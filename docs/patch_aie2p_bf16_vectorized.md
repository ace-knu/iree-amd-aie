# 패치 노트: AIE2P (npu4) bf16 matmul 벡터화

브랜치: `aie2p-bf16-vectorized` (`dev`에서 분기)

`--iree-amdaie-enable-vectorization-passes=true`로 AIE2P(XDNA2 / npu4 / Strix)에서
bf16 matmul을 벡터화하는 작업의 커밋별 기록입니다. 커밋마다 무엇이 바뀌었는지,
왜 바꿨는지, 어떻게 검증했는지를 적습니다. 새 커밋을 올리면 이 문서 맨 아래에
같은 형식으로 항목을 추가합니다.

## 한눈에 보기

| # | 커밋 | 날짜 | 내용 | VGG16 corr (torch 대비) | VGG16 시간 |
|---|---|---|---|---|---|
| 1 | `9b4f831a` | 2026-10-20 | AIE2P bf16 matmul을 bfp16 MAC으로 벡터화 | 0.770, top-1 오답 | — |
| 2 | `d2846c0d` | 2026-10-21 | memtile join의 lock 레이스 수정 (bf16과 무관한 기존 버그) | 0.97917, top-1 정답 | 약 3.3–3.7 s |
| 3 | `822f0fed` cherry-pick (작성: 최지훈) | 2026-10-22 | 코어 반올림 모드를 floor → conv_even으로 설정 | **0.99997**, top-1·top-5 정답 | 약 2.1–3.1 s (변화 없음) |

비교 기준: 벡터화 OFF(스칼라 경로)는 corr 1.00000, 약 68.8 s입니다. 커밋 3까지
적용하면 **스칼라 경로와 거의 같은 정확도로 약 20배 빠릅니다.**

커밋 `f55542b4`는 이 문서와 재현 스크립트를 추가한 문서 커밋이라 표에서 뺐습니다.

측정 조건(모든 커밋 공통):

- 모델: `models/vgg16/vgg16-12.onnx`
- 입력: 고정 시드 랜덤 이미지(`np.random.seed(1)`, `[1,3,224,224]`)
- 정답: 같은 ONNX 그래프를 PyTorch(f32)로 계산한 값
- 컴파일 플래그: [models/vgg16/README.md](../models/vgg16/README.md)와 같고,
  `--iree-amdaie-enable-vectorization-passes`만 `true`로 바꿉니다.
- 명령은 [재현 방법](#재현-방법)을 참고하세요.

---

## 커밋 1 — `9b4f831a` AIE2P bf16 matmul 벡터화 (bfp16 MAC 경유)

### 요약

AIE2P에서 bf16 `vector.contract`는 벡터화되지 않았습니다. `aievec`에 AIE2P bf16
8x8x8 지원이 없었기 때문입니다. AIE2P에는 bf16을 그대로 받는 **행렬** MAC이 없고,
부동소수점 계열 행렬 MAC은 `bfp16`(8개 원소가 지수 1개를 공유하고 원소마다 8비트
가수를 갖는 block floating point)뿐입니다. 이 커밋은 ukernel
(`iree-amd-aie/Target/uKernels/npu4/peano/matmul.cc`의 `load_v64bf16_as_bfp16[_T]`
→ `mac_8x8_8x8T_conf`)이 손으로 하는 순서를 컴파일러가 자동으로 생성하게 합니다.

```
bf16 A, B (8x8)
  → bf16 → f32 승격 (v32bf16.to.v32accfloat × 2)
  → f32 → bfp16 변환 (v64accfloat.to.v64bfp16ebs8)      ※ B는 이 전에 8x8 전치
  → bfp16 × bfp16 MAC (BFP576.BFP576.ACC2048.mac.conf, conf = 780)
  → f32 결과
```

> AIE2P에도 bf16 MAC 명령(`I1024.I1024.ACC2048.bf.mac.conf`, `vmac.f`)은 있지만
> **원소별**(`mac_elem_64`, 명령당 64곱)입니다. 행렬 MAC(`mac_8x8_8x8T`, 명령당
> 512곱)은 bfp16만 받습니다(Peano `aie2p_vmult.h`에서 확인).

### 변경 파일

| 파일 | 변경 |
|---|---|
| `aievec/VectorToAIEVecConversions.cpp` | ① AIE2P matmul 지원 목록에 bf16 8x8x8 추가, ② `vector.transpose` 패턴에 AIE2P 1024비트 bf16 분기 추가, ③ `vector.shuffle`/`vector.bitcast` legal 등록 |
| `aievec/XLLVMOps.td` | AIE2P intrinsic 4개 선언 (bf16→f32, f32→bfp16, bfp16 MAC, `vshuffle`) |
| `aievec/AIEVecToLLVM.cpp` | ① bf16→bfp16→MAC 헬퍼 함수, ② `MatMulOpConversion`에 AIE2P f32 누산 분기, ③ `ShuffleOpConversion` AIE2P 지원 |
| `iree-amd-aie/Transforms/Passes.cpp` | `convert-ub-to-llvm` 패스 추가 |
| `aievec/test/test_shuffle.mlir` | AIE2P `aievec.shuffle` lit 테스트 |

`ShuffleOpConversion` AIE2P 지원과 `AIEVec2PVectorShuffleIntrOp` 선언,
`test_shuffle.mlir`은 아직 머지되지 않은 커밋 `dd26a8a2`(`origin/bert-onnx`)에서
가져와 이 커밋에 포함했습니다.

> 아래 행 번호는 **커밋 1 시점 기준**입니다.

### 상세

#### 1-1. `VectorToAIEVecConversions.cpp`

**① `getSuportedAie2PTypes`에 bf16 8x8x8 등록** (118–131행)

```cpp
117  /// Types currently supported for AIE2P (strix).
118  SmallVector<std::array<Type, 3>> getSuportedAie2PTypes(MLIRContext *context) {
119    SmallVector<std::array<Type, 3>> types;
120    types.push_back(
121        getIntegerMatmulVectorTypes(/* M= */ 8, /* N= */ 8, /* K= */ 8,
122                                    /* A precision (bits)= */ 8,
123                                    /* B precision (bits)= */ 8,
124                                    /*C precision (bits)= */ 32, context));
125    // AIE2P has no native bf16 MAC. bf16 x bf16 -> f32 is lowered by promoting
126    // both operands through the bfp16 (block floating point) MAC pipeline; see
127    // the AIE2P branch of MatMulOpConversion in AIEVecToLLVM.cpp for the
128    // conversion sequence this shape triggers.
129    types.push_back(getBFloatMatmul(/* M= */ 8, /* N= */ 8, /* K= */ 8, context));
130    return types;
131  }
```

> **변경 전 → 후**
> - **기존**: AIE2P 지원 목록에 int8×int8→i32(8x8x8) 하나만 있었습니다. bf16
>   `vector.contract`는 `aievec.matmul`로 바뀌지 않아 벡터화가 적용되지 않았습니다.
> - **추가**: 125–129행. bf16×bf16→f32(8x8x8) 항목 하나와 설명 주석입니다.
> - **결과**: AIE2P에서도 bf16 8x8x8 contract가 `aievec.matmul`로 바뀝니다. 이
>   `aievec.matmul`을 실제 명령으로 내리는 일은 1-3이 맡습니다.

**② `LowerVectorTransposeOpToAIEVecShuffleOpPattern`에 AIE2P 1024비트 bf16 전치
분기 추가** (274–313행, 기존 AIE2용 512비트 로직 앞에 삽입)

```cpp
243  struct LowerVectorTransposeOpToAIEVecShuffleOpPattern
244      : OpConversionPattern<vector::TransposeOp> {
     ...
255    LogicalResult matchAndRewrite(
256        vector::TransposeOp transpOp, OpAdaptor adaptor,
257        ConversionPatternRewriter &rewriter) const override {
     ...
264      // Verify leading dimensions are all 1.
     ...
272      if (perm.back() != static_cast<int64_t>(perm.size() - 2)) return failure();
273
274      // AIE2P's native bf16 register width for the 8x8x8 matmul shape is
275      // 1024 bits (vector<8x8xbf16>), twice AIE2's 512-bit shuffle below. There
276      // is no single shuffle mode at that width, so this mirrors the B-operand
277      // transpose used by the AIE2P bf16 matmul MAC pipeline
278      // (createAie2PBf16MatmulViaBfp16 in AIEVecToLLVM.cpp): split into two
279      // 512-bit halves, shuffle each (T16_4X8), then recombine the transposed
280      // halves (T64_2X8_LO / T64_2X8_HI).
281      if (AMDAIE::isAie2P(device) && elemTyBitWidth == 16 && vBitWidth == 1024 &&
282          resShape.back() == 8) {
283        Location loc = transpOp.getLoc();
284        Type elemTy = resTy.getElementType();
285        auto v32Ty = VectorType::get({32}, elemTy);
286        auto v64Ty = VectorType::get({64}, elemTy);
287        Value flat = rewriter.create<vector::ShapeCastOp>(loc, v64Ty,
288                                                          adaptor.getVector());
289        SmallVector<int64_t> loMask, hiMask, concatMask;
290        for (int64_t i = 0; i < 32; ++i) loMask.push_back(i);
291        for (int64_t i = 32; i < 64; ++i) hiMask.push_back(i);
292        for (int64_t i = 0; i < 64; ++i) concatMask.push_back(i);
293        Value lo = rewriter.create<vector::ShuffleOp>(loc, flat, flat, loMask);
294        Value hi = rewriter.create<vector::ShuffleOp>(loc, flat, flat, hiMask);
295        Value loT = rewriter.create<aievec::ShuffleOp>(
296            loc, v32Ty, lo, nullptr, aievec::ShuffleMode::T16_4X8);
297        Value hiT = rewriter.create<aievec::ShuffleOp>(
298            loc, v32Ty, hi, nullptr, aievec::ShuffleMode::T16_4X8);
299        // T64_2X8_LO/HI's mode verifier requires 64-bit elements; this
300        // second-stage shuffle only reassembles pre-transposed 4-bf16 (64-bit)
301        // chunks from the T16_4X8 outputs above, so reinterpreting as i64 here
302        // does not affect which bf16 lanes end up where.
303        auto v8i64Ty = VectorType::get({8}, rewriter.getI64Type());
304        Value loT64 = rewriter.create<vector::BitCastOp>(loc, v8i64Ty, loT);
305        Value hiT64 = rewriter.create<vector::BitCastOp>(loc, v8i64Ty, hiT);
306        Value resLo64 = rewriter.create<aievec::ShuffleOp>(
307            loc, v8i64Ty, loT64, hiT64, aievec::ShuffleMode::T64_2X8_LO);
308        Value resHi64 = rewriter.create<aievec::ShuffleOp>(
309            loc, v8i64Ty, loT64, hiT64, aievec::ShuffleMode::T64_2X8_HI);
310        Value resLo = rewriter.create<vector::BitCastOp>(loc, v32Ty, resLo64);
311        Value resHi = rewriter.create<vector::BitCastOp>(loc, v32Ty, resHi64);
312        Value resFlat = ...  // vector.shuffle(resLo, resHi) 로 다시 64-lane 합침
313        ...
```

흐름: 8x8 bf16 타일(1024비트) → 64개로 펴기 → 32개씩 반으로 쪼개기 → 각 반쪽에
`T16_4X8` shuffle(반쪽 안에서 전치) → 두 결과를 64비트 단위로 `T64_2X8_LO`/`HI`
shuffle해 재조합 → 다시 8x8로 접기.

> **변경 전 → 후**
> - **기존**: 디바이스 정보 없이 **전체 크기가 정확히 512비트인 벡터만**
>   처리했습니다(`if (vBitWidth != 512) return failure();`). 원소 크기와 마지막
>   차원으로 mode 하나를 골라 `aievec.shuffle` **한 번**으로 끝냈습니다(예:
>   `vector<4x8xbf16>` → `T16_8X4`). AIE2P bf16 타일 `vector<8x8xbf16>`은
>   1024비트라 바로 `failure`가 났습니다. 그래서 VGG16 FC 레이어(`Gemm transB=1`)가
>   `failed to legalize operation 'vector.transpose'`로 컴파일에 실패했습니다.
> - **변경 1**: 패턴이 생성자로 `device`를 받고, 등록할 때 `maybeDevice.value()`를
>   넘깁니다. AIE2P에서만 새 분기를 타게 하기 위해서입니다.
> - **변경 2**: 512비트·원소 크기 검사를 차원·순열 검사 뒤로 옮기고, 그 앞에
>   "AIE2P + 16비트 + 1024비트 + 마지막 차원 8" 분기(shuffle 4번)를 넣었습니다.
>   `T64_2X8_*` 검증기가 64비트 원소를 요구해서 앞뒤로 i64 bitcast를 하지만, 이
>   단계는 이미 전치된 bf16 4개 묶음(64비트)을 통째로 옮기기만 하므로 결과는
>   같습니다. 이 순서는 ukernel의 `load_v64bf16_as_bfp16_T`와 같습니다.
> - **변경 3**: 새 분기가 쓰는 `vector.shuffle`/`vector.bitcast`를 legal로
>   등록했습니다(③).
> - **바뀌지 않은 것**: AIE2용 512비트 경로는 그대로입니다. 검사 순서가 바뀌었지만
>   어느 검사에서 걸리든 `failure`이므로 기존 입력의 결과는 같습니다.

<details>
<summary>참고: shuffle이란?</summary>

shuffle은 **레지스터 안 원소들의 위치(lane)를 정해진 패턴대로 재배치하는
하드웨어 명령**입니다. 값은 바꾸지 않고 순서만 바꿉니다.

AIE 벡터 레지스터 하나는 512비트이고, bf16이면 원소 32개가 한 줄로 들어갑니다.
shuffle은 입력 레지스터 1개 또는 2개를 받아 512비트 레지스터 하나를 만듭니다.
섞는 방법은 **mode** 번호로 고르는데, 임의의 순서를 지정할 수는 없고 하드웨어에
미리 정의된 패턴(전치, 인터리브 등) 중에서 골라야 합니다.

예를 들어 2×4 행렬을 전치하면 레지스터 내용은 이렇게 바뀝니다:

```
행렬            레지스터 상태
[0 1 2 3]       [0 1 2 3 4 5 6 7]
[4 5 6 7]
        ↓ 전치 (4×2)
[0 4]
[1 5]           [0 4 1 5 2 6 3 7]
[2 6]
[3 7]
```

mode 이름 읽는 법:

- **`T16_4X8`**: 16비트 원소 32개를 4×8 행렬로 보고 전치해 8×4로 만듭니다.
  입력은 1개입니다.
- **`T64_2X8_LO` / `T64_2X8_HI`**: 64비트 덩어리 단위로 다룹니다. 입력 2개(a, b)의
  64비트 덩어리 8개씩을 2×8 행렬(1행 = a, 2행 = b)로 보고 전치하면
  `a0 b0 a1 b1 … a7 b7`, 즉 덩어리 16개(1024비트)가 됩니다. 출력 레지스터는
  512비트라서 앞쪽 절반은 `_LO`, 뒤쪽 절반은 `_HI`로 따로 받습니다.

**왜 필요한가.** matmul MAC 명령은 피연산자가 레지스터 안에 특정 배치로 들어 있어야
합니다. 메모리에서 읽어온 B 타일은 그 배치와 다르고, VGG16 FC 레이어처럼 B가 전치된
형태면 행/열이 뒤집혀 있습니다. 그래서 MAC 전에 shuffle로 위치를 맞춥니다. AIE2P
bf16 타일은 8×8 = 1024비트로 레지스터 2개 분량이라 shuffle 한 번으로는 전치가 안
됩니다.

이름이 비슷한 op이 두 가지 있습니다:

- **`aievec.shuffle`**: 위의 하드웨어 shuffle(mode 사용)입니다. 최종적으로 LLVM
  intrinsic `vshuffle`이 됩니다.
- **`vector.shuffle`**: MLIR 표준 op으로, 인덱스 목록으로 원소를 고르는 범용
  연산입니다. 여기서는 64개를 앞 32개 / 뒤 32개로 자르고 다시 이어 붙이는 데만
  씁니다.

</details>

**③ `vector.shuffle`/`vector.bitcast`를 legal로 등록** (912–919행)

```cpp
910  static void configureAIEVecV2Legalizations(ConversionTarget &target) {
911    target.addLegalOp<UnrealizedConversionCastOp>();
912    target.addLegalOp<vector::ShapeCastOp>();
913    // Used by LowerVectorTransposeOpToAIEVecShuffleOpPattern's AIE2P (1024-bit
914    // bf16) case to split/concatenate halves, and to satisfy aievec.shuffle's
915    // per-mode element-bitwidth verifier (T64_2X8_LO/HI require 64-bit
916    // elements); both lower like any other vector op in the generic
917    // vector-to-llvm conversion later on.
918    target.addLegalOp<vector::ShuffleOp>();
919    target.addLegalOp<vector::BitCastOp>();
```

> **변경 전 → 후**
> - **기존**: `vector` 다이얼렉트 op 중 허용된 건 `vector.shape_cast`뿐이었습니다.
>   패턴이 `vector.shuffle`이나 `vector.bitcast`를 만들면 변환 실패로 처리됐습니다.
> - **추가**: 913–919행. 두 op을 legal로 등록하는 두 줄과 이유 주석입니다.
> - **결과**: ②의 새 분기가 만든 두 op이 이 패스를 통과하고, 이후 일반
>   `convert-vector-to-llvm`에서 LLVM IR로 바뀝니다. 다른 경로에서는 이 op을
>   만들지 않으므로 영향이 없습니다.

#### 1-2. `XLLVMOps.td`

AIE2P intrinsic 선언 추가 (83–119행, 아래는 bf16 matmul용 3개):

```tablegen
 83  // ----- BF16 MATMUL (via bfp16; AIE2P has no native bf16 MAC) -----
 84
 85  // Promotes 32 bf16 lanes to 32 f32 accumulator lanes. Two calls plus a
 86  // concatenation build the 64-wide accumulator that
 87  // `AIEVec2PAccFloatToBFP16IntrOp` below consumes.
 88  def AIEVec2PBF16ToV32AccFloatIntrOp :
 89      AIEVec2P_IntrOp<"v32bf16.to.v32accfloat",
 90          [TypeIs<"res", VectorOfLengthAndType<[32], [F32]>>]>,
 91      Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$src)>;
 92
 93  // Converts a 64-wide f32 accumulator into block-floating-point16 (bfp16):
 94  // a 64-byte mantissa vector and an 8-byte exponent vector (one exponent per
 95  // 8-lane block). The real instruction has two results; that is modelled
 96  // here as the single LLVM struct {vector<64xi8>, vector<8xi8>}, which
 97  // callers unpack with two `llvm.extractvalue`s (there is no bf16 -> bfp16
 98  // intrinsic that skips this f32 accumulator step).
 99  def AIEVec2PAccFloatToBFP16IntrOp :
100      AIEVec2P_IntrOp<"v64accfloat.to.v64bfp16ebs8",
101          [TypeIs<"res", LLVM_StructOf<[VectorOfLengthAndType<[64], [I8]>,
102                                       VectorOfLengthAndType<[8], [I8]>]>>]>,
103      Arguments<(ins VectorOfLengthAndType<[64], [F32]>:$src)>;
104
105  // bfp16 x bfp16 -> i32 accumulator MAC. Each bfp16 operand is a
106  // (mantissa, exponent) pair produced by `AIEVec2PAccFloatToBFP16IntrOp`
107  // above. The accumulator and result are declared i32 because that is this
108  // intrinsic's LLVM signature; the bits are the same f32 accumulator used
109  // elsewhere in the bf16 matmul lowering, reached by a bitcast at the call
110  // site (see the AIE2P branch of MatMulOpConversion in AIEVecToLLVM.cpp).
111  def AIEVec2PBFP16MacConfIntrOp :
112      AIEVec2P_IntrOp<"BFP576.BFP576.ACC2048.mac.conf",
113          [TypeIs<"res", VectorOfLengthAndType<[64], [I32]>>]>,
114      Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhsMantissa,
115                     VectorOfLengthAndType<[8], [I8]>:$lhsExponent,
116                     VectorOfLengthAndType<[64], [I8]>:$rhsMantissa,
117                     VectorOfLengthAndType<[8], [I8]>:$rhsExponent,
118                     VectorOfLengthAndType<[64], [I32]>:$acc,
119                     I32:$conf)>;
```

`AIEVec2PAccFloatToBFP16IntrOp`(99–103행)는 실제로 결과를 2개(mantissa, exponent)
돌려주는 명령입니다. intrinsic 선언 베이스 클래스(`ExtIntrOpBase`)가 결과 1개만
지원해서 `LLVM_StructOf`로 묶어 결과 하나로 모델링했습니다.

> **변경 전 → 후**
> - **기존**: AIE2P intrinsic은 int8 matmul용 등만 있었습니다. bf16→f32 승격,
>   f32→bfp16 변환, bfp16 MAC, `vshuffle`은 선언이 없어서 컴파일러가 이 명령들을
>   LLVM IR로 낼 방법이 없었습니다.
> - **추가**: 위 3개와 `AIEVec2PVectorShuffleIntrOp`(`vshuffle`, `dd26a8a2`에서
>   가져옴)까지 4개입니다. 기존 선언은 건드리지 않았습니다.
> - **결과**: 1-3의 코드가 `xllvm::AIEVec2P...::create(...)`로 이 명령들을 만들면
>   `llvm.aie2p.*` intrinsic 호출이 되고, Peano 백엔드가 실제 명령으로 바꿉니다.

#### 1-3. `AIEVecToLLVM.cpp`

**① bf16→bfp16 변환과 MAC 호출 헬퍼** (`createAie2PVShuffle`,
`splitV64Bf16ToHalves`, `promoteBf16HalvesToBfp16`, `createAie2PBf16MatmulViaBfp16`,
631–763행). 핵심인 `createAie2PBf16MatmulViaBfp16`(729–763행):

```cpp
729  static Value createAie2PBf16MatmulViaBfp16(OpBuilder &builder, Location loc,
730                                             Value lhs, Value rhs, Value acc,
731                                             Value confCst) {
     ...
736    // A: no transpose.
737    auto [aLo, aHi] = splitV64Bf16ToHalves(builder, loc, lhs, v32bf16Ty);
738    auto [aMantissa, aExponent] = promoteBf16HalvesToBfp16(builder, loc, aLo, aHi);
739
740    // B: transpose the 8x8 tile before promotion (empirically required -- see
741    // VGG16 e2e correlation history in the aie2p-bf16-vectorized branch: removing
742    // this made correlation against a torch reference WORSE, 0.77 -> 0.057, not
743    // better, so the MAC does consume B pre-transposed regardless of whatever
744    // layout a separate upstream `vector.transpose` may already have applied).
745    // Mirrors the ukernel's `load_v64bf16_as_bfp16_T`.
746    auto [bLo, bHi] = splitV64Bf16ToHalves(builder, loc, rhs, v32bf16Ty);
747    Value bLoT = createAie2PVShuffle(builder, loc, bLo, /*rhs=*/nullptr,
748                                     aievec::ShuffleMode::T16_4X8, v32bf16Ty);
749    Value bHiT = createAie2PVShuffle(builder, loc, bHi, /*rhs=*/nullptr,
750                                     aievec::ShuffleMode::T16_4X8, v32bf16Ty);
751    Value bTLo = createAie2PVShuffle(builder, loc, bLoT, bHiT,
752                                     aievec::ShuffleMode::T64_2X8_LO, v32bf16Ty);
753    Value bTHi = createAie2PVShuffle(builder, loc, bLoT, bHiT,
754                                     aievec::ShuffleMode::T64_2X8_HI, v32bf16Ty);
755    auto [bMantissa, bExponent] = promoteBf16HalvesToBfp16(builder, loc, bTLo, bTHi);
756
757    Value accI32 = bitcastValueToType(builder, loc, acc, v64i32Ty);
758    Value macResI32 = xllvm::AIEVec2PBFP16MacConfIntrOp::create(
759                          builder, loc, v64i32Ty, aMantissa, aExponent,
760                          bMantissa, bExponent, accI32, confCst)
761                          .getResult();
762    return bitcastValueToType(builder, loc, macResI32, v64f32Ty);
763  }
```

A는 승격만 하고, B는 승격 전에 8x8 전치합니다. bfp16 MAC(`mac_8x8_8x8T`)이 B를
전치된 배치로 받기 때문입니다. 이 전치를 빼면 결과가 망가지는 것(corr 0.057)을
실측으로 확인했습니다.

> **변경 전 → 후**
> - **기존**: 이 헬퍼들은 없었습니다. bf16 `aievec.matmul`을 AIE2P 명령으로 바꾸는
>   코드 자체가 없었습니다.
> - **추가**: 모두 새 함수이고, 기존 함수는 고치지 않았습니다.
>   - `createAie2PVShuffle`: `aievec.shuffle` op을 거치지 않고 `vshuffle`
>     intrinsic을 바로 만듭니다(이미 LLVM 변환 단계 안이라서).
>   - `splitV64Bf16ToHalves`: 64개를 32개씩 둘로 자릅니다(`llvm.shufflevector`).
>   - `promoteBf16HalvesToBfp16`: bf16→f32 두 번, 64개로 합치기, bfp16 변환 순서로
>     진행하고 (mantissa, exponent)를 돌려줍니다.
>   - `createAie2PBf16MatmulViaBfp16`: A는 승격, B는 전치 후 승격, 그 다음 bfp16 MAC.
> - **결과**: ukernel `matmul.cc`가 손으로 하던 순서를 컴파일러가 자동으로 생성합니다.

**② `MatMulOpConversion`의 f32 누산 분기에 AIE2P 케이스 추가** (821–845행)

```cpp
821    if (isa<Float32Type>(accVecTy.getElementType())) {
822      if (AMDAIE::isAie2P(device)) {
823        // AIE2P has no native bf16 MAC (unlike AIE2's `bf.mac16.conf` used
824        // below), so this goes through the bfp16 MAC pipeline instead; see
825        // createAie2PBf16MatmulViaBfp16. The `configuration`/`confCst` built
826        // above describe a 16x16 multiplication (from the bf16 operand
827        // types) and do not apply to the bfp16 MAC actually issued here, so
828        // build that MAC's own configuration word: signed x/y, amode=2
829        // (32b float accumulator lanes), bmode=1 (the bfp16 MAC's mode).
830        DataPathConfiguration bfp16Configuration{/*xSigned=*/true,
831                                                 /*ySigned=*/true,
832                                                 /*aMode=*/2, /*bMode=*/1,
833                                                 /*cMode=*/0};
834        Value bfp16ConfCst = LLVM::ConstantOp::create(
835            rewriter, loc, i32ty,
836            rewriter.getI32IntegerAttr(bfp16Configuration.get()));
837        matMulResVal = createAie2PBf16MatmulViaBfp16(rewriter, loc, lhs, rhs,
838                                                     acc, bfp16ConfCst);
839      } else if (AMDAIE::isAie2(device)) {
840        matMulResVal =
841            forceCastOperandsAndCreateTarget<xllvm::AIEVec2MacConfBF16IntrOp>(
842                rewriter, loc, {lhs, rhs, acc, confCst});
843      } else {
844        llvm_unreachable("no support for float matmul on this device, for now");
845      }
846    } else {
```

> **변경 전 → 후**
> - **기존**:
>   ```cpp
>   if (!AMDAIE::isAie2(device))
>     llvm_unreachable("no support for float matmul except for AIE2, for now");
>   matMulResVal = ...AIEVec2MacConfBF16IntrOp...;   // AIE2의 bf16 MAC
>   ```
>   f32 누산 matmul은 AIE2만 지원했고, AIE2P가 들어오면 컴파일러가 죽었습니다.
> - **변경**: `if AIE2P / else if AIE2 / else unreachable` 3갈래로 바꿨습니다.
>   AIE2P 갈래는 bfp16 MAC 전용 conf(signed, amode=2, bmode=1 → 780)를 만들고
>   ①의 `createAie2PBf16MatmulViaBfp16`을 호출합니다. bf16 피연산자 타입으로
>   계산된 범용 `configuration`(16×16 곱셈용)은 이 MAC에 맞지 않아 쓰지 않습니다.
> - **결과**: AIE2P bf16 matmul이 bfp16 MAC으로 내려갑니다. AIE2 경로는 위치만
>   `else if` 안으로 옮겨졌고 동작은 같습니다.

**③ `ShuffleOpConversion` AIE2P 지원** (`dd26a8a2`에서 가져옴)

> **변경 전 → 후**
> - **기존**: 첫 줄이 `assert(AMDAIE::isAie2(device) && ...)`여서 AIE2P에서
>   `aievec.shuffle`을 만나면 죽었습니다. 두 번째 입력이 없을 때는 AIE2 전용 undef
>   intrinsic을 썼고, 결과는 항상 AIE2 `vshuffle`이었습니다.
> - **변경**: AIE2도 AIE2P도 아니면 assert 대신 match failure를 돌려줍니다. AIE2P일
>   때는 두 번째 입력 자리에 일반 `llvm.undef`를 넣고(AIE2P에는 undef intrinsic이
>   없음) AIE2P `vshuffle`을 만듭니다. AIE2 쪽은 그대로입니다.
> - **결과**: 1-1-②가 만든 `aievec.shuffle`이 AIE2P에서도 LLVM으로 내려갑니다.
>   `test/test_shuffle.mlir`에 AIE2P 테스트가 있습니다.

#### 1-4. `Passes.cpp` — `convert-ub-to-llvm` 추가 (963–970행)

```cpp
962    pm.addPass(aievec::createConvertAIEVecToLLVMPass());
963    pm.addPass(createConvertVectorToLLVMPass());
964    // Vector-to-LLVM's shuffle/insert lowering can leave `ub.poison` behind as
965    // a placeholder (e.g. for the AIE2P bf16 vector.transpose sequence); it has
966    // no LLVMIR translation of its own and must be converted to `llvm.mlir.
967    // poison` before translateModuleToLLVMIR, unlike the rest of this pipeline
968    // which only ever produced ops that already had one.
969    pm.addPass(createUBToLLVMConversionPass());
970    pm.addPass(memref::createExpandStridedMetadataPass());
```

> **변경 전 → 후**
> - **기존**: `convert-vector-to-llvm` 다음에 바로 `expand-strided-metadata`가
>   왔습니다. 기존 경로는 `ub.poison`을 만들지 않아 문제가 없었습니다.
> - **추가**: 964–969행. `createUBToLLVMConversionPass()` 한 줄과 주석입니다.
> - **결과**: 1-1-②의 `vector.shuffle`이 남기는 `ub.poison`이 `llvm.mlir.poison`으로
>   바뀌어, `translateModuleToLLVMIR`이 "missing `LLVMTranslationDialectInterface`
>   registration for ub.poison"으로 실패하지 않습니다. `ub.poison`이 없는 모듈에는
>   영향이 없습니다.

### 검증

**IR 레벨.** `aievec.matmul`/`vector.transpose` 입력에 `iree-opt
-convert-aievec-to-llvm`을 돌려, 생성된 시퀀스가 설계와 같은지 확인했습니다. 양쪽
피연산자 모두 승격×2 → 합치기 → bfp16 변환 → extractvalue, B는 추가로
`T16_4X8`/`T64_2X8_{LO,HI}` 전치, MAC 시그니처와 conf=780을 확인했습니다.

**하드웨어, 단일 matmul** (bf16 입력·f32 출력 `linalg.matmul`, npu4):

| Shape (M×K×N) | 결과 |
|---|---|
| 32×32×32 | corr 0.99986, 최대 오차 0.10 |
| 256×32×64 | corr 0.99988, 최대 오차 0.10, 모든 row-tile에서 균일 |
| B 전치 제거 (대조군) | corr **0.057** → B 전치가 꼭 필요함 |

**하드웨어, VGG16 전체**: corr **0.770**, top-1 오답(벡터화 OFF는 1.00000, 정답).

### 이 커밋 시점의 알려진 문제

VGG16 전체에서 정확도가 떨어졌습니다. matmul lowering 자체는 위 단일 matmul에서
정확했고, 다음을 확인한 끝에 **M(행 수)이 커질 때만** 틀린다는 걸 찾았습니다.

- 첫 conv 레이어만 떼어내도 corr 0.857이었습니다.
- 패딩은 원인이 아니었습니다. `iree-amdaie-pad-contraction-dispatches`는 벡터화
  ON/OFF 모두 K-tail을 똑같이 0으로 채우고, OFF는 같은 텐서로 corr 1.0이 나옵니다.
- 같은 bf16 matmul(K=32, N=64)이 M=256에서는 정확하고, M=4096에서 corr 0.78,
  M=50176에서 corr 0.753이었습니다.

원인은 bf16과 무관한 기존 버그였고, 커밋 2에서 고쳤습니다.

---

## 커밋 2 — `d2846c0d` memtile join의 생산자별 lock 쌍 (L2 덮어쓰기 레이스 수정)

### 요약

여러 코어의 결과를 memtile 버퍼 하나에 모으는 **join**에서, 모든 생산자가 lock
한 쌍을 공유해서 생기는 레이스를 고쳤습니다. 빠른 코어가 아직 L3로 나가지 않은
자기 구역을 2웨이브 뒤의 결과로 덮어쓰는 문제였습니다. **int8 벡터화에서도 똑같이
재현되는 기존 버그**이고, 벡터화 커널이 빨라지면서 드러났습니다. upstream(nod-ai)
main의 같은 코드도 동일합니다(upstream 이슈는 아직 올리지 않음).

### 원인

**"손상"의 뜻.** M=4096(K=32, N=64) matmul의 출력 `4096×64`를 numpy 정답과 행 단위로
비교했을 때 값이 틀린 행입니다. 틀린 행은 항상 **64행 단위 덩어리**로 나왔습니다.

- 출력 4096행을 256행씩 16번에 나눠 처리합니다. 256행 묶음 하나가 **웨이브**입니다
  (웨이브 0 = 행 0–255, 웨이브 1 = 행 256–511, …).
- 한 웨이브는 한 column의 core row 4개가 64행씩 나눠 계산합니다. core row `r`(0–3)은
  웨이브 `k`에서 행 `256k + 64r`부터 64행을 맡습니다. 이 64행이 **블록**입니다.
- 각 코어는 자기 블록을 memtile 버퍼의 자기 구역에 쓰고, 4개 블록이 다 모이면 버퍼
  전체가 L3(호스트 메모리)로 나갑니다.

손상된 블록의 값은 쓰레기가 아니라 **같은 core row가 웨이브 `k+2`에서 계산했어야 할
정답(출력 행 `+512` 위치)과 비트 단위로 같았습니다.** 예: 웨이브 0, core row 3의
블록(행 192–255)에 행 704–767의 정답이 들어 있었습니다.

| 실험 (M=4096, K=32, N=64) | 결과 | 의미 |
|---|---|---|
| **int8** 벡터화 | 같은 패턴으로 손상 | bf16과 무관한 기존 버그 |
| 벡터화 OFF (스칼라) | corr 1.0, 손상 0행 | 타일링·채널·lock 구조가 같은데 정확함 |
| 같은 바이너리 5회 실행 | 손상 위치가 매번 다름 | 타이밍 레이스 |
| 손상 블록의 내용 | 웨이브 k+2 정답과 비트 단위로 일치 (int8, 실행마다 14블록) | 한 코어가 2웨이브 앞서 자기 구역을 덮어씀 |
| `--iree-amdaie-num-rows=2` | 1회 실행에서 손상 0행 | 생산자 수가 줄어 레이스 확률이 낮아짐 (1회라 확정 아님) |

**메커니즘.** 4개 core row의 결과는 memtile(`tile_0_1`)의 핑퐁 버퍼 2개(각 2048×f32)에
row별 구역(offset 0/512/1024/1536)으로 모였다가 MM2S 한 채널로 L3에 나갑니다. 이때
4개 S2MM 채널이 **lock 한 쌍을 공유**했습니다.

```
lock_0_1_8 (빈칸)  init = 8   // 생산자 4 × depth 2
lock_0_1_9 (채움)  init = 0
S2MM ch2..5 (row 2..5):  acquire lock_8 >= 1 -> 자기 구역에 쓰기 -> release lock_9 +1
MM2S ch2 (L3로):          acquire lock_9 >= 4 -> 버퍼 전체 전송  -> release lock_8 +4
```

이 카운터는 "어느 row가 어느 버퍼에 썼는지"를 구분하지 못합니다. 한 row가 빠르면
웨이브 k → 버퍼 A, k+1 → 버퍼 B, **k+2 → 다시 버퍼 A**까지 쓸 수 있고(다른 row 몫의
빈칸 토큰을 가져다 쓰므로), 그때 버퍼 A가 아직 L3로 나가지 않았으면 웨이브 k 결과가
덮어써집니다.

**스칼라 경로가 멀쩡했던 이유.** 스칼라 커널은 웨이브 하나 계산이 매우 길어서 4개
코어가 거의 같은 속도로 맞물려 돕니다. 벡터화 커널은 짧아서 입력 도착 시간의 작은
차이만으로도 한 코어가 2웨이브 앞서 나갈 수 있습니다.

**코드 위치.** `AMDAIEObjFifoBufferization.cpp`(수정 전 68–112행)가 생산자가 여러
개인 objectFifo에도 타일당 lock 한 쌍만 만들고 `producerLockInitValue = 생산자 수 ×
depth`로 초기화했습니다(같은 파일 52–57행 TODO가 이 가정을 언급).

### 변경 파일

| 파일 | 기존 | 변경 후 |
|---|---|---|
| `Transforms/Utils/AMDAIEDmaUtils.{h,cpp}` | join 여부와 구역 분할을 판정하는 함수가 없었습니다. | `JoinSlice`(생산자, offset, length)와 `getContiguousJoinSlices()`를 추가했습니다. 아래 두 패스가 같은 판정을 쓰도록 한 곳에 뒀습니다. |
| `Transforms/AMDAIEObjFifoBufferization.cpp` | 생산자 수와 관계없이 타일마다 lock **한 쌍**, 빈칸 초기값 `max(생산자, 소비자) × depth`(여기서는 8). | join이면 lock 쌍을 **생산자 수만큼**(4쌍) 만들고 빈칸 초기값은 `depth`(2). join이 아니면 `numLockPairs = 1`로 기존과 같습니다. |
| `Transforms/AMDAIELowerToAIE.cpp` 생산자(S2MM) | lock이 정확히 1쌍이어야 했고, 모든 S2MM이 `producerLocks[0]`/`consumerLocks[0]`를 공유. | join이면 lock이 생산자 수만큼 있어야 하고, 각 S2MM은 slice 목록에서 자기 위치(`lockPairIndex`)를 찾아 **자기 쌍만** 씁니다. |
| `Transforms/AMDAIELowerToAIE.cpp` 소비자(MM2S) | 버퍼마다 BD 하나: "채움 ≥ 4 대기 → 버퍼 전체 전송 → 빈칸 +4". | 새 함수 `createJoinConsumerDMABlocks`가 버퍼마다 구역별 BD 체인을 만듭니다. 각 BD는 "생산자 j의 채움 ≥ 1 → 구역 j 전송 → 생산자 j의 빈칸 +1". 버퍼 2 × 구역 4 = BD 8개를 돌고 처음으로 돌아갑니다. |
| `Transforms/AMDAIELowerToAIE.h` | — | `createJoinConsumerDMABlocks` 선언, `AMDAIEDmaUtils.h` include. |
| `Transforms/test/obj_fifo_bufferization.mlir` | — | `@join_per_producer_lock_pairs`(2쌍, init 2/0), `@join_overlapping_regions_shared_lock`(구역이 겹치면 기존처럼 1쌍, init 4). |
| `Transforms/test/lower_to_aie.mlir` | — | `@join_per_producer_locks`(S2MM별 lock 쌍, MM2S 4-BD 체인). |

`getContiguousJoinSlices()`는 다음을 모두 만족할 때만 구역 목록을 돌려줍니다.
하나라도 어긋나면 `std::nullopt`를 돌려주고 기존 동작을 그대로 탑니다.

- L2(memspace 1) objectFifo이고 단일 타일에 있음
- 생산자 연결 2개 이상, 소비자 연결 1개
- 각 생산자가 서로 겹치지 않는 연속 구역에 쓰고, 구역들이 빈틈없이 이어짐
- 소비자가 그 전체를 주소 순서대로 연속으로 읽음

`reprogramDmas` 모드에는 circular DMA가 없어서 항상 `std::nullopt`입니다.

**결과.** 생산자 j의 구역이 L3로 나가야만 생산자 j의 빈칸 토큰이 돌아옵니다. 그래서
한 코어가 앞서 나가도 아직 안 나간 자기 구역을 덮어쓸 수 없습니다. 구역이 연속이고
순서대로 나가므로 L3로 가는 데이터는 같습니다. 같은 column-0 memtile 기준으로 join용
lock 4쌍(각 init 2/0), S2MM 2~5번이 각자 자기 쌍, MM2S는 BD 8개가 됩니다. memtile
자원 여유: AIE2P memtile은 lock 64개, BD 48개(채널 짝/홀수에 24개씩)입니다.

### 검증

lit: AMD-AIE 호스트 테스트 전부 통과.

| 하드웨어 테스트 | 수정 전 | 수정 후 |
|---|---|---|
| int8 M=4096, 5회 | 매번 14~40블록 손상 | **5회 모두 손상 0** |
| bf16 M=4096, 10회 | 매번 손상 | **10회 모두 손상 0** |
| VGG16 첫 conv 레이어, 3회 | corr 0.857 | **corr 0.99995** (3회 동일) |
| VGG16 전체, 3회 | corr 0.770, top-1 오답 | **corr 0.97917, top-1 정답** (3회 동일) |

**성능** (2026-10-21, `iree-benchmark-module`, 10회 반복)

| | 벡터화 OFF | 벡터화 ON |
|---|---|---|
| 평균 | 68,839 ms | **3,660 ms** |
| 중앙값 | 68,842 ms | 3,518 ms |
| 표준편차 | 362 ms | 662 ms |
| 변동계수(cv) | 0.53% | 18.09% |

벡터화 ON이 약 19배 빠릅니다. 수정 전 바이너리(3,652 ms)와 같은 범위라 lock 수정은
속도에 영향이 없습니다.

<details>
<summary>참고: 변동계수(cv)란?</summary>

`표준편차 ÷ 평균 × 100%`로, 측정값이 평균에 비해 얼마나 흔들리는지 나타냅니다.
평균 69초에서 0.36초 흔들리는 것과 평균 3.7초에서 0.66초 흔들리는 건 의미가 달라서,
평균으로 나눠야 공정하게 비교할 수 있습니다.

- OFF 0.53%: 10번 돌려도 거의 같은 시간입니다.
- ON 10~18%: 실행마다 ±0.5초 안팎으로 흔들립니다(개별 실행 2.7~5.0초). lock 수정
  전후 모두 그랬고, 원인은 아직 확인하지 않았습니다. 그래서 벡터화 ON끼리 비교할
  때는 여러 묶음을 번갈아 재야 합니다.

</details>

### 이 커밋 시점의 알려진 문제

VGG16 corr가 0.979로, 벡터화 OFF(1.00000)보다 낮습니다. 원인은 bfp16 변환의 반올림
방향이었고, 커밋 3에서 고쳤습니다.

---

## 커밋 3 — `822f0fed` cherry-pick: 코어 반올림 모드 설정 (crRnd = conv_even)

작성: 최지훈 (통합 브랜치 `3devices-f32-aie2p-bf16`의 `822f0fed`). 소스 8개와 lit 테스트만
가져왔고, 원 커밋의 `docs/` 변경(`docs/CHANGES.md`, `docs/specs/`)은 이 브랜치에 넣지
않았습니다.

### 요약

VGG16 corr 0.979의 원인은 **bfp16 변환의 반올림 방향**이었습니다. f32→bfp16 변환 명령
(`v64accfloat.to.v64bfp16ebs8`)은 코어 제어 레지스터 `crRnd`가 정한 방식으로 반올림하는데,
아무도 이 레지스터를 설정하지 않아 리셋값 **floor(항상 내림)**로 돌고 있었습니다. floor는
모든 가수를 같은 방향으로 깎기 때문에, K 방향으로 수천 개를 더하는 동안 오차가 상쇄되지 않고
쌓입니다. 이 커밋은 모든 코어 함수의 맨 앞에서 `crRnd`를 **conv_even(짝수 쪽 최근접 반올림)**으로
설정합니다. 정확도가 corr 0.97917 → 0.99997로 올라가고 속도는 그대로입니다.

### 원인 확인 (이 브랜치에서 독립적으로 한 것)

최지훈 님과 별개로 같은 결론에 도달했습니다.

- **하드웨어 변환 방식**: 하드웨어 bf16 matmul 출력(256×32×64, 32×32×32)에 bfp16 변환 후보를
  대입해 보니, "K 방향 8개 블록이 최대 지수 `E`를 공유하고, 각 원소는 `floor(x / 2^(E-6))`인
  부호 있는 8비트 가수" 모델이 **모든 출력 원소와 비트 단위로 일치**했습니다.
- **VGG16 시뮬레이션**: 이 모델을 PyTorch VGG16에 넣어 계산하면 corr 0.97917로 하드웨어와
  같았고(출력 차이 최대 2.4e-7), floor를 짝수 쪽 반올림으로만 바꾸면 0.99996이었습니다.
  레이어별로는 floor가 conv0 0.99998 → conv12 0.984 → 마지막 FC 0.979로 계속 떨어졌습니다.
- 그래서 0.979는 이 반올림 하나로 전부 설명되고, 커밋 1의 전치, 커밋 2의 lock 수정, 타일링에는
  문제가 없다는 것도 함께 확인됐습니다.

> 다른 보완안(bfp16을 hi/lo로 나눠 MAC 3번, 원소별 bf16 MAC)은 연산량이 3~8배 늘어 쓰지
> 않았습니다. 이 브랜치에서도 matmul마다 반올림을 설정하는 안(새 XLLVM op)을 따로 구현해
> 같은 결과를 얻었지만, 코어 시작에 한 번 설정하고 옵션도 있는 이 커밋으로 통일했습니다.

### 변경 파일

| 파일 | 변경 |
|---|---|
| `aie/AMDAIECoreToStandard.cpp` | 코어 함수 첫 명령으로 `call @llvm.<arch>.set.ctrl.reg(crRnd, mode)` 삽입. `rounding-mode` 옵션 해석 |
| `aie/Passes.td` | `amdaie-standard-lowering` 패스에 `rounding-mode` 옵션 추가 (기본 `auto`) |
| `iree-amd-aie/Target/AIETarget.{h,cpp}` | 컴파일 플래그 `--iree-amdaie-core-rounding-mode` 추가 |
| `iree-amd-aie/Target/XCLBinGen.{h,cpp}` | 위 플래그를 패스 옵션까지 전달 |
| `aie_runtime/iree_aie_runtime.{h,cc}` | `AMDAIEDeviceModel::getRoundingModeCtrlReg()`: `crRnd` 레지스터 번호 (AIE2 6, AIE2P 1) |
| `aie/test/core_rounding_mode.mlir` | lit 테스트: auto / none / floor / chess / 잘못된 이름 |

### 상세

**`AMDAIECoreToStandard.cpp`** (170–183행) — 코어 함수 맨 앞에 반올림 설정 삽입

```cpp
170    // Set the rounding mode before anything else runs on the core, so that every
171    // SRS / float-to-bf16 / float-to-bfp16 conversion in the core (including
172    // those in linked ukernels) sees it.
173    if (roundingMode.has_value()) {
174      rewriter.setInsertionPointToStart(&coreFunc.getBody().front());
175      Location loc = coreOp.getLoc();
176      IntegerType i32 = rewriter.getI32Type();
177      SmallVector<Value, 2> args{
178          rewriter.create<arith::ConstantOp>(
179              loc, i32, rewriter.getI32IntegerAttr(roundingMode->ctrlReg)),
180          rewriter.create<arith::ConstantOp>(
181              loc, i32, rewriter.getI32IntegerAttr(roundingMode->value))};
182      rewriter.create<func::CallOp>(loc, roundingMode->setCtrlReg, args);
183    }
```

옵션 해석(276행~): `auto`면 AIE2P에서 conv_even(12), AIE2에서는 아무것도 하지 않습니다.
`none`이면 설정하지 않고, `floor`·`conv_even` 같은 모드 이름을 주면 그 값을 씁니다. chess
경로에서는 적용하지 않고(이름을 직접 주면 경고), 모르는 이름이면 에러를 냅니다. 모드 번호는
Peano `aie2p_defines.h`의 `rnd_*` 값과 같습니다(129행 `getRoundingModeValue`).

> **변경 전 → 후**
> - **기존**: 코어 시작 코드에서 `crRnd`를 건드리지 않아, 모든 반올림 변환이 리셋값 floor로
>   동작했습니다.
> - **추가**: 코어마다 첫 명령으로 `set.ctrl.reg(1, 12)` 한 번(AIE2P 기본). 생성된 코어
>   코드에서는 `mov crrnd, #12` 명령 하나입니다.
> - **결과**: 벡터화 하강과 ukernel 모두 bfp16 변환이 짝수 쪽 최근접 반올림을 합니다. 같은
>   코어의 다른 반올림 변환(SRS, f32→bf16)도 같은 모드를 따릅니다. AIE2(npu1)는 기본값에서
>   달라지는 것이 없습니다.

**사용법**: 기본값(`auto`)이면 따로 할 일이 없습니다. 비교하려면 컴파일 플래그로 바꿉니다.

```bash
--iree-amdaie-core-rounding-mode=floor   # 이전 동작 (corr 0.979)
--iree-amdaie-core-rounding-mode=none    # 레지스터를 건드리지 않음
```

### 검증

- lit: AMD-AIE 테스트 222개 전부 통과(새 `core_rounding_mode.mlir` 포함). 새 테스트 파일은
  CMake 재구성(`cmake build`)을 해야 테스트 목록에 잡힙니다.
- 하드웨어 (npu4, [재현 문서](repro_aie2p_bf16_vectorized.md)의 스크립트):

| 확인 | 결과 |
|---|---|
| bf16 256×32×64, 기본 / `floor` | corr 0.99996 / 0.99990 (옵션이 실제로 동작) |
| int8 4096×32×64 | 오차 0, PASS (int8은 영향 없음) |
| bf16 4096×32×64 (join 레이스) | PASS, 손상 0 |
| VGG16 첫 conv 레이어 | corr 0.99997 |
| **VGG16 전체** | **corr 0.99997**, top-1 정답(892), top-5가 순서까지 torch와 같음, 3회 실행 비트 단위 동일 |
| VGG16 전체 성능 (`real_time_mean`) | 측정 묶음에 따라 2,096 ms(cv 1.7%) ~ 3,105 ms(cv 13.6%) |

성능은 커밋 2와 같은 범위입니다. 같은 빌드에서도 측정할 때마다 2.1 s와 3.1 s처럼 크게 달라서,
이 커밋이 속도를 바꿨다고 볼 근거는 없습니다.

최지훈 님 쪽 검증(원 커밋 메시지 기준): bf16 벡터화 matmul이 conv_even bfp16 흉내와 난수 입력
4개 모양에서 비트 단위로 일치, int8 출력 그대로, 회귀 26×2와 VGG16(벡터화 OFF) 출력 그대로.

### 이 커밋 시점의 알려진 문제

- 나중에 AIE2P에 SRS나 bf16 출력 하강이 생기면 그 연산도 conv_even을 따릅니다. int8 양자화처럼
  연산마다 반올림이 정해져 있어야 하는 경우에는 그 하강이 직접 저장·설정·복원해야 합니다.
- chess 경로에는 적용되지 않습니다.

---

## 남은 과제

- **분배(distribute) 쪽 레이스**: memtile 버퍼 1개 → 코어 N개로 나눠 주는 쪽도 같은
  공유 lock 구조라 비슷한 레이스가 있을 수 있습니다. 확인하지 않았습니다.
- **벡터화 ON 실행 시간의 큰 흔들림(cv 2~18%, 같은 빌드에서 2.1~3.7 s)**: 원인을
  확인하지 않았습니다. 최지훈 님이 런타임이 NPU dispatch마다 하드웨어 컨텍스트를 만들고
  지우는 비용(dispatch당 약 94 ms)을 찾았고, 이것과 관련 있을 수 있습니다(추정).
- **AIE2P transpose 지원 범위**: 새로 처리하는 1024비트 전치는 `vector<8x8xbf16>`
  하나입니다. AIE2P 512비트 전치는 AIE2 mode 번호를 그대로 쓰는데, int8 lit 테스트
  외에는 따로 확인하지 않았습니다.
- **upstream**: join lock 레이스 이슈를 아직 올리지 않았습니다.

## 재현 방법

NPU에서 직접 확인하는 절차(명령과 기대 결과)는
[repro_aie2p_bf16_vectorized.md](repro_aie2p_bf16_vectorized.md)에 있습니다.

---

## 부록: 쉽게 풀어서 설명

**1. 원래 문제**

AMD NPU(XDNA2)에서 bf16(16비트 소수) 행렬곱을 벡터화해서 빠르게 돌리는 기능이
없었습니다. NPU에는 bf16 행렬을 그대로 곱하는 회로가 없고, `bfp16`(숫자 8개가 지수를
공유하는 압축 형식)으로 바꿔야 행렬 곱셈 회로를 쓸 수 있습니다. 그래서 "bf16 →
bfp16으로 바꾸고 → 곱하고 → 결과를 꺼내는" 과정을 컴파일러에 넣었습니다(커밋 1).

**2. VGG16이 틀린 이유: 버퍼 동기화 버그 (커밋 2)**

작은 행렬곱은 맞았는데 VGG16 전체는 틀렸습니다(상관계수 0.77). 행렬을 키워 보니
커질 때만 틀렸고, 틀린 64행에는 "2라운드 뒤의 정답"이 들어 있었습니다. 4개 코어가
결과를 memtile의 버퍼 2개에 모으는데, "빈칸이 몇 개 남았나"를 **하나의 숫자로
공유**해서, 빠른 코어가 다른 코어 몫의 빈칸까지 쓰며 앞서 나가 아직 안 내보낸 자기
구역을 덮어썼습니다. int8에서도 똑같이 생기는 원래 있던 버그였고, 벡터화로 코어가
빨라지면서 드러났습니다. 코어마다 자기 구역 전용 숫자를 따로 두게 고쳤습니다.
→ 상관계수 0.979, 정답 맞힘.

**3. 남은 오차: 반올림 방향 (커밋 3)**

bfp16으로 바꿀 때 NPU는 기본 설정으로 **항상 내림**을 했습니다. 하나하나는 작은 오차지만
늘 같은 방향이라 수천 개를 더하는 동안 쌓였습니다. 컴퓨터로 이 "항상 내림"을 흉내 내니 NPU
결과와 사실상 똑같이 나와서 원인이 이것 하나라는 게 확인됐습니다. 코어가 시작할 때 반올림
설정을 "가까운 쪽으로"로 바꾸는 명령 하나를 넣었습니다(최지훈 님 커밋).
→ 상관계수 0.99997, top-5까지 정답과 같고 속도는 그대로입니다.

**4. 결론**

벡터화하지 않은 경로(69초, 상관계수 1.00000)와 거의 같은 정확도로, 약 20배 빠른
2~3초에 VGG16을 돌립니다.

## 부록: 정정 이력

이 문서의 이전 판(공유된 적이 있는 판)에 있던 내용 중 틀린 것들입니다.

- "4번째 core row만 lock 순서가 다르다", "memtile DMA 채널 5번이 원인" → 틀렸습니다.
  코드 diff의 비교 구간을 잘못 맞춘 착시였고, 대조 실험으로 배제했습니다. 원인은
  커밋 2의 공유 lock 레이스입니다.
- "lock 수정 후 2,202 ms(약 31배)" → 다음 날 같은 바이너리가 3,408 ms로 재현되지
  않아 철회했습니다. 벡터화 ON 측정은 흔들림이 커서 여러 묶음으로 확인해야 합니다.
- "corr 0.979는 bfp16 양자화 오차가 쌓인 것으로 보임(미확인)" → 양자화 자체가 아니라
  floor 반올림이 원인이었습니다(커밋 3).
