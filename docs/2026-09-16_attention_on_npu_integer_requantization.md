# 2026-09-16 — attention 의 여섯 matmul 과 그 재양자화를 NPU 로

> **아직 "attention 블럭 전체 NPU 연속 실행" 이 아니다.** 이번에 달성한 것은
> attention 의 **matmul 6 개(Q·K·V·QK^T·PV·proj)와 그 재양자화가 NPU 에서 실행되고,
> 블럭 출력이 ORT 골든과 24,576/24,576 비트 동일** 하다는 것이다.
> **softmax, Q/K/V 뒤 transpose 3 개, proj bias 는 여전히 CPU 에 있다** (§3 표 참고).
>
> 막고 있던 것은 DMA 도, lock 도, softmax 도 아니고
> **aie2p 에 스칼라 float 산술이 아예 없다는 것**이었다.

## 0. 한 줄 요약

| | 이전 | 이후 |
|---|---|---|
| attention 컴파일 (융합 켬) | ✗ `llc` 크래시 → program memory 초과 | **✓** |
| NPU 에서 도는 것 | matmul 만 (재양자화는 CPU 별도 디스패치) | **matmul 6 + 그 재양자화** |
| CPU 에 남은 것 | — | **softmax 1, transpose 3, (경계) 2** |
| proj 코어 ELF | 21,680 B (16 KB 초과) | **8,964 B** |
| QK^T 코어 ELF | 18,664 B (초과) | **5,760 B** |
| PV 코어 ELF | 14,300 B | **5,388 B** |
| NPU vs ORT | — | **비트 동일 (maxdiff 0)** |

새 패스 두 개(`iree-amdaie-integer-requantization`, `iree-amdaie-expand-roundeven`)와
`arith-expand` 파이프라인 편입.

## 1. 출발점과 9/14 문서의 오판 정정

`docs/2026-09-14_attention_npu_elementwise_fusion.md` §4 는 이렇게 적었다:

> `bb_kpadq` + 융합 전체 컴파일 → 12 실패, **전부 FC1**.
> **attention 의 matmul 6 개(Q·K·V·QK^T·PV·proj)는 전부 백엔드를 통과했다.**

**이 판정은 틀렸다.** `--iree-flow-enable-executable-deduplication=false` 때문에 NPU
디스패치가 전부 **하나의 `hal.executable.variant`** 에 들어가는데, FC1 의
`no producer DMA channel` 실패가 그 variant 전체를 중단시켜서 **QK^T·PV·proj 의 코어
컴파일까지 가본 적이 없었다.** 에러 목록에 안 나온 것을 "통과" 로 읽은 것이다.

attention 만 잘라낸 모델(FC1 없음)로 컴파일하자 즉시 다음 벽이 드러났다.

## 2. 실험 대상 만들기

`bert_base_kpadq_int8.onnx` 에서 layer 0 attention 만 추출:

- 입력: `/m/embeddings/LayerNorm/LayerNormalization_output_0` `[1,32,768]`,
  `/m/Where_1_output_0` `[1,1,32,32]` (마스크; 실측 결과 **전부 0** — 문서의 all-ones 관찰과 일치)
- 출력: `/m/encoder/layer.0/attention/output/dense/Add_output_0` `[1,32,768]`
- 노드 46 개: MatMul 6, Softmax 1, Transpose 4, Reshape 4, Pad 1, Q/DQ 22

⚠️ **함정**: `onnx.utils.extract_model` 이 그래프 이름을 `Extracted from {main_graph}` 로
붙인다. 이 **공백**이 디스패치 심볼 → 임시 디렉터리 경로로 전파돼 bootgen 이
`Cannot read file - /tmp/.../Extracted` 로 죽는다. `graph.name` 을 공백 없는 이름으로
바꿔야 한다.

아티팩트: `_local/int8_debug/out/attn0_kpadq_int8.{onnx,mlir}`,
`attn0_x.npy/.bin`, `attn0_mask.npy/.bin`, `attn0_ref.npy`, `attn0_fuse.vmfb`.

## 3. flow 레벨: 9/14 예측대로다

NPU 6 / CPU 6 (CPU 6 개 중 2 개는 잘라낸 경계 때문에 생긴 것):

```
 0  elementwise_32x768_f32xi8        CPU   ← 입력 양자화 + Pad (경계 아티팩트)
 1  batch_matmul 1x32x768x832  Q     NPU → i32
 2                             K     NPU → i32
 3                             V     NPU → i32
 4  elementwise_transpose            CPU   (Q)  ← 재양자화 흡수됨(출력 i8)
 5  elementwise_transpose            CPU   (K)
 6  elementwise_transpose            CPU   (V)
 7  batch_matmul 12x32x32x64  QK^T   NPU → i8    ← 재양자화 흡수
 8  softmax 12x32x32                 CPU   (3-operand)
 9  batch_matmul 12x32x64x32  PV     NPU → 32x12x64xi8  ← transpose+재양자화 흡수
10  batch_matmul 1x32x768x768 proj   NPU → i8    ← 재양자화 흡수
11  elementwise_32x768_f32xi8xf32    CPU   ← proj bias (경계 아티팩트)
```

## 4. 벽 1 — `G_FMINIMUM` 합법화 실패

```
LLVM ERROR: unable to legalize instruction: %329:_(s32) = G_FMINIMUM %328, %55
  (in function: core_0_5)   ← dispatch_7 = QK^T
```

융합된 재양자화의 clamp(`arith.maximumf`/`minimumf`)가 내려간 것. `llc` 로 직접 재보니:

| 형태 | aie2p |
|---|---|
| 스칼라 `llvm.minimum/minnum/maximum/maxnum` | ✗ 전부 |
| **스칼라 `fcmp` + `select`** | **✓** |
| 벡터 `<16 x float>` fminimum / fcmp | ✗ |

`arith-expand` 가 정확히 `cmpf`+`select` 로 푸는데(`MaximumMinimumFOpConverter`,
`fcmp ugt`+`select`, `fcmp uno`+`select` — 이 형태도 aie2p 에서 합법화 확인),
IREE 의 **llvm-cpu 파이프라인엔 있고 AMD-AIE 엔 없었다**. 추가.

## 5. 벽 2 — `G_INTRINSIC_ROUNDEVEN`

```
LLVM ERROR: unable to legalize instruction: %356:_(s32) = G_INTRINSIC_ROUNDEVEN %355
```

`llc` 실측: **`roundeven`/`rint`/`nearbyint`/`round`/`floor`/`ceil`/`trunc` 전부 ✗**,
그리고 upstream MLIR 의 `math-expand-ops` 가 roundeven 을 푸는 경로에 있는
**`copysign` 도 ✗**. 반면 `fptosi`/`sitofp`/`fabs` 는 ✓.

→ 새 패스 `AMDAIEExpandRoundEven`: `|x| >= 2^23` 이면 이미 정수이므로 `x`,
아니면 `t = sitofp(fptosi(x))`, `d = x - t` 로 `|d| > 0.5` 또는
(`|d| == 0.5` 이고 `trunc(x)` 가 홀수)일 때 0 에서 멀어지는 쪽으로 한 칸.
**numpy half-to-even 대비 30,021 케이스(tie·경계·대값 포함) 불일치 0 으로 검증.**

## 6. 벽 3 — program memory 초과, 그리고 진짜 원인

```
[AIE ERROR] _XAie_LoadProgMemSection(): Overflow of program memory
```

코어 ELF 심볼 분해(proj, `core_7_3`):

| 심볼 | 크기 |
|---|---|
| `core_7_3` 본체 | 12,128 B |
| `__divsf3` | 1,440 B |
| `__addsf3` | 1,424 B |
| `__mulsf3` | 1,168 B |
| `__cmpsf2`/`__gesf2`/`__ltsf2`/… | ~1,000 B |
| `__fixsfsi` + `__floatsisf` | 464 B |
| **`generic_matmul_0_outlined` (실제 벡터 matmul)** | **208 B** |

**진짜 matmul 은 208 바이트다.** 나머지 전부가 재양자화다.

`llc` 로 확인한 근본 사실: **aie2p 에는 스칼라 float 산술 명령이 없다.**
곱셈조차 `jl #__mulsf3` 소프트 플로트 호출이다. 즉 float 재양자화를 코어에 융합하는 것은
DSP 위에서 소프트 플로트 에뮬레이션을 돌리는 것과 같다.

왜 스칼라냐: `AMDAIEVectorization.cpp:82-91` 이 양자화 elementwise 를 **의도적으로**
제외한다 — 바디에 `truncf`/`trunci`/`yield` 외의 연산이 하나라도 있으면 건너뛴다
(upstream issue #594: 양자화 elementwise 벡터화가 과도하게 넓은 벡터를 만들어 통째로 껐음).
벡터로 가도 벡터 `fcmp` 가 합법화 안 되므로 더 나쁘다.

## 7. 해법 — 정수 고정소수점 재양자화

attention 의 int8 matmul 꼬리는 전부 이 모양이고, **scale 이 전부 컴파일 타임 상수**다
(attn0 의 QK^T·PV·proj 전수 확인, 동적 scale 없음):

```
sitofp(acc:i32) → [× S1] → [÷ S2] → roundeven → [+ zp] → clamp → fptosi → i8
```

`S = S1/S2` 를 컴파일 때 `M / 2^n` (M ≈ 31 비트)로 바꾸면 실행 때는:

```
acc(i64) × M → n 만큼 산술 우시프트 + round-half-to-even → clamp → i8
```

소수 변환·곱셈·나눗셈이 전부 사라진다. `llc` 실측: **40 instructions, 라이브러리는
`__muldi3` 112 B 하나**(소프트 플로트 5.7 KB 대비).

### 7.1 정확도 — 이 scale·accumulator 범위에서 검증된 bit-exact 치환

세 디스패치의 **accumulator 전 범위 전수 비교**:

| 디스패치 | 실효 scale | M / 2^n | 비교한 값 개수 | 불일치 | 최대 오차 |
|---|---|---|---|---|---|
| 7 QK^T | 0.00384738599 | 2115122816 / 2³⁹ | 2,064,513 | **0** | **0 LSB** |
| 9 PV | 0.01030592807 | 1416435968 / 2³⁷ | 1,032,257 | **0** | **0 LSB** |
| 10 proj | 0.003278376302 | 1802306432 / 2³⁹ | 24,774,145 | **0** | **0 LSB** |

0.003 같은 scale 이 2 의 거듭제곱 분수로 정확히 표현되지 않는 것은 맞지만, `acc` 가 i32 로
유계라 **반올림 결과가 갈릴 만큼 경계에 가까이 가는 입력이 하나도 없다.**

⚠️ **"31 비트 multiplier 면 언제나 비트 동일" 은 아니다.** 여기서 증명한 것은
**이 scale 들과 이 accumulator 범위에서** 같다는 것이다. scale 이 아주 작거나 clamp 범위가
다르거나 accumulator 폭이 다르면 달라질 여지가 있다. 그래서 패스는 이걸 가정하지 않고
**rewrite 전에 매번 검사한다**(§7.3). 검사에 실패하면 float 꼬리를 그대로 둔다.

### 7.2 round-half-to-even 구현

`+0.5` 방식은 음수와 정확히 절반인 값에서 틀린다. 실제로 쓴 것:

```
p   = acc × M                       (i64; acc 가 i32 이고 M < 2^31 이라 항상 안전)
q   = p >> n                        (산술 시프트 = floor)
rem = p & (2^n - 1)                 (floor 시프트라 항상 0 <= rem < 2^n)
up  = (rem > 2^(n-1)) | (rem == 2^(n-1) & (q 가 홀수))
q  += up ; q += zp ; clamp ; trunc
```

음수에서도 floor 시프트 + 마스크 remainder 라 판정이 동일하다.

### 7.3 rewrite 전 동치 검사 (guard)

패스는 `M/2^n` 를 고른 뒤 **그 꼴이 그 꼬리의 float 형과 정말 같은지** 확인하고,
아니면 rewrite 를 건너뛴다. 전 범위를 훑지 않고 싸게 하는 방법:

- 두 형태 모두 accumulator 에 대해 **단조**다 (양수 상수의 f32 곱·나눗셈은
  round-to-nearest 에서 단조이고, 정수화 반올림도 단조).
- clamp 후에는 둘 다 `[clampLo, clampHi]` 값만 갖는 **계단 함수**다.
- 같은 치역을 갖는 단조 계단 함수 둘은 **계단이 같은 자리에서 일어날 때에만** 전 구간에서
  같다. 따라서 clamp 가 가질 수 있는 각 값마다 "그 값에 처음 도달하는 accumulator" 를
  비교하면 충분하다 — 계단마다 **이분 탐색 한 번**이다.

int8 이면 계단이 255 개라 검사 비용이 무시할 수준이다. clamp 폭이 4096 을 넘으면
양자화 꼬리가 아니라고 보고 **거절**한다. float 쪽 평가는 호스트 부동소수점이 아니라
`APFloat` 로 f32 의미를 그대로 재현한다(연산마다 한 번씩 반올림).

단위 테스트(`AMDAIERequantUtilsTest.cpp`)가 이 검사를 실제 스윕과 대조한다 — 실제 세
디스패치, 16 비트 accumulator 전수, tie-to-even 의 음수 쪽, zero point, 포화,
그리고 **일부러 어긋나게 만든 multiplier 를 검사가 거절하는지**까지.

## 8. 결과

코어 ELF (최대):

| 디스패치 | 이전 | 이후 |
|---|---|---|
| proj | 21,680 | **8,964** (−59 %) |
| QK^T | 18,664 | **5,760** (−69 %) |
| PV | 14,300 | **5,388** (−62 %) |
| QKV ×3 (epilogue 없음) | 5,592 | 5,592 |

PV 가 epilogue 없는 QKV 보다 작아진 이유: dispatch_9 에는 **같은 상수로 곱하고 나누는
완전히 중복된 두 번째 재양자화**(`×c` 후 `÷c`)가 붙어 있었는데, 정수화하면 M=1·shift=0 이
되어 저절로 접힌다. (ONNX 의 Q→DQ→Q 가 접히지 않은 흔적.)

실 하드웨어:

```
EXEC @attn0
corr     = 1.0000000000
maxdiff  = 0
exact-equal elements: 24576/24576 (100.0000%)
10 회 반복 → 단일 해시 (a6bf4b2a…)
```

⚠️ 이 비트 동일은 **블럭 출력** 기준이다. 블럭 안에서 softmax 와 transpose 3 개는
여전히 CPU 디스패치이고, 그래서 cross-device staging 이 아직 남아 있다.
"attention 전체가 NPU 에서 돈다" 고 말할 수 있으려면 §10 의 1·2 번이 끝나야 한다.

## 9. 바뀐 파일

| 파일 | 내용 |
|---|---|
| `Transforms/AMDAIEIntegerRequantization.cpp` | 신규. 상수 scale float 재양자화 꼬리 → 정수 (동치 검사 통과 시에만) |
| `Transforms/Utils/AMDAIERequantUtils.{h,cpp}` | 신규. multiplier/shift 선택 + 동치 검사 (단위 테스트 대상) |
| `Transforms/AMDAIEExpandRoundEven.cpp` | 신규. `math.roundeven` → peano 가 고를 수 있는 연산. **NaN 을 재현하지 않으므로 좁은 정수 캐스트로 향하는 경우에만 적용** |
| `Transforms/test/AMDAIERequantUtilsTest.cpp` | 신규 gtest 9 개. f32 의미 vs 정수 변환 직접 대조 |
| `Transforms/Passes.cpp` | 위 둘 + `arith::createArithExpandOpsPass()` 를 LLVM 내림 직전에 삽입 |
| `Transforms/{Passes.td,Passes.h,PassDetail.h,CMakeLists.txt}` | 패스 등록, `MLIRArithTransforms`/`MLIRMathDialect` 의존 |
| `Transforms/test/integer_requantization.mlir` | 신규 lit (상수/동적 scale, f32-rooted 음성 케이스 포함) |
| `Transforms/test/expand_roundeven.mlir` | 신규 lit |

## 10. 남은 것

1. **softmax** — 유일하게 커널이 필요한 것. upstream PR #1414 (`nod-ai/iree-amd-aie`,
   Abhishek-Varma, Peano npu4 softmax ukernel)가 있다. **safe softmax 맞다**
   (pass 1 에서 행 최대값을 빼고 `exp2`; 리뷰 전 버전도 safe 였고 리뷰 커밋은
   리덕션을 shift-butterfly 로 바꾼 최적화뿐). 다만 **bf16 전용**이라
   `matchSoftmaxDAGForUKernel` 이 만드는 심볼(`softmax_<elemtype>_<M>x<N>`)과 맞추려면
   softmax 를 bf16 으로 돌려야 한다 — 팀원이 `truncf` 로 감싸는 방식으로 이미 해결.
   추가로 필요한 것: 우리 타일 크기의 인스턴스(`SOFTMAX_BF16_PER_MxN`), 
   `--iree-amdaie-enable-ukernels` 켜기, softmax 의 `general-copy` 와 matmul 의 pack-peel
   타일 파이프라인 공존 확인, 3-operand(마스크) DMA 채널 제약.
   그리고 **홀로 남는 디스패치라 펜스(`executableIsContractionOrConv`) 확장이 같이 필요하다.**
2. **transpose 3 개** — 커널 불필요. **§12 참고: 백엔드 per-party split 수정은 달성 불가능한
   목표였다.** K 는 `transpose → batch_matmul` 을 `batch_matmul_transpose_b` 로 접어 연산
   자체를 없애는 것이 맞고(백엔드가 이미 transpose_b 를 지원), Q/V 는 헤드별 batch_matmul
   재구성이다 — 둘 다 그래프 수준 작업이다.
3. **FFN 으로 확장** — FC1 의 `no producer DMA channel` (접지 않은 f32 bias, N=3072)이 그대로
   남아 있다. 정수 재양자화는 FFN 에서도 같은 이득을 주지만 그 문제는 별개다.
4. **성능** — 아직 안 쟀다. 2026-08-31 측정에서 벽시계의 ~99 % 가 디스패치당 드라이버
   오버헤드였으므로 이득은 cross-device staging 제거분이다. attention-only 스코프의
   가치가 그걸 깨끗이 잴 수 있다는 점이다.

## 11. 회귀

`ctest -R amd-aie` (201 lit): 실패 8 개.
- `controlcode_lowering`, `insert_cores`, `npu_dma_to_half_dma_cpy_nd`,
  `split_logicalobjfifos_for_connection_reuse` — **float/재양자화 연산이 하나도 없고**
  (grep 0) 내 패스보다 앞 단계의 단일 패스 테스트라 영향 불가. 로컬 커밋
  (`65069ce` per-producer lock 등)에서 온 기존 실패.
- `samples/*_e2e` 4 개 — 실 하드웨어가 필요한데 그 컨테이너에 NPU 장치를 안 붙였다.

⚠️ `PEANO_INSTALL_DIR` 을 안 넘기면 `elf_pm_size`/`ctrlpkt_gen`/
`convert_device_to_control_packets` 가 환경 문제로 추가 실패한다 — 진짜 실패가 아니다.


## 12. K transpose 흡수 — 9/14 가 제안한 백엔드 수정은 틀린 방향이다

`--iree-dispatch-creation-propagate-collapse-across-expands` 를 켜면 flow 는 12 → 11 로
줄고 K transpose 가 사라진다. 백엔드는 여전히 깨진다:

```
'aie.memtile_dma' op cannot split this DMA access pattern into per-party BDs:
  its outermost dimension (size 96, stride 1) is not 4 steps of one 768-element
  slice, needed to synchronize against 4 parties
```

### 12.1 정확히 무엇이 깨지나

실패하는 전송은 K matmul(`dispatch_2`)의 memtile → shim 송신이다:

```mlir
%0   = amdaie.logicalobjectfifo.from_buffers({%buffer_6, %buffer_7},
         {%lock_8, %lock_10, %lock_12, %lock_14}, {...})   // memref<3072xi8, 1>, 4 lock 쌍
%222 = amdaie.connection(%220 /*shim memref<768x1x32xi8>*/, %0 /*memtile*/)
%324 = amdaie.npu.circular_dma_cpy_nd %222([] [] [], [0, 0] [96, 32] [1, 96])
                                            ^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^
                                            shim: 연속   memtile: 전치 읽기
```

- memtile 버퍼 3072 B 는 **독립된 코어 4 개**가 각각 768 B 씩 채운다. `65069ce` 이후
  각자 자기 lock 쌍을 갖는다.
- 소비자 읽기 `sizes=[96,32] strides=[1,96]` 는 **흡수된 transpose 그 자체**다.

### 12.2 per-party 분할이 원리적으로 불가능하다

per-party BD 분할이 성립하려면 **소비자 스트림의 k 번째 조각이 party k 에만 의존**해야
한다. 평범한 gather 는 그렇다(각 party 의 몫이 출력 스트림의 연속 구간). 그런데 전치 읽기는
출력의 **열 하나를 내보낼 때마다 네 party 를 전부 훑는다** — 첫 바이트부터 4 개 전원의
완료가 필요하다. 순서를 보존하는 분할이 존재하지 않는다.

"그럼 BD 하나에서 lock 4 개를 acquire 하면 되지 않나" → **하드웨어가 안 된다.**
`Target/AMDAIERT.cpp:139-180` (`configureLocksAndBd`) 를 보면 BD 디스크립터에는
acquire lock **하나**, release lock **하나**뿐이고, 블럭 안에 `use_lock(Acquire)` 를 여러 개
두면 마지막 것이 앞의 것을 덮어쓴다(조용히 틀림). MLIR 쪽 검증기도 이걸 안 막는다.

⇒ **9/14 문서 §6.2 의 "백엔드 memtile DMA splitting 수정이 필요하다" 는 달성 불가능한
목표였다.** 고칠 수 있는 버그가 아니라 설계 전제가 깨진 것이다.

### 12.3 전치를 shim 쪽으로 옮기는 것도 답이 아니다

원리적으로는 가능하다. 순열을 반대쪽에 실으면 된다(직접 유도함):

| | memtile 읽기 | shim 쓰기 | DDR 결과 |
|---|---|---|---|
| 현재 | `[96,32] [1,96]` | 연속 | `DDR[i*32+j] = buf[i+96j]` |
| 옮기면 | 연속 3072 | `[32,96] [1,32]` | 같음 (`DDR[c*32+r] = buf[r*96+c]`) |

그러면 memtile 읽기가 연속이 되어 4 등분 분할이 된다. **그러나 성능이 반대로 간다.**
전치는 어느 쪽에서든 흩어진 접근을 낳는데, 지금은 그 흩어짐이 **memtile SRAM** 에서 일어나고
DDR 쓰기는 연속이다. 옮기면 타일마다 **1 바이트씩 3072 번 흩어진 DDR 쓰기**가 된다.
없애려던 CPU transpose 보다 느려질 공산이 크다. 즉 **지금 배치가 성능상 옳고**, 문제는
그 배치가 per-party lock 과 양립하지 않는다는 것뿐이다.

### 12.4 맞는 방향: 생산자 출력이 아니라 **소비자 입력**에 접어라

"다음 matmul 이 전치된 것처럼 직접 읽게 한다" 는 원칙을 **반대쪽에** 적용하면 된다.
현재 `propagate-collapse-across-expands` 는 transpose 를 **K matmul 의 출력 쓰기**에
접는다(그래서 4-party memtile 을 건드린다). 대신 **QK^T 의 K 입력 읽기**에 접으면:

- QK^T 의 입력 경로는 shim → memtile 이고 **생산자가 하나**라 memtile objectfifo 의
  lock 쌍이 1 개다(덤프의 입력 memtile 들이 전부 그렇다). per-party 분할 자체가 필요 없다.
- 그리고 더 좋은 것: 접을 필요도 없이 **transpose 를 아예 만들지 않는 길**이 있다.
  QK^T 는 `batch_matmul(A=[12,32,64], B=[12,64,32])` 인데, B 를 K 의 자연 레이아웃
  `[12,32,64]` 로 두고 **transpose_b 형태**로 읽으면 transpose 연산이 소멸한다.
- **백엔드가 이미 transpose_b 를 다룬다**: `AMDAIEPadContractionDispatches.cpp:342,711,743`
  이 "RHS = [N,K] (ONNX Gemm 이 내려오는 꼴)" 를 명시적으로 지원하고,
  `AMDAIEInsertLoopsForVectorization.cpp:205` 도 `isMatmulTransposeB` 를 인정한다.

즉 다음 작업은 **백엔드 lock/DMA 수정이 아니라, `transpose → batch_matmul` 을
`batch_matmul_transpose_b` 로 접는 그래프 수준 작업**이다. Q/V 의 head 재구성(§10-2)과도
같은 성격이라 함께 다루는 게 맞다.


## 13. transpose_b 로 접기 — 그래프 변환

§12 의 결론(생산자 출력이 아니라 소비자 입력에 접어라)을 실제로 해봤다.

### 13.1 한 번에 세 transpose 를 없애는 그래프 변환

`_local/int8_debug/gen_attn_headsplit.py` (신규). QKV projection 을 **head 별 batch matmul**
로 바꾼다:

```
기존: MatMul(x[1,S,K], W[K,H*D]) -> Reshape[1,S,H,D] -> Transpose -> [1,H,S,D]
변경: MatMul(x[1,S,K], W[H,K,D]) -> [H,S,D]            (ONNX MatMul 이 배치를 broadcast)
```

가중치 N 축만 head 로 쪼개는 것이라 **컴파일 타임 상수 변형**이고, per-tensor 양자화
(scalar scale, zp=0)라 스케일도 그대로다. K 축 bias 접합도 K 축이라 영향 없다.

- **Q·V**: 소비되는 레이아웃이 정확히 `[H,S,D]` 라 Reshape·Transpose 가 그냥 사라진다.
- **K**: `[H,D,S]` 로 소비되므로 `Transpose(perm=[0,2,1])` 하나가 남는데, 이걸
  **QuantizeLinear 와 DequantizeLinear 사이**(= int8 텐서 위)에 둔다. per-tensor 양자화는
  원소별이라 수치가 동일하고, 그 결과 IREE 가 보는 것이 **순수 permutation** 이 된다.

**ORT 대비 비트 동일** (24,576/24,576).

### 13.2 IREE 가 transpose_b 로 접는 것 확인

`PropagateLinalgTranspose.cpp` 의 `NamedOpConversion<BatchMatmulOp, inputIdx=1>`
(perm `{0,2,1}`, 등록 위치 `:1149`)이 기대대로 발동했다. flow 에서:

```mlir
%5 = linalg.batch_matmul indexing_maps = [
       affine_map<(d0,d1,d2,d3) -> (d0,d1,d3)>,
       affine_map<(d0,d1,d2,d3) -> (d0,d2,d3)>,   // B[batch, N, K]  <- transpose_b
       affine_map<(d0,d1,d2,d3) -> (d0,d1,d2)>]
     ins(%0, %1 : tensor<12x32x64xi8>, tensor<12x32x64xi8>)
```

⚠️ 이 패턴은 **`linalg.transpose` 가 matmul 입력에 직접 물려 있어야** 하고 perm 이 정확히
`{0,2,1}` 이어야 한다. 원래 그래프가 안 걸렸던 이유가 이것이다 — perm 이 4-D `(0,2,3,1)` 이고
중간에 재양자화가 껴 있었다. head split 이 두 조건을 동시에 만족시킨다.

flow 디스패치: **CPU 6 → 4** (transpose 3 개 소멸, broadcast 1 개 신설), NPU 6 유지.

### 13.3 그런데 npu4 백엔드가 transpose_b 를 못 받는다 — 공백 두 개

| 구성 | 결과 |
|---|---|
| transpose_b + 재양자화 융합 + 벡터화 | ✗ `AMDAIEDistributeL1Allocations`: `'linalg.generic' op inferred input/output operand #1 has shape's dimension #2 to be 4, but found 1` |
| transpose_b + `no-fuse` + 벡터화 | ✗ `AIEVecToLLVM.cpp:744` **assert**: `ShuffleOp currently only supports AIE2.` |
| transpose_b + `no-fuse` + 벡터화 **끄기** | ✓ 컴파일됨 |
| 기준선(transpose_b 없음) + 융합 + 벡터화 | ✓ (오늘 §7 의 결과) |

⚠️ 두 번째는 **§14 에서 해결했다**. 첫 번째(융합 시 allocator shape 오류)는 별개 문제로 남아 있다.

### 13.4 Q/V 만 head split (transpose_b 를 피하는 변형)

K 를 건드리지 않으면 transpose_b 가 필요 없고, 따라서 두 공백 모두 피한다
(`gen_attn_headsplit.py --no-k`).

| | 기준선 | Q/V head split |
|---|---|---|
| 디스패치 | NPU 6 / CPU 6 | NPU 6 / **CPU 5** |
| CPU 구성 | 경계 2, transpose 3, softmax 1 | 경계 2, **transpose 1(K)**, softmax 1, **broadcast 1** |
| NPU vs ORT | 비트 동일 | **비트 동일** (24,576/24,576, 5 회 단일 해시) |

transpose 2 개가 사라지고 broadcast 1 개가 생겨 순 −1 이다. ⚠️ broadcast 는 x 를 12 벌
복제한다(`12x26624xi8` = 319,488 B). 소비자가 2 개라
[[project-k-axis-bias-fold-negative]] 에 적힌 `FormDispatchRegions` 의 "소비자 1 개인
생산자만 융합" 제약에 걸려 별도 디스패치로 남는다. **순이득인지는 아직 안 쟀다** —
디스패치 수는 줄지만 바이트는 비슷하다.

### 13.5 다음

1. **broadcast 를 없애거나 융합**시킬 것. stride-0 DMA 로 표현 가능해 보이므로
   `FormDispatchRegions` 의 단일-소비자 제약이나 백엔드 broadcast 처리를 볼 것.
2. **성능 측정** — Q/V head split 이 실제로 이득인지. [[feedback-npu-perf-measurement-method]].
3. ~~**aie2p aievec 공백** (shuffle 등)~~ → **§14 에서 해결.**

### 13.6 아티팩트

| 파일 | 내용 |
|---|---|
| `_local/int8_debug/gen_attn_headsplit.py` | QKV → head 별 batch matmul 변환 (`--no-k` 로 Q/V 만) |
| `out/attn0_hs_int8.{onnx,mlir}` | 3 개 전부 (transpose_b, 백엔드 막힘) |
| `out/attn0_hsqv_int8.{onnx,mlir}`, `out/attn0_hsqv.vmfb` | Q/V 만 (동작함, 비트 동일) |


## 14. AIE2P 에 `vshuffle` lowering 추가 — transpose_b 가 벡터화 경로로 컴파일된다

§13.3 의 두 번째 벽(`ShuffleOp currently only supports AIE2.`)을 해결했다.

### 14.1 먼저 확인한 것: 무엇이 필요하고, AIE2P 가 그걸 할 수 있는가

**어떤 shuffle 이 필요한가.** transpose_b + no-fuse + 벡터화 IR 을 전부 덤프해 수집한 결과
`aievec.shuffle` 은 **24 개, 전부 동일한 하나의 패턴**이었다:

```mlir
%10 = aievec.shuffle %9 [t8_8x8] : vector<64xi8>     // 단일 피연산자
%16 = aievec.matmul %14, %11, %15 : vector<8x8xi8>, vector<8x8xi8> into vector<8x8xi32>
```

`LowerVectorToAIEVec` 가 전치된 B 피연산자를 **64-lane 벡터 안의 8×8 int8 전치**로 만든다.
`t8_8x8` = 모드 35 (`aievec/AIEVecAttributes.td:55`).

**transpose_b 전용인가.** 그렇다 — 기준선 모델과 Q/V-only 변형에서는 `aievec.shuffle` 이
**0 개**다. 이 lowering 은 transpose_b 경로에만 영향을 준다.

**AIE2P 가 할 수 있는가.** Peano 라이브러리 심볼을 뒤져 `llvm.aie2p.vshuffle` 을 찾았고,
최소 LLVM IR 로 직접 확인했다:

```llvm
declare <16 x i32> @llvm.aie2p.vshuffle(<16 x i32>, <16 x i32>, i32)
%r = call <16 x i32> @llvm.aie2p.vshuffle(<16 x i32> %a, <16 x i32> %b, i32 35)
```
```
llc -march=aie2p  ->   mova r0, #35
                       vshuffle x0, x2, x4, r0
```

시그니처가 AIE2 와 동일하고(`(v16i32, v16i32, i32) -> v16i32`), 모드는 명령 인코딩이 아니라
**레지스터 피연산자**다. 즉 표현 가능하다 — 구현해도 되는 경우다.

⚠️ 다만 **모드 35 가 AIE2P 에서도 같은 8×8 전치를 뜻하는지는 정적으로 확답할 수 없었다**
(모드가 런타임 레지스터 값이라 ISA 문서 없이는 표로 확인 불가). 이건 §14.3 의 비트 동일
결과가 실증한다 — 의미가 달랐다면 출력이 비트 동일할 수 없다.

### 14.2 구현 (transpose_b int8 matmul 에 필요한 것만)

| 파일 | 변경 |
|---|---|
| `aievec/XLLVMOps.td` | `AIEVec2PVectorShuffleIntrOp` 추가 (`llvm.aie2p.vshuffle`) |
| `aievec/AIEVecToLLVM.cpp` | `ShuffleOpConversion` 이 device 로 분기. **assert 를 `notifyMatchFailure` 로 교체** (AIE2/AIE2P 외 타깃은 크래시가 아니라 정상 실패) |
| `aievec/test/test_shuffle.mlir` | lit 2 개 (npu4 단일 피연산자 `t8_8x8`, 두 피연산자) |

AIE2 경로는 한 줄도 바뀌지 않았다. AIE2P 에는 undef 인트린식이 없어(`llvm.aie2p.v16int32`
없음) 단일 피연산자 모드의 무시되는 rhs 는 **평범한 `llvm.mlir.undef`** 를 쓴다 — AIE2 의
`AIEVec2UndefV16I32IntrOp` 를 흉내 내지 않았다.

### 14.3 결과 — 5 단계 기준 통과

| | 결과 |
|---|---|
| `transpose_b + no-fuse + 벡터화` 컴파일 | **✓ assert 없음** |
| NPU vs ORT 골든 | **24,576 / 24,576 비트 동일**, maxdiff 0 |
| 반복 실행 | **10 회 단일 해시** (`a6bf4b2a…`, 기준선과 동일) |

생성된 코어 코드에 `vshuffle` 이 실제로 들어갔는지도 확인했다 — **QK^T 디스패치에만 4 개**,
나머지 5 개 디스패치는 0 개다.

### 14.4 회귀

- `ctest -R amd-aie`: 실패 8 개로 **변경 전과 동일**, 새 실패 없음. 새 lit 2 개 통과.
- 기준선 `attn0_fuse.vmfb`, Q/V-only `attn0_hsqv.vmfb` 재컴파일 시 **MD5 동일**.
- 배포본 `bb_kpadq.vmfb` **MD5 동일**.

### 14.5 남은 것 (별도 패치)

**융합된 재양자화 + transpose_b 의 `AMDAIEDistributeL1Allocations` shape 오류는 아직
그대로다.** 즉 지금은 transpose_b 를 쓰려면 `no-fuse` 여야 하고, 그러면 오늘 §7 의 재양자화
융합 이득을 잃는다. 이건 shuffle 과 섞지 않고 다음에 따로 본다.

⚠️ 그래서 **아직 "attention 전체가 NPU" 가 아니다.** softmax 는 여전히 CPU 이고,
transpose_b 를 쓰는 구성은 재양자화가 CPU 디스패치로 돌아간다.

## 15. 융합 재양자화 + transpose_b 의 allocator shape 오류 — 원인 사슬과 수정

§13.3 의 첫 번째 벽. 원인을 끝까지 추적했고, **§15.4 에서 수정됐다**(Codex 작업, 이 문서 저자가 독립 검증).

```
error: 'linalg.generic' op inferred input/output operand #1 has
       shape's dimension #2 to be 4, but found 1
       (AMDAIEDistributeL1Allocations, QK^T 디스패치)
```

### 15.1 사슬

**① transpose_b 가 packing config 를 바꾼다.** QK^T 디스패치의 `outerPerm` 이
`[[0,1,2], [0,2,1], [0,2,1]]` (기준선) → `[[0,1,2], [0,1,2], [0,2,1]]` 로 바뀐다.
B 가 이미 `[N,K]` 라 swap 이 필요 없어진 것이라 그 자체는 옳다.

**② accumulator 의 zero-fill 이 한 단계 얕은 곳에 남는다.**

| | fill 이 적용되는 텐서 |
|---|---|
| 기준선 | `tensor<1x1x4x4x1x8x8xi32>` — **2 단계 pack 후** |
| transpose_b | `tensor<1x1x4x8x32xi32>` — **1 단계 pack 후**, 뒤에 2 단계 pack 이 따로 온다 |

```mlir
%8 = linalg.fill 0 outs(%7 : tensor<1x1x4x8x32xi32>)
bufferization.materialize_in_destination %8 in writable %alloc_8   // ← fill 의 유일한 사용처
%9 = bufferization.to_tensor %alloc_8
%pack = linalg.pack %9 ... -> tensor<1x1x4x4x1x8x8xi32>
```

**③ 그래서 fill 이 forall 로 융합되지 못한다.**
`AMDAIEFuseFillIntoForall.cpp:56-70` 은 **fill 의 use 가 정확히 하나이고 그게 `scf.forall`
일 때만** 융합한다. 여기서 유일한 use 는 `materialize_in_destination` 이라 조용히 포기한다.
(기준선은 per-thread `linalg.fill` 로 잘 융합된다.)

**④ 버퍼화 후 accumulator 가 forall 밖에서 전체 L1 버퍼로 만들어진다.**

```mlir
linalg.fill 0 -> %alloc_12           // memref<1x1x4x8x32xi32, 1>  (memtile)
linalg.generic (copy) -> %alloc_11   // memtile -> memtile
linalg.pack %alloc_11 -> %alloc_5    // -> memref<1x1x4x4x1x8x8xi32, 2>  (L1, 전체)
  scf.forall (%arg3, %arg4) in (4, 1) {
    %subview_20 = memref.subview %alloc_5[0, 0, %arg3, 0, 0, 0, 0] ...   // per-thread 읽기
```

**⑤ `AMDAIEDistributeL1Allocations` 가 rebuild 경로를 탄다.**
`oldAllocIsDest && !insideCoreForall` 이 참이라 `buildDirectComputationNarrowed`
(커밋 `85afac9`, Method B 의 broadcast-bias 용)로 간다.

**⑥ 그 경로가 출력만 좁히고 입력은 안 좁힌다.** 문서화된 전제가
"데이터가 어느 스레드가 읽느냐에 의존하지 않는다" 인데, broadcast bias 는 **입력 맵이
좁혀지는 차원을 아예 참조하지 않아서** 성립했다. 여기 terminal generic 은 평범한 copy 라
입력 맵이 좁혀진 차원(`d2`, 4 → 1)을 그대로 참조한다:

```
maps = [(d0,d1,d2, d4*8+d5, d3*8+d6),  (d0,d1,d2,d3,d4,d5,d6)]
        입력: %alloc_12 (1x1x4x8x32, 안 좁혀짐)   출력: newAlloc (1x1x1x4x1x8x8, 좁혀짐)
```

`d2` 로부터 도메인 extent 를 4 로 추론하는데 출력의 dim2 는 1 → 검증 실패.

### 15.2 고칠 수 있는 지점 세 곳

| | 어디 | 성격 |
|---|---|---|
| **A** | packing/bufferization 순서 — transpose_b 도 기준선처럼 **2 단계 pack 후에 fill** 하도록 | 가장 깨끗한 결과(memtile 왕복 자체가 사라짐). 가장 침습적 |
| **B** | `AMDAIEFuseFillIntoForall` 이 `materialize_in_destination`/`to_tensor` 를 뚫고 보도록 | per-thread fill 복원. 그 패스는 "fill 1 개, use 1 개" 전제로 쓰여 있어 손대면 파급 있음 |
| **C** | `buildDirectComputationNarrowed` 가 **입력도 스레드 오프셋으로 좁히도록**(subview) | 오류 지점에 가장 가깝고 국소적. 이 실패 계열을 일반적으로 해결. 단 memtile 왕복(fill→copy→pack)은 그대로 남아 낭비 |

C 의 구체안: terminal generic 의 각 입력 차원에 대해 합성된 expr 이 단순 `d_k` 이고
`newShape[k] < oldShape[k]` 이면, 호출부가 이미 갖고 있는 per-thread subview
(`earliest->first`)의 `getMixedOffsets()[k]` 를 오프셋으로 입력에 `memref.subview` 를 건다.
`d_a*T + d_b` 같은 합성 expr 은 inner tile 차원이라 좁혀지지 않으므로 그대로 두고,
좁혀진 차원이 합성 expr 안에 나타나면 **잘못된 코드를 내는 대신 정식 실패**시킨다.

### 15.3 채택된 수정 — C 계열의 변형

`AMDAIEDistributeL1Allocations.cpp` 한 파일, +57 줄.

`buildDirectComputationNarrowed` 앞에 `findConstantFillProducer` 를 두어,
`topPackOp` 의 상류가 **layout 을 바꾸기만 하는 pack/identity copy 를 거쳐 결국 상수
`linalg.fill` 에 도달하는지** 증명한다. 그렇다면 그 fill 을 **최종 per-core packed L1
출력에 직접** 다시 낸다.

근거: fill 은 모든 원소가 같은 값이므로 중간의 layout 변환이 결과에 무의미하다. 따라서
좁혀진 목적지에 같은 값으로 채우는 것과 등가다 — §15.1 ⑥ 의 "입력만 안 좁혀짐" 문제를
**입력을 아예 쓰지 않음으로써** 우회한다.

보수적으로 짜여 있다: DPS writer 가 유일하지 않으면 포기, generic 은 입력 1·출력 1 이고
바디가 `yield %arg0` (순수 copy)여야 하며, `visited` 집합으로 순환을 막는다. 조건을
못 맞추면 기존 broadcast-bias 경로로 그대로 떨어진다.

### 15.4 검증 (독립 재현)

| 항목 | 결과 |
|---|---|
| 융합 transpose_b 컴파일 | **✓** (이전 shape 오류 없음), vmfb MD5 재현 일치 |
| NPU vs ORT 골든 | **24,576 / 24,576 비트 동일**, maxdiff 0 |
| 반복 실행 10 회 | **단일 SHA-256** `af57856e…` |
| memtile 왕복 제거 | ✓ `memref<1x1x4x8x32xi32, 1>` accumulator 버퍼가 사라짐 |
| 코어 ELF (최대) | 5,388 ~ 8,964 B — 전부 16 KB 한참 아래 |
| 회귀: `attn0_fuse` / `attn0_hsqv` / `attn0_hs_nofuse` / `bb_kpadq` | **전부 MD5 동일** |
| `ctest -R amd-aie` | 실패 8 개로 변경 전과 동일, 새 실패 없음 |

⚠️ 원래 변경에는 **테스트가 없었다.** `distribute_l1_allocations.mlir` 에는 rebuild 경로
(새 fill 경로도, `85afac9` 의 bias 경로도) 커버리지가 전혀 없었다. 새 fill 경로에 대한
lit 테스트를 추가했다(`@distribute_l1_constant_fill_through_pack`).

### 15.5 이제 이 구성이 성립한다

```
d0 quantize+pad        CPU   (경계 아티팩트)
d1 broadcast           CPU   ← x 를 12 헤드로
d2 Q  12x32x64x832     NPU → i8   (재양자화 융합)
d3 K                   NPU → i8
d4 V                   NPU → i8
d5 QK^T 12x32x32x64    NPU → i8   ← transpose_b, 재양자화 융합
d6 softmax             CPU   ← 유일하게 커널이 필요한 것
d7 PV  12x32x64x32     NPU → i8
d8 proj 1x32x768x768   NPU → i8
d9 dequant+bias        CPU   (경계 아티팩트)
```

**NPU 6 / CPU 4**, transpose 0 개, 재양자화 전부 NPU. 경계 2 개를 빼면 실질 CPU 는
**broadcast 와 softmax 둘**이다.

⚠️ 여전히 **"attention 전체 NPU" 가 아니다** — softmax 가 CPU 에 있고, broadcast 는
§13.4 에 적은 대로 x 를 12 벌 복제하는 비용이며 순이득 여부는 **아직 측정하지 않았다.**

## 16. softmax NPU 이식 — 구현 전 사전 확인 결과

팀원의 09-11 BERT-base softmax 이식 패치를 우리 `attn0_hs`(transpose_b + 재양자화 융합)에
적용하기 전에 확인해야 할 세 가지를 먼저 봤다. **하나는 해결됐고, 하나는 새 blocker다.**

### 16.1 mask 상수 폴딩 — 해결됨

`/m/Where_1_output_0` 를 **all-zero 상수 initializer 로 굽자** IREE 가 mask add 를 통째로
없앤다. softmax 디스패치가 3-operand → **2-operand** 가 된다:

| | operand |
|---|---|
| 이전 | `readonly 12x32x32xi8` + `readonly 32x32xf32`(mask) + `writeonly` |
| 상수 mask | `readonly 12x32x32xi8` + `writeonly 12x32x32xi8` |

**수치 중립이 실측으로 확인됐다**: 상수 mask 버전의 NPU 출력이 3-operand 버전과
**해시가 동일**(`af57856e…`)하고 ORT 골든과 비트 동일하다. 디스패치 수도 10 개로 같다.
아티팩트: `attn0_hsm_int8.{onnx,mlir}`, `attn0_hsm.vmfb`.

⚠️ 이건 attention-only 라서 가능한 것이다. 전체 BERT 에서는 mask 가 `input_ids` 에서
계산되므로 같은 방식으로 상수가 되는지는 따로 봐야 한다.

### 16.2 NPU 후보 디스패치에 softmax 외에 무엇이 있나

mask 를 접어도 **양자화 체인 두 개가 남는다**:

```mlir
%3 = linalg.generic ins(%0 : tensor<12x32x32xi8>) -> f32   // sitofp -> *0.5927 -> *0.125
%4 = linalg.softmax dimension(2) ins(%3) : tensor<12x32x32xf32>
%5 = linalg.generic ins(%4) -> i8    // /0.00747 -> roundeven -> clamp -> fptosi
```

즉 팀원 패치의 affinity 조건인 "**softmax + 순수 fp cast 하나**" 에 해당하지 않는다.
앞은 dequant(정수→부동 변환 + 스케일 2 회), 뒤는 quantize(나눗셈 + roundeven + clamp +
부동→정수)다.

### 16.3 ⚠️ 새 blocker — 남은 양자화 꼬리가 §6 의 소프트 플로트 문제를 다시 부른다

softmax 를 코어로 올리면 **그 앞뒤 generic 도 코어에서 codegen 돼야 한다.** 그런데
[§6](#6-벽-3--program-memory-초과-그리고-진짜-원인) 에서 확인했듯 **aie2p 에는 스칼라 float
산술이 없다.** 그리고:

- **§7 의 `AMDAIEIntegerRequantization` 은 여기에 적용되지 않는다.** 그 패스는 꼬리가
  `sitofp(정수 accumulator)` 에서 시작할 때만 정수로 바꾼다. 여기 quantize 꼬리의 뿌리는
  softmax 출력이라 **진짜 f32** 다. 이 음성 케이스는 lit 테스트
  `@requant_float_rooted` 로 이미 고정돼 있다.
- `AMDAIEExpandRoundEven` + `arith-expand` 덕에 **컴파일은 되겠지만**, 남는 것은
  `__floatsisf`·`__mulsf3`×2 (dequant) + `__divsf3`·`__subsf3`·`__addsf3` (roundeven 확장
  내부) + `__fixsfsi` 의 소프트 플로트 호출 사슬이다. §6 에서 이 라이브러리는 코어 16 KB
  중 **5.7 KB** 를 차지했다.
- `AMDAIEVectorization` 은 양자화 elementwise 를 **의도적으로 제외**하므로(§6,
  upstream issue #594) 벡터화로 피해갈 수도 없다.

### 16.4 그래서 먼저 정해야 할 것

팀원 패치의 5 개 변경 묶음(commute pass, rank≥2 심볼/bf16 scratch, `96x32` 커널 등록,
general-copy 라우팅, affinity)은 **mask 를 접은 지금 그대로 재사용할 수 있다.** 다만
양자화 꼬리를 어떻게 할지 먼저 정해야 한다:

| | 방법 | 대가 |
|---|---|---|
| **A** | QK^T 의 융합 재양자화가 **bf16 을 내도록** 바꿔 dequant generic 을 소멸시키고, softmax 출력 quantize 는 PV 입력 쪽으로 넘김 | 경계 바이트 2 배(i8→bf16). QK^T 의 정수 재양자화 이득 일부 상실 |
| **B** | softmax ukernel 을 **i8 in / i8 out** 으로 확장해 스케일까지 커널 안에서 처리 | PR #1414 ABI 확장. 커널 작업 |
| **C** | 소프트 플로트 꼬리를 그대로 감수 | program memory 초과 위험(§6 전례), 성능 손실 |
| **D** | softmax 만 별도 디스패치로 떼어 양자화는 CPU 에 남김 | 디스패치가 늘어 목적(경계 제거)에 반함 |

**아직 구현하지 않았다** — 16.3 이 원래 계획에 없던 항목이라 방향을 먼저 정하는 게 맞다.

참고: 팀원 BERT-base 측정에서 softmax NPU 는 독립 디스패치라 CPU 보다 약 9.3% 느렸다.
이번 목적은 성능이 아니라 attention 의 CPU 경계 제거와 기능 검증이다.

## 17. B 안 — int8 in/out softmax ukernel: 커널 완성·검증, 통합은 남음

§16.4 에서 고른 B 안(스케일을 커널 안으로 접어 소프트 플로트를 아예 없앰)의
**커널 부분을 끝냈다.** 컴파일러 통합(5 개 변경 묶음)은 아직이다.

### 17.1 결정적 발견 — 스칼라 int↔float 은 소프트 플로트가 아니다

§6 의 교훈("aie2p 에 스칼라 float 산술이 없다") 때문에 B 안이 성립할지가 관건이었는데,
**변환만은 네이티브 명령이다**:

| | 명령 |
|---|---|
| `fix2float(int, sft)` (`aie2p_cnvf2f.h`) | `mov r0, r1.fx2flt, s2` — **1 명령** |
| `float2fix(float, sft)` | `mov r0, r1.flt2fx, s3` — **1 명령** |
| `bfloat16_to_int(v16bfloat16, sft)` (`aie2p_vadd.h:1470`) | `vfloor.s32.bf16` — 벡터 |
| `ssrs(v32acc32, sft, sign)` (`aie2p_srs.h`) | 정수 shift-round-**saturate** |

실측: int8 → bf16 스칼라 변환 루프가 **libcall 0 개**, 총 35 instructions.
⚠️ 반면 **벡터 int↔float 변환은 없다**(전체 builtin 153 개를 훑어 확인). 그래서 입력
확장만 스칼라 루프이고 나머지는 전부 벡터다.

### 17.2 스케일을 접는 방법 — 추가 연산이 0 이다

```
out = clamp(round(softmax(q * s_in) / s_out))
```

- **입력 스케일**: 커널은 이미 `exp2((x - m) * log2e)` 를 계산한다. `x = q * s_in` 이므로
  `(x - m) * log2e = (q - m_q) * (s_in * log2e)` — **`log2e` 상수만 바꾸면 된다.**
  int8 값은 bf16 에 정확히 표현되므로 `q - m_q` 도 정확하다.
- **출력 스케일**: 커널은 이미 `e * (1/sum)` 을 곱한다. 거기에 `1/s_out` 을 곱하고 `+0.5`
  를 더하는 것을 **하나의 `mac_elem_32`** 로 합쳤다. 그 뒤 `bfloat16_to_int`(floor) +
  `ssrs`(saturate) 로 int8 이 된다 — clamp 가 하드웨어로 공짜다.
- 행 최대값의 시프트도 스칼라 곱 대신 `sub(zero_acc, mul_elem_32(m_v, log2e_v))` 벡터
  연산으로 만들어, **행 루프 안에 스칼라 float 산술이 하나도 없다.**

### 17.3 결과

`Target/uKernels/npu4/peano/softmax.cc` — PR #1414 기반에 추가:

| 심볼 | 크기 |
|---|---|
| `softmax_i8_96x32`, `softmax_i8_32x32` | 각 **1,440 B** |
| `softmax_bf16_96x32` (계획상 필요) | 624 B |

**소프트 플로트 호출 0 개** (`llvm-objdump` 로 확인). 비교: C 안이 끌어왔을 라이브러리만
5.7 KB.

### 17.4 수치 검증 (실제 데이터, 시뮬레이션)

실제 QK^T int8 출력을 ORT 로 tap 해 커널 알고리즘(bf16 반올림 포함)을 재현하고, 같은
지점의 ORT int8 softmax 출력과 비교:

| | |
|---|---|
| 정확히 일치 | **12,031 / 12,288 (97.91 %)** |
| ±1 LSB 이내 | **12,288 / 12,288 (100 %)** |
| maxdiff | **1** (mean 0.021) |
| corr | **0.99982** |

bf16 의 8 비트 가수 때문에 출력 상위(127 근처)에서 ulp 가 0.5 라 ±1 LSB 가 나오는 것이며,
이는 bf16 softmax 의 구조적 한계다. **비트 동일은 애초에 목표가 아니다.**

### 17.5 남은 통합 (미착수)

| | 무엇 | 확인된 장애물 |
|---|---|---|
| 1 | affinity 를 softmax 디스패치까지 확장 | 플래그 배선 경로는 확인됨 — `enableAMDAIEUkernels` 가 `demoteContractionInputsToBf16` 와 같은 `AMDAIEOptions` 구조체(`AIETarget.h:106`)에 있어 `PluginRegistration.cpp` 에서 패스 옵션으로 넘길 수 있다 |
| 2 | `LowerToUKernels` 에 **양자화 softmax DAG 매처** | 현 `matchSoftmaxDAGForUKernel` 은 2-D 가정(`getDimSize(0/1)`)이고 스케일을 인자로 넘기지 않는다. `dequant generic → softmax → quantize generic` 을 한 ukernel 호출로 접는 새 매처가 필요 |
| 3 | 타일링 전략 rank≥2 일반화 | ⚠️ `KernelDispatch.cpp:851` 에 **`assert(inputShape.size() == 2)`** — 우리 softmax 는 rank 3 |
| 4 | 모듈이 pack-peel 이어도 이 디스패치만 GeneralCopyPipeline 로 | 미조사 |
| 5 | bf16/i8 scratch 를 L1 예산에 반영 | 커널이 `kMaxRowLanes` 스크래치 2 개를 쓴다 |

⚠️ 1·3·4·5 는 팀원 09-11 패치가 이미 하는 일과 같다. **그 패치를 받을 수 있으면 받아서
쓰는 게 맞고**, 2 만 B 안 전용으로 새로 만들면 된다. 지금은 그 패치가 우리 트리에도
어느 브랜치에도 없다.

## 18. softmax 를 NPU 로 — 컴파일러 통합: 어디까지 됐고 어디서 멈췄나

§17 의 커널 위에 컴파일러 통합을 붙였다. **매처·affinity·타일링은 동작하고, 최종 컴파일은
아직 안 된다.** 멈춘 지점과 이유를 남긴다.

### 18.1 사전 정리 — mask 상수 폴딩 (§16.1 의 실행)

`attn0_hsm_int8` = `attn0_hs_int8` 에서 mask 를 all-zero 상수로 구운 것.
softmax 디스패치가 3-operand → **2-operand** 가 되고, NPU 출력 해시가 3-operand 판과
**동일**(`af57856e…`)하다. 수치 중립이 실측으로 확인된 단순화다.

### 18.2 동작하는 것

**(a) 양자화 softmax DAG 매처** — `Utils/AMDAIESoftmaxUtils.{h,cpp}` 신규.
`matchQuantizedSoftmax(quantizeOp)` 가 `dequantize -> softmax -> quantize` 를 인식하고,
`AMDAIELowerToUKernels.cpp` 의 `matchQuantizedSoftmaxDAGForUKernel` 이 세 op 을
`softmax_i8_<M>x<N>` 호출 하나로 바꾸며 두 스케일을 인자로 넘긴다.
`M = prod(리덕션 아닌 차원)`, `N = 마지막 차원`.

판정은 엄격하다: 모든 op 단일 사용, identity map, 상수 스케일, 마지막 차원 리덕션,
clamp 가 정확히 `[-128, 127]`. 뿌리를 **quantize** 에 두어(마지막 op) 교체가 자연스럽다.

⚠️ 실제 IR 에는 상수 폴딩된 mask 의 잔여물 `addf 0.0` 이 dequant 꼬리에 남는다.
**softmax 는 시프트 불변**이므로 상수 덧셈은 수학적으로 무해하고, 매처가 이를 받아들인다
(스케일만 곱해 모으고 덧셈은 무시).

lit: `test/lower_to_ukernel.mlir` 의 `@quantized_softmax` — **통과**.

**(b) 조건부 affinity** — `isQuantizedSoftmaxOnly()` 가 executable 이 그 DAG **뿐**인지
본다(softmax 1 + generic 정확히 2 + 다른 linalg op 없음 + 매처 성공).
`AMDAIEAssignDeviceAffinities` 에 `enable-softmax-ukernel` 옵션을 추가하고
`PluginRegistration.cpp` 에서 `--iree-amdaie-enable-ukernels` 를 파싱해 넘긴다.

| | softmax 디스패치 |
|---|---|
| 플래그 없음 | **cpu** |
| `--iree-amdaie-enable-ukernels=softmax` | **npu** |

**(c) `AMDAIETileAndFuse` 의 소비자 융합** — `scf::tileConsumerAndFuseProducersUsingSCF` 는
생산자만 융합한다. 타일링 직후 루프의 `parallel_insert_slice` 에
`scf::tileAndFuseConsumerOfSlices` 를 적용하는 `fuseConsumersIntoLoops()` 를 추가했다.
대상은 **단일 사용 + elementwise linalg op** 으로 한정하고 copy/pack/unpack 은 제외한다.

`fuse-consumers` 옵션(기본 **false**)으로 노출하고 general-copy 에서만 켰다. pack-peel 은
자체 `FuseConsumerIntoLoop` 를 뒤에 갖고 있어 무조건 켜면 모든 matmul 산출물이 바뀐다.
`elementwise producer -> root -> elementwise consumer` 는 normalization,
activation-quantization 에도 반복되는 모양이라 기능 자체는 범용이다.

**(d) 타일 루트 유지** — general-copy 에서 `tileElementwise = isElementwiseOp`.
디스패치 자체의 루트가 elementwise 가 아니면 마지막 quantize 를 건너뛰고 **softmax 를
루트로** 잡는다. 그러지 않으면 quantize 만 타일되고 리덕션이 untiled 로 남는다.

**(e) 승격을 체인 경계로 제한** — `AMDAIEInsertCopyOps` 와
`AMDAIEBufferizeToAllocation` 양쪽에 `isInternalChainEdge()` 를 넣어, 같은 블록의 compute
op 이 만든 값(타일 내부 edge)은 버퍼를 주지 않는다. `tensor.extract_slice` 를 통과해서 본다.
이게 L2 버퍼와 copy 를 만들어 **레벨-1 생산자 융합을 막던** 원인이었다
(융합 제어 함수가 `linalg::CopyOp` 를 명시적으로 제외한다).

**(f) 죽은 잔여 op 승격 금지** — 타일링은 원본을 남기고 use 만 바꾼다. 정규화가
버퍼화 전까지 억제되므로 그 클론이 살아 있는데, 승격하면 타일된 것 옆에 **untiled 복사
사슬이 통째로** 생긴다. 결과가 전부 미사용인 op 은 건너뛴다.

**(g) 파이프라인 버그 2 개** — `AMDAIEInsertCopyOps` 가 forall 의 **첫** shared output 을
복사 대상으로 가정했다. 소비자 융합으로 출력이 둘이 되거나 중간 결과의 원소 타입이 다르면
(f32 softmax vs i8 루프 출력) 잘못된 슬라이스를 만든다. 이제 그 op 자신의
`parallel_insert_slice` 를 찾아 목적지를 고르고, 없으면 새 버퍼를 쓴다.

### 18.3 멈춘 지점

최종 컴파일이 `AMDAIEDistributeL1Allocations` 에서 실패한다:

```
'linalg.copy' op inferred input/output operand #1 has shape's dimension #0 to be 12, but found 3
```

원인 사슬은 여기까지 좁혀졌다. 소비자 융합 뒤 루프 인터페이스에 **잉여 결과**가 남는다:

```mlir
%7:2 = scf.forall ... shared_outs(%arg1 = %4 /*f32*/, %arg2 = %3 /*i8*/) {
  %8 = dequant ; %9 = softmax ins(%8) ; %10 = quantize ins(%9)
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %9  into %arg1   ← 루트 자신의 결과, 잉여
    tensor.parallel_insert_slice %10 into %arg2
  }
}
iree_tensor_ext.dispatch.tensor.store %7#1, ...
```

**`%7#0` 은 use-chain 을 끝까지 따라가도 사용처가 없다** — dispatch 출력에 도달하지 않는다.
`%9` 는 루프 **안에서** `%10` 이 소비하므로, 이 escape 는 소비자 융합이 남긴 잉여다.
그대로 두면 `shared_out` 하나와 write-back 하나가 남고, copy/승격이 거기에 버퍼를 만들고,
레벨-1 이 체인을 다시 타일하는 대신 그 버퍼를 읽는다.

**한 일**: `dropDeadForallResults()` — 살아 있는 결과만 가진 `scf.forall` 을 새로 만들고
바디를 `mergeBlocks` 로 옮긴다. 전처리 `detachDeadIterArg()` 는 write-back 을 지운 뒤 남는
배관을 푼다(결과가 안 쓰이는 `insert_slice` 제거, 루트의 **destination** 으로 쓰이는
`extract_slice` 를 `tensor.empty` 로 대체 — destination 은 전부 덮어써지므로 안전하고,
DPS init 용도인지 검사한다).

⚠️ **`scf.forall` 의 정규화 패턴은 일부러 쓰지 않는다.** 시도했더니
`ForallOpSingleOrZeroIterationDimsFolder` 가 **단일 반복 루프까지 접어** 타일링 구조가
통째로 사라졌고, forall 이 교체되면서 뒤따르는 GPU 매핑 코드가 dangling pointer 로
**세그폴트**났다. 그래서 명시적 재작성을 쓰고, 그 이유를 코드 주석에 남겼다.

**결과**: write-back 제거는 된다(위 IR 에서 f32 insert 가 사라짐). 그러나 **재작성이 아직
발동하지 않는다** — `%arg1` 에 배관이 한 겹 더 있다:

```mlir
%inserted_slice   = tensor.insert_slice %9 into %arg1[...]
%extracted_slice_9 = tensor.extract_slice %inserted_slice[...]   ← 이것도 죽었다
```

`insert_slice` 의 결과가 **또 다른 죽은 `extract_slice`** 에 쓰여서 단일 단계 검사가
거절한다. 전이적으로 판정하도록 확장했더니(`isDeadSliceChain`, `SetVector` 로 중복 제거까지)
**세그폴트**가 나서, 추측으로 좇는 대신 직전의 동작하는 상태로 되돌렸다.
크래시 원인은 미확인 — 의심 후보는 `scf::ForallOp` 빌더(`bodyBuilderFn = nullptr`)가 바디
블록/terminator 를 만드는 방식과 `mergeBlocks` 의 상호작용이다.

### 18.4 다음에 여기서 시작할 것

1. **심볼 있는 스택 트레이스부터** 뽑을 것(`LLVM_SYMBOLIZER_PATH`). 위 크래시를 추측으로
   좇지 말 것 — 오늘 두 번 그렇게 시간을 썼다.
2. 전이적 죽은 슬라이스 사슬을 처리해 재작성을 발동시킬 것.
3. 그 뒤 검증 순서: ① 레벨-0 forall 이 i8 출력만 반환 → ② f32 가 루프 밖 L2
   alloc/copy/subview 로 이어지지 않음 → ③ 레벨-1 안에 dequant/softmax/quant 가 전부
   `[3,32,32]` → ④ `softmax_i8_96x32` 치환 → ⑤ NPU vs ORT(비트 동일 요구 금지, corr/maxdiff
   기록) + 반복 단일 해시.

### 18.5 회귀 (매 단계 확인함)

- `ctest -R amd-aie`: 실패 **8 개**로 오늘 내내 불변, 새 실패 없음.
  (중간에 `insert_copy_ops` lit 을 한 번 깨뜨렸다가 오프셋 계산을 원래대로 되돌려 복구했다.)
- `attn0_hs_fuse`, `attn0_hsm`, `attn0_fuse`, `attn0_hsqv`, `attn0_hs_nofuse` 재컴파일
  **전부 MD5 동일**.
- lock 관련 코드는 건드리지 않았다.

⚠️ `tileElementwise` 변경이 general-copy 전체에 영향을 주는데도 회귀가 없는 이유는 현재 이
파이프라인을 타는 다른 디스패치가 없기 때문이다. 순수 elementwise 디스패치가 들어오면
`isElementwiseOp` 분기가 종전 동작을 유지한다.

### 18.6 바뀐 파일 (오늘 전체, 누적)

신규 10, 수정 22, 약 +799/−49 줄.

| 파일 | 내용 |
|---|---|
| `Target/uKernels/npu4/peano/softmax.cc` | PR #1414 + **int8 in/out 커널**(§17) |
| `Transforms/Utils/AMDAIESoftmaxUtils.{h,cpp}` | 양자화 softmax DAG 매처(공유) |
| `Transforms/AMDAIELowerToUKernels.cpp` | 3 op → ukernel 호출 |
| `Transforms/AMDAIEAssignDeviceAffinities.cpp` | 조건부 NPU 배치 |
| `Transforms/AMDAIETileAndFuse.cpp` | 소비자 융합 + dead-result 재작성 |
| `Transforms/AMDAIEInsertCopyOps.cpp` | 목적지 선택 수정, 체인 경계만 승격 |
| `Transforms/AMDAIEBufferizeToAllocation.cpp` | 체인 경계만 승격 |
| `Transforms/Passes.{td,h,cpp}`, `PluginRegistration.cpp` | 옵션·배선 |
| `Transforms/{AMDAIEIntegerRequantization,AMDAIEExpandRoundEven}.cpp` + `Utils/AMDAIERequantUtils.*` | §4-7 의 정수 재양자화 |
| `aievec/{XLLVMOps.td,AIEVecToLLVM.cpp}` | §14 의 aie2p `vshuffle` |
| lit/gtest 6 개 | 위 각각 |

## 19. softmax 결과 버퍼 추적 — 원인이 하나로 수렴했다

§18.3 에서 멈춘 지점을 "루프가 softmax 결과 버퍼를 만드는 것" 으로 잡고 파고든 결과.
**세 겹을 더 벗겨냈고, 남은 하나가 §18.3 의 잉여 loop result 와 같은 뿌리임이 확인됐다.**

### 19.1 확인 — 오류는 정말 그 버퍼였다

실패 직전 IR:

```mlir
%alloc_10 = memref.alloc() : memref<12x32x32xf32, 1 : i32>   ← softmax 결과 버퍼 (L2)
scf.forall (0 to 12 step 3) { ... linalg.copy %alloc -> subview_17 }   ← 루프가 write back
%alloc_11 = memref.alloc() : memref<12x32x32xf32, 2 : i32>
linalg.copy ins(%alloc_10) outs(%alloc_11)
linalg.generic (quantize)                                    ← 12x32x32 로 루프 **밖**
```

레벨-1 에서 quantize 가 루프 밖으로 빠진 이유가 이것이다. softmax 결과가 L2 로 승격되면서
둘 사이에 `linalg.copy` 가 끼고, **융합 제어 함수가 `linalg::CopyOp` 를 명시적으로
제외**하므로 레벨-1 의 소비자 융합이 그 너머를 못 본다.

### 19.2 벗겨낸 세 겹

**(a) 죽은 잔여 사슬이 전이적이다.** 타일링이 남기는 untiled 원본들은 서로가 서로의
사용처라서 **마지막 하나만 직접적으로 죽어 있다.** 한 단계 검사로는 못 걸러서, 승격이
타일된 것 옆에 12×32×32 복사 사슬을 통째로 만들었다. `AMDAIEInsertCopyOps` 에서 부작용
없는 op 에 대해 **전이적 deadness 를 고정점까지 전파**하도록 바꿨더니 함수 스코프의 잔여
복사가 전부 사라졌다.

**(b) `promoteInputs` 가 destination 도 승격한다.** 입력 쪽 판정(`isInternalChainEdge`)은
**생산자**를 보는데, destination 의 생산자는 `tensor.empty` 라 절대 내부 edge 로 안 잡힌다.
destination 에는 반대 방향 판정(`resultsStayInBlock` — 결과가 타일을 벗어나는가)이 맞다.
적용하니 dequant 의 destination 버퍼가 사라졌다.

**(c) 체인 edge 에는 슬라이스가 끼어 있다.** 융합이 `tensor.extract_slice` 를 남기므로
양쪽 판정 모두 그것을 통과해서 봐야 한다(`AMDAIEInsertCopyOps` 와
`AMDAIEBufferizeToAllocation` 양쪽).

### 19.3 남은 하나 — §18.3 과 같은 뿌리

softmax 의 결과 `%13` 의 사용처가 `{quantize, tensor.insert_slice}` 다. 그
`insert_slice` 가 **§18.3 의 잉여 f32 loop result 로 가는 write-back** 이다. 그래서
`resultsStayInBlock(softmax)` 가 거절하고, destination 과 결과가 여전히 승격된다.

즉 **버퍼를 만드는 직접 원인은 잉여 loop result 하나로 수렴했다.**
`dropDeadForallResults` 가 그것을 없애면 (a)(b)(c) 위에서 이 마지막 승격도 자동으로
사라진다. 그 재작성은 `insert_slice -> extract_slice` 전이적 죽은 사슬 처리가 필요하고,
그걸 확장하다 세그폴트가 났다(§18.3).

⚠️ 여기서 배운 것: 추측으로 좇는 대신 `--mlir-print-ir-after-all` 로 **버퍼를 만든 패스를
이분 탐색**했더니 한 번에 `AMDAIEInsertCopyOps` 로 특정됐다. 앞선 두 번의 세그폴트 추적에
그렇게 했어야 했다.

### 19.4 회귀 (이 절의 변경 포함)

- `ctest -R amd-aie` 실패 **8 개**로 불변, 새 실패 없음.
- `attn0_hs_fuse` / `attn0_hsm` / `attn0_fuse` / `attn0_hsqv` / `attn0_hs_nofuse`
  재컴파일 **전부 MD5 동일**.

### 19.5 다음

1. `dropDeadForallResults` 의 전이적 죽은 슬라이스 처리 — **심볼 있는 스택 트레이스부터**.
2. 그 뒤 §18.4 의 ①~⑤ 검증 순서를 그대로 밟을 것.
