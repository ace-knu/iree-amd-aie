# residual add + LayerNorm 을 NPU 단일 디스패치로 (2026-09-21)

목표(M4): attention 출력 → **첫 번째 residual add** → LayerNorm 까지를 CPU 를 거치지
않는 하나의 NPU 디스패치로 만든다. 9/18 까지는 순수 LayerNorm 만 NPU 로 내려갔고,
residual 이 붙으면 디스패치가 융합은 되지만 `'amdaie.connection' op no producer DMA
channel available` 로 빌드가 깨졌다. 오늘 그 원인을 확정하고 두 가지를 고쳐 끝냈다.

## 1. 채널 고갈의 진짜 원인 — 두 개였다

실패 지점(`AMDAIEAssignChannels`) 직전 IR 을 읽어 보니 문제는 "γ/β 때문에 입력이 3 개"
가 **아니었다**. 코어당 입력은 이미 2 개였고, 대신 **타일이 배정되지 않은 L1 버퍼 사슬**이
따로 떠 있었다.

```
%lof = amdaie.logicalobjectfifo.from_memref %alloc_7, {}    <- 타일 없음
  : memref<32x768xi16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<24576xi16, 2 : i32>, 2>
%9  = amdaie.connection(%lof, %lof_9)   <- L1 -> L1, 양쪽 다 타일 없음
%10 = amdaie.connection(%lof_0_1_6, %lof)
```

`memref<32x768xi8, 2 : i32>` 4 개와 `memref<32x768xi16, 2 : i32>` 2 개 —
코어 로컬 메모리(64 KB)보다 큰 버퍼가 **아무 코어에도 속하지 않은 채** L1 주소공간에
만들어져 있었다. 소유 타일이 없으니 채널을 받을 수 없고, 그게 곧 에러였다.

원인: residual producer 가 **블록 레벨 forall 에만** 융합되고 코어 레벨 forall 로는
내려가지 못했다. 그 사이에 `AMDAIEInsertCopyOps` 가 producer 결과와 custom_op 사이에
복사를 끼워 넣어 생산자-소비자 사슬이 끊겼기 때문이다. 9/17 커밋 `ed49da3` 이 넣어 둔
"사슬 내부 edge 는 승격하지 않는다" 규칙이 `linalg.generic`/`linalg.softmax` 만 알고
`iree_linalg_ext.custom_op` 은 몰랐다.

그리고 사슬을 제대로 이어 붙이면 이번엔 **진짜 채널 한계**가 나온다. aie-rt 레지스터
초기화 표(`xaie2pgbl_reginit.c`)로 확인한 AIE2P(npu4)의 타일별 DMA 채널 수:

| 타일 | S2MM | MM2S |
|---|---|---|
| memtile | 6 | 6 |
| **core tile** | **2** | **2** |
| shim tile | 2 | 2 |

residual 을 코어에서 계산하면 코어가 받아야 할 텐서가 p, x, γβ 로 **3 개** → 코어의
S2MM 2 개를 넘는다. 우회로가 없다. 그래서 γ/β 는 DMA 로 들어올 수 없다.

## 2. 고친 것 두 가지

### (a) custom_op 을 사슬의 일부로 인정

`AMDAIEInsertCopyOps` 의 `isInternalChainEdge` / `valueStaysInBlock` 과
`AMDAIEBufferizeToAllocation` 의 같은 이름 두 함수에 `IREE::LinalgExt::CustomOp` 을
추가했다. 이제 producer → custom_op edge 는 승격되지 않고, 다음 타일링 레벨이 둘을
같은 코어 forall 안으로 융합한다. 타일 없는 L1 버퍼 사슬이 통째로 사라진다.

### (b) γ/β 를 코어 로컬 상수로

γ/β 는 더 이상 `custom_op` 의 operand 가 아니다.

- `AMDAIERaiseLayerNorm` 이 raise 시점에 γ, β 상수를 읽어(inline `DenseElementsAttr`
  와 resource blob 둘 다 지원) `gamma/output_scale`, `beta/output_scale` 을 bf16 로
  접어 `[2, 768]` **`DenseElementsAttr`** 하나로 만들고, 이를 `amdaie.layernorm_gamma_beta`
  속성으로 op 에 붙인다. 상수가 아니면 **아무것도 고치지 않고 failure** 를 돌려준다.
- `AMDAIELowerToUKernels` 가 그 속성을 `memref.global "private" constant` 로 만들고
  `memref.get_global` + `bufferization.to_tensor ... restrict` 로 커널 operand 에 넣는다.
  같은 내용이면 global 하나를 공유한다.
- `AMDAIEInsertCores` 는 코어 바깥에 남은 `memref.get_global` 을 코어 안으로 복제한다
  (`amdaie.core` 는 블록째 `aie.core` 로 splice 되므로 바깥 값을 읽으면 남겨진다).
- `AMDAIELowerToAIE` 는 코어가 참조하는 `memref.global` 을 `aie.device` 안으로 옮긴다.
  shim global 이 이미 가는 그 자리다.

주소공간은 **붙이지 않는다**. `2 : i32` 를 붙였더니
`AMDAIENormalizeAddressSpaces` 가 `memref.get_global` 결과 타입만 기본 주소공간으로
바꾸고 `memref.global` 은 그대로 둬서 검증이 깨졌다. 코어 프로그램이 이름 붙일 수 있는
메모리는 코어 로컬 메모리뿐이라 주소공간은 정보를 담지 않는다.

배치는 per-core 링커 스크립트가 한다(`AMDAIETargetLdScript.cpp`): `data` 영역이
`max(stack_size, 마지막 버퍼 끝)` 뒤에서 시작하고 `.rodata` 가 거기로 들어간다.
따라서 채널도, 전송도 쓰지 않는다.

### (c) 타일 크기 계산을 실제 예산으로

`KernelDispatch.cpp` 의 LayerNorm 설정을 두 가지로 다시 썼다.

- **L1 바이트**: DMA 를 건너는 operand 는 2 중 버퍼, 융합된 producer 가 쓰는 중간값은
  1 중, 코어 로컬 상수와 스택은 고정 비용. residual 이 있으면
  `768*(i8*2 + i8*2 + i16*2 + i8*2) - 768*i16 = 6144 B/row`, 고정 `4096 + 3072 = 7168 B`
  → `maxRowsPerCore = (65536 - 7168) / 6144 = 9` → `findLargestFactor(32, 9) = 8`.
- **memtile 채널**: 코어당 스트리밍 입력 `k` 개를 `c` 개 코어에 뿌리면 memtile MM2S 가
  `k*c`, S2MM 이 `k + c` 필요하고 둘 다 6 이하여야 한다. residual 이면 `k=2` → `c ≤ 3`,
  행을 정확히 나누는 2 의 거듭제곱으로 내려 `c = 2`.

결과 구조 (32 행):

```
scf.forall (block) in (2)          # 16 행씩, L2 16x768 3 개
  scf.forall (core) in (2)         # 8 행씩, 코어 2 개
    amdaie.core(%tile, in : [p, x], out : [y]) {link_with = "layernorm.o", stack_size = 4096}
      linalg.generic                    # residual: h = (a*Ma + b*Mb + 128) >> 8, i16
      ukernel.generic "layernorm_i16_8x768" ins(h, @__amdaie_core_constant_0) outs(y) (eps)
```

코어당 L1: `aie.buffer` 49152 B(6144 B × i8 3 종 × 2 depth + 12288 B h) + `.rodata`
3072 B + 스택 4096 B = **56320 / 65536 B (86 %)**. 여유 9216 B.

## 3. 검증

### 합성 모델 `ln_m2` (i8 두 개 → DQ,DQ → Add → LN → Q, [1,32,768])

| 항목 | 값 |
|---|---|
| 디스패치 | **NPU 1 개** (전부 `@npu`) |
| 커널 | `layernorm_i16_8x768`, 코어 2 개 × 블록 2 회 |
| 5 회 실행 | 해시 1 종 (`cbca3a0e…`) |
| float64 레퍼런스 대비 | exact 87.87 %, **≤1 LSB 100.00 %**, maxdiff 1, corr 0.9998805 |

### 실제 BERT layer-0 `attn0res_hsqk4_int8`

| | LN on CPU (대조) | **LN on NPU** |
|---|---|---|
| 디스패치 | NPU 7 / CPU 4 (11) | **NPU 8 / CPU 2 (10)** |
| LN 디스패치 | `reduction_32x768_f32` × 2 (CPU) | `reduction_DxD_f32` × 1 (NPU) |
| 실행 해시 | 3 회 1 종 | **5 회 1 종** |
| ORT 대비 exact | 97.66 % | 97.00 % |
| ORT 대비 ≤1 LSB | 100.00 % | **100.00 %** |
| maxdiff | 1 | 1 |
| corr | 0.9981970 | 0.9976914 |

남은 CPU 2 개는 모델 입출력 경계(f32→i8 양자화, K-pad `slow_memcpy`)로, attention-only
기준선과 같다. **residual add 와 LayerNorm 은 CPU 에 하나도 남지 않았다.**

### 소프트 플로트 / 스택

- 코어 LLVM 모듈의 외부 호출: `layernorm_i16_8x768` 4 회 + `llvm.*` 인트린식뿐.
  `__mulsf3` 계열 **0 개**.
- `stack_size = 4096` 이고 `XCLBinGen::getMaxStackSize` 는 링크된 ELF 의 `.stack_sizes`
  로 검증한다(자동 반영이 아니라 검증이다 — 9/18 확인). 빌드가 통과했다는 것은 프레임이
  실제로 들어간다는 뜻이다. 1 KB 기본값으로 조용히 틀리는 회귀는 구조적으로 불가능하다.

### 회귀

| 검사 | 결과 |
|---|---|
| 순수 LayerNorm `ln_m1` (M=16, i8 커널) | 출력이 9/18 하드웨어 검증본과 **바이트 동일** (`46d321ff…`), 5 회 1 종 |
| attention 기준선 `attn0_kpadq_int8` (O2) | vmfb MD5 `b35a583d…` — 9/18 기록과 **동일** |
| softmax-on-NPU `attn0_hsm_int8` | vmfb MD5 `1b8490a5…` — 9/18 기록 및 `C_noBcast` 와 **동일** |
| `ctest -R amd-aie` | 225 중 실패 4 + timeout 1 — 기존과 **같은 목록**, 새 실패 0 |

## 4. 알려진 잔여물

- 디스패치에 읽히지 않는 3 KB 상수 바인딩이 하나 남는다. custom_op **region 안에**
  fallback 용으로 만든 γ/β `arith.constant` 가 디스패치 형성 시 캡처된 것이고,
  나중에 ukernel 이 region 을 통째로 대체하면서 죽는다. shim DMA 는 생기지 않으므로
  (shim global 3 개 = x, r, out) 장치 비용은 0 이고, 호스트측 3 KB 상수 업로드만 남는다.
- residual 경로의 `layernorm_i16_16x768` 은 도달 불가능하다. 16 행이면 L1 이
  `6144*16 + 7168 = 105472 B` 로 64 KB 를 넘는다. residual 은 항상 8 행, 순수 LN 은 16 행.
  `findLargestFactor` 가 항상 행을 나누는 값을 고르므로 **꼬리 타일은 생기지 않는다**.
- i32 대신 i16: aie2p 에 i32 벡터 MAC 이 없다(int16→acc64 까지). 실제 BERT layer-0
  상수로 재어 i16 + 4 여분 소수 비트(max |h| = 8940)가 ≤1 LSB 100.00 % / exact 99.85 %
  로 i32 의 99.94 % 와 사실상 같음을 9/19 에 확인했다.

## 5. 바뀐 파일

컴파일러 (커밋 없음, `bert` 워킹트리):

- `Transforms/Utils/AMDAIELayerNormUtils.h` — `kLayerNormGammaBeta` 추가
- `Transforms/AMDAIERaiseLayerNorm.cpp` — γ/β 상수 접기(`foldPackedGammaBeta`,
  `readF32Constant`), operand 제거 후 심볼 1 개로, residual 배수 계산을 emission 과 분리
  (`computeResidualMultipliers`) 해서 **거부가 IR 을 건드리지 않게** 함
- `Transforms/AMDAIELowerToUKernels.cpp` — `materializeCoreLocalConstant`
- `Transforms/AMDAIEInsertCores.cpp` — 코어 밖 `memref.get_global` 복제
- `Transforms/AMDAIELowerToAIE.cpp` — `memref.global` 을 `aie.device` 로
- `Transforms/AMDAIEInsertCopyOps.cpp`, `AMDAIEBufferizeToAllocation.cpp` — 사슬 판정에
  `CustomOp` 포함
- `Transforms/KernelDispatch.cpp` — L1 예산과 memtile 채널 예산으로 타일/코어 수 결정
- `Transforms/CMakeLists.txt` — `MLIRBufferizationDialect`

하니스: `_local/ln_lower/{gen_m2_io.py,run2.sh,cmp2.py,dbg4.sh,dbg5.sh,dbg6.sh}`,
`_local/res/{build_ln.sh,build_lnctrl.sh,run_ln1.sh,cmp_ln1.py,cmp_lnctrl.py}`

## 6. NPU 상태

작업 전후 `check-containers.sh` 전 컨테이너 `ok`. 모든 실 NPU 실행은
`scripts/lock/with-npu-lock.sh` 로 감쌌다. 행/타임아웃 없음.

---

# 인코더 레이어 e2e — FC1/GELU/FC2 까지 켜서 (같은 날)

"FC1 GELU FC2 까지 해서 encoder e2e 가 NPU 에서 되나?" 를 직접 돌려서 확인했다.
모델은 `layer0_hsk_int8` (embedding LN 출력 → 인코더 레이어 0 전체 → 두 번째 LN 출력,
head-split + K-fold 적용, 69 노드: MatMul 8, LayerNormalization 2, Softmax 1, Erf 1).

## 7. 결론: 된다 (NPU 10 / CPU 6)

| 디스패치 | softmax only (9/18 구성) | **+ LayerNorm** |
|---|---|---|
| d0 f32→i8 양자화 | cpu | cpu |
| d1 K-pad `slow_memcpy` | cpu | cpu |
| d2–d4 QKV | npu | npu |
| d5 QK^T | npu | npu |
| d6 softmax | npu | npu |
| d7 PV | npu | npu |
| d8 output projection | npu | npu |
| **residual + LayerNorm ①** | **cpu × 2** (`reduction_32x768_f32`) | **npu × 1** (`reduction_DxD_f32`) |
| K-pad `slow_memcpy` | cpu | cpu |
| FC1 + GELU | npu | npu |
| K-pad `slow_memcpy` | cpu | cpu |
| FC2 | npu | npu |
| residual + LayerNorm ② | cpu × 2 | cpu × 2 |
| **합계** | NPU 9 / CPU 8 (17) | **NPU 10 / CPU 6 (16)** |

| | softmax only | + LayerNorm |
|---|---|---|
| 5 회 실행 | 해시 1 종 | 해시 1 종 |
| ORT 대비 corr | 0.9927183 | 0.9917697 |
| maxdiff | 0.54842 | 0.56092 |
| rel-L2 | 0.12057 | 0.12822 |

0.9927183 은 9/18 에 기록한 `L0_kfold_math3` 값과 **정확히 같다** — 그때 구성이 그대로
재현된다. LayerNorm 을 올리면 0.9918 로 0.001 내려간다.

**두 번째 LayerNorm 이 CPU 에 남는 것은 이 추출본의 경계 때문이다.** 이 그래프의 출력이
두 번째 LN 의 **f32 결과 그 자체**라서 뒤에 QuantizeLinear 가 없고, raise 패턴은
quantize 꼬리에 뿌리를 두므로 매칭 대상이 아예 없다. 같은 모델 안의 첫 번째 LN 은
같은 형태로 정상 raise 된다. 실제로 레이어를 쌓으면 두 번째 LN 뒤에도 다음 레이어의
FC1 이 소비하는 QuantizeLinear 가 붙는다(`--whole-layer --i8-out` 경계). **단 그 구성은
아직 돌려보지 않았다.**

남은 CPU 6 개: 모델 입력 경계 양자화 1 개, K-fold 패딩 `slow_memcpy` 3 개,
두 번째 LN 2 개.

## 8. 그 과정에서 고친 IREE 백엔드 버그 하나

layernorm ukernel 을 켜면 QKV batch matmul 3 개가 컴파일에서 죽었다.

```
error: 'memref.subview' op has a non-zero base offset that could not be
       recovered from a backing subspan; not supported by this pass.
```

LayerNorm 과 무관한 디스패치가 죽는 이유는 **버퍼 풀 레이아웃이 바뀌었기 때문**이다.
raise 가 3 KB γ/β 상수를 추가하고(§4 의 잔여물) residual 이 CPU 를 안 거치게 되면서
transient 풀이 26624 → 51200 B 로 합쳐졌고, 그 결과 QKV 입력이 풀 안의 **0 이 아닌
오프셋**에 놓였다. 그 경로가 이렇다:

```
hal.interface.binding.subspan offset(%c24576)   <- 상수 오프셋, 복구 가능해야 함
  -> memref.assume_alignment
  -> memref.expand_shape   [32,832] -> [1,32,832]   <- 여기서 walk 가 끊겼다
  -> memref.subview
```

`AMDAIEConvertToDma` 의 `recoverSubspanElementOffset` 은
`assume_alignment` / `reinterpret_cast` / `subview` 만 따라가고 **reshape 는 몰랐다**.
rank-2 로 바인딩되어 rank-3 로 쓰이는 활성값은 backing subspan 이 아예 없는 것처럼
보여 디스패치 전체가 거부된다. reshape 는 같은 베이스에서 같은 바이트를 다시 읽을 뿐
오프셋을 더하지도 숨기지도 않으므로, `ExpandShapeOp`/`CollapseShapeOp` 를 walk 에
추가하는 것이 맞는 수정이다. 8 줄.

이건 LayerNorm 전용 결함이 아니라 **버퍼 풀 레이아웃이 바뀌면 언제든 터질 수 있던
잠재 버그**다.

## 9. 회귀 (ConvertToDma 를 건드렸으므로 재확인)

| 검사 | 결과 |
|---|---|
| attention `attn0_kpadq_int8` (O2) | MD5 `b35a583d…` — 여전히 **동일** |
| softmax-on-NPU `attn0_hsm_int8` | MD5 `1b8490a5…` — 여전히 **동일** |
| `ctest -R amd-aie` | 실패 4 + timeout 1 — 기존과 같은 목록, 새 실패 0 |

## 10. 남은 것

- `--whole-layer --i8-out` 추출본으로 **두 번째 LN 까지** NPU 에 올리기 (NPU 11 / CPU 5 예상)
- CPU 3 개인 K-pad `slow_memcpy` 흡수 — 9/18 문서의 Pad 생산자 융합 과제와 같은 것
- corr 0.9918 의 분해: bf16 softmax / GELU 다항 근사 / LayerNorm bf16 중 어느 것이 얼마인지
- 12 레이어 스택 및 full BERT 에서의 동작 (미확인)

---

# 인코더 내부 i8 → i8 경계 (같은 날)

목표: 인코더 layer-0 을 **i8 입력 → i8 출력**으로 만들어, 다층 BERT 에서 encoder 0 의
i8 출력이 encoder 1 의 i8 입력으로 그대로 이어질 수 있음을 검증한다.
**컴파일러는 한 줄도 고치지 않았다** — 모델 경계만 바꿨다.

## 11. 경계를 어디로 둘 것인가

양자화 BERT 그래프는 이미 필요한 경계를 갖고 있다. `bert_base_kpadq_int8.onnx` 에서
두 LayerNorm 출력의 소비 구조가 **완전히 대칭**이다.

```
/m/embeddings/LayerNorm/..._QuantizeLinear_Output              (i8 [1,32,768], scale 0.0798715,   zp 0)
  ├─ Pad(kpad_pads_64) -> kpad_x_0 -> QKV 3 개          (K 패딩은 QKV 쪽 일)
  └─ DequantizeLinear  -> residual add                  (residual 은 패딩 안 된 hidden 을 요구)

/m/encoder/layer.0/output/LayerNorm/..._QuantizeLinear_Output  (i8 [1,32,768], scale 0.076748416, zp 0)
  ├─ Pad -> kpad_x_1 -> layer 1 의 QKV
  └─ DequantizeLinear -> layer 1 의 residual add
```

따라서 자연스러운 내부 경계는 **`[1,32,768]` i8** 이지 패딩된 `[1,32,832]` 이 아니다.
832 는 QKV 만의 사정이고 residual 은 768 을 요구하므로, 경계를 832 로 잡으면 residual
입력을 따로 또 넘겨야 한다. 768 i8 은 **layer 1 이 실제로 받는 바로 그 텐서**다.

그리고 경계에 `DQ` 를 새로 붙이지 않았다. 원 그래프의 본체가 i8 을 **그대로** 소비하기
때문이다(Pad 는 양자화 도메인, residual 은 자체 DQ 보유). f32 를 받아 다시 Q 하면
9/18 에 CPU 로 남아 있던 그 quantize 디스패치를 그대로 되살리는 셈이 된다.

## 12. 모델 수술

`_local/int8_debug/tools/make_layer0_i8.py`. 하드웨어 검증이 끝난 `layer0_hsk_int8` 의
**본체는 건드리지 않고** 양 끝만 바꾼다. 스케일/제로포인트는 전부 원 모델의 정적 값이다.

- **입력** 선두 `QuantizeLinear` 를 제거하고 그 i8 출력을 그래프 입력으로 승격
- **출력** 원 모델에 이미 있는 LN② 의 `QuantizeLinear`(scale 0.076748416, zp 0)를 붙이고
  그 i8 결과를 그래프 출력으로

노드 수 69 로 동일(Q 하나 빼고 하나 더함). `func.func @layer0i8(%arg0: si8) -> si8`.

## 13. 결과: NPU 11 / CPU 3 — LN② 도 NPU

| | f32 경계 (§7) | **i8 → i8 경계** |
|---|---|---|
| 입력 f32→i8 양자화 | cpu | **없음** (입력이 이미 i8) |
| K-pad `slow_memcpy` ×3 | cpu | cpu |
| QKV / QK^T / softmax / PV / out-proj | npu 7 | npu 7 |
| **residual + LN ①** | npu 1 | npu 1 |
| FC1+GELU / FC2 | npu 2 | npu 2 |
| **residual + LN ②** | **cpu 2** | **npu 1** |
| **합계** | NPU 10 / CPU 6 (16) | **NPU 11 / CPU 3 (14)** |

raise 가 LN① 과 **동일하게** 매치된 것을 IR 로 확인했다: 마킹된 custom_op 2 개가
서로 다른 `amdaie.layernorm_epsilon_scaled`(1.81775465, 3.56959511)를 갖고, 둘 다
`ukernel.generic "layernorm_i16_8x768"` 로 내려간다 — 즉 **두 LayerNorm 모두 residual add
를 int16 producer 로 흡수**했다.

남은 CPU 3 개는 K-fold 패딩 `slow_memcpy` 뿐이다.

## 14. 검증

입력 2 종, 둘 다 모델 자신의 정적 입력 스케일 0.0798715 / zp 0 기준:

- `l0i8_real.npy` — 실제 BERT embedding-LayerNorm activation(`attn0_x.npy`)을 그 양자화기로
  통과시킨 것. 포화 0 개, 범위 −115..46.
- `l0i8_rand.npy` — int8 전 범위 균등 난수(seed 11). 난수라도 그 스케일로 읽혀야 의미가
  있으므로 스케일을 명시해 둔다.

| | real | rand |
|---|---|---|
| NPU 5 회 SHA-256 | **1 종** `8b5a4908…` | **1 종** `d4f8adc0…` |
| IREE-CPU(llvm-cpu) vs ORT | **바이트 동일** | **바이트 동일** |
| NPU vs IREE-CPU · i8 exact | 79.66 % | 80.61 % |
| NPU vs IREE-CPU · ≤1 LSB | 85.93 % | 86.84 % |
| NPU vs IREE-CPU · i8 maxdiff | 8 | 8 |
| DQ 후 f32 corr | 0.9913976 | 0.9914682 |
| DQ 후 f32 rel-L2 | 0.13085 | 0.13031 |
| DQ 후 f32 maxabs | 0.61399 | 0.61399 |

IREE-CPU 가 ORT 와 바이트 동일하므로 CPU 빌드를 골든으로 써도 된다(NPU vs ORT 수치가
NPU vs IREE-CPU 와 같은 이유).

⚠️ **i8 maxdiff 8 은 §7 의 f32 출력본보다 나빠진 것이 아니다.** 출력 스텝이
0.076748 이므로 8 LSB = 0.614 이고, f32 출력본의 maxabs 0.561 과 같은 크기의 오차다.
출력을 int8 로 다시 양자화하면서 같은 오차가 더 거친 LSB 단위로 표시될 뿐이다.
real 과 rand 가 사실상 같은 수치를 내므로 입력 의존적 병리도 없다.

## 15. 회귀

이번 작업에서 **컴파일러 변경은 없다**(모델/하니스만). 그래도 재확인했다.

| 검사 | 결과 |
|---|---|
| attention `attn0_kpadq_int8` (O2) | MD5 `b35a583d…` **동일** |
| softmax-on-NPU `attn0_hsm_int8` | MD5 `1b8490a5…` **동일** |
| `ctest -R amd-aie` | 실패 4 + timeout 1 — 기존과 같은 목록, 새 실패 0 |

모든 실 NPU 실행은 `with-npu-lock.sh` 경유. 작업 전후 `check-containers.sh` 전부 `ok`.

## 16. 이것이 말해 주는 것

encoder 0 의 i8 출력이 encoder 1 의 i8 입력 형식과 **같은 텐서**(768, zp 0, 정적 스케일)
이므로, 다층 스택은 이 경계를 그대로 이어 붙이면 된다. 레이어당 NPU 11 / CPU 3 이고
CPU 3 은 전부 K-fold 패딩이다. **단 12 레이어 스택 자체는 아직 돌려보지 않았다.**
