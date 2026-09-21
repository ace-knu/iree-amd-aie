# 2026-09-14 — attention 을 NPU 에서 연속 실행하기: elementwise 융합이 막혀 있던 진짜 이유

> 목표는 **attention 블럭 전체를 NPU 에서 돌리는 것**(FFN·전체 BERT 아님).
> 결론부터: 지금 CPU 에 있는 8 개 중 3 개는 **컴파일 플래그 하나 빼면 오늘 바로 사라진다.**
> 마이크로커널이 실제로 필요한 것은 **softmax 하나뿐**이다.

## 0. 결론

레이어 0 의 attention 구간(dispatch 3~16)만 센 것:

| 구성 | attention 디스패치 | 백엔드 |
|---|---|---|
| 현재 배포본 (`no-fuse` 플래그 있음) | NPU 6 + **CPU 8** = 14 | OK |
| **`no-fuse` 플래그 제거 (융합 켬)** | NPU 6 + **CPU 5** = 11 | **OK — attention matmul 6/6 통과** |
| + `propagate-collapse-across-expands` | NPU 6 + **CPU 4** = 10 | ✗ QKV 에서 `memtile_dma` 실패 |

- **재양자화 elementwise 는 원래 matmul 안으로 묶이는 게 맞았다.** 안 묶인 이유는 우리가 켜둔
  `--iree-dispatch-creation-no-fuse-into-contraction-conv-roots` 뿐이고, 그 플래그가 필요했던
  이유는 **접지 않은 f32 bias(3-input)** 였다. 즉 K축 bias 접합 → 융합 해금의 인과가 실증됐다.
- **transpose 가 안 묶인 것은 연산 종류 탓이 아니라 `flow.tensor.reshape` 이 체인을 끊어서다.**
- 융합된 것들은 **펜스(`executableIsContractionOrConv`) 를 안 건드려도 NPU 로 따라간다.**
  펜스 수정이 필요해지는 건 홀로 남는 디스패치(softmax·LN·남은 transpose)부터다.

## 1. 배경 — 무엇을 확인하려 했나

`docs/2026-09-11_k_axis_bias_fold_two_column_fix.md` §5 에 이렇게 적혀 있었다:

> `--iree-dispatch-creation-no-fuse-into-contraction-conv-roots` 를 빼면 **원본조차** 컴파일이
> 깨진다: `'amdaie.connection' op no producer DMA channel available`.

이 판정은 **bias 를 접기 전 원본**에 대한 것이었다. QKV 36 개의 bias 를 K 축에 접은 지금
(`bb_kpadq.vmfb`) 도 여전히 깨지는지는 시험된 적이 없다. 이게 이 문서의 출발점이다.

## 2. 실험 방법

전부 호스트에서 `iree-compile` 만 사용(NPU 미사용, 락 불필요).

```bash
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
COMMON="--iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local \
--iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu \
--iree-amdaie-target-device=npu4 \
--iree-amd-aie-peano-install-dir=$PWD/llvm-aie \
--iree-flow-enable-executable-deduplication=false"

# flow 레벨만 (약 10 초)
./build/tools/iree-compile _local/int8_debug/out/bert_base_kpadq_int8.mlir $COMMON \
  --compile-to=flow --mlir-elide-elementsattrs-if-larger=8 -o flow_fuse.mlir

# 전체 (성공 시 약 3 분, 실패는 20 초 내)
./build/tools/iree-compile _local/int8_debug/out/bert_base_kpadq_int8.mlir $COMMON -o out.vmfb
```

디스패치 인벤토리는 `func.func` 이름으로 센다:

```bash
grep -oE 'func\.func @[a-zA-Z0-9_$]+' flow_fuse.mlir \
  | sed -E 's/func\.func @//; s/^main_graph\$async_dispatch_[0-9]+_//' \
  | sed -E 's/[0-9]+x[0-9]+(x[0-9]+)*/S/g; s/_[0-9]{3,}//g' | sort | uniq -c | sort -rn
```

## 3. 실험 1 — flow 레벨: 플래그를 빼면 무엇이 묶이나

전체 모델 기준 (`bert_base_kpadq_int8.mlir`):

| 종류 | `no-fuse` 有 | `no-fuse` 無 | + collapse 전파 |
|---|---|---|---|
| **총 디스패치** | **304** | **208** | **196** |
| `batch_matmul` (NPU) | 96 | 96 | 96 |
| `elementwise_i32xi8` (재양자화) | **48** | **0** | 0 |
| `elementwise_transpose` | 48 | **36** | **24** |
| `elementwise_f32xi8xi8` | 12 | **0** | 0 |
| pad/crop_executable | 24 | **0** | 0 |
| `reduction_f32` (LayerNorm) | 50 | 50 | 50 |
| `softmax` | 12 | 12 | 12 |
| `slow_memcpy` | 12 | 12 | 12 |

**재양자화 48 개는 전부 matmul 디스패치 안으로 흡수된다.** 흡수의 증거는 출력 타입 변화:

```
no-fuse : dispatch_10 QK^T  → writeonly tensor<12x32x32xi32>   + 별도 requant 디스패치
fuse    : dispatch_10 QK^T  → writeonly tensor<12x32x32xi8>    ← 재양자화 흡수
fuse    : dispatch_12 PV    → writeonly tensor<32x12x64xi8>    ← transpose + 재양자화 둘 다 흡수
fuse    : dispatch_13 proj  → writeonly tensor<1x32x768xi8>    ← 재양자화 흡수
```

## 4. 실험 2 — 백엔드까지: 막히는 곳은 정확히 하나다

`bb_kpadq` (QKV 36 개만 접힌 현 배포본) + 융합, 전체 컴파일:

```
rc=1
12 error: 'amdaie.connection' op no producer DMA channel available
   → 전부 onnx.MatMul(x, [768,3072]) = FC1, 레이어당 1 개
   → 막힌 연결: amdaie.logicalobjectfifo<memref<64xf32, 1>> ← memref<3072xf32>
```

**막힌 것은 FC1 의 접지 않은 f32 bias(3072 개) 가 3 번째 입력으로 들어간 것**이고,
**attention 의 matmul 6 개(Q·K·V·QK^T·PV·proj)는 전부 백엔드를 통과했다.**

부수적으로 드러난 것: **"3-input 이면 무조건 실패" 가 아니다.** 같은 판에서 proj 와 FC2 는
f32 bias 를 그대로 달고도 융합에 성공했다. 실패한 건 **N=3072 인 FC1 하나뿐**이고, 이는
`docs/2026-08-28_shared_channel_fanout_column_threshold.md` 의 열 임계값과 맞아떨어진다 —
채널 예산은 열 수에 비례해 빡빡해진다.

## 5. 실험 3 — bias 를 전부 접으면? 오히려 나빠진다

| 모델 | 융합 | 결과 |
|---|---|---|
| kpadq (QKV 36) | ✗ | OK, 304 디스패치 |
| kpadq | ✓ | **12 실패** (FC1) |
| kpad (전체 72) | ✗ | **OK, rc=0, 198,277,147 바이트** (기준선) |
| kpad | ✓ | **36 실패** (FC1 12, FC2 12, proj 12) |

기준선이 0 에러이므로 **36 개는 K 값 변화가 아니라 융합이 유발한 것**이 확정이다.
bias 3rd input 이 사라졌는데도 proj·FC2 가 새로 깨진다 (`no producer DMA channel` 24 +
`memtile_dma cannot split` 12). K 가 커진 것(proj 768→832~1088, FC2 3072→3136)과 융합이
겹친 결과로 보이지만 **메커니즘은 미확정**이다.

⚠️ 즉 **"bias 를 다 접으면 융합이 열린다" 는 단순한 그림이 아니다.** attention 범위에서는
이미 접혀 있어 문제가 안 되지만, FFN 까지 넓히려면 이 36 개를 따로 규명해야 한다.

## 6. 실험 4 — transpose 는 왜 안 묶이나: reshape 이 끊는다

융합된 transpose 와 안 된 transpose 의 구조 차이:

| | producer 출력 | consumer 입력 | 사이의 reshape | 융합 |
|---|---|---|---|---|
| PV 뒤 | `12x32x64` | `12x32x64` | **없음** | **✅** |
| Q 뒤 | `1x32x768` | `32x12x64` | 있음 (768 → 12×64) | ❌ |
| K 뒤 | `1x32x768` | `32x768` | 있음 (unit-dim 제거) | ❌ |
| V 뒤 | `1x32x768` | `32x12x64` | 있음 | ❌ |

`flow_fuse.mlir:7496` 등에서 직접 확인:

```mlir
%8  = flow.dispatch ..._dispatch_4_batch_matmul...  -> tensor<1x32x768xi32>
%9  = flow.tensor.reshape %8 : tensor<1x32x768xi32> -> tensor<32x12x64xi32>
%10 = flow.dispatch ..._dispatch_7_elementwise_transpose...(%9)
```

**transpose 연산 자체는 융합 가능하다 — PV 가 증명한다.** 못 묶이는 것은 dispatch 형성 시점에
producer 와 consumer 가 직접 이웃이 아니게 되기 때문이고, K 는 unit-dim 하나 떼는 reshape 인데도
끊긴다.

### 6.1 기존 플래그 두 개를 시험

| 플래그 | 총 디스패치 | `elementwise_transpose` |
|---|---|---|
| (없음) | 208 | 36 |
| `--iree-dispatch-creation-enable-aggressive-reshape-movement` | 208 | 36 (**변화 없음**) |
| **`--iree-dispatch-creation-propagate-collapse-across-expands`** | **196** | **24** |
| 둘 다 | 196 | 24 |

`propagate-collapse-across-expands` 가 **K 의 transpose를 matmul 로 흡수**시킨다.
남은 24 개(레이어당 2 개)는 Q 와 V 의 `32x12x64 → 12x32x64`, 즉 (S,H,D)→(H,S,D) 헤드 축 이동이다.

### 6.2 그러나 백엔드가 못 받는다 ← 진짜 "따로 로직"이 필요한 지점

`bb_kpadq` + 융합 + collapse 전파, 전체 컴파일:

```
rc=1, 24 실패
12 × onnx.MatMul(x, [768,3072])  = FC1   : 'amdaie.connection' no producer DMA channel  (기존)
12 × onnx.MatMul(x, [832,768])   = QKV   : 'aie.memtile_dma' op cannot split this DMA
      access pattern into per-party BDs: its outermost dimension (size 96, stride 1) is
      not 4 steps of one 768-element slice, needed to synchronize against 4 parties
      + 'amdaie.logicalobjectfifo.from_buffers' op could not create DMA operations
```

전치된 출력의 접근 패턴을 4 행에 나눠줄 수 없다.
`docs/2026-09-11_k_axis_bias_fold_two_column_fix.md` §8.3 의 K 정렬 결함과 같은 계열이며,
**AMD-AIE 백엔드의 memtile DMA splitting 수정이 필요하다.**

## 7. attention 블럭 지도 (레이어 0)

```
                      현재 (no-fuse 有)          융합 켬
 3  slow_memcpy       CPU                        CPU
 4  Q      1x32x768x832   NPU → i32              NPU → i32
 5  K                     NPU → i32              NPU → i32
 6  V                     NPU → i32              NPU → i32
 7  transpose         CPU                        CPU
 8  transpose         CPU                        CPU
 9  transpose         CPU                        CPU
10  QK^T   12x32x32x64    NPU → i32              NPU → i8        ← 재양자화 흡수
11  requant           CPU                        (사라짐)
12  softmax           CPU                        CPU
13  PV     12x32x64x32    NPU → i32              NPU → 32x12x64xi8 ← transpose+재양자화 흡수
14  transpose         CPU                        (사라짐)
15  proj   1x32x768x768   NPU → i32              NPU → i8        ← 재양자화 흡수
16  requant           CPU                        (사라짐)
────────────────────────────────────────────────────────────
                      NPU 6 / CPU 8 = 14         NPU 6 / CPU 5 = 11
```

`17,18 reduction_f32` (residual + LayerNorm) 부터는 attention 밖이다 — BERT-base 는 post-LN 이라
QKV 앞의 LN 은 이전 레이어 소속이다.

## 8. 남은 CPU 5 개와 각각의 성격

| | 개수 | 필요한 작업 | 마이크로커널? |
|---|---|---|---|
| `elementwise_transpose` (K) | 1 | collapse 플래그 + **백엔드 memtile split 수정** | ✗ |
| `elementwise_transpose` (Q, V) | 2 | 진짜 순열. 그래프 재구성 — W 를 헤드별로 쪼개 `batch_matmul(12,32,64,768)` 로 내면 출력이 네이티브 `12x32x64` 라 transpose 자체가 소멸 (**미검증**, x 를 12 헤드로 broadcast 해야 함) | ✗ |
| `softmax` | 1 | Peano 커널 + 펜스 확장 | **✓ 유일** |
| `slow_memcpy` (K증강 Pad) | 1 | attention-only 면 입력을 미리 증강해 제거 | ✗ |

### 8.1 softmax 에 대해 확인된 것

- **`linalg.softmax` 가 named op 으로 살아남아 있다** (`flow_kpadq.mlir:356` 부근).
  `project_softmax_ukernel_infra` 의 미검증 항목 #1 이 **양성**이다 —
  `matchSoftmaxDAGForUKernel` 이 매칭할 대상이 실제로 존재한다.
- 디스패치는 이미 `dequant+scale+mask → linalg.softmax → quantize` 가 하나로 융합돼 있고
  경계 타입이 i8↔i8 이라 깨끗하다.
- ⚠️ **단 3-operand 다**: `readonly 12x32x32xi8` + `readonly 32x32xf32`(mask) + `writeonly`.
  마이크로커널을 만들어도 DMA 채널 제약을 만날 수 있다. 현 export 의 mask 는 all-ones 이므로
  **상수 폴딩으로 2-operand 가 되는지 먼저 확인하는 게 싸다** (가변 길이 지원은 포기).
- 참고: residual+LayerNorm 은 **4-operand** (`768xf32` + `32x768xi8` ×2). attention 밖이지만
  체인을 끊는 지점이고, residual 은 상수가 아니라 K축 접합 같은 우회가 안 통한다.

## 9. 펜스(`executableIsContractionOrConv`) — 지금은 안 건드려도 된다

`AMDAIEAssignDeviceAffinities.cpp` 의 이 술어가 NPU/CPU 를 가르는 **유일한 분류기**다.

- **제거하면 안 된다.** "원래대로" 라는 상태가 없다 — 이 패스가 없으면 모든 dispatch 가
  `--iree-hal-default-device=npu` 로 가고, embedding gather(`elementwise_32x768_i64xf32xf32xf32`,
  i64 인덱스) 처럼 AMD-AIE codegen 이 없는 것들이 깨진다. 이 펜스는 임의가 아니라
  **백엔드 커버리지를 인코딩한 것**이고, 커버리지가 늘면 같이 늘어나야 한다.
- **그러나 이번 방향에서는 당장 손댈 필요가 없다.** 술어가 "executable 안에 contraction 이
  하나라도 있으면 NPU" 라서, elementwise 가 matmul 디스패치 **안으로 융합되면 자동으로 NPU 를
  따라간다.** 재양자화 48 개 + transpose 12 개가 그 케이스로, 펜스 수정 0 줄로 NPU 에 올라간다.
- 수정이 필요해지는 건 **홀로 남는 디스패치**부터다: softmax 12, LN 50, transpose 24~36.
- 확장 설계는 `docs/device_placement.md` 에 `supportedOps` 테이블 + 정책 seam 으로 이미 있다.

⚠️ 정책 결정이 하나 필요하다: **지금 술어는 executable 안의 elementwise 가 codegen 되는지 안
본다.** §4 의 FC1 실패가 정확히 그 케이스(술어는 NPU 로 보냈는데 백엔드가 못 받음)다.
`supportedOps` 로 가면 이런 걸 조기에 host 로 뺄 수 있지만 **조용히 성능을 잃는다.**
실험 단계에서는 깨지는 쪽이 나을 수 있다.

## 10. 다음

1. **attention 한 블럭만 뽑은 모델을 만들어 융합 켜고 컴파일** (`extract_from_l0.py` 존재).
   FC1 이 없으니 `no-fuse` 플래그만 빼면 되고, 11 디스패치(NPU 6 / CPU 5) 가 나오는지 확인.
2. mask all-ones 상수 폴딩으로 softmax 를 2-operand 로 만들 수 있는지 확인 (싸다).
3. 백엔드 `memtile_dma` per-party split 수정 → K 의 transpose 흡수 (CPU 5 → 4).
4. Q/V 의 헤드별 batch matmul 재구성 시도 (CPU 4 → 2).
5. softmax Peano 커널 + 펜스 확장 (CPU 2 → 1 → 0).

⚠️ 정직한 경고: 다 끝나도 attention 은 14 → 약 10 디스패치(전부 NPU)이지 1 개가 아니다.
contraction→contraction 은 원리적으로 한 커널이 될 수 없다(softmax 가 QK^T 전체 행 리덕션을
요구하고, PV 의 리덕션 축은 softmax 의 비-리덕션 축이라 코어 간 재분배가 든다).
2026-08-31 측정에서 벽시계의 약 99% 가 디스패치당 드라이버 오버헤드였으므로,
**이득은 cross-device staging 제거분**이다. attention-only 스코프의 진짜 가치는 그것을
깨끗하게 측정할 수 있다는 점이다.

## 11. 아티팩트

flow 덤프 (`_local/int8_debug/out/`):

| 파일 | 구성 | 디스패치 |
|---|---|---|
| `flow_nofuse.mlir` | `no-fuse` 有 (현 배포본) | 304 |
| `flow_fuse.mlir` | `no-fuse` 無 | 208 |
| `flow_rsh.mlir` | + aggressive-reshape-movement | 208 |
| `flow_col.mlir` | + propagate-collapse-across-expands | 196 |

입력: `_local/int8_debug/out/bert_base_kpadq_int8.mlir` (QKV 36 접합),
`bert_base_kpad_int8.mlir` (전체 72 접합, ALIGN=64).
