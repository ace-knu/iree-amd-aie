# 2026-09-10 — per-channel int8이 NPU 백엔드에서 막혀 있다 (정확도 +0.09 차단)

## 0. 요약

BERT-base int8 e2e 정확도(§ `2026-09-10_bert_e2e_int8_vectorized.md`)를 올리는 레버를
전수 측정했다. **`per_channel=True` 하나가 가장 크다(+0.085)**. 그런데
**NPU 백엔드가 per-channel 모델을 컴파일하지 못한다.** per-tensor는 된다.

## 1. 정확도 레버 측정 (12층 BERT-base, torch 정답 대비, 전부 CPU/ORT)

| 설정 | corr | maxdiff | NPU 컴파일 |
|---|---|---|---|
| 현재 (per-tensor, MinMax) | 0.883656 | 3.468 | ✓ |
| + matmul 출력의 중간 int8 반올림 제거 | 0.910564 | 2.681 | 미시도 |
| **+ `per_channel=True`** | **0.968350** | 1.871 | **✗ 실패** |
| + `per_channel` + Percentile 99.99 | 0.887959 | 5.766 | — |
| **`per_channel` + 반올림 제거** | **0.973500** | 1.642 | ✗ |
| (참고) bf16 스칼라 | 0.99998 | 0.04 | ✓ |

- **per_channel 이 +0.085** — 중간 반올림 제거(+0.027)의 3배. 둘 합치면 bf16 과의 격차
  0.116 → **0.026** 으로 좁혀진다.
- **Percentile 캘리브레이션은 음성**(0.888 로 악화). 랜덤 토큰 입력에서는 이상치가 실제
  신호이므로 클리핑이 해롭다. 이 입력 분포에서는 MinMax 가 맞다.
- 현재 양자화 설정(`_local/int8_debug/quant_bert_base.py`)에 `per_channel` 이 없어
  기본값 False, `calibrate_method` 도 기본값 MinMax 였다.

## 2. 최소 재현체

`_local/int8_debug/tools/gen_pc_matmul.py [pc|pt]` — npu4 패딩 배수에 정확히 맞춘
단일 matmul (M=32, K=64, N=64). 채널별로 규모가 다른 가중치를 만들어 per-channel 이
의미를 갖게 했다.

| | 가중치 DQ scale dims | 컴파일 |
|---|---|---|
| `pcmm_pt` | `[]` (스칼라) | ✓ rc=0, vmfb 82611 |
| `pcmm_pc` | `[64]` (채널별) | ✗ rc=1 |

40KB 모델이라 수 초에 컴파일된다. bert_base(362MB MLIR)로 디버깅할 필요 없음.

## 3. 즉각적 실패 지점

```
error: expected only one target op, found 2 target ops
  → AMDAIEBufferizeToAllocation.cpp:222
error: failed to run translation of source executable to target executable
  for backend amd-aie / amdaie-pdi-fb
```

`AMDAIEBufferizeToAllocation` 은 디스패치에서 "target op"(elementwise / reduction /
contraction / conv, `fill`·`copy` 및 bias-fold 케이스 제외)을 모아 **정확히 하나**를
요구한다. per-channel 디스패치에는 f32 dequantize generic 이 **둘** 남아 있다:

```
%9  = linalg.generic  maps [(d0,d1,d2,d3)->(d0,d1,d2,d3), 동일]        ← 활성 dequant (identity)
%13 = linalg.generic  maps [(d0,d1,d2,d3)->(d0,d1,d2,d3), ->(d0,d3)]  ← per-channel 가중치 dequant (브로드캐스트)
%15 = linalg.fill f32
%16 = linalg.generic (reduction 포함)                                  ← matmul, f32
```

MLIR 의 `isElementwise` 는 projected permutation 을 허용하므로 브로드캐스트인 `%13` 도
elementwise 로 분류되어, elementwise 모드에서 target 이 2개가 된다.

## 4. 그 아래의 진짜 원인

동작하는 per-tensor 디스패치와 비교하면 차이가 분명하다.

| | 디스패치 내용 |
|---|---|
| per-tensor (동작) | `linalg.fill(i32)` + matmul on **`tensor<8x4x8x8xi32>`** — **dequant generic 이 아예 없음** |
| per-channel (실패) | dequant generic 2개 + `fill(f32)` + matmul on **f32** |

즉 per-tensor 는 `DQ(x)@DQ(w)` 가 **순수 `i8×i8→i32` matmul 로 접히고** 스케일이
디스패치 밖으로 빠진다. per-channel 은 **그 접기가 일어나지 않아** f32 dequant 가
디스패치 안에 남고, 그래서 §3 에서 죽는다.

수학적으로는 per-channel 도 접을 수 있다 —
`out[m,n] = (Σ x_i8·w_i8) · x_scale · w_scale[n]` 이므로 스케일을 출력 쪽
**N 방향 per-channel 곱**으로 올리면 된다. 그걸 하는 단계가 벡터 스케일을 거부하는 것이다.

**위치 확정 (2026-09-10)**: `--compile-to=input` 덤프를 파일로 받아 비교하니
**`input` 단계에서 이미 갈린다** — IREE flow 나 우리 플러그인이 아니라 **torch→linalg 로워링**이다.

```
pt:  linalg.quantized_matmul ins(%2, %cst_1, %c0_i32, %c0_i32
                              : tensor<32x64xi8>, tensor<64x64xi8>, i32, i32)
                                                        ^^^^^^^^^^^^^^^^^^ zero point 가 스칼라 i32

pc:  linalg.generic  maps [(d0,d1)->(d0,d1), ->(d1), ->(d1), ->(d0,d1)]   <- per-channel dequant
     linalg.matmul   ins(%6, %2 : tensor<32x64xf32>, tensor<64x64xf32>)   <- 평범한 f32 matmul
```

torch-mlir 은 per-tensor 일 때만 명명 op `linalg.quantized_matmul` 을 낸다. 그 op 은
**zero point 를 스칼라 `i32` 로만 받으므로 per-channel 을 표현할 수 없다.** 그래서 per-axis 는
f32 matmul + 명시적 dequant generic 으로 폴백하고, 그 f32 dequant 가 디스패치에 남아 §3 에서 죽는다.
즉 torch-mlir 의 버그라기보다 **MLIR 명명 op 자체의 표현력 한계**다.

### 수정 위치 선택지

| | 어디 | 장점 | 단점 |
|---|---|---|---|
| (a) | torch-mlir 의 per-axis DQ 로워링 | 근원 | 명명 op 이 없어 결국 generic 을 내야 함. **3단 로컬 서브모듈 체인** |
| (b) | IREE `GlobalOptimization` (`FuseDequantizationMatmul` 확장) | 상류에서 일반적으로 해결 | 서브모듈 체인 |
| (c) | **우리 플러그인에 호이스팅 패스 추가** | **서브모듈 안 건드림**, 우리 파이프라인에만 필요 | 상류에 기여 안 됨 |

(c) 안: `matmul(dq_scalar(x), dq_perchannel(w))` 를 매칭해
`quantized_matmul(x_i8, w_i8, 0, 0) -> i32` + 출력 쪽 `out[m,n] * x_scale * w_scale[n]` 로 재작성.
수학적으로 동등하다 (`out[m,n] = (Σ x_i8·w_i8) · x_scale · w_scale[n]`).
zero point 가 전부 0 일 때만 발동시키면 안전하고, 실제로 `ActivationSymmetric` + 대칭 가중치라
`dense<0> : tensor<64xi8>` 로 나온다(컴파일 타임 상수라 검증 가능).

⚠️ 미확인 위험: 출력 쪽 스케일 곱은 `w_scale[n]` 을 텐서 오퍼랜드로 받는 **브로드캐스트 generic** 이다.
그게 matmul 디스패치로 융합되면 §3 의 "target op 2개" 가 재발할 수 있다. pt 에서는 출력 스케일이
스칼라 상수라 이 문제가 없었다. 그래서 (c) 를 구현할 때 **디스패치가 어떻게 갈리는지 먼저 확인**해야 한다.

## 5. 곁가지

`AMDAIEFoldBroadcastAddIntoDest.cpp` 가 `DEBUG[refold] ...` 를 `llvm::errs()` 로
**무조건** 출력한다(디버그 매크로 없이). 무관한 op 마다 찍혀 로그를 오염시키고
이번 조사에서 red herring 이 됐다. Method B 작업의 잔재로 보이며 정리 대상.

## 6. 다음

1. §4 의 접기 단계를 특정 (torch-mlir vs IREE core) — `--compile-to=input` 파일 비교
2. per-axis 스케일을 출력 쪽 per-channel 곱으로 올리도록 수정
3. `pcmm_pc` 컴파일 → bert_base per-channel 컴파일 → 하드웨어 실행 → corr 확인
   (CPU 예측치 0.968, 반올림 제거까지 하면 0.974)
