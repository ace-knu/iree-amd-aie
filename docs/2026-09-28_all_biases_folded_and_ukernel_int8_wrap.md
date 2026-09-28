---
date: 2026-09-28
topic: [kfold, layernorm, softmax, encoder-stack, accuracy]
status: resolved
summary: "남은 bias 8개(output projection 7 + L10 FC2)까지 K축에 접어 12층 134 = NPU 131 / CPU 3. 그 과정에서 LayerNorm·softmax ukernel 이 int8 범위 밖 값을 wrap 시키던 버그를 찾아 수정(ssrs 는 포화 모드가 꺼져 있으면 포화하지 않음)"
---
# 모든 bias 접기, 그리고 ukernel int8 wrap 버그 (2026-09-28)

`2026-09-28_kfold_pad_residual_view.md`(K-fold 복사 35→1)의 후속. 남은 CPU 디스패치 19 개 중
LayerNorm 18 개(9 개 × 2)를 없애려고 **아직 K 축에 안 접힌 bias 8 개**를 접었다.

## 0. 결론

| 12층 | 총 | NPU | CPU | NPU vs ORT (real) | NPU vs fp32, 5 입력 평균 |
|---|---:|---:|---:|---:|---:|
| `enc12hv` (접기 8 개 제외) | 142 | 123 | 19 | 0.9310 | 0.7554 |
| **`enc12x` (전부 접음)** | **134** | **131** | **3** | **0.9365** | **0.7588** |

- CPU 에 남은 3 개: 마지막 f32 LayerNorm(2, 출력이 pooler 로 가는 f32 경계) + layer 0 QKV K-fold
  복사(1, 생산자가 모델 입력).
- NPU 의 fp32 대비 정확도가 같은 int8 모델의 ORT(0.7579)와 같다 — NPU 경로 자체의 손실은 없다.
- 반복 실행: 두 모델 × 입력 5 종 전부 run 간 동일.
- ⚠️ **그 과정에서 LayerNorm·softmax ukernel 이 int8 범위 밖 값을 wrap 시키던 버그**를 찾아 고쳤다
  (§3). softmax 쪽은 9/17 부터 있던 것이다.

## 1. 남은 bias 8 개 — 두 가지 다른 이유

| 거부된 matmul | 개수 | 이유 |
|---|---:|---|
| attention output projection (layer 2, 4, 5, 6, 7, 8, 9) | 7 | **열 개수 초과** — bias 를 누산기 단위로 바꾸면 최대 39,017(layer 7) → 값 1 열 기준 308 열 필요, 한도 64 |
| FC2 (layer 10) | 1 | **재양자화 여유 초과** — bias 가 출력 63.7 LSB > `--max-bias-lsb`(48) |

### 1.1 output projection — 열 값 자동 선택 (`--pad-value auto`)

증강 열이 모두 1 이면 열 하나가 1 × 127 을 담는다. Pad 는 모든 열에 같은 값 v 만 채울 수 있으므로
v 를 키우면 v × 127 을 담지만 bias 는 v 의 배수로만 표현된다(반올림 오차 ≤ v/2 누산기 LSB).
그래서 **그룹마다 64 열에 들어가는 가장 작은 v** 를 고른다(1 로 되면 1 = 기존과 동일).
결과 v = 2~5, bias 오차 ≤ **0.003 재양자화 LSB**. 여전히 Pad 라 오늘 만든 흡수 경로를 탄다.
(열마다 다른 값을 쓰면 정확하지만 Concat 이 되어 복사가 생긴다 — 안 씀.)

`_local/int8_debug/tools/fold_bias_into_k_general.py`: `--pad-value auto`,
`--max-bias-err-lsb`(기본 0.05). 옵션 없이 돌리면 출력 SHA 가 이전과 동일함을 확인.

### 1.2 layer 10 FC2 — 실측으로 판단

접으면 "재양자화(±127 clamp) 후 bias 더하기" 가 "bias 를 누산기에 넣은 뒤 clamp" 로 바뀐다.
입력 5 종 122,880 원소에서 **clamp 때문에 달라지는 원소는 2 개**, 둘 다 채널 381 이고 **접은 쪽이
fp32 참값에 더 가깝다**(원래 그래프 오차 15.1 / 17.7 LSB → 접은 쪽 0.4 / 2.2 LSB). 원래 그래프가
bias 를 더하기 전에 잘라 버린 원소다. `--max-bias-lsb 64` 로 접는다(48 을 넘는 matmul 은 이것뿐).

### 1.3 정확도 지표에 대한 교훈

접은 모델과 원래 int8 모델의 **ORT 출력끼리 corr 이 0.93~0.95** 로 나와 문제처럼 보였지만,
bias 오차 0.003 LSB 만 다른 두 변형(v>1 vs v=1 열 수 한도 해제)끼리도 0.956 이었다. **이 12 층 int8
체인은 극소 차이를 크게 증폭한다** — int8 모델끼리의 corr 은 정확도 지표로 못 쓴다. fp32 인코더
(`bert_base.onnx` 에서 같은 구간을 잘라 역양자화 입력을 넣은 것) 대비로 보면 네 변형 모두 같다
(평균 0.7564 → 0.7579).

## 2. NPU 에서만 틀렸다 → 원인 추적

`enc12x` 첫 실행: **NPU vs ORT 0.569**(이전 0.931). IREE CPU 빌드는 ORT 와 0.954 로 정상 →
NPU 백엔드 문제. 이분 탐색(12 층 전체 재컴파일 5 분씩, 이후 1 층 추출로 1 분):

| 변형 | NPU vs ORT |
|---|---:|
| output projection 7 개만 | 0.569 ❌ |
| layer 10 FC2 만 | 0.930 |
| layer 2 / 7 / 6+8 / 4 / 9 만 | 0.93~0.94 |
| **layer 5 만** | **0.788 ❌** |

layer 5 1 층 추출: 접은 matmul 까지 자른 조각은 NPU = ORT(최대 3 LSB), **residual+LayerNorm 만 자른
조각이 틀림** — 단 한 원소, **row 0 col 308: ORT −128, NPU +124**. BERT 의 [CLS] 이상치 차원(308)이
정규화 후 약 −132 → int8 로 포화하면 −128, **wrap 하면 −132+256 = +124**.

접기가 이 층의 LayerNorm 을 CPU 에서 NPU ukernel 로 옮겼고, 그 ukernel 이 범위 밖 값을 처음 만난 것이다.
(재양자화 스택 넘침·int16 residual 합 오버플로·통계 정밀도도 차례로 확인해 배제.)

## 3. ukernel 버그 — `ssrs` 는 포화 모드가 꺼져 있으면 포화하지 않는다

`layernorm.cc`·`softmax.cc` 둘 다 마지막 int8 변환을 `ssrs(...)` 로 하며 주석에 "saturates" 라고
적었다. aie2p 의 `ssrs` 는 코어 `crSat` 모드가 켜져 있을 때만 포화하고 **기본은 wrap** 이다.

- **LayerNorm**: 위 사례.
- **softmax**: 확률 ≥ 0 이라 위로만 넘친다. 출력 스케일이 `1/s_out > 127.5` 이면(캘리브레이션이 1 보다
  작은 확률만 봤을 때) p ≈ 1 이 127 을 넘는다 — **12 층 중 6 층**(0, 1, 8, 9, 10, 11; layer 11 은
  최대 0.72 만 봄). 입력 5 종에서 81 원소가 넘고, NPU 에서 확인: layer 8 `[0,2,7,8]` ORT 127, NPU **−125**.
  softmax 를 NPU 로 올린 9/17 부터 있던 버그다.

수정(`0c34c18`, 로컬 `bert`): 변환 직전 bf16 값을 `min(max(x, -128), 127)` 로 자른다(둘 다 bf16 정확).
floor 가 −128..127 을 정확히 준다.

⚠️ **포화 모드를 켜는 방법(`set_sat()`)은 틀렸다** — 시도했더니 커널의 bf16→int 변환까지 바뀌어
더 나빠졌다(col 308 이 0, 정확 일치 52 %).

| 검사 | 결과 |
|---|---|
| layer 5 residual+LN 조각 | 최대 **1 LSB**, col 308 = −128 |
| layer 5 1 층(접음) | 최대 **3 LSB** (안 접은 모델과 같은 수준) |
| layer 8 softmax 조각 | 최대 **2 LSB**, `[0,2,7,8]` = 127 |
| 12층 `enc12x` | NPU vs ORT 0.569 → **0.937** |
| 12층 `enc12hv` | `l0i8_real` 해시 **불변**(`976028ff…`); `l0i8_rand`·`p_rand2` 는 **해시 변경** — 그 입력들에서 softmax wrap 이 실제로 일어나고 있었다 |

⚠️ 이전 문서들의 "NPU 출력이 기준 해시와 동일" 은 `l0i8_real` 외 입력에서는 **wrap 이 섞인 해시**였다.
새 기준: `enc12hv` rand `37bab720…`, rand2 `0271edb7…`; `enc12x` real `8c9aa4e5…`.

## 4. 재현

```bash
F=_local/int8_debug/tools
python3 $F/fold_bias_into_k_general.py enc12_i8.onnx enc12auto_k_i8.onnx --pad-value auto --max-bias-lsb 64
python3 $F/head_split.py enc12auto_k_i8.onnx enc12autohs_k_i8.onnx --k
python3 $F/hoist_kfold_pad.py enc12autohs_k_i8.onnx enc12x_k_i8.onnx
./scripts/docker/run-dev.sh /workspace/_local/padbe/build_final.sh enc12x_k_i8
./scripts/docker/run-dev.sh /workspace/_local/padbe/revalidate.sh      # 두 모델 × 5 입력 × 2 회
# layer 5 / layer 8 조각: _local/padbe/run_cut.sh, run_l8sm.sh (모델 L5cutB_int8, L8sm_int8)
```
