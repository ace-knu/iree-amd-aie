# 2026-09-18 — residual add + LayerNorm 을 포함한 블록 추출과 디스패치 배치 실측

> 담당 범위: **attention 출력 → 첫 residual add → LayerNorm** 을 NPU 연속 경로로 만드는 것.
> FFN(FC1/GELU/FC2)은 다른 팀 범위다.
>
> 이 문서는 그 작업의 **출발 측정**이다. 설계를 하기 전에, residual+LN 을 붙였을 때
> 지금 컴파일러가 실제로 무엇을 어디에 배치하는지부터 숫자로 고정했다. **컴파일러 변경 없음.**

## 0. 한 줄 답

residual+LN 을 붙이면 **NPU 는 7 그대로, CPU 가 2 → 4 로 둘 는다.** 그리고 그 둘은
서로 다른 원인이다.

| | attention-only (기존) | +LN 만 | +residual+LN |
|---|---:|---:|---:|
| NPU | 7 | 7 | **7** |
| CPU | 2 | 3 | **4** |
| 합계 | 9 | 10 | **11** |

- **+1 은 LayerNorm 자체**: LN 이 표준 2-pass(평균 / 분산)로 내려가 디스패치가 하나 늘어난다.
- **+1 은 residual 의 부작용**: residual 이 입력 `x` 를 두 번째로 소비하면서, 기존에 입력
  quantize 에 융합돼 있던 K 축 Pad 가 떨어져 나와 `slow_memcpy` 디스패치가 된다.
  **residual add 연산 자체는 공짜로 흡수된다.**
- **LN 뒤 QuantizeLinear(= FC1 이 받는 i8 경계)도 공짜다.** f32 출력판과 i8 출력판의
  디스패치 목록이 완전히 동일하다.

## 1. 만든 것

`_local/int8_debug/tools/extract_attn_res_ln.py` (신규). `bert_base_kpadq_int8.onnx` 에서
기존 `attn0_*` 와 **같은 입력**으로 자르되, 끝을 두 op 뒤로 민다.

```text
기존 attn0 :  x ─> ... ─> proj MatMul ─> +bias                      (여기서 끝)
이번 attn0res: x ─> ... ─> proj MatMul ─> +bias ─> +x(residual) ─> LayerNorm ─> Quantize
                └──────────────────────────────────┘ residual 은 모델 입력에서 갈라진다
```

- 입력: `/m/embeddings/LayerNorm/LayerNormalization_output_0` `[1,32,768] f32` (기존과 동일)
- 마스크는 기존 `attn0_hsm` 과 같이 all-zero 상수로 굽는다(softmax 2-operand).
- 출력 두 가지를 다 만들었다.
  - `--i8-out` 없음 → LN 의 f32 출력 (`attn0res_hsm_int8`). 기존 attn0 와 같은 성격의 경계.
  - `--i8-out` → LN 뒤 `QuantizeLinear` 의 i8 출력 (`attn0res_hsq_int8`). **FC1 이 실제로 받는 경계.**
- `extract_model` 이 남기는 함정 두 개를 스크립트가 처리한다: 그래프 이름의 공백(bootgen 이
  임시 경로에서 죽는다)과, batch 차원이 symbolic 으로 남는 것.
- 그 뒤 기존 `gen_attn_headsplit.py` 를 그대로 적용해 Q/K/V 를 head 별 BMM 으로 만든다.

노드 49 개: MatMul 6, Softmax 1, LayerNormalization 1, Add 3, Transpose 4, Reshape 4,
Pad 1, Mul 1, Q/DQ 23.

### 1.1 추출이 맞는지 먼저 확인

새 모델의 ORT 출력이 **기존 attention 골든에 residual+LN 을 직접 씌운 값**과 같은지 봤다.

```python
xq  = dequant(quant(x))            # 그래프가 x 에 하는 것과 같은 per-tensor 양자화
ref = LayerNorm(attn0_hsm_ort + xq, w, b)
```

| | 값 |
|---|---|
| maxdiff | `2.861e-06` |
| mean abs | `1.180e-07` |
| corr | `1.000000000` |

f32 연산 순서 차이 수준이다. 추출은 정확하다.

## 2. 디스패치 배치 실측

플래그는 `docs/2026-09-17_attention_performance_comparison.md` §1.1 의 조합 +
`--iree-amdaie-enable-ukernels=softmax`. 모든 모델이 같은 플래그다.

**대조군 `attn0_hsm_int8` — NPU 7 / CPU 2 (9)**

| # | 장치 | 디스패치 |
|---|---|---|
| d0 | CPU | `elementwise_32x768_f32xi8` — 입력 quantize **+ K 축 Pad 융합**(출력 `32x832`) |
| d1–d3 | NPU | `batch_matmul_12x32x64x832` ×3 — Q / K / V |
| d4 | NPU | `batch_matmul_12x32x32x64` — QKᵀ |
| d5 | NPU | `softmax_12x32x32` |
| d6 | NPU | `batch_matmul_12x32x64x32` — PV |
| d7 | NPU | `batch_matmul_1x32x768x768` — output projection |
| d8 | CPU | `elementwise_32x768_f32xi8xf32` — dequant + bias |

**`attn0res_hsq_int8` (residual+LN) — NPU 7 / CPU 4 (11)**

| # | 장치 | 디스패치 |
|---|---|---|
| d0 | CPU | `elementwise_24576_f32xi8` — 입력 quantize (**Pad 융합 사라짐, 1-D 로 붕괴**) |
| d1 | CPU | **`slow_memcpy`** — `32x768 → 32x832` Pad |
| d2–d8 | NPU | **대조군과 동일한 7 개** |
| d9 | CPU | `reduction_32x768_f32` — dequant×2 + bias + **residual add** + 평균 |
| d10 | CPU | `reduction_32x768_f32` — 분산 + rsqrt + weight/bias + **최종 quantize** |

f32 출력판(`attn0res_hsm_int8`)의 디스패치 목록은 이것과 **완전히 동일**하다. LN 뒤
QuantizeLinear 는 d10 안으로 흡수되어 추가 비용이 0 이다. **encoder 내부 경계를 i8 로
두는 데 드는 디스패치 비용은 없다.**

### 2.1 늘어난 두 개를 원인별로 분리

**(a) LN 본체 = +1.** 기존 d8(dequant+bias)이 사라진 게 아니라 새 d9 안으로 흡수됐고,
LN 이 2-pass 로 내려가면서 하나가 더 생겼다.

```text
d9  : sitofp(proj_i8)*0.0237346236 + sitofp(x_i8)*0.079871498 + bias  ─> h
      h 의 행별 평균 ─> h - mean
d10 : (h-mean)^2 의 행별 합 ─> /768 + 1e-12 ─> rsqrt ─> *w + b ─> quantize(i8)
```

**(b) residual 이 Pad 융합을 깨뜨림 = +1.** 통제 실험으로 확정했다.
같은 모델에서 **residual add 만 제거**하면(`_local/res/no_residual.py`, LN 을 projection
출력에 바로 씌움 — 수치는 달라지지만 소비자 수만 바뀐다):

| | d0 | `slow_memcpy` | 합계 |
|---|---|---|---:|
| residual 있음 | `elementwise_24576` (Pad 분리) | **있음** | 11 |
| residual 없음 | `elementwise_32x768` (**Pad 융합 복귀**) | **없음** | 10 |

즉 residual add 자체는 d9 에 공짜로 융합되지만, `x` 가 두 번째 소비자를 갖는 순간
입력 quantize+Pad 융합이 깨진다. 이는 메모리에 기록된 `FormDispatchRegions.cpp:396-411`
의 "비-aggressive 모드에선 소비자 1 개인 생산자만 융합" 제약과 부합한다.

⚠️ **ONNX 수준 우회는 실패했다.** residual 쪽에 같은 스케일의 QuantizeLinear 를 하나 더
만들어 소비자를 분리해 봤지만(`_local/res/dup_quant.py`), IREE 가 둘을 CSE 로 다시 합쳐서
결과가 동일했다(11 디스패치, `slow_memcpy` 그대로). 해결하려면 컴파일러 쪽이거나,
Pad 자체를 없애는 방향이어야 한다.

### 2.2 `--iree-dispatch-creation-enable-fuse-padding-into-linalg-producer-ops` 는 여기선 쓰면 안 된다

이 플래그는 **대조군을 9 → 10 으로 악화**시킨다(융합돼 있던 quantize+Pad 를 떼어 내
`tensor.pad` 단독 디스패치로 만든다). 신규 모델은 11 → 11 로 변화가 없다
(`slow_memcpy` 가 `tensor.pad` 디스패치로 이름만 바뀐다).
출력은 세 모델 모두 **비트 동일**하므로 수치 중립이지만, 디스패치 수로는 손해다.
(메모리의 "지금 바로 쓸 것" 메모는 full BERT `bb_kpadq` 맥락이었다. attention 계열 모델엔
해당되지 않는다.)

## 3. 하드웨어 실측

기준 플래그로 빌드한 세 vmfb 를 실제 NPU 에서 각 3 회 실행했다(`with-npu-lock.sh`).

| 모델 | 출력 | 3 회 해시 | corr (ORT 대비) | maxdiff | 비고 |
|---|---|---|---|---|---|
| `b_ctrl` (= `attn0_hsm_int8`) | f32 | **1 종** `53463eda89b6f889` | 0.9987321 | 0.0474694 | |
| `b_res_f32` | f32 | 1 종 `5d29866060b3f74f` | 0.9998524 | 0.2068956 | |
| `b_res_i8` | i8 | 1 종 `b53a0df4d62c9f8d` | 0.9980824 | **1 LSB** | 24,576 중 **612 개만** ±1 차이 |

- **대조군 vmfb 의 MD5 가 `1b8490a5b86d86bc89d48ff1f0cefe18`** 로, 9/17 문서의 하드웨어
  검증본 `attn0_hsm_sm_nb.vmfb` 와 **동일**하다. 출력 해시도 기록된 `53463eda…` 와 같다.
  이번 측정 환경이 기존 검증본을 그대로 재현한다는 뜻이다.
- i8 경계 출력은 97.5 % 가 비트 동일하고 나머지도 정확히 1 LSB 다.
- 세 모델 모두 반복 실행 단일 해시 — 비결정론 없음.
- padfuse 플래그로 빌드한 판과 기준 플래그 판의 출력은 세 모델 모두 **바이트 동일**이다.

## 4. 이 측정이 설계에 주는 것

1. **옮겨야 할 대상은 CPU 디스패치 2 개(d9, d10)와, 그 앞의 Pad 문제 1 개다.**
2. **d9 의 앞부분은 정수 산술로 바꿀 수 있는 형태다.** `acc = bias + s_p·p + s_x·x` 에서
   `p`, `x` 가 모두 i8 이고 두 스케일이 상수다. 다만 기존
   `AMDAIEIntegerRequantization` 은 **i32 matmul accumulator 에서 시작하는 꼬리**만
   다루므로 그대로는 적용되지 않는다 — 확장 대상.
3. **LN 본체는 f32 가 본질이다.** `rsqrt`, `divf`, 행별 평균/분산이 들어 있어 aie2p 의
   스칼라 float 부재에 정면으로 걸린다. d0/d8 을 NPU 로 못 올린 이유와 같은 벽이다.
   softmax 를 `softmax_i8_96x32` ukernel 로 흡수한 것과 같은 해법 —
   **i8 in / i8 out 의 LayerNorm ukernel** — 이 현재로선 선례가 있는 유일한 경로다.
4. **경계 자료형은 이미 i8 로 맞아 있다.** d9 의 tensor 입력 두 개가 모두 i8 이고 출력
   경계도 i8 로 공짜다. i32 direct fusion 용 전용 ABI 없이 시작할 수 있다.
5. ⚠️ **d9 는 tensor 모양 입력이 3 개**(bias f32[768], proj i8[32,768], x i8[32,768])다.
   `docs/2026-09-03_method_b_compiles_but_still_hangs.md` 의 결론 — 세 번째 tensor 채널이
   `packet-flow-strategy=inputs` 를 강제하고 그게 hang 패턴으로 이어진다 — 이 그대로
   걸릴 수 있다. NPU 이식 설계에서 **먼저 확인할 위험**이다. bias 를 LN 의 weight/bias 와
   합치거나 K 축으로 접어 채널 수를 줄이는 것이 회피책 후보.
6. **residual 이 Pad 융합을 깨는 문제는 LN 이식과 독립적으로 존재한다.** 지금은 CPU
   `slow_memcpy` 하나지만, encoder 를 이어 붙이면 layer 마다 반복된다.

## 5. 아직 안 한 것

- **성능 측정.** d9+d10 이 host CPU 시간에서 실제로 얼마를 먹는지 재지 않았다.
  `docs/2026-09-17_d1_benchmark_and_d0_feasibility.md` 의 방법(순서균형 쌍체 + CI)으로
  대조군 vs 신규를 재면 "residual+LN 을 NPU 로 옮겨 아낄 수 있는 예산"의 상한이 나온다.
  d0/d9 때처럼 **아낄 예산보다 이식 비용이 큰 경우**가 있으므로 설계 전에 재는 게 맞다.
- LayerNorm ukernel 의 실현 가능성 조사(정밀도, L1 예산, 타일 모양).
- full BERT 연결. FC1 은 여전히 별개 문제이고 이 문서의 범위 밖이다.

## 6. 산출물

| 경로 | 내용 |
|---|---|
| `_local/int8_debug/tools/extract_attn_res_ln.py` | 추출 스크립트 (신규) |
| `_local/int8_debug/out/attn0res_{hsm,hsq}_int8.{onnx,mlir}` | f32 / i8 출력판 모델 |
| `_local/int8_debug/out/attn0res_hsq{d,n}_int8.{onnx,mlir}` | 진단용 변형 (중복 quantize / residual 제거) |
| `_local/int8_debug/out/attn0res_{hsm,hsq}_ort.npy` | ORT 골든 |
| `_local/res/b_{ctrl,res_f32,res_i8}.vmfb` | 기준 플래그 빌드 (MD5 `1b8490a5…` / `5cf91ba6…` / `a8bb163d…`) |
| `_local/res/{flow,stream,flow2,build_full,build_base,run_hw,run_hw2}.sh` | 재현 스크립트 |
| `_local/res/{golden,dup_quant,no_residual}.py` | 골든 생성 · 진단 변형 |
| `_local/res/*.stream.mlir` | 디스패치별 장치 배치 근거 |
