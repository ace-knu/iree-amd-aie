---
date: 2026-09-28
topic: [accuracy, int8]
status: partial
summary: "실제 문장 90개로 재캘리브레이션: 평가 16문장 fp32 대비 0.40 → 0.66 (NPU = ORT). Percentile 99.999 > MinMax. bias 접기가 오히려 정확도를 올림(재양자화-후-bias 손실 제거). 0.9 까지는 per-tensor 양자화 자체의 한계"
---
# 실제 문장으로 재캘리브레이션 (2026-09-28)

`2026-09-28_full_bert_e2e_npu_cpu.md` 에서 실제 문장의 fp32 대비 corr 이 0.33~0.50 이었던 원인이
무작위 토큰 캘리브레이션이었는지 확인한다.

## 0. 결론

평가 16 문장(캘리브레이션에 안 쓴 것), 실제 토큰 위치만, fp32 `bert_base.onnx` 대비:

| 모델 | corr 평균 (최저) |
|---|---:|
| 기존: 무작위 토큰 16 행, MinMax | 0.402 (0.250) |
| 실제 문장 90 개, MinMax | 0.585 (0.518) |
| 실제 문장 90 개, Percentile 99.999 | 0.598 (0.525) |
| ↑ + 모델 준비(bias 접기·head split·hoist) | **0.663 (0.604)** |
| ↑ NPU 실행 | **0.663** (ORT 0.663 과 같음) |

- 캘리브레이션만으로 0.40 → 0.60. **그리고 bias 접기가 0.60 → 0.66 으로 더 올린다** — 원래 그래프는
  matmul 출력을 먼저 int8 로 자른 뒤 bias 를 더하는데, 접으면 bias 가 누산기에 들어간 뒤 잘린다. 실제
  문장에서는 이 "자른 뒤 더하기" 손실이 무작위 토큰보다 훨씬 자주 일어난다(9/28 layer 10 FC2 분석과 같은 현상).
- 이 캘리브레이션에서는 layer 7·8 output projection 의 bias 가 78 / 81 재양자화 LSB 라 `--max-bias-lsb 64`
  에 걸린다. 접는 쪽이 정확하므로 `--max-bias-lsb 128` 로 36 개 전부 접었다(정확도 동일 0.663 vs 0.664,
  대신 두 LayerNorm 이 NPU 로). 디스패치 136 = NPU 131 / CPU 5 그대로.
- 앞서 쓴 3 문장: 0.469 / 0.488 / 0.356 → **0.641 / 0.697 / 0.651**.

## 1. 남은 격차 — 0.66 에서 멈추는 이유(추정)

캘리브레이션을 바꿔도 per-tensor 대칭 양자화라는 방식은 그대로다. BERT 의 residual stream 에는 특정
차원(308 등)에 큰 이상치가 있어서, per-tensor 스케일이 그 이상치에 맞춰지면 나머지 값은 몇 개의 int8
단계에만 몰린다. 9/10 에 측정한 레버(무작위 캘리브레이션 기준):

| 레버 | 효과 | 상태 |
|---|---:|---|
| per-channel 가중치 | +0.085 (최대) | NPU 백엔드 컴파일 실패 (`2026-09-10_per_channel_int8_blocked_in_npu_backend.md`) |
| matmul 출력 중간 반올림 제거 | +0.027 | bias 접기가 일부 해당 |

⚠️ 어느 텐서가 얼마를 잃는지는 **아직 분해하지 않았다**. 다음은 텐서별 민감도 측정(한 종류씩 fp32 로
두고 재기)이 먼저다 — 결과에 따라 per-channel 지원, 특정 텐서만 16-bit, 이상치 처리 중 무엇을 할지 정한다.

## 2. 재현

```bash
C=_local/calib    # calib_sentences.txt(90), eval_sentences.txt(16) → *_ids.npy (bert-base-uncased, 길이 32)
python3 $C/quantize_bert_calib.py models/bert_base/bert_base.onnx $C/q_pct.onnx \
  --calib-ids $C/calib_ids.npy --method percentile --percentile 99.999
# 이후 prepare_bert_i8.sh 의 2~6 단계와 같음, 단 fold_bias_into_k_general 은 --max-bias-lsb 128
python3 $C/eval.py <모델들...>          # 평가 16 문장, fp32 대비
./scripts/docker/run-dev.sh /workspace/_local/padbe/run_eval.sh bertpct2_int8   # NPU
```
