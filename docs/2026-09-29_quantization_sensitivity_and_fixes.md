---
date: 2026-09-29
topic: [accuracy, int8]
status: partial
summary: "BERT int8 정확도 손실은 activation, 그중 FFN·residual 경로(FC2 출력, LayerNorm 출력, GELU 출력)가 지배. ORT 시뮬: FC2 출력 int16 + SmoothQuant(α 0.6) 로 64 문장 fp32 대비 0.595 → 0.837. 상한(해당 48 텐서 fp32) 0.93"
---
# int8 정확도 — 어느 텐서가 지배적이고, 무엇이 고치나 (2026-09-29)

`2026-09-28_real_sentence_calibration.md`(실제 문장 캘리브레이션 0.66)의 후속. 컴파일러·NPU 는 건드리지
않고 **ORT 에서만** 측정했다(NPU 출력 = ORT int8 이므로 ORT 결과가 NPU 결과를 대표한다).

## 0. 결론

- **비정상(버그)이 아니다.** BERT W8A8 per-tensor PTQ 는 원래 크게 무너진다 — Bondarenko et al.
  (EMNLP 2021): GLUE 83.1 → 71.0, MNLI 84.9 → 50.3, QNLI 91.6 → 52.3; 가중치만 int8 은 손실 없음(83.2).
  2026 재현(arXiv 2603.04308): QNLI 89.7 → 54.3.
- 이 모델에서도 **손실은 activation**, 그중 **FFN·residual 경로**가 지배한다.
- ORT 시뮬레이션: **FC2 출력 int16 + SmoothQuant(α 0.6)** 로 **0.595 → 0.837**.

평가: 캘리브레이션과 겹치지 않는 **64 문장**, 실제 토큰 위치만, fp32 `bert_base.onnx` 대비 corr 평균.
기준 모델: 실제 문장 90 개 · Percentile 99.999 캘리브레이션(`q_pct.onnx`).

## 1. leave-one-out — 어느 텐서가 지배적인가 (16 문장)

`_local/calib/loo.py`: 한 종류 텐서의 Q/DQ 쌍만 없애 fp32 로 두고 잰다.

| fp32 로 남긴 것 | corr |
|---|---:|
| 없음 (W8A8) | 0.598 |
| 가중치만 int8, activation 전부 fp32 (W8A32) | **0.948** |
| activation 만 int8 (W32A8) | 0.606 |
| **FC2 출력** (12) | **0.702** |
| LayerNorm 출력 전부 (24; QKV·FC1 입력이자 residual) | 0.661 |
| GELU 출력 (12; FC2 입력) | 0.638 |
| QKV·헤드·score·softmax 확률·context·attn out·FC1 출력 | 0.58~0.61 (영향 없음) |
| FC2 출력 + LN 출력 (36) | 0.783 |
| **FC2 출력 + LN 출력 + GELU 출력 (48 = activation 의 25 %)** | **0.929** |
| FC2 출력, layer 10–11 만 / 6–11 / 0–5 | 0.615 / 0.653 / 0.651 |

- 논문과 같다: 가중치는 무해, activation 이 원인, FFN 출력이 최대 단일 요인.
- 논문과 다르다: 한 텐서로 다 회복되지 않고 **세 종류가 합쳐져야** 한다. 깊은 층에 몰려 있지 않고 층
  전체에 퍼져 있다(사전학습 모델이라 그럴 수 있다 — 논문은 GLUE fine-tune 모델).
- attention 내부는 int8 로 둬도 된다.

(⚠️ 스크립트의 `layer_in` 패턴이 attention 뒤 LN 까지 잡아서, "LN 출력 전부" 로 읽어야 한다.)

## 2. 두 해법의 시뮬레이션 (64 문장)

- **A. FC2 출력 int16** (`fc2_int16.py`): FC2 출력의 Q/DQ 를 같은 범위의 int16 가짜 양자화로.
- **B. SmoothQuant** (`smooth.py`): matmul 입력에서 `s_j = max|X_j|^α / max|W_j|^(1−α)`,
  `X' = X/s`, `W' = diag(s)W`. LN 출력은 γ/β 에 접고, residual 은 **양자화된 X'** 에 s 를 곱해 되돌린다
  (NPU 에서 저장되는 건 int8 하나뿐이므로 그대로 시뮬레이션 — 양자화 도구가 LN 출력의 DQ 를 residual 과
  matmul 이 공유하게 만든 것을 확인). GELU 출력은 FC2 앞에서 ÷s. 변환된 fp32 는 원래와 최대 6.8e-6 차이.
  변환 후 같은 실제 문장으로 재캘리브레이션.

| 변형 | corr (최저) |
|---|---:|
| 무작위 토큰 캘리브레이션 (예전) | 0.425 |
| 기준: 실제 문장 · Percentile | 0.595 (0.525) |
| + bias 접기 (현재 배포 경로) | 0.667 |
| **A** | 0.701 (0.638) |
| B, α 0.5 / 0.6 / 0.75 | 0.664 / 0.716 / 0.674 |
| **C = A + B** | α 0.25 0.48 · 0.4 0.55 · 0.5 0.79 · **0.55 0.839** · **0.6 0.837 (0.812)** · 0.65 0.80 · 0.7 0.74 · 0.75 0.72 |
| A + bias 접기 / C(0.5) + bias 접기 | 0.704 / 0.784 |

- A 와 B 는 **다른 손실을 잡아서 거의 더해진다.** 최고 C(α 0.6) **0.837**, 상한(48 텐서 fp32) 0.929.
- α 에 **민감하다** — 0.55~0.6 에 뚜렷한 봉우리. 0.4 이하는 급락.
- bias 접기의 이득(+0.07)은 FC2 출력 int8 절단을 줄이던 것이라, FC2 가 int16 이면 사라진다.

## 3. NPU 로 옮길 때 필요한 것 (미구현)

| 해법 | 필요한 변경 |
|---|---|
| FC2 출력 int16 | 정수 재양자화가 i16 을 내보내기; LN 의 residual 합 생산자가 i16 + i8 을 받기(현재 i8 + i8 → i16). 합의 범위가 int16 을 넘을 수 있어 **step 선택 재검토 필요** |
| SmoothQuant — LN 출력 | γ/β 접기는 공짜. residual 쪽 ×s 가 **채널별** 이 되어 residual 생산자의 스칼라 승수 Mb 를 채널별 벡터로 |
| SmoothQuant — GELU 출력 | FC1+GELU 디스패치 꼬리에 채널별 ÷s (이미 float 꼬리라 비용 작음) |

모델 쪽(스케일 계산·접기)은 `prepare_*.sh` 의 한 단계로 들어간다.

## 4. 재현

```bash
C=_local/calib
python3 $C/loo.py $C/q_pct.onnx                                         # §1 (16 문장 시점 결과)
python3 $C/fc2_int16.py $C/q_pct.onnx $C/v_A.onnx                        # A
python3 $C/smooth.py models/bert_base/bert_base.onnx $C/sq06_fp32.onnx $C/calib_ids.npy 0.6
python3 $C/quantize_bert_calib.py $C/sq06_fp32.onnx $C/sq06_q.onnx --calib-ids $C/calib_ids.npy \
  --method percentile --percentile 99.999                                # B
python3 $C/fc2_int16.py $C/sq06_q.onnx $C/v_C06.onnx                     # C
python3 $C/eval.py <모델들...>                                           # 64 문장
```

참고: [Bondarenko et al. 2021](https://arxiv.org/abs/2109.12948),
[arXiv 2603.04308](https://arxiv.org/abs/2603.04308), [SmoothQuant](https://arxiv.org/abs/2211.10438).
