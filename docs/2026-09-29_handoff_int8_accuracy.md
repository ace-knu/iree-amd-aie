---
date: 2026-09-29
topic: [team-share, accuracy, int8]
status: reference
summary: "팀원 인계: BERT-base int8 NPU 정확도 개선. 현재 fp32 대비 0.667, 원인 특정(FFN·residual activation), ORT 시뮬로 0.837 까지 확인. 남은 일 = 백엔드 국소 수정 3 개(FC1 채널별 상수, residual 채널별 배수, FC2 출력 int16)"
---
# 인계: BERT int8 정확도 개선 (2026-09-29)

## 0. 한 줄 요약

BERT-base 전체(input_ids → last_hidden_state)가 NPU+CPU 로 돈다(136 디스패치 = NPU 131 / CPU 5).
**NPU 는 같은 int8 모델의 onnxruntime(ORT) 결과와 정확히 같은 정확도**를 내므로 하드웨어·컴파일러 경로는
정상이다. 남은 문제는 **int8 양자화 품질**이다.

| | 실제 문장, fp32 BERT 대비 corr |
|---|---:|
| 현재 배포 경로 | **0.667** |
| ORT 시뮬레이션으로 확인한 목표 | **0.837** |
| 상한 (문제 텐서를 fp32 로 뒀을 때) | 0.929 |

할 일은 **백엔드의 국소 수정 3 개**(§4). 전부 DMA/objectfifo 가 아니라 우리가 만든 패스 범위 안이다.

## 1. 재현 — 지금 상태를 먼저 돌려 볼 것

`bert-onnx` 를 받는다(서브모듈 정정·패치 안내는 `docs/2026-09-28_bert_onnx_submodule_and_patches.md`).

```bash
git fetch origin && git checkout bert-onnx && git pull
git submodule update --init --recursive
./patches/third_party/apply.sh          # torch-mlir 패치 1 개
# IREE 재빌드 (./scripts/build/build.sh — 빌드 lock 자동)
```

모델 만들기·컴파일·실행은 `models/bert_base/README.md` §7. **정확도 작업은 `CALIB=real` 로 시작한다**
(실제 문장 90 개 + Percentile 99.999 캘리브레이션):

```bash
CALIB=real models/bert_base/prepare_bert_i8.sh models/bert_base/bert_base.onnx models/bert_base/out
python3 -m iree.compiler.tools.import_onnx models/bert_base/out/bertx_int8.onnx -o bert.mlir
iree-compile bert.mlir -o bert.vmfb <README §6.2 의 플래그>
./scripts/lock/with-npu-lock.sh iree-run-module --device=amdxdna --device=local-task \
  --module=bert.vmfb --function=main_graph --input=@<input_ids int64 [1,32] .npy>
```

파이썬 의존성: `onnx`, `onnxruntime`(1.29.0), `ml_dtypes`.

기대값:
- 디스패치 **136 = NPU 131 / CPU 5**(임베딩 1, 임베딩 LN 2, 마지막 LN 2). 반복 실행 결과 동일.
- `bertx_int8.onnx` 는 NPU 에서 검증한 모델과 **바이트 동일**(md5 `a053ffb4…`, 2026-09-29 확인).
- 평가 64 문장 fp32 대비 **0.667**(`python3 models/bert_base/quant/eval.py models/bert_base/out/bertx_int8.onnx`).

`CALIB` 없이 돌리면 예전 기본값(무작위 토큰)이라 실제 문장에서 **0.40** 이 나온다 — 정확도 작업엔 쓰지 말 것.

## 2. 정확도를 재는 법 (중요)

- **기준은 fp32 BERT** (`models/bert_base/bert_base.onnx`). 같은 토큰을 넣고 `last_hidden_state` 의
  **실제 토큰 위치만** 비교한다(`[PAD]` 제외). 지표는 corr 평균(최저값도 같이 볼 것).
- **평가 문장은 캘리브레이션 문장과 겹치지 않게**. 지금 세트: 캘리브레이션 90 문장, 평가 64 문장.
- ⚠️ **int8 모델끼리의 corr 은 정확도 지표로 쓰면 안 된다.** 12 층 int8 체인은 극소 차이를 크게 증폭한다
  — bias 오차 0.003 LSB 만 다른 두 모델끼리도 corr 0.956 이 나왔다. 항상 fp32 대비로.
- **NPU 검증** = 같은 모델의 ORT 대비 fp32 corr 과 NPU 대비 fp32 corr 이 같은지. 다르면 백엔드 버그다.
  (실제로 이렇게 LayerNorm·softmax ukernel 의 int8 wrap 버그를 찾았다 — `ssrs` 는 포화 모드가 꺼져 있으면
  포화하지 않는다. 수정됨. 증상: 소수 원소가 256−k LSB 정도 틀림.)
- 성능(속도)은 지금 재지 말 것 — 장비가 디스패치당 ~95 ms 로 느려진 상태다
  (9/23 재부팅 이후 생긴 장비 상태 문제로, 우리 코드 원인은 배제됨. 조사 중).

## 3. 지금까지 알아낸 것

### 3.1 비정상이 아니다 — 알려진 현상

BERT W8A8 per-tensor PTQ 는 원래 크게 무너진다. Bondarenko et al. (EMNLP 2021): GLUE 83.1 → 71.0,
MNLI 84.9 → 50.3, QNLI 91.6 → 52.3. **가중치만 int8 이면 손실 없음**(83.2). 손실은 전부 activation 에서
나오며, 원인은 소수 임베딩 차원의 구조적 이상치(우리 모델에서는 308 번 차원 등).

### 3.2 캘리브레이션

| | corr |
|---|---:|
| 무작위 토큰 | 0.425 |
| 실제 문장, Percentile 99.999 | 0.595 |
| ↑ + bias K 축 접기 (배포 경로) | **0.667** |
| Percentile 99.99 / 99.9 / Entropy (+ 접기) | 0.540 / 0.444 / 0.631 — **클리핑은 해롭다**(이상치가 신호라서) |

### 3.3 어느 텐서가 지배적인가 (leave-one-out, 한 종류만 fp32 로)

| fp32 로 남긴 것 | corr |
|---|---:|
| 없음 | 0.598 |
| activation 전부 (가중치만 int8) | 0.948 |
| **FC2 출력** | **0.702** |
| LayerNorm 출력 전부 (QKV·FC1 입력이자 residual) | 0.661 |
| GELU 출력 (FC2 입력) | 0.638 |
| QKV · score · softmax · context · attn out · FC1 출력 | 0.58~0.61 (무관) |
| **FC2 출력 + LN 출력 + GELU 출력** (activation 의 25 %) | **0.929** |

attention 내부는 int8 로 둬도 된다. 문제는 **FFN·residual 경로**다. 깊은 층에 몰려 있지 않고 전 층에 퍼져 있다.

### 3.4 해법 시뮬레이션 (ORT, 64 문장)

| | corr |
|---|---:|
| A. FC2 출력 int16 | 0.701 |
| B. SmoothQuant (LN 출력 + GELU 출력), α 0.6 | 0.716 |
| B'. SmoothQuant GELU 출력만, α 0.6 (+ 접기) | 0.704 |
| **C. A + B, α 0.6** | **0.837** (최저 0.812) |

SmoothQuant: 입력 채널별 `s_j = max|X_j|^α / max|W_j|^(1−α)`, `X' = X/s`, `W' = diag(s)·W`.
⚠️ **α 에 민감**: 0.55~0.6 에서 봉우리, 0.4 이하는 급락(C 기준 0.5 → 0.79, 0.4 → 0.55).

## 4. 할 일 — 백엔드 국소 수정 3 개 (쉬운 순)

### ① FC1 디스패치에 채널별 상수 넣기 → GELU 쪽 SmoothQuant (+0.04)

GELU 출력을 FC2 앞에서 채널별로 `÷s`(3072 개 상수) 해야 한다. 지금 컴파일하면
`'amdaie.connection' op no producer DMA channel available` — 코어 입력 DMA 채널이 2 개(활성·가중치)뿐인데
상수가 세 번째로 L1 에 들어가야 해서다(stream 단계까지는 바인딩이 안 늘어난다).

**선례**: LayerNorm 의 γ/β 가 똑같은 문제였고 `AMDAIELowerToUKernels.cpp` 의
`materializeCoreLocalConstant` 로 풀었다 — 연산자 operand 가 아니라 코어 데이터 메모리의 `memref.global` 로.
FC1+GELU 꼬리(ukernel 아님, 생성된 elementwise 코드)의 작은 상수 operand 에 같은 처리를 하면 된다.

### ② LayerNorm residual 합의 배수를 채널별로 → LN 쪽 SmoothQuant

LN 출력을 `/s` 하면(γ/β 에 접으면 공짜) residual 쪽은 `×s` 로 되돌려야 한다. residual 합은 LN 디스패치
안에서 `h = (a·Ma + b·Mb + 2^7) >> 8` 로 만들어지는데(`AMDAIERaiseLayerNorm.cpp` 의
`computeResidualMultipliers` / `buildResidualProducer`), `Ma`, `Mb` 가 지금 **스칼라**다. 채널별 벡터로
바꾸면 된다. ukernel 은 그대로(int16 `h` 를 받는 `layernorm_i16_*` 가 이미 있음).

### ③ FC2 출력 int16 → 가장 큰 효과

FC2 의 재양자화를 int8 대신 int16 으로(`AMDAIEIntegerRequantization.cpp` — 지금 clamp 가 int8 범위),
residual 합이 i16 + i8 을 받게. ⚠️ **설계 과제**: `h` 는 int16 이어야 하는데(ukernel 의 int16 통계 경로가
64 비트 누산으로 정확) FC2 출력의 이상치 범위가 크므로, 공통 step 을 `(amax_a + amax_b)/32767` 이상으로
잡아야 넘치지 않는다. 그 step 으로도 int8 보다 충분히 곱지를 먼저 계산해 볼 것.

①~③ 모두 하면 C(0.837). ①만 하면 0.704.

## 5. 모델 쪽 도구 (`models/bert_base/quant/`)

전부 onnxruntime 만 쓴다(NPU·IREE 불필요). 사용법 표는 `models/bert_base/README.md` §8.
fp32 기준 모델은 `models/bert_base/bert_base.onnx`(다른 위치면 `BERT_FP32=`).

| 파일 | 역할 |
|---|---|
| `calib_sentences.txt` / `eval_sentences.txt` → `*_ids.npy` | 캘리브레이션 90 / 평가 64 문장 (bert-base-uncased, 길이 32) |
| `tokenize_sentences.py` | 문장 → ids (`transformers<5` 필요, 호스트에 없어 `pip install --target` 로) |
| `../quantize_bert_base.py` | `--calib-ids`, `--method minmax\|percentile\|entropy`, `--percentile` 추가(기본값은 예전과 바이트 동일) |
| `eval.py <모델들>` | 평가 64 문장, fp32 대비 corr 평균(최저) |
| `deploy_eval.sh <모델들>` | 배포 경로의 접기까지 적용한 것과 함께 평가 |
| `loo.py <int8 모델> [--combos]` | §3.3 leave-one-out(`--combos` 는 조합·층 범위) |
| `smooth.py <fp32> <out> <calib_ids> <α> [all\|ln\|gelu]` | SmoothQuant 변환 (fp32 동치 확인됨, 최대 6.8e-6) |
| `fc2_int16.py <int8> <out>` | FC2 출력 int16 가짜 양자화 (시뮬레이션용) |

2026-09-29 확인: 저장소 버전으로 다시 만든 모델이 문서의 수치를 낸 모델과 바이트 동일
(실제 문장 캘리브레이션, A, C α0.6), `eval.py` 수치 0.595 / 0.667 / 0.701 / 0.837 일치.
날짜 문서의 재현 절차에 나오는 `_local/calib/` 는 이 디렉터리다.

## 6. 검증 절차 (수정마다)

1. ORT 시뮬레이션에서 목표 수치 확인 (`eval.py`)
2. 컴파일 → stream 단계 디스패치 수 확인: **136 = NPU 131 / CPU 5 유지**
3. NPU 실행(lock 경유) → **NPU 대비 fp32 = ORT 대비 fp32** 인지, 반복 실행 동일한지
4. 회귀: attention 모델 vmfb MD5(`attn0_kpadq_int8` → `b35a583d`), 12 층 인코더 해시

## 7. 공유 호스트 주의

- NPU 실행은 반드시 `./scripts/lock/with-npu-lock.sh` 로. 실행 전 `status.sh`, `check-containers.sh`.
- 백그라운드 `docker run` 을 클라이언트째 죽이지 말 것 — 컨테이너를 `docker stop <id>` 로 직접.
- 시간 제한은 lock **안쪽**에만 건다(`with-npu-lock.sh timeout 900 <cmd>`). 바깥에 걸면 lock 을 기다리다
  제한이 끝나 NPU 실행 도중에 죽을 수 있다.
- **디스크**: 모델 파일이 하나 180~440 MB 다. 9/28 에 호스트 디스크가 가득 찼다. 중간 파일은 바로 지울 것.

## 8. 참고

- 문서: `docs/2026-09-28_real_sentence_calibration.md`, `docs/2026-09-29_quantization_sensitivity_and_fixes.md`,
  `docs/2026-09-28_all_biases_folded_and_ukernel_int8_wrap.md`, `docs/2026-09-28_full_bert_e2e_npu_cpu.md`
- [Bondarenko et al. 2021](https://arxiv.org/abs/2109.12948) — per-tensor PTQ 붕괴, leave-one-out, 혼합 정밀도, PEG
- [arXiv 2603.04308](https://arxiv.org/abs/2603.04308) — BERT-base 재현(QNLI 89.7 → 54.3)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
