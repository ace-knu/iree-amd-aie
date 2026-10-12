# BERT-base int8 정확도 레시피 — 재현

이 폴더 하나로, 정확도 작업에서 NPU 로 잰 int8 배포 모델 네 개를 fp32 export 에서부터 다시 만들고,
컴파일하고, NPU 64 문장으로 다시 잴 수 있다. 원래 스크립트(`../quantize_bert_base.py`, `../quant/*.py`,
`../fold_*.py` 등)는 그대로 두고, 바꾼 것은 이 폴더에 복사본으로 넣었다.

| 단계 | NPU 64 문장 | ORT 참조 | 더한 것 | 문서 |
|---|---:|---:|---|---|
| A | 0.9503 | 0.9509 | SmoothQuant α 0.6, FC1·FC2 가중치 per-channel | `docs/2026-09-29_int8_accuracy_work_log.md` §7.7 |
| B | 0.9690 | 0.9697 | A + GELU 출력(FC2 입력) 비대칭 | `docs/2026-10-07_bert_int8_gelu_asymmetric.md` |
| C | 0.9773 | 0.9785 | 규칙 레시피 (구조로 고른 SmoothQuant 위치, 보정 데이터로 고른 위치별 α, head split Q/K/V 외 전부 per-channel, scale 이 1.25 배 이상 주는 곳 비대칭) | `docs/2026-10-07_bert_int8_rule_based_recipe.md` |
| D | 0.9845 | 0.9852 | C + GELU 하한 (FC1 출력 하한 −4) | 같은 문서 §8 |

정확도는 fp32 BERT 대비 피어슨 corr, `../quant/eval_ids.npy` 64 문장의 실제 토큰 위치 평균이다.

## 필요한 것

- 이 브랜치로 빌드한 컴파일러 (`build/tools/iree-compile`, `iree-run-module`). torch-mlir int8 bmm 패치를
  먼저 적용해야 한다: `patches/third_party/apply.sh`.
- Python: `onnx`, `onnxruntime==1.29.0`, `numpy`. 모델 바이트 일치는 onnxruntime 1.29.0 기준이다.
- fp32 모델: `../export_bert_base.py` 로 만든 `../bert_base.onnx`. torch 버전이 달라도 그래프·가중치는 같고
  `producer_version` 만 다르다(확인함). 이 레시피의 결과 바이트에는 영향이 없다.

## 1. 모델 만들기 (코어 2 개로 몇 분)

```bash
models/bert_base/recipe/build_models.sh models/bert_base/bert_base.onnx /tmp/recipe_out
```

마지막에 네 배포 모델의 md5 를 NPU 로 잰 모델의 md5 와 비교한다. 모두 `OK` 면 바이트까지 같은 모델이다.

| 출력 | 내용 |
|---|---|
| `{A,B,C,D}/bertx_int8.onnx` | 배포 모델. 이것을 컴파일한다 |
| `{A,B,C,D}_ref.onnx` | ORT 참조 모델. attn_out·FC2 출력의 재양자화를 int16 가짜 양자화로 바꾼 것 — 컴파일러가 NPU 에서 하는 일(`--iree-amdaie-widen-residual-requantization`, 기본 켜짐)과 같다. 배포 모델을 ORT 로 그대로 돌리면 이 넓히기가 빠져 더 낮게 나온다 |
| `{A,B,C}/log.txt` | bias 접기·head split 로그 |
| `alpha.json` | 단계 C 의 위치별 α 와 오차 표 |

## 2. 컴파일 (약 6 분)

```bash
models/bert_base/recipe/compile.sh /tmp/recipe_out/D/bertx_int8.onnx /tmp/recipe_out/D.vmfb <build_dir>
```

`<build_dir>` 를 생략하면 저장소의 `build/`. peano 는 `$PEANO` (기본 `/workspace/llvm-aie`).
`--iree-flow-inline-constants-max-byte-length=16384` 가 꼭 필요하다(채널별 상수 표를 코어 지역 상수로 만들려면
표가 인라인돼야 한다). 디스패치는 네 단계 모두 136 개(NPU 131 / CPU 5)다.

vmfb 에는 import 한 MLIR 파일의 경로와 이름이 들어간다(모듈·실행 파일 이름, 소스 위치). 그래서 같은 모델이라도
MLIR 경로가 다르면 vmfb 바이트가 달라진다(내용은 같다). 기본값은 출력 옆 `<이름>.mlir` 이고, `MLIR=<경로>` 로
고정할 수 있다. 측정에 쓴 vmfb 는 `MLIR=/tmp/asym/cur.mlir` 로 만든 것과 바이트까지 같다(단계 D 로 확인).

## 3. NPU 에서 재기

공유 NPU 락을 쓴다. 실행 전에 `scripts/lock/status.sh`, `scripts/lock/check-containers.sh` 로 확인.

```bash
models/bert_base/recipe/run_npu.sh /tmp/recipe_out/D.vmfb /tmp/recipe_out/npu_D <build_dir>
BERT_FP32=models/bert_base/bert_base.onnx \
  python3 models/bert_base/recipe/npu_eval.py /tmp/recipe_out/npu_D /tmp/recipe_out/D_ref.onnx
```

`npu_eval.py` 는 ORT 를 intra-op 2 스레드로 돌린다. 문서의 ORT 수치가 모두 이 조건이고, int8 사슬이 민감해서
스레드 수(부동소수점 합산 순서)가 바뀌면 최저값이 0.002 정도 움직인다. `../quant/eval.py` 는 스레드를 정하지 않아
같은 모델에서도 조금 다르게 나올 수 있다.

단계 D 를 이 절차대로 처음부터 다시 해서 확인한 값: 배포 모델 md5 일치, vmfb 바이트 일치,
NPU 0.9845 (최저 0.9796), ORT 0.9852 (최저 0.9801), NPU − ORT −0.0007 (−4.8 SE).

## 파일

| 파일 | 원본 / 하는 일 |
|---|---|
| `build_models.sh` | 전체 파이프라인 (위 표의 네 단계) |
| `compile.sh`, `run_npu.sh`, `npu_eval.py` | 컴파일, NPU 64 문장 실행, 점수 |
| `quantize_pc.py` | `../quantize_bert_base.py` 에 `per_channel=True` 한 줄 |
| `splice_fc2.py`, `splice_mix.py` | per-tensor 모델에 per-channel 모델의 FC2 / FC1 가중치를 끼움 (단계 A) |
| `fold_bias_into_k_general_pc.py` | `../fold_bias_into_k_general.py` 복사본. per-channel 가중치면 bias 행을 `round(bias[n] / (s_x·s_w[n]))` 로 |
| `fold_bias_into_k_general_pcz.py` | 위 + 활성값 zero-point. 덧붙이는 열에 `v + zx` 를 넣어 `x_q − zx = v` |
| `asym_q.py` | GELU 출력 비대칭: 상한 127·scale 그대로, 하한 = 보정 최솟값 (단계 B) |
| `alpha_rule.py` | 위치별 α: 보정 데이터의 matmul 출력 상대 오차 최소 (단계 C) |
| `smooth_rule.py` | `../quant/smooth.py` 복사본. 위치를 구조로 고르고 α 를 위치별 JSON 으로 받음 |
| `splice_pc.py` | 모든 가중치 per-channel, `--skip-head-split` 이면 head split 되는 Q/K/V 는 per-tensor |
| `asym_rule.py` | 통계 비대칭 규칙: scale 이 1.25 배 이상 줄고 컴파일러가 지원하는 위치 |
| `gelu_clip.py` | GELU 하한 규칙: 쓰는 곳이 GELU 하나뿐이면 하한 −4 (단계 D) |
| `both_int16.py` | ORT 참조 모델: attn_out·FC2 출력 int16 가짜 양자화 |

복사본들의 `sys.path.insert(0, "/tmp/pylibs")` 는 개발 컨테이너에서 쓰던 줄이고, 경로가 없으면 아무 일도 하지 않는다.
