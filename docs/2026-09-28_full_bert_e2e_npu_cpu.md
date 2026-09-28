---
date: 2026-09-28
topic: [encoder-stack, accuracy]
status: partial
summary: "전체 BERT(input_ids → last_hidden_state) 이기종 실행: 136 = NPU 131 / CPU 5(임베딩 1 + f32 LayerNorm 2개). NPU 는 같은 int8 모델의 ORT 와 같은 정확도. 단 실제 문장에서는 int8 모델 자체가 fp32 대비 0.33~0.50 — 무작위 토큰 캘리브레이션 탓"
---
# 전체 BERT e2e — 임베딩은 CPU, 인코더는 NPU (2026-09-28)

지금까지 12 층 결과는 **인코더만 떼어낸 모델**(입력 = 임베딩 LayerNorm 의 int8 출력)이었다.
오늘은 토큰(`input_ids`)부터 `last_hidden_state` 까지 한 모델로 돌렸다.

## 0. 결론

| | 값 |
|---|---|
| 디스패치 | **136 = NPU 131 / CPU 5** |
| CPU 5 | 임베딩(Gather ×3 + 더하기) 1, 임베딩 LayerNorm 2(f32 입력), 마지막 LayerNorm 2(f32 출력) |
| K-fold 복사 | **0** — layer 0 QKV 넓히기의 생산자가 이제 임베딩 LN 의 quantize 라 흡수됨 |
| 반복 실행 | 입력 6 종 전부 run 간 동일 |
| 무작위 토큰 3 종 | NPU vs fp32 0.921 / 0.918 / 0.900 = ORT int8 vs fp32 0.924 / 0.920 / 0.884 |
| **실제 문장 3 개** | NPU vs fp32 **0.469 / 0.488 / 0.356**, ORT int8 도 0.460 / 0.497 / 0.324 |

**하드웨어 경로는 정확하다**(NPU = 같은 int8 모델의 CPU 실행). **실제 문장에서 나쁜 건 int8 모델
자체**다.

## 1. 모델 준비 — 전체 모델에서 필요했던 것 하나

`bert_base_kpadq_int8.onnx` 에 인코더와 같은 단계(`fold_bias_into_k_general.py --pad-value auto
--max-bias-lsb 64` → `head_split.py --k` → `hoist_kfold_pad.py`)를 적용하면 head split 이 0 개,
hoist 가 `None` 으로 죽는다. 배치 차원이 기호이고, 임베딩의 position/token-type id 와 attention mask 를
입력 모양으로 펼치는 `Expand`·`ConstantOfShape`·`Where` 사슬 때문에 배치를 1 로 고정해도 모양이
전파되지 않는다(`data_prop=True` 로도 739 개 텐서가 기호로 남음).

이 사슬은 `input_ids` 의 **값이 아니라 모양에만** 의존하므로 배치 1 에서는 상수다.
`_local/int8_debug/tools/fold_shape_consts.py`(신규): 모양 계산 op 만 골라 ORT 로 한 번 계산해
initializer 로 바꾼다(Q/DQ·MatMul·임베딩 Gather 는 제외). 127 노드 → 상수 100 개, 기호 텐서 0 개,
ORT 출력 바이트 동일. 그 뒤 세 단계가 인코더와 똑같이 적용된다(bias 36/36, head split 36/36,
Pad 36 개 흡수 대상 — layer 0 QKV 포함).

attention mask 는 이 export 에서 **상수(전부 1)** 라 `[PAD]` 도 attention 에 들어간다. fp32 기준 모델도
같은 조건이라 비교는 공정하다.

## 2. 실제 문장이 나쁜 이유 — 캘리브레이션

`quantize_bert_base.py` 는 **무작위 `input_ids` 16 행**으로 활성 범위를 잡는다. 무작위 토큰에서는
int8 모델이 fp32 대비 0.9 인데 실제 문장에서는 0.3~0.5 — 실제 텍스트의 활성 분포([CLS] 이상치, 자주
나오는 토큰의 큰 값)가 무작위 토큰에서 본 범위를 넘어 잘린다. 9/10 문서의 "손실의 지배 요인은
캘리브레이션 없이 만든 QDQ" 와 같은 결론이다.

→ 다음: **실제 문장으로 캘리브레이션**해서 다시 양자화하고 같은 파이프라인을 태운다. 컴파일러 변경
없이 1 단계(양자화)만 바뀐다.

## 3. 재현

```bash
S=<scratch>; O=_local/int8_debug/out; T=_local/int8_debug/tools
# 배치 1 고정 후 모양 상수 접기 (bert_b1.onnx = kpadq 에 batch=1 + infer_shapes)
python3 $T/fold_shape_consts.py $S/bert_b1.onnx $S/bert_s.onnx
python3 $T/fold_bias_into_k_general.py $S/bert_s.onnx $S/bertf.onnx --pad-value auto --max-bias-lsb 64
python3 $T/head_split.py $S/bertf.onnx $S/bertfhs.onnx --k
python3 $T/hoist_kfold_pad.py $S/bertfhs.onnx $O/bertx_int8.onnx
./scripts/docker/run-dev.sh /workspace/_local/padbe/build_final.sh bertx_int8     # 함수 main_graph
./scripts/docker/run-dev.sh /workspace/_local/padbe/run_bert.sh                   # 무작위 토큰 ids_{0,1,2}
./scripts/docker/run-dev.sh /workspace/_local/padbe/run_sent.sh                   # 실제 문장 sent_{0,1,2}
```
토크나이저는 `transformers` 를 임시 폴더에 설치해 `bert-base-uncased` 로 만들었다(호스트에 없음).
