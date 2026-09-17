# BERT 전체 NPU 실행 목표 및 로드맵 (2026-09-01)

## 0. 목표

팀 목표: **BERT 모델 적용**.

전략: 지금까지는 "matmul/conv만 NPU, 나머지(softmax/layernorm/gelu/embedding 등)는 무조건 CPU"
구조였다 (`AMDAIEAssignDeviceAffinities.cpp`의 `executableIsContractionOrConv`가 contraction/conv
연산만 NPU로 분류하고 나머지는 전부 host로 보내는 방식). 이제 방향을 바꾼다:

1. **1단계: BERT 인코더 전체를 NPU에서 돌아가게 만든다.** softmax/layernorm/gelu 등 지금까지
   host로 넘기던 non-matmul 연산도 전부 NPU ukernel/커널로 구현해서, 이론상 host round-trip이
   전혀 없는 상태를 먼저 만든다.
2. **2단계: 전부 NPU에서 돌아가는 상태를 기준선으로 두고, 연산을 하나씩 다시 CPU로 빼보면서
   측정한다.** 특정 연산을 CPU로 보내는 게 실제로 이득인지(예: 작은 elementwise 연산은 NPU
   dispatch 오버헤드보다 host 실행이 빠를 수 있음) 검증한 뒤에만 그 연산을 CPU로 되돌린다.
   즉 "기본은 host, 예외적으로 NPU"가 아니라 "기본은 NPU, 측정 근거가 있을 때만 host"로 뒤집는다.

이 방향이 나온 배경: 2026-08-31 측정에서 int8 vectorization이 실측 속도 이득이 없었는데, 원인이
"전체 wall time의 약 99%가 dispatch당 driver overhead이고 실제 compute 비중은 매우 작다"는
것이었다. 즉 지금 구조에서 성능을 좌우하는 건 개별 연산의 compute 속도가 아니라 **host↔NPU
dispatch/round-trip 횟수**이므로, non-matmul 연산들을 host로 흩어놓는 현재 구조 자체가 성능
병목의 핵심 용의자다. "전부 NPU에 올려서 dispatch를 하나로 뭉치고, 그 다음에 진짜 이득이 있는
경우만 다시 뺀다"는 순서가 이 가설과 직접 맞아떨어진다.

## 1. 현재 상태 (2026-09-01 기준, 이미 검증됨)

- BERT-tiny / BERT-base가 현재 구조(matmul만 NPU, 나머지 host)로 **e2e 동작 확인됨**:
  corr 0.99998 (bert-tiny), 0.99998 (bert-base, row-overflow 수정 후).
  (`docs/2026-08-19_bert_tiny_e2e.md`, `models/bert_tiny/README.md`, `models/bert_base/README.md`)
- batch_matmul M/K 패딩 안 되던 row-overflow 버그 root-cause 수정 완료, `bert` 브랜치에 커밋
  (미푸시, 3개 커밋).
- 이 기존 e2e 테스트는 `attention_mask`가 항상 all-ones (실제 padding 없는 입력)이라, **진짜
  padding이 있는 가변 길이 입력의 masking 수치 경로는 아직 한 번도 검증되지 않았다.**

## 2. 왜 이게 어려운가 — 알려진 하드웨어/컴파일러 레벨 블로커

이 계획 전체에 걸쳐 있는 공통 제약들. 개별 단계 작업 전에 팀 전체가 인지하고 있어야 함.

- **코어 타일당 DMA 채널이 2개뿐이다.** 3개의 독립적인 입력 스트림이 필요한 연산(예:
  matmul+bias = X, W, bias 3-input)은 arbiter/msel 리소스가 소진되어 hang을 일으킨다.
  matmul+bias 융합(로드맵 item 2, 이 문서 3.2)과 conv+padding(로드맵 item 1, 다른 팀원 담당)이
  **동일한 하드웨어 제약**에 걸려 있다는 게 2026-08-31에 확인됨 — 별개 버그가 아니라 하나의 근본
  원인.
- **큰 N(N≥3584)의 vectorized int8 matmul은 Peano `G_FMINIMUM` legalization crash로 컴파일이
  안 된다.** BERT-base MLM 디코더 매트멀(N=30528)이 여기 걸림. attention 내부 matmul들(N은
  seq_len/head_dim 수준)은 이 임계값보다 훨씬 작아서 이 크래시는 안 만날 가능성이 높지만
  아직 실측 확인 필요.
- **dispatch-type 전환(batched↔non-batched 등) 시 비결정성 버그가 L≥6 조건에서 재현된다.**
  소스 레벨에서 손댈 수 있는 곳(lock, BD, queue-drain, wait-fold, TCT-sync)을 전부 확인했지만
  미해결 — HW/펌웨어 레벨일 가능성이 높다. 12-layer 인코더를 전부 이어붙이면 dispatch 체인
  길이가 자연스럽게 이 조건을 넘으므로, attention 하나만으론 안 보이다가 풀스택에서 재현될
  위험이 있다.
- **새로운 dispatch 형태 조합은 매번 처음 보는 컴파일러/런타임 버그를 만들어왔다** (batch matmul
  두 개 연쇄 → 크래시, row-overflow, 위 두 버그 등). 이 계획의 각 단계는 전부 "지금까지 시도된
  적 없는 조합"이므로, 기능 구현 자체보다 여기서 나올 새 버그의 bisection 비용을 더 크게
  예산으로 잡아야 한다.

## 3. 단계별 작업 목록

### 3.0 개별 연산 단위 (선행 작업)

1. **non-batch matmul + bias 융합** (Q/K/V/O/FFN projection에 공통 필요)
   - 위 "DMA 채널 2개" 제약에 정면으로 걸림. 현재 진행 중인 접근: IRON 레벨에서 broadcast
     패턴 repro (2026-09-01 기준 1열/8열×4행 규모까지 PASS 확인, 실제 IREE 생성 코드와의
     간극은 미해결).
   - **대안으로 검토할 만한 것**: bias를 별도 입력 스트림으로 두지 말고, X에 1로 채운 열을
     추가하고 W에 bias를 행으로 추가해서 `[X|1] @ [W;b]` 형태로 만들면 구조적으로 2-input
     matmul이 되어 애초에 3-input 문제 자체를 안 만난다. HW hang 원인 규명과 별도로 훨씬 싼
     우회로라 병렬로 시도해볼 가치가 있음.
2. **batch matmul 양자화 (int8)** — QK^T / Attn@V용
   - attention 스케일에서 G_FMINIMUM 크래시(N≥3584) 임계값에 안 걸리는지 먼저 실측 확인.
   - 12-layer 스택 전체에서 dispatch-type 전환 비결정성(L≥6) 재현 여부 확인 필요.
   - **순서 제안**: fusion(dispatch 개수 감소)을 먼저 하고 나서 벡터화 이득을 재측정하는 게
     낫다. 지금처럼 dispatch overhead가 지배적인 상태에서는 벡터화 효과 자체가 측정 잡음에
     묻힐 가능성이 높다.
3. **Softmax ukernel (Peano 포팅)** — LLM으로 생성 가능함을 확인, 진행 예정.
   기존 Chess 전용 구현(`Target/uKernels/npu4/chess/softmax.cc`)이 참고 알고리즘으로 존재.
4. **LayerNorm ukernel (Peano)** — 신규. Chess 쪽에 참고 구현이 있는지 먼저 확인 필요
   (softmax와 달리 아직 확인 안 됨).
5. **GELU** — 필요 여부 결정 필요. 연산 자체는 가볍지만, "1단계 = 전부 NPU"라는 이번 목표
   기준으로는 host에 남겨두는 순간 그 지점에서 dispatch가 끊기므로, NPU ukernel로 만들거나
   matmul 뒤에 붙는 다항식 근사로 처리하는 방안을 검토.
6. **scale(1/√d) + attention mask를 QK^T~softmax 사이에 융합**
   - 안 하면 이 지점에서 host round-trip이 또 하나 생김.
   - **실제 padding 있는 attention mask 수치 경로는 지금까지 한 번도 검증 안 됨** (기존
     export가 항상 all-ones 사용) — 가변 길이 입력을 지원하려면 별도 정합성 검증 필요.
7. **Residual add** — LayerNorm과 fuse할지(고전적인 add+norm 패턴) 별도 dispatch로 둘지 결정.

### 3.1 Attention 블럭 통합

projection(3.0-1) → reshape/transpose → QK^T(3.0-2) → scale+mask(3.0-6) → softmax(3.0-3) →
Attn@V(3.0-2) → transpose → output projection(3.0-1)을 최소 개수의 dispatch로 묶는 단계.

기능적으로는 이미 현재(비융합) 상태로 e2e 정합성이 확인돼 있어 "동작 여부" 리스크는 낮다.
리스크는 이 특정 dispatch 조합을 컴파일러가 처음 겪는다는 점 — §2의 "새 조합마다 새 버그"
패턴대로 새 버그 하나 나올 것을 전제로 일정을 잡을 것.

### 3.2 인코더 블럭 (1 layer e2e)

- 3.0 + 3.1 결과물에 LayerNorm(3.0-4), residual(3.0-7), FFN(matmul+bias ×2 + GELU)을 붙여서
  인코더 1개 레이어를 전부 NPU 위에서 검증.
- FFN도 3.0-1의 matmul+bias 융합을 그대로 재사용.

### 3.3 12-layer 풀스택 재검증

- 한 레이어가 fused 상태로 통과해도 12개를 이으면 스케일에서만 나타나는 버그가 나올 수 있음
  (§2의 L≥6 비결정성 등). 별도 검증 단계로 분리.
- 이 시점에 **실제 dispatch 개수와 latency를 측정**해서 "전부 NPU"가 실제로 얼마나 이득인지
  숫자로 확인. (지금까지는 "컴파일되고 corr 맞으면 성공"으로 판단해왔는데, 이번 목표는
  성능이 핵심이라 정합성만으로는 완료로 볼 수 없음.)

### 3.4 검증 후 선택적 CPU 오프로드 (2단계 목표)

3.3에서 전부-NPU 기준선이 확보되면, 연산 단위로 하나씩 CPU로 옮겨보면서 latency를 재측정.
이득이 실측으로 확인된 연산만 CPU로 남기고, 나머지는 NPU에 유지. (예: embedding gather는
NPU에 올릴 이유가 약해 보이지만, 이것도 가정하지 말고 측정해서 결정.)

### 3.5 Task head (선택, 스코프 확인 필요)

- MLM 헤드의 N=30528 디코더 matmul은 현재 3가지 별개 버그(bf16 timeout, bf16 비결정성,
  int8 G_FMINIMUM crash)로 NPU에서 못 돌아가 host-numpy로 우회 중.
- 최종 목표 태스크가 MLM인지 classification 등 다른 head인지에 따라 이 항목이 스코프에
  들어가는지 결정 필요 — classification이면 훨씬 작은 pooler+linear라 이 문제를 안 만날
  가능성이 높음.

## 4. 담당 매핑 (현재 논의 기준)

- batch matmul 양자화, non-batch matmul+bias 융합, attention 구조 통합 — 본인(3-input 케이스,
  int8 양자화 쪽 담당).
- Softmax / LayerNorm ukernel(Peano) — 팀원 담당, LLM으로 생성.
- conv+padding의 동일 3-input 문제(§2) — 다른 팀원(로드맵 item 1), 같은 근본 원인 공유하니
  진행 상황 공유 필요.

## 5. 참고 문서

- `models/bert_tiny/README.md`, `models/bert_base/README.md` — 현재 e2e 상태
- `docs/2026-08-19_bert_tiny_e2e.md` — bert-tiny 파이프라인 단계별 walkthrough
- `docs/2026-08-20_batch_matmul_row_overflow_fix.md` — row-overflow 버그 root cause/fix
- `docs/2026-08-24_int8_quantization_investigation.md` — int8 양자화, G_FMINIMUM 버그,
  dispatch overhead 측정 결과
- `docs/2026-08-26_matmul_bias_fusion_runtime_hang.md`,
  `docs/2026-08-27_matmul_bias_fusion_hang_root_cause_refined.md`,
  `docs/2026-08-28_shared_channel_fanout_column_threshold.md` — matmul+bias 3-input hang
  조사 전체 기록
