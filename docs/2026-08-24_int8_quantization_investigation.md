# int8 양자화 조사 (일시 중단, 미완료) — 한국어 정리 (2026-08-26 재작성)

목표: npu4(AIE2P/Strix)에서 `vector.contract`를 실제로 vectorize하는 것. aievec의
npu4 lowering(`getSuportedAie2PTypes()`,
`compiler/plugins/target/AMD-AIE/aievec/VectorToAIEVecConversions.cpp:118-126`)은
int8×int8→i32 matmul intrinsic만 있고 bf16은 없음 (npu1/AIE2/Phoenix엔 둘 다
있음 — `getSupportedAie2Types()` 107-115줄. npu4 백엔드에 국한된 갭이지 aievec
자체의 일반적인 한계는 아님). 그래서 bf16 vectorization을 쫓는 대신 모델을 int8로
양자화하는 쪽으로 방향을 잡음.

`bert` 브랜치, BERT e2e 작업(`docs/2026-08-20_batch_matmul_row_overflow_fix.md`,
BERT 모델 README들) 이후에 이어지는 작업. 아래는 전부 한 세션 안에 일어난 일이고,
**대부분 커밋되지 않은 상태로 세션이 종료**됨 (파일별 변경사항은 맨 아래 표 참고).

## 1. int8 vectorization 자체는 되는가? — 예, 실제 하드웨어에서 확인됨

`[32,128]x[128,64]` 단일 matmul을 `onnxruntime.quantization.quantize_static`
(`quantize_dynamic`이 아님 — 설치된 onnxruntime 1.29.0엔 `quant_format` 파라미터가
없고 항상 QOperator 포맷을 냄, `quantize_static`에 `quant_format=QuantFormat.QDQ`를
줘야 진짜 QDQ 노드가 나옴)로 양자화해서 `iree-import-onnx` → 컴파일 → 실행까지
전부 통과. IR 덤프로 `aievec.matmul %_, %_, %_ : vector<8x8xi8>, vector<8x8xi8>
into vector<8x8xi32>`가 20회 나오는 것 확인 (스칼라 fallback 아님). 실제 npu4에서
`iree-run-module` exit 0, fp32 레퍼런스 대비 corr 0.99985 (calibration 노력 거의
없이). **"vectorization이 되긴 하는가"라는 질문은 여기서 확정적으로 해결됨** —
2D matmul 한정, 이 이후 재확인할 필요 없음.

## 2. VGG16 Conv int8 — 업스트림 IREE 버그로 막힘 (별도 큰 작업, 미착수)

VGG16의 13개 Conv 레이어를 양자화하면 `iree-compile`이 크래시함:
`ConvertConvToChannelsLast.cpp:358-359`가 conv의 DPS "output"을 operand index
2로 하드코딩하는데, 양자화된 conv(`linalg::Conv2DNchwFchwQOp`)는 operand가
5개(`input=0,filter=1,inputZp=2,weightZp=3,output=4`)라 실제로는 zero-point
스칼라를 잘못 집어감. channels-last 변환은 AIE conv codegen에 필수라 우회 불가.
진짜 고치려면 vendored `third_party/iree`에 patch(DPS-aware operand 접근 +
양자화된 conv용 channels-last 패턴)가 필요 — 여러 시간짜리 작업, 미착수. VGG16을
Gemm/FC만 양자화하는 건 (2D matmul 케이스랑 같은 메커니즘이라) 될 텐데, BERT로
방향 전환하면서 안 함.

## 3. BERT batch matmul(attention) 양자화 — torch-mlir 패치, 커밋 완료

원인은 IREE가 아니라 torch-mlir이었음: ONNX의 배치 `MatMul`은
`torch.aten.bmm`으로 import되는데(`aten.matmul`이 아님), `torch.aten.bmm`엔
양자화 처리가 아예 없었음. `FuseQuantizedOps.cpp`와 `ConvertAtenBmmOp`에 패치
2개 추가 (input 쪽 `QuantizeOperandsPastCommutingOps<AtenBmmOp,2>`, output 쪽
`QuantizeAccumulator<AtenBmmOp>` — 후자가 없으면 raw int32 accumulator를
`sitofp`로 그냥 캐스팅해버려서 스케일 곱셈 없이 값이 포화됨, corr 0.9998로
검증 완료).

**2026-08-25 업데이트: 이 패치는 이제 로컬 커밋 3단으로 완전히 커밋됨** (git
log/status로 확인, 2026-08-25 아침): torch-mlir 브랜치 `wjjang/bmm-int8-quantization`
커밋 `6ce189ce` → `third_party/iree` 커밋 `9c8b2b7` → 메인 저장소 `bert` 브랜치
커밋 `40654aa`. 전부 로컬에만 있고 어디에도 push는 안 됨 (torch-mlir 서브모듈은
팀 포크가 아니라 순정 upstream을 가리키고 있어서 push할 곳이 없음; `third_party/iree`는
`ace-knu/iree` 포크를 가리켜서 push는 가능한데 아직 안 함). **더 이상 "uncommitted라서
날아갈 위험" 상태는 아님.**

## 4. BERT-tiny 전체 파이프라인 — 성공

BERT-tiny의 16개 `MatMul` 노드(Q/K/V/output/FFN + 배치 attention matmul 전부)를
int8 QDQ로 양자화, vectorization ON으로 컴파일 (기존 bf16 BERT-tiny 레시피는
vectorization off였음), 10,000회 delay fix까지 적용. 실제 npu4에서 **20/20회
전부 성공, corr=0.99678, 완전히 결정론적** (매번 동일한 결과). 2레이어 트랜스포머
전체에 delay fix가 일반화됨을 처음 확인.

## 5. 진짜 원인 규명: batch-0 lock-precharge race — SOLVED, 작동하는 workaround 있음

npu4 + vectorization ON에서 배치 matmul을 돌리면 **batch index 0이 항상 all-zero**로
나오는 문제 발견 (batch 1은 정상). MLIR IR부터 디스어셈블된 기계어까지 전부 훑어서
원인을 찾음 (아래는 요약, 전체 실험 과정은 git 히스토리에 더 상세히 있었음):

- **정적 분석으로는 원인을 못 찾음**: 벡터화 전후 control-code, lock/BD 선언,
  DMA 큐 드레인 전부 동일함을 확인. bypass 테스트(matmul 결과를 상수로 치환)로
  batch 0의 write-back 경로 자체가 실패한다는 것까지는 확인.
- **매크로 스케일 delay 실험이 돌파구**: core 프로그램 시작부에 진짜 busy-wait
  루프(cf.br/cf.cond_br 기반)를 넣어보니 — 0회: 항상 실패, 50,000회: 비결정론적,
  500,000회: batch 0의 앞부분 행은 고쳐지고 뒷부분 행은 여전히 실패 (M=16이
  4개 물리 코어에 타일링되는데, 그중 일부 코어에만 효과가 있었음을 시사).
- **진짜 원인 확정**: delay를 core 시작부가 아니라 **매 lock의 첫 release
  직후**로 옮기니 완전히 해결됨 (batch=2, batch=4 전부 corr 0.9998+, 4회
  재현). **AIE lock은 미리 충전된(pre-charged) 카운팅 세마포어라서, 특정
  lock의 첫 acquire-release 사이클만 실제 하드웨어 확인 게이트를 건너뛴다**는
  게 메커니즘 — 그래서 항상 정확히 "첫 번째" 반복(batch 0)만 레이스가 남.
- **비용 최적화**: 500,000회에서 10,000회로 50배 줄임 (batch=4로 stress-test,
  20/20 성공, batch=2도 10/10 성공). Peano/llc가 이 delay 루프를 AIE 하드웨어
  전용 zero-overhead loop(`lc`/`ls`/`le` 레지스터)로 컴파일해줘서 코드 크기
  비용은 거의 없고(~150-170바이트), 배치 수와 무관하게 코어당 딱 한 번만
  드는 고정 비용.

**결론: 이건 workaround지 진짜 fix는 아님** — lock의 초기 credit 값 설계
자체를 고치는 게 진짜 fix인데, 그건 안 찾음. `AMDAIECoreToStandard.cpp`의
`lockToStd`에 각 lock의 첫 release 직후에만 delay를 넣는 코드로 구현.

## 6. 두 번째, 더 심각한 버그: BERT-base(12레이어)에서 발견, 미해결

BERT-tiny(2레이어)와 같은 레시피를 BERT-base(12레이어)에 그대로 적용하면
corr 0.70~0.90으로 비결정론적. delay를 20k→500k→5M(10배씩 두 번)로 올려도
**전혀 개선 안 됨** — 원래 버그는 delay에 단조롭게 반응했는데, 이건 완전히
평평한(flat) 반응. 다른 메커니즘의 버그.

- 순수 batch matmul을 아무리 크게(레이어 수, 배치 수) 합성해서 반복해도
  재현 안 됨 — **실제 BERT 아키텍처(LayerNorm/Softmax/GELU가 섞인)**를 잘라서
  써야 재현됨: L=1,2는 결정론적, L=4는 거의 결정론적, **L=6, L=8은 정확히 두
  개의 값으로 갈라지는 패턴** (넓게 퍼지는 게 아니라 이산적인 두 값). L=12는
  훨씬 넓게 퍼짐 (0.70~0.90).
- **최소 재현 발견**: non-batched 프로젝션 matmul + Reshape/Transpose + batched
  attention matmul을 번갈아가며 배치하면(같은 물리 타일을 재사용하면서) 비결정론적.
  같은 타입의 dispatch만 반복하면(circuit/circuit 또는 batched/batched) 결정론적.
  **"두 개의 서로 다른 dispatch 타입이 같은 물리 타일을 공유하며 번갈아 실행되는 것"**이
  트리거라는 게 확정됨 — batch 크기, 레이어 수, CPU-op 개입 자체는 원인이 아님.
- **모든 소스 레벨 감사가 클린하게 나옴**: lock 값이 dispatch마다 재초기화되는지
  (재초기화됨, 확인), stale BD 레지스터 내용(BD write는 항상 전체 memset+재작성,
  read-modify-write 아님), dispatch 간 DMA 큐 드레인 레이스(기본 설정에서는
  호스트가 완전히 동기화해서 다음 dispatch 전에 이전 게 끝났음을 보장함),
  `AMDAIEFoldDmaWaits.cpp`가 wait를 비대칭적으로 누락하는지(둘 다 20개 wait를
  1개로 정확히 fold함, coverage gap 없음), `NpuDmaWaitOp`의 lowering이
  dispatch 타입에 따라 다른지 (실제 IR 덤프로 확인, 완전히 동일함).
- **호스트 사이드 settling delay 실험도 전부 음성**: sleep 기반, spin 기반
  (5천~50만 마이크로초) 전부 시도했는데, interleaved A/B 테스트로 제대로
  통제해보니 전부 통계적으로 유의미한 개선 없음 (초기에 "괜찮아 보였던" 신호는
  전부 노이즈로 밝혀짐 — 이 머신 자체의 실패율이 시간에 따라 흔들리는 confound가
  있어서, block sampling이 아니라 반드시 interleaved A/B로 테스트해야 함).

**남은, 소스에서 확인 불가능한 유일한 지점**: 실제 하드웨어/펌웨어의 TCT
completion-token 메커니즘이 `repeat_count>1`일 때 반복당 하나씩 토큰을 정확히
발행하는지 (아니면 첫/마지막 반복에만 발행하는지) — 이 코드베이스엔 문서화가
안 돼있고, AMD 드라이버/펌웨어 문서나 실제 하드웨어 트레이스 없이는 확인 불가.

**결론: 소스/컴파일러/런타임에서 감사 가능한 모든 레이어(lock 값, BD 내용, DMA
큐 드레인, wait-folding, TCT-sync lowering)가 다 정상으로 확인됨. 이건 진짜
하드웨어/펌웨어 레벨의 현상일 가능성이 높고, 이 저장소의 컴파일러/런타임 코드
버그가 아닐 가능성이 높음.**

> 이 두 번째 버그는 2026-08-26에 다시 마주쳤어요 — roadmap item 2(matmul+bias
> 퓨전) 조사에서 완전히 다른 경로로 정확히 같은 결론에 도달했습니다: 독립적인
> mlir-aie 툴체인으로 **packet flow + 하드웨어 `repeat_count`** 조합을 재현했더니
> 똑같이 hang이 났어요. 자세한 내용은
> [docs/2026-08-26_matmul_bias_fusion_runtime_hang.md](2026-08-26_matmul_bias_fusion_runtime_hang.md)
> 참고 — 이번 버그(값이 틀림, dispatch는 완료됨)와 그쪽 버그(dispatch 자체가
> 영원히 안 끝남)는 증상은 다르지만, 둘 다 "TCT/repeat_count 하드웨어 메커니즘이
> 미묘하게 안 맞는다"는 같은 계열의 문제로 보입니다.

## 7. bf16 vectorization 작업 전체 결론 (2026-08-25, 팀 결정)

이 두 번째 버그가 **dtype과 무관**하다는 게 확정적이라(순수 batch matmul
레벨에서는 재현 안 되고 실제 트랜스포머 구조에서만 재현되는 걸 보면, int8이든
bf16이든 vectorized backend라면 똑같이 이 벽에 부딪힐 것), bf16 vectorization
backend를 새로 만들어도 더 적은 비용으로 우회할 방법이 없다고 판단. **팀 결정:
vectorization 작업 자체를 이 시점에서 중단.**

## 파일별 변경사항

| 파일 | 변경 내용 | 상태 |
|---|---|---|
| `compiler/plugins/target/AMD-AIE/aie/AMDAIECoreToStandard.cpp` | §5의 batch-0 lock-precharge race 픽스: 각 lock의 첫 release 직후에 10,000회 busy-wait 삽입 (`lockToStd`) | **커밋됨** — `bert` 브랜치 `1600078` (roadmap item 2 세션에서 2026-08-26에 커밋) |
| torch-mlir (`third_party/iree/third_party/torch-mlir`) `FuseQuantizedOps.cpp`, `ConvertAtenBmmOp` | §3의 `aten.bmm` int8 양자화 지원 (input-side + accumulator-side 패턴 2개) | **커밋됨** (로컬만, push 안 됨) — torch-mlir 브랜치 `wjjang/bmm-int8-quantization` 커밋 `6ce189ce` → `third_party/iree` 커밋 `9c8b2b7` → 메인 저장소 `bert` 브랜치 `40654aa` |
| `runtime/src/iree-amd-aie/driver/amdxdna/direct_command_buffer.cc` | §6 두 번째 버그 조사 중 시도한 호스트 사이드 settling delay 실험 | **되돌림, 커밋 안 됨** — 효과 없음으로 결론, 현재 원본 상태 |
| `_local/int8_debug/` 하위 여러 스크립트/mlir/onnx 파일 | 위 모든 조사에 쓰인 최소 재현 스크립트들 (`gen_bmm_bias.py`, `gen_chained_mixed.py`, `quant_bert_tiny.py`, `export_bert_base_ntrunc.py` 등) | gitignore된 스크래치 디렉토리, 저장소에 커밋 안 됨 (의도적) |

이 문서 자체가 다루는 §1~7 범위에서 **컴파일러/런타임 소스 변경으로 지금 살아있는
건 `AMDAIECoreToStandard.cpp`의 delay fix 하나뿐**이고, 이건 이미 커밋됐습니다.
나머지는 전부 (a) 이미 커밋된 torch-mlir 패치이거나, (b) 검증 후 되돌린 실험이거나,
(c) 애초에 커밋 대상이 아닌 스크래치 스크립트입니다.

## 다음에 이어서 할 것 (우선순위 순)

1. §5(batch-0 레이스)는 메커니즘까지 규명됐고 workaround도 있음 — 남은 건 "진짜
   fix"로 승격시키는 것. lock의 initial credit 값이 어디서/어떻게 결정되는지
   (`2,0,2,0,2,0` 패턴이 뭘 인코딩하는지), `runtime/src/iree-amd-aie`의 HAL
   드라이버/커맨드버퍼 dispatch 순서 코드(이번 세션엔 전혀 안 봄)를 봐야 함.
2. §6(두 번째 버그)은 소스 레벨 감사가 다 끝났음 — 더 파려면 실제 하드웨어
   트레이스 도구가 필요함. 이건
   [docs/2026-08-26_matmul_bias_fusion_runtime_hang.md](2026-08-26_matmul_bias_fusion_runtime_hang.md)에서
   더 진행됐으니 거기 참고.
3. §5 fix가 진짜 fix로 바뀌면: torch-mlir 패치를 더 안전한 곳에 (팀 포크?) 올리는
   것도 고려 — 지금은 순정 upstream repo를 가리키는 서브모듈이라 push할 곳이 없음.
4. VGG16 Conv int8(§2)은 별개의, 더 큰 upstream IREE 작업으로 남아있음 — BERT
   작업 우선순위에 밀려서 의도적으로 보류 중.

## 8. 실행 시간 측정 (2026-08-31, 뒤늦게 채움) — §1의 "int8 벡터화 확인"은 정확성만 봤지 속도는 한 번도 잰 적이 없었음

§1에서 "int8 벡터화가 되는가"를 확정할 때, 확인한 건 IR에 `aievec.matmul`이 실제로
나오는지와 정확도(corr 0.99985)뿐이었고, **실행 시간은 이 조사 전체에서 단 한 번도
측정되지 않았음** (`_local/int8_debug/*.py` 전체에 `time.perf_counter`,
`iree-benchmark-module` 호출이 전혀 없음 — grep으로 확인). 뒤늦게 채움.

**측정 대상**: §1과 동일한 2D matmul(M=32, K=128, N=64, `_local/int8_debug/out/mm_int8.mlir`,
QDQ int8 양자화, 이미 존재하던 아티팩트 재사용) vs 같은 shape의 bf16 스칼라 버전
(`_local/int8_debug/perf_check/mm_baseline.mlir`, 새로 생성 — bf16은 npu4에 벡터화
intrinsic이 없어서(문서 맨 위 참고) 항상 스칼라로만 컴파일됨, 이게 이 백엔드에서
지금 실제로 쓸 수 있는 "베이스라인").

**방법**: `iree-benchmark-module --device=amdxdna --device=local-task
--benchmark_repetitions=5 --benchmark_min_time=2s` (실제 npu4 하드웨어, 5회 반복 ×
각 2초 이상). 두 vmfb 모두 correctness 먼저 재확인(둘 다 참조값과 일치).

| | 평균(mean) | 중앙값(median) | 표준편차 | CV |
|---|---|---|---|---|
| **int8, 벡터화 O** (`mm_int8.vmfb`) | 117 ms | 120 ms | 16.8 ms | 14.4% |
| **bf16, 스칼라** (`mm_baseline_bf16.vmfb`) | 102 ms | 97.5 ms | 11.2 ms | 11.0% |

**결론: 이 크기(M=32,K=128,N=64)에서는 측정 가능한 속도 개선이 없음.** 두 그룹의
차이(~15ms)가 각 그룹 자체의 표준편차(11~17ms)보다 작아서 노이즈 안에 있음 —
오히려 표면적으로는 int8(벡터화)이 근소하게 더 느리게 나왔는데, 이것도 유의미한
회귀가 아니라 노이즈로 보는 게 맞음.

**왜 이런 결과가 나오는지(중요한 발견)**: 두 경우 다 `iree-benchmark-module`이
보고하는 host CPU 시간은 1.3~1.7ms인데 벽시계 시간(real_time)은 90~140ms —
즉 **측정 시간의 99% 이상이 host가 NPU 드라이버 응답을 기다리는 시간**이지 실제
행렬곱 연산 시간이 아님. 이 정도로 작은 matmul(32×128×64)의 실제 코어 연산은
마이크로초 단위일 텐데, 매 호출마다 붙는 고정 오버헤드(커맨드 버퍼 제출, PDI/컨텍스트
재구성, xrt 큐 대기 등으로 추정 — 정확히 어느 구성요소가 지배적인지는 미분리)가
100ms 안팎으로 이 전부를 덮어버림. **즉 이 정도 규모의 단일 dispatch에서는 int8
벡터화가 실제로 빠른지 여부 자체가, 벤치마킹 방법을 더 정교하게 만들지 않는 한
안 보임.**

**더 정밀한 측정을 시도했으나 막힘**: 이 고정 오버헤드를 걷어내고 코어 자체의
반복 실행 시간만 재는 전용 메커니즘이 이미 이 저장소에 있음
(`--iree-amdaie-enable-infinite-loop-around-core-block=true` 컴파일 플래그 +
`--amdxdna_n_kernel_runs`/`--batch_size` 런타임 플래그, `AMDAIEInsertInfiniteLoopAroundCoreBlock`
패스). 두 모델 다 이 플래그를 켜고 재컴파일 시도했으나 **둘 다 서로 다른, 이 벤치마킹
작업과 무관한 기존 버그에 막힘**:
- int8(양자화, 클램프 있는 버전): Peano(`llc`)가 `G_FMINIMUM`(스칼라 f32 min/max,
  requantize 클램프에서 씀)을 legalize 못 하고 크래시 (`unable to legalize
  instruction ... G_FMINIMUM ... in function: core_7_5`). 참고로 **루프 래핑 없이
  컴파일할 땐 이 크래시가 안 남** — 벡터화 경로는 원래도 이 스칼라 클램프를 우회해서
  괜찮았는데, loop-wrap 패스가 뭔가 다른 코드 경로를 타게 만들면서 노출된 것으로
  보임. (이 크래시는 사실 §1과 별개로 순수 스칼라 int8 경로
  (`--iree-amdaie-enable-vectorization-passes=false`)에서도 loop-wrap 없이 그냥
  단독으로 재현됨 — 즉 이 requantize 클램프 코드는 애초에 스칼라 경로 자체가
  깨져 있었던 것으로 보이고, 지금까지 항상 벡터화 경로만 썼기 때문에 안 걸렸던 것.)
- bf16 스칼라: `_XAie_LoadProgMemSection(): Overflow of program memory` — 코어
  프로그램 메모리 초과. loop-wrap이 코드를 줄이는 게 아니라 오히려 (아마 무한 루프
  구조 자체가 요구하는 추가 제어 흐름 때문에) 이미 큰 스칼라 K=128 언롤 코드를 더
  키운 것으로 보임.

**둘 다 이 세션에서 고칠 만한 사이즈가 아니라서(각각 별도의 Peano/코드젠 조사가
필요) 여기서 멈춤** — "코어 연산만 순수 격리한 시간"은 여전히 미확보 상태.

**요약**: 이 int8 양자화 작업 전체의 원래 동기("npu4에서 벡터화가 되긴 하는가")는
§1에서 확정됐지만, **"그래서 실제로 빨라지는가"는 이번에 처음 측정했고, 답은
"이 문제 크기에서는 측정 가능한 차이가 없다"**였음. 벡터화의 실제 이득을 보려면
(a) 이 dispatch-당 고정 오버헤드보다 계산량이 훨씬 큰 문제(더 큰 M/K/N, 또는
여러 레이어를 하나로 묶은 체인)로 다시 재거나, (b) 위에서 막힌 loop-wrap 경로의
두 버그를 각각 고쳐서 코어 전용 시간을 분리 측정해야 함 — 둘 다 미착수.

**재현 자료**: `_local/int8_debug/perf_check/`에 저장 (gitignore됨) —
`mm_baseline.onnx`/`.mlir` (bf16 베이스라인 소스, `mm_int8.mlir`과 동일 shape·동일
시드), `mm_baseline_bf16.vmfb` (컴파일된 베이스라인), `x.npy`/`ref.npy` (입력/참조값,
`_local/int8_debug/out/x.npy`와 값 동일 — 같은 seed=0 재생성). 벤치마크 명령은 위
"방법" 문단 그대로.

## 9. 실행 시간, BERT-tiny 전체 모델로 확장 (2026-08-31, 이어서) — 같은 패턴이 그대로 나옴

§8의 질문("BERT-tiny도 비슷한가?")에 답하기 위해, 이미 §4에서 만들어져 있던 완전한
BERT-tiny e2e vmfb 두 개를 그대로 재사용해서 같은 방식으로 실측:
`_local/int8_debug/out/bert_tiny_int8.vmfb` (§4, int8 QDQ + 벡터화, 10,000회 delay
fix 포함, corr 0.99678 검증됨) vs `_local/int8_debug/out/bert_tiny_bf16.vmfb`
(양자화 없는 원래 bf16 레시피, 벡터화 off). 입력은 `models/bert_tiny/input_ids.npy`
(`[1,32]` 토큰 ID), 둘 다 먼저 `iree-run-module`로 정확성 재확인(두 출력이 서로
거의 동일한 값으로 나옴 — 새 버그 없음).

`iree-benchmark-module` 동일 방식(5회 × 3초 이상, 실제 npu4):

| | 평균 | 표준편차 |
|---|---|---|
| **int8 (벡터화)** | 1615 ms | ±219 ms |
| **bf16 (스칼라)** | 1635 ms | ±197 ms |

**§8과 똑같은 결론: 차이가 노이즈 안에 있음 — 측정 가능한 속도차 없음.** 이번엔
host CPU 시간이 19~24ms, 벽시계 시간이 1.4~1.9초 — 여전히 98%+ 가 오버헤드/대기
시간. BERT-tiny(2레이어 트랜스포머)는 레이어당 여러 개 dispatch(Q/K/V 생성,
배치 attention matmul 2개, softmax, output projection, FFN up/down 등)로
쪼개지는데, 총 벽시계 시간(~1.6초)을 §8에서 측정한 "dispatch 1개당 고정 오버헤드
~100ms"로 나누면 대략 15~16개 dispatch에 해당 — 실제 BERT-tiny 구조(임베딩 +
2레이어 × 레이어당 ~7개 dispatch)와 대략 맞아떨어짐. **즉 여러 dispatch를 체인으로
묶어도 똑같은 메커니즘(dispatch 개수 × 고정 오버헤드)이 전체 시간을 지배하고,
int8 벡터화가 주는 실제 계산 이득은 이번에도 안 보임.**

**결론(§8+§9 종합)**: 이 백엔드에서 "int8 벡터화가 빠르다"는 주장은 **단일
2D matmul에서도, 실제 2레이어 BERT 전체에서도 아직 한 번도 실측으로 확인된 적이
없음** — 둘 다 dispatch당 고정 오버헤드에 완전히 가려짐. 이 오버헤드를 줄이거나
(dispatch 수를 줄이는 퓨전, 혹은 §8에서 시도했다가 막힌 loop-wrap 기반 코어 전용
시간 분리) 계산량 자체가 훨씬 큰 워크로드(BERT-base, 더 긴 seq_len)로 다시 재야
진짜 이득이 보일지 알 수 있음 — 둘 다 미착수.

## 10. BERT-base MLM decoder matmul(N=30528)을 실제로 int8+벡터화해서 확인해봄 (2026-08-31, 이어서) — 컴파일 자체가 새로운 버그로 막힘

`models/bert_base/README.md` §5 / `project-bert-e2e-plan` 메모리에 기록된, 예전에
NPU에서 못 돌려서 host numpy로 우회했던 그 matmul(`[32,768]×[768,30528]`,
BERT-base MLM head의 vocab-projection) — "벡터화하면 그때 문제(타임아웃, 이후
비결정성)가 해결됐을까?"를 직접 확인해보려 했음.

**결과: 확인 자체를 못 함 — 컴파일이 §8/§9와 무관한, 이전에 발견 못 했던 세 번째
Peano 버그에 막힘.** 같은 shape(M=32,K=768,N=30528)을 §8과 동일한 QDQ int8
레시피(가중치 상수 버전, `gen_and_quant.py` 그대로 — §9의 실제 decoder matmul과
가장 가까운 구조)로 양자화 + 벡터화 ON으로 컴파일 시도 → **§8에서 loop-wrap
플래그를 켰을 때 봤던 것과 정확히 같은 크래시**:
```
LLVM ERROR: unable to legalize instruction: %_(s32) = G_FMINIMUM %_, %_ (in function: core_7_5)
```
**중요한 재발견: 이번엔 loop-wrap 플래그 없이, 순정 벡터화 컴파일 자체에서
크래시남.** 즉 §8에서 "loop-wrap이 원인인 줄 알았던" 그 크래시가, 사실은
**loop-wrap과 무관하게 어떤 N 규모 이상에서는 벡터화 경로 자체가 이 requantize
clamp를 못 다루는 것**임이 이번에 드러남 — §8의 "loop-wrap 때문"이라는 설명은
정정 필요.

**이분탐색으로 경계 확인** (같은 K=768, 상수 가중치, 매번 새로 양자화+컴파일만
함 — 하드웨어 실행 없이 컴파일 성공/실패만 확인, 수 초 단위라 빠름):

| N | 결과 |
|---|---|
| 3072 (기존에 이미 검증된 FFN 스케일, `mm_large_int8.vmfb`) | **컴파일 성공** |
| 3584 | **크래시** (`G_FMINIMUM`, `core_7_5`) |
| 4096, 6144, 12288, 24576, 30528(진짜 decoder 크기) | **전부 크래시** (동일) |

**즉 N=3072와 3584 사이 어딘가에 실제 경계가 있고, 그 이상은 이 저장소가 시도한
모든 N에서 100% 크래시함.** BERT-base의 FFN(N=3072)은 마침 이 경계 바로 아래라
지금까지 "벡터화된 int8이 잘 된다"고 검증된 모든 사례가 우연히 이 버그를 피해간
것으로 보임 — decoder matmul(N=30528)은 이 경계를 10배 가까이 넘는 규모라 정면으로
걸림.

**결론**: BERT-base MLM decoder matmul을 벡터화 int8로 바꿔서 "그때의 타임아웃/
비결정성 버그가 없어지는지" 확인하는 건, 그 질문에 답하기도 전에 **컴파일 자체가
안 돼서 막힘**. 원래 bf16 스칼라 버전이 겪었던 문제(타임아웃, 그다음 비결정성)와는
완전히 다른, 세 번째의 별개 버그. 정리:
1. bf16 스칼라: 컴파일됨, 느려서 타임아웃 → 타임아웃 올리면 컴파일+실행은 되는데
   비결정성 버그 (§ models/bert_base/README.md §5, 원인 미규명).
2. int8 벡터화: **이 규모에서는 컴파일 자체가 안 됨** (이번에 새로 발견, 원인 미규명
   — Peano/LLVM의 GlobalISel 레지스터 할당이 이 규모의 requantize epilogue를 어느
   시점부터 못 legalize하는 것으로 보이나 더 파고들지 않음).

**How to apply**: "벡터화하면 다 해결된다"고 가정하지 말 것 — 오히려 이 규모의
큰 matmul은 스칼라든 벡터화든 각자 다른 이유로 막혀 있어서, decoder matmul을
NPU에서 돌리려면 **두 개의 서로 다른 버그**(bf16 비결정성 OR int8 G_FMINIMUM
legalization)를 각각 별도로 고쳐야 함. 당장은 host numpy 우회(`bert_mlm_trunk.vmfb`
방식)가 유일하게 검증된 정답이라는 기존 결론이 그대로 유지됨.

**재현 자료**: `_local/int8_debug/decoder_check/`(gitignore됨) — `bisect_<N>*.onnx/.mlir`
이분탐색에 쓴 각 N별 산출물, `mm_decoder_int8.mlir`/`mm_decoder_w_int8.mlir`
(진짜 decoder 크기 N=30528, 두 가지 양자화 스타일 — activation 2개짜리와 상수
가중치짜리 둘 다 동일하게 크래시함, 구조 문제 아니라 순수 크기 문제임을 확인).
로그 파일은 안 남겨뒀지만 컴파일 자체가 몇 초면 끝나서 위 "방법"대로 재컴파일하면
바로 재현됨.

## 11. 배경 지식: QDQ vs QOperator, dequantize가 왜/어디서 나오는가 (2026-09-01, 참고용)

지금까지 이 문서 전체가 `quant_format=QuantFormat.QDQ`(§1)를 전제로 쓰여있는데,
"QDQ가 뭐고 왜 그걸 골랐는지", "dequantize는 왜 필요한지"를 나중에 다시 볼 때
헷갈리지 않도록 개념을 정리해둠. 코드 조사 결과 기반, 실험은 아님.

**QDQ는 별도의 양자화 "방법"이 아니라 양자화 그래프의 표현 방식.** ONNX에는 두 가지
표현이 있음:
- **QOperator**: `QLinearMatMul`/`QLinearConv` 등 int8 전용 연산자로 그래프를 짬.
- **QDQ**: 원래 fp32 연산(`MatMul`, `Conv` 등)은 그대로 두고 앞뒤에
  `QuantizeLinear`/`DequantizeLinear` 노드 쌍을 끼워 넣음.

installed onnxruntime 1.29.0의 `quantize_static`은 `quant_format`을 안 주면
QOperator를 냄 — QDQ를 쓰려면 명시적으로 지정해야 함(§1에 이미 기록됨). 이 저장소가
QDQ를 고른 이유: torch-mlir의 `FuseQuantizedOps` → IREE의
`LinalgQuantizedMatmulToMatmul`(`third_party/iree/compiler/src/iree/compiler/GlobalOptimization/QuantizedMatmulToMatmul.cpp`)
패스 체인이 Q→matmul→DQ 패턴을 인식해서 순수 int8 matmul로 재구성하도록 설계되어
있고, 이 컴파일 경로가 실제로 검증/커밋된 경로이기 때문(QOperator용
`QLinearMatMul` 임포터도 torch-mlir에 존재하긴 함 —
`third_party/iree/third_party/torch-mlir/lib/Conversion/TorchOnnxToTorch/DefaultDomainQtoZ.cpp:602` —
but 이 저장소에서 실사용/검증된 쪽은 QDQ).

**dequantize가 나오는 이유는 세 가지, "안전장치"처럼 보이지만 실제로는 대부분
수학적으로 필수:**
1. **int32 누산값 rescale (근본 원인)**: int8×int8 matmul 결과는 int32
   누산기에 쌓이고, 이 값엔 `scale_x * scale_w`라는 합성 스케일이 곱해진 상태라
   실제 값과 다름. `real ≈ (int32_acc - zp보정) * (scale_x*scale_w)`로 되돌리는
   계산은 다음 레이어로 넘기기 전에 반드시 필요 — 안 하면 값 자체가 틀어짐.
2. **정밀도 민감 연산 경계**: softmax/layernorm/GELU처럼 int8로 안 도는(혹은 안
   도는 게 나은) 연산 앞에서는 float으로 복원해서 넘겨야 함([[project_softmax_ukernel_infra]]
   참고 — 이 저장소는 애초에 softmax를 NPU에 못 올리고 있어서 이 경계가 실제로
   존재함).
3. **weight-only quantization**: activation은 float 그대로 두고 weight만
   압축 저장한 경우, 실행 시 weight를 다시 float으로 되돌려서(dequant) float
   matmul을 돎. IREE의 `FuseDequantizationMatmulPass`
   (`third_party/iree/compiler/src/iree/compiler/GlobalOptimization/FuseDequantizationMatmul.cpp`)가
   찾는 패턴이 정확히 `extui → uitofp → subf → mulf`(정수→float 변환 → zero_point
   빼기 → scale 곱하기) — dequantize 그 자체.

**QOperator는 dequant가 "없는" 게 아니라 op 안에 캡슐화된 것.**
`QLinearMatMul`은 입력 8개(`a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp`)를
받는데, 이건 "int8 곱 → int32 누산 → y_scale/y_zp로 rescale"이라는 계산을 op
하나에 통째로 숨긴 것 — QDQ에서 별도 노드로 노출되던 계산이 QOperator에서는 op
파라미터로 흡수됐을 뿐 계산량은 동일함. 또한 QOperator도 quantized 버전이 없는
연산(위 2번 케이스)을 만나면 그 경계에서 여전히 `DequantizeLinear`/`QuantizeLinear`
쌍을 그대로 남김 — 완전히 사라지는 게 아니라 "지원되는 연산에 한해서만" fuse된
것.

| | QDQ | QOperator |
|---|---|---|
| rescale 수학 | 있음, `DequantizeLinear` 노드로 명시적 노출 | 있음, op 내부 파라미터로 은닉 |
| 컴파일러가 fuse 못하면 | float로 fallback해서 그냥 돌아감(정확도 손실만) | fallback 없음 — 그 op를 backend가 지원 안 하면 컴파일 실패 |
| int8 미지원 연산 만나면 | 앞뒤에 그대로 Q/DQ 노드 유지 | 그 op만 예외적으로 Q/DQ 쌍이 그대로 남음 |

**How to apply**: 이후 다른 모델/레이어를 양자화하다가 "이상하게 dequantize
연산이 그래프/IR에 남아있다"고 놀랄 필요 없음 — 정밀도 민감 연산 경계이거나
weight-only 케이스일 가능성이 높고, 후자라면 `FuseDequantizationMatmulPass`가
정상적으로 처리 중인 것. 반대로 컴파일된 IR에서 dequant 노드가 하나도 안
보이면 `LinalgQuantizedMatmulToMatmul`이 완전 fuse에 성공해서 순수 int8
matmul이 된 것(§1의 `aievec.matmul ... i8, i8 into i32` 케이스가 이거).

## §11 — 2026-09-04: 두 번째 버그(BERT-base 비결정론)를 IREE 밖, 순수 IRON(손으로 짠 `aie.device` MLIR)에서 재현 시도 — 최소 버전은 음성(negative) 결과

**동기**: 지금까지 이 두 번째 버그(§"Thirteenth"~"Twentieth")의 최소 재현은
전부 ONNX→`iree-import-onnx`→`iree-compile` 경로로만 만들어졌고, IRON으로
손으로 검증한 적이 없었음. 첫 번째 버그(batch-0 lock pre-charge race)는
IREE 컴파일러 산출물이 아니라 AIE 락 자체의 하드웨어 동작(pre-charged
counting semaphore)이 근본 원인이었던 선례가 있으므로, 이번에도 "IREE가 만드는
특정 코드 패턴 때문"이 아니라 "타일 공유 + dispatch 전환"이라는 더 일반적인
하드웨어 레벨 현상일 가능성을 IREE 컴파일러 경로와 완전히 무관하게 먼저
검증하기로 함 (`_local/mlir_aie_repro/2026-09-04_dispatch_type_alternation/`).

**최소 repro 설계**: 같은 물리 타일 `tile(0,2)` 위에서 실행되는, 구조적으로
다른 두 개의 단일 실행(배치 루프 없음) 프로그램을 각각 별도 xclbin으로 컴파일:
- TYPE A: `out = in + 10`, 락 4개(0-3).
- TYPE B: `out = in*3 + 5`, 락 6개(0-5, extra scratch buffer/lock pair 추가) —
  단순히 상수만 다른 게 아니라 타일 위 프로그램 이미지(.elf, 락 테이블)가
  실제로 다르도록 구성.

의도적으로 배치 루프(이미 고친 버그 #1의 메커니즘)는 아예 넣지 않아서, 여기서
뭔가 깨진다면 그 원인은 순수하게 "다른 타입의 프로그램을 같은 타일에 번갈아
로드/실행하는 것" 하나로 좁혀지도록 설계.

**실행 방법**: 하나의 `xrt::device`에 xclbin A/B를 모두 `register_xclbin`하고
별도의 `xrt::hw_context`(즉 실제 PDI reconfigure가 일어남)를 만들어 `A, B,
A, B, ...`를 반복 실행하며 매번 출력 전체를 검증 (`diag_alternate.cpp`).

**결과**: 개별 sanity check(A만 1회, B만 1회) 모두 정상(`run state 4`, 출력
일치). Alternation 테스트: **10회 alternation(20회 실행) 0/20 실패, 100회
alternation(200회 실행) 0/200 실패** — 전부 `run state 4`, 매 실행 출력
100% 일치, 행(hang)이나 손상 전혀 없음.

**결론 (negative, 하지만 유의미)**: "구조적으로 다른 두 프로그램을 같은 물리
타일에 번갈아 로드하는 hw-context/PDI 스위치" 자체는, 이 정도의 최소 스케일
(타일 1개, 컬럼 1개, 배치/반복 메커니즘 없음, 코어 프로그램 자체도 아주 작음)
에서는 200번을 시도해도 전혀 문제를 일으키지 않음. 즉 실제 버그는 단순
"dispatch type이 바뀐다"는 사실만으로 트리거되는 게 아니라, 실제 BERT-base
dispatch가 갖는 다음 요소 중 하나 이상이 진짜 필요조건일 가능성이 높음:
1. **진짜 batch 실현 메커니즘** — §"Twentieth"에서 확인했듯, 실제 batched
   dispatch는 코어 내부 `scf.for` 루프가 아니라 **shim DMA의 하드웨어
   repeat_count**(`push_to_queue`를 N번 재사용)로 batch를 구현함. 이 repro는
   그 메커니즘을 전혀 쓰지 않았음 — 다음 시도에서 넣어야 할 첫 번째 후보.
2. **컬럼/타일 규모** — 실제 dispatch는 8개 컬럼(`NumCols=8`)에 걸쳐 여러
   타일이 동시에 참여하지만, 이 repro는 컬럼 1개·코어 1개뿐.
3. **IREE가 실제로 생성하는 TCT sync / lock 초기화 코드**는 `aiecc`(mlir-aie
   툴체인)가 생성하는 control code와 다른 코드베이스(`AMDAIEControlCodeToTransaction.cpp`
   vs mlir-aie 자체 lowering)라서, 여기서 "깨끗함"이 IREE 쪽 control code가
   똑같이 깨끗하다는 보장은 아님 — 이 repro가 검증한 건 "AIE 하드웨어/드라이버
   자체는 최소 스케일에서 문제 없다"는 것이지, "IREE의 특정 control-code
   생성 로직도 문제 없다"는 것은 아님.

**How to apply**: 이 최소 버전을 근거로 "타일 공유는 무해하다"고 단정하지
말 것 — 스케일/메커니즘이 실제 버그 조건과 아직 충분히 다름. 다음 단계는
repeat_count 기반 batch 실현(§1 후보)을 이 repro에 추가해서 재시도하는 것이
가장 저렴하고 유력한 다음 실험. 재현 자료: `_local/mlir_aie_repro/
2026-09-04_dispatch_type_alternation/`(`gen_type.py`, `diag_single.cpp`,
`diag_alternate.cpp`, gitignored) — 컴파일/실행은 컨테이너 밖에서
`~/NPU/mlir-aie/ironenv` + `/opt/xilinx/xrt` + `PEANO_INSTALL_DIR=llvm-aie`로
직접 진행(`solo_test.sh` 패턴과 동일), 실행은 반드시 `scripts/lock/with-npu-lock.sh`로 감쌀 것.

## §12 — 2026-09-04, 같은 날 이어서: "애매한 toy 실험 말고 진짜 non-batch→batch 구조로 바로 테스트" — 사용자 요청으로 즉시 업그레이드, 역시 negative (더 강한 증거)

**변경 사항**: §11의 TYPE B(toy, 단일 실행+extra scratch lock만 다름)를 버리고,
진짜 **batched dispatch 구조**로 교체 — `tile(0,2)` 코어 내부에 진짜
`scf.for` BATCH(=4) 루프를 넣고, 매 iteration마다 같은 락(0-3)을
재사용(`gen_type_b_batched.py`). 이건 정확히 버그 #1의 메커니즘(락의 첫
사용에서 pre-charge 레이스)이 다시 나타날 수 있는 구조이므로, 실제 프로덕션
수정(commit `1600078`)과 동일하게 **rep==0의 release 직후에만 10,000회
busy-wait**를 조건부로 삽입(`scf.if %is_first`)해서 버그 #1을 무력화 — 이후
남는 손상이 있다면 그건 순수하게 "dispatch TYPE 전환(버그 #2)" 때문이라고
귀속할 수 있게 설계.

**중간에 발견한 진짜(도구) 버그, 실제 발견과 헷갈리지 않도록 기록**: 첫 컴파일
결과 batch 0만 정상(8), batch 1-3은 전부 정확히 0으로 실패 — 처음엔 버그 #2와
비슷한 패턴처럼 보였지만, `gen_debug_plain_repeat.py`(delay/scf.if 없는
순수 반복만)로 최소화해도 **동일하게 재현**돼서, 이게 실제 HW 현상이 아니라
**이 repro 자체의 작성 버그**임이 드러남: `aiex.npu.dma_memcpy_nd`의 4-튜플
stride 배열에서 batch 선택 차원(`memref<4x64x64xi8>`의 dim1)의 stride를
0으로 잘못 넣어서, 4번의 출력 전송이 전부 offset 0으로만 겹쳐 쓰임 (batch
1-3 슬롯은 아예 한 번도 안 쓰여서 host 버퍼의 초기 0이 그대로 남은 것).
stride를 4096(=64×64 원소)으로 고치자 즉시 4/4 PASS로 해결됨 — **이 리포에서
"host-repeat으로 여러 번 같은 버퍼/락을 재사용하는 패턴"이 N>1로 실제 검증된
건 이번이 처음**(`gen_degree2ch.py`도 문서상 `N=1`로만 실행됐었음, §5절 인용
참고) — 이 부분 자체가 사전 검증 안 된 새 영역이었다는 뜻이므로, 앞으로 이
패턴을 다시 쓸 때는 항상 toy add-10 같은 가장 단순한 형태로 먼저
단독(비-alternation) 검증부터 할 것.

**stride 수정 후 진짜 실험**: TYPE A(비-batched, 단일 실행, `out=in+10`)와
TYPE B(진짜 batched, BATCH=4, delay-fix 내장, `out=in*3+5`)를 각각 별도
xclbin/hw_context로 등록하고 `tile(0,2)` 하나를 공유하며 A→B→A→B... alternation:
- TYPE B 단독 안정성 먼저 확인: 5회 연속 4/4 배치 전부 PASS.
- **Alternation 10회(A 10 + B 10×4batch=40 checks): 전부 PASS.**
- **Alternation 100회(A 100 + B 100×4batch=400 checks): 전부 PASS, 실패 0건.**

**결론: 이번에도 negative, 그러나 이번엔 §11보다 훨씬 실제 구조에 가까운
조건에서 나온 negative임.** "비배치 단일 실행 dispatch"와 "코어 내부에 진짜
batch 루프 + 프로덕션 delay-fix가 들어간 batched dispatch"를 같은 물리
타일에서 hw-context 스위치로 500회 넘게 검사했는데도 전혀 손상이 없었음.
남은 구조적 차이는 이제 명확하게 좁혀짐: (1) 실제 배치가 core-internal
`scf.for`가 아니라 shim DMA `repeat_count` 하드웨어 메커니즘으로 구현된다는
점(§11에서 이미 지적, 이번에도 미반영 — core 루프 방식은 검증했지만
repeat_count 방식은 아직임), (2) 컬럼/타일 스케일(8컬럼 vs 1컬럼), (3)
`aiecc` control-code lowering이 IREE의 것과 다른 코드베이스라는 점.

**How to apply**: "core-internal 루프 기반 batch면 재현 안 된다"는 것까지는
이제 확인됨. 다음으로 저렴하고 유력한 실험은 여전히 (1)
`aiex.npu.push_to_queue`의 `repeat_count` 필드를 실제로 써서 shim-DMA
하드웨어 반복 방식으로 batch를 구현한 TYPE B'을 만들어 같은 alternation
테스트를 반복하는 것 — 이게 실제 bert-base 스케일 dispatch가 쓰는 진짜
메커니즘이므로 구조적 gap을 가장 크게 줄임.

## §13 — 2026-09-04, 같은 날 이어서: 컬럼 수까지 실제와 동일하게(8컬럼) 맞춰서 재검증 — 여전히 negative

**사용자 요청**: "1컬럼만 하지 말고 IREE와 완전히 동일한 조건으로" — 실제
bert-base dispatch가 쓰는 `NumCols=8`(Twentieth 실험에서 확인)까지 스케일을
맞춰서 재검증.

**구현**: §12의 검증된 단일-타일 TYPE A(비배치)/TYPE B(배치, delay-fix
내장) 설계를 `gen_8col.py`의 이미 검증된 컬럼 복제 패턴(컬럼마다 독립된
shim/memtile/core 타일 3개, plain circuit flow, 컬럼별로 유일하지만 rep에는
불변인 DMA id)을 따라 8개 컬럼(0-7)으로 그대로 복제(`gen_multicol.py`).
출력 텐서는 §12에서 겪은 3D stride 버그를 피하려고 `gen_8col.py`와 동일한
"평평한 2D 텐서 + 절대 row offset" 방식으로 주소 지정(컬럼/배치별로 독립된
`dma_memcpy_nd` 호출, 각자 자기 블록의 절대 row에서 시작).

**검증 순서**: 컴파일 클린(양쪽 다) → TYPE A 단독(8컬럼×1 block=8개 블록)
PASS → TYPE B 단독(8컬럼×4배치=32개 블록) PASS, 3회 반복 안정성 확인 →
**A↔B alternation 10회(A 80블록 + B 320블록) 전부 PASS** → **100회(A 800블록
+ B 3200블록) 전부 PASS, 실패 0건**.

**결론: 8컬럼 스케일까지 맞춰도 여전히 완전히 negative.** 실제 BERT-base
dispatch와 동일한 컬럼 수(8), 동일한 batch 크기(4), 실제 커밋된 delay-fix
그대로 넣은 상태에서 비배치↔배치 dispatch를 hw-context 스위치로 4000회
가까이(800+3200) 검사했는데도 손상이 전혀 없었음.

**남은 구조적 차이는 이제 정말 하나로 좁혀짐**: core-internal `scf.for`
루프가 아니라 **shim DMA의 `repeat_count` 하드웨어 필드**로 batch를
구현하는 것 — 이것과 `aiecc`(mlir-aie) vs IREE의 서로 다른 control-code
lowering 코드베이스라는 점만 남음. 컬럼 수는 더 이상 변수가 아님이 확인됨.

**How to apply**: 다음 실험은 반드시 `repeat_count` 기반 batch 실현으로
가야 함 — 컬럼 수를 더 올리거나 다른 변수를 바꾸는 건 더 이상 우선순위가
아님. 재현 자료: `gen_multicol.py`, `diag_8col_single.cpp`,
`diag_alternate_8col.cpp` (모두 `_local/mlir_aie_repro/
2026-09-04_dispatch_type_alternation/`, gitignored).

## §14 — 2026-09-04, 같은 날 이어서: 진짜 `repeat_count` 메커니즘 구현(1컬럼 + 8컬럼) — 역시 negative

**정확한 문법 확보**: `~/NPU/mlir-aie/test/npu-xrt/nd_memcpy_linear_repeat/aie2.py`
(mlir-aie 자체 공식 테스트)를 컴파일해서 실제 생성되는 raw MLIR을 직접
확인. 핵심: `repeat_count`는 별도 필드가 아니라 `aiex.npu.dma_memcpy_nd`의
**4-튜플 access pattern에서 가장 바깥쪽(leftmost) 차원의 size를
repeat_count로, 그 차원의 stride를 0(같은 데이터 반복 전송, 예: weight)
또는 실제 원소 단위 stride(반복마다 다른 위치, 예: 우리 output)로** 주는
것 — **호출 1번**으로 하드웨어가 N번 자동 반복하고, **`dma_wait`도 1번만
필요**함. 지금까지 만든 TYPE B(§12,§13)는 이 호출을 N번 따로따로 한
것이었으므로, 이번엔 정확히 이 방식으로 다시 구현(`gen_type_b_repeatcount.py`,
`gen_multicol.py`에 `MODE=repeat` 추가). 코어 쪽(scf.for 배치 루프 +
rep==0 delay-fix)은 완전히 그대로 유지해서 이 차이 하나만 격리.

**1컬럼**: 단독 검증(5회 연속 4/4 배치 PASS) → A(비배치)↔B(repeat_count
배치) alternation 10회 PASS, **100회(400개 체크) 전부 PASS**.

**8컬럼**: 단독 검증(32/32 블록 PASS, 3회 반복 안정) → alternation 10회
PASS, **100회(A 800블록 + B 3200블록, 총 4000개 체크) 전부 PASS**.

**결론: 이제 core-internal 루프 방식과 진짜 repeat_count 방식 둘 다,
1컬럼과 8컬럼 스케일 모두에서 완전히 negative.** 지금까지 식별했던
"실제 dispatch와 다른 점" 후보(배치 실현 메커니즘, 컬럼 수)를 전부
하나씩 실제와 동일하게 맞춰봤지만 손상이 재현되지 않음.

**남은 후보, 이번 라운드에서 새로 발견한 것 포함**:
1. **컬럼 간 결합된 TCT sync (`col_num`)** — Twentieth 실험 원문을 다시
   보면 실제 IREE 컨트롤 코드는 컬럼마다 독립적으로 `dma_wait`을 하는 게
   아니라, **연속된 여러 컬럼의 push_to_queue를 하나의 `amdaie.npu.tct_sync`로
   묶어서(`col_num=8`/`8`/`4`) 한 번에 기다림**. 이번 8컬럼 repro는 컬럼마다
   **독립적으로** `dma_wait`을 8번 호출했음 — 이 "여러 컬럼을 묶어서 하나의
   완료 신호로 기다리는" 구조 자체를 아직 재현 안 함. 이게 지금 특정할 수
   있는 것 중 가장 구체적이고 실제와 다른 지점.
2. `aiecc`(mlir-aie) vs IREE `AMDAIEControlCodeToTransaction.cpp`가 여전히
   다른 코드베이스라는 점 자체(구조적으로 똑같이 만들어도 최종 CDO
   바이너리가 byte-identical이라는 보장은 없음).
3. 실제 bert-base는 단순히 "타입 A ↔ 타입 B" 2종류만 반복하는 게 아니라
   한 레이어 안에서 Q/K/V/output/FFN(비배치) + QK^T/Attn@V(배치)가 뒤섞인
   더 긴/다양한 dispatch 시퀀스임 — 2종류 alternation보다 훨씬 복잡한 순서.

**How to apply**: 다음 가장 유력한 실험은 (1) — 여러 컬럼의 완료 대기를
하나의 결합된 sync로 묶는 구조를 IRON에서 재현해보는 것. 이게 재현되면
"컬럼을 묶어서 기다리는 로직 자체의 버그"라는 훨씬 구체적인 후보로
좁혀짐. 재현 자료: `gen_type_b_repeatcount.py`, `gen_multicol.py`
(`MODE=repeat` 인자 추가됨), `diag_alternate_repeatcount.cpp`,
`diag_alternate_8col_rc.cpp` (모두 `_local/mlir_aie_repro/
2026-09-04_dispatch_type_alternation/`, gitignored).

## §15 — 2026-09-04, 같은 날 이어서: 문서 대신 실제 IREE를 다시 컴파일해서 직접 검증 → 결합 TCT sync까지 IRON으로 재현 — 역시 negative

**사용자 요청**: "옛날 문서 보고 확인하지 말고 지금 직접 IREE로 다시 컴파일해서
차이를 확인하자." 옳은 지적 — `docs/`의 Twentieth 실험 기록을 그대로
믿는 대신, `_local/int8_debug/out/chainmix_L12_int8.mlir`(그때 그 파일,
아직 디스크에 있음)을 **지금 이 순간** 다시 `build/tools/iree-compile`로
컴파일하고 `--mlir-print-ir-after=iree-amdaie-controlcode-lowering`로
control code를 직접 덤프해서 확인 (NPU 실행 아니고 컴파일만이라 락 불필요).

**직접 확인/재확인한 사실**:
- `amdaie.npu.tct_sync {channel=0,col=0,col_num=8,direction=0,...}`,
  `{channel=0,col=0,col_num=8,direction=1,...}`,
  `{channel=1,col=0,col_num=4,direction=1,...}` — 정확히 이 3개 조합이
  **매 dispatch마다** 나타남 (grep으로 72개 tct_sync = dispatch당 3개 ×
  24개 dispatch, L=12 chainmix와 일치).
- **비배치(repeat_count=2) dispatch와 배치(repeat_count=12) dispatch 둘 다
  동일한 col_num=8/8/4 패턴** — 옛 문서 주장 그대로 재확인됨(byte-identical).
- 소스 추적: `AMDAIEControlCodeToTransaction.cpp`의 `appendTCTSync`가
  `col`/`row`/`colNum`/`rowNum`/`direction`/`channel`을 그대로 raw
  `XAIE_IO_CUSTOM_OP_TCT` 트랜잭션 워드에 패킹 — 이건 `aie-rt`/XAIE
  드라이버가 실제로 지원하는 하드웨어 프리미티브(여러 컬럼을 한 번에
  폴링)이지, IREE가 소프트웨어적으로 흉내낸 게 아님.
- **결정적 발견**: `~/NPU/mlir-aie`(aiecc가 쓰는 그 저장소) 자체 소스
  `lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp`의 `DmaWaitToSyncPattern`을
  직접 읽어보니, 지금까지 제 모든 repro가 써온 고수준 `aiex.npu.dma_wait`는
  **`column_num`을 무조건 1로 하드코딩**해서 저수준 `aiex.npu.sync`로
  변환함(주석 그대로: "Create with column_num == 1 and row_num == 1 to
  check for a single column and row"). 즉 지금까지 쓴 API로는 애초에
  멀티컬럼 결합 sync를 만들 수 없는 구조였음 — 컬럼 수를 8로 맞춰도 이
  차이 자체는 여전히 재현이 안 되고 있었던 것.
- 우회책 확인: `aiex.npu.sync`는 mlir-aie 자체 raw MLIR에서 직접 쓸 수
  있는 저수준 op이고(`test/npu-xrt/tile_dmas/writebd_tokens/aie.mlir`에
  실제 사용례 존재), `column`/`column_num`/`row`/`row_num`/`channel`/`direction`을
  전부 노출함 — `dma_wait`를 우회해서 이걸 직접 쓰면 IREE와 동일한
  구조를 만들 수 있음.

**구현**: `gen_multicol.py`에 `WAITMODE` 인자 추가(`percolumn`=기존,
`combined`=8개 컬럼 각각의 `dma_wait` 대신 **`aiex.npu.sync {channel=1,
column=0, column_num=8, direction=0(S2MM), row=0, row_num=1}` 딱 1개**로
교체). §14의 repeat_count 기반 TYPE B와 결합해서 지금까지 나온 것 중
IREE의 실제 control code 구조에 가장 가까운 버전을 만듦.

**검증**: TYPE A(combined-wait) 단독 PASS(8/8) → TYPE B(repeat_count +
combined-wait) 단독 PASS(32/32), 3회 반복 안정 → **A↔B alternation 10회
PASS(80+320) → 100회 PASS(800+3200), 실패 0건.**

**결론: 지금까지 특정 가능했던 실제 IREE 구조(비배치↔배치, 8컬럼,
repeat_count 기반 batch, 컬럼 결합 TCT sync)를 전부 하나로 합쳐도
여전히 완전히 negative.** 이 시점에서 IRON으로 소스 레벨에서 재현 가능한
구조적 차이는 사실상 소진됨.

**남은, 아직 손 안 댄 부분(솔직한 gap)**:
1. 이번에 재현한 건 3개의 tct_sync 중 **출력 쪽 1개(channel=1, col_num=4
   조합)뿐** — 실제로는 입력 쪽 두 그룹(channel=0, direction=0/1,
   col_num=8)도 있고, 이건 아직 안 만듦. 완전히 동일한 3-sync 세트를
   전부 재현하진 않음.
2. `aiecc`(mlir-aie)와 IREE의 `AMDAIEControlCodeToTransaction.cpp`가
   여전히 물리적으로 다른 코드베이스 — 구조적으로 동일해 보여도 최종
   CDO 바이너리가 완전히 같다는 보장은 없음(직접 바이트 비교는 안 함).
3. 실제 bert-base는 2종류 dispatch가 아니라 한 레이어 안에 여러
   dispatch(Q/K/V/output/FFN + QK^T/Attn@V)가 뒤섞인 훨씬 긴 실제
   시퀀스 — 제 테스트는 항상 정확히 2개 타입만 반복.

**How to apply**: 여기서부터는 "한 가지 메커니즘을 더 추가하면 재현될
것"이라는 가설보다, (a) 3-sync 세트를 완전히 재현하거나 (b) 실제 12-layer
전체 dispatch 시퀀스를 그대로 IRON으로 옮기는 등 훨씬 큰 재현 비용이
드는 방향으로 가거나, (c) 이 지점에서 소스/구조 비교는 사실상 소진됐다고
보고 실제 HW 트레이스 도구로 전환하는 결정이 필요함 — 사용자와 상의
필요. 재현 자료: `gen_multicol.py`(`WAITMODE=combined` 추가됨),
`diag_alternate_8col_combined.cpp` (모두 `_local/mlir_aie_repro/
2026-09-04_dispatch_type_alternation/`, gitignored).

## §16 — 2026-09-04, 같은 날 이어서: 실제 chainmix_L12를 지금 다시 실행해서 버그 생존 확인 + hw_context 생성 방식 자체를 소스로 재확인 — 역시 negative

**1) 진짜 아티팩트로 버그가 아직 살아있는지부터 확인 (사용자 요청).** 8/24에
컴파일된 `_local/int8_debug/out/chainmix_L12_int8.vmfb`를 지금 15번 실행
(`iree-run-module`, NPU 락으로 감쌈). **결과: 완전히 재현됨** — 15번 중
12번은 비트 단위로 완전히 동일한 값(corr=0.97074), 나머지 3번(run1, 10,
11)은 각각 다른 값(corr=0.96522/0.97103/0.97056). 이산적(discrete) 패턴
그대로, 노이즈 아님. 딜레이 fix가 이미 적용된 상태에서도 여전히 남아있는
바로 그 두 번째 버그가 오늘도 살아있음을 확인.

**2) 사용자 지적: "그건 IRON에서 해보라는 거였다" — 맞는 지적, 처음엔
잘못 이해함.** 실제 요청은 "실제 12-layer 전체 dispatch 시퀀스를 IRON으로
옮겨서 테스트"였음. 이 방향으로 가기 전에, `iree-dump-module`로 실제
vmfb 구조를 먼저 까봄: **24개 dispatch가 24개의 개별 xclbin이 아니라
`hal.executable.create` 호출 1~2번뿐인 "linked" 실행파일**
(`chainmix_L12_int8_linked_amd_aie`)로 묶여있음 — 지금까지 만든 "TYPE A용
hw_context, TYPE B용 hw_context를 각각 별도로 만들어 전환"하는 제 모델과
다른 것 아니냐는 우려가 생김.

**3) 스키마/런타임 소스를 직접 읽어서 확인 — 실제로는 제 모델이 맞았고,
다른 부분에서 진짜 차이를 발견함.** `runtime/src/iree-amd-aie/schemas/
pdi_executable_def.fbs`의 주석: "`pdi_indices`는 현재 그냥 (0, entry point
개수] 범위다 — 나중에 kernel merging을 하면 바뀔 것" → **지금 이
코드베이스에서는 dispatch(entry point)마다 각자 별도 PDI를 가짐**(아직
병합 안 됨), 제 모델의 기본 전제(타입마다 별도 PDI)는 맞았음. 그런데
`runtime/src/iree-amd-aie/driver/amdxdna/direct_command_buffer.cc:977`를
읽어보니, "reconfiguration"(하나의 context를 재사용하며 가볍게 재구성)
경로는 `AIETarget.cpp`의 `options.enableCtrlPkt` 플래그가 켜져 있을 때만
작동함 — **이번 조사 전체에서 쓴 표준 컴파일 플래그 레시피는 이 플래그를
한 번도 켠 적이 없음** → 즉 `num_reconfigurations == 0` 경로만 항상 탐,
이 경로는:
```cpp
iree_hal_amdxdna_native_device_create_context(...)  // 캐싱/재사용 없이 매번 새로
```
**dispatch 타입에 상관없이, 심지어 같은 타입을 반복 호출해도 매번 완전히
새로운 context를 생성**함이 소스에서 직접 확인됨.

**진짜 차이 발견**: 지금까지 만든 모든 alternation 테스트(§11-15)는
`ctxA`/`ctxB`를 **딱 한 번씩만 생성**하고 계속 재사용했음 — 실제로는
매 dispatch 호출마다 매번 새로 만들어야 정확히 맞음.

**수정 후 재검증**: `diag_alternate_8col_freshctx.cpp` — 매 dispatch
호출 직전에 `xrt::hw_context`를 새로 생성(같은 xclbin UUID로, BO는
재사용 유지, 실제로도 버퍼는 dispatch 간 유지되고 context만 재생성되는
것과 일치). §15의 8컬럼+repeat_count+combined-wait 버전에 이 수정을
적용해서 재실행: **10회, 100회(800+3200개 체크) 전부 PASS, 실패 0건.**

**결론: hw_context를 매번 새로 만드는 정확한 실제 패턴으로 고쳐도
여전히 완전히 negative.** 지금까지 소스에서 확인 가능한 모든 구조적
차이(비배치↔배치, 8컬럼, repeat_count, 결합 TCT sync, 매번-새-context)를
전부 하나로 합쳐도 재현이 안 됨 — 이 시점에서 소스 레벨 재현 시도는
사실상 소진됐다고 보는 게 맞음.

**남은 후보 (전부 훨씬 비용이 큰 것들)**:
1. 3개 tct_sync 세트 중 1개만 재현(입력 쪽 두 그룹 미반영)
2. 실제 24-dispatch 전체를 정확한 순서/구성으로 복제(2종류 단순 alternation이 아니라)
3. PDI 크기/내용 자체가 훨씬 큼(실제 matmul vs 제 toy add/mul) — 로딩
   시간이나 타이밍에 의존하는 레이스라면 이 스케일 차이가 결정적일 수 있음
4. `aiecc`와 IREE가 물리적으로 다른 코드베이스라는 근본적 한계

**How to apply**: 다음 유력 후보는 실제 matmul 스케일(K=768 등 실제 크기)의
core 프로그램으로 교체해서 PDI 로딩 시간을 실제와 비슷하게 맞춰보는 것 —
지금까지 전부 매우 작은 toy 프로그램이라 PDI 로드 자체가 순식간에 끝나서,
혹시 존재할 "PDI 로드 시간이 길어야 드러나는 레이스"를 놓치고 있을 가능성.
재현 자료: `diag_alternate_8col_freshctx.cpp` (`_local/mlir_aie_repro/
2026-09-04_dispatch_type_alternation/`, gitignored).

## §17 — 2026-09-04, 같은 날 이어서: 실제 matmul 스케일(K=768, 진짜 리덕션)로 키워서 재검증 — 역시 negative, 새로운 미검증 변수 하나 발견

**사용자 요청**: "실제 matmul 스케일로 키워봐, batch-0 문제도 그랬는데
레이스 컨디션일 확률이 높을 것 같다."

**구현**: toy add/mul을 진짜 int8 리덕션 matmul로 교체 — K=768(BERT
hidden dim과 동일)을 12×64로 K-타일링(64x64 블록 버퍼 크기는 그대로
유지해서 AIE2P 코어 로컬 메모리 한도(~64KB)를 넘지 않게 함 — 64x768
버퍼를 그대로 쓰면 X만으로 48KB라 코어 메모리 예산을 넘어감). 가중치는
`initial_value = dense<1>`로 코어에 상주하는 상수 버퍼(실제 weight가
dispatch 간 안 바뀌는 것과 유사). 누산기(i32)는 12개 K-타일에 걸쳐
지속되며, 최종 결과는 (X=Y=1 상수 입력이라) `K=768=3×256`이라 truncate하면
정확히 0이 되는 계산상 우연이 있어서, 기존 진단 드라이버가 그대로 쓰던
기대값(TYPE A=11, TYPE B=8)에 정확히 맞도록 epilogue bias를 조정 —
기존 모든 diag 드라이버를 그대로 재사용 가능.

**결과**: xclbin 크기가 확 커짐(toy 20KB → matmul768 54-56KB, PDI/프로그램
크기가 실제로 커졌다는 직접 증거). TYPE A/B 단독 검증 PASS, TYPE B 3회
반복 안정 확인. **8컬럼 + repeat_count + combined-wait + fresh-hw_context(모든
검증된 요소 결합) + 실제 matmul768 스케일로 alternation 10회, 100회
(800+3200개 체크) 전부 PASS, 실패 0건.**

**결론: 실행 시간/PDI 크기를 실제 규모로 키워도 여전히 완전히 negative.**
지금까지 소스에서 확인 가능한 구조적 차이는 사실상 전부 닫혔음에도
재현이 안 됨.

**이번에 새로 떠오른, 아직 전혀 안 건드린 변수 하나**: 지금까지 만든
모든 core 컴퓨트(toy든 matmul768이든)는 전부 **스칼라 연산**
(`arith.muli`/`arith.addi`)이었고, **한 번도 실제 AIE 벡터 명령어
(`aievec.matmul` 등 vectorized codegen)를 쓴 적이 없음**. 원래 첫 번째
버그(batch-0 lock pre-charge)는 **정확히 vectorized 코드에서만** 나타났고
scalar는 멀쩡했다는 게 최초 조사에서 이미 확인된 사실(2026-08-24 세션).
두 번째 버그(이번에 계속 재현 시도 중인 것)가 만약 첫 번째와 비슷하게
**vectorized codegen 경로에만 국한된 현상**이라면, 지금까지의 모든
scalar 기반 IRON repro는 애초에 구조적으로 재현이 **불가능**했을 수
있음 — 다른 조건을 아무리 실제와 똑같이 맞춰도 이 축 하나가 다르면
원천적으로 못 잡아냄.

**How to apply**: 다음 실험은 scalar 대신 실제 `aievec.matmul` 벡터
인트린식을 쓰는 코어 프로그램으로 바꿔서(기존 `gen_bmm_pure_smallN.py`류
스크립트가 이미 손으로 짠 vectorized int8 matmul 패턴을 갖고 있으므로
그 패턴을 이 alternation 테스트에 이식) 같은 alternation 테스트를
재시도하는 것이 가장 유력한 다음 단계. 재현 자료:
`gen_multicol.py`(`SCALE=matmul768` 추가됨),
`diag_alternate_8col_freshctx_matmul.cpp` (모두 `_local/mlir_aie_repro/
2026-09-04_dispatch_type_alternation/`, gitignored).

## §18 — 2026-09-04, 같은 날 이어서: 실제 벡터화(aievec.matmul) 시도 — `aiecc` 툴체인 자체의 한계로 막힘, int8은 불가능·bf16도 실패

**사용자 요청**: scalar 대신 실제 벡터 명령어로 시도. "애초에 지금 버그가
양자화 + NPU 구조에서 나온 거라" — vectorization 경로 자체를 의심.

**정확한 op 확보**: 실제 IREE가 컴파일한 int8 vectorized matmul
(`mm_int8.mlir`, §1의 그 파일)을 지금 다시 `--mlir-print-ir-after-all`로
컴파일해서 `LowerVectorToAIEVec` 패스 직후 IR을 직접 확인 —
`aievec.matmul %10, %11, %12 : vector<8x8xi8>, vector<8x8xi8> into
vector<8x8xi32>` 확인. 그런데 `aiecc`(mlir-aie 독립 툴체인)의 aievec
dialect `.td` 정의를 보니 이 8x8x8-int8 조합은 `aievec.matmul`(AIE2
전용)이 아니라 `aievec.matmul_aie2p`(AIE2P/npu4용, 우리 타겟)에서만
지원됨 — IREE는 자체 vendored/fork된 aievec dialect 사본을 쓰기 때문에
op 이름이 겹쳐도 실제 구현은 다를 수 있음.

**4가지 서로 다른 구조로 시도, 전부 동일한 지점에서 크래시**:
1. `vector<8x8xi8>` 2D strided 읽기 → `aie-opt` 단계에서
   `vector.extract_strided_slice`가 "explicitly marked illegal"로 거부
2. 전체 버퍼를 하나의 큰 벡터로 읽고 레지스터 내에서 슬라이스 추출 →
   같은 거부
3. 4D 타일 레이아웃(`memref<8x8x8x8xi8>`) + `subview`+`collapse_shape`로
   진짜 contiguous하게 만들어서 시도 → `aie-opt`는 통과하지만 `llc`에서
   크래시
4. 완전히 flat한 `memref<64xi8>` 선언(실제 IREE 덤프와 거의 동일한
   패턴, `vector.shape_cast`로 8x8 reinterpret) → 역시 동일하게 `llc`
   크래시

**4번 모두 정확히 동일한 crash**: `LLVM ERROR: unable to legalize
instruction: %925:_(<64 x s8>) = G_CONCAT_VECTORS %25:_(<8 x s8>),
%924:_(<8 x s8>) × 7...` — 소스 구조를 아무리 바꿔도 바이트 단위로
동일한 실패 지점. 추가로 mlir-aie 자체의 `test/Conversion/AIEVecToLLVM/
matmul-aie2p.mlir`(이 op의 lowering을 검증하는 공식 테스트)를 확인해보니
**bf16 조합에 대한 테스트는 여러 개 있는데 int8×int8→int32(8x8x8) 조합에
대한 테스트가 단 하나도 없음** — `.td`에는 "valid" 타입 조합으로 선언은
돼 있지만 실제 lowering 패턴이 구현/테스트 안 된 상태로 보임.

**결론: 이건 제가 짠 MLIR 문제가 아니라 `aiecc`(mlir-aie 업스트림)
툴체인 자체가 int8 8x8x8 `matmul_aie2p`의 코드 생성을 완전히 지원하지
않는 것으로 보임.** IREE는 자체 vendored aievec dialect 사본을 쓰기
때문에 이 gap을 우회/수정한 상태이고, 업스트림 mlir-aie(아이콘 이 세션
전체에서 IRON 작업에 쓴 그 툴체인)에는 이 수정이 없음.

**bf16으로도 재시도 — 다른 지점에서 역시 크래시.** 사용자가 "int8이
막혔으면 bf16으로 vectorization 경로 자체가 문제인지만 확인해보자"고
방향 조정. `matmul_aie2p_8x8x8`(bf16×bf16→f32, AIEVecToLLVM 테스트에
실제 존재하는 검증된 조합)로 시도 — `aie-opt` 단계는 통과했지만 `llc`에서
**다른** 크래시: `unable to legalize instruction: %59:_(<64 x s32>) =
G_INSERT_VECTOR_ELT ...` (int8 때와는 다른 명령어, f32 결과를 다시
i8로 requantize하는 후처리 부분에서 발생 — bf16 matmul 자체보다는
epilogue 변환 쪽 문제일 가능성).

**현재 상태: `aiecc` 기반 IRON으로는 int8 vectorized matmul 재현이
사실상 불가능하다고 결론. bf16도 다른 지점에서 막혀 미해결.** "벡터화
경로 자체가 두 번째 버그의 필요조건인가"라는 질문은 **negative가 아니라
inconclusive**로 남음 — scalar로는 계속 재현 안 됐지만, 벡터화를 아예
테스트할 수 없었기 때문에 이 축은 아직 열려있음.

**How to apply**: 이 방향을 계속 파려면 (a) `aiecc` 대신 IREE 자체
컴파일러(`iree-compile`+AMD-AIE 백엔드)에 손으로 짠 저수준 MLIR을 직접
태워서(ONNX/양자화 프론트엔드는 생략) IREE의 실제로 작동하는 aievec
lowering을 활용하는 방법을 찾아야 함 — 진입점을 새로 찾아야 해서 품이
더 듦. (b) 또는 bf16 epilogue 크래시를 마저 디버깅해서 최소한 "벡터화
경로 자체"(dtype 무관)가 원인인지만이라도 확인. (c) 또는 여기서 벡터화
축은 미해결로 남기고 scalar 기반의 종합적 negative 결과(§11-17)를
최종으로 삼아 다른 방향(3-sync 완전 재현, 전체 dispatch 체인, HW 트레이스
도구)으로 전환. 재현 자료: `/tmp/vecmatmul_probe/probe*.mlir` (세션
스크래치, 저장소 밖).

## §19 — 2026-09-04, 같은 날 이어서: 기존 IREE 컴파일 int8 vectorized matmul 아티팩트 재활용 + chainmix 임계값 재측정 — L=6→10으로 상승 확인

**사용자 질문**: "matmul 양자화로 돌린 iron 예제 없어?" — 정곡을 찌른 질문.
확인해보니 `aiecc`로 손으로 짠 게 아니라, **원래 버그#1 조사(8/24) 때
IREE로 컴파일해서 실제 하드웨어에서 검증까지 끝난 진짜 vectorized int8
matmul**이 이미 있었음: `mm_int8.vmfb`(비배치, 기존 파일 재사용),
`bmm_int8.vmfb`(배치, 소스 `gen_and_quant_bmm.py`가 남아있어서 재컴파일,
exit=0 — 이 빌드에 torch-mlir bmm int8 패치가 이미 반영돼 있음을 재확인).
둘 다 개별 실행 정상(bmm corr=0.9998). `iree-run-module`로 직접 15회씩
번갈아 실행(별도 프로세스) → 완전히 결정론적 — 다만 이건 L=1급 스케일 +
프로세스 매번 새로 뜨는 약한 테스트라 당연한 negative, 정보량 낮음.

**사용자 질문 2: "L=12 corr가 거의 100%인데 예전엔 70%대였다, 안 고쳐도
되는 거 아니냐?"** — 원소 단위로 직접 세어봄: run1은 corr=0.965였지만
실제로는 **24576개 중 2728개(11.10%)가 다르고 최대 절대 차이 0.28**
(출력값 자체가 보통 0.01~0.15 스케일임을 감안하면 결코 작지 않음).
Pearson correlation은 "일부 원소가 상당히 다르지만 나머지 대부분이
일치"하는 상황에 안 민감해서 착시를 일으킴 — corr 숫자만 보고 "거의
정상"이라고 판단하면 안 됨. 왜 예전 0.70-0.90과 다른지는 이 조사 자체가
이미 기록한 "레이스 발생 빈도/심각도가 시간에 따라 통제 안 되는 요인으로
흔들린다"(Seventeenth 실험)로 설명 가능 — 버그가 약해진 게 아니라 이번
샘플이 우연히 덜 심각했을 뿐일 가능성이 높음.

**L을 낮춰서 더 저렴한 재현 크기 재탐색 (사용자 요청)**: `gen_chained_mixed.py`로
L=6,8,10을 새로 생성/임포트(컨테이너 안, ONNX/양자화)+컴파일(호스트,
`iree-compile`, 동일 플래그)해서 각각 20회씩 재실행:
- **L=6: 45/45 완전히 결정론적** (기존 문서는 "L=6에서 이산적 두 값 분리"라고 기록돼 있었음 — 재현 안 됨)
- **L=8: 20/20 완전히 결정론적** (기존 문서는 "L=8도 이산적 분리"라고 기록 — 재현 안 됨)
- **L=10: 재현됨 — 20회 중 15/5로 갈림**, 다른 그룹은 578/24576(2.35%) 원소가 다르고 maxdiff=0.029

**결론: 버그 자체는 여전히 살아있지만(L=12 today도 재현, L=10도 재현),
재현에 필요한 최소 L(임계값)이 원래 문서화된 6에서 10으로 올라간 것으로
보임.** 가장 유력한 설명: 8/24 이후 이 컴파일러에 여러 커밋(row-overflow
수정, batch matmul padding 수정, accumulator rescale 수정, batch-0
lock-precharge delay-fix 등)이 반영되면서, 소스 MLIR은 동일해도 실제
생성되는 control-code/타이밍 특성이 달라져 정확한 트리거 조건이 이동한
것으로 추정(직접 diff는 안 함). 버그 원인은 바뀌지 않았을 가능성이
높지만, 최소 재현 크기가 바뀌었으므로 앞으로는 **L=10을 최소 재현
기준으로 사용**.

**How to apply**: 앞으로 이 버그를 대상으로 한 저비용 반복 실험은
L=6이 아니라 **L=10**(컴파일 ~32초, 20회 중 25% 발생률 확인)을 기준으로
삼을 것 — L=6/8을 재현 안 된다고 폐기하지 말고, 임계값이 이동했다는
사실 자체를 기억할 것. 재현 자료(모두 gitignored):
`_local/int8_debug/out/chainmix_L{6,8,10}_int8.{onnx,mlir,vmfb}`,
`_local/int8_debug/vecalt_test/`(mm_int8/bmm_int8 alternation 테스트).

## §20 — 2026-09-04, 같은 날 이어서: chainmix를 더 저렴하게 축소 시도 — D(hidden dim)를 줄이면 L=12에서도 재현 안 됨, 크기 자체가 필요조건

**시도**: `gen_chained_mixed_mini.py` 작성 — chainmix와 완전히 동일한
구조(비배치 MatMul → Reshape/Transpose → 배치 MatMul → Transpose/Reshape,
L번 반복)를 유지하되 D(hidden dim)/HEADS를 파라미터화해서 실제
BERT-base 크기(D=768, HEADS=12)보다 훨씬 작게 줄여 컴파일/반복 비용을
낮추려 시도.

**D=64, HEADS=2, L=12**: 컴파일 정상(34초, chainmix와 비슷 — 즉 크기를
줄여도 컴파일 시간 자체는 별로 안 줄어듦, per-dispatch AIE/Peano
컴파일 오버헤드가 지배적). `--mlir-print-ir-after-all`로 확인해보니
**vectorization은 여전히 켜져 있음**(aievec.matmul 576회 등장, 최초
`num_cols=8` grep이 잘못된 pass를 봐서 0으로 나왔던 건 착오였고 재확인
결과 벡터화는 문제 없이 적용됨). 그런데 **20회 실행 전부 완전히
결정론적 — 재현 안 됨.**

**D=384, HEADS=6(원래의 절반), L=12**: 마찬가지로 **20회 전부
결정론적 — 재현 안 됨.**

**결론: 벡터화가 켜져 있어도, dispatch 반복 횟수(L=12)가 원래
재현되던 값과 같아도, D/HEADS를 줄이면(768→384, 심지어 768→64)
재현이 안 됨.** 즉 "타입 alternation 횟수"나 "벡터화 여부"만으로는
설명이 안 되고, **실제 텐서 크기(정확히는 이게 몇 개의 물리적 AIE
컬럼에 걸쳐 타일링되는지)가 함께 필요조건**인 것으로 보임 — D=768은
여러 컬럼에 걸쳐 타일링되지만 D=384/D=64는 훨씬 적은 컬럼만 쓸 가능성이
높음(직접 컬럼 배치까지 diff하진 않음). 즉 버그는 "충분한 횟수 ×
충분한 컬럼 스프레드"의 곱 같은 조건일 가능성이 있음 — 어느 한쪽만
줄여도 재현이 사라짐.

**실용적 함의: 이 버그는 D=768(또는 그에 가까운 크기) 없이는 값싸게
축소 재현하기 어려움** — 지금까지 시도한 축소는 전부 실패. 저비용
반복 실험이라는 원래 목표는 L 방향으로만 가능(L=10이 현재 최소
확인값), D/HEADS 방향으로는 축소가 안 먹힘.

**How to apply**: 크기를 더 줄이는 시도(D=128, D=256 등 중간값)는
추가로 해볼 수 있으나, 이미 절반(D=384)에서도 실패한 걸 보면 급격한
비선형 임계치일 가능성이 높음 — 무작정 더 줄이기보다는 실제 컴파일된
IR에서 D=768 vs D=384/64의 **타일/컬럼 배치가 정말 다른지 직접
diff**하는 게 다음으로 저렴하고 확실한 검증. 재현 자료(gitignored):
`_local/int8_debug/gen_chained_mixed_mini.py`,
`_local/int8_debug/out/minichain_L12_D{64,384}_H{2,6}_int8.{onnx,mlir,vmfb}`.

## §21 — 2026-09-04, 같은 날 이어서: 타일/컬럼 배치 직접 diff — 컬럼 개수 가설 반증, "dispatch당 전송량"이 진짜 변수로 확정

**사용자 요청**: "타일/컬럼 배치부터 diff해보자." `--mlir-print-ir-after-all`로
D=768(chainmix_L12, 기존 덤프)과 D=64/384(minichain, 새로 덤프)를 직접
비교.

**1) `col_num`(몇 개 컬럼에 걸쳐 완료를 기다리는지) 비교 — 완전히
동일함, 반증됨.** D=768과 D=64 양쪽 다 `tct_sync`가 정확히
`col_num=8/8/4`로 byte-identical. **즉 컬럼 스프레드는 D 크기와 무관하게
항상 고정(8개)** — "작은 크기는 컬럼을 덜 써서 안전하다"는 가설은
틀림.

**2) `push_to_queue`의 `repeat_count`(배치 실현 메커니즘) 비교 —
D=64,HEADS=2는 배치 개수(2)가 너무 작아서 `repeat_count` 메커니즘을
아예 안 쓰고 전부 `repeat_count=1`(완전 unroll)로 컴파일됨. 이건
원본(D=768,HEADS=12, `repeat_count=12`)과 근본적으로 다른 배치
구현이라 순수 크기 비교가 아니었음 — 교란 변수 발견.** D=384,HEADS=6은
`repeat_count=6`을 정상적으로 사용(원본과 동일 메커니즘 유지) — 이
경우가 진짜 깨끗한 비교 대상.

**3) `write_bd`의 `buffer_length`(실제 DMA 전송 바이트 수) 비교 —
D에 명확히 비례해서 스케일링됨.**

| | 최대 buffer_length | 배치 메커니즘 | 재현 여부 |
|---|---|---|---|
| D=768, HEADS=12 | **18432 bytes** | repeat_count=12 | 재현됨 |
| D=384, HEADS=6 | 4608 bytes (정확히 1/4) | repeat_count=6(동일 메커니즘) | 재현 안 됨 |
| D=64, HEADS=2 | 512 bytes | repeat_count=1(다른 메커니즘) | 재현 안 됨 |

D=384 케이스는 배치 메커니즘까지 원본과 동일하게 유지한 가장 깨끗한
비교인데도 재현이 안 됨 — "메커니즘이 달라서"라는 반론도 배제됨.

**결론: 컬럼 개수 가설은 명확히 반증. "dispatch당 실제 DMA 전송
데이터량(D에 비례)"이 진짜 필요조건으로 확정.** 이 버그는:
1. **충분한 반복 횟수(L)** — L=10 이상 (D=768 고정 상태에서 확인, §19)
2. **충분한 dispatch당 전송량(D)** — D=768 필요, 절반(D=384)만 돼도
   사라짐 (L=12 고정 상태에서 확인, 이번 §21)

**두 개의 독립된 임계값을 동시에 넘어야 재현되는 구조**로 최종 정리됨.
이는 "호스트 딜레이를 아무리 넣어도 안 고쳐졌다"(Seventeenth/Eighteenth
실험)는 기존 사실과도 정합적 — 문제는 "경과 시간"이 아니라 "그
dispatch 자체가 실제로 하드웨어를 점유하는 시간/데이터량"이라는 뜻.

**How to apply**: 앞으로 이 방향(코드 레벨, 드라이버 비접근 원칙)에서
다음으로 저렴한 검증은 D를 중간값(예: 512, 640)으로 스캔해서 정확한
D 임계값을 찾는 것. 다만 D=384(절반)에서도 이미 실패한 걸 보면 임계값이
768에 상당히 가까울 가능성이 있어 큰 폭의 축소는 어려울 수 있음 — 저비용
반복 실험은 계속 L 축(L=10)이 중심이 되어야 함. 재현 자료(gitignored):
`/tmp/minichain_d384_full.err.log`, `/tmp/minichain_full.err.log`
(세션 스크래치, `--mlir-print-ir-after-all` 전체 덤프, 용량 커서
저장소엔 안 남김).
