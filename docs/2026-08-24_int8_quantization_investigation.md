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
