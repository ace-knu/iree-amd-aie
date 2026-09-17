# 2026-09-08: int8 벡터화 파이프라인 전체 구조 조사 + 안 본 pass 전수 감사

이 문서는 `docs/2026-08-24_int8_quantization_investigation.md`(§1-21),
`docs/2026-09-06_second_bug_scale_threshold_summary.md`,
`docs/2026-09-07_bf16_vectorized_matmul_iron_and_batch0_recheck.md`에 이어지는
당일치 추가 조사. 배경: bf16 IRON 재검증(9/7)이 완전히 negative로 나오면서
"버그 2가 IREE 자체의 int8 컴파일 파이프라인 쪽 결함일 수 있다"는 가설이
힘을 얻었고, 사용자의 박사 선배가 "이런 버그는 대부분 DMA/설정 로직
문제일 것"이라고 조언 — 이번엔 **L=10,12,13이 왜 특정적으로 재현되는지**를
더 파기보다, **컴파일 파이프라인 자체의 완결성**을 순서대로 감사하는 데
집중함.

## 1. 전체 파이프라인 구조 (4단계)

**(파이프라인 진입 전)**: ONNX 양자화(QDQ)가 이미 끝난 int8 linalg 연산으로
시작 — scale/zero-point 등은 이 파이프라인 밖에서 결정됨.

### Stage A — 코드젠/타일링 (dtype 무관, 공통 로직 — 이번에도 안 봄)

`createTypePropagationPass → ... → AMDAIELoweringStrategyPass`(PackPeel 등
타일 파이프라인 선택) `→ AMDAIELowerExecutableTargetPass`(실제 타일링:
`AMDAIETileAndFuse`, `AMDAIEPackAndTranspose`로 8x8x8 타일 형태로 패킹,
L3→L2→L1 승격, `AMDAIEBufferizeToAllocation`) `→ LowerUKernelOpsToCallsPass`.
batch 차원은 여기서 "타일링할 차원 하나 더"로 취급되고, matmul/batch_matmul
둘 다 같은 프레임워크를 탐 — 구조적으로 특별 취급 없음.

### Stage B — objectFifo/DMA 구성 (`addAMDAIEObjectFifoLoweringPasses`)

실제 "DMA 설정" 로직이 몰려있는 단계. 순서대로:

erase-HAL-descriptor-type → fold-memref-alias → `AMDAIEDistributeL1Allocations`
→ **`AMDAIEConvertToDma`**(linalg→objectFifo 기반 DMA) → normalize-loop-bounds
→ **`AMDAIEInsertCores`**(타일별 `amdaie.core` 몸체 생성) → 함수 아웃라이닝
→ **벡터화 1단계**(`AMDAIEInsertLoopsForVectorization` → `AMDAIEVectorizationPass`,
linalg→`vector` 방언, 아직 AIE 전용 명령어 아님) → `AMDAIELocalizeLogicalObjectFifo`
→ `AMDAIEDistributeCoresAndObjectFifos` → 로지컬objFifo 분할 →
`AMDAIEAssignLogicalObjectFifoDepth` → **`AMDAIEAssignTiles`**(물리 타일 배치)
→ `AMDAIEDmaToCircularDma` → `AMDAIECreateAIEWorkgroup` → ... →
**`AMDAIEGenerateControlOverlay`** → **`AMDAIEAssignConnectionTypes`** →
control-code-forall-to-for → **`AMDAIEDmaComposition`**(배치 차원이 여기서
DMA repeat_count 접근패턴으로 접힘 — 핵심 분기점) → dma-cse →
**`AMDAIEAssignChannels`** → **`AMDAIEAssignNpuDmaBdIds`** →
control-code-loop-unroll → ... → **`AMDAIEObjFifoBufferizationPass`**(락
ID/초기값) → `AMDAIETemporaryAllocBufferization` → `AMDAIEConnectionToFlow`
→ `AMDAIEAssignPacketIds` → `AMDAIENpuDmaToHalfDmaCpyNd` →
**`AMDAIEInsertDmaBdChain`** → `AMDAIEFoldDmaWaits` → `AMDAIEControlCodeLowering`
→ `AMDAIEInsertDmaOutOfOrderBlock` → `AMDAIEControlCodeToTransaction`.

### Stage C — amdaie → aie/aiex 방언 변환 (`addAMDAIEToAIEPasses`)

`AMDAIEAcquireReleaseToUseLock` → `AMDAIESinkIntoCore` →
**`AMDAIELowerToAIE`**(논리적 objectFifo/커넥션을 진짜 물리적
`aie.buffer`/`aie.lock`/`aie.flow`/`aie.dma_bd`로 구체화, 1126줄, 이 파일
안의 `connectionToAIE`가 실제 DMA BD 블록/락 acquire-release 구조를 만듦).

### Stage D — mlir-aie 백엔드 (`addMLIRAIELoweringPasses`)

**벡터화 2단계, 진짜 마지막**: `aievec::buildConvertVectorToAIEVec` —
`vector.contract`가 실제 npu4 `aievec.matmul`(8x8x8 int8→i32 융합곱셈누산)로
변환됨. **모든 락/DMA/control-code pass가 끝난 다음, 파이프라인 제일
마지막에 일어남**(놀라운 점). 이어서 `AMDAIEAssignBufferDescriptorIDs` →
`AMDAIEAssignBufferAddresses` → **`AMDAIERouteFlowsWithPathfinder`**(스위치박스
라우팅) → ... → **`AMDAIELocalizeLocks`** → `AMDAIENormalizeAddressSpaces` →
... → **`AMDAIELoadStoreAlignmentReset`** → **`amdaie-standard-lowering`**
(`AMDAIECoreToStandard.cpp`, 배치-0 delay-fix가 있는 곳) → Peano/`llc` →
오브젝트 코드 → xclbin/PDI.

## 2. 안 본 pass 전수 감사 결과 (Stage B 10개 + Stage C/D 4개, 총 14개)

| Pass | 결론 |
|---|---|
| `AMDAIEAssignConnectionTypes` | **죽은 코드** — `packetFlowStrategy` 기본값이 `None`(pass 자체 기본값도, 최상위 컴파일 옵션 기본값도), chainmix는 이 플래그를 준 적 없음 → 모든 connection이 무조건 Circuit. congestion/packet 분기 로직 전체가 미사용(참고: 이 로직 자체는 완전히 별개 트랙인 matmul+bias fusion 행 버그의 원인이었지만, 거긴 `packet-flow-strategy=inputs`를 명시적으로 줬었음). |
| `AMDAIEDmaComposition` | **활성** — `populateDmaLoopSubsumptionPattern`이 batch 차원을 DMA의 outer-dim repeat_count 접근패턴으로 접는 바로 그 로직. 실제 출력 확인: batched matmul에서 `[%c12,%c64][%c0,%c1]`(outer size=12, stride=0=브로드캐스트) — IRON에서 검증한 패턴과 정확히 일치. `onlyZeroStrideOnOuterDim=true`(AIE2+ 기본값)도 일치. 샘플 점검에서 이상 없음. |
| `AMDAIEAssignChannels` | 활성이지만 Circuit-only 경로에선 단순 first-available 배정. non-batched/batched 채널 배정 완전히 동일. packet-flow 전용 복잡 로직(멀티타일 분리, BD-id 풀 밸런싱)은 죽은 코드. |
| `AMDAIEAssignNpuDmaBdIds` | 복잡한 **회전(rotate) BD-ID 로직**(`iv % effSize + offset`, scf.for 루프 안에서 BD ID 재사용) 존재하지만 **실제로는 발동 안 함** — `AMDAIEDmaComposition`이 이미 루프를 DMA 접근패턴으로 흡수해버려서 control-code 레벨엔 scf.for가 안 남음. 모든 BD ID가 상수(0,1,2...)로만 배정됨(`affine.apply` 0회 확인). |
| `AMDAIETemporaryAllocBufferization` | 사소함 — 코어 내 남은 `memref.alloc`을 `amdaie.buffer`로 바꾸는 것뿐, 위험 없음. |
| `AMDAIEConnectionToFlow` | 단순 1:1 변환(connection→flow), 위험 없음. |
| `AMDAIEAssignPacketIds` | **죽은 코드** — packet-flow 전용, `isPacketFlow` 아니면 즉시 skip. |
| `AMDAIENpuDmaToHalfDmaCpyNd` | 결합 DMA op를 source/target half로 나누는 기계적 변환, 위험 없음. |
| **`AMDAIEInsertDmaBdChain`** | 코드상 **진짜 분기 있음**: `repeatCount>1`이면 체이닝 스킵 → 이론상 non-batched(체이닝 가능)와 batched(체이닝 불가)가 다르게 처리될 수 있음. 하지만 **실측: chainmix_L12 24개 dispatch 전부(양쪽 타입 다) `next_bd`가 단 한 번도 세팅 안 됨** — 같은 타일+커넥션에 wait 전 DMA가 2개 이상 몰리는 경우가 없어서 체인 길이가 항상 1. 이론상 제일 유력했는데 실측으로 무관 확인. |
| `AMDAIEInsertDmaOutOfOrderBlock` | **죽은 코드** — packet-flow 전용, S2MM 채널당 `DMAStartOp`가 항상 1개뿐이라 첫 줄에서 no-op. |
| `AMDAIELowerToAIE` (1126줄) | 제일 큰 두 함수(`connectionToAIE`, `workgroupToAIE`) 정독. source/target을 대칭적으로 처리, dispatch 타입별 특수 분기 없음. control-code(NPU 명령어) 생성은 이미 앞 단계에서 끝나있음을 재확인. |
| `AMDAIERouteFlowsWithPathfinder` | mlir-aie의 congestion-aware Dijkstra 라우팅. **새로운 검증: 같은 소스를 두 번 독립 컴파일해서(스레드 끄고) 라우팅 결과 diff — byte-identical**(임시 디렉터리명/ASLR 주소만 다름). **컴파일러 자체가 완전히 결정론적임을 확인** — "컴파일할 때마다 운이 갈리는 것 아닌가"라는 가능성도 닫힘. (부수적으로: 이 재컴파일 시도 자체는 8/24에 이미 기록한 known int8 벡터화 `llc` legalization 크래시로 끝까지 성공은 못 함 — 기존 `chainmix_L12_int8.vmfb`는 이미 성공적으로 컴파일된 파일이라 무관, 재현용 플래그 세트가 원본 레시피와 정확히 일치하진 않는다는 뜻.) |
| `AMDAIELocalizeLocks` | 락 **초기값은 전혀 안 건드림**(어제 걱정과 달리) — 코어가 자기 타일+이웃 4방향(남서북동) 타일의 락을 통합 로컬 인덱스로 참조하도록 `cardinalMemOffset = 방향*numLocks`만 더하는 순수 기하학적 리매핑. 타일 배치가 이미 타입/L 무관 동일하다고 확인했으므로 이것도 자동으로 동일. |
| `AMDAIELoadStoreAlignmentReset` | 완전히 무관 — `convert-vector-to-llvm`이 붙인 과보수적 alignment 속성을 지워서 후속 단계가 벡터 폭 기반으로 재추론하게 하는 workaround(이미 알려진 엣지케이스, "future work"로 명시돼 있음). |

## 3. 배치 행렬곱 vs 비배치 행렬곱 — 실제 차이

**공통점(실측 확인)**: 둘 다 완전히 같은 8x8x8 `aievec.matmul` 명령어 사용,
K=768 리덕션을 K-타일(예: 384씩 2개)로 나누는 것도 동일, 타일/컬럼/락/채널
구조 동일, BD 체이닝 안 걸리는 것도 동일. **"배치 전용 하드웨어 명령어"는
존재하지 않음.**

**유일한 차이 — DMA의 repeat_count 접근패턴**: 배치(HEADS=12)는 컴파일된
DMA에서 outer 차원 size=12로 나타남 — 헤드마다 다른 데이터(X)는 stride>0,
헤드마다 공유되는 데이터(weight)는 stride=0(같은 데이터 12번 재사용,
IRON에서 흉내낸 repeat_count 그대로). 코어 쪽은 그냥 같은 8x8x8 matmul
루프를 12번 반복 실행하는 것뿐 — 배치는 순수하게 DMA/반복 트릭.

## 4. 결론 — 코드 레벨 감사는 이제 사실상 완료

이번 라운드로 fork가 짚어준 "한 번도 안 본 pass" 14개를 전부 감사했고,
**어디에서도 dispatch 타입 간 실행 차이나 명백한 결함을 못 찾음** — 대부분
활성 로직이 우리 컴파일 플래그(packetFlowStrategy=None) 하에서는 실질적으로
단순하거나 아예 죽은 코드였고(패킷 플로우 전용 로직 다수), 이론상 위험해
보였던 두 곳(`AssignNpuDmaBdIds`의 회전 BD-ID, `InsertDmaBdChain`의
repeat_count 분기)도 실측으로는 발동 자체를 안 하는 것으로 확인됨.

이번 라운드의 진짜 새로운 소득은 **컴파일러 자체의 결정론성 확인**(같은
소스 재컴파일 시 라우팅 결과 byte-identical) — 이걸로 "L=10,12,13이 왜
특정적으로 재현되는가"를 정적 코드 레벨에서 설명하려는 시도는 사실상
막다른 길에 도달했다고 봐야 함: 락 초기값, 타일/컬럼 배치, 버퍼 주소,
채널 배정, BD 체이닝, 라우팅까지 — 우리가 확인할 수 있는 모든 컴파일
결과물이 L값과 dispatch 타입 조합에 대해 완전히 결정론적이고 예상대로
동작함.

## 5. 전체 파이프라인 MLIR 덤프 저장 (탐색용)

`_local/int8_debug/out/full_pipeline_dumps/` (gitignored):
- `bmm12_full_pipeline.log` (74만 줄) — batch matmul 단독, 모든 pass 후 IR
- `mm_full_pipeline.log` (68만 줄) — non-batched matmul 단독, 동일
- `pass_index.tsv` / `pass_index_mm.tsv` — pass별 줄번호 색인(각 1300여 개)
- `INDEX.md` — 핵심 pass 줄번호만 골라낸 목차
- `chainmix_L{10,11}_bf16_*.log` — bf16 체인의 4개 핵심 pass 덤프

**배치 vs 비배치 실측 diff (체크포인트 3곳)**: DmaComposition에서만 다름
(batched는 `[%c12, %c256][%c0, %c1]` outer repeat dim, non-batched는 `[256][1]`
평면) — BD 체이닝(둘 다 next_bd 0회), 락 init 분포(완전 동일)는 차이 없음.

## 6. 양자화 vs 비양자화(bf16 스칼라) 비교

동일 shape의 단일 dispatch(`mm`/`bmm12`)를 int8(양자화+벡터화)과
bf16(비양자화+스칼라, `--iree-amdaie-demote-contraction-inputs-to-bf16
--iree-amdaie-enable-vectorization-passes=false`)로 각각 컴파일해 비교:

- **DMA repeat_count 패턴**: 완전히 동일(`[%c12,...][%c0,%c1]`).
- **BD 체이닝**: 둘 다 0회.
- **락 init 분포 형태**: 동일(0/2/8 세 종류), 단 **개수가 정확히 5배 차이**
  (int8 580/540/40 vs bf16 116/108/8) — 양자화가 추가로 요구하는
  pack/unpack/scale 변환용 objectFifo 때문으로 추정.

→ **구조적 패턴은 양자화 여부와 무관하게 동일**하고, 차이는 "같은 종류의
락/DMA가 5배 더 많다"는 규모 차이뿐. 이건 "int8만 문제인 이유"를 구조가
아니라 리소스 규모(동시에 돌아가는 락/채널 수)로 설명할 수 있다는 새 단서.

bf16 체인(L=10,11)도 만들어 비교: 락 개수/타일 수가 L에 정확히 선형 비례,
DmaComposition 패턴 동일, next_bd 0회 — int8과 똑같이 **L값에 따른 구조
차이 전무**.

## 7. bf16 체인 실측 시도 → 하드웨어에서 TIMEOUT (미해결, 별개 이슈)

bf16 chainmix L=10~13을 컴파일(4개 모두 성공)해서 실제 하드웨어에서
결정론성을 확인하려 했으나, **L=10이 2회 연속 `ert state 8`으로 실패**.
`third_party/XRT/.../ert.h` 확인 결과 `ERT_CMD_STATE_TIMEOUT = 8` — 즉
장비 에러가 아니라 **디스패치 타임아웃**(`tdr_timeout_ms=60000` 초과).
bf16은 npu4에서 벡터화 lowering이 없어 순수 스칼라로 도는데, 단순 연산량
추정으로는 60초가 걸릴 규모가 아니라서 **데드락 가능성이 더 큼**(별개
버그 후보). 안전 프로토콜대로 즉시 중단하고 검증된 `chainmix_L12_int8.vmfb`
로 복구 확인 → **정상 실행(RC=0), NPU 건강함 확인**. `xrt-smi examine`도 정상.
→ bf16을 "동일 스케일 비양자화 대조군"으로 쓰려면 이 타임아웃 원인부터
따로 규명해야 함(작은 L부터 스케일업하며 어디서 깨지는지 확인 필요).

## 8. ⚠️ 가장 중요한 발견: L-의존성이 "탐지 가능성" 인공물일 가능성

`_local/int8_debug/gen_chained_mixed.py`를 다시 정독하다 발견한 사실
(29, 46, 55, 75, 89번째 줄):

```python
rng = np.random.default_rng(0)      # 단일 RNG 스트림, seed 0
for i in range(N_LAYERS):
    wv0 = rng.standard_normal((D, D)) ...   # 레이어별 weight
    wv1 = rng.standard_normal((HEADS, HD, HD)) ...
x = rng.standard_normal((32, D)) ...        # ← 루프가 끝난 "다음"에 뽑음!
class Calib: ... rng.standard_normal(...)   # ← 캘리브레이션 데이터도 그 다음
```

즉 **입력 x와 양자화 스케일(캘리브레이션)이 L값마다 전부 다름** — RNG를
weight 생성에 L번 소비한 뒤에 x를 뽑기 때문. 앞쪽 레이어들의 weight는
L끼리 동일하지만, 입력과 스케일이 달라짐.

**따라서 각 L은 "레이어만 하나 더 붙은 같은 모델"이 아니라 완전히 다른
수치 문제(different numerical instance)임.**

이게 중요한 이유: 지금까지 우리는 "왜 L=10,12,13에서만 재현되는가"를
하드웨어/코드 관점에서 찾으려 했지만, **각 dispatch가 독립적으로 동일하게
컴파일되고 fresh context로 로드되며 호스트가 완전 동기화한다면, 실패
확률은 L에 대해 단조 증가해야 함**(P=1-(1-p)^L). 관측된 비단조 패턴
(L=10 재현, L=11은 100/100 클린, L=12 재현, L=13 최악, L=14/16 클린)은
이 모델과 정면으로 모순됨.

**대안 설명**: 레이스는 모든 L에서 비슷한 확률로 발생하지만, 그로 인한
미세한 섭동(perturbation)이 **뒤따르는 int8 양자화(round/clamp) 단계들을
살아남아 최종 출력까지 보이는지**가 그 인스턴스의 구체적 수치(입력값,
스케일, weight 조합)에 따라 갈린다. 양자화는 이산화·수축(contractive)
연산이라 대부분의 미세 섭동을 반올림으로 흡수함. 즉 **L-의존성은 트리거의
성질이 아니라 "탐지 가능성"의 인공물일 수 있음.**

같은 논리로 **"D=768 필요, D=384면 사라짐" 결론도 재검토 대상** — D를
바꾸면 weight 개수가 바뀌어 RNG 상태가 달라지고, 따라서 입력/스케일도
전부 달라지는 다른 인스턴스가 되기 때문.

### 다음 세션 첫 실험 (값싸고 결정적)

`gen_chained_mixed.py`에 **seed를 인자로 추가**하고, 같은 L을 여러 seed로
재생성해서 재현 여부를 비교:
- L=11(현재 100/100 클린)을 seed 1,2,3으로 재생성 → **하나라도 재현되면
  "L-의존성"은 착시**이고, 진짜 변수는 인스턴스별 수치 민감도임.
- L=10(현재 재현)을 seed 1,2,3으로 재생성 → **모두 클린해지면 같은 결론**.

이 결과에 따라:
- (a) 착시로 판명되면 → "왜 L=10,12,13인가"는 애초에 잘못된 질문이었고
  (정적 코드 감사에서 아무것도 못 찾은 것도 당연 — 찾을 게 없었음),
  가장 민감한 인스턴스를 골라 통계를 몰아서 진짜 레이스를 추적하면 됨.
- (b) seed를 바꿔도 L 패턴이 유지되면 → L 자체가 진짜 변수이므로,
  런타임 타이밍 쪽(스코프 밖으로 정해둔 영역)으로 넘어가야 함.

부수적으로 확인해볼 것: 호스트에서 numpy/ORT로 양자화 체인을 시뮬레이션해
중간 레이어에 ±1 LSB 섭동을 주입했을 때 최종 출력이 바뀌는지를 L별로
측정하면, 위 가설을 하드웨어 없이도 직접 검증 가능.
