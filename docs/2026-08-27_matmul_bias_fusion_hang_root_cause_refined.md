# Matmul+bias 퓨전 hang — 진짜 원인 재규명 및 우회 성공 (2026-08-27)

[docs/2026-08-26_matmul_bias_fusion_runtime_hang.md](2026-08-26_matmul_bias_fusion_runtime_hang.md)의
후속. 어제 결론("packet flow + 하드웨어 `repeat_count` 조합이 hang 원인")을
**정정**하고, 실제로 작동하는 우회 방법을 실제 하드웨어로 검증했습니다.

## 요약 (TL;DR)

어제 "repeat_count가 트리거"라고 결론 냈는데 **틀렸습니다.** batch 없는(즉
`repeat_count`가 전부 1인) 순수 2D matmul+bias도 packet flow만 걸리면 여전히
hang이 났어요. mlir-aie로 다시 격리 실험해서 진짜 원인을 찾았습니다:

**진짜 원인은 `object_fifo_link`류의 "하나의 소스를 여러 목적지로 나누는
distribute(fan-out)" 구조 자체입니다.** 완전히 독립적인 별개의 커넥션들(같은
개수 이상이어도)로 바꾸면 packet flow 없이 circuit만으로 정상 작동한다는 걸
실제 npu4 하드웨어로 확인했어요.

IREE 쪽 IR을 다시 보니 L3→relay hop은 이미 독립 커넥션들로 돼있고, **relay→memtile
hop에서 하나의 connection이 여러 채널을 한번에 갖는 distribute 형태**가 보입니다
— 이게 의심되는 지점이고, 지금부터 이걸 실제로 고치는 작업을 시작합니다 (이 문서
작성 시점 기준 아직 미착수, 코드 수정은 이후 별도로 진행).

## 1. 어제 결론 정정: repeat_count는 원인이 아니었다

**실험 1 — batch 없는 순수 2D matmul+bias.** `_local/int8_debug/gen_matmul_bias_2d.py`로
`M=32,K=64,N=64` 2D matmul(배치 차원 없음) + bias 재현 생성. batch가 없으니
DMA loop subsumption이 접을 반복 자체가 없어서, 컴파일된 control code의
`repeat_count`가 **전부 1**임을 직접 확인 (`grep -oE "repeat_count = [0-9]+"` →
28개 전부 `= 1`). `--iree-amdaie-packet-flow-strategy=inputs`로 컴파일 →
정상 컴파일 → 실제 npu4에서 실행 → **동일하게 hang** (`ert state 8`).

**실험 2 — `packet-flow-strategy=auto`로 packet 커넥션 개수 줄이기.**
`inputs`는 congestion 여부와 무관하게 모든 input 방향 커넥션을 packet으로
바꿔버리는 걸 소스에서 확인함(`AMDAIEAssignConnectionTypes.cpp`의
`simpleManualAssignment`, `enableInputPacketFlow`가 켜지면 congestion 체크
없이 무조건 packet). `auto`(congestion-aware, 진짜 필요한 것만 packet)로
바꿔서 packet 커넥션 수를 24개로 줄여서(`inputs`보다 적음) 재시도 —
**여전히 hang.** 즉 packet 커넥션 개수를 줄이는 것만으로는 해결 안 됨.

## 2. mlir-aie로 진짜 트리거 격리

**실험 3 — fan-out(distribute)만 있고 repeat는 없는 케이스.**
`test/npu-xrt/objectfifo_repeat/simple_repeat`(단일 경로, repeat 있음, packet
걸면 hang — 어제 확인)를 기반으로, 이번엔 반대로 **fan-out은 있고 repeat는
없는** 버전을 새로 작성 (`_local/mlir_aie_repro/aie2_broadcast.py`,
`distribute_repeat` 구조 참고해서 `repeat_counter=1`로 고정, `MemTile`이
`ComputeTile2`/`ComputeTile3` 2곳으로 `object_fifo_link(of_in, [of_in2,
of_in3], ...)`로 나눠주는 구조). circuit flow는 `PASS!`, **packet flow
(`--packet-sw-objFifos`)는 hang** (`RUN_EXIT=124`, repeat 전혀 안 썼는데도).

이걸로 정리하면:

| 케이스 | packet flow | fan-out(distribute) | repeat_count>1 | 결과 |
|---|---|---|---|---|
| mlir-aie `packet_switch` (1:1 라우팅) | O | 사실상 X | X | PASS |
| mlir-aie `simple_repeat` (단일 경로) | O | X | O | HANG |
| mlir-aie `aie2_broadcast.py` (방금) | O | **O** | X | HANG |
| IREE `matmul_bias_2d` (배치 없음) | O | O (8컬럼) | X | HANG |
| IREE `bmm_bias` (배치 있음) | O | O | O | HANG |

**공통분모는 fan-out이지 repeat_count가 아님.** repeat_count는 confound였음.

## 3. 우회 검증: 독립 커넥션은 정상 작동 (실제 하드웨어 확인)

**실험 4 — `object_fifo_link` distribute 없이, 완전히 독립적인 2개의 경로로
재작성.** `_local/mlir_aie_repro/aie2_independent.py`: `MemTile`이 아니라
아예 별개의 objectFifo 체인 2세트(`inA2/inB2`, `inA3/inB3`)를 각자 독립적으로
shim→memtile→compute로 연결, 공유되는 distribute 노드 없음. circuit flow로
컴파일(**packet flow 자체가 필요 없어짐** — 컴파일 통과) → 실제 npu4에서
`test_broadcast.exe -x indep_circuit.xclbin ...` → **`PASS!`**

**핵심**: 동시에 필요한 shim 쪽 커넥션 개수는 오히려 늘었는데(1개 distribute
→ 2개 독립 커넥션) 성공했습니다. **문제는 "커넥션이 몇 개 필요한가"가 아니라
"하나의 소스가 여러 목적지로 나뉘는 distribute 구조 자체"**라는 게 확정적으로
증명됨.

## 4. 8-way 스케일 테스트 — 별개의 이유로 미완

우리 실제 케이스(8컬럼)에 맞춰 `_local/mlir_aie_repro/gen_independent_n.py`로
8-way 독립 커넥션 버전을 자동 생성해서 컴파일 시도. **circuit 채널이 아니라
memtile의 lock 개수가 부족해서 크래시** (`AIEObjectFifoStatefulTransform.cpp:429:
Assertion 'prodLockID >= 0 && "No more locks to allocate!"' failed`,
objectFifo depth를 2→1로 줄여도 동일). 이건 8개의 완전히 별개인 objectFifo
쌍을 한 memtile에 naive하게 다 만들어서 lock을 과소비한, **테스트 스크립트
자체의 자원 관리 문제**로 보임 (하드웨어/아키텍처 한계라는 증거는 아직 없음) —
더 lock을 아껴 쓰는 구조로 다시 짜면 될 가능성이 높으나, 이번 세션에서는
완결하지 못함.

## 5. IREE 쪽 실제 connection 구조 재조사

`matmul_bias_2d`(2D, packet-flow-strategy=inputs)의 컴파일된 control-code를
다시 자세히 보니:

- **L3→relay hop**: 이미 8개의 독립적인 `amdaie.connection` op (각각 소스/타겟
  채널이 1개씩인 단순 point-to-point) — 이 hop은 이미 우리가 검증한 "독립
  커넥션" 패턴과 구조적으로 비슷해 보임.
  ```
  %106 = amdaie.connection(%0 {%channel_153}, %104 {%channel_152}, flow = %105)
    {connection_type = Packet} : (memref<8xf32,1:i32>,2>, memref<64xf32>>)
  ```
- **relay→memtile hop**: 여기서 **하나의 connection이 4개의 채널을 한번에
  갖는 형태**가 나타남 — 이게 mlir-aie에서 hang났던 `object_fifo_link`
  distribute 패턴과 구조적으로 같아 보이는 지점.
  ```
  %169 = amdaie.connection(%135 {%channel_719, %channel_720, %channel_721,
    %channel_722}, %0 {%channel_810}, flow = %168)
    {connection_type = Packet} : (memref<8xf32,2:i32>,2>, memref<8xf32,1:i32>,2>)
  ```

**가설**: relay→memtile hop을 만드는 코드가 (아마 `AMDAIELogicalObjFifoSplittingUtils.cpp`나
`AMDAIEConvertToDma.cpp` 근처, 혹은 더 이후 lowering 단계) 여러 목적지를
하나의 connection에 다중 채널로 묶어서 표현하고 있고, 이게 mlir-aie의
`object_fifo_link` distribute와 같은 부류의 하드웨어 문제를 일으키는 것으로
추정됨. **아직 정확히 어느 pass/파일이 이 구조를 만드는지는 특정 못함** —
다음 단계에서 찾아야 함.

## 6. 실제 구현 시도 (같은 날, 이어서) — 진짜 원인 지점 특정 + 코드 수정, 크래시는 다 잡았지만 자원 예산 벽에 부딪힘

`AMDAIEAssignChannels.cpp`(connection에 채널을 배정하는 pass)를 보니: connection의
target(또는 source) logical objectFifo가 여러 물리 타일에 걸쳐있으면, 그 타일
개수만큼 채널을 하나의 connection에 다 부여하는 구조였습니다 (`for (Value tile :
targetLogicalObjFifo.getTiles())`). 이게 "connection 하나에 채널 4개" distribute
구조의 직접적인 원인.

**타일 배정 자체는 `AMDAIEAssignTiles.cpp`가 훨씬 이전에 하는데(§5에서 처음 시도한
지점), packet인지 circuit인지는 `AMDAIEAssignConnectionTypes`가 그보다 *나중에*
결정합니다** — 그래서 `AssignTiles`에서 고치면 안 되는 것도 확인함 (circuit으로
남는, 원래 멀쩡한 케이스까지 망가뜨림 — 진짜 operand(B/Y)가 여러 row에 공유되는 건
정상이고 hang 안 남; **packet으로 바뀌는 경우만** 문제). 그래서 로직을
`AMDAIEAssignChannels.cpp`로 옮겨서, packet-type connection이면서 target/source가
multi-tile인 경우만 골라 독립 connection들로 쪼개도록 구현.

**구현 중 use-after-free/dangling-use 버그를 3단계에 걸쳐 잡음** (전부 실제
크래시로 재현, 디버그 프린트로 위치 특정):
1. `replaceWithNewTiles`가 넘겨준 op를 **지우고 새 op를 반환**하는데, 옛날(이미
   지워진) 포인터를 계속 참조 — use-after-free.
2. connection의 결과(async 토큰)가 `amdaie.core`의 `in:`/`out:` 의존성으로
   직접 쓰이는 경우 리다이렉트 누락.
3. 같은 토큰이 core 안에 nested된 `logicalobjectfifo.acquire`/`release`로도
   쓰이고, **거기다 core에 전혀 속하지 않는 control-code 레벨의
   `amdaie.npu.circular_dma_cpy_nd`가 connection을 직접 참조**하는 경우까지
   있었음 — 이건 재귀적으로 전체 "non-core 참조 체인"을 찾아서 타일별로 통째로
   복제하는 방식으로 해결.

**검증 결과**:
- `bmm_pure_repro`(packet flow 자체가 필요 없는 케이스): 수정 전후 **완전히
  동일한 바이너리 크기, 회귀 없음** — 새 로직이 packet 커넥션이 없으면 진짜
  no-op임을 확인.
- `matmul_bias_2d`(M=32, 우리가 고치려던 실제 케이스, packet-flow-strategy=inputs):
  **크래시는 완전히 사라짐**, 실제 저수준 AIE dialect(`aie.dma_start`,
  `aie.dma_bd`, `aie.use_lock`)까지 정상적으로 lowering됨. 근데 그 다음 단계에서
  **`'aie.memtile_dma' op could not find and assign a valid BD id`** — memtile의
  BD id 예산을 넘어섬.
- M을 16, 8로 줄여서 재시도 — 여전히 실패. 근데 **M=8은 제 수정 없는 원본
  컴파일러에서도 이미 실패**한다는 걸 확인함(다른 에러: `'aie.device' op could
  not create a valid routing configuration`) — 그러니 M=8/16 실패는 제 수정 탓이
  아니라 그 shape 자체가 원래도 힘든 케이스. **공정한 비교는 M=32뿐**이고, 거기서
  "원본: 컴파일은 되는데 hang" → "수정 후: BD id 부족으로 컴파일 자체가 실패"로
  바뀐 것.

**즉 로직 방향은 맞다는 게 검증됨** (크래시 없음, 회귀 없음, §3에서 mlir-aie로
증명한 "독립 connection이면 hang 안 남" 패턴을 그대로 구현함) — **근데 4-way
분할이 이 특정 shape에서 memtile BD id 예산을 넘어서는, 진짜 자원 트레이드오프에
부딪힘.** 이건 로직 버그가 아니라, §4의 mlir-aie 8-way 실험에서 lock이 부족했던
것과 정확히 같은 종류의 현상 — "공유 connection 하나 → 독립 connection N개"로
바꾸면 그만큼 하드웨어 자원(채널, BD id, lock)을 더 씀.

## 다음 단계

1. **BD id 예산 문제 해결** — 후보: (a) 분할된 N개의 connection이 서로 다른 BD
   id를 매번 새로 받는 대신 BD chaining(`AMDAIEInsertDmaBdChain.cpp`, 순환
   재사용)으로 하나의 BD id를 돌려쓰게 하기, (b) 이 분할을 무조건 하지 말고
   실제로 hang의 원인이 되는 경우(대략: fan-out 대상 타일 수가 일정 이상)에만
   선택적으로 적용, (c) 애초에 memtile 레벨에서 각 row가 정말 "독립된
   connection"이 필요한지 재검토 — row들이 진짜 동일한 데이터를 읽는다면, PACKET
   ID를 다르게 줘서 하나의 physical connection으로도 순서 문제 없이 처리되는
   방법이 있을 수도 있음 (mlir-aie의 `packet_switch` 예제가 이런 식으로 packet
   ID 기반 분기를 씀 — §3에서 이건 hang 안 났었음, fan-out과 packet-ID-routing의
   차이를 더 정확히 파봐야 함).
2. 위 해결 후 `matmul_bias_2d`(M=32)부터 실제 하드웨어 검증 → 되면
   `bmm_bias_repro`(배치 있는 원래 케이스)로 확장.
3. 8컬럼 전체 스케일에서 자원 예산 확인.

## 파일별 변경사항 (이번 구현 시도, 전부 uncommitted 상태로 시작 — 커밋 여부는
별도 확인)

| 파일 | 변경 내용 |
|---|---|
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEAssignChannels.cpp` | **EXPERIMENTAL, 미완**: packet-type connection의 target/source가 multi-tile이면 독립 connection들로 분할하는 로직 추가 (`splitPacketConnectionAcrossTiles`, `splitMultiTilePacketConnections`). `bmm_pure` 회귀 없음, `matmul_bias_2d`는 크래시 없이 BD id 부족까지 진행 (아직 컴파일 성공 못 함) |

`AMDAIEAssignTiles.cpp`에 먼저 구현했다가(§5 초반) pass 순서 문제로 전량
되돌림 — 최종적으로 `AMDAIEAssignChannels.cpp`에만 남아있음.

## 재현 자료

이번 세션에서 만든 mlir-aie 재현 스크립트들은 `_local/mlir_aie_repro/`에
저장해뒀습니다 (저장소 관례상 `_local/`은 gitignore, 커밋 안 됨):
- `aie2_broadcast.py` — fan-out(2-way distribute) + repeat 없음, packet flow에서 hang
- `aie2_independent.py` — 완전 독립 2-way 커넥션, circuit로 PASS (우회 검증)
- `gen_independent_n.py` — N-way 독립 커넥션 IRON 스크립트 자동 생성기 (8-way는 lock 부족으로 미완)
- `test_broadcast.cpp`, `test_indep.cpp` — 호스트 사이드 XRT 테스트 (전자가 더 안정적으로 검증됨)

실행 환경은 어제 문서의 "재현 방법 — mlir-aie 쪽"과 동일
(`~/NPU/mlir-aie/ironenv`, `/opt/xilinx/xrt`, 이 저장소의 `llvm-aie`를 Peano로 재사용).

`_local/int8_debug/`에도 오늘 만든 관련 스크립트가 있습니다: `gen_bmm_bias_smallN.py`,
`gen_bmm_bias_tiny.py`, `gen_matmul_bias_2d.py` (N/M을 줄인 재현들, circuit-only
채널 예산이 스케일과 무관하게 부족함을 확인하는 데 씀).
