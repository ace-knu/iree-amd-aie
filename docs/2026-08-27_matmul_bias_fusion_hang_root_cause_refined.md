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

---

## §7. BD id 예산 문제 — 실측 후 밸런싱으로 해결

다음 세션(같은 날 후반)에서 "다음 단계 1"을 실제로 시도했습니다.

**견적(실측):** memtile의 BD id 예산은 총 48개(even 채널군 24 + odd 채널군
24)로 넉넉한데, `matmul_bias_2d`(M=32) 실패 지점을 실측하니 **odd 그룹은
24/24로 꽉 찼고 even 그룹은 20/24로 여유가 있었습니다.** 즉 "예산 절대 부족"이
아니라 "채널 배정이 한쪽으로 쏠린" 문제였습니다.

**원인:** `runtime/.../Utils/ChannelGenerator.cpp`의 채널 선택 로직
(`findFirstAvailableChannel`)이 그냥 인덱스 순서(0,1,2...)로 첫 빈 채널을
고르고, BD id 풀이 어느 쪽에 여유가 있는지 전혀 모름.

**수정 (3번의 반복, 매번 재측정 후 재수정):**
1. 새 채널 탐색 경로에 `preferredOrder`(풀 여유량 기준 재정렬)를 추가 — **효과
   없음.** 원인: 이미 배정된 채널을 재사용하는 LRU 폴백 경로가 그대로였고,
   문제되는 마지막 연결들이 전부 그 경로를 탐.
2. LRU 재사용 폴백도 `preferredOrder`를 참고하도록 수정 — 불균형이
   **odd→even으로 반전**(even 24/24, odd 22/24). 원인: producer(MM2S)/consumer
   (S2MM) 사용량을 따로 카운트했는데, 실제로는 한 타일의 BD id 풀을 양쪽
   방향이 **공유**함(`AMDAIEAssignBufferDescriptorIDs.cpp`가 방향 무관하게 타일당
   제너레이터 하나만 씀).
3. producer/consumer 카운터를 하나로 통합 — **여전히 효과 없음.** 원인: circuit
   connection이 먼저 채널을 가져가며 이미 한쪽 풀을 채워놓는데, 카운터를 packet
   모드일 때만 증가시켜서 packet 밸런싱 로직이 circuit이 만든 쏠림을 못 봄.
4. circuit 배정도 카운터에 반영(재정렬 대상은 여전히 packet만) — **`could not
   find and assign a valid BD id` 에러가 로그에서 완전히 사라짐.** BD id 문제
   해결 확인.

## §8. 그 자리를 대신한 새 병목 — 라우팅/arbiter 자원

BD id가 뚫리자 컴파일이 훨씬 더 진행됐고(런타임 시퀀스, `npu_instructions`까지
생성), 대신 그 다음 단계에서 새 에러가 남:

```
error: 'aie.device' op could not create a valid routing configuration
error: failed to convert packet flows to amsels and rules
```

같은 근본 원인(1개 connection → N개 독립 connection으로 쪼개면 모든 하류
하드웨어 자원을 N배로 소비)이 BD id 다음으로 라우팅 자원에서도 나타난 것.

## §9. 원인 재규정 — "쪼개기"가 아니라 "타입"이 문제였다

사용자와의 논의로 관점이 바뀜: split 로직은 fan-out(하나가 여러 개로 뻗는 구조)
자체는 없앴지만, 쪼개진 N개 connection이 여전히 **`ConnectionType::Packet`**로
남아있어서(원본 타입을 그대로 clone) 패킷 스위치 전용 자원(BD id → 라우팅)을
계속 소비하고 있었음이 드러남. → "쪼갠 뒤 Circuit 타입으로 재분류할 수 있는가"
라는 새 가설(§10) 로 이어짐.

## §10. 실험 (B): split 후 Circuit 재분류 — 실패, 예상된 방식으로

`splitPacketConnectionAcrossTiles`에서 clone한 connection을
`ConnectionType::Circuit`로 재생성하도록 임시 수정 → 컴파일 시도:

```
error: 'amdaie.connection' op no producer DMA channel available
```

**원래 `packet-flow-strategy=inputs`를 도입했던 바로 그 문제(circuit 채널
예산 고갈)가 정확히 재발.** split된 4개 connection이 circuit 채널(타일당 6개)
을 다른 operand들과 나눠 쓰다가 바닥남. → **(B) 기각, 코드 되돌림.**

## §11. bias의 실제 데이터 패턴 확인 — 여러 번 정정된 끝에

이 지점에서 "bias가 진짜 브로드캐스트(같은 값)인지, distribute(다른 슬라이스)
인지"를 여러 차례 재확인/정정했습니다 (세션 내 시행착오 기록):

1. 최초 실측(잘못됨): memtile의 한 채널 그룹에서 offset 0/64/128/192로 4등분된
   버퍼를 발견 → "bias는 distribute"라고 결론. **나중에 이 버퍼가 bias가 아니라
   다른 operand(X/Y)였다고 판명.**
2. 재확인: bias의 진짜 소스(`memref<64xf32>`)를 추적하니 모든 `dma_bd`가
   `offset=0, len=64`(전체) → "bias는 완전 브로드캐스트"로 정정.
3. 사용자 지적으로 재재확인: pre-lowering IR
   (`scf.forall (%arg2, %arg3) in (4, 8) {...} {mapping = [thread<y>, thread<x>]}`,
   `%arg2`=row(4), `%arg3`=col(8))을 직접 추적한 결과, **최종 정답**은:
   - L3→L2(shim→memtile): 8개 column이 각자 독립적으로 bias 전체 64개를
     DRAM에서 재읽음 (`aie.shim_dma_allocation`이 column마다 별도 심볼, fan-out
     아님, 안전)
   - L2→L1(memtile→4코어): 그 column 몫인 **8개만** 잘라서(`%lof[%arg3,0]`,
     row `%arg2`는 인덱싱에 안 쓰임) 같은 column의 4개 row 코어에 **동일하게
     브로드캐스트**. `{%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}` 같은 식으로
     4개 타일이 `memref<8xf32,2>` 하나를 공유.

   즉 **column(8) 방향은 distribute, row(4) 방향은 broadcast** — 사용자가 처음에
   제안한 모델이 정확했음. (제가 1번에서 반대로 짚었던 게 완전히 다른 버퍼를
   착각한 것이었고, 2번의 "완전 브로드캐스트"도 column 방향 차이를 놓친
   불완전한 결론이었음.)

## §12. 실하드웨어 검증: "브로드캐스트도 hang" — 결정적 확정

§9의 재규정에 따라, split을 하지 않고 원본 bias connection(4타일 공유, 진짜
브로드캐스트)을 그대로 둔 채 IREE 전체 파이프라인(mlir-aie 아님, 실제
`iree-compile`+`iree-run-module`+실 npu4)으로 직접 테스트:

- `AMDAIEAssignChannels.cpp`에 `isGenuineDistribute` 휴리스틱(소스/타깃
  objectFifo 크기 비교)을 임시 추가해 **크기가 같으면(브로드캐스트) split을
  건너뛰도록** 수정.
- 결과: **BD id, 라우팅 에러 둘 다 완전히 사라지고 컴파일 성공**(304479 바이트
  vmfb). bias를 안 쪼개니 자원 소모가 사라진 것 확인.
- 하지만 `scripts/debug/pipeline_dump.py`로 실 npu4에서 실행하니:
  ```
  INTERNAL; amdxdna dispatch did not complete: ert state 8 (TIMEOUT)
  ```
  **여전히 hang.** → "브로드캐스트는 안전하다"는 가설이 실하드웨어로 반박됨.

**최종 결론:** distribute든 broadcast든, "**1개의 물리 자원(BD)이 여러 물리적
목적지로 동시에 fan-out하는 구조 자체**"가 데이터 내용과 무관하게 이 하드웨어
(npu4/Strix)에서 hang을 유발함. → `isGenuineDistribute` 휴리스틱 되돌림
(최종 커밋에는 미포함, BD id 밸런싱만 남음).

## §13. 컴파일러 자원 검사(`detect-arbiter-deadlock`)가 이걸 못 잡는 이유

§12의 hang이 "컴파일러가 자원 초과를 놓친 것 아니냐"는 가설을 검증:

- `--iree-amdaie-detect-arbiter-deadlock`은 **기본값이 이미 `true`**(항상
  켜져 있었음). 다만 검사 범위가 좁아서 "한 arbiter가 msel 그룹을 2개 이상
  쓰는가"만 봄 — "arbiter를 몇 개 쓰는가"나 "브로드캐스트가 하드웨어적으로
  안전한가"는 애초에 검사 대상이 아님.
- §12에서 컴파일된 IR을 직접 스캔: arbiter 인덱스 최댓값 4(한도 5 이내), msel
  최댓값 0(한도 3 이내, 사실상 미사용) — **자원 초과 흔적 전혀 없음.**
- 즉 "컴파일러가 자원 폭증을 놓쳤다"는 가설은 숫자로 반박됨. 하드웨어 문서상
  한도 내인데도 hang 나는 걸 보면, 컴파일 타임 카운팅으로는 원천적으로 못 잡는
  실리콘/펌웨어 레벨 문제일 가능성이 높음 — §12 결론을 재확인함.

## §14. Upstream 사례 조사 — nod-ai/iree-amd-aie#644, PR #709

같은 클래스 문제("matmul+elementwise 융합 시 connection 폭증")를 upstream
팀도 겪었는지 웹 검색:

- **[Issue #644 "ObjectFifo Matmul + Elementwise"](https://github.com/nod-ai/iree-amd-aie/issues/644)**:
  정확히 같은 문제. 팀이 검토한 3가지 접근법 중 "1. 패킷 라우팅으로 스트림
  재사용"은 **명시적으로 기각**하고 "3. 가능한 connection들을 하나로 통합
  (재사용)"을 채택.
- **[PR #709](https://github.com/nod-ai/iree-amd-aie/pull/709)**: approach 3의
  실제 구현. `amdaie.logicalobjectfifo.placeholder` 메커니즘으로 **L3(DDR)
  쪽에서만** 하나의 물리 connection을 여러 논리적 데이터가 시간차로(순차적으로)
  재사용하게 함.
- **저장소 내 확인**: `AMDAIE_LogicalObjectFifoPlaceholderOp`
  (`AMDAIEOps.td:1459`)와 `AMDAIECreateAIEWorkgroup.cpp:160-200`에 이미
  존재. `AMDAIEDistributeCoresAndObjectFifos.cpp`(#625),
  `AMDAIEFlattenLogicalObjectFifo.cpp`(#638/#652)도 존재하고 기본 파이프라인에
  이미 포함됨.
- **직접 확인**: 이 두 패스 전후로 IR을 덤프해서 봤더니, X/Y/bias는 이 패스들
  실행 전후로도 **여전히 별개의 `dma_cpy_nd`**로 남아있음 — 즉 이 기존
  인프라는 "connection 결합(bias를 X/Y 채널에 얹기)"을 자동으로 해주지
  않음. approach 3 인프라는 있지만 **L3 레벨 시간차 재사용에 한정**돼 있어서,
  우리 문제(L1/L2에서 bias가 4개 코어로 fan-out)에는 그대로 적용 안 됨 —
  새로운 확장 구현이 필요함 (예: K-loop 시작 전 prologue로 bias를 X/Y 채널에
  한 번 얹는 방식 — 순차적 재사용이라면 구조적으로는 말이 됨, 미구현).

## §15. 크기 재검증 — 8 vs 64, 결론 재확인

§12의 실하드웨어 hang이 정확한 크기(8개 원소)로 검증된 것인지 재확인 요청 →
`AMDAIEAssignChannels` 직전 IR을 직접 덤프해서 해당 connection을 확인:

```
%56 = amdaie.connection(%lof_0_r_76 : memref<8xf32,2>(4타일), %lof_0_1 : memref<8xf32,1>)
      {connection_type = Packet}
```

**양쪽 다 정확히 8개 원소.** §12에서 hang을 확인한 connection은 처음부터
authentic한 8개짜리였음 (별도로 만든 mlir-aie N=64 합성 테스트만 크기가 안
맞았던 것뿐, 결정적 증거인 실 IREE 테스트는 애초부터 정확했음). **§12의 결론은
그대로 유효.**

## 오늘 최종 커밋 상태 및 파일별 변경사항

BD id 밸런싱(§7)만 최종 반영, 그 외 실험적 코드(circuit 재분류 §10, broadcast
skip 휴리스틱 §12)는 전부 되돌림.

| 파일 | 변경 내용 |
|---|---|
| `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEAssignChannels.cpp` | `computeBdIdPools`/`buildBdIdAwareChannelOrder` 추가, packet-flow 채널 배정 시 BD-id 풀(짝/홀) 여유량 기준으로 재정렬하도록 `assignChannels` 수정. `splitMultiTilePacketConnections`(§6, 어제 구현)는 그대로 유지 |
| `runtime/src/iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h` | `findFirstAvailableChannel`/`getAndAssignProducerDMAChannel`/`getAndAssignConsumerDMAChannel`에 `order`/`preferredOrder` 파라미터 추가 |
| `runtime/src/iree-amd-aie/aie_runtime/Utils/ChannelGenerator.cpp` | 위 파라미터 구현 — 새 채널 탐색과 LRU 재사용 폴백 둘 다 `preferredOrder`를 따르도록 |

**검증:** `matmul_bias_2d`(M=32) 기준 BD id 에러 완전 해소(재현 로그 3종,
`bdcount_v3~v6.log` 계열로 반복 확인). `bmm_pure_repro`(패킷 connection 없는
케이스) 회귀 없음.

## 다음 단계 (미결, §12 결론 기준으로 갱신)

1. **§10/§12로 "connection 쪼개기" 계열 우회는 전부 막힘** (circuit 재분류는
   채널 고갈, split 자체는 라우팅 고갈, split 없이 두면 hang). 남은 방향은
   §14의 **connection 결합(approach 3 확장)** — bias를 X 또는 Y operand의
   기존 connection에 prologue 방식(순차 재사용)으로 얹는 새 구현. 아직 미착수.
2. 위가 막히면 §1의 (c), 즉 packet-ID 기반 라우팅(mlir-aie `packet_switch`
   예제 방식)이 진짜 대안인지 — `distribute`(hang)와 `packet_switch`(안전
   확인됨)의 IR/하드웨어 레벨 메커니즘 차이를 규명하는 조사가 필요. 아직
   미착수.
3. 어느 쪽이든 해결되면 `matmul_bias_2d`(M=32) 실하드웨어 검증 →
   `bmm_bias_repro`(배치 포함 원래 케이스) → 8컬럼 전체 스케일 확장.
