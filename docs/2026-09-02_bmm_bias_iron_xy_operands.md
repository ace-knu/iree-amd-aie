# X/Y 오퍼랜드까지 포함한 진짜 matmul+bias IRON 재현, 그리고 bias를 reduction에 접어넣는 우회책 (2026-09-02)

[docs/2026-08-28_shared_channel_fanout_column_threshold.md](2026-08-28_shared_channel_fanout_column_threshold.md)의
후속. 이 문서부터 날짜별로 별도 파일에 정리하기로 함(이전까지는 한 파일에 계속 이어붙였음).
matmul+bias 퓨전 hang 조사(로드맵 아이템 2)의 연장선.

## 0. 세션 시작 배경 — 호스트(`ace-amd01`) 자체가 죽었다 리부트됨

`ace-amd01`(공유 호스트)가 이 세션 시작 직전 15:48경 죽었다가 15:50에 리부트됨(직전 세션
13:22~15:48). `journalctl -b -1 -p err` 확인 결과 NPU/커널 관련 에러는 없어서 hang/TDR
계열 사고는 아니었던 것으로 보임. 리부트로 락/컨테이너 상태는 깨끗하게 초기화됨.

죽기 직전(14:42~14:55) 마지막으로 만들고 있던 `gen_broadcast_dbuf_x.py`(ping-pong bias +
row별 전용 X 채널, DEGREE=2)는 컴파일까지만 완료되고 실제 HW 실행 로그는 없는 상태로 남아
있었음.

## 1. `gen_broadcast_dbuf_x.py` 재실행 — HANG, 그러나 베이스라인이 이미 hang이라 무의미

리부트 후 상태 확인(`NPU: free`, `build: free`, 활성 컨테이너 없음) 뒤 이 컴파일된 바이너리를
그대로 실행: **`run state: 8`, 123.7초 — hang.** `dest 0`만 성공(15), `dest 1`은 도달 못함(0).

**문제 발견 (사용자 지적)**: 이 재현이 얹은 베이스라인(`gen_broadcast_dbuf.py`, ping-pong bias
DEGREE=2)은 **X 없이도 이미 hang**이었던 걸로 8/31에 확정된 케이스임. 그러니 "X 추가했더니
hang"이라는 결과는 X의 영향을 전혀 말해주지 못함 — 애초에 양성(positive)이던 베이스라인에
뭘 얹어도 양성으로 나올 뿐. 지난 세션 내내 반복됐던 "성공이라 보고한 게 사실 hang이었다"는
방법론 문제의 연장선에서, 이번엔 "hang인 베이스라인에 얹었다"는 실험 설계 자체의 문제였음.

## 2. 올바른 대조군으로 재설계 — `gen_broadcast_x.py`(single-buffer PASS 베이스라인 + X)

**정정된 실험**: X는 **검증된 PASS 베이스라인**(`gen_broadcast.py`, single-buffer/ping-pong
없음, 2026-09-01에 DEGREE=4·0.25초로 PASS 확인됨)에 얹어야 함. `gen_broadcast_x.py` 작성
(DEGREE=2로 축소 — 메모리타일 채널 예산상 DEGREE=4는 X 전용 채널 4개를 못 댐).

같은 스케일(DEGREE=2)에서 재확인/신규 확인 두 가지를 조용한 조건(직전 `docker top`으로
gylee/yekim 활동 없음 확인, gylee가 `with-npu-lock.sh`로 NPU 쓰는 도중엔 자동 대기 후 실행)
에서 순서대로 실행:

| 구성 | 결과 |
|---|---|
| `gen_broadcast.py` DEGREE=2, X 없음 (재확인) | **PASS**, `state=4`, 0.25초, dest0=13/dest1=14 정상 |
| `gen_broadcast_x.py` DEGREE=2, X 있음 | **PASS**, `state=4`, 실측 dispatch 0.52ms, dest0=15/dest1=16 정상 |

**결론**: single-buffer(ping-pong 없음) bias broadcast는 다른 채널에 진짜 트래픽(X)이 얹혀도
안 깨짐. X는 변수가 아니고, **ping-pong 유무만이 IRON 재현에서 결과를 가르는 유일한 변수**임을
재확인.

## 3. 사용자 지적: X는 진짜 matmul이 아님 — Y와 실제 K-reduction MAC 추가

X는 그냥 상수 덧셈이었을 뿐 진짜 matmul이 아니라는 지적(사용자)에 따라 `gen_bmm_bias_iron.py`
작성: `out[i,j] = bias[i,j] + Σ_k X[i,k]·Y[k,j]` — 진짜 K-reduction MAC 루프(K=64, i32
누산 후 i8로 truncate).

**1차 시도 실패**: X와 Y를 각각 독립 채널로(row당 채널 2개씩, 총 6개 동시 memtile flow) 넣으려
하니 `aiecc`가 **어서션으로 크래시**(`AIEPathFinder.cpp:596`, `i < sb.srcPorts.size()` —
`gen_degree_1chan_BROKEN_router_crash.py`와 같은 실패 계열). Y만 packet-broadcast로 바꿔서
memtile flow 개수를 5개로 줄여도 여전히 크래시.

**진짜 원인 확인**: memtile 채널 예산 문제가 아니라 **코어 타일 자체가 하드웨어적으로 S2MM
채널이 2개뿐**(`third_party/aie-rt/driver/src/lite/xaie_lite_hwcfg.h`의
`XAIE_TILE_DMA_NUM_CH=2U` — Strix B0 포함 모든 AIE2(P) variant 공통). bias(채널0) +
X(채널1) + Y(채널2)로 3개를 요구한 게 애초에 존재하지 않는 채널을 요청한 것이었음.

**교차검증**: 옆 팀이 정리해둔 문서(claude.ai 아티팩트, 상용 NPU 조사 — Ethos-U/TPU/NVDLA/
Tenstorrent)도 정확히 같은 결론: "채널 3개로 각각 전달은 불가능하고 의도되지 않음... 조사한
어떤 상용 NPU도 이 방식을 쓰지 않는다." 우리가 오늘 실측으로 부딪힌 벽과 독립적으로 확인된
같은 하드웨어 한계. **부수 발견**: 실제 `bmm_bias`의 CDO/바인딩 분석(이미 기록된 것,
"① DDR_PATCH 개수 vs 바인딩 개수" 섹션)에서도 X와 Y가 **이미 바인딩 하나를 offset으로 나눠
공유**하고 있었음 — 즉 실제 IREE 컴파일러도 이미 이 2채널 한계에 맞춰 X+Y를 합쳐서 내보내고
있었다는 뜻. "3개를 각각 넣는 방법론"은 파볼 가치가 없는 방향으로 확정.

**해결책**: X와 Y를 하나의 `128x64` 버퍼로 합쳐서(위 64행=X, 아래 64행=Y) 채널 1개로 전송 —
코어의 2채널 예산(bias, XY) 안에 정확히 맞음, 컴파일 클린.

**실제 HW 실행 결과 (DEGREE=2, K=64, 조용한 조건 확인 후)**: **PASS**, `run state: 4`, 실측
dispatch 1.6ms, `dest 0`=77(정답, K+13+0), `dest 1`=78(정답, K+13+1) — 진짜 X/Y 오퍼랜드로
실제 K-reduction MAC을 돌리는데도 single-buffer bias는 여전히 안 깨짐.

## 4. XY 채널도 진짜 ping-pong(더블버퍼)으로 바꿔서 재검증 — 여전히 PASS

실제 CDO에서 X 경로도 "ping-pong 4쌍"(K-루프 반복으로 추정)이 있었다는 기존 기록에 맞춰,
`gen_bmm_bias_iron_dbuf.py` 작성: bias는 그대로 single-buffer(안전한 구조 유지)로 두고,
**XY 채널만 BATCH=2 물리 버퍼로 ping-pong**하게 만듦.

단, 이 XY 채널은 row당 전용(consumer 1개, fan-out 없음)이라 예전에 hang을 낸 bias의 "1채널
+ 2 consumer가 집계 lock 하나 공유" 패턴과는 구조가 다름 — memtile 쪽만 BATCH개 물리 버퍼를
순환하고(자기 신호(self-signaling) lock 하나로 체인만 늘림), row 쪽 수신 버퍼는 기존처럼
단일 버퍼 재사용. 별도 lock 추가 없이 기존 `lock{r}_xysrc`의 체인 엔트리만 BATCH개로 늘리고,
호스트/코어 반복 횟수 `N=BATCH=2`로 설정해서 두 버퍼가 실제로 다 쓰이게 함.

**실제 HW 실행 결과 (DEGREE=2, BATCH=2, N=2, K=64)**: **PASS**, `run state: 4`, 실측
dispatch 2.8ms, `dest 0`=77, `dest 1`=78 — 둘 다 정답.

**해석**: 예상 가능했던 결과. 예전에 hang을 낸 bias의 ping-pong 구조는 "**1채널이 2개 이상의
목적지(consumer)로 fan-out**하면서 그 여러 consumer가 **하나의 집계 lock을 공유**"하는 게
핵심이었는데, XY 채널은 애초에 그 위험 패턴이 아님(fan-out 없음). "ping-pong 자체"가 아니라
"**fan-out + 공유 lock + ping-pong**"의 조합이 진짜 트리거였다는 기존 결론이 한 번 더
보강됨.

## 5. 전략 전환: bias를 별도 broadcast 대신 reduction(K) 축에 접어넣기 (Method B) — PASS

지금까지의 3-input(bias/X/Y 각각 독립 채널) 시도는 §3에서 확인했듯 **하드웨어가 막고 있어서**
갈 수 없는 길(코어 타일 S2MM 2채널 하드 리밋 + 옆 팀 조사 결과 둘 다 일치)로 확정됨. 그럼
bias를 아예 별도 broadcast 연결로 안 보내고, **reduction(K) 축에 접어넣는** 두 가지 방식을
검토(옆 팀 문서, claude.ai 아티팩트 cf200d8c...의 "방법 A/B"):

- **Method A (augmented matmul)**: X를 `[M,K]`→`[M,K+1]`로 확장, 마지막 열을 상수 `1`로.
  Y를 `[K,N]`→`[K+1,N]`으로 확장, 마지막 행에 bias. `X_aug·Y_aug = X·Y + 1·bias`.
- **Method B (accumulator 초기화)**: reduction 루프의 accumulator를 `0`이 아니라
  `bias[i,j]`로 초기화하고 그 위에 K-루프 MAC을 누적. 컴파일러 관점에서
  `linalg.fill(0)` → `linalg.fill(bias)` 교체 하나로 표현 가능, X/Y shape 변경 불필요.

둘 다 채널/lock 구조 관점에서는 완전히 동일(bias 전용 broadcast 연결 자체가 사라짐)하므로,
Method B(더 저비용)로 IRON 구현: `gen_bmm_bias_iron_folded.py` — bias를 X/Y와 같은 row별
전용 버퍼(`192x64`: 0-63행=X, 64-127행=Y, 128-191행=bias)에 합쳐서 **채널 1개**로 전송,
bias용 broadcast/packet_flow/공유 lock 기계장치를 통째로 제거. 코어는
`acc0 = bias[i,j]`로 시작해서 K-reduction만 얹고 바로 저장(별도 add 단계 없음). 부수 효과:
row 타일이 이제 S2MM 채널 1개만 쓰므로(기존 2개에서 절반으로 감소) 하드웨어 여유가 더 생김.

**실제 HW 실행 결과 (DEGREE=2, K=64)**: **PASS**, `run state: 4`, 실측 dispatch 1.6ms,
`dest 0`=65, `dest 1`=65 — 둘 다 정답(bias=1 + K=64 reduction = 65).

**Method A vs B 비교** (안전성은 동일, 구현 비용/효율만 다름):

| | Method A (augmented matmul) | Method B (accumulator init) |
|---|---|---|
| 컴파일러 변경 | X/Y shape를 K+1로 패딩 — 타일링 로직 침습적 변경 필요 | `linalg.fill(0)`→`linalg.fill(bias)` 하나만 교체 |
| 실행 비용 | K+1번째 반복에서 `1×bias` 곱셈 낭비 | 초기화 한 번, 낭비 없음 |
| 적용 범위 | reduction 있는 연산 전반 | accumulator 있는 연산(matmul+bias, conv+bias 등) |
| 리스크 | shape 변경이 다른 pass에 부작용 가능 | 국소적 변경, 부작용 범위 작음 |

**결론: Method B가 명확히 더 낫다.** 실제 IREE 코드베이스에서 `linalg.matmul` 앞의
`linalg.fill(0)`을 bias로 바꾸는 패턴(이미 흔한 linalg 관용구)만 넣으면 되고, 텐서 shape/
바인딩 구조 변경이 필요 없음.

## 6. 오늘 세션 종합 결론

1. **X(다른 채널의 진짜 트래픽)도, 진짜 matmul reduction 연산도, XY 채널의 ping-pong도**
   IRON의 single-buffer bias broadcast PASS 결과를 안 깨뜨림. 결과를 가르는 유일한 변수는
   "**fan-out(여러 consumer) + 공유 credit-lock + ping-pong**"의 조합 하나뿐임이 재확인됨.
2. **코어 타일 S2MM 채널 = 2개가 하드 리밋**이라는 새 정적 사실 확보 (실측 크래시 + 상용 NPU
   조사 문서 교차검증 + 실제 bmm_bias 바인딩 공유 패턴, 세 곳에서 독립적으로 수렴). "3-input
   각각 독립 채널" 방향은 조사 가치 없음으로 확정.
3. **실질적 우회책 확보 및 검증 완료**: bias를 별도 broadcast 없이 accumulator 초기값으로
   접어넣는 Method B가 IRON에서 PASS 확인됨. 이게 실제 IREE 컴파일러 fix로 이어질 수 있는
   가장 유력하고 저비용인 방향.

## 재현 자료

`_local/mlir_aie_repro/2026-08-28_column_threshold/`:
- `gen_broadcast_x.py` — single-buffer bias broadcast + row별 전용 X 채널 (검증된 PASS
  베이스라인에 X를 얹은 올바른 버전). `diag_broadcast_x2.cpp`/`.exe`, `diag_baseline2.cpp`
  (X 없는 대조군, `DEGREE_PLACEHOLDER` 치환 방식).
- `gen_bmm_bias_iron.py` — 진짜 bias+matmul(X,Y) IRON 재현. X+Y 합침 128x64 버퍼 방식
  (row당 전용 채널, bias는 여전히 single-buffer broadcast). `diag_bmm_bias_iron.cpp`/`.exe`.
  `python3 gen_bmm_bias_iron.py <DEGREE> <host-repeat N> <K>` (기본 DEGREE=2, N=1, K=64).
- `gen_bmm_bias_iron_dbuf.py` — 위에서 XY 채널만 BATCH개 ping-pong으로 확장한 버전.
  `diag_bmm_bias_iron_dbuf.cpp`/`.exe`. `python3 gen_bmm_bias_iron_dbuf.py <DEGREE> <BATCH>
  <N> <K>` (기본 DEGREE=2, BATCH=2, N=BATCH, K=64).
- `gen_bmm_bias_iron_folded.py` — **최종/권장 구조**. bias를 별도 broadcast 없이 accumulator
  초기값으로 접어넣음(Method B), row당 채널 1개(X+Y+bias 합침 192x64 버퍼)만 사용.
  `diag_bmm_bias_iron_folded.cpp`/`.exe`. `python3 gen_bmm_bias_iron_folded.py <DEGREE> <N>
  <K>` (기본 DEGREE=2, N=1, K=64).
- (참고, 폐기됨) `gen_broadcast_dbuf_x.py` — ping-pong bias(이미 hang인 베이스라인) + X.
  실험 설계 결함으로 결과 무의미, 코드는 보존.

## 7. Method B를 실제 IREE 컴파일러에 적용 — 3개 버그 수정, 4번째에서 막힘 (같은 날, 계속)

§6의 "다음 세션 시작점 1번"(Method B를 실제 컴파일러 패치로 옮기기)을 같은 세션에서 바로 이어서
시작함. `linalg.fill(0)` → `linalg.fill(bias)` 로 직접 바꾸는 새 코드를 짤 필요 없이, **이미
존재하는 미사용 pass**를 찾음: `AMDAIEFoldBroadcastAddIntoDestPass`
(`AMDAIEFoldBroadcastAddIntoDest.cpp`, 2026-08-26에 구현) — `linalg.broadcast(bias) →
matmul(outs=broadcast)` 형태로 정확히 Method B를 구현하고 있었지만, 8/27 조사에서 밝혀진 대로
IREE 표준 pass `DetachElementwiseFromNamedOpsPass`가 9패스 뒤에 이걸 그대로 되돌려서
(`zero-fill → matmul → separate add`) 사실상 죽은 코드였던 상태.

**버그 A — FIXED (`third_party/iree`, `DetachElementwiseFromNamedOps.cpp`).** 그 되돌리는
로직 자체를 읽어보니: matmul의 `outs`를 만든 op가 `linalg.FillOp`일 때만 "이미 처리된 것"으로
스킵하고, 그 외 어떤 `LinalgOp`든(우리 pass가 넣은 `linalg.broadcast` 포함) 무조건 분리해버리는
구조였음. `isa<linalg::FillOp>` 조건에 `isa<linalg::BroadcastOp>`를 추가해서 broadcast로 만든
초기값도 fill과 동일하게 "건드리지 마라"로 스킵하도록 수정. `--compile-to=global-optimization`
로 검증: 이제 `linalg.batch_matmul outs(브로드캐스트 결과)`가 그대로 살아남고 별도 add가 다시
생기지 않음 — 최초 목표(되돌림 방지) 달성 확인.

여기서 끝이 아니었음 — 그 다음부터는 **AMD-AIE 자체 pack-peel 코드젠 파이프라인**이 이 새로운
모양(matmul의 dest가 fill이 아니라 broadcast인 경우)을 한 번도 겪어본 적이 없어서, bugs #1-8과
똑같은 종류의 "없는 케이스" 문제가 다시 하나씩 나타남.

**버그 B — FIXED (`AMDAIEBufferizeToAllocation.cpp` + 신규 유틸 `AMDAIEUtils.cpp`).**
`'linalg.generic' op failed bufferizing to allocations` 에러 발생. 원인: 이 pass의 elementwise
walk가 bias broadcast generic을 (matmul의 dest가 아니라) **독립된 elementwise 타겟**으로
오인해서, 그 generic의 raw 입력(bias의 `iree_tensor_ext.dispatch.tensor.load`)을 직접
`bufferizeToAllocation`하려다 실패(`outOfPlaceOperands.size()=0` — 순수 로드 op는 이 메커니즘
자체가 적용될 수 없음, upstream `ConvertToDestinationStyle.cpp:562-564`에 임시 디버그 프린트
넣어서 확인 후 다시 원복). **수정**: `isElementwiseFeedingContractionDest()` 유틸 함수 신설 —
generic의 결과가 `linalg.pack`/`linalg.copy`/`scf.for`/`scf.forall`(루프 캐리 값, `iterArg`)를
거쳐서 최종적으로 contraction의 dest에 도달하는지 재귀적으로 추적. 맞으면 `linalg.fill`과
동일하게 취급해서 walk가 독립 타겟으로 잡지 않게 스킵.

**버그 C — FIXED (`AMDAIEDistributeL1Allocations.cpp`).** 버그 B 수정 후 `'linalg.pack' op
needs logic implemented for handling.` 에러로 진행. 원인: 이 pass의 `TypeSwitch`가
`linalg::LinalgOp` 인터페이스를 구현하는 op만 처리하는데, `linalg.pack`/`linalg.unpack`은 그
인터페이스를 구현하지 않아서(bugs #3에서 `linalg.generic`/`linalg.fill`용으로 이미 일반화했던
것과 같은 종류의 누락) L1 alloc을 subview 없이 직접 읽는 pack이 `.Default` 케이스로 떨어짐.
`linalg::LinalgOp` 케이스와 동일한 오퍼랜드 교체 로직을 `PackOp`/`UnPackOp` 전용 케이스로 추가.

**버그 D — 발견, 아직 미해결. 오늘 세션은 여기서 중단.** 버그 C 수정 후 새 에러:
```
expected 'memref<1x1x4x1x1x8x8xf32>' for the packed domain value, got 'memref<1x1x1x1x1x8x8xf32>'
  linalg.pack(...) : (memref<1x1x4x8x8xf32,2>, memref<1x1x1x1x1x8x8xf32,2>) -> ()
```
shape 불일치. 정황상 원인: X/Y 오퍼랜드는 이 지점(코어별 `scf.forall (4,1)` 스레드 분할)에서
`memref.subview`(스레드 인덱스 기반 offset 포함)를 거쳐서 자기 몫만 슬라이스해서 pack에 넘기는
반면, bias의 broadcast→pack 체인은 그 subview 단계가 없어서 L1 alloc **전체**를 그대로 pack에
넘기고 있음 — 버그 C에서 추가한 "오퍼랜드를 통째로 newAlloc으로 교체"가 딱 이 경우엔 부족함
(같은 shape일 때만 맞는 방법이었는데, 여기선 스레드별로 더 쪼개야 함). 아직 어느 pass가 X/Y에는
이 subview를 넣어주면서 bias(우리가 새로 접어넣은 accumulator init)에는 안 넣어주는지 못 찾음 —
다음 세션에서 이어서 추적 필요.

**커밋 완료** (오늘 마무리, `bert` 브랜치): iree 서브모듈에 버그 A 커밋(`d9e3447`,
"Don't let DetachElementwiseFromNamedOps undo a broadcast-folded accumulator init"), 메인
레포에 버그 B/C 커밋(`4c3bd22`, "[AMD-AIE] Follow a broadcast-folded accumulator init through
pack/loop lowering" — 서브모듈 포인터 bump 포함). 둘 다 `iree-compile` 재빌드해서 각 단계별로
검증 완료. 실제 HW 실행은 아직 안 해봄 (컴파일 자체가 버그 D에서 막혀 있어서 바이너리가 안 나옴).
디버깅용으로 upstream `llvm-project`(`ConvertToDestinationStyle.cpp`)에 넣었던 임시 프린트는
커밋 전에 원복 완료 — 서브모듈 포인터 변경 없음.

**사용자 판단**: 이건 버그가 아니라 "새로 만든 경로라 하나씩 채워야 하는 정상적인 과정"("없는
걸 만들어 냈으니 하나씩 추가해야 하는거잖아") — bugs #1-8 때와 동일한 패턴이 반복되는 것으로
받아들이고 계속 진행하기로 함. 예상보다 라운드가 걸릴 수 있음.

## 다음 세션 시작점

1. **버그 D 계속 추적** — 코어 스레드 분할(`scf.forall (4,1)`) 시 X/Y는 어느 pass에서
   `memref.subview`로 자기 몫만 슬라이스받는지 찾아서, bias의 pack 체인도 같은 처리를 받게
   만들기. `AMDAIEDistributeL1Allocations.cpp`의 `getDistributedType`/subview-탐색 로직부터
   보는 게 자연스러운 다음 지점.
2. 버그 D까지 풀려서 `bmm_bias_repro_N8.mlir`이 실제로 `.vmfb`까지 컴파일되면, **Method B가
   실제 컴파일러 경로에서도 hang을 없애는지** 실제 HW로 최종 검증 (지금까지는 전부 IRON
   재현이었고, 실제 IREE 경로에서의 검증은 아직 없음 — 이게 이 전체 작업의 원래 목적임을 잊지
   말 것).
3. memtile output 집계 팩터-2 미스터리 (여전히 미해결)
4. core ELF의 정확한 compute 루프 반복 횟수 (여전히 미해결)
