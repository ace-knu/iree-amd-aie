# Method B(accumulator-fold) 실제 컴파일러 경로 완주, 그러나 실제 HW hang은 그대로 (2026-09-03)

[docs/2026-09-02_bmm_bias_iron_xy_operands.md](2026-09-02_bmm_bias_iron_xy_operands.md) §7의
연장. bugs D, E를 마저 고쳐서 실제 IREE 컴파일러가 `bmm_bias_repro_N8.mlir`을 끝까지 컴파일하는
데 성공했지만, 실제 NPU에서 돌려보니 **hang이 그대로 재현됨**. matmul+bias 퓨전 hang 조사
(로드맵 아이템 2)의 핵심 가설을 뒤집는 결과.

## 버그 D — FIXED: pack 대신 단일 generic으로 직접 계산

전날 §7에서 막혔던 지점: bias accumulator(`%alloc`)를 만드는 마지막 pack이 `scf.forall` 밖에서
"4개 스레드 분량 전체"를 한 번에 만든 뒤 안에서 슬라이스하는 구조라, "operand만 바꿔치기"로는
pack의 source/dest 크기 정합성이 깨졌음(`AMDAIEDistributeL1Allocations.cpp`).

여러 단계로 원인을 좁힘:
1. **1차 시도(subview 재사용)** — 실패: `newAlloc`이 identity layout인데 subview 결과 타입은
   strided layout이라 타입 비교가 항상 실패. shape/element-type만 비교하도록 수정.
2. **처리 순서 문제** — K-tile 언롤로 생긴 2개의 후보 forall 중 아무거나 골라서 잘못하면 두
   번째 K-chunk 앞에 잘못 꽂혀 이미 누적된 부분합을 bias로 리셋할 위험. `isBeforeInBlock`으로
   "가장 먼저 실행되는" forall을 고르도록 수정.
3. **memref는 `getDefiningOp()`가 실제 writer가 아님** — buffer semantics에서 pack/generic은
   결과값이 없는 side-effect 연산이라, `value.getDefiningOp()`는 그냥 `memref.alloc()`을
   가리킴. writer는 "이 값을 DPS init으로 쓰는 연산"을 use-list에서 찾아야 함. 크래시
   (`cast<DestinationStyleOpInterface>` 어서션 실패)로 발견, 수정.
4. **타일된 차원 처리 누락** — 첫 번째 pack은 M(32)을 4개 그룹으로 **타일링해서** dim2(=4)를
   만드는데, narrowing 로직이 "타일된 차원은 무조건 원본 그대로 유지"로 잘못 처리해서
   32를 그대로 둠(8이어야 함). 타일된 차원도 "narrow 대상이면 타일 크기만큼만 남기기"로 수정.

**최종 구조 변경**: "pack 체인을 그대로 복제"가 아니라, **pack들의 인덱스 대수(affine
map)를 합성해서 broadcast generic 하나로 직접 최종 shape에 쓰는** 방식으로 전면 재설계
(`composeThroughPack` + `buildDirectComputationNarrowed`). 이렇게 한 이유는 버그 E에서 밝혀짐.

## 버그 E — FIXED: L1-to-L1 DMA는 지원 안 됨 → pack 자체를 없애야 했음

버그 D를 pack 복제 방식으로 처음 고쳤을 때 새 에러: `'amdaie.dma_cpy_nd' op dma op with both
source and target on L1 is not supported` (`AMDAIEInsertCores.cpp`). 원인 추적:
`AMDAIEConvertToDma.cpp`가 **모듈 전체의 모든 `linalg.pack`/`unpack`을 무조건 DMA로 변환**함 —
스레드 forall 안에 있는지, 메모리 공간이 뭔지 전혀 안 가림. 우리가 새로 만든 2단계 pack 체인은
둘 다 L1(코어 로컬)이라, forall 안에 들어가는 순간 "L1→L1 DMA"로 변환되고 `AMDAIEInsertCores`가
이걸 거부함(코어가 자기 자신에게 DMA를 걸 수는 없으므로).

기존 아키텍처에서 이 pack이 항상 **forall 밖(공유 prelude)**에 있었기 때문에 이 검사(포럴 안만
훑음) 대상에 아예 안 걸렸던 것 — 즉 원래도 잠재적으로 말이 안 되는 구조였지만, forall 밖에
있어서 한 번도 걸린 적이 없었을 뿐. bias 값이 스레드에 의존하지 않는다는 점(broadcast라 M을
전혀 안 읽음)을 이용해서, pack 두 개를 거치는 대신 **pack들의 인덱스 변환을 역으로 합성한
affine map 하나로 원본 bias generic을 최종 shape에 직접 다시 실행**하는 식으로 전면 재설계함
(위 "버그 D 최종 구조 변경"과 동일). 이러면 pack 자체가 아예 안 생겨서 DMA 변환 대상이 없음.

**부가 문제**: 이 재설계로 원래(가짜) pack 체인이 완전히 안 쓰이게 됐는데, "죽은 코드로 남겨두면
나중 DCE가 치울 것"이라는 이 파일의 기존 관례를 그대로 따랐다가 또 다른 에러 발생:
`'amdaie.flow' op with no source channel is unsupported`. 원인: `AMDAIEConvertToDma`가 **죽은
pack도 살아있는 pack과 똑같이 무조건 DMA로 변환**하기 때문에, 아무 데도 연결 안 된 고아
L1-to-L1 DMA/connection이 만들어지고 나중 단계에서 검증 실패함. **해결**: 죽은 pack 체인을
`eraseDeadPackChain`으로 명시적으로 삭제(원본 generic/fill 같은 non-pack dead code는 여전히
방치해도 안전 — `AMDAIEConvertToDma`가 pack/unpack만 건드리므로).

## 컴파일 성공

위 두 버그를 다 고친 뒤 `bmm_bias_repro_N8.mlir` → `.vmfb` **컴파일 완전히 성공**
(`COMPILE EXIT CODE: 0`, CPU 타겟 관련 경고 하나만 있음). CDO 생성도 정상 확인.
`_local/int8_debug/out/bmm_bias_repro_N8_final.vmfb`.

이로써 오늘/전날 세션에서 고친 컴파일러 버그 요약(모두 `bert` 브랜치에 커밋됨):
- 버그 A: `DetachElementwiseFromNamedOps.cpp` (iree 서브모듈, 커밋 `d9e3447`)
- 버그 B, C: `AMDAIEBufferizeToAllocation.cpp` + `AMDAIEUtils.cpp` (메인 레포, 커밋 `4c3bd22`)
- 버그 D, E: `AMDAIEDistributeL1Allocations.cpp` (이번 세션, 아직 미커밋 — 아래 "커밋 상태" 참고)

## 실제 HW 검증 — **hang이 그대로 재현됨** (핵심 가설 반증)

`with-npu-lock.sh`로 안전하게 감싸서 실행:
```
build/tools/iree-run-module --device=amdxdna --device=local-task \
  --module=bmm_bias_repro_N8_final.vmfb --function=bmm_bias \
  --input=@bmm_bias_N8_x.npy --input=@bmm_bias_N8_y.npy \
  --output=@final_verify_out.npy
```
**결과: `ert state 8`(TDR 타임아웃), 2분 4.6초.** 정상 baseline(100~140ms)의 1000배 — 명백한
hang, §6까지의 그 hang과 완전히 동일한 시그니처. dmesg 확인: `aie2_set_cmd_timeout` 에러 딱
1번만 발생, SMU/FLR 연쇄 없음 — NPU는 정상 자동 복구됨, 장비 이상 없음.

**이 결과가 뒤집는 것**: 8/28~9/2까지 쌓아온 핵심 가설 — "bias의 별도 broadcast 전달 방식
(1채널 + 다중 consumer + 공유 credit-lock + ping-pong 조합)이 hang의 원인"이었음. Method B는
**그 구조를 통째로 없앴다** (bias 전용 broadcast connection/공유 lock 자체가 이제 존재하지
않음 — bias는 이제 그냥 각 코어가 스스로 계산하는 로컬 값). 그런데도 hang이 똑같이
재현된다는 것은:

1. IRON 재현(§5, `gen_bmm_bias_iron_folded.py`)에서 Method B가 PASS했던 것은 **진짜 실제
   dispatch의 hang 메커니즘과 무관한, 우연히 안전했던 축소 모델**이었을 가능성이 높음 — IRON
   재현은 항상 실제 dispatch보다 단순화된 부분(예: 실제 K-tiling 언롤 구조, 실제 X/Y ping-pong
   세부사항, 실제 컨트롤 코드 생성 경로)이 있었고, 그 차이 어딘가에 진짜 원인이 있었다는 뜻.
2. 즉 bias 경로는 (8/31 발견대로) **원인이 아니라 상관관계였을 가능성**이 커짐 — 지금까지의
   모든 정적 분석(락/BD/큐/lowering)과 IRON 격리 실험들이 "진짜 원인"이 아닌 곁가지를
   조사했을 수 있음.

## 사용자 판단 및 다음 방향

이번 결과로 "컴파일러 버그 몇 개 고치면 hang이 풀린다"는 낙관적 시나리오는 확정적으로
배제됨. 지금까지 고친 컴파일러 버그들(A-E) 자체는 **독립적으로 유효하고 유용한 수정**이며(각각
실제 존재하는 결함이었고, Method B 자체를 컴파일 가능하게 만들었음), 계속 유지할 가치가 있음 —
다만 **"matmul+bias 퓨전 hang을 고친다"는 원래 목적은 달성하지 못함**.

## 순수 matmul(bias 완전히 없음, 동일 N8 shape) — 재현 안 됨

`_local/int8_debug/gen_bmm_pure_smallN.py` 작성(bias/add 없이 MatMul 노드 하나만, B=2,M=32,
K=64,N=8 — bias 버전과 완전히 같은 shape). **표준 파이프라인 그대로**(컴파일러 패치 전혀 필요
없음 — 애초에 accumulator-fusion 경로를 안 타므로 이번 세션 수정과 무관) `--iree-amdaie-
packet-flow-strategy=inputs` 플래그 **없이도** 컴파일 성공. 실제 HW 실행: **PASS, 0.323초,
exit 0.** matmul+bias(어떤 아키텍처든)는 전부 hang나고, bias 없는 순수 matmul만 정상 — 명확한
대조.

## CDO 직접 비교로 진짜 차이 특정 — bias가 아니라 Y가 문제였음

`cdoutil -output-source -rewrite-sequential`로 세 가지의 memtile BD를 직접 디코딩해서 비교
(`decode_bd.py` 재사용):
- **순수 matmul (정상)**: BD 18개 전부 `en_pkt=0` — packet-mode 자체가 하나도 없음. 각 lock쌍은
  BD 2개(더블버퍼용 ping-pong)에 acq/rel=1인 평범한 circuit 연결.
- **Method B 컴파일 결과 (hang)**: BD 30개, 그중 **16개가 packet-mode**. base
  `0x030000`/`0x034000`(ping-pong 2버퍼) 그룹이 lock64/65(fetch acq124/rel4, consumer
  acq127/rel1)로 **4개 목적지 전부 같은 pkt_id=0**을 씀 — 9/1에 `gen_broadcast_dbuf.py`로 hang
  확정했던 바로 그 "fan-out + 공유 credit-lock + ping-pong" 패턴과 100% 일치. shape상 이건
  **bias가 아니라 Y**임(Y=[K,N]은 M에 안 걸리므로 M-tile 4개 코어에 broadcast 필요 — bias를
  Method B로 로컬화해도 Y의 이 broadcast는 원래부터 있었음, 한 번도 의심 안 했을 뿐).

**메커니즘**: 코어 타일의 circuit 채널 예산은 한정적(X,Y 2개면 충분, packet-flow 불필요 —
순수 matmul이 증거). bias가 끼면(어떤 아키텍처든) 채널 수요가 예산을 넘어서 컴파일러가
`--iree-amdaie-packet-flow-strategy=inputs`를 강제로 써야 컴파일이 됨 — 이 packet-flow
모드의 구현 방식이 "여러 목적지 → 공유 credit-lock + ping-pong" 패턴이고, **이게 실제
hang 트리거**. 즉 "bias냐 아니냐"가 아니라 **"3번째 텐서급 채널 수요가 생겨서 packet-flow로
강제 전환되느냐"**가 진짜 변수.

## Method A(reduction 축에 bias 접합) 시도 — 평범한 circuit 멀티캐스트로도 여전히 HANG

사용자 제안: bias를 accumulator init이 아니라 **reduction 축**에 붙이기(X_aug=[X|1] shape
[M,K+1], Y_aug=[Y;bias] shape [K+1,N] — X_aug·Y_aug = X·Y + 1·bias). 컴파일러 관점에서
텐서가 X_aug, Y_aug **2개**뿐이라 순수 matmul과 동일하게 packet-flow 없이 컴파일될 것이라는
가설.

**IRON에서 먼저 검증** (사용자 지시대로): `_local/mlir_aie_repro/2026-08-28_column_threshold/
gen_bmm_augmatmul_iron.py` 작성 — X_aug는 행마다 전용 채널(distribute), Y_aug는 **순수 matmul이
실제로 쓰는 것과 동일한 방식**(순수 `aie.flow`를 같은 소스 채널에서 여러 개 선언해서 만드는
circuit 멀티캐스트, packet_flow 전혀 없음)으로 4개 행에 ping-pong 전달, 진짜 K+1 reduction
수행. `aiecc` 컴파일 클린.

**실제 HW 실행: HANG.** `run state: 8`, 63.3초, 목적지 값 전부 0(도달 못함). 순수 circuit
멀티캐스트로 만들어도 여전히 hang — "packet-mode만 위험하다"는 가설이 틀렸거나(순수 circuit
멀티캐스트 자체가 여러 독립 코어에게는 원래 위험할 수 있음), 이 손으로 짠 IRON 스크립트에
버그가 있어서(진짜 컴파일러가 만드는 circuit 멀티캐스트에는 있는 동기화 장치를 안 넣었을
수 있음) 만든 hang일 가능성 둘 다 열려있음 — **결론 미확정**.

## 안전 사고: 이번 라운드 2연속 hang + 평소와 다른 복구 에러, 확인 후 정상 복구 확인

`timeout 70`으로 먼저 죽인 시도(RUN_EXIT=124, 출력 없음)와 그 다음 `timeout 180`으로 제대로
관찰한 시도(`run state 8`) — dmesg 확인 결과 **TDR이 2번**(17:20:59, 17:22:38) 찍힘, 즉 둘 다
진짜 hang이었음(2연속 hang 규칙 위반). 게다가 두 번째 이후 **평소와 다른 에러**:
```
aie2_config_cu: Lookup GEM object failed
aie2_hwctx_restart: Config cu failed, ret -22
aie_send_mgmt_msg_wait: command opcode 0x106 failed
aie2_hwctx_restart: Map host buf failed, ret -22
```
이 조사 내내 봤던 "TDR 한 번 뜨고 깔끔하게 자동 복구"와 다른 패턴이라 **라이브 HW 테스트
즉시 전면 중단**. 이후 `sudo xrt-smi examine`으로 디바이스 정상 인식/응답 확인, 이미
PASS 확인됐던 `diag_bmm_constacc.exe`를 다시 돌려서 **완전히 정상 동작 확인**(`run state 4`,
0.0015초, 정답). **NPU는 최종적으로 정상 복구됨**, 세션 종료 시점 기준 이상 없음. 다만 위
"Config cu failed"류 에러가 새로 관찰된 만큼, 다음 세션에서도 시작 전 상태 확인은 평소보다
한 번 더 꼼꼼히 할 것.

## 재현 자료 (모두 `_local/`, gitignored, 커밋 안 됨)

- `_local/int8_debug/gen_bmm_pure_smallN.py` — bias 없는 순수 matmul ONNX 생성기
  (`python3 gen_bmm_pure_smallN.py <N>`).
- `_local/int8_debug/out/bmm_pure_repro_N8.{onnx,mlir,vmfb}` — 컴파일된 순수 matmul (PASS 확인됨).
- `_local/int8_debug/out/final_dump/`, `pure_n8_dump/` — 각각 Method B/순수matmul의 CDO 덤프.
- `_local/mlir_aie_repro/2026-08-28_column_threshold/gen_bmm_constacc_iron.py` —
  accumulator를 컴파일타임 상수로 시드 + 실전용 채널(전용, non-broadcast). **PASS**
  (`diag_bmm_constacc.exe`).
- `_local/mlir_aie_repro/2026-08-28_column_threshold/gen_bmm_augmatmul_iron.py` — Method A,
  Y_aug를 순수 circuit 멀티캐스트로 전달. **HANG** (`diag_bmm_augmatmul.exe`). 원인 미확정
  (진짜 HW 한계 vs 스크립트 버그).
- `/tmp/real_cdo_init2.src.txt`, `/tmp/iron_folded_cdo2.src.txt`, `/tmp/pure_cdo_init.src.txt`
  — `cdoutil -output-source -rewrite-sequential`로 뽑은 각 CDO의 decode_bd.py 입력용 텍스트
  (임시 경로, 세션 종료 시 사라짐 — 재현하려면 위 cdoutil 명령 다시 실행).

## 다음 세션 시작점

1. **Method A(circuit 멀티캐스트) hang의 원인 규명이 최우선.** 두 가능성을 갈라야 함:
   (a) 진짜 하드웨어가 "여러 독립 코어에게 순수 circuit 멀티캐스트"를 못 버티는 것이라면
   Method A도 폐기하고 완전히 다른 접근 필요. (b) `gen_bmm_augmatmul_iron.py`의 손으로 짠
   동기화 로직에 버그가 있는 것이라면(예: 순수 matmul이 실제로 쓰는 코어 실행 순서/컨트롤코드
   동기화를 안 넣었을 수 있음), 그걸 고쳐서 재검증. 후자를 먼저 의심하는 게 나을 듯 —
   pure matmul 자체가 이미 "circuit 멀티캐스트+ping-pong은 정상 작동"의 실증 사례이므로,
   메커니즘 자체가 위험하다기보다 재현이 부정확했을 가능성이 더 높음. `bmm_pure_repro_N8`의
   실제 lower-to-aie IR(`_local/int8_debug/out/pure_lower_to_aie.log`)을 훨씬 더 꼼꼼히
   따라가서 컨트롤코드/락 프로토콜의 정확한 세부사항을 다시 맞춰볼 것.
2. (a)로 판명나면: HW trace tooling(`docs/2026-08-27_matmul_bias_fusion_hang_root_cause_refined.md`)
   투자가 사실상 유일한 다음 수. (b)로 판명나서 Method A가 IRON에서 PASS하면: 실제 IREE
   컴파일러에 X_aug/Y_aug 구현(예: linalg 레벨에서 K차원 padding+concat 프리프로세싱 패스) →
   real HW 최종 검증.
3. 세션 시작 시 평소보다 한 번 더 dmesg/xrt-smi로 NPU 상태 확인할 것(이번 세션 끝에 평소와
   다른 복구 에러가 있었으므로).

## 커밋 상태

버그 A, B, C, D, E 전부 커밋 완료(`d9e3447`, `4c3bd22`, `85afac9`, 전부 `bert` 브랜치, 미push).
이번 세션의 IRON 실험/CDO 비교/Method A 시도는 전부 `_local/`(gitignored) 안에만 있어서 커밋
대상 없음 — 다음 세션에서 이어가려면 위 "재현 자료" 목록 참고.
