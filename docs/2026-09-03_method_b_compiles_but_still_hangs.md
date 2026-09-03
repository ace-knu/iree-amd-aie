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

**다음 세션 시작점**:
1. bias 경로가 원인이 아니라면, 남은 유력 후보는 X/Y 자체의 전달 구조(실제 ping-pong,
   K-tile 언롤, 실제 컨트롤 코드) — §8/31에서 "plain matmul+bias(배치 없음)도 CDO
   byte-identical하게 hang"이 나왔던 것과 일치하는 방향. bias를 완전히 제거한(순수 matmul만)
   경우도 hang나는지 재확인해볼 가치 있음(`bmm_pure_repro.vmfb`가 이미 있으나 shape이
   달라서 이번엔 안 씀 — N8 shape으로 순수 matmul만 있는 버전 필요).
2. 이전에 스코프만 해두고 안 만든 HW trace tooling(`docs/2026-08-27_matmul_bias_fusion_hang_root_cause_refined.md`)이
   이제 훨씬 더 필요해 보임 — 정적 분석/구조적 재현만으로는 한계에 도달한 것으로 보임.
3. 컴파일러 버그 D/E 커밋 완료 필요(아직 안 함 — 세션 종료 전 커밋할 것).

## 커밋 상태

버그 A, B, C는 이미 커밋됨(`d9e3447`, `4c3bd22`). 버그 D, E는 이번 세션에서 작업했고 아직
**미커밋** — `AMDAIEDistributeL1Allocations.cpp` 변경사항.
