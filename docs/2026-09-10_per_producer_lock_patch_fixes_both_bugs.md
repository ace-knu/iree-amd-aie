# 2026-09-10 — 이분 탐색 결론: batch-0 을 고친 것은 커밋이 아니라 per-producer lock 패치였다

## 0. 한 줄 요약

`1600078` 의 delay 를 **끄고** `AMDAIEObjFifoBufferization` 의 **per-producer lock 패치를 켜면**,
이 조사에서 나온 batch-0 all-zero 와 batched-matmul lane 오답이 **둘 다 동시에** 사라진다.
chainmix L=12 전체가 CPU 기준 대비 `maxerr = 0`, 10/10 실행 완전 동일, `corr = 1.000000`.

## 1. 출발 질문

> "어느 커밋이 고쳤는지 이분 탐색 해줘. 그리고 이거 내가 바꾼건가?"

전제: 8/24~8/25 에 관측된 batch-0 all-zero 가 오늘(9/10) HEAD 에서는 재현되지 않았다.
그러니 그 사이 어떤 커밋이 우연히 고쳤을 것이다 — 라는 가정으로 이분 탐색을 시작했다.
**이 전제가 틀렸다.**

## 2. 이분 탐색 결과 — 커밋은 아무것도 고치지 않았다

재현체: `_local/int8_debug/b0repro/bmm_int8.mlir` (8/24 원본 `gen_and_quant_bmm.py` 를 그대로
재실행해 만든 것, ONNX 가 원본과 **바이트 동일**). `B,M,K,N = 2,16,16,64`, 두 피연산자 모두 런타임 입력.
모든 스텝에서 delay(`1600078` 블록)는 제거한 상태로 측정.

| 플러그인 상태 | 날짜 | vmfb 크기 | batch 0 (6회) |
|---|---|---|---|
| `40654aa` | 8/25 | 81031 | **ALL-ZERO 5/6** + 부분오염 1/6 |
| `552c8c2` (= `ee37c48` + 빌드픽스) | 8/26 | 81031 | **ALL-ZERO 6/6** |
| `4c3bd22` | 9/02 | 81031 | **ALL-ZERO 6/6** |
| `85afac9` (= HEAD 의 플러그인 상태) | 9/03 | 81031 | **ALL-ZERO 6/6** |
| HEAD + **per-producer lock 패치** | (미커밋) | 83911 | **정상 12/12** |

`85afac9` 이후 플러그인을 건드린 커밋은 없다(`866ce6e` 는 문서만). 즉 **git 상 HEAD 의
플러그인 상태 = `85afac9`** 이고, 그 상태는 여전히 batch-0 을 재현한다.

"HEAD 에서는 재현 안 된다"고 봤던 그 바이너리(`b0_off.vmfb`, 83911)는 순수 git HEAD 가 아니라
**내 워킹트리 패치가 들어간 빌드**였다. 패치를 복원해 다시 빌드하니 나온 vmfb 가
`b0_off.vmfb` 와 **md5 바이트 동일**(`d6e1cd04…`) — 출처가 확정됐다.

## 3. "이거 내가 바꾼건가?" → 아니다

`ee37c48` / `4c3bd22` / `85afac9` 는 전부 wjjang 커밋이지만, batch-0 을 **고치지도 만들지도** 않았다.
셋 다 위 표에서 ALL-ZERO 6/6 으로 재현한다.

- `ee37c48` (8/26, bias fusion WIP): batched-matmul 경로 4개 파일을 실제로 바꾸긴 했지만
  이 재현체의 vmfb 크기는 `40654aa` 와 같은 81031 이고 증상도 그대로다.
  (오히려 `40654aa` 의 5/6 → 6/6 으로 **더 결정론적**이 됐다.)
- `4c3bd22`, `85afac9`: 둘 다 `AMDAIEDistributeL1Allocations.cpp` 를 건드리지만,
  새 경로는 `oldAllocIsDest && !insideCoreForall` (broadcast-fold 된 bias 누산기) 에서만
  발동하므로 bias 없는 이 재현체에는 걸리지 않는다.

즉 **8/24 의 batch-0 관측은 정확했고, 지금까지 한 번도 고쳐진 적이 없었다.**

## 4. 진짜 원인·진짜 수정: per-producer lock

패치 위치: `AMDAIEObjFifoBufferization.cpp` (+ `AMDAIELowerToAIE.{h,cpp}` 의 BD 분리)

원래 코드는 producer 가 여러 개여도 lock 쌍을 **1개만** 만들고,
초기값을 `numProducers * depth` 로 미리 충전한다:

```cpp
int8_t producerLockInitValue =
    copyLikeProducers.size() >= copyLikeConsumers.size()
        ? copyLikeProducers.size() * depth
        : copyLikeConsumers.size() * depth;
```

이러면 consumer 의 `AcquireGreaterEqual` 이 **어느 producer 가 몇 번 릴리즈했는지 구분하지 못한다.**
producer A 가 자기 몫을 두 번 릴리즈하면 아직 아무것도 쓰지 않은 producer B 의 슬롯까지
"준비됨"으로 보이고, consumer 는 B 의 버퍼(=초기화 안 된 메모리 또는 0)를 그대로 읽어간다.
batched matmul 에서 이 B 가 batch 0 이면 → **batch 0 이 통째로 0**.

패치는 producer 가 여러 개이고 consumer 가 하나일 때 lock 쌍을 **producer 당 하나씩** 만든다:

```cpp
lockPairPerProducer = numProducers > 1 && numConsumers == 1;
numLockPairs        = lockPairPerProducer ? numProducers : 1;
producerLockInitValue = lockPairPerProducer ? depth : (기존 식);
```

consumer 전송도 party 별 BD 로 쪼개서, 각 producer 의 릴리즈가 자기 슬롯만 열도록 한다.

## 5. 검증 — 세 모델 모두 통과 (패치 ON, delay OFF)

| 모델 | 형상 | 결과 |
|---|---|---|
| `b0repro/bmm_int8` (8/24 원본 재현체) | B=2, M=16, K=16, N=64 | batch 0 **정상 12/12**, 출력 1종 |
| `bmmlane_b12_m32_k64_s0` | HEADS=12, M=32, HD=64 | **maxerr = 0** (비트 완전 일치) 6/6, 틀린 head 없음 |
| `chainmix_L12_H12` (24 디스패치, D=768) | L=12 전체 | **maxerr = 0**, 틀린원소 0/24576, `corr = 1.000000`, 10/10 완전 동일 |

`chainmix_L12_H12` 는 이 조사의 실제 목표 형상이다. 오차가 "작다"가 아니라 **정확히 0** 이고,
실행간 편차도 0 이다.

## 6. delay(`1600078`)는 이제 해로운 쪽이다

- delay 는 batch-0 을 **우회**만 했다 (락 구분 문제를 고치지 않고 타이밍으로 가렸다).
- 그리고 규모가 커지면 스스로 오답을 만든다: delay ON 이면 `bmm` 12개 전부 오답
  (0.12–0.45), `col%8==0` 위치에 집중. delay OFF 면 12/12 정확.
- 반면 per-producer lock 패치는 batch-0 을 **원인에서** 고치고, delay 없이도 lane 오답이 없다.

→ **권고: delay 를 제거하고 per-producer lock 패치를 커밋한다.**
현재 워킹트리는 delay 를 `AMDAIE_DISABLE_BATCH0_DELAY` 환경변수로 끌 수 있게 해 둔 상태다
(기본값은 여전히 delay ON). 기본값 전환/제거는 커밋 여부와 함께 결정할 사안.

## 7. 앞선 "패치 반증" 판정과의 관계

9/08 에 이 패치를 "반증됨"으로 기록했는데, 그건 **버그 ①**(옛 아티팩트 `proj2`/`proj8` 의
core-store vs DMA-read 경합)에 대한 반증이었다. 그 판정은 그대로 유효하다 —
패치를 켜도 옛 아티팩트의 5개 기준 해시가 그대로 재현됐다.

이번에 밝혀진 것은 **같은 패치가 다른 버그(batch-0 all-zero)의 진짜 수정**이라는 점이다.
두 건은 별개 증상이고, 패치는 그중 하나만 고친다. 버그 ① 은 여전히 열려 있다.

## 8. 부수적으로 고친 것

`bisect_step.sh` 가 스텝 1 에서 출력을 잃은 원인:

```bash
set -e
build.sh | grep -E "^FAILED|error:"   # 빌드 성공 → grep 매치 0건 → rc=1 → set -e 로 즉시 종료
```

빌드가 **성공했을 때만** 죽는 함정이었다. `set -uo pipefail` + 명시적 rc 검사로 교체했다.

## 9. 커밋 완료 (`65069ce`)

`bert` 브랜치에 커밋했다(미푸시). 내용:

- `AMDAIECoreToStandard.cpp` 를 `1600078^` 로 되돌려 **delay 완전 제거** (−88 줄).
  내가 조사용으로 넣어둔 실험 블록(`AMDAIE_DISABLE_BATCH0_DELAY`,
  `AMDAIE_EXPERIMENTAL_PRE_RELEASE_{DELAY,NOPS,BARRIER}`)도 같이 사라졌다.
- `AMDAIEObjFifoBufferization.cpp`: producer 당 lock 쌍.
- `AMDAIELowerToAIE.{h,cpp}`: consumer 전송을 producer 슬라이스별 BD 로 분할 +
  분할이 원본과 같은 바이트·같은 순서인지 컴파일 타임 검증.

### 커밋 상태에서의 최종 검증

delay 를 소스에서 제거했으므로 환경변수 없이 컴파일했고, 나온 vmfb 3개가 앞서 검증한
`AMDAIE_DISABLE_BATCH0_DELAY=1` 빌드와 **md5 바이트 동일**이었다 — delay 제거가
코드 생성에 아무 영향이 없음을 확인(즉 5절의 검증 결과가 그대로 유효).

| 모델 | 실행 | 결과 |
|---|---|---|
| `b0repro/bmm_int8` | 10/10 | batch 0 정상, 출력 1종 |
| `bmmlane_b12_m32_k64_s0` | 10/10 | **max = 0**, 틀린 head 없음, 출력 1종 |
| `chainmix_L12_H12` | 10/10 | **maxerr = 0**, 0/24576, corr 1.000000, 출력 1종 |

회귀 테스트 `obj_fifo_bufferization.mlir`, `lower_to_aie.mlir` 통과.
