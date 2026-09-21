# memtile 출력 gather의 multi-producer 공유 lock 결함 — 2026-09-08 (Opus 재검토 세션)

이 문서는 "BERT-base 두 번째 버그(dispatch-type-transition 비결정론)"를 **처음부터 다시**
생각해보자는 요청으로 시작한 세션의 기록이다. 같은 날 앞서 작성된
`docs/2026-09-08_int8_vectorized_pipeline_full_pass_audit.md`(pass별 감사)와는 별개 세션이고,
결론도 다른 층위다 — 그 문서가 "정적 코드 감사는 소진됐다"로 끝났는데, 이 문서는 **그 감사가
놓쳤던 구조적 결함을 실제로 하나 찾았다**.

배경 요약: int8 양자화 성공 → batch matmul에서 batch 0 출력이 0으로 나오는 버그(#1) →
lock 첫 release 뒤 delay 삽입으로 우회 해결(`1600078`) → 그 방식으로 BERT를 돌리니
특정 L에서 실행마다 다른 출력이 나오는 버그(#2). #2는 delay로 안 고쳐지고, L 의존성이
불규칙(L=10,12,13 재현 / L=11,14,16 clean)해서 미해결로 남아 있었다.

---

## 1. 핵심 발견 — `AMDAIEObjFifoBufferization`의 multi-producer 공유 counting semaphore

`AMDAIELowerToAIE` 출력 IR을 직접 읽어서 확인한 구조 (`bmm12_int8` 및 실제 `chainmix_L12`,
컬럼 0 기준. 8개 컬럼 전부 동일):

```
buff_4, buff_5 : memref<256xi8, 1>   ← 물리 버퍼 2개 (핑퐁, depth=2)
lock_4 (free)  : init = 8
lock_5 (ready) : init = 0

aie.memtile_dma(%tile_0_1) {
  S2MM ch2 ← core tile_0_2 : acquire free>=1; dma_bd(buff, len=64, offset=0);   release ready+=1
  S2MM ch3 ← core tile_0_3 : acquire free>=1; dma_bd(buff, len=64, offset=64);  release ready+=1
  S2MM ch4 ← core tile_0_4 : acquire free>=1; dma_bd(buff, len=64, offset=128); release ready+=1
  S2MM ch5 ← core tile_0_5 : acquire free>=1; dma_bd(buff, len=64, offset=192); release ready+=1

  MM2S ch2 → tile_0_0(shim, 호스트 출력) : acquire ready>=4; dma_bd(buff, len=256); release free+=4
}
```

flow로 producer 신원을 확인한 결과(8컬럼 전부): **row 2,3,4,5의 서로 다른 코어 4개**가
각자 독립 S2MM 채널로 **하나의 256B 버퍼의 서로 다른 64B slice**에 쓰고, **lock 1쌍을 공유**한다.

### 결함의 정확한 진술

`AcquireGreaterEqual(ready, 4)`는 **"release가 4번 있었다"만 표현할 수 있고,
"서로 다른 4개 producer가 같은 버퍼 세대에 대해 release했다"는 것을 확인할 수 없다.**
카운팅 세마포어에는 producer 신원도 버퍼 세대 정보도 없다.

깨지는 시나리오 (코어 2개가 각각 1 이터레이션만 앞서 나가면 충분):

| 이벤트 | ready | consumer |
|---|---|---|
| core 0_2 iter k 완료 (buff_4 slice0) | 1 | 대기 |
| core 0_3 iter k 완료 (buff_4 slice1) | 2 | 대기 |
| core 0_2 iter k+1 완료 (buff_5 slice0) | 3 | 대기 |
| core 0_3 iter k+1 완료 (buff_5 slice1) | 4 | **성립 → buff_4 256B 전송** |

이때 `buff_4`의 slice2/slice3(core 0_4, 0_5 담당)은 아직 안 쓰였다. 첫 패스면 초기화된 0,
이후엔 이전 이터레이션 잔여 데이터가 그대로 호스트로 나간다.

`free` 쪽도 과잉 구독이다 — 버퍼 2개인데 credit 8개라, **한 코어가 혼자 BD 이터레이션 4번
(=핑퐁 2바퀴)을 앞서 나가** consumer가 읽고 있는 버퍼를 덮어쓸 수 있다.

### 컴파일러 코드 위치

`compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEObjFifoBufferization.cpp`:

```cpp
if (copyLikeConsumers.size() > 1 && copyLikeProducers.size() > 1) {
  logicalObjFifo.emitOpError()
      << "has a multi-producer, multi-consumer DMA "
         "pattern, which is currently not supported";   // ← 둘 다 다수면 거부
  return WalkResult::interrupt();
}
int8_t consumerLockInitValue{0};
int8_t producerLockInitValue =
    copyLikeProducers.size() >= copyLikeConsumers.size()
        ? copyLikeProducers.size() * depth      // ← 4 producers × depth 2 = 8
        : copyLikeConsumers.size() * depth;

for (AMDAIE::TileOp tileOp : tiles) {           // ← 루프가 "타일" 단위
  ...
  // Every set of buffers needs one producer and one consumer lock.
  //  → 타일 1개당 lock 1쌍. producer 4개가 그 1쌍을 공유하게 됨
}
```

- multi-producer + multi-consumer는 **명시적으로 거부**하지만,
  **multi-producer + single-consumer는 허용**하고 lock init을 `producers × depth`로
  스케일하는 것만으로 처리한다. 총 credit 수는 맞지만 **identity가 사라진다.**
- lock 배정 루프가 **producer 단위가 아니라 타일 단위**여서, 한 타일에 몰린 producer 4개가
  자동으로 lock 1쌍을 공유하게 된다.
- 바로 위 `TODO(jornt)` 주석이 이 accounting이 "pass 간 가정"에 의존한다는 것을
  작성자 본인이 이미 인지하고 있었음을 보여준다.

반대로 **코어 타일 쪽 lock은 전부 `init=2`로 정상**이다 — 루프가 타일 단위라서 broadcast의
consumer 4개는 각자 자기 타일에서 자기 lock 쌍을 받는다. 결함은 **여러 producer가 한 타일에
몰리는 경우(= memtile 출력 gather)에 국한**된다.

### 실제 chainmix에서의 규모

`chainmix_L12_int8.mlir`을 `--mlir-print-ir-after=iree-amdaie-lower-to-aie`로 덤프:

- dispatch 모듈 24개 (레이어 12 × 2 = 비배치/배치 교대), 컬럼 0~7 사용
- `AcquireGreaterEqual, 4)` **384회** = 24 dispatch × 8 컬럼 × 2 핑퐁 BD
- `init = 8` lock **192개** = 24 dispatch × 8 컬럼

즉 **모든 dispatch의 모든 컬럼에** 이 공유 lock gather가 존재한다.

## 2. 이 메커니즘이 설명하는 것 (가설로서의 정합성)

| 기존 관측 | 설명 |
|---|---|
| batch 0 출력이 0 (버그 #1) | 첫 패스에서 안 쓰인 slice는 초기화된 0 |
| batch 1 이후는 항상 정상 | 첫 패스 이후 버퍼엔 그럴듯한 값이 있고 skew가 유지되지 않음 |
| 첫 release 뒤 delay가 #1을 고침 | 코어 4개가 전부 같은 10k-cycle 통행료를 내며 **de-skew**됨 (순서 보장이 아니라 정렬) |
| delay로 안 고쳐지는 #2가 남음 | 첫 release에서 정렬해도 이후 skew가 다시 쌓임 |
| `L>=10` 필요 | 노출 기회 ∝ dispatch 수 |
| `D=768` 필요, `D=384`는 안 됨 | per-dispatch DMA 18432B vs 4608B — skew가 쌓일 시간 4배 |
| 호스트 delay가 전혀 안 먹힘 | skew는 코어 간 **상대** 시간차 — 호스트에서 기다려도 무관 |
| 손으로 짠 IRON 재현 8번 전부 negative | 그 8개 중 어느 것도 "코어 4개가 lock 1쌍을 공유하는 gather"를 만들지 않았다. 이 구조는 IREE의 objectFifo bufferization이 생성하는 것 |
| 출력이 노이즈가 아니라 이산적으로 다른 값 | 64B slice 단위로 stale/fresh가 갈리는 이산 이벤트 |

matmul+bias fusion 조사(9/1, 9/3)에서 별개로 "hang한다"고 입증된
**fan-out + shared-lock + 핑퐁** 안티패턴과 **구조적으로 같은 것**이다. 두 조사가 한 결함으로
수렴한다.

## 3. int8만 재현되는 이유 — lock 개수가 아니라 벡터화(=증폭기)

메모리에 "int8이 bf16보다 objectFifo/lock 5배"라고 기록돼 있었는데, chainmix에서 직접 세어보니
**틀렸다**:

| | 총 lock | **dispatch당** | `init=8` (공유 gather) |
|---|---|---|---|
| int8 L12 | 5568 | **232** | 192 = 24×8 |
| bf16 L10 | 4640 | **232** | 160 = 20×8 |

dispatch당 lock 수가 완전히 동일하고, **bf16에도 공유 lock gather가 똑같이 있다.**
(그 "5배"는 아마 다른 비교 — 단독 mm/bmm — 였던 것으로 보인다.)

그럼 왜 int8만 재현되는가: **벡터화가 증폭기다.** int8은 native `aievec.matmul`로 코어 계산이
매우 짧고, bf16 chainmix는 순수 스칼라라 계산이 압도적으로 길다. 계산이 길면 코어 4개가 다
같이 느려 상대 skew가 이터레이션 길이에 비해 미미하지만, 계산이 짧으면 skew가 이터레이션
길이와 맞먹는다. 버그 #1 커밋 메시지가 "**on vectorized** batched matmul"이라고 적은 것과
정확히 일치한다.

주의: **"bf16은 negative"라는 결과는 존재하지 않는다** — bf16 chainmix는 하드웨어에서
timeout(`ert state 8`)이라 비결정론 측정 자체가 된 적이 없다.

## 4. 시도했지만 **증거로 쓸 수 없는** 실험 — `--iree-amdaie-num-rows` 스윕

`--iree-amdaie-num-rows`(기본 4)가 정확히 이 producer 개수를 정하므로, L과 D를 고정한 채
공유 lock만 없애는 통제 실험이 될 것으로 기대했다. 정적으로는 의도대로 동작했다:

| | rows=4 | rows=2 | rows=1 |
|---|---|---|---|
| `init=8` lock | 192개 | — | **0개** |
| `AcquireGreaterEqual, 4)` | 384회 | — | **0회** |
| 사용 컬럼 | 0~7 | 0~7 | 0~7 |
| dispatch / L / D | 24 / 12 / 768 | 동일 | 동일 |

HW 결과 (L=12, 20회씩):

| | 서로 다른 비트 패턴 | float 기준 corr |
|---|---|---|
| rows=4 | **7개** (최대 22.8% 원소 차이) | 0.965~0.971 |
| rows=2 | 1개 (결정론적) | 0.209 |
| rows=1 | 1개 (결정론적) | 0.229 |

**그런데 L=1 대조군에서 이게 무효임이 드러났다.** L=1은 dispatch 2개라 버그가 안 터지고
양쪽 5/5 결정론적이므로, 두 출력 차이는 순수하게 num-rows의 수치적 영향이다:

| | float 기준 corr | rows=4 대비 |
|---|---|---|
| rows=4 | **0.9978** (int8 양자화 오차 수준, 정상) | — |
| rows=1 | **0.8713** | **24.78% 원소가 다름** |

**`--iree-amdaie-num-rows=1`은 그 자체로 수치를 깨뜨린다** — L=1에서 이미 레이어당 ~13%
corr 손실, 12레이어 누적이면 L=12의 corr=0.229가 그대로 설명된다. 따라서 rows=1/2의
결정론이 "공유 lock 제거 때문"인지 "계산이 달라져 비결정론이 안 보이게 된 것"인지
구분할 수 없다. **인과관계 근거로 쓰면 안 된다.**

### 부수 발견 — 별개의 새 버그

이 int8 벡터화 파이프라인에서 **`num-rows≠4`는 에러 없이 조용히 틀린 결과를 낸다**
(결정론적으로 틀림, L=1에서 이미). 기본값이 4이고 npu4+int8+벡터화 조합에서
non-default num-rows 경로가 테스트되지 않은 것으로 보인다.
**과거·미래에 num-rows를 바꾼 실험은 전부 이 confound를 의심해야 한다.**

## 5. `enable-vectorization-passes=false` — 완벽한 단일 변수 조작이지만 HW에서 실패

L=1 덤프로 vec vs novec의 IR을 비교한 결과, **flow/DMA/lock/buffer/sync 구조 5248줄이
바이트 단위로 완전히 동일**하고 코어 계산 코드만 다르다. 즉 타일링·DMA·lock을 전혀
건드리지 않는 이상적인 단일 변수 조작이다.

그러나 **HW에서 실패한다** — L=1 novec조차 실행당 ~2분 걸리고 출력 파일을 안 남긴다
(bf16 chainmix와 같은 `ert state 8` 계열). 스칼라 경로의 별개 미해결 문제로,
이 실험 축은 현재 막혀 있다.

### 이 과정에서 NPU를 일시적으로 wedge시킨 사고

실패 반복 중인 novec 런을 중단하려고 백그라운드 작업을 kill했더니 **docker 클라이언트만
죽고 컨테이너는 NPU를 계속 사용**했다(`scripts/lock/status.sh`는 `NPU: free`로 표시 —
CLAUDE.md가 경고하는 "락에 안 보이는 컨테이너" 상황). 이어서 `docker stop`이 실행 중인
dispatch를 끊어 디바이스가 wedge됐고, 이후 모든 dispatch가 알려진-정상 아티팩트까지
`ert state 8`로 실패했다.

**복구는 자동이었다** — 실패한 dispatch 1회가 TDR 리셋을 트리거해 run 2부터 정상화,
`chainmix_L12_int8.vmfb`가 rc=0 / corr=0.970738로 복귀(세션 초반 관측값과 정확히 일치).
`xrt-smi examine -r platform`도 정상(8 컬럼 열거).

교훈: NPU 실행 작업을 중단할 때는 래퍼 프로세스를 kill하지 말고 `docker ps` →
`docker stop <id>`로 컨테이너를 직접 정지시킬 것. 강제 중단 후에는 알려진-정상
아티팩트로 반드시 재검증할 것.

## 6. `gen_chained_mixed.py`의 rng 오염 수정 (같은 날 앞 세션의 계획된 실험)

같은 날 앞 세션이 찾아낸 confound: `x`와 양자화 calibration 데이터가 per-layer weight 루프
**뒤에서 같은 `default_rng(0)` 스트림**에서 뽑혀서, L마다 완전히 다른 수치 인스턴스가 됐다
(다른 입력, 다른 int8 scale). "같은 모델 + 레이어 하나 추가"가 아니었으므로 모든 L-스윕
결론이 오염됐다.

수정: weight는 `default_rng(0)` 그대로(레이어 i의 weight는 L·SEED와 무관하게 동일),
`x`/calibration은 `default_rng(1000 + SEED)`라는 **독립 스트림**에서 뽑는다.
출력 파일은 항상 `chainmix_L{N}_s{SEED}_*`로 접미사를 붙여 기존 이력 아티팩트
(`chainmix_L{N}_*`)를 덮어쓰지 않게 했다.

검증: seed 0에서 L=10,11,12,13의 `x`가 전부 바이트 동일, seed 1/2/3은 서로 다름.

---

## 7. rng 오염 수정 후의 L-스윕 — 그리고 그것을 무효화한 TDR 리셋 confound

수정된 생성기로 `x`를 L에 대해 고정한 첫 L-스윕(각 20회, rows=4 기본, 벡터화 on):

| 인스턴스 | 서로 다른 비트 패턴 | corr |
|---|---|---|
| L=10 seed 0 | 1 (20/20 결정론적) | 0.9757 |
| L=11 seed 0 | 1 (20/20) | 0.9746 |
| L=12 seed 0 | 1 (20/20) | 0.9736 |
| L=13 seed 0 | 1 (20/20) | 0.9718 |
| L=11 seed 1 | 1 (20/20) | 0.9776 |
| L=11 seed 2 | 1 (20/20) | 0.9744 |
| L=11 seed 3 | 1 (20/20) | 0.9765 |
| L=12 seed 1 | 1 (20/20) | 0.9765 |

반면 **기존 아티팩트(옛 `x`/scale)로 만든 L=12는 같은 세션에서 20회 중 7개의 서로 다른
비트 패턴**(최대 22.8% 원소 차이, maxdiff 0.28)을 냈다. 그 빌드는 오늘 컴파일러로 8/24
MLIR을 재컴파일한 것이므로 **컴파일러 버전 차이가 아니다.** 그리고 두 MLIR은
`sed`로 숫자 리터럴을 지우면 **완전히 동일**하고, `LowerToAIE` 구조도 동일하다
(dispatch 24개, `init=8` lock 192개, `AcquireGreaterEqual,4)` 384회 — 양쪽 같음).

즉 차이는 **수치 인스턴스(입력 `x` + 그로부터 나온 int8 scale)뿐**이었다. 그래서
"L 불규칙성은 인스턴스 민감도 착시"라는 결론으로 향했는데, **두 가지가 그걸 막았다.**

### (a) 인스턴스 민감도 가설은 CPU 측정으로 반증됨

onnxruntime CPU에서 입력 1개 원소를 약 1 LSB 만큼 흔들어 출력이 얼마나 바뀌는지 측정
(각 12회 probe 평균):

| 인스턴스 | 1-원소 교란이 바꾸는 출력 비율 | maxdiff |
|---|---|---|
| L12 옛 (**dirty**, 7패턴) | **0.20%** | 0.0412 |
| L12 s0 (clean) | 0.62% | 0.0535 |
| L11 s0 (clean) | 0.62% | 0.0447 |
| L13 s0 (clean) | 0.62% | 0.0593 |

**dirty 인스턴스가 오히려 교란을 덜 증폭한다** — "민감한 인스턴스라서 보인다"는 설명과
정반대다. 입력 통계도 동일하다(전부 std≈0.0497, absmax 0.188~0.215).

### (b) 그리고 더 큰 문제 — 측정 시점이 TDR 리셋을 사이에 두고 갈린다

dirty 결과(7패턴)는 세션 초반, **§5의 NPU wedge/TDR 리셋 이전**에 측정했다.
새 인스턴스 측정은 **전부 리셋 이후**다. 모델과 `x`를 교차해봤더니:

| 모델 | `x` | 결과 | 측정 시점 |
|---|---|---|---|
| 옛 | 옛 | **7패턴 (dirty)** | 리셋 **이전** |
| 옛 | 새 | 20/20 결정론적 | 리셋 이후 |
| 새 | 옛 | 20/20 결정론적 | 리셋 이후 |
| 새 | 새 | 20/20 결정론적 | 리셋 이후 |

**리셋 이후 측정한 모든 조합이 clean이다.** 리셋 직후 건강 검사에서도 옛 아티팩트가
3/3 비트 동일했다(원래는 최대 그룹이 7/20이므로 3연속 동일 확률 ~12%).

TDR 리셋은 펌웨어/전력(DPM) 상태를 재초기화하므로 코어 대 DMA 클럭 비율이 바뀌어
skew 창이 달라질 수 있다 — 그래서 원래 구성(옛 모델 + 옛 `x`)을 리셋 이후에 다시
40회 돌려 이 confound를 직접 검증했다.

### (c) 그 통제 실험 결과 — **리셋 confound는 반증됨**

리셋 이후 원래 구성 재실행 (40회, `retest_orig`):

```
21회 시점: 5개의 서로 다른 비트 패턴
  [deb99c6b2ec2] n=9  corr=0.969921   (다수)
  [9b3c6b123dd9] n=6  corr=0.969832   1092/24576 (4.44%) 차이, maxdiff=0.0412
  [743fbb9c82a3] n=4  corr=0.970738   5629/24576 (22.90%) 차이, maxdiff=0.2803
  [d639b983c876] n=1  corr=0.969860    516/24576 (2.10%) 차이
  [11949bf471e6] n=1  corr=0.969852   3324/24576 (13.53%) 차이
  -> NON-DETERMINISTIC
```

**세션 초반(리셋 이전)에 나온 것과 동일한 해시들이 그대로 재등장했다**
(`deb99c6b2ec2`, `743fbb9c82a3`, `9b3c6b123dd9`, `11949bf471e6`). 즉 TDR 리셋은
버그에 아무 영향이 없었고, §7의 인스턴스 비교는 **유효하다.**

### 따라서 확정되는 것

| 구성 | 결과 |
|---|---|
| 옛 모델 + 옛 `x` | **비결정론적** (20회→7패턴, 재검증 21회→5패턴, 동일 해시 재등장) |
| 옛 모델 + 새 `x` | 20/20 결정론적 |
| 새 모델 + 옛 `x` | 20/20 결정론적 |
| 새 모델 + 새 `x` (L=10,11,12,13 × seed 0) | 각 20/20 결정론적 |
| 새 모델 + 새 `x` (L=11 × seed 1,2,3) | 각 20/20 결정론적 |
| 새 모델 + 새 `x` (L=12 × seed 1,2,3) | 각 20/20 결정론적 |

**버그의 가시성은 `x`와 int8 scale의 정확한 조합에 달려 있다** — 둘 중 하나만 바꿔도
사라진다. 그리고 그 의존성은 §7(a)의 단순 "교란 증폭도"로는 설명되지 않는다
(dirty 인스턴스가 오히려 증폭이 작다).

**과거 L-스윕에 대한 함의:** 옛 생성기는 L마다 `x`와 scale이 같이 바뀌었으므로,
"L=10,12,13 재현 / L=11,14,16 clean"은 **L 의존성이 아니라 L과 인스턴스가 뒤섞인
측정이었다.** 특히 "L=13이 가장 심각(6패턴)"과 "L=11은 100/100 면역"은 각각 다른
인스턴스에서 나온 관측이므로 L의 성질로 볼 근거가 없다. 실제로 seed 0 인스턴스에서는
L=10,11,12,13이 **전부** 결정론적이다.

---

## 8. 컴파일러 패치로 결정적 실험 — **패치는 올바르게 동작하지만 버그는 그대로**

§7까지의 정적 근거가 강했으므로, 남은 유일한 confound-free 실험(패치)을 실제로 구현했다.

### 패치 내용

**(a) `AMDAIEObjFifoBufferization.cpp`** — multi-producer/single-consumer objFifo에
**producer마다 lock 쌍 1개**를 할당(각 producer lock init = `depth`, consumer lock init = 0).
기존의 `producers × depth` 스케일링은 총 credit만 맞추고 producer 신원을 잃는다는 설명을
주석으로 남겼다. 나머지 경우(1-producer/N-consumer 포함)는 기존 동작 그대로.

**(b) `AMDAIELowerToAIE.{h,cpp}`** — `createDMABlocks`가 lock 쌍 **리스트**를 받는다.
쌍이 여러 개면 consumer 전송을 **slice별 BD로 분할**해 각 BD가 자기 producer의 쌍을
1회 acquire/release한다(AIE2 BD는 acquire/release lock을 각 1개만 가지므로 이게 유일하게
표현 가능한 형태). producer 측은 자기가 쓰는 **base offset의 순위**로 자기 쌍을 고르므로
consumer가 slice를 읽는 순서와 정확히 일치한다. 처리 불가한 기하는 조용히 넘기지 않고
컴파일 에러(`verifyProducersTileBufferInOrder`, strided/repeated 분할 거부).

**첫 시도는 내 가드에 걸려 컴파일 실패했다** — `bmm12_int8` 단독 덤프에서 본 "연속 256B"가
chainmix 전체에는 해당되지 않았다. 실제로는 두 종류가 있었다:

| consumer 기하 | 예 |
|---|---|
| 연속 | `len=256`, dims 없음 |
| 2D | `len=1536`, `dims=[<size=4, stride=384>, <size=384, stride=1>]` |

2D 쪽의 **최외곽 차원 `size=4`가 정확히 producer 개수, `stride=384`가 slice 길이**였다
(producer들은 `len=384`를 offset 0/384/768/1152에 쓴다). 즉 논리적으로 연속인데 2D로
표현된 것뿐이라, **최외곽 차원을 벗겨내는** 방식으로 일반화했다.

### 정적 검증 — 의도대로 정확히 동작

| | 베이스라인 | 패치 후 | 검산 |
|---|---|---|---|
| dispatch 모듈 | 24 | 24 | 동일 |
| `init=8` lock (공유 gather) | **192** | **0** | 제거 |
| `AcquireGreaterEqual, 4)` / `Release, 4)` | **384** / 384 | **0** / 0 | 제거 |
| lock 총계 | 5568 | 6720 | +1152 = 192 objFifo × (8−2) lock |
| `aie.dma_bd` | 7680 | 8832 | +1152 = 384 consumer BD × 3 |

IR에서 짝짓기를 직접 확인(컬럼 0, dispatch 1): consumer 체인이 BD 8개(버퍼 2 × slice 4)로
쪼개지고, 각 BD가 그 slice를 쓰는 producer의 쌍과 1:1로 맞는다.

| consumer BD | offset | acquire (ready) | release (free) | producer |
|---|---|---|---|---|
| buf0 slice0 | 0 | `lock_0_1_1` | `lock_0_1` | S2MM ch2 (offset 0) |
| buf0 slice1 | 64 | `lock_0_1_3` | `lock_0_1_2` | S2MM ch3 (offset 64) |
| buf0 slice2 | 128 | `lock_0_1_5` | `lock_0_1_4` | S2MM ch4 (offset 128) |
| buf0 slice3 | 192 | `lock_0_1_7` | `lock_0_1_6` | S2MM ch5 (offset 192) |
| buf1 slice0~3 | 동일 | 동일 쌍 | 동일 쌍 | 핑퐁 유지 |

수치도 보존된다 — corr 0.9698~0.9707로 베이스라인과 동일 범위.

### 하드웨어 결과 — **NEGATIVE**

원래 dirty 구성(옛 모델 + 옛 `x`)을 패치 빌드로 40회:

| 출력 해시 | 베이스라인 (40회) | 패치 후 (40회) |
|---|---|---|
| `deb99c6b2ec2` | 18 | 10 |
| `9b3c6b123dd9` | 12 | 15 |
| `743fbb9c82a3` | 8 | 10 |
| `d639b983c876` | 1 | 2 |
| `11949bf471e6` | 1 | 2 |
| `b0dcf9221716` | — | 1 (신규 싱글톤, majority 대비 13.64% — `11949bf471e6`의 13.59%와 사실상 같은 계열) |
| **서로 다른 패턴** | **5** | **6** |

**베이스라인의 5개 해시가 하나도 빠짐없이 그대로 재등장한다.** 8컬럼 × 24 dispatch 전부의
memtile gather 동기화를 완전히 교체했는데도 **가능한 오답의 집합이 불변**이고, 비율만
이동했다(타이밍이 바뀌었으니 예상되는 결과).

**결론: multi-producer 공유 카운팅 세마포어는 이 비결정론의 원인이 아니다.** §1~2의 가설은
원인으로서 **반증됐다**.

단, §1의 구조적 결함 자체는 여전히 실재하고 unsound하다(카운팅 세마포어로 producer 신원을
표현할 수 없다는 사실은 변하지 않는다). 이 버그의 원인은 아니지만 별건의 잠재 결함으로
upstream 가치가 있다 — 비용은 해당 채널의 BD 3배 증가.

### 이 negative가 알려주는 새 정보

오답 집합이 큰 구조 변경에도 불변이라는 건, 원인이 **작고 이산적인 상태 집합**이라는 뜻이다.
memtile 출력 gather가 배제됐으므로 남은 후보(유력순):

1. **호스트 측 출력 읽기 타이밍**(TCT sync / fence). dispatch 레벨이라 이 패치의 영향을
   전혀 받지 않고, "stale 전체 텐서 vs fresh"라는 소수의 이산 상태를 정확히 만들어낸다.
2. 입력 broadcast(memtile → 코어 4개) 또는 shim → memtile 입력 경로.
3. 코어 자체의 L1 처리.

## 9. 부수 수정 — `scripts/lock/`가 사용자 간에 조용히 고장나 있었다

빌드 락을 잡으려다 발견했다. `ace-amd01`은 `fs.protected_regular = 2`이고 `$LOCK_ROOT`는
의도적으로 sticky + world-writable(`chmod 1777`)이다. 이 조합에서 커널은 **파일 소유자가
호출자도, 디렉토리 소유자도 아닐 때 write 열기를 거부**한다. `lock_run`은 `200>"$file"`,
`lock_status`는 `200<>`로 열었으므로, 정확히 락이 필요한 사용자 간 상황에서 EACCES가 났다:

- `status.sh`가 **영구히 `BUSY`로 오보**(fail-closed)
- `lock_run`은 **아무도 잡지 않은 락을 무한 대기**

실측: `gylee`의 `build.lock`에 write 열기는 EACCES, **읽기 전용 fd로는 flock 즉시 획득** —
즉 락은 실제로 free였고 `status.sh`가 틀렸다.

**수정**: `flock`은 열린 파일 서술자를 잠그므로 읽기 전용 fd로도 배타 락이 된다.
`lock_run`/`lock_status` 모두 `200<"$file"`로 변경. lock 파일 생성 실패 시 조용히 넘어가지
않도록 명시적 에러도 추가(읽기 전용 fd는 `200>`와 달리 파일을 만들지 않는다).

비대칭 주의: 락 디렉토리 소유자(`wjjang`)가 만든 파일은 모두에게 정상 동작한다 — 그래서
`npu.lock`은 계속 잘 됐고 `gylee`가 만든 `build.lock`만 막혔다. `.meta` 파일에는 같은
함정이 남아 있어(스크립트 주석이 이미 인지) 홀더 *표시*는 사용자 간에 깨질 수 있다.

---

## 결론 / 남은 것

### 확정된 것 (정적 — 재확인 가능)

1. 모든 chainmix dispatch의 모든 컬럼에서 **독립된 코어 4개**가 하나의 memtile 버퍼의 서로
   다른 slice에 쓰면서 **lock 1쌍을 공유**하고, consumer는 `AcquireGreaterEqual(ready, 4)`로
   기다린다(L=12에서 `init=8` lock 192개, `acquire>=4` 384회).
2. `AMDAIEObjFifoBufferization.cpp`가 이걸 lock init `producers × depth` 스케일링만으로
   처리하며, lock 배정 루프는 producer 단위가 아니라 타일 단위다.
3. 카운팅 세마포어로는 producer 신원도 버퍼 세대도 표현할 수 없으므로, 이 스킴은
   **강제되지 않은 lockstep 가정에 의존한다 — 그 자체로 unsound하다.**
4. lock init 조정만으로는 못 고친다("한 producer가 credit 2개를 먼저 집는" 시나리오가 남음).
   올바른 형태는 producer별 lock 쌍 + consumer BD 분할이고, **그걸 실제로 구현·검증했다**(§8).
5. 부수 버그: 이 int8 벡터화 파이프라인에서 **`--iree-amdaie-num-rows≠4`는 에러 없이
   조용히 틀린 결과**를 낸다(L=1에서 이미 corr 0.9978 → 0.8713).
6. 부수 버그(수정 완료): `scripts/lock/`가 `fs.protected_regular=2` + sticky 디렉토리 조합에서
   사용자 간에 조용히 고장나 있었다(§9).

### 확정된 것 (동적)

7. 버그는 살아있고 안정적으로 재현된다 — 원래 구성 40회에서 5개 패턴(18/12/8/1/1),
   최대 22.9% 원소 차이.
8. **가시성은 `x`와 int8 scale의 정확한 조합에 달려 있다.** 둘 중 하나만 바꾸면 13개 구성이
   전부 20/20 결정론적이 된다.
9. 따라서 **기존의 "L 의존성"은 신뢰할 수 없다** — 옛 생성기가 L마다 인스턴스를 같이 바꿨다.
   seed 0에서 L=10~13 전부, L=11 seed 1/2/3, L=12 seed 1/2/3 전부 결정론적.
10. TDR 리셋은 버그에 영향 없다(리셋 전후 동일 해시 재등장).
11. §7(a)의 "교란 증폭도"로는 (8)이 설명되지 않는다 — dirty 인스턴스가 오히려 덜 증폭한다.
12. **§1~3의 공유 lock 가설은 원인으로서 반증됐다**(§8). producer별 lock + consumer BD 분할
    패치가 정적으로 완벽히 동작하는데도(공유 lock 0개, 짝짓기 1:1 확인, 수치 보존)
    **베이스라인의 5개 오답 해시가 하나도 빠짐없이 그대로 재현된다**(40회, 6패턴).

### 확정되지 않은 것

- **진짜 원인.** memtile 출력 gather는 §8로 배제됐다.
- **(8)의 메커니즘.** 왜 정확한 `x`+scale 조합에서만 보이는지.

### 다음 후보 (§8의 negative가 좁혀준 순서)

오답 집합이 큰 구조 변경에도 불변이라는 사실이 가장 강한 단서다 — 원인은 **작고 이산적인
상태 집합**이며, DMA/lock 층의 연속적 타이밍 레이스와는 성질이 다르다.

1. **호스트 측 출력 읽기 타이밍(TCT sync / fence).** dispatch 레벨이라 §8 패치의 영향을
   전혀 받지 않고, "stale 전체 텐서 vs fresh"라는 소수의 이산 상태를 정확히 만들어낸다.
   가장 유력. (메모리에 "wait-fold/TCT-sync lowering 감사 clean"이 있지만 감사는 증명이 아니다.)
2. 입력 broadcast(memtile → 코어 4개) 또는 shim → memtile 입력 경로.
3. 코어 자체의 L1 처리.
4. **`num-rows≠4` 오답 버그**를 별건으로 조사/보고(기본값 외 경로 미테스트로 보임).
5. §1의 공유 lock 결함은 **별건의 잠재 unsoundness**로 upstream 검토 — 이 버그의 원인은
   아니지만 실재한다. 비용은 해당 채널 BD 3배.

### 재현 자료 (모두 gitignored)

`_local/int8_debug/out/rowsweep/`:
- `L12_rows{4,2,1}.vmfb` — num-rows 스윕 (rows=4가 원래 dirty 구성)
- `L1_rows{4,1}.vmfb`, `L1_novec.vmfb`, `L12_rows4_novec.vmfb` — 대조군
- `L1{0,1,2,3}_s0.vmfb`, `L11_s{1,2,3}.vmfb`, `L12_s{1,2,3}.vmfb` — 오염 없는 인스턴스
- `L12_fixed.vmfb` — **패치된 컴파일러 산출물**(§8)
- 실행 출력: `r{4,2,1}/`, `L1r{4,1}/`, `run_L1*_s*/`, `cross_{oldm_newx,newm_oldx}/`,
  `retest_orig/`(베이스라인 40회), `fixed_orig/`(패치 40회), `health*/`

생성기: `_local/int8_debug/gen_chained_mixed.py` (rng 스트림 분리 + `SEED` 인자).
패치: `compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEObjFifoBufferization.cpp`,
`AMDAIELowerToAIE.{h,cpp}`, `scripts/lock/lock-common.sh` (전부 uncommitted).
