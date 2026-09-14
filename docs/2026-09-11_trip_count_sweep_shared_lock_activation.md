# 공유 lock 결함의 활성 조건 — trip-count 스윕 (2026-09-11)

`65069ce`(per-producer lock)가 고친 결함이 **왜 배치 matmul에서만 관측됐는가**를 통제 실험으로
확인한 기록. 결론부터: **bmm 전용이 아니다.** 비배치 2D matmul도 조건만 맞추면 똑같이 깨진다.

## 0. 배경 — 정적 구조는 matmul도 동일하다

`AMDAIEObjFifoBufferization`이 내보내는 memtile 출력 fifo의 lock 구성은 비배치 matmul과
배치 matmul이 **같다**. `mm_bf16`(32×64×128, 비배치) 덤프:

```
%buffer, %buffer_10 : memref<256xf32, 1>   ← depth 2
%lock = amdaie.lock(%tile_0_1(4), 8)       ← init 8 = 4 producers × depth 2 (공유 1쌍)
```

즉 "bmm에만 생기는 버그"가 아니라, **컴파일러는 둘 다에게 unsound한 동기화를 내보내고 있고
matmul에서는 그것이 실행되지 않았을 뿐**이라는 가설.

관측된 차이는 producer의 `release` 횟수뿐이었다:

| | produce acquire/release 위치 | 코어당 release 횟수 T |
|---|---|---|
| mm 32×64×128 | `scf.for` **밖** | 1 |
| bmm 12×32×64×32 | `scf.for 0..12` **안** | 12 |

T=1이면 `ready`가 4를 넘을 수 없어 `AcquireGreaterEqual(4)`가 필연적으로 "4개 producer 각 1회"를
의미한다 — 세대가 하나뿐이라 신원 소실이 관측 불가능해진다. 반면 chainmix의 비배치
`matmul_32x768x768`은 produce가 `scf.for 0..3` 안에 있어 **전제조건을 이미 충족하는데도**
실측에서 항상 정확했다. 그래서 T를 직접 스윕했다.

## 1. 실험 장치

`AMDAIEObjFifoBufferization.cpp`에 디버그 전용 탈출구를 추가(커밋 대상 아님):

```cpp
static const bool legacySharedLock = std::getenv("AMDAIE_LEGACY_SHARED_LOCK") != nullptr;
if (legacySharedLock) lockPairPerProducer = false;
```

`AMDAIELowerToAIE`는 lock 쌍 개수(`producerLocks.size() > 1`)를 따라가므로 이 한 줄로
파이프라인 전체가 pre-fix 동작으로 돌아간다. 동일 shape을 두 모드로 컴파일해 확인:

```
legacy : lock_inits [2, 8]   ← 공유 1쌍, init = 4×depth
fixed  : lock_inits [2]      ← producer당 1쌍, init = depth
T, R(내부 K 루프) 동일 → lock 스킴만 격리된다
```

- 생성: `_local/int8_debug/gen_trip_sweep.py <tag> <M> <K> <N> [BATCH]` (int8 QDQ, shape마다 고정 seed)
- 컴파일: `_local/int8_debug/tripsweep/compile.sh <tag> legacy|fixed [dump-pass]`
- 실행: `_local/int8_debug/tripsweep/run.sh <tag...>` — 모드당 10런, 런마다 별도 프로세스
- 판정: fixed 결과를 정답으로 두고 legacy 10런을 비트 비교 + legacy 런들 간 distinct 개수

총 29 shape × 2 모드, 케이스당 10~40런 = **700런 이상, 실행 실패 0건**.

## 2. 결과

### (a) T 사다리 — K=768 고정, N으로 T를 올림 (비배치 2D)

| N | T | legacy distinct | legacy==fixed | 오답 비율 |
|---|---|---|---|---|
| 64 | 1 | 1 | ✅ True | 0.00% |
| 256 | 1 | 1 | ✅ True | 0.00% |
| 512 | 1 | 1 | ✅ True | 0.00% |
| 768 | 2 | 1 | ✅ True | 0.00% |
| 1536 | 3 | 1 | ✅ True | 0.00% |
| 3072 | 6 | 1 | ✅ True | 0.00% |

**T만 올리는 것으로는 활성화되지 않는다.** T=6까지 공유 lock으로도 비트 동일.

### (b) K 사다리 — N=1536 고정 (비배치 2D, T=2~3)

| K | T | legacy distinct | legacy==fixed | maxdiff | 오답 비율 | corr |
|---|---|---|---|---|---|---|
| **32** | **2** | **4** | ❌ **False** | **0.2420** | **17.39%** | **0.9320** |
| 64 | 2 | 1 | ✅ True | 0 | 0.00% | 0.9998 |
| 128 | 3 | 1 | ✅ True | 0 | 0.00% | 0.9998 |
| 256 | 3 | 1 | ✅ True | 0 | 0.00% | 0.9998 |
| 768 | 3 | 1 | ✅ True | 0 | 0.00% | 0.9998 |

**⚠️ 핵심 결과: `K=32, N=1536`은 배치가 전혀 없는 평범한 2D matmul인데 재현한다.**
10런에서 서로 다른 출력 4종, 원소의 17.4%가 틀리고 corr 0.932로 떨어진다.
같은 자리에서 fixed는 10런 1종, 비트 동일. **"bmm 전용"이라는 설명은 반증됐다.**

### (c) B 사다리 — K=32, N=64 고정 (배치)

| B | T | legacy distinct | legacy==fixed | 오답 비율 | corr |
|---|---|---|---|---|---|
| 2 | 2 | 2 | ❌ False | 24.51% | 0.8655 |
| 4 | 4 | 3 | ❌ False | 12.28% | 0.9220 |
| 8 | 8 | 4 | ❌ False | 14.47% | 0.9653 |
| 12 | 12 | 3 | ❌ False | 23.44% | 0.9772 |
| 24 | 24 | 4 | ❌ False | 21.07% | 0.7938 |
| 12 (**K=64**) | 12 | 1 | ✅ True | 0.00% | 0.9998 |

**B=2, 즉 T=2면 이미 깨진다** — 높은 trip count가 필요한 게 아니라 "producer가 같은 버퍼
세트에 대해 두 번 release할 수 있다"는 최소 조건이면 충분하다. 그리고 마지막 줄이 결정적이다:
**T=12로 동일한데 K만 32→64로 늘리면 완전히 깨끗해진다.**

## 3. 오염의 지문 — 기전을 직접 확인

corr/오답비율은 심각도 지표로 못 쓴다. 대신 **틀린 원소가 어디에 어떤 값으로 나타나는지**를 봤다
(`legacy` 각 런 vs `fixed` 비트 비교, 출력 32행은 producer 4개가 8행씩 나눠 씀).

### (i) 0-오염 = 한 번도 쓰이지 않은 메모리 (RAW)

`b_B12`(B=12) 10런의 지배적 클래스(8/10런): **batch 0, 행 0-15, 값 전부 0**.
다른 클래스는 **행 0-7, 전부 0**. 즉 오염 영역이 항상 **producer slice의 정수배**다:

| 0인 행 수 | 안 쓴 producer 수 | consumer의 4 credit 출처 |
|---|---|---|
| 8행 | 1개 | 나머지 3개 (그중 하나가 2번 release) |
| 16행 | 2개 | 나머지 2개 (각각 2번 release) |

**비분할 counting semaphore가 정확히 허용하고, per-producer lock이 정확히 금지하는 조합이다.**
"경합이 있었다"가 아니라 "credit 조합이 산술적으로 이렇게 맞아떨어졌다"까지 보인다.

비배치 `t_K32_N1536`도 **같은 지문**이다: 행 0-7/0-11, 값 전부 0, 열은 1536 중 768개
(= T=2의 첫 출력 타일). 행 수가 8의 배수가 아닌 클래스(10행, 12행)는 producer가 **쓰는 도중**에
consumer가 읽어간 부분 전송이다.

모든 0-오염이 **첫 세대**(batch 0 / 첫 출력 타일)에만 나타난다는 점도 일관된다.

### (ii) 앞서나간 producer의 덮어쓰기 (WAR)

같은 근본 원인의 두 번째 위험도 직접 잡혔다. `b_B12` run 8을 **타일 단위**로 보면 (2026-09-14 정밀화):

batch 7 에서 정답과 다른 곳은 **정확히 8×8 타일 8개**이고, 각 타일이 **통째로** 정답 batch 9 의
같은 위치 타일과 비트 동일하다. 8×8 은 코어 출력 objectFifo 버퍼 `memref<64xi32, 2>` 크기 그대로다.

```
행 8-15  · 열 0-7    → batch 9 와 타일 전체 동일
행 8-15  · 열 32-39  → batch 9
행 8-15  · 열 48-55  → batch 9
행 16-23 · 열 8-15   → batch 9
행 16-23 · 열 16-23  → batch 9
행 16-23 · 열 56-63  → batch 9
행 24-31 · 열 24-31  → batch 9
행 24-31 · 열 40-47  → batch 9
```

**우연 배제**: 같은 마스크에서 나머지 11개 batch 와의 원소 일치율은 0.0~2.6% 인데 batch 9 만 100%.
그리고 8개 타일의 **열-블록이 0~7 하나씩** — 8개 하드웨어 column 에서 각각 한 번씩 일어난 것과 일치.

⚠️ **정정**: 앞선 초안의 "batch 10 의 40개 원소가 batch 8 과 일치"는 *그 40개 원소에 한해서만* 맞다.
타일 단위로 보면 batch 10 은 **타일 5개의 첫 행만** batch 8 데이터이고 나머지 7행은 자기 정답이다 —
전송 도중에 읽힌 **부분 덮어쓰기**이고, batch 7 의 완전한 타일 교체보다 약한 사례다.

consumer가 아직 읽지 않은 버퍼에 **다음 세대 데이터가 덮어쓰여** 있었다. init=8은 총량으로는
맞지만(4 producer × depth 2 = 슬라이스 8칸) 분할이 없어 **한 producer가 혼자 credit을 여러 개
가져가 세대를 앞질러 갈 수 있다.** 신원 소실과 run-ahead 한계 소실은 같은 결함의 두 증상이다.


### (iii) 오염 데이터의 출처 세대 — 거리가 ±2 로 강제된다 (2026-09-14)

run 8 의 손상 타일 **125개 전수 조사**. 각 타일이 어느 batch 의 정답인지 맞춰본 결과:

| 손상 batch | 들어있던 데이터 | 거리 | 타일 수 | 열-블록 |
|---|---|---|---|---|
| 0 | 전부 0 (미기록) | — | 16 | 0–7 각 2개 |
| 2 | batch 0 / batch 4 | −2, +2 | 12 | — |
| 3 / 4 / 5 / 6 / 7 / 8 / 9 | batch 5 / 6 / 7 / 8 / 9 / 10 / 11 | **+2** | 각 8 | 0–7 각 1개 |

출처가 특정되는 68개 타일의 거리 분포: **+2 가 61개, −2 가 7개, 그 외 0개.**
나머지는 0-타일 16개와 부분 덮어쓰기 41개. **±1, ±3 은 하나도 없다.**

**왜 2인가 — 두 조건이 동시에 2를 강제한다:**

1. `depth = 2` 이므로 세대 g 와 g+2 가 **같은 물리 버퍼**(`g mod 2`)를 쓴다. 1 앞선 producer 는
   다른 버퍼에 쓰므로 세대 g 를 건드릴 수 없다.
2. run-ahead 용량이 `⌊D_in/R⌋ = ⌊2/1⌋ = 2` 이므로 3 이상 앞설 수 없다.

⇒ 가능한 거리는 **오직 ±2** 이고 관측이 100% 그렇다. 부호는 순서다 — `+2` 는 producer 가 이미
다음다음 세대를 써버린 뒤 consumer 가 실어간 경우(WAR), `−2` 는 아직 안 써서 두 세대 전 내용이
남아 있던 경우(RAW). **batch 0 의 "−2" 는 존재하지 않는 세대라 초기값 0 — batch-0 all-zero
증상의 정체가 이것이다.**

batch 3–9 가 전부 8타일씩 8개 열-블록에 하나씩인 것도 일관된다: 한 column 에서 발동하면
그 코어는 남은 dispatch 동안 **2세대 리드를 유지한 채 계속 달린다**(일회성이 아니라 정상 상태).

## 3b. 해석 — 무엇이 증명됐고 무엇이 가설인가

**증명된 것**

1. **결함의 정체**: credit이 producer별로 분할되지 않아 (a) consumer의 join이 부분집합으로
   만족되고 (b) 한 producer가 세대를 앞질러 간다. → 위 (i)(ii)의 산술적 지문이 직접 증거.
2. **T ≥ 2 가 필요조건**: 논리적으로 T=1이면 `ready` 최대값이 producer 수와 같아 부분집합 조합이
   불가능. 실측 통제쌍도 있다 — **K=32 고정**, N=768(T=1) 깨끗 / N=1536(T=2) 깨짐.
3. **K 의존은 구조 변화가 아니다**: K=32와 K=64의 L2 출력 fifo IR이 동일(버퍼 2개, 3072×i32,
   depth 2, lock init [2,8], memtile S2MM 채널 수 동일). 순수하게 동적인 현상.
4. **bmm 전용이 아니다**: 비배치 2D가 같은 지문으로 재현.

**정정 — 이전 설명은 틀렸다**

앞선 초안에서 "K-reduction 루프가 **공유 입력 objectFifo lock**을 통해 4개 코어를 암묵적
lockstep에 묶어 skew를 막는다"고 썼는데, **IR로 반증된다.** 브로드캐스트 입력 `%86`은

```
from_buffers({buf×8}, {lock×4}, {lock×4})   ← 코어당 자기 버퍼 2개 + 자기 lock 쌍
```

코어마다 **자기 L1 버퍼와 자기 lock**을 갖는다. 입력 쪽에 공유 랑데부는 없고, 따라서 lockstep
결합도 없다.

**한때의 가설과 그 반증 — 계산 시간은 무관하다**

한때 "세대 계산시간(∝K) vs 코어 간 시작 지터"로 설명했으나 **통제 실험으로 반증됐다.**
K는 고정(=32)한 채 M/N만 키워 **세대당 계산량만** 2×·4×로 늘렸다:

| 케이스 | M | N | 세대당 ukernel 호출 | ukernel 출력 타일 | 결과 |
|---|---|---|---|---|---|
| `b_B12` | 32 | 64 | 1 | 1×8×8 | ❌ 10/10 실패 |
| `b_M64` | 64 | 64 | 1 | **2**×8×8 | ❌ 20/20 실패 (오답 22.3%) |
| `b_M128` | 128 | 64 | 1 | **4**×8×8 | ❌ 20/20 실패 (오답 34.5%) |
| `b_N128` | 32 | 128 | 1 | 2×1×8×8 | ❌ 20/20 실패 |
| `b_K48` | 32 | 64 | **2** | 1×8×8 | ✅ 0/20 |

**계산량을 4배로 늘려도 전혀 나아지지 않고, 호출 수만 1→2가 되면 완전히 깨끗해진다.**
시간이 아니라 **구조**가 판별자다.

## 3c. 진짜 기전 — 입력 fifo 크레딧이 run-ahead를 제한한다

호출 1회마다 입력 objectFifo를 acquire/release 한다. 그래서 판별자는
**출력 세대 1회당 입력 acquire 라운드 수**다. 코어가 출력 lock에서 앞설 수 있는 세대 수는

```
run-ahead(세대) = floor( 입력 L1 depth / 세대당 입력 라운드 수 )
```

**출력 L1 depth는 들어가지 않는다** — 아래 §3d에서 분리 실험으로 확인했다.

이고, 버그는 leader가 laggard보다 **2번 release**해야 성립하므로 이 값이 **≥ 2**여야 한다.
라운드가 2 이상이면 코어는 자기 **입력** fifo의 크레딧을 먼저 소진해 스스로 막히고,
출력 lock에서 남을 앞지를 수 없다.

### 양방향 개입으로 증명

L1 depth를 디버그 탈출구(`AMDAIE_L1_DEPTH`)로 직접 바꿔, 보호가 있던 곳에서 **없애고**
없던 곳에 **넣었다**. 6/6 예측 적중:

| 케이스 | L1 depth | 라운드/세대 | run-ahead | 예측 | 실측 |
|---|---|---|---|---|---|
| `b_B12` | 2 | 1 | 2 | 실패 | ❌ 10/10 실패 |
| `b_B12d1` | **1** | 1 | **1** | 깨끗 | ✅ 0/20 |
| `b_K48` | 2 | 2 | 1 | 깨끗 | ✅ 0/20 |
| `b_K48d4` | **4** | 2 | **2** | 실패 | ❌ 20/20 실패 (22.3%) |
| `b_B12K64` | 2 | 2 | 1 | 깨끗 | ✅ 0/40 |
| `b_B12K64d4` | **4** | 2 | **2** | 실패 | ❌ 20/20 실패 (21.6%) |

깨끗하던 두 케이스를 **깊이만 바꿔 깨뜨렸고**, 깨지던 케이스를 **깊이만 바꿔 고쳤다.**
shape도 lock 스킴도 그대로다.

## 3d. 입력 크레딧인가 출력 크레딧인가 — 분리 실험

`AMDAIE_L1_DEPTH`는 입력·출력 L1 깊이를 동시에 바꾸므로 귀속이 안 됐다. DMA 방향으로 판별하는
탈출구를 새로 만들어(`AMDAIE_L1_IN_DEPTH` / `AMDAIE_L1_OUT_DEPTH`; 코어 입력 fifo는 그것을
채우는 `dma_cpy_nd`의 **target**, 출력 fifo는 그것을 비우는 dma의 **source**) 2×2로 갈랐다:

| 케이스 | 라운드 | 입력 depth | 출력 depth | 결과 |
|---|---|---|---|---|
| `b_B12` | 1 | 2 | 2 | ❌ 10/10 실패 |
| `b_B12_in1out2` | 1 | **1** | 2 | ✅ 0/20 |
| `b_B12_in2out1` | 1 | 2 | **1** | ❌ **20/20 실패** (21.6%) |
| `b_K48` | 2 | 2 | 2 | ✅ 0/20 |
| `b_K48_in4out2` | 2 | **4** | 2 | ❌ **20/20 실패** (20.3%) |
| `b_K48_in2out4` | 2 | 2 | **4** | ✅ 0/20 |

**입력 깊이만 바꾸면 양방향으로 켜고 끌 수 있고, 출력 깊이는 아무 영향이 없다.**
출력 depth를 1까지 줄여도 실패가 그대로고(`b_B12_in2out1`), 4로 늘려도 깨끗한 게 그대로다
(`b_K48_in2out4`). **run-ahead를 제한하는 것은 오직 입력 크레딧이다.**

이유는 결함 자체와 맞물린다: 코어의 출력 release는 **과잉 충전된 공유 lock**으로 던져지므로
back-pressure가 애초에 없다(그게 버그다). 따라서 코어를 붙잡아 둘 수 있는 유일한 것은
자기 입력 fifo의 크레딧뿐이다. per-producer lock으로 고치면 출력 쪽에 제대로 된 back-pressure가
생기고, 그 순간 입력 깊이는 무의미해진다.

## 3e. producer 수 P 로 일반화되는가

P 는 memtile 버퍼 하나를 나눠 쓰는 코어 수이고 `--iree-amdaie-num-rows` 로 바꾼다.
legacy 의 producer lock init = `P × depth` 라 IR 에서 P 를 직접 읽을 수 있다(확인: 4/6/8 → P=2/3/4).

⚠️ 과거에 "num-rows≠4 는 조용히 틀린 결과를 내는 별개 버그가 있어 증거로 못 쓴다"고 기록해 뒀는데,
이번에는 **같은 num-rows 의 legacy vs fixed 비트 비교**라 그 오염이 상쇄된다. 실제로 P=2 의
fixed 는 20런 1종으로 완전히 결정론적이어서 비교 기준으로 유효함을 확인했다.

### 이론이 예측하는 것

consumer 는 `ready ≥ P` 를 기다린다. 일부 producer(≥1개)가 안 썼는데 발사되려면 **남은
producer 들만으로 P 번의 release** 가 나와야 하고, 각자는 run-ahead R 번까지만 앞설 수 있다:

```
(P − 결손수) × R ≥ P ,  결손수 ≥ 1   →   최소조건 (P−1)·R ≥ P   →   R ≥ P/(P−1)
```

R 은 정수이므로 **P ≥ 2 인 모든 P 에서 조건은 똑같이 R ≥ 2** 가 된다.
즉 이론은 **P 가 임계값을 바꾸지 않는다**고 예측한다.

### 실측 (2×2, P × R)

| 케이스 | P | 라운드 | run-ahead R | 슬라이스 | fail/runs | 결손 producer | 남은 producer의 release 수 |
|---|---|---|---|---|---|---|---|
| `b_B12_r2` | 2 | 1 | **2** | 16행 | ❌ 16/20 | 1 (16행) | 2/1 = **2.0회씩** |
| `b_K48_r2` | 2 | 2 | 1 | 16행 | ✅ 0/20 | — | — |
| `b_B12` | 4 | 1 | **2** | 8행 | ❌ 10/10 | 2 (16행) | 4/2 = **2.0회씩** |
| `b_K48` | 4 | 2 | 1 | 8행 | ✅ 0/20 | — | — |

**임계값은 P 와 무관하게 R ≥ 2 로 동일하다 — 예측대로다.** 그리고 오염 지문이 P 에 맞춰
스케일한다: 0-오염 영역의 최소 단위가 정확히 `32/P` 행(=producer 1개 몫)이고, 두 경우 모두
**남은 producer 가 정확히 2회씩 release 한 것**으로 산술이 맞아떨어진다. P=2 는 훨씬 심한데
(출력 17종·오답 38%) leader 하나만 2번 앞서면 성립하기 때문이다.

### 못 한 것

- **P=8 은 하드웨어적으로 불가능**하다. npu4 의 core row 는 1~4 범위로
  (`Invalid number of core rows (8), must be in the range [1, 4]`), 한 memtile 을 공유하는
  producer 는 최대 4개다.
- **P=3 은 hang** 했다(legacy, 20런 중 1번째에서 정지, TDR 60s 도 넘김). M=32 가 3으로
  나누어떨어지지 않는 구성이라 기존 num-rows 버그 쪽일 가능성이 크지만 **확인하지 않았다** —
  공유 장비라 컨테이너를 직접 정지하고(락 해제 확인) 알려진 정상 모듈로 복구를 확인한 뒤
  더 밀지 않았다. 복구 직후 2런이 rc=1 이었고 이어진 6런은 6/6 정상.

## 3f. 규칙은 유도된 것인가 맞춘 것인가 — 조각별 구분

규칙 세 조각의 지위가 다르다. 섞어 쓰면 안 된다.

### (1) T ≥ 2 — **유도됨**

consumer 는 `ready ≥ P` 를 기다린다. 각 producer 가 최대 1회만 release 할 수 있으면(T=1)
`ready ≤ P` 이고, `ready = P` 는 **P개 전부가 각 1회 release 했을 때만** 성립한다. 부분집합
조합이 원리적으로 불가능하다. 실험이 아니라 lock 의미론에서 바로 나온다.

### (2) 임계값 "2" — **유도됨**

결손이 생기려면 남은 producer 집합 S(|S| ≤ P−1)만으로 P번의 release 가 나와야 하고,
각자는 run-ahead 용량 R<sub>a</sub> 회까지만 앞설 수 있다:

```
|S| · Ra ≥ P ,  |S| ≤ P−1   →   (P−1)·Ra ≥ P   →   Ra ≥ P/(P−1)
```

P ≥ 2 에서 `P/(P−1) ∈ (1,2]` 이고 R<sub>a</sub> 는 정수이므로 **모든 P 에서 R<sub>a</sub> ≥ 2**.
"2" 는 맞춘 값이 아니라 이 부등식의 올림값이다. P=2,4 실측이 이를 확인했다(§3e).

### (3) run-ahead = ⌊D<sub>in</sub> / R⌋ — **유도 + 판별 실험**

유도: 코어는 자기 L1 입력 fifo 에 최대 D<sub>in</sub> 개의 버퍼만 들고 있을 수 있고,
출력 세대 1회가 그중 R 개를 소비한다. 따라서 **새 전달 없이 연속 실행할 수 있는 세대 수는
⌊D<sub>in</sub>/R⌋** 이다.

⚠️ 문제: 기존 5개 데이터점 `(D,R)` = (2,1)F, (1,1)C, (2,2)C, (4,2)F, (2,3)C 는
**`D ≥ 2R`(이 유도)과 `D > R`(경쟁 가설)에 똑같이 들어맞는다.** 둘을 가르는 점을 따로 잡았다:

| 케이스 | D<sub>in</sub> | R | `D ≥ 2R` 예측 | `D > R` 예측 | 실측 |
|---|---|---|---|---|---|
| `b_K48_in3` | 3 | 2 | clean (3 ≥ 4 거짓) | FAIL (3 ≥ 3 참) | ✅ **0/20 clean** |
| `b_K96_in4` | 4 | 3 | clean (4 ≥ 6 거짓) | FAIL (4 ≥ 4 참) | ✅ **0/20 clean** |

**두 점 모두 `D ≥ 2R` 편 — 경쟁 가설 `D > R` 은 반증됐다.**

### (4) 상류(L2) 버퍼링은 기여하지 않는다

"코어가 앞설 수 있는 양"이 L1 만의 성질인지, 상류 버퍼까지 합한 것인지 확인했다.
`AMDAIE_L2_DEPTH` 탈출구로 L1 입력 depth·R 을 그대로 둔 채 L2 depth 만 2→4 로 올렸다:

| 케이스 | D<sub>in</sub> | R | L2 depth | 실측 |
|---|---|---|---|---|
| `b_K48` | 2 | 2 | 2 | ✅ 0/20 |
| `b_K48_l2d4` | 2 | 2 | **4** | ✅ **0/20 — 변화 없음** |

상류를 아무리 채워도 데이터는 결국 L1 에 앉아야 코어가 쓴다. **한계는 코어 자신의 L1 입력
크레딧이고, 공식에 L2 가 들어가지 않는 이유가 이것이다.**

### 남은 가정

⌊D<sub>in</sub>/R⌋ 은 "새 전달이 없을 때"의 상한이다. 이 상한이 실제 run-ahead 와 같으려면
**뒤처진 코어가 있는 동안 앞선 코어로의 전달이 진행되지 못해야** 한다(브로드캐스트 원본 버퍼가
모든 소비자를 기다리므로 그럴 것으로 본다). L2 실험이 이와 모순되지 않지만,
전달 타이밍을 직접 계측하지는 않았다. 공식이 7개 점에서 예외 없이 맞는다는 것이 현재 근거다.

### 최종 규칙 (관측 34건, 예외 0)

```
오답 발생  ⟺  (multi-producer 공유 lock)  ∧  T ≥ 2  ∧  floor(입력L1depth / 라운드수) ≥ 2

(P=2, 4 에서 확인. 임계 2 는 (P−1)·R ≥ P 에서 나오며 모든 P ≥ 2 에 대해 R ≥ 2 로 같다.)
```

전부 **컴파일 시점에 계산 가능한 정적 조건**이다. 실행 시간·입력값·배치 여부는 들어가지 않는다.

이 규칙은 그동안의 관측을 전부 설명한다:
- 호스트측 지연이 한 번도 도움이 안 된 이유 — 지연은 크레딧 구조를 바꾸지 않는다
- 임계가 K=32/48 사이에서 칼같이 갈린 이유 — K가 라운드 수를 양자화해서 정한다
- attention의 배치 matmul이 대표 피해자인 이유 — head 차원이 작아 라운드가 1
- FFN급 큰 K의 2D matmul이 같은 결함을 안고도 통과한 이유 — 라운드가 3


## 3g. lowered AIE IR — 결함을 직접 읽기 (2026-09-14 추가)

`--mlir-print-ir-after=iree-amdaie-lower-to-aie` 덤프에서 결함이 그대로 읽힌다.

**legacy, memtile 0_1** — producer 4개가 offset 0/64/128/192 에 64B 씩 쓰는데 **락 쌍이 하나**다:

```mlir
%lock_0_1_8 = aie.lock(%tile_0_1, 0) {init = 8 : i32}   // free   (= 4 producers × depth 2)
%lock_0_1_9 = aie.lock(%tile_0_1, 1) {init = 0 : i32}   // ready

%3 = aie.dma_start(S2MM, 2, ^bb10, ^bb12)               // 코어 0_2
^bb10:
  aie.use_lock(%lock_0_1_8, AcquireGreaterEqual, 1)
  aie.dma_bd(%buffer_0_1_6 : memref<256xi32, 1 : i32>) {len = 64 : i32}
  aie.use_lock(%lock_0_1_9, Release, 1)
  aie.next_bd ^bb11
...  S2MM 3/4/5 (코어 0_3/0_4/0_5) 는 offset 만 64/128/192 로 다르고 락은 동일 ...

%8 = aie.dma_start(MM2S, 2, ^bb25, ^bb27)               // memtile → 호스트
^bb25:
  aie.use_lock(%lock_0_1_9, AcquireGreaterEqual, 4)     // ← 결함의 전부
  aie.dma_bd(%buffer_0_1_6 : memref<256xi32, 1 : i32>) {len = 256 : i32}
  aie.use_lock(%lock_0_1_8, Release, 4)
```

**fixed, 같은 자리** — 락 쌍이 producer 마다(init 2) 생기고 consumer 전송이 BD 4개로 쪼개진다:

```mlir
%lock_0_1_8  = aie.lock(%tile_0_1, 0) {init = 2 : i32}   // 코어 0_2 전용
%lock_0_1_10 = aie.lock(%tile_0_1, 2) {init = 2 : i32}   // 코어 0_3 전용
%lock_0_1_12 = aie.lock(%tile_0_1, 4) {init = 2 : i32}   // 코어 0_4 전용
%lock_0_1_14 = aie.lock(%tile_0_1, 6) {init = 2 : i32}   // 코어 0_5 전용

%8 = aie.dma_start(MM2S, 2, ^bb25, ^bb33)
^bb25: acquire %lock_0_1_9  >= 1 ; dma_bd len=64            ; release %lock_0_1_8
^bb26: acquire %lock_0_1_11 >= 1 ; dma_bd len=64 offset=64  ; release %lock_0_1_10
^bb27: acquire %lock_0_1_13 >= 1 ; dma_bd len=64 offset=128 ; release %lock_0_1_12
^bb28: acquire %lock_0_1_15 >= 1 ; dma_bd len=64 offset=192 ; release %lock_0_1_14
```

**T 는 코어 코드에 그대로 있다.** 출력 락은 `%lock_0_2_383` 이다(코어가 release 하고, 같은
덤프의 `aie.mem(%tile_0_2)` 안 `MM2S` 가 `AcquireGreaterEqual(%lock_0_2_383, 1)` 로 받아
`64xi32` 를 memtile 로 보낸다):

| | 코어 본문 | `%lock_0_2_383` Release 횟수 |
|---|---|---|
| `b_B12` (배치) | `scf.for %arg0 = 0 to 12 step 2` 안, 본체 2개 언롤 | **12** |
| `t_K32_N768` (2D) | 루프 자체가 없는 직선 코드 | **1** |

산출물: `_local/int8_debug/tripsweep/{b_B12,t_K32_N768}_*_aie.ir`.

## 4. 덤으로 나온 fix의 한계

`t_K32_N3072`(M=32, K=32, N=3072)는 **fixed 모드에서 컴파일이 거부된다**:

```
'aie.memtile_dma' op cannot split this DMA access pattern into per-party BDs:
its outermost dimension (size 32, stride 1) ...
```

`verifyProducersTileBufferInOrder`가 "조용히 재정렬하느니 거부한다"는 설계대로 동작한 것이지만,
결과적으로 **legacy가 (unsound하게나마) 컴파일하던 shape을 fixed는 컴파일하지 못한다.**
현재 워크로드에서는 안 걸리지만 커버리지 공백으로 기록해 둔다.

## 5. 산출물

- `_local/int8_debug/gen_trip_sweep.py` — shape 생성기 (shape별 고정 seed)
- `_local/int8_debug/tripsweep/{compile,run}.sh`, `probe.py` — 컴파일/실행/IR 파서
- `_local/int8_debug/tripsweep/*_{legacy,fixed}.{vmfb,ir}`, `runs/<tag>_<mode>/o_*.npy`
- `AMDAIEObjFifoBufferization.cpp`의 `AMDAIE_LEGACY_SHARED_LOCK` 탈출구 (**미커밋**, 디버그 전용)
- `AMDAIEAssignLogicalObjectFifoDepth.cpp`의 `AMDAIE_L1_DEPTH` / `AMDAIE_L1_IN_DEPTH` /
  `AMDAIE_L1_OUT_DEPTH` / `AMDAIE_L2_DEPTH` 탈출구 (**미커밋**, 디버그 전용)
- ⚠️ L1 입력 depth ≥ 5 는 컴파일 불가(objectFifo 타입 검증 assert) — R=3 의 양성 대조군
  (D=6)을 만들 수 없었다
- `scripts/docker/run-dev.sh`: 비대화형 실행을 위해 `-it` → `-i` + tty일 때만 `-t` (미커밋)
