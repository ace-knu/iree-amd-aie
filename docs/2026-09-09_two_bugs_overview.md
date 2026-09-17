# int8 NPU 경로에 문제가 **두 개** 있다 — 전체 개요 (2026-09-09)

이 문서가 **시작점**이다. 세부 근거는 각 절이 가리키는 문서를 보면 된다.

- 문제 ① 경합: `docs/2026-09-09_core_store_vs_dma_read_race_at_vector_store_granularity.md`
- 문제 ② 배치 레인 버그: `docs/2026-09-09_batched_int8_matmul_deterministic_lane_bug.md`
- 그 앞 단계(국소화): `docs/2026-09-08_bug_localized_to_one_8x8_tile_of_proj2.md`
- 반증된 가설(공유 lock): `docs/2026-09-08_multiproducer_shared_lock_gather_defect.md`

---

## 0. 무엇을 돌리고 있나 — 이건 BERT가 아니다

`chainmix_L12`는 **합성 테스트 모델**이다(`_local/int8_debug/gen_chained_mixed.py`).
BERT의 LayerNorm/Softmax/GELU 등 실제 의미는 전부 빼고, BERT가 만들어내는
**"비배치 matmul → 배치 matmul" 교대 패턴만** 남겼다. 레이어 1개 = **matmul 2개**:

```
[32,768] --비배치 MatMul [768,768]--> proj_k
   --reshape [32,12,64] --transpose--> [12,32,64]
   --배치 MatMul [12,64,64]--> bmm_k
   --transpose/reshape--> [32,768] = y_k
```

L=12 → **NPU dispatch 24개**. `D=768`, `HEADS=12`, `HD=64`만 BERT-base 크기를 따랐다.
실제 BERT 레이어는 비배치 6개(Q,K,V,output + FFN 2개) + 배치 2개(QK^T, Attn@V)이므로
크게 축약된 캐리커처다.

**용어 주의**: 이 문서들의 "**레이어 2**"는 체인 길이 L=2가 아니라 **12개 레이어 중 인덱스 2
(3번째)** 다.

## 1. 판을 바꾼 계측 방법 — per-stage residual

원래 문제: 오차가 전파되기 때문에 레이어 2에서 틀리면 3~11도 다 틀려 보이고,
**"어디서 새로 생겼는지"** 를 알 수 없었다.

해결:

> **각 단계마다 NPU가 *실제로 받은* 입력을 CPU에 똑같이 넣고, 그 단계의 출력만 비교한다.**

`onnx.utils.extract_model`로 단계별 부분 그래프를 뽑아 쓴다. 전파분이 빠지고 그 단계에서
**새로 생긴 손상만** 남는다. 그리고 실행 간 **편차**로 성격을 가른다:

| 편차 | 뜻 |
|---|---|
| 0 | 매번 같음 → **구조적(결정론적) 버그** |
| > 0 | 실행마다 다름 → **경합(race)** |

계측 tap을 만드는 방법(중요): **기존 양자화 ONNX의 그래프 출력 목록에 기존 중간 텐서 이름을
덧붙인다.** 가중치·스케일을 건드리지 않으므로 재현되는 인스턴스를 그대로 쓸 수 있고,
컴파일 후 dispatch 24개 구조도 유지된다. **tap은 f32 텐서만 쓸 것** — int8 텐서를 그래프
출력으로 노출하면 `runtime/src/iree-amd-aie/driver/amdxdna/buffer.cc:247` assertion으로
죽는다(별건 버그).

## 2. 24개 dispatch 전수 결과 — 문제가 두 개다

### 비배치 matmul (`proj0`~`proj11`)

| 단계 | 오차 최소 | 오차 최대 | 실행간 편차 | 판정 |
|---|---|---|---|---|
| proj0,1,3,4,5,6,7,9,10,11 | 0 ~ 1.2e-07 | 동일 | **0** | **비트 수준 정확** |
| **proj2** | 0.1677 | **3532** | **3532** | 경합(심각) |
| **proj8** | 5.96e-08 | 0.6326 | 0.6326 | 경합(작음) |

### 배치 matmul (`bmm0`~`bmm11`)

| 단계 | 오차 최소 | 오차 최대 | 실행간 편차 | 판정 |
|---|---|---|---|---|
| bmm0,1,3,5,6,7 | 0.1235 ~ 0.3421 | 동일 | **0** | **계통 오차** |
| bmm2,4,8,9,10,11 | 0.195 ~ 0.3712 | 0.2128 ~ 0.4521 | 0.0024 ~ 0.081 | 계통 + 작은 경합 |

**⇒ 24개 중 12개(배치 전부)는 항상 틀리고, 8개는 실행마다 달라진다.**

지금까지 **최종 출력의 corr(0.97)만** 보고 있었기 때문에 배치 쪽 계통 오차는 **전혀 안 보였다** —
하류 int8 재양자화의 **포화(clamp)** 가 다 가렸다. corr/maxdiff는 이 버그들의 심각도 지표로
부적합하다.

---

## 문제 ① — 코어 저장 vs DMA 읽기 경합

**상태: 진단은 지지되나 미해결. 툴체인에 고칠 수단이 없음.**

### 현상

일꾼(코어)이 8×8 int32 누산기를 L1에 저장하고 "데이터 준비" lock을 release하면 DMA가
읽어간다. **저장이 아직 도착하지 않았는데 release가 먼저 반영되어**, DMA가 안 채워진
부분을 실어간다.

### 증거 — 64바이트(벡터 레지스터 폭) 단위 경계

`proj2`의 손상은 항상 `rows 24~31 × cols 520~527`(8×8 타일 하나)이고, 행별로 보면
**모든 손상 그룹이 정확히 같은 패턴 `XXpp....`**:

| 행 | 상태 | 함의 누산기 (정상 ≈4.3e4) |
|---|---|---|
| 24, 25 | 쓰레기 | 6.89e8 / 6.71e8 (그룹 간 동일) |
| 26, 27 | 부분 또는 **정확히 0** | 3.5e3 ~ 7.8e3 |
| 28~31 | 정상 | 2.6e4 ~ 4.5e4 |

8열 × int32 = 행당 32B → **2행 = 64B = AIE2 벡터 레지스터 폭**. 생성 코드에도 정확히
그 단위로 저장 명령 4개가 있다:

```
aa: vst bmll0, [p3, #0x0]     ← 0~63B    → 행 24-25
ae: vst bmlh0, [p3, #0x40]    ← 64~127   → 행 26-27
b4: vst bmhl0, [p3, #0x80]    ← 128~191  → 행 28-29
ba: vst bmhh0, [p3, #0xc0]    ← 192~255  → 행 30-31
```

마지막 저장부터 release까지 **명령 10개**뿐이고, 중간 `acq`는 lock이 available이면 즉시
통과해 배리어가 못 된다.

### 확정한 것

- **입력은 무관**: 두 피연산자가 매 실행 비트 동일(`y1` 13/13 동일, 가중치는 상수)인데
  출력만 갈린다 ⇒ 손상은 dispatch **내부**에서 생성.
- **누산 결과가 아니다**: 쓰레기가 함의하는 누산기 6.89e8은 그 타일 이론상한(1.9e6)의 **360배**,
  모델 전체 상한(12.4e6)의 **55배**.
- **정적으로 실패 dispatch를 구별할 수 없다**: 레이어 0/1/2의 dispatch는 코어 ELF 32개 전부,
  CDO init/enable/elfs가 **동일**하고, 제어 코드도 **16줄만 다르며 전부 호스트 버퍼 주소**.
- **순서가 아니라 시간이 문제**:

| 개입 | 시간 증가 | 수치 중립 | 버그 수정 |
|---|---|---|---|
| `sched.barrier`(순서만) | ❌ 0사이클 | ✅ **비트 동일 증명** | ❌ 안 됨 |
| busy-wait 루프 | ✅ | ❌ clean 0.9736→0.9031 | ✅ 됨 |
| 직선 volatile store | ✅ | ❌ clean이 비결정론화 | ✅ 됨 |

### 왜 고칠 수 없나

1. **툴체인에 수단이 없다.** Peano는 aie2p에서 **인라인 asm 미지원**(`__asm__ volatile("nop")`
   한 줄에도 `unable to translate instruction: call`로 죽음), aie2p 빌트인 141개 중 nop/지연
   **없음**, Chess `chess_separator()` 계열은 Peano 헤더에서 **빈 매크로**,
   `sched.barrier`는 순서 전용(0사이클).
2. **어떤 사이클 추가도 시스템을 흔든다.** 직선 store 8개만 넣어도 **20/20 완벽했던 clean
   인스턴스가 6회 중 5개 패턴으로 깨진다.** ⇒ 한 곳의 결함이 아니라 **handoff 전반의 타이밍
   여유가 거의 0**이고, 지연은 경합을 **옮기는** 것일 뿐 fix가 못 된다.

또한 Peano 자신의 lock 헤더에 힌트가 있다 — 버퍼 포인터로 메모리 의존성을 표현할 수 있는
`release(void*, id, val)` 변형을 만들어두고 **"The ptr arg is intentionally not used"**,
`sched_barrier`는 `done()`에만 쓰고 `release()`에는 없다.

### 이전 버그와의 관계

버그 #1(batch-0 출력 0)을 `1600078`의 **첫 release *뒤* 10k busy-wait**로 우회했는데,
release **뒤**는 저장 가시성에 아무 도움이 안 된다. **근본 원인이 delay로 가려진 채 살아남아
delay가 못 덮는 지점에서 다시 나타난 것**으로 보인다 — 사용자가 처음 물은 "버그 #1 재발
아니냐"와 일치.

### 반증된 가설 (기록용)

- **memtile 출력 gather의 multi-producer 공유 lock**: 구조적으로 unsound한 건 사실이고
  (`AMDAIEObjFifoBufferization.cpp`가 lock init을 `producers × depth`로 스케일하고 lock을
  타일 단위로 배정) 버그 #1 증상과도 맞지만, **producer별 lock + consumer BD 분할 패치를
  실제로 구현·정적 검증까지 했는데 베이스라인의 오답 해시 5개가 그대로 재현**됐다 ⇒ 원인 아님.
  (패치는 별건 잠재 결함으로 워킹트리에 남아 있음)
- **`--iree-amdaie-num-rows≠4`로 producer 수 줄이기**: `num-rows≠4`가 **에러 없이 조용히 틀린
  결과**를 내는 별개 버그가 있어서(L=1에서 이미 corr 0.9978→0.8713) 증거로 쓸 수 없다.
- **L 의존성**: 옛 생성기가 L마다 입력·스케일을 같이 바꿨던 confound. 고친 뒤엔 L=10~13이
  전부 결정론적. "L=13이 최악", "L=11은 면역" 같은 서술은 근거 없음.
- **TCT sync / fence 회계**: 24개 dispatch 전부에서 push 20 = TCT 대기 20이 `(dir,ch,col,row)`
  단위로 정확 일치, BD-id 재사용-전-미대기 0건. 깨끗함.

---

## 문제 ② — 배치 int8 matmul의 결정론적 레인 버그

**상태: 규칙 정확히 특정. 원인 코드는 아직 안 봄. ← 다음 작업 대상**

### 손상 규칙

출력 `[12 head, 32행, 64열]`, 손상 항상 **128개 원소**:

```
head  8, row%8 = 0   -> 32
head 10, row%8 = 2   -> 32
head 10, row%8 = 4   -> 32
head 10, row%8 = 6   -> 32
열 분포: 128개 전부 col%8 == 0
```

8×8 타일 좌표로:

| head | 타일 내부에서 틀린 (row, col) | 개수 |
|---|---|---|
| **8** | **(0, 0)** — 타일의 첫 원소 | 4 row-tile × 8 col-tile = 32 |
| **10** | **(2, 0), (4, 0), (6, 0)** | 4 × 8 × 3 = 96 |

- **128개 전부 `col % 8 == 0`** — 8칸 벡터 그룹의 **첫 번째 레인**
- 그 head의 **모든 타일에서 같은 레인**이 틀림 ⇒ 32개 코어 전부 동일 증상
- 값은 절반가량이 **정확히 0**(NPU exact-zero 68 vs 기준 4), 나머지는 다른 틀린 값

예 (head 8, row 0의 손상 열 `{0,8,...,56}`):

```
NPU   : 0        0        0       0       0       0        0        0
기준값: 0.0667  -0.0518   0.0358  0.0559  0.0856  0.0833  -0.1235  -0.0028
```

### 순수 구조적 버그로 확정

| 확인 층위 | 결과 |
|---|---|
| 실행 간 | 손상 마스크가 **비트 단위 동일** |
| 12개 레이어 간 | **동일 좌표** (레이어별 가중치·스케일이 다 다른데도) |
| 인스턴스 간 | 별도 입력·스케일의 clean 인스턴스에서도 **한 글자도 안 다름** |

⇒ **값에 전혀 의존하지 않는 코드 생성 버그.** 문제 ①(값·타이밍 의존)과 성격이 정반대.

### NPU가 틀린 것이 맞다

layer 0의 배치 matmul을 numpy로 int8 정확 계산(int8→int32 정확 누산 → `× s_a·s_w`)해 비교:

| | exact와의 차이 |
|---|---|
| **onnxruntime** | **1.0e-08** ✅ |
| **NPU** | **0.1235** ❌ |

계측 방법 문제가 아니다.

### 중요한 정정

지금까지 **"clean"으로 분류한 인스턴스들은 경합이 없었을 뿐**이고, 배치 matmul은 거기서도
똑같이 틀리고 있었다.

---

## 3. 두 문제 비교 — 왜 ②부터 파는가

| | ① 경합 | ② 배치 레인 버그 |
|---|---|---|
| 영향 범위 | 24개 중 8개 | **배치 12개 전부** |
| 재현성 | 확률적(11/13 등) | **매번 100%** |
| 값 의존 | 있음 | **없음** |
| 정적 근거 | 없음(코드·설정 전부 동일) | **규칙 정확히 특정** |
| 고칠 수단 | **툴체인에 없음** | 미조사 |
| 실제 BERT에서 | — | **attention 핵심**(QK^T, Attn@V) |
| 추적 난이도 | 개입이 시스템을 흔듦 | **결정론적 → 수월** |

## 4. ② 조사 계획

1. **왜 head 8과 10인가.** batch=12가 하드웨어에 어떻게 매핑되는지 확인.
   출력 `[12,32,64]`는 M=32를 코어 4행, N=64를 8컬럼으로 나누고(코어당 8×8 타일)
   **batch 12개는 시간축으로 반복**되는 구조로 보인다. 그렇다면 손상은 **특정 반복 회차
   (head 8, 10)의 특정 레인**이라는 뜻 ⇒ **배치 루프가 언롤되어 특정 언롤 사본만 잘못
   생성됐다**는 가설이 가장 유력하다. head 8은 레인 1개, head 10은 3개인 **비대칭**도
   언롤 사본별 차이라면 설명된다.
2. **생성 코드 확인.** 배치 matmul dispatch의 코어 코드를 디스어셈블해
   (a) 배치 루프가 언롤되었는지, (b) 회차별로 코드가 다른지,
   (c) 8×8 누산기의 레인 저장/추출(`vst bmll0/bmlh0/bmhl0/bmhh0` 및 `aievec.matmul` 결과
   재배치)에서 **첫 레인(`col%8==0`)이 누락·덮어써지는지** 본다.
3. `col%8==0`이 가장 강한 단서다 — 벡터 레인 0의 추출/저장 경로를 우선 볼 것.

## 5. 재현 자료 (모두 gitignored)

| 용도 | 경로 |
|---|---|
| 재현되는 인스턴스(경합) | `_local/int8_debug/out/chainmix_L12_int8.{onnx,mlir}` + `chainmix_L12_x.npy` |
| clean 인스턴스(경합 없음) | `chainmix_L12_s0_int8.{onnx,mlir}` + `chainmix_L12_s0_x.npy` |
| `y1`+`proj2` tap (경합 계측용) | `chainmix_L12_y1p2_int8.*`, `rowsweep/L12_y1p2.vmfb`, `rowsweep/y1p2/` |
| 전 `proj` tap | `chainmix_L12_allproj_int8.*`, `rowsweep/L12_allproj.vmfb`, `rowsweep/allproj/` |
| 전 `bmm` tap | `chainmix_L12_allbmm_int8.*`, `rowsweep/L12_allbmm.vmfb`, `rowsweep/allbmm/` |
| clean 인스턴스 전 `bmm` tap | `chainmix_L12_s0_allbmm_int8.*`, `rowsweep/L12_s0_allbmm.vmfb`, `rowsweep/s0_allbmm/` |
| 베이스라인 40회 (경합) | `rowsweep/retest_orig/` (5개 패턴) |
| 실행 파일 덤프 | `--iree-hal-dump-executable-files-to` → 코어 ELF, CDO, `npu_inst.txt` |

생성기: `_local/int8_debug/gen_chained_mixed.py <L> [SEED]`
(2026-09-08에 rng 스트림 분리 — weight는 `default_rng(0)`, 입력/calibration은
`default_rng(1000+SEED)`. 출력은 항상 `chainmix_L{N}_s{SEED}_*`)

컴파일러 실험 블록(전부 env-var opt-in, 기본 빌드 불변,
`compiler/plugins/target/AMD-AIE/aie/AMDAIECoreToStandard.cpp`):
`AMDAIE_EXPERIMENTAL_PRE_RELEASE_DELAY`(busy-wait),
`AMDAIE_EXPERIMENTAL_PRE_RELEASE_NOPS`(직선 store),
`AMDAIE_EXPERIMENTAL_PRE_RELEASE_BARRIER`(sched.barrier).
