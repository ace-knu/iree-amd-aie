# 두 번째 버그를 하나의 8×8 타일로 국소화 — 2026-09-08 (TCT sync 조사에서 출발)

`docs/2026-09-08_multiproducer_shared_lock_gather_defect.md`에서 공유-lock 가설이 컴파일러
패치로 반증된 뒤, 그 negative가 지목한 다음 후보(**호스트 측 TCT sync / fence**)를 조사한
기록이다. TCT 쪽은 깨끗했지만, 그 과정에서 **버그를 단일 8×8 타일까지 국소화**했고 그 결과
지금까지의 문제 인식 자체가 바뀌었다.

---

## 1. TCT sync / fence 회계는 깨끗하다

가설: 호스트가 NPU 출력 완료를 다 기다리지 않고 버퍼를 읽는다면, "stale vs fresh"라는 소수의
이산 상태가 나오고 memtile gather 재구성에 영향받지 않는다 — 관측과 일치.

`chainmix_L12`의 control code(`--mlir-print-ir-after=iree-amdaie-controlcode-lowering`)를
덤프해 전수 검사했다. dispatch 1개당 `push_to_queue` 20개, `tct_sync` 3개:

```
tct_sync {dir=0(S2MM), ch=0, col=0, col_num=8, row_num=1}   → 8 TCT
tct_sync {dir=1(MM2S), ch=0, col=0, col_num=8, row_num=1}   → 8 TCT
tct_sync {dir=1(MM2S), ch=1, col=0, col_num=4, row_num=1}   → 4 TCT
                                                      합 20 = push 20
```

- `push_to_queue`는 전부 `async`이고, `AMDAIEControlCodeToTransaction.cpp`에서
  `issueToken = static_cast<bool>(op.getAsyncToken())` → 20개 모두 토큰 발행.
- **24개 dispatch 전부에서 `(dir, ch, col, row)` 단위 커버리지가 정확히 일치**(스크립트로 전수
  확인, mismatch 0). 즉 control-code 레벨의 under-wait는 없다.
- `AMDAIEFoldDmaWaits`는 "큐의 마지막 wait만 남긴다"는 최적화지만 `maxQueueSize`와
  `isDuplicateBdId`를 모두 확인하고, 큐 키가 `(tile, connection)`으로 하드웨어 큐
  `(tile, channel, dir)`보다 **더 세밀**하므로 필요한 것보다 적게 기다리는 방향으로는 틀리지
  않는다.
- **dispatch 내 BD id 재사용-전-미대기 이벤트 0건**(전수 검사). col 0에서 쓰이는 BD id는
  0/1/2뿐으로 고갈도 없다.

미해결로 남긴 세부: `repeat_count=N`(우리 경우 12 또는 2)인 push가 TCT를 1개 내는지 N개 내는지
aie-rt 헤더 주석("Number of times the **task** is to be repeated" / "issue a token when
**task** is completed")만으로는 확정하지 못했다. 다만 아래 §2의 국소화 결과가 이 축의 우선순위를
크게 낮춘다.

## 2. 레이어별 tap으로 최초 분기 지점을 찾다

블랙박스 재실행을 늘리는 대신, **어느 레이어에서 처음 갈리는지** 국소화했다. 재현되는
인스턴스는 옛 것이라 scale을 재생성할 수 없으므로, **기존 양자화 ONNX에 중간 텐서를 추가
출력으로 노출**해 scale과 `x`를 그대로 보존했다(가중치·스케일 일절 미변경).

- 1차: `y0`~`y10`(레이어 경계) → 컴파일 후 dispatch 24개 유지 확인 후 18회 실행
- 2차: `proj0..3`(비배치 matmul 출력)과 `bmm0..3`(배치 matmul 출력) → 20회 실행

### 결과 — 최초 분기는 `proj2`, 그것도 8×8 타일 하나

| tap | 서로 다른 그룹 | NPU `|max|` (CPU 기준값) | 판정 |
|---|---|---|---|
| `proj0` | 1 | 0.2784 (0.2784) | 정확, err 2.2e-07 |
| `bmm0` | 1 | 0.2741 (0.2741) | 정상 |
| `proj1` | 1 | 0.3072 (0.3072) | 정상 |
| `bmm1` | 1 | 0.2853 (0.2853) | 정상 |
| **`proj2`** | **5** | **3532** (0.3738) | **손상** |
| `bmm2` | 4 | 0.4436 (0.2898) | 포화 후 전파 |
| `proj3` | 4 | 0.3944 (0.3944) | 전파 |
| `y_final` | 5 | 0.9565 (1.0472) | 전파 |

`proj2`의 손상 범위는 **항상 정확히 `rows 24~31 × cols 520~527`** — 8×8 타일 하나다.
8×8은 int8 `aievec.matmul`의 명령 타일(8x8x8)이자 패킹 내부 크기(`packedSizes=[...,8,8,8]`).

CPU 기준값 대비:

| 그룹 | 실행 수 | 그 타일 maxerr | 타일 밖 maxerr | corr |
|---|---|---|---|---|
| `db9d09d348` | 10 | **3531.95** | 0.1677 | 0.0047 |
| `e480c9b332` | 4 | **3531.97** | 0.1677 | 0.0047 |
| `93e62e8581` | **3** | **0.026** ✓ | 0.1677 | **0.9958** |
| `d972b89ece` | 2 | 3531.96 | 0.1677 | 0.0047 |
| `3ef3ff38ae` | 1 | 3531.95 | 0.1677 | 0.0047 |

**20회 중 17회에서 이 타일이 완전한 쓰레기값**(±3532, 정상은 ±0.2)이고, 올바른 건 3회뿐이다.
타일 밖 오차는 모든 그룹에서 동일(0.1677) — 결정론적인 int8 양자화 오차다.

전파 경로도 정확히 맞는다: `cols 520-527`은 head 8(512-575) 안에 있어 reshape/transpose 후
`bmm2`의 **head 8만** 손상되고(`ax0:1/12[8]`), 그 다음부터 전 텐서로 퍼진다.

## 3. 그래서 지금까지의 문제 인식이 바뀐다

1. **"작은 비결정론적 교란"이 아니다.** 특정 8×8 타일이 대부분의 실행에서 심각하게 손상된다.
   최종 출력이 corr 0.97로 "가볍게 틀린" 것처럼 보였던 이유는 **하류 int8 재양자화의 포화가
   심각도를 가렸기 때문**이다(`bmm2`의 `|max|`가 0.4436로 잘려 나온다).
   → 그동안 써온 corr/maxdiff는 이 버그의 심각도 지표로 부적합하다.
2. **"결정론적"이었던 인스턴스는 진짜로 정확하다.** `L12_s0`을 같은 방식으로 tap해서 6회
   실행한 결과 모든 tap이 1그룹이고 NPU `|max|`가 기준값과 일치(쓰레기 타일 없음).
   즉 §7(seed 스윕)의 결정론 결과는 "정확함"을 뜻했다 — 이 우려는 해소됐다.
3. **트리거는 코드 구조가 아니라 수치다.** dirty/clean 두 인스턴스의 MLIR은 숫자 리터럴 외
   완전 동일하고 `LowerToAIE` 구조 카운트도 같다. 그런데도 한쪽만 손상된다.
   그리고 같은 인스턴스 안에서도 `proj0`/`proj1`은 정확하고 `proj2`만 손상된다 — 12개 비배치
   dispatch는 구조가 동일하므로, 위치나 구조가 아니라 그 dispatch에 흐르는 **값**이 조건이다.

## 4. 손상된 누산기의 정량

NPU는 `int8 matmul → int32 acc → acc × (s_a·s_w)`로 계산한다. ONNX에서 뽑은 스케일:

- `y1_scale` = 0.00243499, `w0_2_scale` = 0.00210634 → `s_a·s_w` = **5.13e-6**
- `proj2_scale`(이후 양자화용) = 0.00325954

따라서:

| | proj2 값 | 함의된 int32 acc |
|---|---|---|
| 정상 (최대) | 0.374 | ≈ **72,000** (K=768 int8 MAC로 타당) |
| 쓰레기 (최대) | 3532 | ≈ **6.9e8** (정상의 ~9,500배, int32 범위 내) |

그리고 쓰레기 값이 실행마다 **거의 같지만 완전히 같지는 않다**
(1708.5433 / 1708.5427 / 1708.558 → acc 차이 ~수천). int32 누산은 정확하므로, 이는
**거의 고정된 거대한 오염 항(≈3.3e8) + 실행마다 변하는 정상 부분합**의 합으로 읽힌다.
즉 누산기가 남은 데이터를 물고 시작하는 형태이지, 무작위 메모리 쓰레기는 아니다.

단순한 "누산기 zero-init 누락"으로는 `proj0`/`proj1`이 정확한 것이 설명되지 않는다.

## 5. 다음 단계 (우선순위)

1. **`proj2`를 계산하는 dispatch의 8×8 누산기 초기화/에필로그 코드를 직접 본다.**
   손상 타일이 항상 같은 자리이므로 대상 코어/타일이 특정된다: M=32를 코어 4행이 나누면
   `rows 24-31`은 마지막 코어 행, `cols 520-527`은 해당 N-타일. 그 타일의
   `linalg.fill`/누산기 초기화와 requantize 에필로그를 벡터화 경로에서 확인.
2. **그 타일의 int8 *입력*이 올바른지 tap한다** (`y1_QuantizeLinear_Output`,
   `w0_2` 양자화 가중치). 입력이 이미 오염됐는지 / 누산 단계에서 오염되는지 가른다.
   입력이 깨끗하면 누산/에필로그 문제로 확정된다.
3. **정상 3회 vs 손상 17회의 차이가 무엇인지** — 손상이 거의 결정론적이므로, 3회만 정상인
   조건을 찾으면 트리거가 드러난다. 같은 타일의 값이 clamp 경계 근처인지 확인
   (메모리의 "Peano `G_FMINIMUM` legalization 크래시(int8 clamp 경로)"와 연결될 수 있음 —
   int8 clamp lowering이 이 툴체인에서 취약하다는 선례가 있다).
4. `repeat_count>1`의 TCT 발행 개수(1개 vs N개)는 미확정으로 남겨두되, §2 결과로 우선순위는 낮다.

## 재현 자료 (모두 gitignored)

- `_local/int8_debug/out/chainmix_L12_taps_int8.{onnx,mlir}` — `y0..y10` tap (레이어 경계)
- `_local/int8_debug/out/chainmix_L12_taps2_int8.{onnx,mlir}` — `proj0..3`/`bmm0..3` tap
- `_local/int8_debug/out/chainmix_L12_s0_taps2_int8.{onnx,mlir}` — clean 인스턴스 대조군
- `_local/int8_debug/out/rowsweep/L12_taps.vmfb`, `L12_taps2.vmfb`, `L12_s0_taps2.vmfb`
- 실행 출력: `rowsweep/taps/`(18회), `rowsweep/taps2/`(20회), `rowsweep/taps2_s0/`(6회)

tap 추가는 가중치/스케일을 건드리지 않는다 — 기존 양자화 ONNX의 그래프 출력 목록에만
기존 텐서 이름을 덧붙이는 방식이다. dispatch 24개 구조가 유지되는 것을 컴파일 후 확인했다.
