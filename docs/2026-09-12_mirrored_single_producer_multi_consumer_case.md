# mirrored 케이스(1 producer / N consumer) — 정적 확인, 동적 미재현 (2026-09-12)

`65069ce` 커밋이 명시적으로 남겨둔 결함:

> The mirrored single-producer/multi-consumer case keeps the scaled single pair --
> it has the same weakness, but is not exercised by the workloads this was written for.

이걸 실제로 재현해 보려 한 기록. **결론: 패턴은 실재하고 수정판에도 그대로 남아 있음을 정적으로
확인했으나, 40런에서 발현시키지 못했다.** 그리고 그 과정에서 별개의 유용한 사실을 하나 확정했다.

## 1. 어디서 방출되는가 — `num-rows=3`

보유한 IR 51개(2D/배치 matmul 전 스윕 + chainmix/mm/bmm bf16 full-pipeline 덤프)를
**디스패치 함수 단위로** 훑었다.

⚠️ 첫 시도는 틀렸다. dump 전체를 한 덩어리로 보면 SSA 이름이 함수마다 재사용되기 때문에
chainmix 에서 mirrored 가 75개 나오는 것처럼 보인다. `func.func` 로 쪼개면 **chainmix 에는 없다.**

정확한 결과: mirrored 패턴은 **`--iree-amdaie-num-rows=3` 에서만** 나온다.

```
b_B12_r3 / b_K48_r3 (batch_matmul 12x32x64xK, num-rows=3)
  L2 fifo %8 (또는 %16): memref<1536xi8, 1>, depth 2
    ← 채우는 connection 1개 (shim → L2, circular repeat 12)
    → 소비 connection 3개 (L2 → 코어 3개의 L1, 각 512xi8 disjoint slice)
    lock 쌍 1개, producer lock init = 6 = 3 consumers × depth 2
```

즉 1536B 버퍼 하나를 3개 DMA 가 512B 씩 **서로 겹치지 않게** 나눠 읽어가는 distribute 이고,
producer 는 `AcquireGreaterEqual(free, 3)` 로 "release 가 3번 있었다"만 확인한다 — N-producer
쪽과 정확히 대칭인 신원 소실이다.

**num-rows=2, 4 에서는 나오지 않는다.** 32행이 2·4로는 균등하게 쪼개져 코어마다 별도 L2 버퍼(1:1)를
받지만, 3으로는 안 나뉘어 컴파일러가 버퍼 하나를 공유시키기 때문이다.

## 2. 수정판이 덮지 않는다 — 정적 확인

`b_B12_r3`, `b_K48_r3` 각각 legacy/fixed 두 모드의 IR 을 비교했다:

| | mirrored fifo | consumer connection | lock 쌍 | producer lock init |
|---|---|---|---|---|
| legacy | 있음 | 3 | **1** | **6** |
| fixed | 있음 | 3 | **1** | **6** |

**완전히 동일하다.** 커밋 메시지 그대로, 수정은 N-producer 쪽만 덮는다.

⇒ 방법론상 중요한 결과: **legacy vs fixed 비트 비교로는 mirrored 를 절대 검출할 수 없다**(양쪽이
똑같이 결함이다). 기준은 `num-rows=4` 의 fixed 출력이어야 한다 — 정수 matmul 이라 i32 누산이
타일링 순서와 무관하게 정확하므로 비트 동일해야 한다(실측으로 확인됨).

## 3. 실측

기준: `b_K48_fixed` (num-rows=4, per-producer lock) 출력.

| 설정 | 모드 | 출력/런 | 타임아웃 | 출력 종류 | 기준과 다름 |
|---|---|---|---|---|---|
| P=3, D<sub>in</sub>=2 (N-producer 결함 살아있음) | legacy | 13/20 | **7** | 1 | 0 |
| P=3, D<sub>in</sub>=2 (mirrored 만) | fixed | 20/20 | 0 | 1 | 0 |
| P=3, D<sub>in</sub>=4 (mirrored 만) | fixed | 20/20 | 0 | 1 | 0 |

- 타임아웃은 `ert state 8` = `ERT_CMD_STATE_TIMEOUT`(60s TDR). 장치는 매번 복구되고 다음 런이
  계속 진행된다 — 영구 wedge 가 아니다.
- **완료된 런은 legacy 든 fixed 든 전부 기준과 비트 동일**하다. 오답은 한 건도 없다.

### 3.1 num-rows=3 의 데드락은 mirrored 가 아니다

타임아웃은 **legacy 에서만** 나오고 fixed 에서는 0 이다. mirrored fifo 는 두 모드가 동일하므로,
이 데드락은 **기존 N-producer 공유 lock 결함**의 증상이다. num-rows=3 에서 그 결함은 오답이 아니라
**교착**으로 나타난다.

⇒ 예전에 기록해 둔 "`num-rows≠4` 는 조용히 틀린 결과를 내는 별개 버그라 스윕 증거로 못 쓴다"는
메모는 **갱신되어야 한다.** 그 증상의 정체는 N-producer 결함이고, `65069ce` 로 **이미 고쳐졌다**
(fixed 에서 20/20 무결·비트 정확). num-rows 스윕은 이제 쓸 수 있다.

### 3.2 mirrored 자체는 발현되지 않았다

fixed 모드(= N-producer 쪽이 막힌 상태)에서 소비자 L1 깊이를 2 와 4 로 바꿔가며 40런을 돌렸으나
타임아웃 0, 오답 0 이었다.

발현 조건이 안 갖춰진 이유는 **T 가 아니다** — 이 L2 버퍼는 `circular_dma_cpy_nd ... [12, 256]`
로 디스패치당 12번 다시 채워지므로 T=12 다. 남은 설명은 **3개의 소비자 DMA 가 서로 한 세대
이상 벌어지지 못한다**는 것이고, 목적지 L1 깊이를 4로 올려도 그대로였다.

⚠️ L1 입력 depth ≥ 5 는 컴파일이 안 된다(objectFifo 타입 검증 assert)라 더 올려보지 못했다.

## 4. 현재 상태 정리

| 항목 | 상태 |
|---|---|
| mirrored 패턴이 실제로 방출되는가 | ✅ 예 — `num-rows=3` |
| 수정판이 덮는가 | ❌ 아니오 — legacy/fixed IR 동일 |
| 정적으로 결함인가 | ✅ 예 — lock 쌍 1개, init = 3×depth, 소비자 3개 |
| 동적으로 재현되는가 | ❌ **40런 미재현** |
| num-rows=3 의 교착은 이것인가 | ❌ 아니오 — N-producer 결함, `65069ce` 로 해결됨 |

## 5. 재현하려면 다음에 무엇이 필요한가

1. **소비자 DMA 간 skew 를 키울 수단.** 목적지 L1 깊이는 4까지가 한계였다(5+ 컴파일 불가).
   세 소비자의 하류 소비 속도를 비대칭으로 만드는 shape(예: 코어별 연산량이 다른 구성)이 필요하다.
2. **mirrored 를 더 흔하게 방출시키기.** 지금은 `num-rows=3` 이라는 특수 설정에서만 나온다.
   `32 / P` 가 나누어떨어지지 않는 다른 (M, P) 조합을 찾으면 표본이 늘어난다.
3. **소비자별 lock 쌍 패치.** N-producer 쪽 수정을 대칭으로 확장한 뒤, 그 패치가 있을 때/없을 때를
   비교하면 legacy-vs-fixed 와 같은 격리 비교가 가능해진다. 지금은 비교 기준 자체가 없어
   `num-rows=4` 결과에 의존하고 있다.

## 6. 산출물

- `_local/int8_debug/tripsweep/b_{B12,K48}_r3{,in1,in4}_*` — IR, vmfb, 런 출력
- 분석: 디스패치 단위 mirrored 스캔, lock init/소비자 수 대조

---

## 부록 (2026-09-14) — 락 프로토콜 시뮬레이션

`_local/int8_debug/tripsweep/sim_lock2.py`, `sim3.py` (목표: `target_col1.json`).
free=8 / acquire(ready,4) / depth-2 핑퐁 / 행 단위 쓰기·읽기를 그대로 구현하고,
타이밍 파라미터만 탐색해 실측(열 1, 12 batch × 4 slice × 8 행 = 384셀)과 대조했다.

- 출력 락만 모델링: **357/384 (93%)**
- 입력 공급(shim 도착 시각)까지 추가: **366/384 (95%)**

재현되는 것: batch 0 의 미기록 슬라이스 2개, batch 1 완전 정상, batch 2 의 −2 stale,
batch 3~9 의 "슬라이스 하나만 +2" 정상 상태, batch 10·11 꼬리.

재현 안 되는 것(18셀): 빠른 코어가 gen0·gen1 에는 정시였다가 gen2 부터 +2 로 올라서는 전이,
그리고 batch 3·6·10 의 **1행짜리 찢어진 타일**. 둘 다 세대 하위 단위의 타이밍
(코어 연산 지연, 입력 DMA 지터)에 달려 있어 이 추상화 수준에서는 안 잡힌다.

⇒ **프로토콜만으로 구조(거리 ±2, 열당 코어 1개, 정상 상태)는 설명되지만
정확한 순서까지는 재현되지 않는다.** 더 가려면 코어별 연산 지연을 실측해 넣어야 한다.
