# 2026-09-18 — 태초(matmul만 NPU) vs 최종(NPU 7 / CPU 2) 성능 비교

> `docs/2026-09-17_attention_performance_comparison.md` (A/B/C 비교)의 연장. 여기서는
> **출발점**, 즉 attention 에서 **matmul 류만 NPU 로 돌리던 원래 배포 구성**을 실제로
> 빌드해 최종본과 같은 조건에서 쟀다. **코드 변경 없음.**

## 0. 한 줄 답

**태초 → 최종: wall time −8.30 %, 호스트 CPU time −74.87 %, throughput +9.03 %, 디스패치 15 → 9.**
(12/12 라운드 전부 최종본이 빠름, 95 % CI `[−8.54 %, −8.06 %]`)

단, 그 이득은 **한 방향으로 단조롭게 쌓인 게 아니다** — §3 참고.

## 1. 비교 대상

| | O1 = 태초 | O2 = 중간 | C = 최종 |
|---|---|---|---|
| 모델 | `attn0_kpadq_int8.mlir` (head-split 없음) | 같음 | `attn0_hsm_int8.mlir` (head-split + mask 상수화) |
| 융합 플래그 | `--iree-dispatch-creation-no-fuse-into-contraction-conv-roots` **켬** | **끔**(융합 켬) | 끔 |
| softmax ukernel | 없음 | 없음 | `--iree-amdaie-enable-ukernels=softmax` |
| NPU 에 있는 것 | **matmul 6 개만** | matmul 6 + **재양자화** | matmul 6 + 재양자화 + **softmax** |
| 디스패치 | **15 (NPU 6 / CPU 11)** | 12 (NPU 6 / CPU 6) | **9 (NPU 7 / CPU 2)** |
| vmfb MD5 | `158c15ef8192bf9fe17d3fa6e903feaa` | `b35a583d03e74e02b0cc2e51f9fe0e1c` | `1b8490a5b86d86bc89d48ff1f0cefe18` |

O2 를 끼운 이유: O1 → C 사이에 바뀐 것이 **네 가지**(재양자화 융합 / head-split 로
transpose 소멸 / softmax NPU / d1 broadcast 제거)라 한 덩어리로 보면 해석이 안 된다.
O2 는 그중 **재양자화 융합만** 적용한 지점이다.

### 1.1 디스패치 목록

**O1 (태초) — NPU 6 / CPU 11**

| # | 장치 | 디스패치 | 역할 |
|---|---|---|---|
| d0 | CPU | `elementwise_32x768_f32xi8` | 입력 quantize + K 패딩 |
| d1–d3 | **NPU** | `batch_matmul_1x32x768x832` ×3 | Q / K / V projection (**i32 출력**) |
| d4–d6 | CPU | `elementwise_transpose_*_i32xi8` ×3 | Q/K/V 재양자화 **+ transpose** |
| d7 | **NPU** | `batch_matmul_12x32x32x64` | QKᵀ (i32 출력) |
| d8 | CPU | `elementwise_12288_i32xi8` | QKᵀ 재양자화 |
| d9 | CPU | `softmax_12x32x32xf32_generic` | softmax |
| d10 | **NPU** | `batch_matmul_12x32x64x32` | PV (i32 출력) |
| d11 | CPU | `elementwise_transpose_12x32x64_i32xi8` | PV 재양자화 + transpose |
| d12 | **NPU** | `batch_matmul_1x32x768x768` | output projection (i32 출력) |
| d13 | CPU | `elementwise_24576_i32xi8` | proj 재양자화 |
| d14 | CPU | `elementwise_32x768_f32xi8xf32` | 최종 dequant + bias |

**O2 — NPU 6 / CPU 6**: 재양자화 4 개(d8, d13 그리고 d4–d6·d11 의 재양자화분)가 matmul
안으로 흡수되어 CPU 는 입력 quantize, transpose 3 개, softmax, 최종 dequant 만 남는다.

**C (최종) — NPU 7 / CPU 2**: transpose 3 개가 head-split 로 소멸, softmax 가 NPU,
d1 broadcast 가 Q/K/V 로 흡수. CPU 는 **입력 quantize+pad** 와 **최종 dequant+bias** 둘뿐.

## 2. 결과

방법은 A/B/C 비교와 동일하다: `iree-benchmark-module`, **12 라운드 × 3 구성**, 6 순열
2 바퀴로 순서 균형(각 구성이 1/2/3 위치에 4 회씩, 실측 확인), 라운드당
`--benchmark_repetitions=20` → 구성당 **240 repetition**, 캠페인 전체를
`with-npu-lock.sh` 하나로 감쌈, Release 빌드, 동일 런타임 옵션.

### 2.1 절대값

| 구성 | 디스패치 | wall mean | median | p95 | p99 | sd | CV | 호스트 CPU | throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **O1** 태초 | 15 (6/11) | **14.9792 ms** | 14.9503 | 15.1424 | 15.4366 | 0.2220 | **1.48 %** | **4.6704 ms** | 66.77 inv/s |
| **O2** 중간 | 12 (6/6) | **13.2513 ms** | 13.2425 | 13.3960 | 13.4965 | 0.0875 | 0.66 % | 3.4056 ms | **75.47 inv/s** |
| **C** 최종 | **9 (7/2)** | 13.7362 ms | 13.7286 | 13.8558 | 13.9267 | 0.0737 | **0.54 %** | **1.1738 ms** | 72.80 inv/s |

### 2.2 쌍체 대조

| 대조 | wall mean | 95 % CI | 라운드 | 호스트 CPU | throughput |
|---|---:|---|---|---:|---:|
| **O1 → O2** 재양자화 융합 | **−1.7279 ms (−11.54 %)** | `[−11.83, −11.24]` | **12/12** | −27.08 % | **+13.02 %** |
| **O2 → C** head-split + softmax NPU + d1 제거 | **+0.4850 ms (+3.66 %)** | `[+3.54, +3.78]` | 0/12 | **−65.53 %** | −3.53 % |
| **O1 → C** 태초 → 최종 | **−1.2429 ms (−8.30 %)** | `[−8.54, −8.06]` | **12/12** | **−74.87 %** | **+9.03 %** |

wall median 기준도 부호·크기가 같다 (−11.42 % / +3.67 % / **−8.17 %**).

### 2.3 정확도

| 구성 | ORT 대비 | 출력 SHA-256 | 3 회 반복 |
|---|---|---|---|
| **O1** | **corr 1.0000000, 24,576/24,576 비트 동일** | `af57856e1da9c4ea…` | 단일 해시 |
| **O2** | **corr 1.0000000, 비트 동일** | `af57856e1da9c4ea…` | 단일 해시 |
| **C** | corr 0.9987321, maxdiff 0.0474694 | `53463eda89b6f889…` | 단일 해시 |

- **O1 과 O2 의 출력은 바이트 동일**하고, A/B/C 비교의 **A(softmax CPU) 와도 같은 해시**다.
  즉 재양자화 융합·head-split·mask 상수화는 모두 **수치 중립**이었다.
- **C 만 corr 0.9987321** 인데 그 원인은 **bf16 softmax 커널** 하나다(d1 제거는 §A/B/C
  문서에서 바이트 동일로 확인됨).
- 두 골든 `attn0_ref.npy`(O1/O2 용)와 `attn0_hsm_ort.npy`(C 용)는 **바이트 동일**함을
  확인했다 — 모델 변형 간 비교가 같은 기준 위에 있다.

### 2.4 캠페인 간 일관성

C 는 두 캠페인에 모두 들어 있다: 어제 13.7584 ms, 오늘 13.7362 ms — **0.16 % 차이**.
두 캠페인의 수치를 나란히 읽어도 된다는 뜻이다.

## 3. 해석 — 이득이 한 방향으로 쌓인 게 아니다

```
태초 O1   14.979 ms  (NPU 6 / CPU 11)
   │  −11.54 %   ← 재양자화를 matmul 안으로 흡수 (가장 큰 이득, 수치 중립)
   ▼
중간 O2   13.251 ms  (NPU 6 / CPU 6)   ← wall time 은 여기가 최저
   │  +3.66 %    ← head-split(transpose 소멸) + softmax NPU + d1 제거
   ▼
최종 C    13.736 ms  (NPU 7 / CPU 2)   ← 호스트 CPU 는 여기가 최저 (1.17 ms)
   ═══════════
   순변화 −8.30 % wall / −74.87 % 호스트 CPU
```

1. **wall time 이득은 거의 전부 "재양자화 융합"에서 나왔다** (−11.5 %).
   `i32` matmul 출력을 CPU 로 내려 재양자화하던 4 개의 왕복이 사라진 것이라, 이건
   **디스패치 수 감소 + cross-device staging 제거**가 동시에 온 경우다.
2. **O2 → C 는 wall 로는 +3.66 % 손해다.** 안에 세 변화가 섞여 있는데, A/B/C 문서의
   분리 측정에 따르면 그중 **softmax NPU 이식이 +2.99 %** 이고 **d1 제거가 −0.99 %** 다.
   나머지(head-split 로 transpose 3 개 소멸 + mask 상수화)가 순증 쪽이라는 뜻이며,
   transpose 를 없애는 대신 **Q/K/V 가 2D `1x32x768x832` 에서 배치 `12x32x64x832` 로
   바뀐 비용**이 붙은 것으로 보인다. ⚠️ 이 마지막 분해는 **측정으로 분리하지 않았다** —
   O2 와 C 사이에 별도 중간 구성을 만들지 않았으므로 추정이다.
3. **호스트 CPU time 은 단조롭게 줄었다**: 4.670 → 3.406 → **1.174 ms (−74.9 %)**.
   CPU 예산 관점에서는 모든 단계가 이득이었다.
4. **변동성도 크게 줄었다**: CV 1.48 % → 0.54 %. CPU 디스패치가 11 개에서 2 개로 줄면서
   호스트 지터가 그만큼 빠졌다 (p99 − p50 이 0.49 ms → 0.20 ms).

## 4. 발표용 문구

> attention 블럭을 "matmul 만 NPU" 에서 "NPU 7 / CPU 2" 까지 올렸다.
> 디스패치 15 → 9, **wall time −8.3 %**, **호스트 CPU 작업 −75 %**, throughput **+9 %**,
> 실행 간 변동(CV) 1.5 % → 0.5 %.
> 다만 wall time 이득은 대부분 **재양자화를 matmul 안으로 접은 것**(−11.5 %)에서 왔고,
> 그 뒤 단계는 CPU 경계를 없애는 대가로 wall 을 +3.7 % 되돌려 놓았다 —
> 그 적자의 대부분이 **softmax 를 독립 디스패치로 돌리는 경계 비용**이다.
> 다음 과제는 QK^T → softmax → PV 를 한 스케줄로 묶는 것.

## 5. 재현 정보

| 항목 | 값 |
|---|---|
| 빌드 | HEAD `173748a` (`bert`), `third_party/iree` `d9e3447`, `CMAKE_BUILD_TYPE=Release` |
| O1/O2 는 패치 없는 HEAD 로 충분 | head-split 이 없어 broadcast 흡수 패치가 발동하지 않음 (O2 MD5 가 패치 유무와 동일함을 2026-09-17 에 확인) |
| C 는 HEAD + broadcast 흡수 패치 4 파일 | d1 제거가 곧 그 패치라 불가피 |
| 입력 (O1/O2) | `attn0_x.npy` + `attn0_mask.npy`, 함수 `attn0` |
| 입력 (C) | `attn0_x.npy`, 함수 `attn0hsm` |
| 골든 | `attn0_ref.npy` = `attn0_hsm_ort.npy` (바이트 동일) |
| 런타임 | `--device=amdxdna --device=local-task` (전부 동일) |
| 원자료 | `_local/perf3/benchO_raw.csv` (720 행), `benchO_agg.csv` (36 행) |
| 스크립트 | `_local/perf3/build_orig.sh`, `bench_orig.sh`, `analyzeO.py`, `acc_orig.sh` |

기존 산출물은 덮어쓰지 않았다. 새 파일은 전부 `_local/perf3/` 에 있다.
⚠️ 9/16 에 저장된 `attn0_fuse.vmfb`(MD5 `216248c3…`)는 지금 HEAD 재컴파일본
(`b35a583d…` = O2)과 바이트가 다르다 — 9/16 이후 커밋들의 표류이고 **패치 탓이 아니다**
(2026-09-17 에 stash 대조로 확인).

## 6. NPU 상태

캠페인 전후 `status.sh` = `NPU: free`, `check-containers.sh` 전 컨테이너 `ok`.
세 구성 각 3 회 실행이 전부 단일 해시이고 O1/O2 는 ORT 와 비트 동일이므로 측정 구간
동안 장치 정상이 데이터로 확인된다.
