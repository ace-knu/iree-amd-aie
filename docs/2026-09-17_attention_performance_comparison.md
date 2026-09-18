# 2026-09-17 — attention-only 최종 경로 성능 비교 (A / B / C)

> 발표용 정리. **코드 변경 없음** — 이미 검증된 경로를 같은 조건에서 다시 재서 비교만 했다.
> d0(input quantize+pad)·d9(final dequant+bias)를 NPU 로 올리는 실험은 **의도적으로 제외**했다
> (`docs/2026-09-17_d1_benchmark_and_d0_feasibility.md` — soft-float 비용과 `ert state 8`).

## 0. 한 장 요약

| | A | B | C |
|---|---|---|---|
| 구성 | softmax **CPU** + d1 broadcast | softmax **NPU** + d1 broadcast | softmax **NPU**, d1 **제거** |
| 디스패치 | 10 (**NPU 6 / CPU 4**) | 10 (**NPU 7 / CPU 3**) | **9 (NPU 7 / CPU 2)** |
| wall mean | **13.492 ms** | 13.896 ms | 13.758 ms |
| 호스트 CPU time | 2.601 ms | 1.439 ms | **1.174 ms** |
| ORT 대비 | **corr 1.0000000 (비트 동일)** | corr 0.9987321 | corr 0.9987321 |
| A 대비 wall | — | **+2.99 %** | **+1.98 %** |
| A 대비 CPU time | — | −44.70 % | **−54.88 %** |

세 줄로:

1. **d1 broadcast 제거(B→C)는 순이득이다** — wall **−0.99 %**, 호스트 CPU **−18.4 %**, 12/12 라운드.
2. **softmax NPU 이식(A→B)은 기능적으로는 CPU 경계를 없앴지만 현재 성능은 손해다** —
   wall **+2.99 %** (12/12 라운드 전부 A 가 빠름). 호스트 CPU 는 −44.7 % 로 크게 줄지만
   그 이득이 wall time 으로 넘어오지 않는다.
3. 그래서 **A → C 순변화는 여전히 wall +1.98 %** 다. 즉 attention 을 NPU 로 더 밀어 넣는
   작업의 다음 과제는 **디스패치를 더 옮기는 것이 아니라 QK^T → softmax → PV 사이의
   launch/DMA 경계를 줄이는 것**이다.

⚠️ **B 는 A 보다 정확도도 약간 낮다.** A 의 CPU softmax 는 ORT 골든과 **비트 동일**이고,
B/C 의 NPU softmax 는 bf16 커널이라 corr 0.9987321 이다. 성능·정확도 둘 다 A 가 낫고,
B/C 의 가치는 **CPU 경계 제거(기능)** 와 **호스트 CPU 절반 이하**라는 점에 있다.

## 1. 구성 정의와 재현 정보

세 구성은 **같은 모델·같은 입력**이고 다음만 다르다.

| | 컴파일러 | softmax ukernel 플래그 | d1 broadcast |
|---|---|---|---|
| **A** | HEAD `173748a` (패치 없음) | 없음 | CPU 디스패치로 존재 |
| **B** | HEAD `173748a` (패치 없음) | `--iree-amdaie-enable-ukernels=softmax` | CPU 디스패치로 존재 |
| **C** | HEAD `173748a` **+ broadcast 흡수 패치 4 파일** | `--iree-amdaie-enable-ukernels=softmax` | **없음** (Q/K/V 로 흡수) |

⚠️ **A·B 와 C 가 서로 다른 빌드인 것은 피할 수 없고, 그게 정확히 측정 대상이다.**
d1 제거는 컴파일러 패치 그 자체이므로, B 는 패치 없는 빌드에서, C 는 패치 있는 빌드에서
나올 수밖에 없다. **A vs B 는 같은 빌드**라 softmax 효과가 깨끗하게 분리된다.

### 1.1 공통 컴파일 플래그

```
--iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local
--iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu
--iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=/workspace/llvm-aie
--iree-flow-enable-executable-deduplication=false
--iree-amd-aie-enable-chess-for-ukernel=false
```

B·C 는 여기에 `--iree-amdaie-enable-ukernels=softmax` 하나만 더한다.
(`--iree-dispatch-creation-no-fuse-into-contraction-conv-roots` 는 **빼야** 한다 —
attention 은 융합을 켠 상태가 기준선이다.)

### 1.2 산출물·입력

| 항목 | 값 |
|---|---|
| 소스 | `_local/int8_debug/out/attn0_hsm_int8.mlir` (MD5 `8a7878d1…`) |
| 입력 | `_local/int8_debug/out/attn0_x.npy` `[1,32,768] f32` (MD5 `851b4ee4…`) |
| 골든 | `_local/int8_debug/out/attn0_hsm_ort.npy` (MD5 `958f09b6…`) |
| A vmfb | `_local/perf3/A_softmaxCPU.vmfb` MD5 **`cd8b82ef46471dd64e112b1ddf09b2fd`** |
| B vmfb | `_local/perf3/B_softmaxNPU.vmfb` MD5 **`c47813aeed5c79359cc7c857635e1d08`** |
| C vmfb | `_local/perf3/C_noBcast.vmfb` MD5 **`1b8490a5b86d86bc89d48ff1f0cefe18`** |
| 빌드 | `CMAKE_BUILD_TYPE=Release`, 셋 다 동일 트리·동일 툴체인 |
| 런타임 옵션 | `--device=amdxdna --device=local-task` (셋 다 동일) |

- C 의 MD5 는 **하드웨어 검증본 `attn0_hsm_sm_nb.vmfb` 와 동일**하다.
- ⚠️ 저장돼 있던 `attn0_hsm_sm.vmfb`(MD5 `7e6ebd0d…`)는 지금 HEAD 에서 재컴파일하면
  `c47813ae…` 가 나온다. 바이트는 다르지만 **NPU 출력 SHA-256 은 동일**(`53463eda…`)이라
  기능 차이는 없다. 이 문서의 B 는 **새로 빌드한 쪽**이다(A 와 같은 빌드로 맞추기 위해).
- 기존 산출물은 **하나도 덮어쓰지 않았다.** 새 파일은 전부 `_local/perf3/` 에 있다.

## 2. 디스패치 토폴로지

flow 레벨(`--compile-to=flow`)에서 `stream.affinity` 로 센 것이다.

### A — softmax CPU + d1 broadcast : 10 개 (NPU 6 / CPU 4)

| # | 장치 | 디스패치 | 역할 |
|---|---|---|---|
| d0 | **CPU** | `elementwise_32x768_f32xi8` | 입력 quantize + K 패딩(768→832) |
| d1 | **CPU** | `elementwise_broadcast_12x26624_i8` | 활성화를 12 head 로 복제 (319,488 B) |
| d2 | NPU | `batch_matmul_12x32x64x832` | **Q** projection (+ 재양자화 융합) |
| d3 | NPU | `batch_matmul_12x32x64x832` | **K** projection |
| d4 | NPU | `batch_matmul_12x32x64x832` | **V** projection |
| d5 | NPU | `batch_matmul_12x32x32x64` | **QKᵀ** (transpose_b) + 재양자화 |
| d6 | **CPU** | `softmax_12x32x32xf32_generic` | dequant → softmax → quant |
| d7 | NPU | `batch_matmul_12x32x64x32` | **PV** (출력 transpose 흡수, `[32,12,64]`) |
| d8 | NPU | `batch_matmul_1x32x768x768` | **output projection** + 재양자화 |
| d9 | **CPU** | `elementwise_32x768_f32xi8xf32` | 최종 dequant + bias |

### B — softmax NPU + d1 broadcast : 10 개 (NPU 7 / CPU 3)

A 와 동일하되 **d6 softmax 가 NPU** 다 (`softmax_i8_96x32` ukernel).
CPU 에 남는 것: d0, d1, d9.

### C — softmax NPU + d1 제거 : **9 개 (NPU 7 / CPU 2)**

| # | 장치 | 디스패치 | 역할 |
|---|---|---|---|
| d0 | **CPU** | `elementwise_32x768_f32xi8` | 입력 quantize + 패딩 |
| d1 | NPU | `batch_matmul_12x32x64x832` | Q |
| d2 | NPU | `batch_matmul_12x32x64x832` | K |
| d3 | NPU | `batch_matmul_12x32x64x832` | V |
| d4 | NPU | `batch_matmul_12x32x32x64` | QKᵀ |
| d5 | NPU | `softmax_12x32x32xf32_generic` | i8 softmax ukernel |
| d6 | NPU | `batch_matmul_12x32x64x32` | PV |
| d7 | NPU | `batch_matmul_1x32x768x768` | output projection |
| d8 | **CPU** | `elementwise_32x768_f32xi8xf32` | 최종 dequant + bias |

broadcast 는 **NPU 디스패치로 옮긴 게 아니라 없어졌다.** Q/K/V 가 각자 디스패치 안에서
같은 `32x832` L3 버퍼를 head 마다 다시 읽는다(DMA fan-out). Q/K/V 의 LHS 바인딩이
`12x32x832xi8`(319,488 B) → `32x832xi8`(26,624 B) 로 줄고, transient 버퍼와 그 CPU 쓰기가
사라졌다.

⚠️ 과제 지시문에는 A 가 "NPU 7 / CPU 3" 으로 적혀 있었으나 **실측은 NPU 6 / CPU 4** 다.
softmax 가 CPU 에 있으므로 당연하다.

## 3. 벤치마크 방법

[[feedback-npu-perf-measurement-method]] 규칙 그대로:

- `iree-benchmark-module` (벽시계 `iree-run-module` 아님 — 모듈 로드가 실행 시간과 같은 자릿수)
- 시작 전 `./scripts/lock/status.sh` = `NPU: free`, `./scripts/lock/check-containers.sh`
  전 컨테이너 `ok` 확인 → **캠페인 전체를 `with-npu-lock.sh` 하나로 감쌌다.**
- **12 라운드 × 3 구성 = 36 측정**, 라운드당 `--benchmark_repetitions=20`
  → 구성당 **240 repetition**.
- **순서 균형**: 3 구성의 6 가지 순열을 2 바퀴 돌려, **각 구성이 1/2/3 번째 위치에 정확히
  4 회씩** 나오게 했다(실측 확인함). 라운드 내 쌍체(paired) 비교.
- 셋 다 같은 Release 빌드·같은 입력·같은 `--device` 옵션.

원자료: `_local/perf3/bench3_raw.csv` (720 행), `_local/perf3/bench3_agg.csv` (36 행),
스크립트 `bench3.sh`, 분석 `analyze.py`.

## 4. 결과

### 4.1 구성별 절대값

| 구성 | wall mean | wall median | wall p95 | wall p99 | sd | CV | CPU process time | throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** softmax CPU | **13.4918 ms** | 13.4887 | 13.6229 | 13.6653 | 0.0792 | 0.59 % | 2.6014 ms | **74.12 inv/s** |
| **B** softmax NPU | 13.8958 ms | 13.8858 | 14.0504 | 14.2176 | 0.0904 | 0.65 % | 1.4386 ms | 71.97 inv/s |
| **C** +d1 제거 | 13.7584 ms | 13.7501 | 13.8905 | 14.0540 | 0.0804 | 0.58 % | **1.1737 ms** | 72.69 inv/s |

p95/p99 는 구성당 240 repetition 전체에서 계산했다. CV 는 0.6 % 대로 세 구성이 같다.

### 4.2 쌍체 대조 (같은 라운드, 순서 균형)

| 대조 | wall mean | 95 % CI | 라운드 |
|---|---:|---|---|
| **A → B** (softmax CPU→NPU) | **+0.4040 ms (+2.99 %)** | `[+2.87 %, +3.12 %]` | **0/12** (B 가 빠른 적 없음) |
| **B → C** (d1 broadcast 제거) | **−0.1374 ms (−0.99 %)** | `[−1.10 %, −0.87 %]` | **12/12** (C 가 항상 빠름) |
| **A → C** (순변화) | **+0.2666 ms (+1.98 %)** | `[+1.87 %, +2.08 %]` | 0/12 |

| 대조 | 호스트 CPU process time | 95 % CI |
|---|---:|---|
| A → B | **−1.1628 ms (−44.70 %)** | `[−45.24 %, −44.16 %]` |
| B → C | **−0.2649 ms (−18.41 %)** | `[−19.34 %, −17.48 %]` |
| A → C | **−1.4277 ms (−54.88 %)** | `[−55.29 %, −54.47 %]` |

| 대조 | throughput |
|---|---:|
| A → B | −2.15 inv/s (−2.91 %), CI `[−3.03 %, −2.79 %]` |
| B → C | **+0.72 inv/s (+1.00 %)**, CI `[+0.88 %, +1.11 %]` |
| A → C | −1.44 inv/s (−1.94 %), CI `[−2.04 %, −1.84 %]` |

wall median 기준 값도 mean 과 부호·크기가 같다 (A→B +2.94 %, B→C −0.98 %, A→C +1.94 %).

### 4.3 정확도·결정성

| 구성 | ORT 대비 corr | maxdiff | mean abs | 출력 SHA-256 | 3 회 반복 |
|---|---:|---:|---:|---|---|
| **A** | **1.0000000** | **0** | 0 | `af57856e1da9c4ea…` | 단일 해시 |
| **B** | 0.9987321 | 0.0474694 | 0.0060457 | `53463eda89b6f889…` | 단일 해시 |
| **C** | 0.9987321 | 0.0474694 | 0.0060457 | `53463eda89b6f889…` | 단일 해시 |

- **A 는 ORT 골든과 비트 동일**하다 — CPU softmax 경로가 기준으로 확실하다는 뜻이고,
  `docs/2026-09-16_...` §18.1 에 기록된 해시 `af57856e…` 와 일치한다.
- **B 와 C 의 출력은 서로 바이트 동일**하다 ⇒ **d1 제거는 수치적으로 완전 중립**이다.
  (성능 변화가 수치 변화와 섞이지 않았음을 보장한다.)
- B/C 의 corr 0.9987321 은 **bf16 softmax 커널의 한계**이지 d1 과 무관하다.

## 5. 요소별 해석

### 5.1 d1 broadcast 제거 (B → C) — CPU 물질화를 없애 wall 도 소폭 개선

- 호스트가 매 호출마다 쓰던 **319,488 B 의 12 벌 복제와 그 transient 버퍼가 사라진다.**
- 호스트 CPU time **−18.4 %**, wall **−0.99 %**, 12/12 라운드 일관, 출력 **바이트 동일**.
- **0.87 % 미만·1.10 % 초과의 이득은 95 % CI 로 배제**된다.
- ⚠️ L3 → NPU 로 실제로 흐르는 바이트 수는 **줄지 않는다**(12 head 가 같은 26 KB 를 각각
  읽는다). 이득의 출처는 **호스트 쓰기 + 디스패치 launch 1 개 + transient 할당**이다.
  그래서 크기가 −1 % 수준인 것이 자연스럽다.

### 5.2 softmax NPU 이식 (A → B) — 기능은 성공, standalone 디스패치 성능은 손해

- CPU 경계 하나가 사라지고 호스트 CPU time 이 **−44.7 %** (2.60 → 1.44 ms) 다.
- 그런데 **wall 은 +2.99 %** 이고 **12 라운드 전부 A 가 빠르다.** 즉 호스트에서 던
  1.16 ms 가 wall 로 회수되지 않고, 오히려 0.40 ms 를 더 쓴다.
- 원인은 `[12,32,32]` 짜리 softmax 를 **독립 NPU 디스패치**로 돌리기 때문이다 —
  launch + L3↔L2↔L1 왕복 + 동기화가 계산 이득보다 크다.
  (2026-09-17 핸드오프 문서가 debug 빌드에서 +7.2 % 로 기록한 것과 **부호가 같다**.
  이번은 Release 빌드·12 라운드 순서균형·쌍체 CI 라 **+2.99 % `[+2.87, +3.12]`** 가
  더 신뢰할 수치다. 절대값이 23.6/25.3 ms → 13.49/13.90 ms 로 바뀐 것도 빌드 타입 차이다.)
- 덤으로 **정확도도 비트 동일에서 corr 0.9987 로 내려간다**(bf16 커널).

### 5.3 디스패치 하나의 값은 어느 장치냐로 자릿수가 갈린다

| | 변화 | wall | 호스트 CPU |
|---|---|---:|---:|
| **CPU** 디스패치 1 개 제거 (B→C) | 10 → 9 | −0.137 ms (−0.99 %) | −0.265 ms |
| **NPU** 디스패치 1 개 추가 (A→B) | CPU softmax → NPU softmax | +0.404 ms (+2.99 %) | −1.163 ms |

⇒ "디스패치 수를 줄이면 빠르다" 는 성립하지 않는다. **CPU→NPU 이동은 호스트 CPU 예산을
사서 wall 을 파는 거래**이고, 지금 규모에서는 그 거래가 손해다.

### 5.4 디스패치별 시간 — 이번엔 제시하지 못한다

- 이 빌드는 `IREE_ENABLE_RUNTIME_TRACING=OFF`, `IREE_BUILD_TRACY=OFF` 다.
  Tracy 트레이스를 켜려면 런타임을 다시 빌드해야 하는데 이번 작업은 **코드 변경 없음**
  범위라 하지 않았다.
- 개별 Q/K/V·QKᵀ·PV·projection 을 떼어 단독 실행하는 방식은 **쓰지 않았다** —
  dependency/DMA 스케줄이 원래대로 재현되지 않아 수치가 오해를 부른다.
- 그래서 **end-to-end 비교만 제시**하고, 요소 분리는 §5.1/5.2 처럼 **구성 차이(A/B/C)로만**
  했다. 이건 디스패치를 건드리지 않고도 softmax 효과와 d1 효과를 깨끗이 가른다.

## 6. 결론과 다음 과제

1. **d1 broadcast 제거는 유지한다.** CPU 물질화(319 KB)와 호스트 CPU 18 %, wall 1 % 를
   줄이고 **출력은 바이트 동일**이다. 비용 없는 개선이다.
2. **softmax NPU 이식은 "기능적 성과"로 보고해야 한다.** attention 블럭에서 CPU 경계를
   없앤 것은 맞지만, **standalone 디스패치인 한 성능은 CPU 보다 불리하다(+2.99 %)** 이고
   정확도도 비트 동일에서 내려간다. 성능 근거로 쓰면 안 된다.
3. **다음 성능 과제는 디스패치를 더 옮기는 것이 아니라 경계를 줄이는 것이다.**
   구체적으로 **QK^T → softmax → PV 를 하나의 스케줄로 묶어** 그 사이의 launch/DMA
   왕복을 없애는 것. 지금 이 셋은 `[12,32,32]`/`[12,32,64]` 짜리 작은 텐서를 매번
   L3 까지 내렸다 올린다. §5.2 가 재는 비용이 정확히 그 왕복이다.
4. d0/d9 를 NPU 로 올리는 것은 **선행 조건(정수/벡터 quantize 경로)이 생기기 전까지 보류**다
   — 근거는 `docs/2026-09-17_d1_benchmark_and_d0_feasibility.md` (코어 ELF 의 63 % 가
   소프트 플로트, 외삽 3.57 ms, 아낄 수 있는 호스트 CPU 예산 전체가 1.17 ms).

### 발표용 한 문장

> attention 블럭을 NPU 7 / CPU 2 까지 올렸고 호스트 CPU 작업을 55 % 줄였다.
> 다만 end-to-end wall time 은 CPU-softmax 기준선 대비 아직 +2.0 % 인데,
> 그 적자는 전부 **softmax 를 독립 디스패치로 돌리는 경계 비용**에서 나온다 —
> 다음 작업은 QK^T→softmax→PV 를 한 스케줄로 묶는 것이다.

## 7. NPU 상태

캠페인 전후로 `./scripts/lock/status.sh` = `NPU: free`, `check-containers.sh` 전 컨테이너
`ok`. 세 구성 각각 3 회 실행이 전부 단일 해시이고 A 는 ORT 와 비트 동일이므로, 측정 구간
동안 장치가 정상이었음이 데이터 자체로 확인된다.
