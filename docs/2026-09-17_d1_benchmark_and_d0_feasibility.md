# 2026-09-17 — d1 제거의 성능 측정, 그리고 d0 를 NPU 로 올릴 수 있는가

> 앞선 `docs/2026-09-17_qkv_broadcast_absorbed_into_npu_matmuls.md` 에서 d1(12-head
> broadcast) 을 없앴다. 이 문서는 ① 그 변경의 실제 성능과 ② d0(input quantize+pad) 를
> NPU 로 올리는 것이 성립하는지를 측정한 결과다. d9 는 손대지 않았다.

## 0. 결론

1. **d1 제거는 실이득이다.** wall time **−0.98 %** (95 % CI `[−1.16 %, −0.80 %]`),
   호스트 CPU time **−18.2 %**, 12/12 라운드 전부 빠름.
2. **d0 를 NPU 로 올리는 것은 지금 성립하지 않고, 성립시켜도 손해다.**
   - Q/K/V 에 clone/fuse 하는 구성은 `AMDAIEBufferizeToAllocation` 에서 막힌다.
   - 독립 NPU 디스패치 구성은 컴파일·실행·**비트 정확**까지 되지만 **실제 크기에서 행**한다.
   - 코어 ELF 의 **63 % 가 소프트 플로트 라이브러리**다.
   - 실측 스케일링으로 외삽하면 quantize 하나에 **≈ 3.6 ms** — 현재 모델 전체 wall time 이
     13.75 ms 이고 **CPU 디스패치 2 개를 합친 호스트 CPU time 이 1.18 ms** 다.
     즉 없애려는 비용의 **3 배**를 새로 낸다.

## 1. d1 제거본 vs 기준선 — 성능

[[feedback-npu-perf-measurement-method]] 3종 세트 그대로: `iree-benchmark-module`,
캠페인 전체를 NPU lock 하나로 감싸고, **12 라운드 AB/BA 교차 + 쌍체 통계**.
라운드당 `--benchmark_repetitions=20`.

- A = `attn0_hsm_sm.vmfb` (기준선, NPU 7 / CPU 3)
- B = `attn0_hsm_sm_nb.vmfb` (d1 제거, NPU 7 / CPU 2)

| | A | B | Δ (B−A) | 95 % CI |
|---|---:|---:|---:|---|
| wall mean | 13.885 ms | **13.749 ms** | **−0.136 ms (−0.98 %)** | `[−1.16 %, −0.80 %]` |
| wall median | 13.878 ms | 13.729 ms | −0.150 ms (−1.08 %) | `[−1.23 %, −0.92 %]` |
| **CPU process time** | 1.441 ms | **1.179 ms** | **−0.262 ms (−18.2 %)** | `[−19.1 %, −17.2 %]` |
| throughput | 72.02 inv/s | 72.74 inv/s | +0.71 (+0.99 %) | `[+0.82 %, +1.17 %]` |

- **12/12 라운드 전부 B 가 빠르다.** within-run CV 는 A 0.59 % / B 0.64 %.
- **1.16 % 이상의 이득도, 0.80 % 미만의 이득도 95 % CI 로 배제된다.**

⚠️ 호스트 CPU 작업은 18 % 줄었는데 end-to-end 는 1 % 다. 지워진 d1 이 NPU 디스패치가
아니라 **값싼 CPU 복사 디스패치**여서다. softmax 를 NPU 로 올렸을 때 +7.2 % 였던 것과
크기가 비대칭인 것도 같은 이유 — 그쪽은 **NPU** 디스패치가 하나 늘어난 것이었다.
즉 "디스패치 하나 = 몇 %" 라는 환산은 성립하지 않는다. **어느 장치의 디스패치인지가
자릿수를 결정한다.**

원자료: `_local/scratch_bcast/bench_ab.csv`, 스크립트 `bench_ab.sh`.

## 2. d0 를 Q/K/V 에 clone/fuse — 막힌 지점

d0 는 이렇게 생겼다(§flow):

```mlir
%2 = flow.tensor.splat %c1_i8 : tensor<32x832xi8>          // pad 는 splat 으로 밖에 있다
%3 = flow.dispatch @..._dispatch_0_elementwise_32x768_f32xi8(%1, %2)
       {stream.affinity = #hal.device.affinity<@cpu>}
       : (tensor<32x768xf32>, tensor<32x832xi8>) -> %2      // readwrite, [0..768) 만 덮어씀
```

디스패치 본체:

```
divf(x, 0.079871498) -> roundeven -> addf(zp=0) -> maximumf(-128) -> minimumf(127) -> fptosi i8
```

### 2.1 타일링은 된다

손으로 병합한 `hal.executable.source`(§5)로 백엔드만 단독 시험했다.
`AMDAIETileAndFuse` 는 **quantize 를 matmul 의 `scf.forall` 안으로 제대로 융합한다** —
루프 안에 `linalg.generic`(divf…fptosi)이 들어간 IR 을 확인했다.

### 2.2 막히는 곳은 `AMDAIEBufferizeToAllocation` 이다

```
error: 'linalg.generic' op failed bufferizing to allocations
```

pack-peel 파이프라인의 세 번째 호출("Promote the elementwise input to shared memory",
`Passes.cpp:247-255`)이 **elementwise op 을 matmul 의 *꼬리*로 가정**한다. 꼬리라면 그
입력은 matmul 결과라 버퍼화 가능한 정의 op 을 갖는다. 그런데 d0 는 **머리**여서 그 입력이
바인딩의 raw `iree_tensor_ext.dispatch.tensor.load` 이고, `bufferizeToAllocation` 이
그걸 처리하지 못한다. (같은 파일 `AMDAIEBufferizeToAllocation.cpp:299-307` 의 주석이
이미 이 실패 모양을 다른 맥락에서 기술해 두었다.)

### 2.3 pad 는 두 번째 벽이다

`fill + tensor.insert_slice` 든 `tensor.pad` 든 `AMDAIETileAndFuse` 의 fusion control
(`AMDAIETileAndFuse.cpp:565-581`)이 **`tensor::PadOp` 을 "융합하지 않을 op" 목록에**
넣어두었고, `insert_slice` 는 linalg op 이 아니라 기본 분기에서 탈락한다. 그래서
pad 는 루프 **밖**에 남는다. 2.2 를 고쳐도 이건 따로 풀어야 한다.

### 2.4 고쳐도 방향이 틀렸다 — 정적 비용

레벨-0 타일은 `scf.forall (0..12, 0..32, 0..64) step (1,32,64)` = **디스패치당 12 회**,
LHS 타일은 매번 **전체 `32x832`** 다(M·K 는 이 레벨에서 안 잘림).

| | 현재 (CPU d0) | Q/K/V 에 융합 |
|---|---:|---:|
| quantize 원소 연산 | 24,576 회 **1 번** | 24,576 × 12 × 3 = **884,736 회 (36×)** |
| L3 → NPU 활성화 읽기 | i8 26,624 B × 36 = 958,464 B | **f32 98,304 B × 36 = 3,538,944 B (3.69×)** |

즉 연산량 36 배, DMA 3.7 배를 내고 디스패치 하나를 아낀다.
그 디스패치 하나의 값은 §1 에서 실측됐다 — **0.136 ms**.

## 3. d0 를 독립 NPU 디스패치로 — 여기까지는 된다

clone/fuse 가 막혀서, **비중복 형태**(quantize 를 딱 한 번만 하는 형태)로 바꿔 측정했다.

### 3.1 파이프라인 선택이 먼저 막는다

- 기본 pack-peel 에서 elementwise 를 루트로 하면 `setRootConfig` 이
  `Unhandled pass pipeline in setRootConfig.` 로 실패한다(`KernelDispatch.cpp:985`).
- 경로는 있다: `setRootConfigForElementwiseCopyPipeline` (`KernelDispatch.cpp:916`).
  단 **`TilePassPipeline::GeneralCopyPipeline` 일 때만** 불린다.
- 그런데 general-copy 는 `AMDAIELowerExecutableTarget.cpp:87` 에서 **루트가
  `linalg::SoftmaxOp` 일 때만** 선택된다. 실모델에서 d0 를 NPU 로 보내려면 softmax 와
  똑같이 elementwise 루트도 per-dispatch 로 라우팅해야 한다(2 군데).
- 추가로 그 함수는 `assert(inputShape.size() == 1)` 이라 **1-D 만 받는다.**
  d0 는 `32x768` 이므로 collapse 가 필요하다.

실험에서는 `--iree-amdaie-tile-pipeline=general-copy` 를 전역으로 주고 입력을 1-D
(`24576xf32 -> 24576xi8`)로 써서 이 두 가지를 우회했다.

### 3.2 컴파일된다 — 그리고 코어 ELF 는 소프트 플로트다

`--iree-hal-dump-executable-intermediates-to` 로 받은 `core_0_2.elf` (text **8,432 B**):

| 심볼 | 크기 |
|---|---:|
| `core_0_2` (본체) | 4,192 B |
| `__divsf3` | 1,440 B |
| `__addsf3` | 1,424 B |
| `__nesf2`/`__ltsf2`/`__lesf2`/`__gtsf2`/`__gesf2`/`__eqsf2`/`__cmpsf2` (272 × 7) | 1,904 B |
| `__fixsfsi` | 256 B |
| `__floatsisf` | 208 B |
| `__unordsf2` | 48 B |
| `__subsf3` | 32 B |
| `__muldi3` | 112 B |
| `__fe_raise_inexact`, `__fe_getround` | 32 B |

**소프트 플로트 합계 5,344 B = text 의 63 %.** [[project-aie2p-no-scalar-float]] 가
말하는 그대로다. program memory(16 KB)는 넘지 않으므로 **크기는 blocker 가 아니다** —
비용이 blocker 다.

### 3.3 정확도는 비트 정확하다

ONNX `QuantizeLinear` 의 의미(round-half-to-even + clamp)를 numpy 로 재현한 골든 대비:

| 원소 수 | 결과 |
|---|---|
| 128 | **128 / 128 exact**, maxdiff 0 |
| 256 | **256 / 256 exact**, maxdiff 0 |
| 512 | **512 / 512 exact**, maxdiff 0 |
| 640 | **640 / 640 exact**, maxdiff 0 |

즉 소프트 플로트 경로가 **수치적으로는 정확하다.**

### 3.4 결정성

640 원소 판으로 **별도 프로세스 5 회** 실행 → **단일 SHA-256** (`ceeed14e…`),
골든과 640/640 일치.

### 3.5 ⚠️ 그런데 실제 크기에서 행한다

| 원소 수 | 디바이스 루프 반복 | 결과 |
|---:|---:|---|
| 128 | 1 | ✓ |
| 256 | 2 | ✓ |
| 512 | 4 | ✓ |
| 640 | 5 | ✓ |
| 1,024 | 8 | ✗ `ert state 8` (타임아웃) |
| 4,096 | 32 | ✗ `ert state 8` |
| **24,576 (실제 d0 크기)** | **192** | ✗ `ert state 8`, 5/5 회 전부 |

⚠️ **내 변경 탓이 아니다**: 변경을 stash 하고 재빌드해 같은 소스를 다시 컴파일하니
`d0_only1d.vmfb` 가 **MD5 동일**(`8a57a1e2…`)이다. general-copy elementwise 경로가
원래 갖고 있던 성질이다.

**관찰된 차이**(인과는 미확정): 제어 코드의 shim BD 접근 패턴이 경계에서 바뀐다.

```
N=640  (동작): sizes = [0, 0, 640],  buffer_length = 640     ← 1-D 연속으로 접힘
N=1024 (행)  : sizes = [0, 8, 128],  buffer_length = 1024    ← 바깥 차원 8 이 남음
```

`foldLinearDims` 가 BD 의 `WrapMax` 안에 들어갈 때까지만 전체를 하나의 연속 전송으로
접고, 넘어가면 바깥 차원이 남는다. 그 **바깥 차원이 남은 4-코어 fan-out objectFifo** 가
데드락하는 것으로 보인다. 단 이건 상관이지 증명된 인과가 아니다 —
행의 원인을 확정하려면 별도 조사가 필요하다.

### 3.6 성능 — 외삽하면 결정적으로 손해다

동작하는 크기들에서 `iree-benchmark-module` (20 repetitions, lock 사용):

| 원소 수 | wall mean | cpu mean |
|---:|---:|---:|
| 128 | 0.6598 ms | 0.0833 ms |
| 256 | 0.6779 ms | 0.0844 ms |
| 512 | 0.7045 ms | 0.0864 ms |
| 640 | 0.7226 ms | 0.0864 ms |

선형 적합: **wall = 0.6455 ms + 0.11891 µs/원소** (절편 = 디스패치 launch).

→ 실제 d0 크기 24,576 원소로 외삽: **3.57 ms** (계산분만 2.92 ms).

코어당으로 환산하면 24,576/4 = 6,144 원소에 2.92 ms → **원소당 ≈ 475 사이클**(1 GHz 가정).
§3.2 의 소프트 플로트와 자릿수가 맞는다.

**비교 기준** (§1 실측):

| | |
|---|---:|
| 모델 전체 wall time | 13.75 ms |
| **CPU 디스패치 2 개(d0+d9) 합친 호스트 CPU time** | **1.18 ms** |
| 디스패치 하나 제거의 실측 가치 | 0.136 ms |
| d0 를 NPU 로 올릴 때 추가 비용(외삽) | **≈ 2.9 ~ 3.6 ms** |

⚠️ 외삽의 한계: 적합 구간(128~640)은 BD 가 1-D 로 접히는 영역이고 24,576 은 구조가
다르다(§3.5). 그래서 3.57 ms 는 **하한**으로 읽어야 한다. 그래도 아끼려는 값의
**3 배 이상**이라 부호가 뒤집힐 여지가 없다.

## 4. 그래서 d0 는

**지금 형태로는 올리지 말 것.** 막는 것은 파이프라인 배선이 아니라 **aie2p 에 스칼라 float
산술이 없다**는 것이고, 그건 배선을 고쳐도 남는다.

성립시키려면 [[project-attention-npu-fusion]] 의 softmax 와 같은 길밖에 없다 —
**float 를 코어에서 몰아내는 것**:

1. **정수/벡터 quantize**. `AMDAIEIntegerRequantization` 은 꼬리가
   `sitofp(i32 accumulator)` 로 시작할 때만 적용된다. d0 의 뿌리는 **진짜 f32 모델 입력**
   이라 해당 없음(lit `@requant_float_rooted` 가 이 음성 케이스를 이미 고정).
   대신 **bf16 벡터 변환**(`bfloat16_to_int` → `vfloor.s32.bf16`)은 존재하므로
   softmax 처럼 **ukernel 로 내리는 길**은 있다. 단 `AMDAIEVectorization` 이 양자화
   elementwise 를 의도적으로 제외하므로(upstream #594) 자동 벡터화로는 안 된다.
2. 그 위에 §3.1 의 배선 2 군데 + 1-D assert, §3.5 의 행, §2.2/2.3 의 융합 벽.

**순서상 1 번이 먼저다.** 1 번 없이 2 번을 해도 §3.6 의 부호는 안 바뀐다.

⚠️ **d9(final dequant+bias)는 같은 구조다** — f32 뿌리, 스칼라 float, 모델 I/O 경계.
d0 의 결과가 그대로 적용되므로, 1 번(정수/벡터 quantize 경로) 없이 d9 를 시작하면
같은 벽을 다시 만난다.

## 5. 산출물

| 파일 | 내용 |
|---|---|
| `_local/scratch_bcast/bench_ab.sh`, `bench_ab.csv` | §1 의 AB/BA 교차 캠페인 |
| `_local/scratch_bcast/d0_fused_src.mlir` | d0+d1 을 Q/K/V 디스패치에 합친 손작업 소스 (§2, 막힘) |
| `_local/scratch_bcast/d0_pad_src.mlir` | 위의 `tensor.pad` 변형 (§2.3) |
| `_local/scratch_bcast/d0_nopad_src.mlir` | pad 를 뺀 변형 — §2.2 의 벽이 pad 와 무관함을 보임 |
| `_local/scratch_bcast/d0_only{128,256,512,640,1024,4096,1d}_src.mlir` | §3 의 독립 NPU quantize, 크기별 |
| `_local/scratch_bcast/d0dump/core_0_2.elf` | §3.2 의 코어 ELF |
| `_local/scratch_bcast/bench_q.sh`, `run_qn.sh`, `elfsyms.sh` | §3 의 측정 |

컴파일 플래그는 d1 문서와 같고, §3 만 `--iree-amdaie-tile-pipeline=general-copy` 를 더한다.

## 6. NPU 상태

행이 여러 번 났으므로 매 구간 끝에 known-good `attn0_hsm_sm.vmfb` 로 확인했다 —
**4 회 전부 corr 0.9987321 / maxdiff 0.0474694** 로 기준선을 정확히 재현.
세션 종료 시점 `NPU: free`.
