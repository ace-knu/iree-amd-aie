# 변경 장부 (CHANGES.md)

git HEAD 대비 **모든 변경**을 한곳에 모은 장부다. 소스(플러그인·IREE 서브모듈)를 고치는 세션은 끝날 때 이 파일을 갱신한다(CLAUDE.md 1-4의 4, 3-1-2).

- 첫 판: 2026-10-20 세션 1 챕터 01 (`docs/2026-10-20/1_변경장부_명세_커밋준비/01_변경_장부/`)
- 주제별 명세: `docs/specs/` (아래 "명세" 열의 S1~S10. S6·S10은 dev 통합 때 삭제)
- **dev-POOL 통합(2026-10-28) 후 상태는 바로 아래 절을 본다.** 그 다음 절은 3devices 통합(2026-10-25) 때 상태, 0~6절은 10/20 스냅숏이다.

## dev-POOL 통합 후 상태 (2026-10-28, 브랜치 `pool-integration`)

`origin/dev-POOL`(GEMV pipeline, NPU max pooling pipeline)을 dev `2c5763b8`에 병합했다(`7060d5f3`, 작성자 보존). 3devices 통합과 같이 겹치는 부분은 dev가 우선이다.

### 기준과 원칙

- **GEMV**: M==1 transpose_b matmul(FC)은 GEMV pipeline으로 컴파일한다. N축만 32개 core에 나누고, 나뉜 tile은 ukernel로 벡터화한다(bf16). f32도 같은 pipeline으로 돈다(scalar, f32 ukernel 없음).
- **N-split 삭제**: FC를 Flow에서 N으로 쪼개던 pass는 NPU DMA cliff 우회였으므로 GEMV pipeline이 들어오면서 완전히 지웠다.
- **pool**: NPU max pooling은 옵션이다(`--iree-amdaie-accelerate-pooling`). 기본은 CPU이고, 스크립트는 `--pool {cpu,accel}`로 고른다.
- **ukernel 출처**: upstream mlir-aie `aie_kernels/`(aie_api)를 먼저 가져온다. 없을 때만 직접 쓴다. 통합 중에 peano ukernel 넷(matmul, matvec, zero_fill, trunci)을 이 원칙에 맞췄다.
- **ukernel unroll**: upstream 소스의 unroll 지시를 그대로 따른다. core stack(기본 1024 B)이나 program memory(16 KB)를 넘을 때만 소스에서 끄고 이유를 주석에 남긴다. core 코드의 `-unroll-full-max-count=0`(`XCLBinGen.cpp`)은 소스의 unroll pragma를 막지 못하므로 ukernel에는 쓰지 않는다. 우리 커널도 pragma는 `aie_kernel_utils.h`의 `AIE_LOOP_*` macro로 쓴다.

### dev 기준으로 본 변경

#### 1. 겹친 부분 (병합 충돌 5곳)

| 파일 | dev | dev-POOL | 통합 후 | 커밋 |
|---|---|---|---|---|
| `AMDAIEAssignDeviceAffinities.cpp` | 가속기 여러 개, round-robin | amd-aie target config `pooling`이면 pool을 NPU로 | pool 허용을 배정 pass 옵션 `accelerate-pooling` 하나로 정하고, 가속기 전체에 round-robin(장치별 op 선택은 나중) | `7060d5f3`, `f44d40e0` |
| `AMDAIEPadContractionDispatches.cpp` | 장치별 core 수 기준 pad | GEMV N pad, pool OH/C pad | 둘 다. 이후 GEMV pad 규칙을 GEMM과 같은 `getPaddingMultiples`로 합침 | `7060d5f3`, `14c757a2` |
| `AMDAIEUtils` | dev helper | GEMV/pool helper | 둘 다 | `7060d5f3` |
| pad lit | dev 케이스 | GEMV 케이스 | 합집합. dev `matmul_tb_uk`는 M=1이 GEMV가 되므로 M=2로 | `7060d5f3` |
| 플래그 이름 | — | `--iree-amdaie-enable-pooling` | `--iree-amdaie-accelerate-pooling`(배정 pass 옵션) | `f44d40e0` |

#### 2. 추가된 부분

| 영역 | 내용 | 커밋 |
|---|---|---|
| GEMV pipeline (dev-POOL) | `tile_pipeline="gemv"`: copy-based, N을 최대 32 core 1-D forall로(core당 출력 2개 이상), K tile은 K의 256 이하 최대 약수(matvec이면 256 또는 64) | `7060d5f3`, `7854b375`, `33b86238` |
| pool pipeline (dev-POOL) | 2-D max pooling을 spatial x channel core grid로. `--iree-amdaie-pool-core-split`, `--iree-amdaie-pool-columns-policy`(기본 `max`), OH/C pad, halo pre-split | `7060d5f3` |
| channels-last (dev-POOL) | `--iree-amdaie-pooling-channels-last`: NCHW pooling을 transpose로 감싼 NHWC pooling으로 바꿔 NHWC conv 사이 transpose가 상쇄되게 함(기본 꺼짐) | `7060d5f3` |
| matvec ukernel | upstream `mv_bf16.cc`를 옮긴 `uKernels/npu4/peano/matvec.cc`(f32 누적, K는 instance 값, `(ptr, offset)` 인자). bf16 GEMV tile을 이 ukernel로 lower | `69a5e8b7` |
| aie_api 내장 | submodule `third_party/aie_api`(Xilinx/aie_api, MIT), 헤더를 바이너리에 embed하고 ukernel이 `<aie_api/aie.hpp>`나 `aie_kernels/` 헤더를 쓸 때만 꺼내 `-I`로 넘김. `aie_arch.h`, `aie_kernel_utils.h`, `common/zero.h`는 mlir-aie에서 복사 | `286008ec`, `d3486957`, `ab97d20a`, `75ebbebf` |
| zero_fill ukernel | upstream `common/zero.h`. native vector 배수면 compile-time 커널, 아니면 runtime 길이 커널(정렬을 스스로 맞춤)이라 fill 크기 제한이 없음 | `ab97d20a`, `f0ef9de3` |
| matmul ukernel | upstream `mm_aie2p.h`의 `matmul_vectorized_2x2_mmul`(bf16은 bfp16, i8). tile pointer만 iree L1 pack 순서(A `[k][m]`, C `[n][m]`, B는 transpose_b에 따라 `[n][k]`/`[k][n]`)로 바꿈. upstream은 i8 K < 416에서 K 루프를 다 펼치지만 stack 1472 B로 core stack을 넘어 펼치지 않음 | `a6874c90` |
| trunci ukernel | upstream에 없어 우리 커널 유지. unroll pragma만 `AIE_LOOP_UNROLL`로 | `75ebbebf` |
| 스크립트 | `--pool {cpu,accel}` | `31f1e935` |

ukernel 이전의 A/B(npu4 CI Performance, i8): 커널 시간이 결과를 좌우하는 512x512x512 callrepl_100이 30.75 → 29.18 ms(−5%). 나머지는 같거나 측정 편차 범위(같은 빌드도 512x4096x512 pack-peel이 750 / 820 us로 갈림). bf16 matmul stack은 576 → 64 B. MLP·VGG bf16/f32 출력은 이전과 바이트 단위로 같다.

#### 3. 통합 중에 생긴 문제와 처리

| 문제 | 원인 | 처리 | 커밋 |
|---|---|---|---|
| f32 GEMV hang(ert state 6), dev-POOL은 weight를 1.0으로 pad해 피함 | memtile weight buffer 하나(S2MM 1개)를 core별 MM2S 4개가 lock pair 하나로 나눠 읽음. core 속도가 다르면 lock count가 꼬임. dev의 producer별 lock pair 수정은 이 반대 방향(consumer 여럿)을 다루지 않았음 | consumer별 lock pair. pad 값을 0으로 되돌림 | `6ef3dc1f`, `0c5bc58f` |
| ukernel all + GEMV에서 컴파일러 crash | matmul ukernel matcher가 2-D GEMV op에 rank 4/6 전제의 `getTilingInfo`를 호출. `zero_fill` template이 1x32에서 `static_assert` 실패 | rank 확인 후 호출. fill은 처음에 `M*N % 16 != 0`을 걸렀다가 upstream `zero.h`로 옮긴 뒤 조건을 없앰 | `247f29d2`, `f0ef9de3` |
| K = 128, 192 GEMV 컴파일 실패(stack 초과) | 우리 `matvec.cc`(f32에 더함)의 stack이 upstream(bf16 덮어씀)의 약 2배. K tile 128/192에서 1024/1088 B | matvec K tile을 256 또는 64로만 고름 | `7854b375` |
| aie_api 헤더 추출 실패 | 하위 폴더를 만들지 않고 파일을 씀 | 폴더 생성 | `d3486957` |
| f32 GEMV N 4096 → 4160 pad(host pad 34 ms/run) | GEMV pad가 L1 한도로 per-core N을 다시 고름 | pad 규칙을 GEMM과 통일(N은 core 수 x 8의 배수, K는 matvec이면 64) | `14c757a2` |
| M == 1, weight `[K, N]` matmul 컴파일 실패(dev에서는 됨) | pad pass는 M == 1이면 GEMV로 보고 M을 pad하지 않았지만, GEMV pipeline은 `[N, K]` weight만 받음 | 두 곳이 같은 판정 `isGemvShape(m, transposeB)`을 씀. `[K, N]`은 dev처럼 M을 32로 pad해 pack-peel | `0b80251e` |
| L1 tile + core stack이 64 KB를 넘을 수 있음 | pool과 pack-peel tile 선택이 core stack(1024 B)을 빼지 않음(GEMV는 뺌) | 셋 다 stack을 뺀 예산을 씀. L1 탐색은 예산과 같은 크기도 받음(`<=`, L2와 같게). MLP·VGG tile은 그대로 | `93c03407` |
| core당 출력 1개인 GEMV가 틀린 값을 냄 | 원인 미상(남은 일). host 있는 빌드는 N을 core x 8 배수로 pad해 만나지 않음 | GEMV tiling이 core당 출력 2개 이상만 고름. 소수 N(host 없는 빌드)은 컴파일 오류 | `33b86238` |
| npu1에서 matvec용 K 규칙을 씀 | pad pass와 KernelDispatch의 matvec 판정에 AIE2P 확인이 없었음(matcher에는 있음) | 세 곳 모두 AIE2P 확인. rank 2/4/6이 아닌 zero fill은 ukernel로 내리지 않음(이전에는 assert) | `916141ca` |

#### 4. 지운 것

| 항목 | 이유 | 커밋 |
|---|---|---|
| `AMDAIESplitLargeContractionDispatches`(N-split pass), lit, concat helper | FC DMA cliff 우회. GEMV pipeline으로 대체 | `7d49efaa` |
| `--iree-amdaie-enable-gemv-pipeline` | 임시 스위치. 필요하면 최신 구조에 맞춰 새로 만든다 | `7d49efaa` |
| GEMV weight 1.0 pad, `chooseGemvNPad`, `core_stack_size` config | lock 수정과 pad 규칙 통일로 불필요 | `0c5bc58f`, `14c757a2` |

### 검증 (`31f1e935`, ukernel 이전은 위 A/B와 각 커밋)

- ctest amd-aie 230/230, npu4 CI(`run.py --skip_tests=Performance`) 32/32(lock 수정 시점), 회귀 27/27(수동·자동).
- CPU+NPU (ukernel all):

| 모델 | dev `2c5763b8` | 통합 후 | 정확도 |
|---|---|---|---|
| MLP bf16 / f32 | 0.17 / 0.21 s | 0.19~0.21 / 0.20 s | 출력이 dev와 바이트 단위로 같음(GEMV 없음, 시간 차는 측정 편차) |
| VGG bf16 | 2.19 s | 1.62 s | corr 0.99998, top-1 일치 |
| VGG f32 | 35.04 s | 31.10 s | corr 1.0, top-1 일치 |

- VGG bf16 FC dispatch: fc1 30.7 → 14.0 ms, fc2 12.0 → 7.4 ms, fc3 5.1 → 3.8 ms(scalar GEMV 대비 matvec ukernel). e2e는 hwctx 생성/해제(약 1.2 s)가 대부분이다.
- GPU 조합(f32): CPU+GPU VGG 0.40 s, CPU+NPU+GPU 13.73 s, CPU+GPU+NPU 17.84 s(dev 14.16 / 21.32 s). 모두 corr 1.0, top-1 일치. FC가 8조각에서 1개 dispatch가 되어 round-robin 배치가 바뀌었다.
- `--pool accel`: CPU+NPU VGG bf16 1.95 s, f32 31.43 s, CPU+GPU f32 0.70 s(pool CPU 0.40 s), CPU+NPU+GPU f32 17.15 s(13.73 s). 정확도는 같다.

### 알려진 제한

| 항목 | 내용 |
|---|---|
| pool이 가속기에서 느림 | NPU pool 4개 kernel 합 22.3 ms + dispatch당 hwctx 약 95 ms, CPU는 합 1.7 ms. ukernel을 넣어도 hwctx 비용이 남아 기본값은 CPU |
| pool의 장치 선택 | `--pool accel`은 GPU에도 round-robin으로 보낸다(amd-aie만 고르는 기능 없음) |
| transpose와 fuse된 pool | 뒤에 NHWC conv가 없는 pool(VGG pool5)은 transpose와 한 dispatch가 되어 CPU에 남는다 |
| M ≥ 2 큰 FC | N-split이 없어져 batch ≥ 2 classifier 같은 큰 transpose_b FC는 DMA cliff(ert state 6)에 걸릴 수 있다 |
| host 없는 빌드의 GEMV | NPU만 선언하면 Flow pad pass가 돌지 않아 GEMV N이 pad되지 않는다. N이 소수(예: 1009)면 컴파일 오류. e2e 모델은 host가 필요하므로 단일 op 테스트에서만 해당 |
| weight `[K, N]`인 M == 1 matmul | GEMV pipeline을 못 타고 M을 32로 pad해 pack-peel로 돈다(32행 중 31행은 0 계산 + host pad/crop). GEMV L2 split이 `[K, N]` weight를 K로 잘라 모든 core가 memtile 하나를 읽기 때문. MLP·VGG에는 없음 |
| NPU 없는 `--pool accel` | channels-last 전처리는 pool을 독립 dispatch로 만들어 어느 가속기로든 보내게 하므로 NPU가 없어도 필요하다. 다만 그 안의 C pad와 halo split은 npu4 geometry(row 4, column 8)로 한다. C가 4의 배수가 아니거나 window가 겹치는 pool을 GPU로 보낼 때만 불필요한 pad/gather가 생김(VGG는 해당 없음) |
| 작은 N의 GEMV pad | N은 항상 core 수 x 8 = 256의 배수로 pad된다(N = 10 → 256). 필요한 core만큼만 pad하는 규칙은 측정상 이득이 없었고(NPU 시간 3~4 ms가 고정 비용) core당 출력 1개 문제를 드러내 넣지 않음 |

### 남은 일

- matvec stack 줄이기(f32 C 누적의 spill). 줄이면 K tile 128/192를 다시 허용. 4-lane f32 load/store로 바꾸는 시도는 Peano backend가 처리하지 못함(`unable to legalize ... <2 x s64>`).
- NPU pool ukernel(upstream에 없음, aie_api로 작성). hwctx 재사용 뒤에 측정.
- 장치별 op 선택(pool을 amd-aie에만 등), hwctx 재사용, lock pair 코드 정리.
- core당 출력 1개 GEMV가 틀리는 원인(core의 weight tile이 activation tile과 같은 `memref<1x256xbf16>`이 되고 C가 [1, 1]인 경우. 의심만 함).
- weight `[K, N]` M == 1 matmul을 GEMV로: 상수 weight를 컴파일 때 `[N, K]`로 전치하거나, GEMV L2 split이 N 위치를 따라 자르게 일반화.
- 구조 정리(동작 변화가 커서 이번 통합에서는 기록만):
  - GEMV/pool pipeline 선택이 dev의 `TilePassPipeline` enum이 아니라 문자열 override(`amdaie.tile_pipeline`)로 되어 있다.
  - pool column policy가 세 경로(target config, lowering strategy 옵션, preprocessing 옵션)로 전달된다. 같은 값이어야 한다.
  - 실험용 옵션이 일반 옵션으로 노출돼 있다: `--iree-amdaie-pool-core-split`, `--iree-amdaie-pool-columns-policy`(odd/cost/full/max), `--iree-amdaie-gemv-pad-n`, `--iree-amdaie-gemv-n1`, VGG pool로 맞춘 cost model 상수(`PoolColumnCost`). 남길 것을 정한 뒤 정리.
  - `AMDAIEPadPoolingDispatches`가 `AMDAIEPadContractionDispatches.cpp` 안에 있다. 공용 helper(target 조회, pad/crop executable 생성)를 헤더로 빼야 나눌 수 있다.

## dev 통합 후 상태 (2026-10-25, 브랜치 `3devices-integration`)

0~6절은 2026-10-20 시점 3devices 작업 트리의 스냅숏이다(행 내용은 그대로 두고, 4·5·6절의 상태 칸에만 통합 결과를 덧붙였다). 통합 이후의 상태는 이 절이 기준이다.

### 기준과 원칙

| 저장소 | 브랜치 | 구성 |
|---|---|---|
| 플러그인 `ace-knu/iree-amd-aie` | `3devices-integration` | dev `45c18ae7` + `origin/3devices-f32` 병합(`c1c4d937`) + 정리 커밋 |
| IREE 서브모듈 `ace-knu/iree` | `amd-aie-flow-hook` | dev가 쓰던 `cd1fa2a1e` + 3devices 서브모듈 `54f0fb464` 병합(`ebfa2a629`) + 정리 커밋 |

- dev가 우선이다. 겹치는 부분은 dev 구현을 남기고, 불필요하거나 디버그용인 것은 지웠다.
- 통합 후 dev는 CPU + NPU + GPU를 정식으로 지원한다. 3devices-f32 대비 빠진 것은 아래 "장부 ID별 처리"의 의도적 제거뿐이다(라인 단위 대조와 이 장부 217행 대조에서 누락 0건, S8-1).

### dev 기준으로 본 변경

dev와 3devices-f32가 같은 파일을 고친 곳은 세 군데뿐이다(`AIETarget.h`, amdxdna `direct_command_buffer.cc`, IREE 서브모듈 포인터. 서브모듈 안에서는 겹치는 파일이 없다). 나머지 겹침은 기능 단위다. dev에 이미 있던 기능을 3devices가 다른 방식으로 확장했거나, 3devices 작업 트리의 우회를 dev가 먼저 정식으로 고친 경우다.

#### 1. 겹친 부분: dev 기능을 어떻게 바꿨나

처리: **dev** = dev 것을 남기고 3devices 쪽을 지움, **합침** = 두 쪽을 함께 고려해 고침, **dev 확장** = dev 구조에 3devices 기능을 얹음.

| 기능 | dev (통합 전) | 3devices-f32 | 통합 후 | 처리 | 커밋 |
|---|---|---|---|---|---|
| 연산 배정 (`AMDAIEAssignDeviceAffinities`) | contraction/conv는 첫 accelerator(NPU) 하나에, 나머지는 CPU에. host만 있는 모듈도 배정 | accelerator 사이 round-robin, vulkan-spirv는 `TEMP` accelerator, respect-marks(디버그), placement-map(실험), accelerator가 없으면 early return | round-robin, vulkan-spirv 정식 accelerator, placement-map은 실험용(기본 꺼짐). respect-marks와 early return은 제거 | 합침 | `4a798514`, `e7060e40`, `131799d7` |
| 링크 표 (`linkFlags`) | CPU → NPU true, NPU → CPU false | CPU ↔ GPU, GPU ↔ NPU 4행 추가(GPU → CPU true) | dev의 2행은 그대로. GPU 4행을 넣되 GPU → CPU는 false로 바꿈(아래 3절) | 합침 | `e3d86485` |
| 플러그인 pass 등록 (`PluginRegistration`) | Flow 단계에 배정 + pad | stream 단계 hook에 ResolveBufferOwner, AnnotateInputDevices. 배정에 옵션 전달 | 둘 다 | 합침 | 병합 `c1c4d937` |
| 컴파일 옵션 (`AIETarget.h`) | ukernel 기본 컴파일러를 peano로 | `respect-affinity-marks`, `placement-map` | dev 변경 + `placement-map` | 합침 | `e7060e40` |
| amdxdna dispatch (`direct_command_buffer.cc`) | tracing 메시지(dispatch 이름) | stderr 로그, hwctx 캐시(`IREE_AMDXDNA_CACHE_HWCTX`) | dev 그대로. 3devices 변경은 모두 빠져 dev 대비 diff 0 | dev | `06aa6260` |
| 실행 시 진단 | `build-trace/` console tracing(`--verbose`) | `[T]`·`[DBG]`·`[BR]` stderr 로그, `IREE_VULKAN_BR_DUMP`, `TEST_EXPORT` | dev 방식만. GPU 로그가 필요하면 tracing으로 넣음 | dev | `06aa6260`, fork `effaf17b4` |
| ONNX 실행 스크립트 (`scripts/onnx/`) | 장치 고정(CPU + NPU), `--numeric`, `--ukernel`, `--verbose` | 스크립트 없이 손으로 쓴 컴파일·실행 명령(W14~W21) | `--devices`(`local-host`, `amdxdna-npu4`, `vulkan-gfx1150`), CPU가 첫 장치가 아니면 오류, GPU + bf16이면 오류, `--verbose`에 GPU 시간 안내 | dev 확장 | `d8e98aec`, `50f8aaa9` |
| 빌드와 컨테이너 | `configure.sh`, Dockerfile, `run-*.sh`(NPU만 전달) | GPU용 HOME·ccache를 따로 씀(`.gitignore` 항목) | dev 스크립트에 Vulkan driver/backend, Vulkan 런타임, render 노드 전달 추가. 3devices의 개인 항목 제거 | dev 확장 | `270fcda3`, `5e2cca69` |
| NPU f32 | f32 pad(`b492865a`), DMA 반복(`990bcab2`)으로 해결 | 작업 트리 우회 W07, W22, W23, W27 | dev | dev | — |
| peano unroll | `-unroll-full-max-count=0` (`XCLBinGen.cpp`) | `-unroll-threshold=0` (W13) | dev | dev | — |
| ukernel bf16 반올림 | `cff1e56e`(반올림 모드 저장·복원 포함) | `set_rnd` 우회(O04, K1)와 S10 명세 | dev. S10 명세 삭제 | dev | `fbf075ab` |
| `mlp_2layer` 가중치 | 상수 가중치 | 런타임 가중치(Pp09~11) | dev | dev | — |
| IREE 서브모듈 | `cd1fa2a1e`(console tracing 시작 시각) | `54f0fb464`(3장치 커밋 8개) | 병합. 겹치는 파일 없음 | 합침 | fork `ebfa2a629` |

#### 2. 추가된 부분: 3devices-f32에만 있던 것

dev에 없던 기능이라 그대로 들여오고, 로그·디버그 스위치·`TEMP` 표식만 정리했다.

| 영역 | 추가된 기능 | 위치 |
|---|---|---|
| 컴파일러: 버퍼 소유 | `AMDAIEResolveBufferOwner`: 여러 장치가 쓰는 버퍼의 `#hal.device.optimal` 후보 중 소유자 하나를 링크 표로 고름 | 플러그인 `Transforms/`, stream 단계 hook |
| 컴파일러: 입력 장치 기록 | `AMDAIEAnnotateInputDevices`: 각 입력을 둘 장치를 reflection(`amdaie.input_devices`)에 기록. 툴링이 `--input_device` 없이 입력을 배치 | 플러그인 `Transforms/`, lit `annotate_input_devices.mlir` |
| 컴파일러: 배치 실험 | `--iree-amdaie-placement-map`: dispatch 번호·이름 규칙으로 장치 지정, 배정 표를 stderr에 출력 | 플러그인, lit `assign_device_affinities_placement_map.mlir` |
| 컴파일러: IREE | stream 단계 plugin hook(`PluginAPI`, `Pipelines.cpp`). `ElideAsyncCopies`가 링크 방향을 보고 transfer를 지울지 판정(S1) | fork |
| 런타임: amdxdna | `export_buffer`(BO를 dma-buf fd로 내보냄), 할당 버퍼에 `placement.device` 기록, `EXPORTABLE` 호환성 | amdxdna driver, `allocator_test` |
| 런타임: Vulkan | BRIDGE: dispatch·copy·fill·update가 다른 장치 버퍼를 만나면 기록 시점에 import(NPU BO는 dma-buf, CPU 버퍼는 host allocation). 다른 장치 semaphore는 host에서 기다리고 signal. external memory fd 확장 활성화, host-visible 메모리 영구 매핑 | fork `vulkan/` |
| 런타임: HAL | heap 버퍼를 export할 때 DISCARD로 map하지 않음(debug build의 0xCD 덮어쓰기 방지) | fork `allocator_heap.c` |
| 툴링 | `iree-run-module`이 입력을 장치별로 옮김(`--input_device` 또는 reflection), lead 장치 | fork `run_module.c`, `context_util.c` |
| 테스트 | 3장치 회귀 26케이스(+ 통합 중 `fc_matvec_lv` 추가로 27) | `tests/regression/` |
| 문서 | 명세 S1~S5, S7~S9, 이 장부 | `docs/specs/`, `docs/CHANGES.md` |

#### 3. 통합 중에 생긴 문제와 처리

두 브랜치를 합치거나 검증하면서 새로 드러난 것이다. 어느 브랜치에도 해결책이 없어서 새로 고쳤다.

| 문제 | 원인 | 처리 | 커밋 |
|---|---|---|---|
| host만 있는 모듈의 lit 실패 | 3devices의 early return이 dev의 host-only 배정을 건너뜀 | early return 제거 | `4a798514` |
| CPU + NPU + GPU에서 `PERMISSION_DENIED` (W20 플래그가 있어야 동작) | GPU가 쓴 버퍼를 다른 장치가 host map하는데 Vulkan 버퍼에 mapping usage가 없음 | host-visible + coherent 메모리에 `MAPPING \| MAPPING_PERSISTENT`. W20 플래그 제거 | fork `8122a888b` |
| CPU와 GPU가 같이 쓰는 transient 버퍼: "transient buffer has not been committed" (W12) | GPU → CPU true라 CPU가 소유자가 될 수 있는데, BRIDGE는 기록 시점에 import하므로 아직 메모리가 없는 transient 버퍼를 못 가져옴 | GPU → CPU를 false로. 공유 버퍼는 GPU가 소유하고 어떤 lead로도 동작 | `e3d86485` |
| lead 장치가 정해지지 않음 | 3devices는 `--device` 순서에 따라 lead가 바뀜 | CPU(`local-host`)를 첫 장치로 고정. lead이자 기본 장치. `--iree-amdaie-stack-size=2048`(W21)도 불필요해져 제거 | `50f8aaa9` |
| `--devices`의 accelerator 순서가 무시됨 | iree-compile은 이름 붙인 장치(`name=driver`)를 `DictionaryAttr`에 넣어 이름순으로 정렬하므로, round-robin이 늘 `amdxdna_npu4`부터 시작함(CPU+NPU+GPU와 CPU+GPU+NPU의 IR이 같았음) | `options.py`가 장치 symbol에 `d<순번>_`을 붙여 이름순을 `--devices` 순서와 맞춤 | `options.py` |
| GPU의 FC·matvec 결과가 정확히 2배 (W08) | SPIR-V `SubgroupReduce`가 subgroup 크기(32)를 실행 파일에 기록하지 않아 RADV가 64로 실행 | `KernelConfig.cpp`가 subgroup 크기를 넘김. 회귀 `fc_matvec_lv` 추가. FC도 GPU에 둘 수 있음 | fork `f1a30ce2c`, `fb26853c` |
| GPU bf16 matmul 실패 | 미해결 | GPU는 f32만. `options.py`가 오류를 냄 | `d8e98aec` |
| opaque-fd(dma-buf) import 메모리 누수 | import한 VkDeviceMemory와 NPU BO 참조를 해제하지 않음(20회 실행에 import 40 / 해제 0) | import 메모리를 소유하는 release, 실패 경로의 fd·메모리 정리 | fork `64d222702` |
| BRIDGE dispatch의 원본 버퍼 수명 | import한 버퍼만 보관하고 원본(host 포인터 alias 대상)은 보관하지 않음 | 원본 binding도 resource set에 보관 | fork `a2b0dfd18` |
| `TEMP` dispatch 메모리 타입(`base_buffer.c`) | heap 보고에만 쓰이고 할당에는 영향이 없음을 확인 | upstream으로 되돌림 | fork `b006be8fc` |
| msync 검사 `#if 0` | 켜면 CPU 버퍼 import가 막혀 회귀 10건 실패(heap 할당이 anonymous mmap이라 msync가 성공) | 꺼 둔 채로 이유를 주석에 남김 | fork `b006be8fc` |

### 장치 구성과 동작

- **장치**: `scripts/onnx/options.py`의 `--devices`로 고른다. 이름은 `<driver>-<target>`이다: `local-host`(llvm-cpu, `target-cpu=host`), `amdxdna-npu4`, `vulkan-gfx1150`. 기본값은 `local-host,amdxdna-npu4`다.
- **CPU는 필수이고 첫 장치다**: host 코드를 실행하는 장치이므로 lead(툴링이 입력을 할당하고 출력을 받는 장치)이자 기본 장치(`--iree-hal-default-device`)다. 첫 장치가 `local-host`가 아니면 `options.py`가 오류를 낸다.
- **배정**(`AMDAIEAssignDeviceAffinities`): contraction/conv dispatch는 accelerator(amd-aie, vulkan-spirv)에 `--devices` 순서대로 round-robin하고, 나머지는 첫 host에 둔다. `--iree-amdaie-placement-map`은 실험용 수단이고 기본값은 비어 있다. 모델별 맵은 커밋하지 않는다.
- **버퍼 소유**(`AMDAIEResolveBufferOwner`): 링크 표(`AMDAIEDevicePlacementUtils.cpp` `linkFlags`)만으로 정한다. 공유 버퍼의 소유자 X는 다른 모든 사용자 Y에 대해 (Y→X) transparent일 때만 될 수 있다. lead는 소유를 정하지 않는다.

| 링크 (src → dst, "src가 dst 메모리에 직접 접근") | 값 | 근거 |
|---|---|---|
| CPU → NPU | true | amdxdna BO는 host 메모리 |
| NPU → CPU | false | amdxdna에 `import_buffer`가 없음 |
| CPU → GPU | true | 이 장비(UMA)의 Vulkan dispatch 메모리는 host-visible. 외장 GPU라면 false로 바꿔야 함 |
| GPU → CPU | false | BRIDGE가 기록 시점에 import하는데, queue 순서 할당(transient)은 그때 아직 메모리가 없음. 공유 버퍼는 GPU가 소유 |
| GPU → NPU | true | amdxdna가 dma-buf를 export하고 Vulkan이 import |
| NPU → GPU | false | amdxdna에 import가 없고 Vulkan은 export하지 않음 |

- **수치 형식**: GPU는 f32만 쓴다(GPU에 `--numeric bf16`을 주면 `options.py`가 오류를 낸다, `# workaround:`). 기준값은 llvm-cpu f32다.
- **빌드와 컨테이너**: `build/`(Vulkan driver + vulkan-spirv backend)와 `build-trace/`(Vulkan driver)를 켰다. dev·deploy 이미지에 Vulkan 런타임이 들어 있고, docker 스크립트가 render 노드와 그 gid를 넘긴다(`270fcda3`).
- **회귀**: `tests/regression/`(v9, 27케이스). PASS = 컴파일 rc + 실행 rc + 기대값. 3장치 관련 변경 때만 ctest와 MLP/VGG e2e에 더해 돌린다.

### 장부 ID별 처리

위 내용을 0~6절의 장부 ID로 다시 묶은 표다.

| 묶음 | 장부 ID | 통합 후 | 커밋 |
|---|---|---|---|
| `[T]`·`[DBG]`·`[BR]` 로그 | D1~D3, Pp·Pi 로그 덩어리, I32 | 제거. 이후 GPU 로그가 필요하면 dev의 `build-trace` 방식으로 넣는다 | `06aa6260`, fork `effaf17b4` |
| 잔재 전역·include | D4, D5 | 제거 | fork `14973745c` |
| 환경 변수 스위치 | D9, W30, W31·W32 | 제거(`CACHE_HWCTX`, `TEST_EXPORT`, `IREE_VULKAN_BR_DUMP`) | `06aa6260`, fork `effaf17b4` |
| respect-marks | O02·O03·O06~O08·O10·O14의 해당 부분, N04, K3, S6 | 제거(옵션, lit, 명세) | `e7060e40` |
| vulkan-spirv 역할 `TEMP` | D6(플러그인) | 정식 Accelerator | `e7060e40` |
| dispatch 메모리 타입 `TEMP` | D6(vulkan `base_buffer.c`), Pi10 | upstream으로 복원. 실제 할당이 아니라 heap 보고에만 쓰여 없어도 회귀·VGG·MLP가 그대로임을 확인 | fork `b006be8fc` |
| msync 검사 `#if 0` | D7, Pi30 | 유지하고 이유를 주석으로 남김. 켜면 CPU 버퍼를 GPU로 import하는 회귀 10건이 실패함 | fork `b006be8fc` |
| 무조건 `MAPPING` | D8, Pi25 | 대체: host-visible + coherent 메모리에 `MAPPING \| MAPPING_PERSISTENT`(CPU dispatch는 scoped, 툴링·BRIDGE는 persistent로 map) | fork `8122a888b` |
| GPU → CPU 링크, CPU↔GPU 동률 | O17, W12 | 링크를 false로 바꿔 동률을 없앰(공유 버퍼는 GPU 소유). 어떤 lead로도 동작 | `e3d86485` |
| `--iree-hal-external-resources-mappable` | W20 | 불필요해져 제거 | `e3d86485` |
| `--iree-amdaie-stack-size=2048` | W21 | 불필요(CPU lead에서 확인), 제거 | `50f8aaa9` |
| FC(M=1)가 GPU에서 2배 | W08, W29 | 해결: SPIR-V `SubgroupReduce`가 subgroup 크기를 기록하지 않아 RADV가 64로 실행하던 것을 고침. 회귀 `fc_matvec_lv` 추가 | fork `f1a30ce2c`, `fb26853c` |
| 배정 맵 | W01~W06, W09, W10, K2 | 맵은 커밋하지 않음(기본은 round-robin, 맵은 실험 도구) | — |
| NPU f32 우회 | W07, W22, W23, W27 | dev 수정으로 대체: f32 pad `b492865a`, DMA 반복 `990bcab2`. W23의 N=1000·N-split f32는 VGG f32 e2e(corr 1.0)로 확인 | — |
| `-unroll-threshold=0` | W13 | dev의 `-unroll-full-max-count=0`(`XCLBinGen.cpp`) | — |
| ukernel `set_rnd` | O04, K1, S10 | dev `cff1e56e`(반올림 모드 저장·복원 포함). S10 명세 삭제 | `fbf075ab` |
| `mlp_2layer` 런타임 가중치 | Pp09~11 | dev 버전(상수 가중치) 유지 | — |
| 회귀 | N09의 `docs/regression/`, W28, D2·D3의 판정 의존 | `tests/regression/`(v9)로 이동. 로그 판정·지표 열·stream 컴파일 지표 제거, `IREE_BUILD=/workspace/build`, 출력 `build/regression/` | `a3b2c714`, `f8fca56e` |
| host-only early return | (3devices `3853fbfa` 계열) | 제거(host만 있는 모듈도 다시 배정) | `4a798514` |
| 실험 모델 | `models/chain_nn`, `models/mlp_parallel`, N07 | 삭제(chain2는 회귀 모델과 같음) | `3f677c95` |
| `.gitignore` 개인 항목 | — | 컨테이너 HOME 파일만 루트 기준으로 남김 | `5e2cca69` |

유지한 것: D10(placement-map stderr 표, 플래그를 줄 때만), D11(`EXPERIMENT` 표식), W11·K4(`accelIdx = 0`, round-robin 시작점), W14~W17(VGG 레시피 플래그, `options.py`), W19(회귀 수동 모드), amdxdna export·`placement.device`·allocator_test(O18~O24, S2), `AMDAIEAnnotateInputDevices`와 입력 자동 배치(S5), `AMDAIEResolveBufferOwner`와 stream hook(S8), `ElideAsyncCopies` 양방향 판정(S1), Vulkan BRIDGE와 영구 매핑(S3·S4·S9).

### 검증 (S8-3, repo `d0c9960e` / fork `32dad7667`)

- ctest amd-aie 223/223, SPIR-V lit(`ctest -R SPIRV`) 65/65, 회귀 27/27(수동·자동) 2회.
- CPU+NPU: MLP·VGG의 bf16(ukernel all), f32(all), f32(none) 출력과 CPU-only 기준값이 dev(`45c18ae7`)와 바이트 단위로 같다. VGG bf16 배정은 92 dispatch(npu 23 / cpu 69)이고 dispatch별 장치까지 같다. VGG bf16 실행 2.13 s.
- GPU 조합(f32): CPU+GPU MLP 0.03 s / VGG 0.37 s, CPU+NPU+GPU MLP 0.14 s / VGG 14.19 s, CPU+GPU+NPU VGG 14.16 s. 모두 corr 1.0, VGG top-1 일치.
- 오류 확인: GPU + bf16, CPU가 첫 장치가 아닌 경우 모두 rc 1.

### 알려진 제한

| 항목 | 내용 |
|---|---|
| W11 | NPU 3입력 dispatch의 `no producer DMA channel available`. 원인 미규명, round-robin 순서로 피하고 있음 |
| GPU bf16 | GPU bf16 matmul이 실패해 GPU는 f32만 씀 |
| `--verbose` GPU 시간 | GPU dispatch 시간은 따로 잡히지 않고 host 시간에 들어감 |
| 링크 표 전제 | CPU → GPU true는 host-visible dispatch 메모리(UMA) 전제. GPU → CPU는 BRIDGE가 실행 시점에 import하게 되면 true로 되돌릴 수 있음 |
| semaphore BRIDGE | 외부 signal semaphore는 submit당 16개까지(넘으면 버림), fence 실패 시 외부 semaphore를 fail 처리하지 않음 |
| dispatch BRIDGE | binding이 9개 이상이면 BRIDGE 없이 오류(의도) |
| opaque-fd import | `external_memory_dma_buf`가 없으면 OPAQUE_FD 형식으로 질의(spec 위반, 이 장비는 dma_buf 있음) |
| 툴링 device group | 전역 device group을 release하지 않음(프로세스당 한 번) |
| amdxdna `export_buffer` | 입력 버퍼가 amdxdna 버퍼인지 검사하지 않음 |
| `ElideAsyncCopies` | 양쪽 방향 중 하나라도 transparent면 transfer를 지움(S1). upstream 동작과 다름 |

### 남은 일

- SPIR-V subgroup 크기 수정(`f1a30ce2c`)을 upstream(iree-org/iree)에 PR로 보내고, 반영되면 fork 커밋을 뺀다. `ElideAsyncCopies` 변경(S1)도 같은 메모에 적는다.
- 미룬 것: 날짜별 문서 재배치, hwctx 재사용 최적화, `--verbose`의 GPU 시간, 장치별 수치 형식, GPU bf16.

## 0. 기준

| 저장소 | 기준 커밋 | 작업 트리 상태 |
|---|---|---|
| 플러그인 `ace-knu/iree-amd-aie` | HEAD `44e6bcd4` (브랜치 `junho/npu-gpu-dmabuf`, `origin/dev` `f7a5c86f` + Junho Kwak 8/31 커밋 3개) | 추적 파일 20개 수정, 새 파일 5개(+ 스크립트·문서) |
| IREE 서브모듈 `third_party/iree` (`ace-knu/iree`) | `c73928f0b9` (detached, `origin/amd-aie-flow-hook`·`origin/bert-onnx`에 포함) | 추적 파일 28개 수정, 빈 새 파일 6개(출처 불명). **서브모듈 포인터는 HEAD와 같다**(상위 저장소에서 서브모듈은 "변경 없음"으로 보인다. 변경은 서브모듈 작업 트리 안에만 있다) |

출처 구분 방법(확인함):
- **전임자** = 9/28 인수인계 시점 작업 트리 − HEAD. 9/28 기준선 patch(`docs/2026-09-28/01_인수인계_코드현황_파악/baseline_plugin.patch`, `baseline_iree.patch`, `untracked_AMDAIEResolveBufferOwner.cpp`)를 HEAD 파일에 적용해 기준선 파일을 재구성했다(두 patch 모두 깨끗이 적용됨).
- **우리** = 현재 작업 트리 − 기준선. 덩어리마다 세션 코드 patch(`변경내역*.patch`)의 추가·삭제 줄과 대조해 출처 세션을 정했다(대조 스크립트·결과: 챕터 01 폴더 `attribute.py`, `attribution.tsv`). **출처를 못 찾은 덩어리는 0개다.**
- 근거 diff: 챕터 01 폴더 `current_plugin.patch`·`current_iree.patch`(HEAD 대비 전체), `ours_since_0928_plugin.patch`·`ours_since_0928_iree.patch`(기준선 대비 우리 몫).

## 1. 규모 요약

| 구분 | 플러그인 | IREE 서브모듈 | 합계 |
|---|---|---|---|
| HEAD 대비 수정 파일 | 20 | 28 | 48 |
| 그중 전임자 몫이 있는 파일 | 15 (모델 2 포함) | 23 | 38 |
| 그중 우리 몫이 있는 파일 | 14 | 8 | 22 |
| (둘 다 있는 파일) | 9 | 3 | 12 |
| 전임자 덩어리(9/28 기준선 hunk) | 51 (기능 19, 디버그 25, 기능+디버그 6, 바이너리 1) | 57 (기능 30, 디버그 1, 기능+디버그 22, 빈 줄만 4) | 108 |
| 우리 덩어리(기준선 대비 hunk) | 24 | 32 | 56 |
| 우리 변경 줄(+/− 합, 빈 줄 제외) | 509 | 830 | 1,339 |

분류 기준: "디버그" = 추가 줄이 모두 로그(`[T]`/`[DBG]`/`[BR]` 출력, 타이밍 변수)나 잔재(`g_last_bindings`, `execinfo`). `TEMP:`·`#if 0`·`EXPERIMENT`는 동작을 바꾸므로 "기능"으로 셌다(4절 D6~D8, D11).
| 새 소스·테스트 파일 | 5 (우리 4, 전임자 1) | 0 | 5 |

디버그·임시 표식(현재 작업 트리, 수정·신규 소스 대상 grep): `[T]` 30, `[DBG]` 21, `[BR]` 19, `BRIDGE` 주석 17, `TEMP:` 2(코드 표식), `EXPERIMENT` 5, 환경 변수 스위치 3. 9/28 대비 `[BR]` 3줄이 늘었고(우리: BRIDGE 오류 로그 이동, 매핑 없는 바인딩 로그, 입력 자동 배치 로그), 나머지는 전임자 것 그대로다. 4절 참고.

## 2. 우리 변경 — 덩어리별 (9/28 기준선 이후)

줄 번호는 **현재 작업 트리** 기준(`+시작,길이`). 명세 S번호는 챕터 02 문서.

### 2-1. 플러그인 (`compiler/plugins/target/AMD-AIE/iree-amd-aie/`, `runtime/src/iree-amd-aie/`)

| ID | 파일 | 덩어리 | 내용 | 출처(세션·챕터) | 이유 문서 | 명세 |
|---|---|---|---|---|---|---|
| O01 | `PluginRegistration.cpp` | +74,24 | (a) stream hook에 `AMDAIEAnnotateInputDevices` 등록(`ResolveBufferOwner` 바로 뒤) (b) flow hook에서 `AssignDeviceAffinities`에 옵션 2개 전달 | (a) 9/29 세션2 챕터02 (b) 10/1 세션2 챕터02 + 10/2 세션2 챕터02 | `docs/2026-09-29/2_VGG16상수묶음확인_input_device자동화/02_input_device_자동화_구현/README.md`, `docs/2026-10-01/2_C안_1단계_S1_respect_marks/02_구현과_기본값_불변_검증/README.md`, `docs/2026-10-02/2_f32_3장치_VGG16_임의배정/02_배정_지정_플래그/README.md` | S5, S6, S7 |
| O02 | `Target/AIETarget.h` | +67,15 | 옵션 필드 `respectAffinityMarks`, `placementMap` | 10/1 세션2 챕터02, 10/2 세션2 챕터02 | 위 10/1·10/2 문서 | S6, S7 |
| O03 | `Target/AIETarget.h` | +187,27 | CLI 옵션 `--iree-amdaie-respect-affinity-marks`, `--iree-amdaie-placement-map` | 10/1, 10/2 (같음) | 같음 | S6, S7 |
| O04 | `Target/uKernels/npu4/peano/matmul.cc` | +185,7 | `matmul_vectorized_bf16_f32` 첫 줄 `set_rnd(rnd_conv_even);` | 10/19 세션1 챕터04 | `docs/2026-10-19/1_NPU_bf16_벡터화_범위조사/04_U1_반올림_설정_확인/README.md` | S10 |
| O05 | `Transforms/AMDAIEAssignDeviceAffinities.cpp` | +33,10 | include 4개 (StringExtras, StringSwitch, MemoryBuffer, raw_ostream) | 10/2 세션2 챕터02 | 10/2 문서 | S7 |
| O06 | 같은 파일 | +115,240 | `readExecutableMark`(respect-marks) / placement-map 규칙 구조체·파서·`ruleMatches`(placement-map) / 옵션 생성자 | 10/1(표식 읽기·생성자) + 10/2(map) | 10/1·10/2 문서 | S6, S7 |
| O07 | 같은 파일 | +395,102 | dispatch walk 앞단: placement-map 분기 → respect-marks 분기 → (기존) 라운드로빈 | 10/2 + 10/1 | 같음 | S6, S7 |
| O08 | 같은 파일 | +504,67 | `markFailed` 실패 처리(10/1), 쓰이지 않은 번호 규칙 오류 + stderr 배정 요약(10/2) | 10/1 + 10/2 | 같음 | S6, S7 |
| O09 | 같은 파일 | +583,12 | topology 주입 대상에 "첫 번째가 아닌 host"도 포함(placement-map일 때만) | 10/2 | 10/2 문서 | S7 |
| O10 | 같은 파일 | +607,9 | `createAMDAIEAssignDeviceAffinitiesPass(options)` | 10/1 | 10/1 문서 | S6 |
| O11 | `Transforms/CMakeLists.txt` | +50,7 | `AMDAIEAnnotateInputDevices.cpp` 추가 | 9/29 세션2 챕터02 | 9/29 문서 | S5 |
| O12 | `Transforms/PassDetail.h` | +25,7 | `GEN_PASS_DEF_AMDAIEANNOTATEINPUTDEVICES` | 9/29 | 9/29 문서 | S5 |
| O13 | `Transforms/Passes.h` | +82,14 | 배정 패스 생성자 옵션 인자(10/1), AnnotateInputDevices 생성자(9/29) | 10/1 + 9/29 | 각 문서 | S5, S6 |
| O14 | `Transforms/Passes.td` | +85,22 | 배정 패스 옵션 `respect-marks`, `placement-map` | 10/1 + 10/2 | 각 문서 | S6, S7 |
| O15 | `Transforms/Passes.td` | +110,13 | `AMDAIEAnnotateInputDevices` 정의 | 9/29 | 9/29 문서 | S5 |
| O16 | `Transforms/test/CMakeLists.txt` | +10,15 | lit 3개 등록(annotate 9/29, placement_map 10/2, respect_marks 10/1) | 9/29, 10/2, 10/1 | 각 문서 | S5, S6, S7 |
| O17 | `Transforms/Utils/AMDAIEDevicePlacementUtils.cpp` | +66,12 | `linkFlags` GPU→CPU 줄의 옛 TODO 주석을 "BRIDGE로 지원됨, cg_topo lead=0 검증"으로 교체(**주석만**) | 10/2 세션1 챕터00 | `docs/2026-10-02/1_C안_2단계_연산단위배정/00_사전정리/README.md` | S8 |
| O18 | `runtime/…/amdxdna/allocator.cc` | +39,19 | allocator 구조체에 unowned `parent_device` | 9/28 챕터10 | `docs/2026-09-28/10_amdxdna_동기할당_placement_기록/README.md` | S2 |
| O19 | 같은 파일 | +144,11 | 동기 할당 placement에 `.device = parent_device` | 9/28 챕터10 | 같음 | S2 |
| O20 | 같은 파일 | +204,7 | `iree_hal_amdxdna_allocator_create` 시그니처에 `parent_device` | 9/28 챕터10 | 같음 | S2 |
| O21 | 같은 파일 | +214,8 | 생성자 호출 | 9/28 챕터10 | 같음 | S2 |
| O22 | `runtime/…/amdxdna/allocator.h` | +12,10 | 선언·주석 | 9/28 챕터10 | 같음 | S2 |
| O23 | `runtime/…/amdxdna/device.cc` | +119,8 | `allocator_create`에 device 넘김 | 9/28 챕터10 | 같음 | S2 |
| O24 | `runtime/…/amdxdna/test/allocator_test.cc` | +16,8 | 테스트를 새 시그니처에 맞춤(`/*parent_device=*/nullptr`) | 9/29 세션1 챕터02 | `docs/2026-09-29/1_구조전환_offset회귀_input_device준비/02_allocator_test_시그니처수정/README.md` | S2 |

### 2-2. IREE 서브모듈 (`third_party/iree/`)

| ID | 파일 | 덩어리 | 내용 | 출처 | 이유 문서 | 명세 |
|---|---|---|---|---|---|---|
| I01 | `compiler/…/Stream/Transforms/ElideAsyncCopies.cpp` | +723,16 | `hasPinnedPlacement(Value)` 공개 진입점 | 9/28 챕터06 | `docs/2026-09-28/06_ElideAsyncCopies_양방향판정_수정/README.md` | S1 |
| I02 | 같은 파일 | +838,62 | `hasPinnedPlacement` 본체(import/상수/global/lifetime constant·variable, 같은 lifetime transfer·clone은 따라감) | 9/28 챕터06 | 같음 | S1 |
| I03 | 같은 파일 | +1100,38 | `isSafeToElideTransferOp` topology 판정을 양방향으로 | 9/28 챕터06 | 같음 | S1 |
| I04 | `compiler/…/Stream/Transforms/test/elide_async_copies_topology.mlir` | +239,153 | lit 5모듈(비대칭 transient 2, 고정 import 보존 2(직접·cast/clone), 고정 import elide 1, 상수 보존 1) | 9/28 챕터06 | 같음 | S1 |
| I05 | `runtime/…/vulkan/direct_command_buffer.cc` | +7,11 | include `<cinttypes>`, `<cstdlib>`, `<cstring>` | 9/28 챕터11, 10/2 세션3 챕터01 | 각 문서 | S3, S4 |
| I06 | 같은 파일 | +19,11 | include `native_buffer.h`, `sparse_buffer.h` | 9/28 챕터09 | `docs/2026-09-28/09_Vulkan_남의버퍼_명시적에러(D)/README.md` | S3 |
| I07 | 같은 파일 | +74,217 | `is_vulkan_buffer`·`foreign_buffer_error`·`require_vulkan_buffer`(D) + `bridge_error`·**`resolve_buffer_ref`**(A, 전임자 dispatch BRIDGE 본문을 옮겨 공통화, 실패 시 에러 반환, `byte_offset` 반영) | 9/28 챕터09 + 11 | 09 문서, `docs/2026-09-28/11_Vulkan_copy_fill_update_BRIDGE(A)/README.md` | S3 |
| I08 | 같은 파일 | +580,13 | `execution_barrier`의 buffer barrier에 남의 버퍼면 에러 | 9/28 챕터09 | 09 문서 | S3 |
| I09 | 같은 파일 | +680,13 | `wait_events`의 buffer barrier에 같은 검사 | 9/28 챕터09 | 09 문서 | S3 |
| I10 | 같은 파일 | +781,20 | `fill_buffer` target BRIDGE | 9/28 챕터11 | 11 문서 | S3 |
| I11 | 같은 파일 | +930,13 | `update_buffer` target BRIDGE | 9/28 챕터11 | 11 문서 | S3 |
| I12 | 같은 파일 | +944,7 | 위 update의 ref 교체 | 9/28 챕터11 | 11 문서 | S3 |
| I13 | 같은 파일 | +992,17 | `copy_buffer` source/target BRIDGE | 9/28 챕터11 | 11 문서 | S3 |
| I14 | 같은 파일 | +1012,7 | copy 기록에 resolved ref 사용 | 9/28 챕터11 | 11 문서 | S3 |
| I15 | 같은 파일 | +1036,19 | `iree_hal_vulkan_br_dump_enabled()`(환경 변수 `IREE_VULKAN_BR_DUMP=1`) | 10/2 세션3 챕터01 | `docs/2026-10-02/3_Vulkan_map_겹침_수정_VGG16_3장치/01_디버그로그_map_분리_가설확인/README.md` | S4 |
| I16 | 같은 파일 | +1064,18 | 덤프 꺼짐이면 바인딩 값을 map하지 않고 ref 정보만 `[BR]`로 출력 | 10/2 세션3 챕터01 | 같음 | S4 |
| I17 | 같은 파일 | +1099,30 | dispatch 바인딩 BRIDGE를 `resolve_buffer_ref` 호출로 교체(전임자 99줄 삭제), 모든 바인딩·indirect 인자 Vulkan 버퍼 검사 | 9/28 챕터11 + 09 | 09·11 문서 | S3 |
| I18 | `runtime/…/vulkan/native_buffer.cc` | +11,7 | include `mutex.h` | 10/2 세션3 챕터02 | `docs/2026-10-02/3_Vulkan_map_겹침_수정_VGG16_3장치/02_map_참조수_근본수정/README.md` | S4 |
| I19 | 같은 파일 | +20,13 | `mapping_mutex`, `mapped_ptr` 필드 | 10/2 세션3 챕터02 | 같음 | S4 |
| I20 | 같은 파일 | +39,10 | `iree_hal_vulkan_native_buffer_isa` | 9/28 챕터09 | 09 문서 | S3 |
| I21 | 같은 파일 | +74,8 | 필드 초기화 | 10/2 세션3 챕터02 | 같음 | S4 |
| I22 | 같은 파일 | +93,14 | destroy에서 영구 매핑 해제 | 10/2 세션3 챕터02 | 같음 | S4 |
| I23 | 같은 파일 | +147,43 | `map_range`: 처음 한 번 `vkMapMemory(0, VK_WHOLE_SIZE)`, 이후 `mapped_ptr + offset`, non-coherent 읽기는 invalidate | 10/2 세션3 챕터02 | 같음 | S4 |
| I24 | 같은 파일 | +207,25 | `unmap_range`: unmap 안 함, non-coherent 쓰기는 flush | 10/2 세션3 챕터02 | 같음 | S4 |
| I25 | `runtime/…/vulkan/native_buffer.h` | +45,9 | `_isa` 선언 | 9/28 챕터09 | 09 문서 | S3 |
| I26 | `runtime/…/vulkan/sparse_buffer.cc` | +31,10 | `iree_hal_vulkan_sparse_buffer_isa` | 9/28 챕터09 | 09 문서 | S3 |
| I27 | `runtime/…/vulkan/sparse_buffer.h` | +35,9 | 선언 | 9/28 챕터09 | 09 문서 | S3 |
| I28 | `runtime/…/tooling/run_module.c` | +52,8 | `--input_device` 도움말에 reflection 대체 설명 | 9/29 세션2 챕터02 | `docs/2026-09-29/2_VGG16상수묶음확인_input_device자동화/02_input_device_자동화_구현/README.md` | S5 |
| I29 | 같은 파일 | +218,32 | `iree_tooling_find_device_by_id`(장치 ID 패턴 `<id>*`로 `--device` 목록 검색) | 9/29 | 같음 | S5 |
| I30 | 같은 파일 | +275,28 | 플래그가 없으면 reflection `amdaie.input_devices` 사용 | 9/29 | 같음 | S5 |
| I31 | 같은 파일 | +307,66 | 항목별 처리: `""` 둠, `?` 에러, `a|b` 동률은 lead가 있으면 lead, 아니면 첫 후보, 목록에 없는 ID는 에러 | 9/29 | 같음 | S5 |
| I32 | 같은 파일 | +411,9 | `[BR] input[i] -> device N (ok, flag/reflection)` 로그 | 9/29 | 같음 | S5 |

### 2-3. 새 파일 (untracked)

| ID | 경로 | 출처 | 내용 | 이유 문서 | 명세 |
|---|---|---|---|---|---|
| N01 | `compiler/…/Transforms/AMDAIEAnnotateInputDevices.cpp` | 우리, 9/29 세션2 챕터02 | 입력 장치 reflection 기록 패스 | 9/29 세션1 챕터04(설계), 세션2 챕터02(구현) | S5 |
| N02 | `compiler/…/Transforms/AMDAIEResolveBufferOwner.cpp` | **전임자**(9/28 기준선 사본과 바이트 동일, 확인함 `cmp`) | 소유자 결정 패스 | `docs/2026-09-28/01_인수인계_코드현황_파악/README.md` 3-2, `03_소유자결정_topology_검증보강/` | S8 |
| N03 | `compiler/…/Transforms/test/annotate_input_devices.mlir` | 우리, 9/29 | lit | 9/29 세션2 챕터02 | S5 |
| N04 | `compiler/…/Transforms/test/assign_device_affinities_respect_marks.mlir` | 우리, 10/1 | lit | 10/1 세션2 챕터02 | S6 → 삭제(`e7060e40`) |
| N05 | `compiler/…/Transforms/test/assign_device_affinities_placement_map.mlir` | 우리, 10/2 | lit | 10/2 세션2 챕터02 | S7 |
| N06 | `scripts/lock/{check-containers,host-load,lock-common,status,with-npu-lock}.sh` | **팀**(`origin/bert-onnx` 커밋 `49514d9c` 계열에서 `git show`로 꺼냄, 9/29) | 공용 NPU·빌드 잠금 | `docs/2026-09-29/3_VGG16_2장치재현_3장치/03_…`, CLAUDE.md 0 "공용 잠금" | 커밋 대상 아님(팀 브랜치에 이미 있음) |
| N07 | `models/mlp_parallel/debug_out/` (12페이즈 덤프 + vmfb, 9/1) | 전임자 임시 산출물 | 디버그 덤프 | 9/28 01 3-1 | 커밋 제외 → 폴더째 삭제(`3f677c95`) |
| N08 | `third_party/iree/runtime/src/iree/hal/drivers/vulkan/{access,, memory_type,, memory_types.dispatch_idx, type, type,, usage,}` (0바이트 6개, 8/27 21:04) | **출처 불명**(전임자 시기). 이름이 C 코드 조각이라 셸 리다이렉트 실수로 생긴 파일로 **추정** | 없음 | 9/28 01 `untracked_files.txt` | 커밋 제외 |
| N09 | `docs/` 아래 날짜 폴더(9/28~10/20), `docs/regression/`, `docs/CHANGES.md` | 우리 | 작업 문서·회귀 스크립트 | — | 커밋 여부는 챕터 04 결정 → 회귀는 `tests/regression/`(`a3b2c714`), 날짜 폴더는 커밋 안 함 |

## 3. 전임자 변경 — 덩어리별 (HEAD → 9/28 기준선)

줄 번호는 **9/28 기준선 patch의 hunk 머리**(그때 작업 트리 줄)다. "현존" = 그 덩어리의 추가 줄이 지금 파일에 남아 있는 정도(확인함, 줄 문자열 대조). "분류": 기능 / 디버그(로그만) / 기능+디버그(한 덩어리에 섞임). 내용 설명과 판정(동작·로그·보류)은 `docs/2026-09-28/01_인수인계_코드현황_파악/README.md` 3-2(파일 목록), 3-4(디버그 인벤토리 A·B·C)를 따른다.

이유 문서 연결(덩어리 묶음별):
- stream hook(Pi01~03), `ResolveBufferOwner` 등록(Pp01, Pp03~07), `linkFlags`(Pp08): 9/28 01 3-2, `docs/2026-09-28/03_소유자결정_topology_검증보강/README.md`, `04_transparent_access_방향정의_확인/README.md` → 명세 S8
- `accelIdx = 0`(Pp02): 9/28 01 3-5, CLAUDE.md 3절 "배정" → 명세 S8(배정 부분)
- amdxdna dma-buf export(Pp12~17 allocator, Pp46 native.h, Pp47~48 native_linux_kmq), hwctx 캐시(Pp43), Vulkan 남의 버퍼 dispatch BRIDGE(Pi13~14), Vulkan dma-buf import·memory type·확장(Pi10, Pi21~33, Pi37~40), heap export DISCARD 제거(Pi04), Vulkan 세마포어 BRIDGE(L1)(Pi15~20 direct_command_queue, Pi41~42 vulkan_device), allocator.select·fence 로그(Pi43~45), `--input_device`·device group(Pi46~49 context_util, Pi55~57 run_module): 9/28 01 3-2·3-4 A, `docs/2026-09-28/08_Vulkan_copy_buffer_BRIDGE_조사/README.md`, `docs/2026-10-19/3_세마포어_Host중개_조사/` → 명세 S9
- mlp_2layer 모델(Pp09~11): 9/28 01 3-5 "판단 보류" → 명세 없음(모델)
- 로그만(디버그): 9/28 01 3-4 B → 4절

**이유 문서가 9/28 01의 "목적(추정)" 한 줄뿐인 덩어리**(전임자가 남긴 설명 없음, 우리가 코드로 추정): Vulkan `base_buffer.c` memory type HOST_VISIBLE 우선(Pi10, TEMP), `native_allocator.cc` 무조건 `MAPPING`(Pi25)·msync 검사 `#if 0`(Pi30), `module.c` allocator.select 로그(Pi43~45), `buffer.c`·`command_buffer_validation.c` 로그(Pi05~09). 이 중 동작에 영향을 주는 것(Pi10, Pi25, Pi30)은 명세 S9에 "근거 문서 없음"으로 표시했다.

### 3-1. 플러그인 (51덩어리)

| ID | 파일 | 9/28 hunk | 함수 | +/− | 분류 | 현존 | 첫 추가 줄 |
|---|---|---|---|---|---|---|---|
| Pp01 | `iree-amd-aie/PluginRegistration.cpp` | `-70,6 +70,12` | struct AMDAIESession | +4/-0 | 기능 | 전부 | void extendStreamTransformPassPipeline(OpPassManager &passMa |
| Pp02 | `Transforms/AMDAIEAssignDeviceAffinities.cpp` | `-159,7 +159,7` | void AMDAIEAssignDeviceAffinitiesPass::r | +1/-1 | 기능 | 전부 | unsigned accelIdx = 0; |
| Pp03 | `Transforms/CMakeLists.txt` | `-49,6 +49,7` | iree_cc_library( | +1/-0 | 기능 | 전부 | "AMDAIEResolveBufferOwner.cpp" |
| Pp04 | `Transforms/CMakeLists.txt` | `-133,6 +134,7` | iree_cc_library( | +1/-0 | 기능 | 전부 | iree::compiler::Dialect::Stream::IR |
| Pp05 | `Transforms/PassDetail.h` | `-24,6 +24,7` | namespace mlir::iree_compiler::AMDAIE { | +1/-0 | 기능 | 전부 | #define GEN_PASS_DEF_AMDAIERESOLVEBUFFEROWNER |
| Pp06 | `Transforms/Passes.h` | `-83,6 +83,7` | std::unique_ptr<Pass> createAMDAIEAssign | +1/-0 | 기능 | 전부 | std::unique_ptr<Pass> createAMDAIEResolveBufferOwnerPass(); |
| Pp07 | `Transforms/Passes.td` | `-87,6 +87,13` | def AMDAIEAssignDeviceAffinities : | +5/-0 | 기능 | 전부 | def AMDAIEResolveBufferOwner : |
| Pp08 | `Utils/AMDAIEDevicePlacementUtils.cpp` | `-26,43 +26,72` | DeviceRole deviceRole(StringRef backend) | +53/-28 | 기능 | 44/53줄 | // 'transparent_access' on a link (src -> dst) means "src ca |
| Pp09 | `mlp_2layer/export_mlp_2layer.py` | `-1,33 +1,26` |  | +3/-18 | 기능 | 전부 | Weights w1, w2 are runtime inputs (x, w1, w2), not baked ini |
| Pp10 | `mlp_2layer/export_mlp_2layer.py` | `-35,11 +28,11` | graph = h.make_graph( | +2/-3 | 기능 | 전부 | [vi("x"), vi("w1"), vi("w2")],  # weights are runtime inputs |
| Pp11 | `mlp_2layer/mlp_2layer.onnx` | `BINARY` |  | +0/-0 | 바이너리 | - |  |
| Pp12 | `amdxdna/allocator.cc` | `-5,6 +5,8` |  | +2/-0 | 기능 | 전부 | #include <unistd.h> |
| Pp13 | `amdxdna/allocator.cc` | `-13,6 +15,26` |  | +18/-0 | 기능 | 전부 | static iree_status_t iree_hal_amdxdna_allocator_export_buffe |
| Pp14 | `amdxdna/allocator.cc` | `-40,7 +62,9` | iree_hal_amdxdna_allocator_query_buffer_ | +3/-1 | 기능 | 전부 | // Every amdxdna allocation is a DRM BO exportable via PRIME |
| Pp15 | `amdxdna/allocator.cc` | `-80,16 +104,22` | iree_hal_amdxdna_allocator_query_buffer_ | +5/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp16 | `amdxdna/allocator.cc` | `-125,6 +155,21` | static iree_status_t iree_hal_amdxdna_al | +13/-0 | 기능+디버그 | 전부 | if (getenv("IREE_AMDXDNA_TEST_EXPORT")) { |
| Pp17 | `amdxdna/allocator.cc` | `-238,7 +283,7` | const iree_hal_allocator_vtable_t iree_h | +1/-1 | 기능 | 전부 | .export_buffer = iree_hal_amdxdna_allocator_export_buffer, |
| Pp18 | `amdxdna/buffer.cc` | `-96,6 +96,7` | static iree_status_t iree_hal_amdxdna_bu | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp19 | `amdxdna/buffer.cc` | `-149,6 +150,7` | static iree_status_t iree_hal_amdxdna_bu | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp20 | `amdxdna/device.cc` | `-473,6 +473,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp21 | `amdxdna/device.cc` | `-617,6 +618,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp22 | `amdxdna/device.cc` | `-703,6 +705,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp23 | `amdxdna/device.cc` | `-874,6 +877,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp24 | `amdxdna/device.cc` | `-921,6 +925,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp25 | `amdxdna/device.cc` | `-940,6 +945,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp26 | `amdxdna/device.cc` | `-972,6 +978,7` | static iree_status_t iree_hal_amdxdna_de | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[DBG] amdxdna queue_dispatch called\n"); |
| Pp27 | `amdxdna/device.cc` | `-1092,15 +1099,24` | iree_status_t iree_hal_amdxdna_device_ge | +9/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[T] ctx HIT  '%.*s' t=%.3fms\n", |
| Pp28 | `amdxdna/direct_command_buffer.cc` | `-1,3 +1,4` |  | +1/-0 | 기능 | 전부 | #include <cstdlib> |
| Pp29 | `amdxdna/direct_command_buffer.cc` | `-179,6 +180,7` | static void iree_hal_amdxdna_direct_comm | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp30 | `amdxdna/direct_command_buffer.cc` | `-194,6 +196,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp31 | `amdxdna/direct_command_buffer.cc` | `-215,6 +218,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp32 | `amdxdna/direct_command_buffer.cc` | `-227,6 +231,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp33 | `amdxdna/direct_command_buffer.cc` | `-242,6 +247,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp34 | `amdxdna/direct_command_buffer.cc` | `-254,6 +260,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp35 | `amdxdna/direct_command_buffer.cc` | `-286,6 +293,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp36 | `amdxdna/direct_command_buffer.cc` | `-324,6 +332,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp37 | `amdxdna/direct_command_buffer.cc` | `-343,6 +352,14` | static iree_status_t iree_hal_amdxdna_di | +6/-0 | 기능+디버그 | 전부 | const float* f = (const float*)source_mapping.contents.data; |
| Pp38 | `amdxdna/direct_command_buffer.cc` | `-533,6 +550,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp39 | `amdxdna/direct_command_buffer.cc` | `-713,6 +731,7` | static iree_status_t iree_hal_amdxdna_su | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp40 | `amdxdna/direct_command_buffer.cc` | `-774,6 +793,8` | static iree_status_t iree_hal_amdxdna_di | +2/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp41 | `amdxdna/direct_command_buffer.cc` | `-861,6 +882,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp42 | `amdxdna/direct_command_buffer.cc` | `-938,6 +960,9` | static iree_status_t iree_hal_amdxdna_di | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[T] dispatch enter t=%.3fms\n", iree_time_n |
| Pp43 | `amdxdna/direct_command_buffer.cc` | `-975,12 +1000,24` | static iree_status_t iree_hal_amdxdna_di | +17/-6 | 기능 | 전부 | // Reuse a cached hw context keyed on (PDI, kernel) instead  |
| Pp44 | `amdxdna/direct_command_buffer.cc` | `-1051,6 +1088,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] dispatch exit t=%.3fms\n", iree_time_no |
| Pp45 | `amdxdna/direct_command_buffer.cc` | `-1058,6 +1096,7` | static iree_status_t iree_hal_amdxdna_di | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[T] %s t=%.3fms\n", __func__, iree_time_now |
| Pp46 | `amdxdna/native.h` | `-125,6 +125,12` | uint64_t iree_hal_amdxdna_native_buffer_ | +4/-0 | 기능 | 전부 | // Exports the buffer's underlying BO as a dma-buf file desc |
| Pp47 | `amdxdna/native_linux_kmq.cc` | `-4,6 +4,7` |  | +1/-0 | 기능 | 전부 | #include <fcntl.h> |
| Pp48 | `amdxdna/native_linux_kmq.cc` | `-416,6 +417,28` | iree_device_size_t iree_hal_amdxdna_nati | +17/-0 | 기능 | 전부 | iree_status_t iree_hal_amdxdna_native_buffer_export_fd( |
| Pp49 | `amdxdna/native_linux_kmq.cc` | `-452,10 +475,12` | iree_status_t iree_hal_amdxdna_native_co | +2/-0 | 디버그 | 전부 | int64_t tc0 = iree_time_now(); |
| Pp50 | `amdxdna/native_linux_kmq.cc` | `-465,7 +490,9` | iree_status_t iree_hal_amdxdna_native_co | +2/-0 | 디버그 | 전부 | int64_t td0 = iree_time_now(); |
| Pp51 | `amdxdna/native_linux_kmq.cc` | `-577,11 +604,19` | iree_status_t iree_hal_amdxdna_native_qu | +8/-0 | 기능+디버그 | 전부 | int64_t t0 = iree_time_now(); |

### 3-2. IREE 서브모듈 (57덩어리)

| ID | 파일 | 9/28 hunk | 함수 | +/− | 분류 | 현존 | 첫 추가 줄 |
|---|---|---|---|---|---|---|---|
| Pi01 | `Pipelines/Pipelines.cpp` | `-413,6 +413,9` | void buildIREEVMTransformPassPipeline( | +2/-0 | 기능 | 전부 | if (hooks.pipelineExtensions) { |
| Pi02 | `PluginAPI/Client.h` | `-88,6 +88,12` | public: | +5/-0 | 기능 | 전부 | // Adds passes to the \|buildStreamTransformPassPipeline\| p |
| Pi03 | `PluginAPI/PluginManager.h` | `-140,6 +140,12` | public: | +3/-0 | 기능 | 전부 | void extendStreamTransformPassPipeline(OpPassManager &passMa |
| Pi04 | `hal/allocator_heap.c` | `-275,11 +275,14` | static iree_status_t iree_hal_heap_alloc | +7/-4 | 기능 | 전부 | // BRIDGE: never request DISCARD when exporting: the content |
| Pi05 | `hal/buffer.c` | `-5,7 +5,7` |  | +1/-1 | 기능 | 전부 | #include <stdio.h> |
| Pi06 | `hal/buffer.c` | `-275,6 +275,10` | IREE_API_EXPORT iree_status_t iree_hal_b | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] mem_type actual=0x%x expected=0x%x\n" |
| Pi07 | `hal/buffer.c` | `-339,6 +343,9` | iree_hal_buffer_validate_usage(iree_hal_ | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] usage allowed=0x%x required=0x%x\n", |
| Pi08 | `hal/buffer.c` | `-878,6 +885,12` | IREE_API_EXPORT iree_status_t iree_hal_b | +6/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] map_persistent buf=%p dev=%p type=0x% |
| Pi09 | `hal/command_buffer_validation.c` | `-118,6 +118,16` | static iree_status_t iree_hal_command_bu | +9/-0 | 기능+디버그 | 전부 | if (!iree_all_bits_set(iree_hal_buffer_allowed_usage(binding |
| Pi10 | `vulkan/base_buffer.c` | `-137,10 +137,24` | static void iree_hal_vulkan_populate_dis | +15/-1 | 기능 | 전부 | // TEMP: experiment. Prefer host-visible device-local memory |
| Pi11 | `vulkan/base_buffer.c` | `-296,6 +310,13` | iree_status_t iree_hal_vulkan_populate_m | +6/-0 | 기능+디버그 | 전부 | fprintf(stderr,                                    // ← 여기 |
| Pi12 | `vulkan/completion_watcher.cc` | `-340,6 +340,7` | static int iree_hal_vulkan_completion_wa | +1/-0 | 디버그 | 전부 | fprintf(stderr, "[BR] vk complete sem=%p value=%llu t=%.3fms |
| Pi13 | `vulkan/direct_command_buffer.cc` | `-1,3 +1,4` |  | +1/-0 | 기능 | 전부 | #include <stdio.h> |
| Pi14 | `vulkan/direct_command_buffer.cc` | `-782,15 +783,176` | static iree_status_t iree_hal_vulkan_dir | +129/-0 | 기능+디버그 | 108/129줄 | iree_hal_buffer_ref_t g_last_bindings[8]; |
| Pi15 | `vulkan/direct_command_queue.cc` | `-5,6 +5,10` |  | +3/-0 | 기능+디버그 | 전부 | extern iree_hal_buffer_ref_t g_last_bindings[8]; |
| Pi16 | `vulkan/direct_command_queue.cc` | `-28,7 +32,8` | DirectCommandQueue::~DirectCommandQueue( | +2/-1 | 기능 | 전부 | VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena*  |
| Pi17 | `vulkan/direct_command_queue.cc` | `-41,22 +46,47` | iree_status_t DirectCommandQueue::Transl | +30/-7 | 기능+디버그 | 전부 | iree_host_size_t wait_count = 0; |
| Pi18 | `vulkan/direct_command_queue.cc` | `-68,24 +98,24` | iree_status_t DirectCommandQueue::Transl | +4/-4 | 기능 | 전부 | static_cast<uint32_t>(wait_count); |
| Pi19 | `vulkan/direct_command_queue.cc` | `-103,17 +133,66` | iree_status_t DirectCommandQueue::Submit | +42/-2 | 기능+디버그 | 전부 | bool any_foreign_signal = false; |
| Pi20 | `vulkan/direct_command_queue.h` | `-35,7 +35,8` | class DirectCommandQueue final : public  | +2/-1 | 기능 | 전부 | VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena*  |
| Pi21 | `vulkan/dynamic_symbol_tables.h` | `-243,7 +243,7` | namespace vulkan { | +1/-1 | 기능 | 전부 | DEV_PFN(OPTIONAL, vkGetMemoryFdPropertiesKHR)                |
| Pi22 | `vulkan/extensibility_util.cc` | `-213,6 +213,16` | iree_hal_vulkan_populate_enabled_device_ | +6/-0 | 기능 | 전부 | } else if (strcmp(extension_name, |
| Pi23 | `vulkan/extensibility_util.h` | `-84,6 +84,13` | typedef struct iree_hal_vulkan_device_ex | +4/-0 | 기능 | 전부 | // VK_KHR_external_memory_fd is enabled. |
| Pi24 | `vulkan/native_allocator.cc` | `-1,3 +1,5` |  | +2/-0 | 기능+디버그 | 전부 | #include <execinfo.h> |
| Pi25 | `vulkan/native_allocator.cc` | `-198,7 +200,11` | iree_hal_vulkan_native_allocator_query_b | +5/-1 | 기능 | 전부 | // BRIDGE: on UMA parts (890M) every allocation lands in hos |
| Pi26 | `vulkan/native_allocator.cc` | `-288,6 +294,30` | static iree_status_t iree_hal_vulkan_nat | +19/-0 | 기능+디버그 | 전부 | // BRIDGE: report the memory type the buffer actually landed |
| Pi27 | `vulkan/native_allocator.cc` | `-295,7 +325,7` | static iree_status_t iree_hal_vulkan_nat | +1/-1 | 기능 | 전부 | placement, actual_type, params->access, actual_usage, alloca |
| Pi28 | `vulkan/native_allocator.cc` | `-333,7 +363,7` | static iree_status_t iree_hal_vulkan_nat | +1/-1 | 기능 | 전부 | placement, actual_type, params->access, actual_usage, alloca |
| Pi29 | `vulkan/native_allocator.cc` | `-586,7 +616,14` | static VkDeviceSize iree_hal_vulkan_nati | +5/-1 | 기능+디버그 | 전부 | // [BR] Read the pointer value as a pointer, then convert. T |
| Pi30 | `vulkan/native_allocator.cc` | `-626,7 +663,7` | static iree_status_t iree_hal_vulkan_nat | +1/-1 | 기능 | 전부 | #if 0 && defined(IREE_PLATFORM_LINUX) |
| Pi31 | `vulkan/native_allocator.cc` | `-723,6 +760,10` | static iree_status_t iree_hal_vulkan_nat | +0/-0 | 빈 줄만 | - |  |
| Pi32 | `vulkan/native_allocator.cc` | `-790,6 +831,83` | static void iree_hal_vulkan_native_alloc | +60/-0 | 기능 | 전부 | static iree_status_t iree_hal_vulkan_native_allocator_import |
| Pi33 | `vulkan/native_allocator.cc` | `-865,6 +983,14` | static iree_status_t iree_hal_vulkan_nat | +4/-0 | 기능 | 전부 | case IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD: |
| Pi34 | `vulkan/native_buffer.cc` | `-112,6 +112,13` | static iree_status_t iree_hal_vulkan_nat | +7/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] vkmap buf=%p dev=%p type=0x%x usage=0 |
| Pi35 | `vulkan/native_semaphore.cc` | `-7,7 +7,7` |  | +1/-1 | 기능 | 전부 | #include <cstdio> |
| Pi36 | `vulkan/native_semaphore.cc` | `-30,6 +30,10` | extern const iree_hal_semaphore_vtable_t | +4/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] semaphore_cast: value=%p vtable=%p ex |
| Pi37 | `vulkan/vulkan_device.cc` | `-6,6 +6,9` |  | +1/-0 | 기능 | 전부 | #include <stdio.h> |
| Pi38 | `vulkan/vulkan_device.cc` | `-227,6 +230,18` | IREE_API_EXPORT iree_status_t iree_hal_v | +9/-0 | 기능 | 전부 | // VK_KHR_external_memory_fd: |
| Pi39 | `vulkan/vulkan_device.cc` | `-1046,6 +1061,9` | static iree_status_t iree_hal_vulkan_que | +0/-0 | 빈 줄만 | - |  |
| Pi40 | `vulkan/vulkan_device.cc` | `-1146,6 +1164,12` | iree_status_t iree_hal_vulkan_device_cre | +5/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] vk ext: memory_fd=%d dma_buf=%d host= |
| Pi41 | `vulkan/vulkan_device.cc` | `-1912,7 +1936,28` | static iree_status_t iree_hal_vulkan_dev | +19/-1 | 기능 | 전부 | // BRIDGE: only native semaphores can be watched; foreign on |
| Pi42 | `vulkan/vulkan_device.cc` | `-1921,7 +1966,7` | static iree_status_t iree_hal_vulkan_dev | +1/-1 | 기능 | 전부 | device->completion_watcher, &native_signal_list, resource_se |
| Pi43 | `hal/module.c` | `-351,6 +351,11` | IREE_VM_ABI_EXPORT(iree_hal_module_alloc | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] alloc_select: no policy, pairs=%d ->  |
| Pi44 | `hal/module.c` | `-368,12 +373,21` | IREE_VM_ABI_EXPORT(iree_hal_module_alloc | +8/-1 | 기능+디버그 | 전부 | for (iree_host_size_t i = 0; i < pair_count; ++i) { |
| Pi45 | `hal/module.c` | `-1661,6 +1675,12` | IREE_VM_ABI_EXPORT(iree_hal_module_fence | +4/-0 | 기능+디버그 | 전부 | iree_string_view_t dev_id = iree_hal_device_id(device); |
| Pi46 | `tooling/context_util.c` | `-6,8 +6,11` |  | +2/-0 | 기능 | 전부 | iree_hal_device_group_t* iree_tooling_experimental_device_gr |
| Pi47 | `tooling/context_util.c` | `-185,6 +188,8` | static iree_status_t iree_hal_module_dev | +0/-0 | 빈 줄만 | - |  |
| Pi48 | `tooling/context_util.c` | `-463,6 +468,9` | static iree_status_t iree_tooling_load_h | +3/-0 | 기능 | 전부 | // EXPERIMENT: expose the group so run_module can place inpu |
| Pi49 | `tooling/context_util.h` | `-106,6 +106,9` | iree_status_t iree_tooling_create_instan | +2/-0 | 기능 | 전부 | // EXPERIMENT: device group created from --device flags (NUL |
| Pi50 | `tooling/function_util.c` | `-5,7 +5,7` |  | +1/-1 | 기능 | 전부 | #include <stdio.h> |
| Pi51 | `tooling/function_util.c` | `-16,6 +16,8` | iree_status_t iree_tooling_append_async_ | +2/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] abi.model='%.*s'\n", (int)model.size, |
| Pi52 | `tooling/function_util.c` | `-42,6 +44,9` | iree_status_t iree_tooling_append_async_ | +3/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] async_fence sem=%p dev=%p\n", |
| Pi53 | `tooling/function_util.c` | `-94,6 +99,7` | static iree_status_t iree_tooling_setup_ | +0/-0 | 빈 줄만 | - |  |
| Pi54 | `tooling/function_util.c` | `-128,6 +134,8` | static iree_status_t iree_tooling_submit | +2/-0 | 기능+디버그 | 전부 | fprintf(stderr, "[DBG] transfer_fence sem=%p dev=%p\n", |
| Pi55 | `tooling/run_module.c` | `-6,6 +6,9` |  | +2/-0 | 기능 | 전부 | #include <stdlib.h> |
| Pi56 | `tooling/run_module.c` | `-45,6 +48,12` | IREE_FLAG_LIST( | +5/-0 | 기능 | 전부 | IREE_FLAG_LIST( |
| Pi57 | `tooling/run_module.c` | `-237,6 +246,77` | static iree_status_t iree_tooling_run_fu | +59/-0 | 기능+디버그 | 53/59줄 | // EXPERIMENT: relocate inputs to per-input devices. |

현존이 "전부"가 아닌 덩어리(우리가 덮어쓴 것, 확인함):
- Vulkan `direct_command_buffer.cc` dispatch BRIDGE(Pi14, 108/129줄): 9/28 챕터11이 본문을 `resolve_buffer_ref`로 옮기며 실패 시 `continue`(조용한 실패)를 에러 반환으로 바꿈(I07, I17).
- `AMDAIEDevicePlacementUtils.cpp` `linkFlags`(Pp08, 44/53줄): 10/2 세션1 챕터00이 GPU→CPU 주석의 옛 TODO를 교체(O17).
- `run_module.c` `--input_device`(Pi57, 53/59줄): 9/29 세션2 챕터02가 reflection 경로를 넣으며 반복문·로그를 바꿈(I30~I32).

## 4. 디버그·임시 코드 (현재 작업 트리)

| 묶음 | 위치 | 출처 | 기본 동작 | 의존하는 곳 | 비고 |
|---|---|---|---|---|---|
| D1 `[T]` 타이밍 로그 30줄 | amdxdna `allocator.cc`, `buffer.cc`, `device.cc`, `direct_command_buffer.cc`, `native_linux_kmq.cc` | 전임자 | 항상 stderr | 없음(확인함: 회귀 스크립트 grep 0) | 9/28 01 3-4 B **통합 후**: 제거(`06aa6260`) |
| D2 `[DBG]` 로그 21줄 | amdxdna 위 파일들, IREE `buffer.c`, `command_buffer_validation.c`, vulkan `base_buffer.c`·`native_buffer.cc`·`native_semaphore.cc`·`vulkan_device.cc`, `modules/hal/module.c`, `tooling/function_util.c` | 전임자 | 항상 stderr | **회귀 스크립트가 `module.c`의 `[DBG] alloc_select` 로그로 lead 선택을 판정**(`docs/regression/run_regression.sh:154`) | 지우면 회귀 판정이 깨진다 **통합 후**: 제거(`06aa6260`, fork `effaf17b4`). 회귀는 이제 로그를 읽지 않음 |
| D3 `[BR]` 로그 19줄 | vulkan `direct_command_buffer.cc`(BRIDGE import 결과 5, dispatch 바인딩 4), `direct_command_queue.cc` 3, `completion_watcher.cc` 1, `native_allocator.cc` 1(+주석 1), amdxdna `direct_command_buffer.cc` 1, `run_module.c` 2 | 전임자 15 + 우리 4(I07의 3줄은 전임자 로그를 옮긴 것, I16 1줄, I31 1줄, I32는 전임자 줄 수정) | 항상 stderr. dispatch 바인딩 값 덤프(map)는 `IREE_VULKAN_BR_DUMP=1`일 때만(I15) | 회귀 스크립트가 `[BR] input[i]`(placed 열, `:174`)와 `[BR] amdxdna copy`(xdna_copy 지표, `:283`)를 읽음 | 지우면 회귀 지표가 깨진다 **통합 후**: 제거(fork `effaf17b4`). 회귀는 이제 로그를 읽지 않음 |
| D4 `g_last_bindings[8]` 전역 | vulkan `direct_command_buffer.cc`, `direct_command_queue.cc` extern | 전임자 | 쓰기만, 읽는 곳 없음(9/28 확인), 스레드 안전 아님 | 없음 | 잔재 **통합 후**: 제거(fork `14973745c`) |
| D5 `#include <execinfo.h>` | vulkan `native_allocator.cc:1` | 전임자 | 쓰는 곳 없음 | 없음 | 잔재 **통합 후**: 제거(fork `14973745c`) |
| D6 `TEMP: experiment` | 플러그인 `AMDAIEDevicePlacementUtils.cpp:24`(vulkan-spirv → Accelerator, **HEAD에 이미 커밋됨**), vulkan `base_buffer.c:140`(HOST_VISIBLE 우선) | 전임자 | **동작** | GPU 배정 전제, CPU→GPU HOST_VISIBLE 표의 전제 | 9/28 01 3-4 A **통합 후**: 플러그인은 정식 Accelerator(`e7060e40`), `base_buffer.c`는 upstream 복원(fork `b006be8fc`) |
| D7 `#if 0 &&` msync 검사 끔 | vulkan `native_allocator.cc:666` | 전임자 | **동작**(host ptr import 전 검사 생략) | GPU→CPU host ptr import | 9/28 01 3-5 부채 **통합 후**: 유지 + 이유 주석(fork `b006be8fc`) |
| D8 무조건 `MAPPING` | vulkan `native_allocator.cc:203-207` | 전임자 | **동작** | CPU가 GPU 버퍼를 host-map | UMA 전제, 외장 GPU에선 성립 안 함 **통합 후**: 대체: `MAPPING \| MAPPING_PERSISTENT`(fork `8122a888b`) |
| D9 환경 변수 스위치 | `IREE_AMDXDNA_TEST_EXPORT`(allocator.cc:168, 테스트 훅), `IREE_AMDXDNA_CACHE_HWCTX`(direct_command_buffer.cc:1008, hwctx 캐시 opt-in), `IREE_VULKAN_BR_DUMP`(vulkan direct_command_buffer.cc:1046, 우리) | 전임자 2 + 우리 1 | 기본 꺼짐 | 없음 | — **통합 후**: 제거(`06aa6260`, fork `effaf17b4`) |
| D10 `[amdaie placement-map]` stderr 요약 | `AMDAIEAssignDeviceAffinities.cpp:558-566` | 우리(10/2) | **`--iree-amdaie-placement-map`을 줄 때만** | 실험 기록 | 실험 옵션의 출력이라 디버그로 분류하지 않음(챕터 04에서 판단) **통합 후**: 유지 |
| D11 `EXPERIMENT` 표식 5곳 | `run_module.c`(`--input_device`), `context_util.c/.h`(device group 전역) | 전임자 | **동작** | 회귀 수동 모드, 입력 자동 배치(S5) | 9/28 01 3-4 A **통합 후**: 유지 |

## 5. 코드가 아닌 우회 (배정 맵, 플래그, IR 편집, 설정 주입, 배정으로 피하기)

전체 목록(32개, 근거 줄 포함)은 `docs/2026-10-20/1_변경장부_명세_커밋준비/01_변경_장부/우회_목록.md`. 아래는 요약이다. "CLAUDE.md" 열은 **2026-10-20 수정 후 줄 번호**(1-3 행, 3-2 문단)다. 상태: **씀** = 지금 실험·데모에 씀, **기본** = 회귀·F3 등 기본 경로가 의존, 안 씀 = 기록만.

| ID | 우회 | 종류 | 피하는 것 | 원인 규명 | 없앨 계획 | 상태 | CLAUDE.md(1-3·3-2) |
|---|---|---|---|---|---|---|---|
| W01 | 맵 NPU1/GPU15/CPU43 (`docs/2026-10-02/2_…/03_…/*.map`) | 맵 | NPU f32 코드젠 실패 | 표면만 | 3-2 없앨 계획 1~3 | 안 씀(FC GPU 2배) → 통합 후: 커밋 안 함(맵은 실험 도구) | L305, L311-313 |
| W02 | 맵 t4 1/12/46 (`docs/2026-10-02/3_…/01b_…/map_t4_fc_cpu.txt`) | 맵 | NPU f32 실패 + GPU M=1 2배 | GPU 쪽 규명 | 3-2 L311-313, L325 | 데모 → 통합 후: 커밋 안 함 | L321, L325 |
| W03 | 맵 NPU10/GPU3 | 맵 | — | — | — | 안 씀(실행 실패) → 통합 후: 커밋 안 함 | L321 |
| W04 | **맵 NPU9/GPU4/CPU46** (`docs/2026-10-18/2_…/02c_…/…npu9_gpu4.map`) | 맵 | d6 반복 392, conv5 ②, FC ③, GPU 2배 | 반복 392 이유는 규명, **멈춤 기전 미확인**, ②③ 설계 수준 미규명 | L311·L312, L325 | **씀**(10/20 세마포어 시험, 커밋 후 시험 목록) → 통합 후: 커밋 안 함 | L152, L154, L321 |
| W05·W06 | 맵 NPU13·NPU16 (`docs/2026-10-18/4_…/03_…/*.map`) | 맵 | 편집 IR(W22·W23)용 | W22와 같음 | L311-313 | 데모 → 통합 후: 커밋 안 함 | L321 |
| W07 | NPU가 f32로 못 하는 dispatch를 배정으로 피함 | 배정 | ①②③ + DMA 반복 | 표면만 → **챕터 03** | L311-313 | 씀 → 통합 후: dev 수정으로 대체(`b492865a`, `990bcab2`) | L143, L275, L154 |
| W08 | FC(M=1)를 GPU에 안 보냄 | 배정 | SPIR-V SubgroupReduce subgroup 크기 미기록 | **규명** | L325(서브모듈 수정, 일정 미정) | 씀 → 통합 후: **해결**(fork `f1a30ce2c`) | L325 |
| W09 | d6(conv1_2)를 GPU로 | 배정 | 제어 코드 반복 392 > 256 | 392 이유 규명, 멈춤 기전 미확인 | L312 | 씀 → 통합 후: 커밋 안 함 | L147, L152 |
| W10 | 반복 ≤ 256 규칙 | 배정(규칙) | 256 초과 펼침 멈춤 | **미확인**(fc6 782 반례) | L312, L152 "추가 조사" | 씀 → 통합 후: 커밋 안 함 | L152, L321 |
| W11 | NPU 3입력 dispatch를 라운드로빈 순서로 피함(전임자) | 배정 | `no producer DMA channel available` | **미규명** | **없음**("고치려 하지 말 것") | **기본**(회귀 기준선) → 통합 후: 유지(원인 미규명) | L244, L296 |
| W12 | lead 기본값(0) 유지 → CPU↔GPU 동률은 GPU 소유 | 런타임 선택 | BRIDGE 기록 시점 transient export | 코드 경로 규명 | **없음**(수정 방향만) | **기본** → 통합 후: **해결**(GPU→CPU false, `e3d86485`) | L324, L151 질문 3 |
| W13 | `-unroll-threshold=0` | 플래그 | ① soft-float 펼침 코드 16 KB 초과 | 규명 | L313(벡터화 후 뺌) | 씀 → 통합 후: dev `-unroll-full-max-count=0`으로 대체 | L146, L314-319 |
| W14 | dedup off | 플래그 | 정적 offset만 지원하는 `address_patch` | 규명 | **없음**(팀 백엔드 제약) | 씀(VGG) | L298 |
| W15 | no-fuse-into-contraction | 플래그 | 융합 contraction tile 불가(팀 기록) | 팀 기록뿐 | **없음** | 씀(VGG) | L298 |
| W16 | vectorization off | 플래그 | aievec bf16/f32 미지원 | 규명(10/19 세션1) | **명시 없음**(벡터화 작업 연계 추정) | 씀(VGG) | L298 |
| W17 | im2col·channels-last·detach | 플래그(레시피) | NPU conv 경로 | 팀 기록 | 없음(레시피) | 씀(VGG) | L298 |
| W18 | `--device_lead_allocator=2` | 플래그 | Vulkan map SIGSEGV | 근본 수정됨 | — | 안 씀 | L324 |
| W19 | `--input_device` 수동 | 플래그 | 자동 배치 전 수단 | — | 없음(비교용) | **기본**(회귀 수동 모드) | L70 |
| W20 | `--iree-hal-external-resources-mappable=true` | 플래그(F3) | **docs에 이유 없음** | **미규명** | 없음 | **기본** → 통합 후: **제거**(불필요, `e3d86485`) | L57-65 |
| W21 | `--iree-amdaie-stack-size=2048` | 플래그(F3) | stack 부족(일반론) | 3장치 필요성 미확인 | 없음 | **기본** → 통합 후: **제거**(불필요, `50f8aaa9`) | L57-65 |
| W22 | conv5 M 196→224 패딩 IR 편집 | IR 편집 | ② 팀 pad가 f32 건너뜀 | 표면만 → **챕터 03** | L311(1) | 데모(vmfb만) → 통합 후: dev 수정으로 대체(`b492865a`) | L145, L154 |
| W23 | FC M 1→32 `linalg.generic` 패딩, fc6 N 512×8 분할, fc8 N 1000→1024 | IR 편집 | ③ M=1 pack, N 1000 `ConvertToDma` 실패, 팀 N-split f32 건너뜀 | 표면만 | L311(1) | 데모 → 통합 후: dev 수정으로 대체, VGG f32 e2e로 확인 | L145, L154 |
| W24 | FC 패딩 v1(`tensor.pad`) | IR 편집 | — | 0 splat CSE 실패(규명) | — | 폐기 | — |
| W25·W26 | 단독 손 패딩 모델 8개, conv1_2 M 2분할 | IR 편집 | 검증용 | — | — | 안 씀 | L321 |
| W27 | conv1_2 1단계 타일 M 256 executable 치환(`--iree-hal-substitute-executable-source`) | 설정 주입 | 반복 392 | 392 이유 규명, 멈춤 미확인 | L312(타일 ≤ 256) | 데모 → 통합 후: dev 수정으로 대체(`990bcab2`) | L147, L152 |
| W28 | `REG_OUT=/home/mkchoi/iree-work/regression/…` | 환경 변수 | `REG_ROOT` 기본 `/tmp` | 규명 | 제안만(미승인) | **기본** → 통합 후: `REG_ROOT` 기본 `build/regression`으로 대체(`a3b2c714`) | L55 |
| W29 | `RADV_PERFTEST=cswave32` | 환경 변수 | GPU 2배 | 효과 없음 | — | 안 씀 → 통합 후: 안 씀(W08 해결) | L325 |
| W30 | `IREE_VULKAN_BR_DUMP=1` | 환경 변수(우리 코드) | 디버그 map 충돌 | 근본 수정됨 | 챕터 04 "빼야 할 것" | 기본 꺼짐 → 통합 후: **제거**(fork `effaf17b4`) | L250, L323 |
| W31·W32 | `IREE_AMDXDNA_CACHE_HWCTX`, `IREE_AMDXDNA_TEST_EXPORT` | 환경 변수(전임자) | hwctx 캐시 opt-in, export 테스트 | 전임자 주석뿐 | 없음 | 안 씀 → 통합 후: **제거**(`06aa6260`) | — |

코드이지만 우회 수단인 것: K1 ukernel `set_rnd`(O04, 팀 반영 전, 규민 상의), K2 `--iree-amdaie-placement-map`(W01~W06의 통로, S7), K3 `--iree-amdaie-respect-affinity-marks`(디버그, S6), K4 `accelIdx = 0`(전임자, W11의 바탕). **통합 후**: K1은 dev `cff1e56e`, K2는 실험 옵션으로 유지, K3는 삭제(`e7060e40`), K4는 유지(round-robin 시작점).

- **원인 미규명 우회**: W10(DMA 멈춤 기전) / W07·W22·W23(f32 패딩 설계 수준·배수 → 챕터 03에서 조사) / W23의 N 1000 실패 / W11 NPU 3입력 / W20 mappable / W21 stack-size / W15 no-fuse
- **없앨 계획이 없는 우회**: W11, W12, W14, W15, W16, W17, W19, W20, W21, W31·W32, K4
- **통합 후**: 위 두 목록에서 W08·W12는 해결, W20·W21·W31·W32는 제거, W07·W22·W23은 dev 수정으로 대체됐다. 남은 원인 미규명 우회는 W10(맵을 안 쓰므로 지금은 해당 없음), W11, W15다.

## 6. 커밋 계획 대응 (2026-10-20 세션 1 챕터 04, 승인 대기)

| 커밋 | 장부 ID |
|---|---|
| K00a 인수인계 기준선(컴파일러) | Pp01~08, Pi01~03, N02 |
| K00b 인수인계 기준선(런타임) | Pp12~51 중 기능 덩어리, Pi04~57(U3 = Pi25 제외안), 로그 전용 덩어리 처리는 계획 3-5 |
| K01 S1 | I01~I04 | 
| K02 S3-D / K04 S3-A | I06, I08, I09, I17(일부), I20, I25~I27 / I05, I07, I10~I14, I17 |
| K03 S2 / K05 S2 test | O18~O23 / O24 |
| K06 S5 | O01a, O11~O13, O15, O16, N01, N03, I28~I32 |
| K07 S6 / K09 S7 | O01b~O03, O06~O08, O10, O13, O14, N04 / O05~O09, N05 |
| K08 S8 주석 | O17 |
| K10 / K11 S4 | I15, I16 / I18, I19, I21~I24 |
| K12 S10(보류 추천) | O04 → dev `cff1e56e`로 대체, S10 삭제 |
| 제외 | N06(lock 스크립트), N07(덤프), N08(빈 파일 6개), Pp09~11(모델, 추천), Pi25(U3, 추천) |

## 7. 갱신 이력

| 날짜 | 세션 | 바뀐 것 |
|---|---|---|
| 2026-10-20 | 세션 1 챕터 01 | 첫 판(9/28 기준선 대비 우리 56덩어리, 전임자 108덩어리, 새 파일 9묶음, 디버그 11묶음, 우회 목록) |
| 2026-10-20 | 세션 1 CP1 후·챕터 04 | 소스 변경 없음. 6절 커밋 계획 대응 추가. 명세 알려진 한계(S7 L1, S3·S9 L-fd) 추가. U1~U4 명세(`04_커밋_계획/01_…`) |
| 2026-10-25 | dev 통합 S9 | "dev 통합 후 상태" 절 추가, 명세 경로를 `docs/specs/`로, 4·5·6절 상태 칸과 N04·N07·N09에 통합 결과를 덧붙임 |
| 2026-10-28 | dev-POOL 통합 P9 | "dev-POOL 통합 후 상태" 절 추가, ukernel upstream 이전과 unroll 규칙 반영 |
