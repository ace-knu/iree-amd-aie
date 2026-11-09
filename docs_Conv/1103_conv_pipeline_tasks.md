# Convolution NPU 파이프라인 — 현황 분석과 작업 정리

> 작성 2026-11-03 · 기준 브랜치 `conv-npu`(= origin/dev `786d1d19`, dev-POOL `9c6ea263` 포함) · 타겟 **npu4** (XDNA2, 8 열 × 4 행 = 32 코어)
>
> 요청: "현재 IREE-AMD-AIE 환경에서 Convolution 에 대한 NPU 파이프라인을 구축하려고 해. 필요한 작업들을 전체 정리해 봐. 구현이 안 된 부분이 주가 될 것 같은데, 부족한 부분·해야 하는 부분을 전반적으로 정리해 줘."
>
> 검증 정책: 컴파일 · 정적 IR · lit/ctest 만. NPU 실행은 명시 승인 시에만 ([CLAUDE.md](../CLAUDE.md) §3).
> 조사 방법: 리포·브랜치·문서를 읽기 전용으로 분석(7 갈래 병렬) → 계획 3 안 작성 → 심사·통합 → 핵심 주장 14 건 검증 → 조립된 문서를 비평 3 건(정확성·누락·순서)으로 검토해 반영(§10.4–§10.5) → VGG 를 im2col 없이 Flow·전처리 끝까지 컴파일해 conv 형태 확인(§11). 본문에서 **미검증** 으로 표시한 항목은 컴파일이나 실행으로 확인되지 않은 것이다.
>
> **갱신(같은 날)**: origin/dev 가 `2c5763b8` → `786d1d19` 로 올라갔다(71 커밋: dev-POOL 머지 `7060d5f3`, consumer 별 lock `6ef3dc1f`, bias k+1 fold, mlir-aie aie_api 기반 ukernel, 버퍼 64 B 정렬 `33c20094` 등). 문서 전체를 그 기준으로 고쳤고, 바뀐 점은 §10.6 에 모았다. conv 브랜치 `conv-npu` 는 이 커밋에서 만들었다.

---

## 0. 결론 요약

1. **지금 NPU 에서 도는 conv 경로는 im2col 하나뿐이다.** `--iree-global-opt-use-im2col-for-convs` 로 conv 를 CPU gather + NPU matmul 로 바꾼다. dev-POOL 에서는 matmul 코어 코드가 **스칼라**(ukernel 없음, 벡터화 off, Peano v19)라 VGG-16 e2e 가 55 s 였다. origin/dev(`786d1d19`)는 같은 경로를 bf16 ukernel 로 돌려 **1.71 s**(3 회 중앙값, default 전원 모드, corr 0.99996, top-1 일치, NPU dispatch 16 개)이고, bias 는 K 에 접고 ReLU 는 NPU dispatch 안에서 끝낸다.
2. **origin/dev 가 dev-POOL 을 포함한다.** 곽준호 님이 `7060d5f3`(pool-integration)로 dev-POOL 을 머지했고(이 문서가 예측한 5 파일 충돌, dev 우선 해소), 그 위에 GEMV matvec ukernel, bias k+1 fold, consumer 별 lock 쌍(`6ef3dc1f`), mlir-aie aie_api 기반 ukernel, 버퍼 정렬(`33c20094`)이 들어왔다. conv 브랜치 `conv-npu` 는 이 커밋에서 만들었다. 김규민 님의 aievec bf16 벡터화(`origin/aie2p-bf16-vectorized`, 약 2.1–3.1 s, Peano v19)는 여전히 dev 밖이지만 origin/dev 에 텍스트 충돌 없이 적용된다.
3. **네이티브 conv(conv-decompose) 는 origin/dev 에서도 업스트림 상태 그대로다**: 1 열 4 코어, IC 전체를 L1 에 올려 IC ≥ 256 인 8 층이 L1 초과(depth 1 로 줄여도 IC = 512 인 5 층), bf16 벡터화 불가, conv ukernel 없음, conv+matmul 혼합 모듈 라우팅 불가, npu4 테스트 0 건. VGG conv 13 개는 batch 가 접힌 6-loop generic 으로 도착해 **conv 로 인식조차 되지 않는다**(Flow 컴파일로 확인). `dev-conv` 의 24 커밋(라우팅·6-D 인식·Flow 패딩·BD 수정)은 dev 에 하나도 없다. origin/dev 의 bias fold 는 `linalg.matmul` 만 접으므로, im2col 을 빼면 네이티브 conv dispatch 는 bias 를 세 번째 입력으로 가진 채 남는다(코드 추론, 미검증).
4. **따라서 작업은 세 층으로 나뉜다**: (A) 기준선·측정 도구 — origin/dev 에서 분기해 받았다(55 s → 1.71 s, `scripts/onnx`·`build-trace`). (B) 네이티브 conv 기반(라우팅·인식·패딩·**conv bias fold**·BD·테스트)과 L1 IC 타일링. (C) 전 배열(8 열) 분배 + bf16 벡터 커널/ukernel + 코어 내 ReLU epilogue. e2e 바닥은 **hwctx 생성·해제 약 1.2 s(NPU dispatch 16 개)** 라 conv 자체 성능은 **dispatch 별 NPU 시간과 DDR 바이트**로 평가해야 한다.
5. **네이티브 conv 의 권고 설계는 OC 를 열, OH halo 청크를 행에 두는 격자(G2)다.** 오늘 im2col matmul 이 32 코어에서 쓰는 통신 패턴(행 broadcast · 열 broadcast · 열별 4-producer join)과 같아, 연속 슬라이스에서만 동작하는 consumer 별 lock(`6ef3dc1f`)에 기대지 않는다. 예비 산술(§5.0, 실측값을 재현하는 L1 식)로 오늘 하강(모든 L1 버퍼 depth 2)에서도 13 층 모두 들어가는 타일이 있고, 바이트당 MAC 은 26–37 로 지금 conv-decompose 타일(2.8)의 약 9–13 배다. 코어 내 epilogue 를 넣으면 누산기 depth 1 이 필수다. 아직 반증되지 않은 가정이 셋 있다: DDR 연속 구간(IC 청크 그대로면 16–32 B), **64 B 벡터 정렬**(AIE2P 의 512-bit 접근은 64 B 정렬이 아니면 오류 없이 다른 주소를 읽는다 — `33c20094` 실측. kw 시프트와 행 간격이 이를 깬다), 열 분배·채널 예산(6/6). 그래서 재빌드 없는 스파이크로 이 가정들부터 확인한다.
6. **목표 수치를 먼저 정해야 한다.** 팀은 CPU 단독 실행(f32, HX 370 호스트) **0.34 s** 를 NPU 최적화의 목표로 적어 두었다. origin/dev 의 NPU 연산 합(0.120 s)도 CPU 단독의 dispatch 합(0.085 s)보다 길고, 배열 피크 대비 효율은 0.4 % 수준이다. e2e 는 hwctx 재사용(런타임, 이 계획 밖)이 먼저이고, conv 파이프라인의 몫은 NPU 연산 효율이다(D4).
7. 업스트림(nod-ai/iree-amd-aie)은 fork 지점(2026-06-03) 이후 사람 커밋이 없어 **끌어올 conv 작업이 없다.** 다만 IREE 서브모듈에는 conv 를 dispatch 안에서 `im2col` op + matmul 로 바꾸는 implicit GEMM 경로(`iree-codegen-convolution-to-igemm`)가 있고 지금은 GPU 백엔드만 쓴다. pack-peel + bf16 ukernel + 융합 ReLU 는 origin/dev 에서 하드웨어로 검증된 조합이라, 이를 재사용하는 대안으로 스파이크(T2.6)해 G2 와 비교한다(D13).
8. **작업 순서**: (완료) origin/dev 에서 `conv-npu` 분기와 실행 환경 구성 → 재빌드 없는 반증 스파이크 → 빌드 락 커밋(dev 의 build.sh 에는 아직 없다) → 단일 형상 수직 슬라이스 → conv bias fold 를 포함한 일반화 → 전 배열·커널·epilogue. 남은 작업 37 개(전체 42 개 중 완료·생략 5), 열린 소유자 결정 10 건(§5–§7).

---

## 1. 범위와 환경

| 항목 | 값 |
|---|---|
| 호스트 | ASUS Zenbook S16 UM5606GA, Ryzen AI 9 HX 470 (NPU "Gorgon Point 1" = npu4 급, 8 열 × 6 행, 코어 32, H clock 1800 MHz, 드라이버 계산 최대 58 TOPS) |
| 컴파일러 | 브랜치 `conv-npu` = origin/dev `786d1d19`(dev-POOL `9c6ea263` 포함, dev-POOL 보다 124 커밋 앞). 업스트림 fork 지점 `fddfec1b`(2026-06-03). `third_party/iree` `32dad766`(+ 미커밋 FormDispatchRegions.cpp 패치, 실험 전용), `third_party/aie_api` `b0422227`(새 서브모듈). Peano v22 |
| 대상 모델 | VGG-16 (ONNX → torch dialect: `onnx.Conv` 13 개, 3×3 / pads 1 / stride 1 / NCHW / bias 포함, ReLU 15, MaxPool 5, Gemm 3) |
| 대상 연산 | 3×3 stride-1 bf16(f32 → demote) conv, f32 누산. NCHW·stride·depthwise 는 범위 밖(§4 에서 거부 가드만) |
| 빌드 | 컨테이너 전용, `JOBS=4 ./scripts/build/build.sh --target ...`(빌드 락은 로컬 수정), `build/` + `build-trace/`. 모델 실행은 `scripts/onnx`([docs/EXPERIMENT_ENV.md](../docs/EXPERIMENT_ENV.md)) |
| 검증 | 컴파일 + `--mlir-print-ir-after=<pass>` + lit/ctest. NPU 실행은 건별 승인 |

### 1.1 VGG-16 conv 13 층 (224 입력, 3×3, pad 1)

| # | OH=OW | IC | OC | GMAC | 가중치 bf16 | 입력 act bf16 | im2col M | K=9·IC | N=OC | im2col 행렬 bf16 | L1 가중치 타일 (OC_t=8) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 224 | 3 | 64 | 0.087 | 3 KB | 294 KB | 50176 | 27 | 64 | 2.6 MB | 432 B |
| 2 | 224 | 64 | 64 | 1.850 | 72 KB | 6.1 MB | 50176 | 576 | 64 | 55.1 MB | 9,216 B |
| 3 | 112 | 64 | 128 | 0.925 | 144 KB | 1.5 MB | 12544 | 576 | 128 | 13.8 MB | 9,216 B |
| 4 | 112 | 128 | 128 | 1.850 | 288 KB | 3.1 MB | 12544 | 1152 | 128 | 27.6 MB | 18,432 B |
| 5 | 56 | 128 | 256 | 0.925 | 576 KB | 784 KB | 3136 | 1152 | 256 | 6.9 MB | 18,432 B |
| 6–7 | 56 | 256 | 256 | 1.850 ×2 | 1.1 MB | 1.5 MB | 3136 | 2304 | 256 | 13.8 MB ×2 | 36,864 B |
| 8 | 28 | 256 | 512 | 0.925 | 2.3 MB | 392 KB | 784 | 2304 | 512 | 3.4 MB | 36,864 B |
| 9–10 | 28 | 512 | 512 | 1.850 ×2 | 4.5 MB | 784 KB | 784 | 4608 | 512 | 6.9 MB ×2 | **73,728 B** |
| 11–13 | 14 | 512 | 512 | 0.462 ×3 | 4.5 MB | 196 KB | 196 | 4608 | 512 | 1.7 MB ×3 | **73,728 B** |
| 합계 | | | | **15.35 GMAC = 30.7 GFLOP** | 28.1 MiB | 17.3 MiB | | | | **155.9 MiB** | |

- 크기 단위 KB/MB 는 KiB/MiB 다.
- L1 가중치 타일 = KH·KW·IC·OC_t·2 B (npu4 bf16 명령 8×8×8 → OC_t = 8). 64 KiB 를 넘는 층이 5 개(9–13), IC=256 층도 이중 버퍼면 초과한다.
- 산술 하한: 배열 59 TOPS(1024 ops/cycle/core × 1.8 GHz × 32) 기준 30.7 GFLOP = **0.52 ms (100 %) / 1.7 ms (30 %) / 5.2 ms (10 %)**. 현재 55 s 는 피크의 1e-5 수준이다.
- NPU 가 DDR 에서 읽고 쓰는 양: im2col 경로 ≈ 235.6 MiB/추론(gather 출력 155.9 + 가중치 28.1 + f32 출력 51.7) vs 네이티브 ≈ 97.1 MiB(입력 17.3 + 가중치 28.1 + f32 출력 51.7). 측정된 shim 경로 대역폭 2.6 GB/s(풀링 1 dispatch 실측) 기준 95 ms vs 39 ms. 이 비교는 DMA 효율이 같다고 가정한다. 연속 구간이 짧으면(§5.0, G16) 네이티브 쪽 실효 대역폭이 떨어진다. 2.6 GB/s 는 run 뒤 모든 binding 을 sync 하던 옛 HAL 에서 잰 값이라 입력 sync 가 섞였을 수 있다(`3d454250` 이후 읽기 전용 binding 은 sync 하지 않는다). origin/dev 런타임으로 다시 잰다.

---

## 2. 현재 상태 (사실)

### 2.1 지금 NPU 에서 도는 conv = im2col 경로

플래그(origin/dev `scripts/onnx/options.py:60-126` 이 정본, `models/vgg16/README.md` 동일): 이종 타겟(`d0_local_host` = llvm-cpu `--iree-llvmcpu-target-cpu=host` 가 기본·lead 장치, `d1_amdxdna_npu4`), `--iree-global-opt-use-im2col-for-convs`, `iree-preprocessing-convert-conv-to-channels-last`, `--iree-amdaie-pooling-channels-last`(항상, `05c7f6a3`), dedup off(남은 유일한 fork 플래그), `--iree-amdaie-enable-ukernels=all`·`--iree-amdaie-enable-vectorization-passes=false`(스크립트 기본), `--numeric bf16` 이면 `--iree-amdaie-demote-contraction-inputs-to-bf16`(스크립트 기본은 f32). no-fuse·detach 플래그는 bias 를 K 에 접으면서 레시피에서 지웠다(`38613bbb`). pooling 은 기본 CPU, `--pool accel` 이면 `--iree-amdaie-accelerate-pooling`(옛 `--iree-amdaie-enable-pooling`, `f44d40e0`). dev-POOL 레시피(`log_shjo/poolpad/compile_vgg.sh`: no-fuse·detach·ukernel none·enable-pooling)는 이제 팀 레시피와 다르다.

층마다 생기는 dispatch(origin/dev 레시피). `conv-npu` 에서 VGG bf16 CPU+NPU 를 11-03 에 컴파일했고(112 s, NPU 실행 없음), vmfb 안 실행 파일의 entry point 이름으로 셌다. NPU 는 16 개(conv matmul 13 + FC 3)다. CPU 는 im2col gather 13, max pool 5, `slow_memcpy` 8, 입력 transpose 1, host pad 32, crop 9 다. 괄호는 dev-POOL 관찰이다.

| dispatch | 장치 | 내용 |
|---|---|---|
| `generic_HxWx3x3xIC_f32xbf16` | CPU | im2col gather + truncf (6-D all-parallel generic, 9× 확장) |
| `matmul_MxNxK_bf16xbf16xf32` (+ ReLU) | **NPU** | pack-peel 32 코어 / 8 memtile, bf16 ukernel. bias 는 K+1 번째 행으로 접히고(init = fill 0, `dd05ed3e`) ReLU 는 같은 dispatch 의 epilogue(dev-POOL 은 스칼라 코어 코드) |
| `pad_dispatch_*` ×2 | CPU | im2col LHS 에 ones 열(fold)과 K 배수 zero pad — LHS 를 층마다 두 번 더 복사(VGG bf16 CPU 약 18 ms, `CHANGES.md:91`). 가중치 pad 와 bias 행은 컴파일 때 새 global(`5769144f`). (dev-POOL 은 이 자리에 bias + ReLU f32 elementwise) |
| `slow_memcpy` | CPU | 다음 층의 SAME pad 버퍼에 복사(dev-POOL 관찰). origin/dev 에도 8 개 있고, dispatch 번호로 보면 다음 층이 conv 인 conv 8 곳 바로 뒤에 있다 |
| `pad_*` / `crop_*` | CPU | ukernel all 배수 M 64·N 128·K 32(`AMDAIEPadContractionDispatches.cpp:132-140`): conv1_x N 64→128(D3), conv4_x M 784→832, conv5_x M 196→256, 모든 층 K+1 → 다음 32 배수(예: 577→608). crop 9 개는 이 규칙의 예측(conv 8 층 + FC 출력 1000→1024)과 수가 맞는다. (dev-POOL ukernel none: K 27→32, M 784→800, 196→224, 낭비 2.0 %) |

- dev-POOL 이 스칼라였던 이유: `enableAMDAIEUkernels{"none"}`, 벡터화 off(aievec 의 AIE2P 지원은 i8 8×8×8 뿐이라 켜면 bf16 `vector.contract` 가 실패, [docs/0908_im2col_e2e.md](../docs/0908_im2col_e2e.md) §3), Peano v19. origin/dev 는 `scripts/onnx` 기본이 `--ukernel all` 이고 Peano v22 다.
- dev-POOL 실측: e2e **55.09 s**(R=3 중앙값, corr 1.00000, `log_shjo/poolpad/e2e_full.log`). dispatch 별 시간 측정 수단이 없었다.
- 팀 측정 사다리: `STAGES.md`(측정 호스트 HX 370) 74.5 s(Peano v19) → 8.8 s(Peano v22) → 2.29 s(ukernel all, 23 dispatch, hwctx 1.80 s). dev-POOL 머지 뒤(GEMV + matvec ukernel) 1.62 s(hwctx 약 1.2 s, `CHANGES.md:173-176`). k+1 bias fold 뒤 origin/dev **1.71 s**(compile 108.3 s, corr 0.99996, maxabs 0.0479, top-1 일치, `models/vgg16/README.md:44-48`). 같은 브랜치 1.68 s 실행의 분해는 NPU 0.120 s·CPU 0.062 s, NPU dispatch 16 개(`CHANGES.md:77-83`). 다른 구성: `--pool accel` 2.04 s, `--ukernel none` 6.80 s, f32 31.64 s, CPU+GPU f32 0.37 s. CPU 단독(f32) 0.34 s 는 팀이 적은 NPU 최적화 목표다(HX 370; 이 호스트는 HX 470). `3d454250` 부터 HAL 이 읽기 전용 binding 의 run 뒤 sync 를 건너뛰므로(VGG sync 41 → 1.3 ms) 그 이전 NPU 시간과 직접 비교하지 않는다.
- im2col 의 구조적 비용: gather 출력 163.5 MB(활성값의 9×)를 CPU 가 쓰고 NPU 가 읽는다. origin/dev 에서는 bias/ReLU 가 NPU dispatch 안에서 끝나지만, CPU 가 im2col LHS 를 gather·ones 열 pad·tile pad 로 세 번 쓴다. 층마다 NPU<->CPU 경계를 오간다.

### 2.2 네이티브 conv (conv-decompose) — origin/dev 도 업스트림 그대로

`git diff fddfec1b..dev-POOL` 에 conv 관련 hunk 가 0 개였고, origin/dev 도 conv-decompose root config 는 기능상 같다(TODO 주석만 추가, `KernelDispatch.cpp:827` → origin/dev `:894`). 아래 줄 번호는 dev-POOL 기준이고 origin/dev 와 다르면 함께 적었다.

| 항목 | 현재 | 근거 |
|---|---|---|
| 파이프라인 선택 | 모듈 전역 `--iree-amdaie-tile-pipeline=conv-decompose` 뿐. dev-POOL 의 dispatch 별 override 는 `gemv`/`pool` 만 안다 → **conv + FC matmul 혼합 모듈은 어느 값으로도 컴파일 불가** | `KernelDispatch.h:116-118`(origin/dev `:141-143`), `AMDAIELowerExecutableTarget.cpp:84-118`, `KernelDispatch.cpp:1148/1528/1563` |
| conv 인식 | 명명 `conv_2d_nhwc_hwcf`/`depthwise_conv_2d_nhwc_hwc`, 또는 iterator 가 정확히 `[P,P,P,P,R,R,R]` 인 mul+add generic. IREE 는 명명 conv 를 dispatch 전에 generic 화하고 N=1 배치를 접는다. **VGG conv 13 개가 모두 6-loop generic `[P,P,P,R,R,R]` 로 도착 → conv 로 인식되지 않음**(im2col 없이 `--compile-to=flow` 로 확인, §11: 맵 `(oh+kh, ow+kw, ic)`·`(kh,kw,ic,oc)`·`(oh,ow,oc)`, bf16 입력, f32 zero-fill 누산) | `Utils/AMDAIEUtils.cpp:577-609`, dev-conv P9(`0812bb51`), §11 |
| 타일링 | L0 `{N:1, OH:4, OW:8, OC:8}`, L1 `{1,1,8,8}` → **1 열 4 코어 1 memtile**, reduction `scf.for` KH·KW·IC_packed 각 1, pack `{OC:8, IC:8}`(npu4 bf16 8×8×8). `numRows/numCols` 는 인자조차 아님(TODO #821) | `KernelDispatch.cpp:827-871`, `AIETarget.h:381-396` |
| L1 | IC 전체 가중치 슬랩(KH·KW·IC·OC_t)과 3 행 이미지 슬랩을 reduction 타일링 **전에** L1 승격. 모든 L1 버퍼가 depth 2(기본)라 npu4 에서 IC ≥ 256 인 8 층이 초과(IC=256 실측 105,984 B); dev-conv `780c4acd`(depth 1)로도 IC=512 인 5 층이 초과(105,728 B) | `Passes.cpp:758-781`, `AMDAIEAssignLogicalObjectFifoDepth.cpp:39-50`, `log_shjo/e2e/bd_vgg_56x56x256_npu4.txt`, docs_Conv/0907 |
| 벡터화 | 코어 본문은 generic linalg 벡터화 → bf16 contract 가 AIE2P 에서 하강 불가(i8 만) → **npu4 에서는 스칼라만**. 벡터화 스위치는 모든 함수에 공통(기본 true, VGG 레시피는 false) | `VectorToAIEVecConversions.cpp:118-126`, `AIEVecToLLVM.cpp:669-673`, `Passes.cpp:978-986`, `AIETarget.h:58` |
| ukernel | conv 매처·커널 없음. 이름은 matmul/matvec/zero_fill/trunci/softmax 이고 matvec 은 GEMV 의 unpacked 2-D 타일 전용(`69a5e8b7`). matmul 매처는 packed rank-4/6 타일만, zero_fill 은 rank 2/4/6 fill 만 받는다. conv-decompose 파이프라인은 ukernel 패스를 부르지 않는다 | origin/dev `AMDAIELowerToUKernels.cpp:122-129, :268-271, :468-472`, `Passes.cpp:724-804` |
| 패딩 | `AMDAIEPad` 는 nofold 스테이징 복사(배수 1)일 뿐. SAME pad 는 bf16 변환(truncf)과 합쳐진 **host elementwise dispatch** 가 패딩된 bf16 입력(예: 226×226×3)을 새로 쓴다(§11). 형상 패딩(IC 3, OH/OW/OC 비배수)은 없다. dev-POOL 의 Flow 패딩은 `getMatmulInfo` 가 명명 `linalg.matmul` + fill init 만 받으므로 conv 를 암묵적으로 건너뛴다 | `AMDAIEPad.cpp:70-73`, `AMDAIEPadContractionDispatches.cpp:192`(origin/dev `:162-200`: epilogue 허용·정적 shape 요구가 늘었고 여전히 matmul 만), §11 |
| epilogue | origin/dev 의 matmul 계열은 bias fold(`f3079605`·`dd05ed3e`, 필수)와 unary epilogue(GEMV `33a7112e`, pack-peel 융합 consumer, pad `72a730d5`)를 지원한다. conv-decompose 는 둘 다 없고 fold 매처는 `linalg.matmul` 만 본다. origin/dev 레시피는 no-fuse 플래그를 쓰지 않으므로 네이티브 conv dispatch 에는 bias add + ReLU 가 융합된다(코드 추론, 미검증) | origin/dev `AMDAIEFoldBiasIntoContractionDispatches.cpp:69-81`, `38613bbb` |
| 타입 | bf16→f32 8×8×8, i8→i32 8×8×8; f32 는 vector 경로 없음(demote 필수) | `iree_aie_runtime.cc` |
| 테스트 | lit·샘플·CI 모두 **npu1_4col** (CI: N=2 14×14 IC32 OC64, i32/bf16/i8), depthwise CI 비활성, **npu4 conv 테스트 0 건** | `run.py:38, 2495-2513`, `Passes.td:677` |
| 기타 TODO | transpose-on-source DMA, Sequential alloc "to avoid numerical error", SplitLogicalObjFifos 제외, AIR 경로 canonicalizer — 모두 전역 파이프라인 값으로 분기 | `Passes.cpp:952-963, 998-1001, 1105-1137, 1195-1212` |

### 2.3 dev-conv (2026-08-24 ~ 09-08, 24 커밋) — 성과와 블로커

dev-POOL 과의 공통 조상은 `f7a5c86f`; dev-POOL 에도 origin/dev 에도 **하나도 포함돼 있지 않다.** 문서는 `docs_Conv/`(31 편)·`docs_BD/`(8 편), 검증 스크립트는 `log_shjo/e2e/`(미추적).

**달성(컴파일·lit·정적 IR 로 검증, 하드웨어 미검증)**

| 커밋 | 내용 | dev-POOL 적용 시 |
|---|---|---|
| `93d66148` + `7d3eb3e1` | P1 root op 기준 dispatch 별 파이프라인 라우팅(`getTilePipelineForRootOp`) + P3 `numRows` de-hardcode + matmul4d 가드 | `AMDAIELowerExecutableTarget.cpp` **충돌**(dev-POOL 의 translation_info override 와 같은 switch) |
| `0812bb51` | P9 batch-folded 6-D NHWC conv 인식 + rank 별 설정 벡터 | 자동 머지 |
| `dd8f5c06` | P6 op 별 pack transpose side(`amdaie.conv_decompose` func 마커) | 자동 머지 |
| `e153c172`·`2509390c`·`c417145b` | Flow 경계 IC(3→8)/OH/OW/OC zero-pad + crop (host dispatch) | dev-POOL: 자동 머지, 테스트 파일 충돌. origin/dev: `.cpp` 3 곳·lit 6 곳 충돌하고, dispatch 안 `tensor.pad`(IC+1 처럼 inner dim 이 1 늘면 llvm-cpu 실패, `710d6b0e`)와 가중치 런타임 pad 를 쓰므로 cherry-pick 대신 이식(T2.4) |
| `855c3168` | DMA write-side repetitionCount 접기: memtile BD 7224 → 36 (npu4 IC=512), 다른 vmfb 바이트 동일 | 자동 머지 |
| `780c4acd` | conv-decompose `l1BufferDepth = 1` (IC=256 통과, DMA/compute 오버랩 상실) | 자동 머지 |
| `69ec5c13` | softmax → GeneralCopy 라우팅 (conv 와 무관, 분리 권고) | — |

컴파일 표(npu4, dev-conv 트리): `conv_ow16_bf16`, `vgg_112x128`, `vgg_56x56x256` ✅ / `vgg_28x28x512`, `vgg_14x14x512` ❌ L1 `105728 > 65536`. 전 층 1 열 4 코어 1 memtile, 스칼라. 13 층 전체 표는 없다(5 형상만). §5.0 의 교정된 L1 식이 dev-conv 실측값(52,480 B, 105,728 B)과 dev-POOL 의 depth 2 실측값(105,984 B)을 그대로 재현한다.

**블로커(미해결, wip 패치 9 개로만 남음)**

| 블로커 | 측정 | 시도와 결과 |
|---|---|---|
| L1: IC 전체 가중치 | 73,728 B (IC 512) | IC 용량 청킹 재구성(`wip/0903_p3_restructure_wip.patch`): L1 52,736 B 달성 → **활성값 L2→L1 BD 5 차원 > memtile 4 차원**에서 정지 |
| 1 열만 사용 | 32 코어 중 4 | `colFactor` 열 타일링: forall 32 스레드 달성 → split 없이는 **DMA 채널 소진**; `SplitLogicalObjFifos` 는 "첫 >1 차원"(KH)을 쪼개고 OC 를 못 고름; split 축 강제는 matmul 회귀 |
| OC 선두 L2 레이아웃 | — | 2 단계 packing(두 순서 모두): bf16 **innermost stride 1 규칙**과 원리적 충돌 → 실패 확정 |
| 조용한 오답 경로 | HW corr 0.111778 | `getNewOffsetConfig` 가 offset 불일치 시 `objFifoIndex = 0` 반환(`AMDAIELogicalObjFifoSplittingUtils.cpp:673`) — 컴파일 타임 가드 없음 |
| submodule 패치 | — | `FormDispatchRegions.cpp` 1 줄(`isaConvolutionOpInterface`)이 **작업 트리에만** 존재. dev-POOL 레시피에서는 없으면 bias 가 conv root 에 융합돼 공간 패딩이 적용되지 않았다. origin/dev 레시피는 그 플래그(no-fuse)를 지웠으므로 이제 실험 전용(G02) |

### 2.4 재사용 자산 (dev-POOL 의 GEMV·MaxPool, origin/dev 추가분)

| 자산 | 위치 | conv 에 주는 것 |
|---|---|---|
| Flow host pad/crop dispatch 빌더 | origin/dev `AMDAIEDispatchRewriteUtils.{h,cpp}` 의 `createPaddingDispatch`(high pad 만, pad 값 하나, splat + tied in-place copy, `710d6b0e`), `createCropDispatch`(`AMDAIEPadContractionDispatches.cpp:260`), `AMDAIEPadPoolingDispatchesPass`. `createConcatDispatch`·N-split 패스는 `7d49efaa` 로 삭제 | IC/OH/OW/OC 패딩을 텐서 수준 항등 변환으로. 단 SAME(low+high) 패딩은 이 빌더로 표현하지 못한다 |
| Flow 상수 재작성 helper(origin/dev) | `getConstantValue`(dense_resource 포함), `createReplacementConstant`, 컴파일 때 상수 zero pad(`createPaddedConstant`, `5769144f`) | T2.4 의 가중치 사전 pack·IC/OC 패딩과 T5.1 의 bias 탭을 런타임 dispatch 없이 만든다(VGG 가중치 16 개 모두 Flow 에서 상수로 인식). 비용은 컴파일 중 상수 복제(+14 s) |
| host halo pre-split | `AMDAIEConvertPoolingToChannelsLast.cpp` `PoolingNhwcMaxHaloSplit`(212-304) | 3×3/s1 처럼 window ≠ stride 인 연산을 OH 조각(halo 2 행 포함)으로 잘라 열에 분배 — conv 가 그대로 필요로 하는 패턴. 단 `linalg.pooling_nhwc_max` 전용 패턴이라 conv 용 재구현 필요 |
| 공간×채널 코어 격자 + 열 정책 | `choosePoolColumns`(odd/cost/max/full), `spatialChannelPoolTiling`, `swapThreadXY` | OH 를 열, 채널을 행에. 정책을 preprocessing·Flow pad·root config 가 **같은 Utils 함수**로 결정 |
| copy 기반 reduction 파이프라인 | `addGemvPassPipeline` | 누산기를 L1 에 두고 K 루프를 바깥에서 도는 구조(peel first/last), memtile → 열 4 코어 broadcast. origin/dev 의 코어 본문은 matvec ukernel(`69a5e8b7`)이다: unpacked 타일 + K 스트리밍 누산의 ukernel 선례 |
| dispatch 별 파이프라인 override | origin/dev `KernelDispatch.h:141-143`(`amdaie.tile_pipeline` = gemv/pool), `AMDAIELowerExecutableTarget.cpp:84-103`(그 밖 값은 에러) | conv 용 값 추가 지점. 팀은 '문자열 override → `TilePassPipeline` enum' 정리를 남은 일로 기록했다(`CHANGES.md:200-201`). 형식은 곽준호 님과 합의 |
| 옵션 배관 | `AIETarget.h` → `addConfig` → executable target attr → `getConfig*` | conv 정책 플래그 전달. 배치(placement)만 바꾸는 스위치는 `AssignDeviceAffinities` pass 옵션으로 둔다(`--iree-amdaie-accelerate-pooling`, `f44d40e0`). target config 키는 target_device·ukernels·gemv_pad_n·gemv_n1·pool_columns_policy·num_rows·num_cols 뿐이다(`pooling`·`gemv_pipeline`·`core_stack_size` 삭제) |
| lit 템플릿 | `pad_*_dispatches.mlir`, `lowering_strategy_pool.mlir`, `pool_pipeline.mlir`, `gemv_pipeline.mlir` | npu4 conv 테스트 골격 |
| 알려진 결함(해결) | docs_POOL/1012 §8.10 → origin/dev `6ef3dc1f`, `0c5bc58f` | memtile 가중치를 코어별 MM2S 4 개가 lock 한 쌍으로 나눠 읽던 GEMV 교착은 consumer 별 lock 쌍으로 해결됐고, 1.0 패딩은 0 으로 되돌렸다(팀 NPU 확인: f32 GEMV hang 케이스 통과, VGG GEMV-on corr 1.0). 조건은 consumer 슬라이스가 producer 전송을 연속·오름차순으로 나눌 때뿐이다. 인터리브·겹침 슬라이스는 공유 쌍 그대로 |
| 원인 미상 오답(주의) | origin/dev `33b86238` | GEMV 가 코어당 출력 1 개(C [1,1], 가중치·활성값 L1 타일이 같은 memref 타입)일 때 npu4 에서 전부 0 을 낸다. 정적 검사로 안 보였고 NPU 로만 발견됐다. copy 기반 conv 파이프라인도 퇴화 타일을 피한다(§8) |

### 2.5 팀 브랜치 자산 (읽기 전용 확인)

`origin/dev` == `origin/k1-fold-integration` `786d1d19`(곽준호; dev-POOL 을 `7060d5f3` 로 포함, dev-POOL 보다 124 커밋 앞). `origin/3devices-integration` 은 `2c5763b8` 에 머물고, `origin/pool-integration` 은 `bdd7d47e`(origin/dev 의 조상)다. `origin/3devices-f32-aie2p-bf16`(최지훈)만 김규민 브랜치를 포함한다. 이기영 님의 `origin/feature/k1-fold`(`3bad9591`)·`gecv-k1fold`(`a66400ad`)는 미병합이고, origin/dev 는 이를 참고만 해 fold 를 새로 설계했다(`CHANGES.md:11, :44-49`). 사용자 브랜치 가운데 dev-POOL·dev-GEMV(GECV)는 dev 에 들어갔고 dev-conv 는 들어가지 않았다.

| 브랜치 / 커밋 | 내용 | conv 관련성 | `origin/dev` 포함 |
|---|---|---|---|
| `origin/aie2p-bf16-vectorized` `9b4f831a` (김규민) | AIE2P bf16 8×8×8 `vector.contract` → bfp16 MAC (`aievec/VectorToAIEVecConversions.cpp`, `AIEVecToLLVM.cpp`, `XLLVMOps.td`) | **직접** — 네이티브 conv 코어 본문을 벡터화할 유일한 컴파일러 경로 | ✗ |
| 〃 `d2846c0d` | memtile join(다중 producer) producer 별 lock 쌍 | 간접 (producer 측 race) | ✗ (origin/dev 는 `6499776a` 계열로 join 을, `6ef3dc1f` 로 fan-out 을 다르게 수정 → 건너뛴다) |
| 〃 `17986af8`/`822f0fed` (최지훈) | 코어 진입 시 crRnd = conv_even (`--iree-amdaie-core-rounding-mode`) | **직접** — ukernel 밖의 bfp16 변환에 필수(없으면 corr 0.979) | ✗ |
| `origin/dev` `8c8a9b31`+`7d485626` | Peano v22 + full unroll 비활성 | 간접 — 스칼라 fallback 6–16× | ✓ |
| `origin/dev` `92736f26`,`462325de`,`65ece963`,`cff1e56e`,`9177dcf6` | IR 에서 peano ukernel 인스턴스 생성·transpose_b, 짝수 블록 타일, 템플릿 표, ukernel 내부 RNE(`a6874c90` 이후 aie_api `swap_rounding` 으로 같은 동작), peano 기본 | **직접** — im2col matmul 고속화(STAGES 2.29 s, 지금 1.71 s) | ✓ |
| `origin/dev` `286008ec`,`d3486957`,`ab97d20a`,`a6874c90`,`75ebbebf`,`f0ef9de3`,`7854b375`,`69a5e8b7` | peano ukernel 을 mlir-aie `aie_kernels`(aie_api) 기반으로 교체: `third_party/aie_api` embed, matmul = `matmul_vectorized_2x2_mmul`(bf16 은 bfp16 에뮬레이션 + RNE), zero_fill(크기 무관), matvec(GEMV, K tile 256/64) | **직접** — conv ukernel(T4.4)의 작성 기반. MLP/VGG 출력 바이트 동일, bf16 matmul 스택 576 → 64 B | ✓ |
| `origin/dev` `f3079605`,`dd05ed3e`,`5769144f`,`38613bbb`,`33a7112e`,`72a730d5` | bias k+1 fold(필수, FC·im2col), 상수 가중치 compile-time pad, no-fuse·detach 레시피 삭제, GEMV unary epilogue, pad 의 epilogue 처리 | **직접** — 네이티브 conv bias fold(T5.1)와 코어 epilogue(T5.2)의 틀 | ✓ |
| `origin/dev` `6ef3dc1f`,`0c5bc58f`,`33c20094`,`3d454250` | consumer 별 lock 쌍(연속 슬라이스), GEMV 0 패딩 복귀, L1/L2 버퍼 64 B 정렬(AIE2P), 읽기 전용 binding sync 생략 | **직접** — G09 해결, 정렬 제약(C12), NPU 시간 정의 변화 | ✓ |
| `origin/dev` `990bcab2`,`18cafea6`,`2119b20f` | DMA loop-subsumption 반복 한도, wait 접기, BD 체인 분할 | 간접 — K=4608·M 반복 >256 형상(conv4/5)에서 필요 | ✓ |
| `origin/dev` `b492865a` | vector 명령 없는 타입(f32)을 스칼라 pack 크기로 패딩, `getScalarMatmulPackSize` | 간접 | ✓ |
| `origin/dev` `scripts/onnx/{compile,run,e2e,dispatch_times}.py` + build-trace 런타임 | **dispatch 별 NPU/CPU/hwctx 시간** 측정 도구 | 측정 필수 | ✓ |
| `origin/feature/gecv-k1fold` (이기영) `502af3ab`,`0d6a805d`,`c82b61c2` | bias 를 K 열/추가 입력 채널로 접기(`AMDAIEFoldBiasIntoContraction`), GEMV unary epilogue(ReLU), conv/bias pad-contraction | 참고 — origin/dev 는 Flow 단계 fold 를 새로 설계했고 Preprocessing fold 패스·옵션(이 브랜치의 conv 경로)은 넣지 않기로 했다(`CHANGES.md:46`) | ✗ |
| `origin/bert-*` (장우진·김예진) | 다중 producer shared-lock 결함 문서(2026-09-08/12), 코어 상수 슬라이싱 | 간접 — lock 설계 참고 | 일부 |

`git merge-tree`(드라이런, 11-03 재확인): `origin/aie2p-bf16-vectorized` 의 `9b4f831a`·`17986af8` 은 origin/dev 에 충돌 없이 적용된다. dev-conv 의 Flow 패딩 커밋은 origin/dev 에서 `AMDAIEPadContractionDispatches.cpp` 3 곳·lit 6 곳이 충돌한다. `855c3168`(BD 수정)은 충돌 없다. dev-POOL + origin/dev 머지는 팀이 끝냈다(`7060d5f3`). `AMDAIEPadContractionDispatches.cpp` 는 origin/dev 에서 pad 패스·fold 패스·`AMDAIEDispatchRewriteUtils` 세 파일로 나뉘었다(`198651de`).

### 2.6 업스트림

nod-ai/iree-amd-aie 는 2026-06-03 이후 사람 커밋·머지 PR·이슈 활동이 없다(열린 PR #1414 softmax ukernel, #1417 DMA chain fusion 뿐). iree-org/iree 로 이전되지도 않았다. conv 관련 마지막 변경은 2025-03 (#1193); #821(conv 4×4 배열)은 BD 오류만 고치고 다열 설정은 추가되지 않았고, #529(conv ukernel)는 결론 없이 열려 있다. mlir-aie 의 conv 커널은 int8/uint8 뿐(bf16 2-D conv 없음); `programming_examples/ml/conv2d_14x14` 가 8 열 × 4 행 32 코어 분배(OC 를 열, 행은 1/4 씩, 입력 행 방향 broadcast, 가중치 열 방향 broadcast)의 구조 참고가 된다.

IREE 서브모듈(`c73928f0`)에는 `iree-codegen-convolution-to-igemm`(conv → tileable `iree_linalg_ext.im2col` + contraction, `IGEMMConfigFn` 훅)과 `populateConvToIm2colOpPatterns(patterns, controlFn)` 이 있다. 사용처는 LLVMGPU 뿐이고 플러그인은 쓰지 않는다(`git grep` 0 건). 이 리포가 재사용할 수 있는 유일한 외부 conv 인프라다(G21, T2.6).

---

## 3. 설계가 지켜야 하는 하드 제약

| # | 제약 | 값 / 근거 | conv 에 미치는 영향 |
|---|---|---|---|
| C1 | 코어 L1 | 64 KiB − 스택 1 KiB. origin/dev 는 pack-peel(`KernelDispatch.cpp:295`)·GEMV(`:705`)·pool(`:1518`) 모두 `getCoreTileLocalMemorySize() - stackSize` 를 쓰고 비교는 `<=` 다(`93c03407`). 버퍼마다 시작 주소를 정렬한다(npu4 코어 64 B 이상 버퍼는 64 B, 그 밖 32 B, memtile 4 B, `33c20094`). 오늘 하강은 모든 L1 objFifo 에 같은 depth(기본 2)를 줘 누산기도 이중 버퍼다(`AMDAIEAssignLogicalObjectFifoDepth.cpp:39-50`). conv-decompose(`:894-`)만 L1 검사 없이 고정 타일 | IC=512 가중치 타일 73,728 B → **IC reduction 타일링 필수**. 누산기만 depth 1 로 두려면 새 기능 필요. 코어 내 epilogue 를 넣으면 그 출력 버퍼도 L1 에 더해진다 |
| C2 | BD 차원 | 코어·shim 3-D, memtile 4-D (`aie/AIEDialect.cpp:474-484`); stride 0 금지; 차원 크기 < 1023; **bf16 은 innermost stride 1** (`:519-521`); **BD 전송 길이와 주소는 4 B 배수**(origin/dev `:440-444, :567-569`; `5eb5a583` 이 bf16 K tile 251 = 502 B 에서 겪음) | IC 청킹 후 활성값 5-D BD 문제, 2 단계 pack 불가, pack 레이아웃 선택 제약. bf16 은 BD 당 원소 수가 짝수여야 한다(IC 8 배수 패딩이면 충족, 패딩 없는 IC=3 탭 전송 6 B 는 위반) |
| C3 | memtile BD 수 | 48 (짝/홀 24 씩, 채널 {0,2,4}/{1,3,5} 공유) | IC=512 에서 7224 BD 폭증 사례 — write-side repetition 접기로 36 |
| C4 | DMA 채널 | shim 열당 MM2S 2 / S2MM 2, memtile 6/6, 코어 2/2; control overlay 가 shim MM2S 1 개 선점 | 피연산자 3 개(act·weight·bias) dispatch 는 shim 채널 초과(`no producer DMA channel available`); ctrlpkt 모드는 전 배열 dispatch 컴파일 실패 |
| C5 | lock | objFifo 타일당 기본 lock 1 쌍(producer init = max(producer, consumer) × depth). N→1 join 은 producer 별(`6499776a` 계열), 1→N fan-out 은 consumer 별(`6ef3dc1f`) 쌍(init = depth). 둘 다 슬라이스가 상대 전송을 연속·오름차순으로 나눌 때만 적용. N→M 은 에러(origin/dev `AMDAIEObjFifoBufferization.cpp:132-137`) | 인터리브·겹침 슬라이스의 1→N fan-out 만 공유 쌍으로 남아 코어 속도 차이로 교착할 수 있다. consumer 별 쌍은 producer BD 를 consumer 수 × depth 로 늘린다(C3) |
| C6 | 벡터 명령 | AIE2P 행렬 MAC 은 bfp16 만(bf16 은 원소별); aievec 지원 목록 dev-POOL = i8 8×8×8 | bf16 벡터화는 김규민 패치 + crRnd 필요, 결과는 **bit-identical 이 아님**(corr 0.99997) |
| C7 | 컴파일러 라우팅 | 파이프라인은 모듈 전역 + override(gemv/pool); objFifo 하강의 conv 분기도 전역 값 | 혼합 모듈에서 conv 가 matmul 설정을 받음 |
| C8 | 런타임 | dispatch 당 hwctx 생성·해제 약 75–95 ms(default 전원 모드; performance 모드에서는 거의 0, 단 root 필요), 컨텍스트 16 개. origin/dev VGG 는 NPU dispatch 16 개로 hwctx 약 1.2 s(`CHANGES.md:176, :184`) | e2e 는 conv 와 무관한 바닥 약 1.2 s. 지표는 dispatch 별 NPU 시간 |
| C9 | 검증 정책 | NPU 실행 금지(승인 시 예외) | lock·동기화 결함은 IR 로 증명 불가 → 승인 세션을 마일스톤에 명시 |
| C10 | 빌드 | 컨테이너 전용, `JOBS=4`, 트리 되돌리면 재빌드. 호스트 빌드 락은 아직 미커밋(G18) | 스테일 바이너리 오진 2 회 전력 |
| C11 | shim DMA 반복 | BD 는 intra 3 차원 + inter 1 차원, inter 반복 최대 63 (`xaie2pgbl_reginit.c:2066`, `iree_aie_runtime.h:355`); repeat 256 은 stride 0 차원만(`990bcab2`) | 접히지 않는 루프는 제어 코드로 풀려 명령이 폭증(45 KB → 42 MB 사례)하거나 stall. DDR 연속 구간이 짧으면(16–32 B) 대역폭이 떨어진다(GEMV 에서 64 B → 512 B 로 고친 전례, docs_GEMV/0909_gemv_solution_catalog.md:27, :43) |
| C12 | 벡터 load 정렬 | 장치 모델은 정렬 256 bit, 최대 벡터 1024 bit(`iree_aie_runtime.h:387-390`, npu4 재정의 없음)이고 aievec `align-transfer-reads` 도 이 값(32 B)으로 판정한다. 그러나 **AIE2P 의 512-bit 이상 접근은 64 B 정렬이 필요하고, 어긋나면 오류 없이 다른 주소를 접근한다**(`33c20094`: 16 B 경계 버퍼로 f32 GEMV corr ~0.92; 할당기는 이제 64 B 이상 코어 버퍼를 64 B 에 둔다). 2 배 읽기로 맞출 수 없으면 경고만 내고 어긋난 읽기를 남긴다(`aievec/VectorToVectorConversions.cpp:1416-1428`) | kw=1(16 B)뿐 아니라 kw=2(32 B), 그리고 `[t_b+2][IC_c/8][W+2][8]` 의 (행, ic_o) 블록 간격 (W+2)·16 B(W 가 8 의 배수면 32 mod 64)도 512-bit 읽기를 어긋나게 한다. 32 B 어긋남은 aievec 가 경고 없이 통과시킨다 → 레이아웃·커널 선택 문제 |
| C13 | bias fold 전제(origin/dev) | 상수 float bias·상수 weight, caller 1 개, tied 결과 없음, 정적 shape. residual·split-K 부분합 accumulator 는 오류. host 장치가 선언된 이종 컴파일에서만 돈다(`AMDAIEFoldBiasIntoContractionDispatches.cpp:334-399`, `CHANGES.md:16, :90`) | dedup off 유지 필수(같은 모양 conv 가 executable 을 공유하면 caller 여러 개), T3.4 부분합은 accumulator 로 넘기지 않음, i8 conv bias 는 fold 불가, 단일 장치 컴파일에서는 bias 가 세 번째 입력으로 조용히 남는다 |

---

## 4. 결함 · 미구현 목록 (해야 하는 일의 근거)

심각도: **blocker** = 이것이 풀리기 전에는 네이티브 conv 가 VGG 에서 돌 수 없음 / **major** = 성능·정확성·재현성·안전을 크게 깎음 / minor.

| ID | 결함 / 미구현 | 심각도 | 근거 | 푸는 작업 |
|---|---|---|---|---|
| G01 | dev-POOL 의 conv matmul 은 **스칼라 bf16**(ukernel none, 벡터화 off, Peano v19)이었다 | 해결(origin/dev) | origin/dev 는 Peano v22 + bf16 ukernel + GEMV matvec 로 VGG 1.71 s. `conv-npu` 에 들어와 있다 | T1.3(재현 확인) |
| G02 | **미커밋 submodule 패치**(`FormDispatchRegions.cpp`, no-fuse 플래그를 generic conv 에도 적용) | minor(실험 전용) | origin/dev 레시피는 no-fuse·detach 를 쓰지 않는다(`38613bbb`). 팀은 두 fork 옵션을 정리하고 dispatch 형성은 플래그 대신 패스로 다루기로 했다(`CHANGES.md:98, :100`). 패치는 §11 같은 비교 실험에만 필요 | T1.1(생략) |
| G03 | ~~dev-POOL ↔ origin/dev 머지~~ → 팀 머지(`7060d5f3`) 완료. GEMV 판정은 `getPaddingMultiples` 안에서 짝수 블록 규칙보다 먼저 한다(`isGemvShape(m, transposeB)`). pool/GEMV 코어는 팀이 Peano v22 에서 컴파일·실행했다. **남은 것: ukernel 아래 N=64 → 128 패딩** | minor | origin/dev `AMDAIEPadContractionDispatches.cpp:101-110, :132-139`, lit `pad_contraction_dispatches.mlir:428-470`, `247f29d2`, `31f1e935` | D3 |
| G04 | dispatch 별 **conv 라우팅 없음**, 6-D(batch 접힘) 인식 없음 → VGG conv 는 conv 로 분류되지 않고, conv 가 FC/GEMV/pool 과 한 모듈에 공존 불가. objFifo 하강의 분기들이 전역 파이프라인 값에 묶임 | blocker | `KernelDispatch.h:116-118`(origin/dev `:141-143`); `AMDAIELowerExecutableTarget.cpp:94-101`; `AMDAIEUtils.cpp:577-586`; `Passes.cpp:958-960, :998-1001, :1109-1110, :1135-1136, :1210`; §11 | T2.2, T2.1 |
| G05 | **L1 용량·산술 강도**: conv-decompose 는 KH·KW·IC·OC_t 가중치 슬랩 전체(IC=512 에서 73,728 B)를 reduction 루프 전에 승격하고 코어당 블록마다 1×8×8 출력 타일만 계산(≈2.8 MAC/B). L1 초과만으로 dev-POOL(모든 L1 버퍼 depth 2)에서는 IC ≥ 256 인 **8 층**, dev-conv `780c4acd`(depth 1)로도 IC = 512 인 **5 층**이 컴파일 불가. 나머지도 벡터화되면 DMA 바운드 | blocker | `Passes.cpp` conv-decompose 순서(L1 승격이 level-2 `scf.for` 앞); `KernelDispatch.cpp:865-871`; `AMDAIEAssignLogicalObjectFifoDepth.cpp:39-50`; 실측 105,984 B(IC 256, depth 2, `log_shjo/e2e/bd_vgg_56x56x256_npu4.txt`), 105,728 B(IC 512, depth 1) | T0.3, T3.0, T3.1, (T3.4) |
| G06 | npu4 에서 **matmul 이 아닌 conv 본문용 bf16 벡터 경로 없음**(aievec AIE2P = i8 뿐, ukernel 매처 = linalg.matmul 형상뿐) — dev-POOL 과 origin/dev 모두 | blocker | `VectorToAIEVecConversions.cpp:118-126`; `AIEVecToLLVM.cpp:669-673`; `AMDAIELowerToUKernels.cpp` 매처; `9b4f831a`·`17986af8` 은 origin/dev 조상 아님 | T0.6, T4.1, T4.2, (T4.4) |
| G07 | 네이티브 conv 는 **1 열 4 코어**: `numCols` 미사용, HWCF L2 가중치 레이아웃은 split 안전하지 않음(split 패스가 "첫 >1 차원" = KH 를 고름), 열 시도는 채널 소진, 2 단계 pack 은 stride 규칙 위반 | major | `KernelDispatch.cpp:865-869`; `AMDAIESplitLogicalObjFifos.cpp:201-203`; `AIEDialect.cpp:521`; docs_Conv/0907 §2.5, §3.3 | T0.7, T2.4, T3.3 |
| G08 | **공간 SAME 패딩, 3×3 행 겹침(window ≠ stride), 비배수 형상**(IC=3, OW 14/28, OH 14)을 네이티브로 처리 못 함. dev-conv 의 Flow 패딩은 root config 와 공유되지 않는 규칙으로 패딩 | major | `AMDAIEPad.cpp` padToMultipleOf=1; `KernelDispatch.cpp:1365-1367`(pool 규칙); dev-conv `getConvTiles`; `models/vgg16/vgg.mlir:38` pads=[1,1,1,1] | T2.2, T2.3, T2.4 |
| G09 | **1 producer / N consumer 공유 lock 쌍** — origin/dev `6ef3dc1f` 로 해결(조건부): consumer 들의 source offset 이 서로 다르고 producer 전송을 연속·오름차순으로 정확히 나눌 때만 consumer 별 lock 쌍(init depth/0)을 주고 producer 전송을 슬라이스별 BD 로 나눈다. 인터리브·strided 슬라이스는 공유 쌍(init = consumer 수 × depth)을 유지하고, 겹치는(halo) 슬라이스도 연속 분할 검사를 통과하지 못해 같다(코드 기준). GEMV 1.0 패딩은 `0c5bc58f` 로 폐기 | minor(조건 밖 레이아웃만) | origin/dev `AMDAIEObjFifoBufferization.cpp:30-86, :157-185`, `AMDAIELowerToAIE.cpp:797-835`; lit `obj_fifo_bufferization.mlir:250-329`; 팀 NPU 확인 | T4.3(감사) |
| G10 | **bias + ReLU + truncf** — im2col 경로는 origin/dev 에서 해결: 상수 bias 를 K 에 접고(`AMDAIEFoldBiasIntoContractionDispatches`, 필수) ReLU 는 NPU dispatch 의 epilogue(VGG contraction 16 개 모두). 네이티브 conv 는 여전히 밖: fold 매처는 `linalg.matmul` 전용, epilogue staging 은 `linalg::ContractionOpInterface` producer 전용이다. 3 입력 dispatch 실패는 팀이 DMA 채널 부족으로 기록 | major(네이티브 한정) | origin/dev `AMDAIEFoldBiasIntoContractionDispatches.cpp:69-81`, `AMDAIEInsertCopyOps.cpp:197-203`, `CHANGES.md:11, :26, :340` | T5.1, T5.2 |
| G11 | **정적 제약 감사 도구가 없다**. dispatch 별 시간 측정은 `conv-npu` 에 들어왔다(`scripts/onnx`, `build-trace`). 과거 진단은 사후 에러 읽기에 의존했고 스테일 바이너리 오진 2 회, 존재하지 않는 패스 이름으로 빈 덤프를 읽은 사례(`log_shjo/e2e/bd512.txt`)가 있다 | major | docs_GEMV/0910_perf_methodology.md §1, §3.4; docs_Conv/0825_ci_vs_vgg.md §1-2 | T0.2, T0.3 |
| G12 | **npu4 conv 테스트 0 건**, conv/VGG 회귀 케이스 없음, split 의 조용한 fallback(`objFifoIndex = 0`)에 가드 없음 | major | `run.py:38, :2494-2513`; `Passes.td:677`; origin/dev tests/regression README 의 conv 0 건; `AMDAIELogicalObjFifoSplittingUtils.cpp:673` | T2.5, T6.1 |
| G13 | **e2e 목표의 모호성**: origin/dev VGG 1.71 s 중 hwctx 약 1.2 s(16 dispatch), 팀 목표는 CPU 단독 0.34 s. 네이티브 conv 도 NPU dispatch 수가 그대로면 바닥을 못 내린다. hwctx 캐시(`IREE_AMDXDNA_CACHE_HWCTX`)는 dev 통합 때 제거됐고, ctrlpkt 컨텍스트 공유는 npu4 전 배열 dispatch 에서 컴파일 실패 | major | `CHANGES.md:176, :197, :234, :353`; docs/1030_vgg_ctrlpkt.md §0 | D4, T5.4, §5.9 |
| G14 | **팀 분기**: join lock 수정 2 종(`d2846c0d` vs `6499776a` 계열), 반올림 수정 2 종(`17986af8` vs `cff1e56e`, 후자는 `a6874c90` 로 aie_api 코드로 대체)이 같은 파일; pad 패스는 origin/dev 에서 세 파일로 나뉘었고 이기영 님의 `feature/k1-fold` 는 미병합 | major | 각 커밋 `git show --stat`; `git log f7a5c86f..origin/dev -- KernelDispatch.cpp` = `65ece963, b492865a, 462325de` | T4.1(-x), T6.3 |
| G15 | **정확도 정책 미정**: bfp16 MAC 은 corr 0.99997, 스칼라는 1.00000 | minor | STAGES.md:24; aie2p docs/patch_aie2p_bf16_vectorized.md | T4.5, D10 |
| G16 | **L3(DDR↔shim) 쪽이 설계에 없음**: IC 를 8–16 씩 청크로 스트리밍하면 NHWC 활성값·FHWC 가중치의 DDR 연속 구간이 16–32 B 뿐이다. shim BD 는 intra 3 차원 + inter 1 차원(최대 63 반복)이라 64 청크(conv4_2/4_3, IC_c=8)는 한 BD 로 못 접혀 제어 코드가 반복 BD 프로그래밍으로 불어난다 | major | `xaie2pgbl_reginit.c:2057-2066`(Shim IterWrapMax 63); `iree_aie_runtime.h:355`(dmaNbInterDims 1); `git show 9c6ea263:…/AMDAIEPadContractionDispatches.cpp:700-708`(삭제된 N-split 패스 주석: 큰 가중치 shim DMA 가 63 반복을 넘으면 직렬 BD 체인이 되어 stall); origin/dev `CHANGES.md:187`(N-split 삭제 뒤 큰 가중치는 다시 DMA cliff 위험); `990bcab2`(접히지 않은 루프 → NPU 명령 45 KB → 42 MB); docs_GEMV/0909_gemv_solution_catalog.md:27, :43(64 B 버스트 병목 → 512 B) | T0.3, T2.3(R9·R10), T2.4, D14 |
| G17 | **aievec bf16 경로의 kw 시프트 읽기 정렬**: npu4 bf16 lhs 는 `vector<8x8xbf16>` = 1024 bit 이고, `[W+2][8]` 레이아웃에서 kw=1 이면 16 B 어긋난다. 벡터 load 정렬은 256 bit, 최대 벡터는 1024 bit 라 `align-transfer-reads` 가 2 배 읽기로 맞추지 못하고 **경고만 낸 뒤 어긋난 읽기를 그대로 둔다**. npu1(4×8 = 512 bit)에서는 맞출 수 있어 CI 의 npu1 결과는 근거가 되지 못한다. AIE2P 512-bit 이상 접근의 실제 요구는 **64 B** 이고(`33c20094`, 어긋나면 조용한 오답 실측), kw=2·홀수 블록의 32 B 어긋남은 경고도 없다 | major (aievec 경로의 blocker 후보) | `iree_aie_runtime.h:387-390`(npu4 재정의 없음); `aievec/VectorToVectorConversions.cpp:1416-1428`; origin/dev `aie/AMDAIEAssignBufferAddresses.cpp:31-44`(`33c20094`) | T0.6, T2.3(R11), T4.2, D5 |
| G18 | **환경·안전 전제가 미커밋**: CLAUDE.md §1 의 호스트 빌드 락(`scripts/lock/`, `build.sh` 의 `lock_run`)과 NPU 락(run-debug/deploy)이 로컬 수정·미추적 파일로만 있다. origin/dev 의 `build.sh` 도 락 없이 `build` 와 `build-trace` 를 빌드한다. `conv-npu` 로 옮기면서 로컬 락 수정을 다시 얹었지만(11-03) 여전히 미커밋 | major | `git status`(` M scripts/build/build.sh`, `?? scripts/lock/`); `git show origin/dev:scripts/build/build.sh`; `scripts/lock/lock-common.sh:13-15` | T1.0 |
| G19 | **전역 스위치 결합**: 벡터화는 objectFifo 하강에서 모든 함수에 붙는 하나의 플래그(기본 true)라, conv 를 위해 켜면 GEMV(M=1)·pool 코어도 벡터화된다. 이 조합은 시험된 적 없고 GEMV 설계 문서가 legalize 실패 가능성을 경고한다 | major | `Passes.cpp:978-986`; `AIETarget.h:58`; docs_GEMV/0909_gemv_pipeline_design.md §8 | T2.1 |
| G20 | **기존 conv-decompose 사용처**: npu1_4col CI conv 3 종(i32/bf16/i8), lit `lowering_strategy_conv.mlir`(depthwise bf16·i8 기대값), 샘플 `conv2d_nhwc_objectfifo_e2e.mlir` 이 conv-decompose 를 쓴다. 새 `conv` override 와 거부 진단이 이들을 조용히 재라우팅하거나 깨뜨릴 수 있다 | minor | `run.py:2494-2513`; `lowering_strategy_conv.mlir:1, :37-78`; `Test/samples/conv2d_nhwc_objectfifo_e2e.mlir:1`; `AMDAIELowerExecutableTarget.cpp:84-101`(override 우선) | T2.2, T2.1, T6.1 |
| G21 | **IREE 서브모듈의 implicit GEMM 경로 미검토**: `iree-codegen-convolution-to-igemm`(conv → `iree_linalg_ext.im2col` + contraction, `IGEMMConfigFn` 훅), tileable im2col op, `populateConvToIm2colOpPatterns(patterns, controlFn)` 이 있고 LLVMGPU 만 쓴다. pack-peel + 검증된 bf16 ukernel 을 재사용하면서 CPU gather 를 없앨 수 있는 중간 단계 후보다 | major (설계 결정) | `third_party/iree/.../Codegen/Common/ConvolutionToIGEMM.cpp`; `Codegen/Common/Passes.td:238-239`; `LinalgExt/Transforms/Transforms.h:46`; `LLVMGPU/Passes.cpp:438`; 플러그인 사용 0 건 | T2.6, D13 |

---

## 5. 작업 계획

### 5.0 전략과 설계 결정

**반증 우선 측정 게이트 사다리.**

1. 재빌드 없이(또는 증분 빌드로) 가장 위험한 가정 네 개를 먼저 반증한다: aievec 의 kw 시프트 읽기 **64 B** 정렬(T0.6), G2 의 objectFifo 분배·채널 예산(T0.7), IC 루프를 코어 forall 밖에 두는 구조(T3.0 사전 확인), origin/dev 레시피에서 conv 가 Flow 에 도착하는 모양(T0.4, bias 융합 여부).
2. (완료, 11-03) origin/dev `786d1d19` 에서 `conv-npu` 를 만들고 실행 환경(`scripts/onnx`, `build-trace`, Vulkan, Peano v22)을 갖춰 빌드했다. 남은 것은 빌드 락 커밋(T1.0)과 기준선 재현(T1.3, 승인 시 T1.5).
3. 네이티브 conv 를 **단일 형상 수직 슬라이스**로 먼저 증명한다(T3.0 → T3.1).
4. 그 결과로 인식·라우팅·타일 결정·**conv bias fold**·전처리를 일반화하고(T2.x, T5.1), IGEMM 대안과 비교해 설계를 확정한다(T2.6, D13).
5. 전 배열·벡터 커널·epilogue·모델로 넓힌다. 모든 단계는 정적 AIE 제약 감사로 게이트하고, NPU 실행은 승인 배치로만 한다.

세 계획안(점진 / 네이티브 우선 / 리스크 감사)을 심사해 통합하고, 비평 3 건(정확성 / 누락 / 순서)을 반영했다. 상충점은 리포 증거로 해소했다.

| 쟁점 | 결론 | 증거 |
|---|---|---|
| dev-POOL + origin/dev 머지 | 팀이 `7060d5f3` 로 머지 완료(이 문서가 예측한 5 파일 그대로, dev 우선 해소). conv 브랜치는 origin/dev 에서 분기 | `CHANGES.md:103-125` |
| dev-conv BD 수정(`855c3168`) 적용성 | origin/dev(`786d1d19`)도 `AMDAIEDmaUtils.cpp` 를 건드린 적 없음 → **그대로 적용**(11-03 재확인, 충돌 0) | `git log f7a5c86f..origin/dev -- …/AMDAIEDmaUtils.cpp` 비어 있음 |
| ukernel 켤 때 N=64 층 | `getPaddingMultiples` 의 n = numCols·2·8 = 128 → **conv1_1/1_2 가 N=128 로 패딩**(MAC 2 배). origin/dev 에 그대로 남음(`7060d5f3` 가 dev 의 perCore 규칙과 lit 를 유지) — D3 | origin/dev `AMDAIEPadContractionDispatches.cpp` `perCore`; lit `pad_contraction_dispatches.mlir:428-470` |
| "행이 OC 를 나누면 IC_c=8 강제" 규칙 | HWCF L2 레이아웃에서만 성립. **FHWC 상수 가중치** `[OC][3][3][IC]` 면 memtile→코어 가중치 DMA 가 `[9][IC_c/8][OC_t][8]` **4-D, innermost stride 1**(주소 산술, §10.5 V6). IR 확인은 T0.7 | `AIEDialect.cpp:474-477, :521` |
| 코어 격자 | **G2 = OC 를 열에, OH halo 청크를 행에**. multi-party objFifo 가 broadcast(consumer DMA 1 개, 타겟 여러 개)와 4-producer 출력 join 뿐이라 연속 슬라이스에서만 동작하는 consumer 별 lock(`6ef3dc1f`)에 기대지 않고 GEMV 교착 패턴을 **구조적으로 회피**. pack-peel matmul 의 32 코어 배치(A 행 broadcast, B 열 broadcast, C 열별 join)와 같은 통신 패턴이고 오늘 im2col 경로가 이 패턴으로 하드웨어에서 돈다. 단 가중치 읽기에서 OC 는 **셋째** 접근 차원이라 pack-peel 의 B 와 같지 않다 → T0.7 로 M0 에서 확인 | docs_POOL/1012:320-352; origin/dev ObjFifoBufferization:144-160; docs_GEMV/0909 §8 |
| 베이스 브랜치 | (완료) origin/dev `786d1d19` 에서 `conv-npu` 분기. dev-POOL 은 dev 에 포함 | `git merge-base --is-ancestor 9c6ea263 origin/dev` |
| 커널 경로 | im2col matmul 은 **ukernel**(팀 검증). 네이티브 본문의 aievec bfp16 경로는 **64 B 정렬(C12, G17)** 에 걸릴 수 있어 T0.6 으로 M0 에서 go/no-go. no-go 면 kw 복제 레이아웃(L1 비용), conv ukernel(T4.4, origin/dev 의 aie_api 인프라로 진입 비용이 낮아짐), IGEMM(T2.6) 중 선택 | `33c20094`, `VectorToVectorConversions.cpp:1416-1428`, `a6874c90` |
| 네이티브 G2 vs IGEMM | IGEMM 은 pack-peel + ukernel 을 재사용하고, DMA 가 im2col 주소를 만들어 코어는 정렬된 matmul 타일만 본다(G17 회피, stride-2/1×1 도 커버). 대신 A 가 탭마다 L2→L1 로 다시 읽혀 온칩 강도가 G2 보다 낮고, tiled im2col → `amdaie.dma_cpy_nd` 하강이 새로 필요하다. pack-peel + bf16 ukernel + 융합 ReLU 는 origin/dev 에서 하드웨어로 검증됐다(VGG corr 0.99996). T2.6 스파이크 뒤 **D13** 에서 결정 | `ConvolutionToIGEMM.cpp`, `Transforms.h:46` |
| L1 모델 | 오늘 하강은 **모든 L1 objFifo 에 같은 depth**(기본 2)를 줘 누산기도 이중 버퍼다. 버퍼마다 64 B 정렬(`33c20094`). 코어 내 epilogue 를 넣으면 그 출력 버퍼가 더해진다. 아래 식이 두 실측값(105,984 B / 105,728 B)을 재현한다 | `AMDAIEAssignLogicalObjectFifoDepth.cpp:39-50`; `gemvL1Bytes = 2·(A+B+C)`(origin/dev `AMDAIEUtils.cpp:253-257`); `KernelDispatch.cpp:274-290, :666-673` |
| 성능 지표 | 네이티브 conv 의 정직한 이득은 NPU 연산·DMA 효율과 dispatch 구조(DDR 97 vs 236 MiB, CPU gather/epilogue 제거). **hwctx 바닥 약 1.2 s 는 못 내림** → 지표를 결정(D4) | STAGES.md:38 |
| 작업 순서 | 위험한 가정을 XL 작업 뒤가 아니라 **앞에서** 반증. dev-conv 가 같은 구조(IC 루프를 코어 forall 밖으로)에서 이미 실패 사슬을 문서화했으므로 그 패스 수정(T3.0)이 슬라이스(T3.1)보다 먼저 | dev-conv `docs_Conv/0903_restructure_plan.md` §5.3–§7, `docs_Conv/wip/0903_p3_restructure_wip.patch` |

**G2 격자의 데이터 흐름** (층당 1 dispatch, 32 코어 / 8 memtile / 8 열):

- 코어 (r, c) 는 OH 청크 r 과 OC 블록 c 를 계산한다.
- 가중치(FHWC 상수, 또는 열·IC 청크 우선으로 사전 pack — T2.4): shim → 열 c 의 memtile 에 OC/8 슬랩. 8-way split 은 split 패스가 고르는 "첫 >1 차원" = OC 이므로 패스 수정 없이 열 분배된다(T0.7 로 확인). 열 c 의 4 코어는 **같은** OC 블록을 쓰므로 슬랩은 열 안에서 **broadcast**(consumer DMA 1 개, 타겟 4 개).
- 활성값: 호스트가 OH 를 numRows=4 개의 halo 포함 청크(t+2 행)로 미리 잘라 N=4 로 concat → 행 r 의 청크는 8 열의 코어 8 개에 **broadcast**(connection 1 개, 타겟 8 개 — GEMV A 피연산자 multicast 선례).
- 출력: 코어별 `t_b × W × OC_t` f32 타일 → 열 memtile 의 4-producer join → shim.
- 채널 예산(설계값): 활성값 조각을 가진 memtile 은 S2MM 6/6(가중치 1 + 활성값 1 + join 4), MM2S 3/6; 코어 S2MM 2/2(A, B); shim 열당 MM2S 2/2(가중치, 활성값). **여유가 없다**(§8).
- IC 청크 루프는 코어 forall **바깥**(shim 제어 코드 루프, pack-peel 의 K 루프와 같은 꼴), 누산기는 L1 에 상주(GEMV 선례).
- **L3 연속 구간**(G16): L2 청크를 IC_c 그대로 두면 DDR 읽기 연속 구간이 IC_c·2 B(16–32 B)다. 해결 후보는 (a) L2 청크를 IC_L2 ≥ 64(128 B 연속)로 키우고 L1 은 그 안에서 IC_c 로 나누는 2 단계 청크, (b) 호스트가 활성값을 청크 우선 레이아웃으로 쓰고 가중치를 사전 pack(추가 dispatch 0). 청크 루프 반복은 shim inter 차원 63 이하여야 한다(R10). 선택은 D14.
- **kw 시프트 정렬**(G17, C12): AIE2P 의 512-bit 이상 읽기는 64 B 정렬이 필요하다. `[t_b+2][IC_c/8][W+2][8]` L1 레이아웃에서는 kw=1(16 B)·kw=2(32 B) 시프트와 (행, ic_o) 블록 간격 (W+2)·16 B 가 모두 이를 깬다. aievec 경로라면 kw 복제 레이아웃(DMA 가 시프트 사본 3 개를 64 B 정렬로 만든다, 활성값 L1 약 3 배)이 필요하다. ukernel 은 커널 안에서 정렬 로드 + 레지스터 시프트로, IGEMM 은 DMA 가 시프트를 만들어 이 문제를 피한다(R11, T0.6).

**L1 예산식**(교정): `d_in·(⌈9·IC_c·OC_t·2⌉ + ⌈(t_b+2)(W+2)·IC_c·2⌉) + d_acc·⌈t_b·W·OC_t·4⌉ + d_out·⌈t_b·W·OC_t·b_out⌉ ≤ 65,536 − 스택 1,024`. ⌈x⌉ 는 버퍼 시작 주소 정렬(npu4 코어 64 B 이상 버퍼는 64 B, `33c20094`)이고, 마지막 항은 코어 내 epilogue 출력(T5.2, bf16 이면 b_out = 2)이다. 오늘 하강은 `d_in = d_acc = l1BufferDepth`(기본 2)이고 epilogue 가 없다. 이 식에 conv-decompose 의 오늘 타일(t_b=1, W=8, OC_t=8, IC 전체)을 넣으면 실측 105,984 B(IC=256, depth 2, 스택 포함)와 105,728 B(IC=512, depth 1, 스택 포함)가 그대로 나온다(이 버퍼들은 모두 64 B 배수다). 코어가 L2→L1 로 받는 바이트당 MAC = `t_b·W·OC_t·9 / (18·OC_t + 2(t_b+2)(W+2))` 이고 **IC_c 와 무관**하다(IC_c 는 L1 크기와 청크 반복 수만 바꾼다).

**예비 산술**(정적 계산, 컴파일 미검증; OH 를 4 의 배수, OW 를 8 의 배수, IC 를 8 의 배수로 패딩; 버퍼 정렬 포함; MAC/B 최대 → IC_c 최대 순으로 선택):

| 층 | 열당 OC | A 타일 t_b·W·OC_t·IC_c | A: 오늘 하강(전부 depth 2) L1 B | A + 코어 epilogue(누산기 depth 1, bf16 출력 depth 2) L1 B | A: MAC/B | B 타일(누산기 depth 1) | B: L1 B | B + 코어 epilogue L1 B | B: MAC/B | B: IC 청크 수 / L3 연속 구간 |
|---|---|---|---|---|---|---|---|---|---|---|
| conv1_x | 8 | 28·16·8·8 | 48,256 | 48,256 | 26.4 | 28·32·8·8 | 63,616 | 92,288 ❌ | 29.5 | 1·8 / 16 B |
| conv2_x | 16 | 14·16·16·16 | 56,320 | 56,320 | 37.3 | 28·16·16·8 | 50,560 | 79,232 ❌ | 47.2 | 8·16 / 16 B |
| conv3_x | 32 | 14·8·32·16 | 57,344 | 57,344 | 36.0 | 7·56·16·8 | 46,464 | 71,552 ❌ | 42.4 | 16·32 / 16 B |
| conv4_x | 64 | 7·32·16·16 | 57,472 | 57,472 | 35.8 | 7·32·32·8 | 47,744 | 76,416 ❌ | 54.3 | 32·**64** / 16 B |
| conv5_x | 64 | 4·16·64·8 | 54,656 | 54,656 | 26.9 | 4·16·64·16 | 60,160 | 76,544 ❌ | 26.9 | 32 / 32 B |
| (참고) 오늘 conv-decompose | — | 1·8·8·IC 전체 | 105,984 ❌(IC 256, depth 2) | — | 2.8 | 같음, depth 1(dev-conv) | 105,728 ❌(IC 512) | — | 2.8 | 1 / — |

- 오늘 하강(A)으로도 13 층 모두 들어가는 타일이 있고 MAC/B 는 26–37(오늘 타일의 약 9–13 배)이다. 코어 내 epilogue 를 넣으면 누산기를 depth 1 로 내리고(pack-peel 이 elementwise consumer 가 있을 때 하는 방식, `KernelDispatch.cpp:274-290`) bf16 출력을 depth 2 로 두어 L1 이 그대로다. B 안(누산기 depth 1 + 큰 타일, MAC/B 27–54)은 epilogue 를 넣으면 모든 층이 넘친다. 따라서 **코어 내 epilogue 를 쓰는 최종 구성의 MAC/B 는 A 안의 26–37** 이고, L1 objFifo 별 depth(누산기 1)는 epilogue 와 함께 필수다.
- 코어 S2MM 2 채널 × 채널당 약 4 B/cycle(추정, docs_GEMV/0914:166-171, 하드웨어 미검증)이면 26 MAC/B 는 피크 512 MAC/cycle 의 약 41 %, 37 은 약 58 %, 54 는 약 85 % 를 먹일 수 있다. 오늘 타일은 약 4 % 다.
- 이 표는 **L1 만** 본다. 남은 제약: L3 연속 구간(16–32 B, G16)과 청크 반복 ≤ 63(B 의 conv4_2/4_3 은 64 → 2 단계 청크로 해소), 64 B 정렬(G17; kw 복제 레이아웃이면 활성값 항이 약 3 배), L2 는 블록 단위 스테이징(행 청크 전체와 출력 join 을 한 memtile 에 두면 conv1_2 에서 512 KiB 초과), BD 전송 4 B 배수(C2). 모두 T0.3 이 검사한다.
- 패딩 낭비: conv5_x 는 OH·OW 14→16 으로 MAC 1.31 배, conv4_x 는 OW 28→32 로 1.14 배, conv1_1 은 IC 3→8 로 2.67 배(절대량은 작음).

### 5.1 P0 — 설계 전 데이터와 반증 스파이크 (M0)

| ID | 작업 | 내용 | 왜 | 의존 | 검증 | 크기 |
|---|---|---|---|---|---|---|
| T0.1 | 제약 표 · 실패 분류 · e2e 예산 · 머지 충돌 목록 문서 | `docs_Conv/MMDD_conv_constraints.md`: §3 의 C1–C12 를 출처와 함께(채널 6/2/2, BD 차원 3/4, 크기·stride 한도, stride-1 규칙, L1 = 65,536 − 1,024 와 depth 규칙, memtile BD 48, shim inter 반복 63·repeat 256(stride 0 만), 벡터 정렬 256/1024 bit, Peano 16 KB 프로그램 메모리, dispatch 당 hwctx). dev-conv 실패를 제약 위반 / 버그 / 프로세스로 분류. e2e 예산(origin/dev VGG 1.71 s ≈ hwctx 약 1.2 s + NPU 0.120 s + CPU 0.062 s + 나머지), 팀 목표 0.34 s, bias fold 전제(C13). 11 마커/5 파일 충돌 목록, N 64→128 | 이후 모든 작업의 점검 기준 | — | 문서 검토: 모든 행이 file:line / sha / 문서 인용 | S |
| T0.2 | **정적 AIE 제약 감사 스크립트** | `scripts/debug/aie_audit.py`. 읽는 지점(파이프라인 순서): `--mlir-print-ir-after=iree-amdaie-assign-channels`(타일별 DMA 채널 vs 6/2/2), `--mlir-print-ir-after=iree-amdaie-assign-npu-dma-bd-ids`(shim BD), `--mlir-print-ir-before=amdaie-assign-bd-ids`(memtile·코어 `aie.dma_bd` 수·차원·stride; 배정이 실패해도 덤프가 나옴), `--mlir-print-ir-after=iree-amdaie-lower-to-aie`(코어/memtile/shim/열 수, objFifo 별 lock 쌍과 producer·consumer 수, 다중 타겟 connection), `--mlir-print-ir-after=amdaie-assign-buffer-addresses`(코어 타일별 max(address + 크기) = 스택·정렬 여유가 든 실제 L1, 64 B 이상 코어 버퍼의 `address % 64 == 0`), `npu_instructions` 크기와 제어 코드의 shim DMA op 수, L3 접근의 연속 구간 분포. BD 전송 길이·주소 4 B 배수도 검사. `allocated buffers exceeded available memory`·`could not find and assign a valid BD id`·`An upper bound for the stack size ... insufficient` 진단은 위반으로 집계. 기대한 `IR Dump Before/After` 헤더가 없으면 비정상 종료(빈 덤프 통과 금지). `iree-amdaie-assign-bd-ids` 는 `--iree-amdaie-reprogram-dmas` 일 때만 돌아 기본 경로에서는 쓰지 않는다. 씨앗: 로컬 `log_shjo/e2e/{bdc,bdcount,chk,eqv}.sh`, `scripts/debug/condense_flow.py` | NPU 금지 정책에서 설계가 제약을 지키는지 패스를 쓰기 전에 증명하는 유일한 수단 | T0.1 | 골든 재생: im2col VGG conv dispatch(32 코어/8 memtile/8 열), npu4 CI conv 샘플(4 코어/1 열), `vgg_28x28x512` 의 `855c3168` 전/후 memtile BD(7224 vs 36), IC=256 depth 2 의 L1 진단(105,984 B) | M |
| T0.3 | **타일/L1/L2/L3/BD/채널/정렬 계산기** | `scripts/debug/conv_tile_calc.py`(§5.0 예비 산술을 확장): 13 VGG 형상 × 후보(G2 / G1 / 단일 열 / IGEMM) × (t_b, W, OC_t, IC_c, IC_L2, d_in, d_acc)에 대해 L1 식, 열당 L2 ≤ 512 KiB(블록 단위), 전송별 BD 차원(shim 3+1, memtile→코어 A `[t_b+2][IC_c/8][W+2][8]`, B `[9][IC_c/8][OC_t][8]`), memtile 채널 수, BD 수, **L3 연속 구간**, **청크 반복 ≤ 63**, **kw 시프트 정렬**(오프셋 kw·16 B 와 (행, ic_o) 블록 간격 (W+2)·16 B 가 **64 B** 배수인가 — AIE2P 512-bit 접근, `33c20094`; 아니면 kw 복제 레이아웃의 L1 비용), 버퍼별 64 B 정렬, 코어 내 epilogue 출력 항(유무·출력 타입을 인자로), BD 전송 4 B 배수, MAC/B vs 채널당 4 B/cycle(추정)·512 MAC/cycle, 나눗셈 조건 | 제약을 숫자로. dev-conv 의 L1 벽은 이 산술로 예측 가능했음 | T0.1 | 단위 assert: 52,480 B(IC 256, depth 1, 스택 제외), 105,984 B(IC 256, depth 2, 스택 포함), 105,728 B(IC 512, depth 1, 스택 포함), 가중치 슬랩 73,728 B. 세 실측은 `33c20094` 이전 값이지만 버퍼가 모두 64 B 배수라 그대로다. 새 브랜치에서 한 번 다시 잰다 | S |
| T0.4 | **13 층 기준 행렬과 VGG 컴파일 게이트**(`scripts/onnx` 기반) | 플래그는 `scripts/onnx/options.py` 를 정본으로 쓴다(`--conv` 는 T1.3 에서 추가). `run-onnx.sh compile` 과 `--compile-to=flow` 덤프로 dispatch 목록·affinity, NPU dispatch 수(origin/dev 기본 16 = conv 13 + FC 3, `--pool accel` 이면 20 — pool5 는 transpose 와 융합돼 CPU), lowering_config 히스토그램, ukernel 호출 수(matmul 13 + matvec 3), host pad/crop 수를 뽑는다. `scripts/debug/gen_conv_layer.py` 가 층별 (a) 명명 `conv_2d_nhwc_hwcf` bf16→f32(N=2, pre-padded) (b) im2col matmul 을 생성해 npu4 에서 컴파일 + 감사. origin/dev 레시피에서 im2col 만 뺀 Flow 덤프로 conv dispatch 에 bias+ReLU 가 융합되는지(3 입력) 확인. `docs_Conv/MMDD_vgg_layer_matrix.md` | 지금까지 5 형상만 측정됐고 레시피가 바뀌었다(no-fuse·detach 삭제, enable-pooling → accelerate-pooling) | T0.2, T0.3 | 컴파일 전용. (a) 는 IC ≥ 256 의 8 층이 L1 실패, (b) 는 13 층 통과 예상 | M |
| T0.5 | 실제 VGG conv 의 프런트엔드 확인 — **이번에 수행(§11)** | Flow: 13 conv 모두 6 iterator `[P,P,P,R,R,R]`, NHWC·HWCF 맵, bf16 입력, f32 zero-fill 누산; SAME pad 는 truncf 와 합쳐진 host elementwise, bias+ReLU 는 뒤 elementwise. 전처리 끝: 명명 `conv_2d_nhwc_hwcf` 13 개; 입력 체인 `tensor.pad`(5-D, 이미 NHWC) → `tensor.collapse_shape` → truncf generic → conv; 가중치 상수 → `linalg.transpose`(OIHW→HWCF) → truncf; init = bias `linalg.broadcast` → `linalg.transpose`. 이 관찰은 dev-POOL 플래그(no-fuse·detach + 미커밋 패치) 기준이다. origin/dev 레시피의 Flow 모양은 T0.4 에서 다시 본다. 남은 일: 두 덤프를 `docs_Conv/MMDD_vgg_frontend.md` 로 옮김 | dtype·IR 계약 고정, T2.4 패턴의 입력 | — | 완료(컴파일 전용) | S |
| T0.6 | **aievec kw 시프트 스파이크**(반증) | (a) 재빌드 없음: 컨테이너에서 `build/tools/iree-compile log_shjo/ci_conv_bf16.mlir --iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu4 --iree-amdaie-tile-pipeline=conv-decompose --iree-amdaie-enable-vectorization-passes=true --iree-amd-aie-peano-install-dir=/workspace/llvm-aie --mlir-print-ir-after=align-transfer-reads`. "can't be aligned with a read twice as large" 경고가 나오면 컴파일러가 kw 시프트 읽기 정렬을 보장하지 못한다는 뜻이고, kw=1 오프셋이 16 B 라는 주소 산술과 합치면 레이아웃을 바꾸지 않는 aievec 경로는 기각된다. (b) 증분 재빌드: `conv-npu` 곁가지에 `git cherry-pick -x 9b4f831a 17986af8`(origin/dev 에도 충돌 0, Peano v22) 후 `--mlir-print-ir-after=test-lower-vector-to-aievec` 로 `aievec.matmul` 존재·`vector.contract` 0·경고 0 **그리고** 512-bit 이상 lhs/rhs load 오프셋이 모두 64 B 배수일 때 통과(`align-transfer-reads` 는 32 B 어긋남을 경고하지 않으므로 IR 의 오프셋 식을 직접 보고, `core_*.elf` 를 Peano `llvm-objdump -d` 로 확인). 실패하면 kw 복제 레이아웃으로 같은 확인을 반복 | 커널 경로(D5)를 M0 에서 결정. 계획의 가장 큰 기술 내기 | — | 컴파일 전용; `docs_Conv/MMDD_aievec_kw_probe.md` | S |
| T0.7 | **G2 objectFifo 스파이크**(iree-opt, 재빌드 없음) | `Transforms/test/distribute_cores_and_objectfifos.mlir`·`split_logicalobjfifos.mlir` 를 본뜬 손 입력(target attr: num_cols 8, num_rows 4, npu4): 가중치 L2 `memref<64x3x3x8xbf16, 1>` 를 열 c 의 코어 4 개가 `[9][1][8][8]`(stride `[8, 8, 72, 1]`, offset 576·c)로 읽고, 활성값 L2 `memref<4x30x34x8xbf16, 1>` 의 r 번째 조각을 행 r 의 코어 8 개가 읽는다. `iree-amdaie-distribute-cores-and-objectfifos` → `iree-amdaie-split-logical-objectfifos` → `iree-amdaie-assign-channels` 순으로 실행. 기각 조건: 가중치 split factor ≠ 8, 어느 열 consumer 라도 조각 0 을 가리킴(`objFifoIndex = 0` 경로), 활성값 8 코어 읽기가 connection 8 개로 갈림, memtile S2MM > 6. 대조군: npu4 pack-peel bf16 32 코어 matmul 의 `-after=iree-amdaie-assign-channels` 덤프 | G2 의 세 가정(OC split, 다중 타겟 broadcast, 6/6 채널)을 XL 작업 전에 반증. split factor 는 consumer objFifo 수와 numCols 의 gcd 로 정해진다(`AMDAIESplitLogicalObjFifos.cpp:272, :276`) | — | iree-opt 출력 검토, 결과 문서화 | S |
| T0.8 | ~~dev-POOL 기준 산출물 저장~~ — **생략** | 팀이 dev-POOL 머지를 검증했고(ctest 232/232, npu4 CI 32/32, 회귀 27/27, `CHANGES.md:71, :167`), 11-03 에 곧바로 `conv-npu` 로 옮겼다 | — | — | — | — |

### 5.2 P1 — 재현 가능한 빠른 기준선 (M1)

| ID | 작업 | 내용 | 왜 | 의존 | 검증 | 크기 |
|---|---|---|---|---|---|---|
| T1.0 | **빌드·NPU 락 스크립트 커밋** | 지금 미커밋인 `scripts/lock/`(미추적), `scripts/build/build.sh` 의 `lock_run` 래퍼(origin/dev 판 build + build-trace 를 락 안에서, build-trace 는 구성돼 있을 때만), `scripts/docker/run-{dev,debug,deploy,onnx}.sh` 의 락 마운트·npu.lock 래퍼(와 관련 문서)를 경로 명시 `git add` 로 영어 메시지 커밋. 같은 변경을 origin/dev 로 PR(팀 빌드도 같은 락을 잡아야 직렬화가 성립) | G18: CLAUDE.md §1 의 호스트 빌드 직렬화가 머지 순간 조용히 사라짐 | — | 커밋 뒤 빌드 중 `scripts/lock/status.sh` 가 build.lock 보유를 표시 | S |
| T1.1 | ~~submodule 패치 커밋 + 업스트림 PR~~ — **생략** | origin/dev 레시피는 패치가 넓히는 no-fuse 플래그를 쓰지 않고(`38613bbb`), 팀은 그 fork 옵션을 지울 예정이다(`CHANGES.md:98, :100`). 필요하면 비교 실험 동안만 로컬 적용(커밋·bump·PR 없음) | G02 | — | — | — |
| T1.2 | **`conv-npu` = origin/dev 에서 분기** — **완료(11-03)** | `git fetch origin dev:dev`(빨리 감기) → `git switch -c conv-npu dev`. 로컬 락 수정 6 파일은 stash 후 dev 의 새 내용 위에 다시 얹음. `git submodule update --init third_party/iree third_party/aie_api`. Peano v22 설치(v19 는 저장소 밖에 보존). dev 이미지 재빌드, `configure.sh`(Vulkan, `build-trace/`), `build.sh` 로 `iree-compile`·`iree-run-module`·`iree-lld`·`IREECompilerPythonModules` 와 trace 런타임 빌드. 자세한 기록은 [docs/1103_experiment_env_setup.md](../docs/1103_experiment_env_setup.md) | dev-POOL 은 팀 `7060d5f3` 로 dev 에 포함 | T1.0(락은 로컬 수정으로 유지) | 빌드 성공(ccache 적중 88 %). NPU 없이 MLP CPU·CPU+GPU e2e 와 `--verbose`, MLP·VGG 의 CPU+NPU 컴파일 통과. `ctest -R amd-aie` 는 남음 | L |
| T1.3 | **VGG ukernel 컴파일 게이트 + `--conv` 스위치** | `run-onnx.sh compile models/vgg16/vgg16-12.onnx --numeric bf16`: conv matmul 13 개가 `matmul_bf16_bf16_f32_<MxNxK>_8x8x8` + `zero_fill`, FC 3 개가 `matvec`(K tile 256; 128/192 는 스택 초과로 금지, `7854b375`)을 `ukernel_instance` 로 호출하는지, pool·GEMV 코어 스택 진단 0 건인지 본다. origin/dev `options.py` 에 `--conv {im2col,native}` 를 추가한다(im2col 이 고정 graph 플래그라 `--extra-flag` 로는 뺄 수 없다. `--pool` 은 이미 있다). 스크립트는 `PEANO_INSTALL_DIR`(기본 `/workspace/llvm-aie`)로 Peano 를 고른다. npu4 CI 부분집합 컴파일 전용(`--do_not_run_aie`) | 기준선을 컴파일 측에서 먼저 증명; 이후 모든 컴파일의 플래그 단일 출처 | T1.2 | 정적 IR: matmul 13 + matvec 3 호출; `<dir>/<entry>/matmul.o`·`ukernel_include/aie_api/` 생성(aie_api embed 확인); ctest; `--conv im2col` vmfb 생성. `--conv native` 는 T5.4 전까지 `--compile-to=flow` 까지만 | M |
| T1.4 | `.gitignore` 정리 커밋 | `log_shjo/`, `docs_shjo/`, `pass_dump/`, `build_shjo/`, `build_root_bak/`, `*.npy`, `sources.mlir`, `Output.rootbak.*/` 무시(docs_Conv/0908_merge_plan.md §7); `git add -A` 금지 유지 | dev-conv 문서 커밋 2 개가 스크래치를 포함(`34a40674`, `54d37b34`) | — | `git status` 에 스크래치 없음 | S |
| T1.5 | **(승인) NPU 배치 B1**: im2col + ukernel 시간·정확도 | `./scripts/docker/run-onnx.sh e2e models/vgg16/vgg16-12.onnx --numeric bf16 --runs 1 --warmup 0 --verbose` 를 전원 모드 2 개(default, performance)로 = **총 2 회**(+ `--verbose` 의 trace 실행 각 1 회), npu.lock 아래에서. FC 파이프라인 스위치(`--iree-amdaie-enable-gemv-pipeline`)는 `7d49efaa` 로 삭제돼 FC A/B vmfb 는 만들 수 없다(팀 측정 2.19 → 1.62 s 로 대신). `tests/regression/run_regression.sh` 는 제외 | 하드웨어 전용 결함은 컴파일로 안 보임; 기준 바(origin/dev ~1.7 s, NPU ~0.12 s, corr ≥ 0.9999, top-1 892) | T1.3 | 소유자 승인(D9) 시에만; 아니면 M1 은 컴파일 전용으로 닫고 "하드웨어 미검증" 표기 | S |

### 5.3 P2 — 네이티브 경로 일반화: 인식 · 라우팅 · 타일 결정 · 전처리 (M2)

T3.1 슬라이스(§5.4)가 먼저 구조를 증명하고, 이 단계가 그 결과를 일반화한다. 표는 실행 순서대로 적었다.

| ID | 작업 | 내용 | 왜 | 의존 | 검증 | 크기 |
|---|---|---|---|---|---|---|
| T2.2 | **indexing map 기반 conv 인식 + 공유 `ConvInfo`**, 조기 거부, **공존 규칙** | `AMDAIEUtils.cpp:564-609` 의 iterator 전용 판정을 맵 검사로: 입력 `(n?, oh+kh, ow+kw, ic)` unit stride/dilation, 가중치 `(kh,kw,ic,oc)` 또는 `(oc,kh,kw,ic)`(`ConvLayout`), 출력 `(n?,oh,ow,oc)`, batch 선택적. `ConvInfo {차원 위치, layout, hasBatch}` 를 conv root config·Flow pad·affinity 가드가 소비. 거부 진단과 `conv` override 는 `--iree-amdaie-enable-native-conv` 가 켜지고 npu4·bf16·NHWC·3×3·stride 1 일 때만 적용하고, 그 밖(npu1_4col, i8/i32, depthwise, 전역 `--iree-amdaie-tile-pipeline=conv-decompose`)은 기존 conv-decompose 를 그대로 탄다 | G04/G08/G20 | T0.5 | lit `lowering_strategy_conv.mlir` 기존 3 케이스 불변 + npu4 케이스(6-D generic, 7-D N=4 FHWC generic), 플래그 on 에서 NCHW·stride-2 expected-error | M |
| T2.1 | **`conv` tile-pipeline override, 함수별 분기(벡터화 포함), dev-conv 커밋 선별** | `KernelDispatch.h`: `kConvPipelineName = "conv"`; conv root 에 `amdaie.tile_pipeline = "conv"` 를 어느 전역 파이프라인에서도 부여(isMatmul 을 isConvOp 보다 먼저 보는 순서 유지, `7d3eb3e1`). `AMDAIELowerExecutableTarget.cpp:94-101` 에 `conv` 분기 + funcOp attr(`dd8f5c06` 의 `amdaie.conv_decompose` 대체). `Passes.cpp:958-960, :998-1001, :1109-1110, :1135-1136, :1210` 와 **벡터화(`:978-986`)** 가 `useTilePipeline` 대신 func attr 를 읽음: conv(·pack-peel) 함수만 벡터화하고 gemv/pool 함수는 건너뜀. `855c3168` 그대로 이식(origin/dev 에도 충돌 0). 팀은 문자열 override 를 `TilePassPipeline` enum 으로 옮기는 정리를 남은 일로 적었으므로(`CHANGES.md:201`), `conv` 를 enum 값으로 넣을지 그 정리 PR 과 순서를 맞출지 먼저 합의한다. `getTilePipelineForRootOp`·aie shim 헤더 2 개·softmax 라우팅(`69ec5c13`)·전역 depth 1(`780c4acd`)은 제외 | G04/G19: VGG 는 conv root·FC GEMV·pool·pack-peel matmul 이 섞임; 모듈 수준 패스에 함수별 마커 필요 | T1.2, T2.2 | 새 lit `lowering_strategy_conv_routing.mlir`(한 모듈: 6-D conv generic → conv, 2-D matmul → pack-peel, M=1 → gemv, pooling → pool; npu4); 모든 `lowering_strategy_*.mlir`; 혼합 모듈 compile rc=0; 벡터화 on/off 에서 GEMV·pool 코어의 `-after=iree-amdaie-lower-to-aie` 덤프 동일; `Test/samples/conv2d_nhwc_objectfifo_e2e.mlir` 통과; run.py conv 3 종(npu1_4col, `--do_not_run_aie`) AIE IR 불변; `vgg_28x28x512` memtile BD 36(`-before=amdaie-assign-bd-ids`) | M |
| T2.3 | **`chooseConvTiling`**: 전처리·Flow 패딩·root config 가 공유하는 예산 탐색 1 개 | `Utils/AMDAIEUtils.{h,cpp}`: `struct ConvTiling {grid, cols, rows, t_b, W, OC_t, IC_c, IC_L2, d_in, d_acc, padMultiples}`; 규칙 R1 L1 식(§5.0, 버퍼별 64 B 정렬과 epilogue 출력 항 포함) ≤ `getCoreTileLocalMemorySize() − stackSize`(비교는 `<=`; 기존 `gemvL1Bytes`·`AMDAIETileSizeSelectionUtils` 는 정렬 여유가 없어 그대로 쓰지 않는다); R2 BD 차원(shim 3+1, memtile→코어 4-D); R3 innermost stride 1 + BD 전송 길이·오프셋 4 B 배수; R4 열 memtile 채널 ≤ 6/6, 코어 S2MM = A + B, shim MM2S 2; R5 broadcast 만(G2); R6 8 배수·약수; R7 memtile BD ≤ 48; R8 열당 L2 ≤ 512 KiB(블록 단위); **R9 L3 연속 구간 ≥ 128 B 또는 청크 우선 레이아웃**; **R10 각 L3 전송이 블록·청크 루프까지 shim intra 3 + inter 1(≤ 63) + repeat(≤ 256, stride 0 만) 안에 접힘**; **R11 kw 시프트 정렬**(aievec 경로면 512-bit 이상 읽기 오프셋이 모두 64 B 배수: kw 복제 레이아웃, 행 폭 W+4 패딩은 블록 간격만 해결; ukernel 은 정렬 로드 + 레지스터 시프트; IGEMM 은 해당 없음); **R12 코어 타일 퇴화 금지**(코어당 출력 ≥ 2, 서로 다른 피연산자의 L1 타일이 같은 memref 타입이 되지 않게, `33b86238`). 목적함수: MAC/B 최대 → L3 연속 구간 최대 → shim 바이트 최소. 플래그 `--iree-amdaie-conv-grid-policy`, `--iree-amdaie-conv-columns`, `--iree-amdaie-enable-native-conv`(기본 off) | G05/G07/G08/G16/G17: GEMV 는 `0b80251e`·`8df29041` 로 형상·K 규칙을 pad 패스와 KernelDispatch 가 공유하게 됐고 남은 TODO 는 Flow 가 전역 tile pipeline 을 못 보는 것뿐이다(origin/dev `KernelDispatch.cpp:747-752`). 반면 Flow 에서 L1 예산으로 타일을 다시 고르던 `chooseGemvNPad` 는 과패딩으로 `14c757a2` 에서 삭제됐다. 그래서 conv 도 Flow 패딩은 기하에서 나온 배수만 쓰고 예산 탐색은 KernelDispatch 가 한 번 한다. 정책 값은 target config 한 경로로만 전달한다(팀이 pool 정책의 3 경로 전달을 부채로 기록, `CHANGES.md:202`) | T0.3, T2.2, T3.1 | 표 기반 lit `lowering_strategy_conv_npu4.mlir`(13 형상, 기대값은 `conv_tile_calc.py`); 선택 타일이 R1–R11 통과 | M |
| T2.4 | **conv Flow 준비**(가중치 사전 pack, IC/OW/OH 패딩, OH halo pre-split, 청크 우선 활성값) — 전처리 또는 Flow 패스 | 기본 위치는 origin/dev 의 Flow 단계(affinity → **fold(T5.1)** → **pad** → pool pad, `PluginRegistration.cpp:104-132`)다: 가중치는 `getConstantValue`·`createReplacementConstant`·compile-time 상수 pad 로 FHWC 또는 열·IC 청크 우선(`[OC/OC_col][ICp/IC_L2][OC_col][3][3][IC_L2]` 또는 L1 타일 순서)으로 사전 pack(런타임 dispatch 0, const-eval 의존 없음); 입력 IC 3→8, OW 8 배수, OH numRows 배수 패딩과 `PoolingNhwcMaxHaloSplit` 방식 halo split(N=numRows)은 host dispatch 로. 새 helper 접두어는 `firstFreeHelperIndex` 에 더한다. 전처리 패턴을 쓴다면 루트는 `linalg.conv_2d_nhwc_hwcf`(bf16 입력, init = bias broadcast → transpose)이고 입력 producer 를 truncf generic → `tensor.collapse_shape` → `tensor.pad`(5-D NHWC) 순으로 따라간다(§11). dev-conv 의 `getConvInfo`/`padConvDispatch` 는 옛 dispatch 안 `tensor.pad` 를 써서 cherry-pick 하지 않고 이식한다 | G07/G08/G16: 행별 L2 조각이 서로소, 형상 패딩이 host op, L2 의 최외곽 축이 OC 라 수정 없는 split 패스가 OC 를 분배. 팀은 split 패스를 N 위치로 일반화하거나 상수 가중치를 전치하는 방안을 남은 일로 적었다(`CHANGES.md:199`) — 방식은 곽준호 님과 합의해 GEMV [K,N] 과 함께 쓴다 | T2.2, T2.3, T5.1 | lit(IC=3, OH=14, OH=56 층); VGG `--compile-to=flow`: conv executable 13 개 피연산자 레이아웃, 가중치 전치 dispatch 0. host 변환(IC pad, halo split, 청크 우선 레이아웃)을 기존 pad·truncf dispatch 에 합칠 때 llvm-cpu 가 받는 형태인지 먼저 확인(inner dim 이 1 늘어나는 `tensor.pad` 는 llvm-cpu tiling 실패 `710d6b0e`, 두 pad 를 합치는 세 방식도 거부 `CHANGES.md:91`). '층당 host 보조 dispatch ≤ 2' 는 ones 채널·IC 배수 pad 를 세어 다시 정한다 | L |
| T2.5 | **fail-fast 가드 + split 패스 `objFifoIndex` 가드** | origin/dev 는 fail-fast 를 fold 패스에서 한다(접을 수 없는 bias 는 컴파일 오류, `CHANGES.md:16`). T5.1 전까지는 fold 매처를 conv generic 으로 넓혀 'conv bias fold 미구현' 오류를 내고, T5.1 뒤에는 접고 남은 consumer 가 `isUnaryElementwiseEpilogue`(`28bdc408`) 일 때만 NPU 에 둔다. affinity 판정(origin/dev `AMDAIEAssignDeviceAffinities.cpp:108-132`)은 융합 op 와 무관하게 conv 를 모든 가속기(vulkan 포함)에 round-robin 하므로, NPU 전용 conv 형태를 amd-aie 에만 보내는 장치별 선택을 pass 옵션으로 둔다(`accelerate-pooling`·`placement-map` 과 같은 방식; 팀 남은 일 '장치별 op 선택' `CHANGES.md:197` 과 합침). `getNewOffsetConfig`(`AMDAIELogicalObjFifoSplittingUtils.cpp:673`): splitFactor > 1 인데 consumer 가 index 0(또는 bare loop-IV block arg)으로 풀리면 에러(corr 0.111778 사례) | G10/G12: 'fill 뿐' 규칙이면 origin/dev 레시피에서 VGG 네이티브 conv 13 개가 모두 걸린다; 조용히 틀린 열 매핑에 대한 유일한 방어 | T1.2 | lit `fold_bias_into_contraction_dispatches.mlir` 에 conv + bias 케이스(T5.1 전 expected-error, 뒤 fold); `split_logicalobjfifos.mlir` 부정 케이스; ctest | S |
| T2.6 | **IGEMM 스파이크**(설계 대안) | 플래그 `--iree-amdaie-conv-igemm`: conv root 에 대해 `iree-amdaie-lowering-strategy` 앞에서 IREE 의 `convertToIGEMMAndSetConfig`(configFn = pack-peel root config)를 적용. conv3_1·conv4_2 를 npu4 compile-only 로 돌려(conv1_1 은 IC 를 4 이상으로 패딩해야 BD 전송이 4 B 배수가 된다) tiled `iree_linalg_ext.im2col` 이 막히는 첫 패스를 기록하고, 필요한 새 하강(im2col 타일 → `amdaie.dma_cpy_nd`: K 타일 = 한 (kh,kw) 탭 안의 IC 연속, M 타일 = 출력 한 행 안; 입력 슬랩은 L2 에 두어 L3 읽기 ≈ 1 배 + halo)의 작업량을 산정 | G21 → D13. 엔지니어링이 가장 작은 네이티브 경로일 수 있음: pack-peel + bf16 ukernel + 융합 ReLU(누산기 depth 1)는 origin/dev 에서 하드웨어 검증됐다. 단 Flow 에서는 conv generic 이므로 conv bias fold(T5.1)는 공통 선행이고, IGEMM 의 contraction 이 pack-peel 의 `isMatmul` 판정을 통과하는지는 미검증 | — (dev-POOL 에서 가능; ukernel 결합은 T1.2 뒤 재확인) | 컴파일 전용; 결과 문서 | M |

### 5.4 P3 — 네이티브 하강 (M3 → M4)

| ID | 작업 | 내용 | 왜 | 의존 | 검증 | 크기 |
|---|---|---|---|---|---|---|
| T3.0 | **패스 결함 선수정** | `git show dev-conv:docs_Conv/wip/0903_p3_restructure_wip.patch` 의 `AMDAIEDistributeL1Allocations`(fill/pack/unpack 의 L1 alloc 쓰기, 상수 offset rank-reducing subview, 같은 rank subview 우선, rank-reduced 타입 재추론), `AMDAIEInsertCores`(사용자가 모두 코어 안인 `memref.subview` 이동), `AMDAIEPeelForLoop`(최외곽만), `AMDAIEFuseConsumerIntoLoop`(affine.min 접기) 수정을 hunk 별 lit 와 함께 먼저 적용. (gecv-k1fold `502af3ab` 의 `distributeWholeBufferInitPack` 은 비교하지 않는다: origin/dev 는 누산기를 fill(0) 으로 바꿔 bias L2 버퍼가 없으므로 불필요하다고 판단했다, `CHANGES.md:47`.) 사전 확인(재빌드 없음): `addGemvPassPipeline` 의 패스·옵션 순서를 `iree-opt --pass-pipeline` 텍스트로 손으로 쓴 7-D N=4 FHWC conv generic(손 lowering_config)에 적용해 처음 실패하는 패스를 기록 | dev-conv 가 같은 구조에서 실패 사슬을 이미 문서화함(fill 로직, packed rank, `amdaie.connection` 단일 circular user, `amdaie.flow` 다중 소스, InsertCores subview). 이 수정 없이는 T3.1 이 검증을 통과할 수 없음 | — | 수정별 lit; ctest; pack-peel/GEMV/pool lit 불변 | M |
| T3.1 | **단일 형상 수직 슬라이스**: `addConvPassPipeline` | 먼저 conv5_1, 다음 conv1_2. 단일 dispatch 모듈에 전역 `--iree-amdaie-tile-pipeline` 새 값(`conv`)을 추가하고 타일은 `conv_tile_calc.py` 출력으로 손 설정(라우팅·타일 결정 일반화는 P2). 파이프라인(`addGemvPassPipeline` 본뜸): L0 블록 forall(N 청크 × OC 블록, cols=1 먼저) → `AMDAIEPad` nofold 복사 → 출력 L2 승격 → reduction 타일링 전에 C 의 두 번째 복사 단계 `useInitAsResultDest`(누산기 L1 상주) → IC 청크 `scf.for` → `FuseProducerIntoLoop` → 입력 청크 L2 승격 → 코어 forall(N → 행 DimY, OC → 열 DimX) → `FuseFillIntoForall` → `PackAndTranspose` level 0 는 IC 만 8 로 pack, `packTransposeOnSource = true` → L1 승격 → 코어 안 `scf.for` KH, KW(shifted subview) → `LinalgFoldUnitExtentDims` → `(ow, oc, ic_i)` 꼴 generic → `addPeelAndFusePasses`(첫 청크에 fill, 마지막에 write-out) → bufferize. **L1 objFifo 별 depth 옵션**(입력 2, 누산기 1) 추가 — 코어 내 epilogue(T5.2)와 함께 필수(§5.0). 누산기 L1 레이아웃은 2-D `[t_b·W][OC_t]` 또는 pack 4-D 로 둔다(zero_fill ukernel 은 rank 2/4/6 fill 만 받는다, origin/dev `AMDAIELowerToUKernels.cpp:268-271`). IC 청크 누산을 ukernel 로 할 경우 GEMV 의 matvec 경로(LowerToUKernels 가 `addGemvPassPipeline` 안에서 돈다, origin/dev `Passes.cpp:444`)를 본뜬다. 1 열 → 8 열 | G05. 라우팅·타일 결정·전처리(P2)는 이 결과로 일반화 | T3.0, T0.3, T0.7 | npu4 컴파일 전용: 두 형상 rc=0; 감사(L1 = 계산기 예측, memtile BD ≤ 4-D, 코어/shim ≤ 3-D, stride 1, 채널); lit `conv_pipeline.mlir`(블록·코어 forall 사이 IC 청크 `scf.for` 1 개, fill 은 첫 반복, write-out 은 마지막); matmul/GEMV/pool 샘플 불변; `--ukernel all` 에서도 conv 타일이 matmul ukernel(rank 4/6 전용)에 잡히지 않음을 lit 로 확인 | XL |
| T3.2 | T3.1 이 새로 드러내는 결함 수정 | T3.0 에 없던 결함만, 수정별 lit 와 함께. 임시 진단은 `// XXX TEMPORARY DIAGNOSTIC -- REMOVE` 로 표시하고 확인 직후 제거 | 숨은 결합은 개별 적용·시험으로만 드러남 | T3.1 | 수정별 lit; ctest | S |
| T3.3 | **8 열 × 4 행 (G2)**: split·broadcast·채널·BD·lock 의 정적 검증 | 파이프라인 순서대로 확인: `distribute-cores-and-objectfifos` 뒤 코어 32 개와 L2 objFifo 의 consumer 집합 → `split-logical-objectfifos` 뒤 가중치 OC 8-way·활성값 N 4-way 조각과 조각별 consumer(가중치 = 그 열 4 코어, 활성값 = 8 코어, connection 1 개 다중 타겟) → `assign-channels` 뒤 memtile ≤ 6/6·shim ≤ 2 MM2S → `-before=amdaie-assign-bd-ids` 에서 memtile BD ≤ 4-D·≪ 48 → `lower-to-aie` 뒤 출력 join 의 producer 별 lock 쌍(`dd16b81c` 연속 순서 경우)과 broadcast 의 consumer DMA 1 개(lock init = depth). 첫 단계는 T0.7 결과를 실제 파이프라인 출력으로 재확인. 동일 슬라이스에 코어별 DMA 가 생기면 `AMDAIEDistributeCoresAndObjectFifos` 에 connection 병합(+M) 또는 T3.4. 열당 제어 코드 루프 수(블록 × IC 청크)와 반복 한도도 감사 | G07: matmul 을 회귀시킨 split 축 재작성 대신 R4/R7 안에서 코어 8 배 | T3.1, T2.5, T0.7 | conv1_2·conv4_2·conv5_1 의 IR 덤프·감사 표 보관; lit `conv_pipeline_columns.mlir` | XL |
| T3.4 | **Fallback**: Flow 수준 IC 분할 + host add | T3.1 청크 스트리밍이 BD/채널 제약에서 멈출 때만: 새 Flow 패스 `AMDAIESplitConvInputChannels`(Flow 순서 affinity → fold → pad 안에 둔다. N-split 패스·`createConcatDispatch` 는 `7d49efaa` 로 삭제됐으므로 청크 분할과 부분합 host add 를 새로 작성하고, 참고 구현은 `git show 9c6ea263:…/AMDAIEPadContractionDispatches.cpp`): L1 footprint 초과 시 IC 를 같은 청크로(512 → 2×256) host 빌더로 분할, 청크당 conv dispatch 가 f32 부분합을 쓰고 host add dispatch 로 합친다(origin/dev 에는 합칠 CPU bias/ReLU dispatch 가 없다. 부분합을 matmul 형 accumulator 로 넘기면 fold 가 오류를 낸다, C13) | 저위험으로 G05; conv4_x/conv5_x 를 네이티브로 컴파일 | T3.1 | lit `split_conv_input_channels.mlir`; 네이티브 행렬 13/13; 추가 부분합 트래픽·dispatch 수(hwctx 비용) 기록 | M |
| T3.5 | **13 층 네이티브 컴파일 행렬**(npu4, 1 열 → 8 열) | 일반화된 경로(P2)와 T3.3 뒤 `gen_conv_layer.py` + 감사 재실행: rc, 실패 패스, L1 바이트 vs 예측, 코어/memtile/열, BD 수·차원, 채널, lock fan-out, 이동 바이트, L3 연속 구간; `14x14x512` 의 0901 vs 0903 불일치 정리; `docs_Conv/MMDD_native_conv_table.md` | M3/M4 증거; dev-conv 의 5 형상 표 대체 | T3.3, T2.3, T2.4 | 컴파일 전용; 13 층 rc=0, 감사 green, 코어 32/memtile 8/열 8 | S |

### 5.5 P4 — 벡터화 · lock 안전 · 수치 명세 코어 (M5)

| ID | 작업 | 내용 | 왜 | 의존 | 검증 | 크기 |
|---|---|---|---|---|---|---|
| T4.1 | **AIE2P bf16 aievec 경로 + 코어 반올림 모드 cherry-pick**(중복 lock 수정 제외) | 1 단계(T0.6 (b) 와 같음): `conv-npu` 곁가지에서 `git cherry-pick -x 9b4f831a 17986af8`(origin/dev 에도 텍스트 충돌 0). 2 단계: Peano v22 에서(origin/dev 의 핀·코어 opt 플래그는 2c5763b8 과 같다) aievec lit 과 npu4 bf16 8×8×8 matmul 1 개를 ELF 까지 컴파일 — `9b4f831a` 가 `XLLVMOps.td` 에 묶는 Peano intrinsic(`vshuffle`, `v32bf16.to.v32accfloat`, `v64accfloat.to.v64bfp16ebs8` 등)의 이름·시그니처와 `-unroll-full-max-count=0` 아래 `.text` ≤ 16 KB·스택 ≤ 1,024 B 확인, 실패하면 바인딩 포팅. `d2846c0d` 는 건너뜀(origin/dev `6499776a`/`dd16b81c` 와 같은 함수). 반올림 수정 2 종은 합성 가능(`cff1e56e` 는 종료 시 모드 복원) | G06. 팀의 2.1–3.1 s / corr 0.99997 은 Peano v19 수치이고 v22 와 결합한 브랜치는 없음 | — (1 단계), T1.2 (2 단계) | aievec lit(`test_shuffle.mlir`), `core_rounding_mode.mlir`, ctest; VGG im2col `--iree-amdaie-enable-ukernels=none --iree-amdaie-enable-vectorization-passes=true` 컴파일 성공; `--iree-amdaie-core-rounding-mode=none` 으로 컴파일한 ukernels=all vmfb 가 cherry-pick 전과 md5 동일(기본 auto 는 모든 npu4 코어 진입에 crRnd 를 넣음) | M |
| T4.2 | **벡터화 conv 코어 본문**: 프로브 → 본문 → 정적 asm | 덤프 지점: `iree-amdaie-insert-loops-for-vectorization`, `iree-amdaie-vectorization`, `canonicalize-vector-for-aievec`, `align-transfer-reads`, `test-lower-vector-to-aievec`(aievec 변환은 `lower-to-aie` 뒤 `addMLIRAIELoweringPasses` 에서 돎, `Passes.cpp:1125`). 확인: 최내 generic 이 `(ow, oc, ic_i)` 맵이라 `isMatmulTransposeB`(`AMDAIEUtils.cpp:485-510`; `AMDAIEInsertLoopsForVectorization.cpp:192-206`)가 8×8×8 로 타일; lhs/rhs `vector<8x8xbf16>`, acc `vector<8x8xf32>`, `aievec.matmul`; **`align-transfer-reads` 경고 0 그리고 512-bit 이상 load 오프셋이 모두 64 B 배수**(경고가 없어도 32 B 어긋남은 남을 수 있다, C12); asm 은 `--iree-hal-dump-executable-intermediates-to=<dir>` 의 `core_<col>_<row>.elf` 를 Peano `llvm-objdump -d` 로(asm 파일은 생성되지 않음), 누산기 spill, `.text` < 16 KB(`llvm-size`); B 타일 방향(`[oc][ic_i]` = Bᵀ)이 `createAie2PBf16MatmulViaBfp16` 의 레지스터 내 transpose 와 맞는지 | G06/G17; D5 확정 | T4.1, T0.6, T3.1 | 정적 IR + asm: 코어당 `aievec.matmul` 수 = `W/8 · OC_t/8 · 9`/청크; `vector.contract` 0; 정렬 경고 0; im2col vmfb 불변; `docs_Conv/MMDD_conv_aievec_check.md` | M |
| T4.3 | **consumer 별 lock 쌍 — origin/dev `6ef3dc1f` 로 구현됨. conv 레이아웃이 조건을 만족하는지 감사** | conv 의 1→N distinct fan-out(T5.6 G1 변형 등)이 서로소·연속·오름차순 슬라이스로 내려가는지 `--mlir-print-ir-after=iree-amdaie-objfifo-bufferization` 에서 확인(lock 쌍 수 = consumer 수). producer 쪽 memtile BD 는 consumer 수 × depth 로 늘어나므로 R7(BD ≤ 48)에 포함. 인터리브 레이아웃이면 L2 레이아웃을 바꾸거나 일반화를 팀 남은 일과 협의. lock 사용량 vs `getNumLocks` 기록 | G09 잔여 | T1.2 | 기존 lit(`@multi_consumer_contiguous`, `@multi_consumer_interleaved`) + conv 레이아웃 케이스 1 개; 감사 | S |
| T4.4 | **조건부**: npu4/peano 용 bf16 3×3 conv ukernel(aie_api) | T0.6/T4.2 no-go 또는 변환·spill 바운드 측정 시에만: `uKernels/npu4/peano/conv2d_bf16.cc` 를 **aie_api 로** 작성한다. origin/dev `matmul.cc`(mlir-aie `mm_aie2p.h` 의 `matmul_vectorized_2x2_mmul`: `aie::mmul<8,8,8,bfloat16,bfloat16,accauto>` + `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, 2×2 누산기 블록, `aie::swap_rounding(conv_even)`/복원, `a6874c90`)를 kh·kw·IC 청크 루프로 감싸고, kw 시프트는 커널 안에서 정렬 로드 + 레지스터 시프트로 처리(64 B, C12). `<aie_api/aie.hpp>` 를 include 하면 XCLBinGen 이 헤더 추출·`-std=c++20` 등 mlir-aie 플래그를 붙인다(`286008ec`). 구현 틀은 matvec(`69a5e8b7`): `Target/CMakeLists.txt` embed + `kNpu4Templates`(origin/dev `AMDAIEUtils.cpp:109-116`) 항목 + `ukernel_instance`(`92736f26`) + L1 수준 conv generic 매처 + conv 파이프라인에 `LowerToUKernels` 추가(conv-decompose 는 지금 안 부른다). 팀 원칙: upstream mlir-aie `aie_kernels` 우선, 없을 때만 aie_api 로 직접; unroll 은 `AIE_LOOP_*` 매크로, 스택 1,024 B·프로그램 16 KB 를 넘을 때만 끄고 주석(`CHANGES.md:112-113`). f32 누적 커널은 스택이 upstream 의 약 2 배라(matvec K tile 128/192 → 1,024/1,088 B 초과) 청크·타일을 스택 ≤ 1,024 B 로 제한. 먼저 내부 루프 3 변형을 컴파일해 정적 명령 히스토그램·`--stack-sizes` 비교 | 팀 방향은 ukernel 우선이고 upstream aie_kernels·aie_api 에 bf16 3×3 conv 가 없어 직접 작성 대상 | T4.2 또는 T0.6 no-go | 13 형상 컴파일 + ELF 링크; 코어 ELF 스택 상한 ≤ 1,024 B(`XCLBinGen.cpp:1044-1052` 진단 0); lit lower_to_ukernels conv 케이스; MAC 스칼라 에뮬레이션 호스트 C++ 단위 테스트; 수치는 승인 시에만 | XL |
| T4.5 | **정확도 정책 + CPU bfp16 에뮬레이터 골든 파일** | `tools/bfp16_emulator/bfp_strict_v3.py` 를 origin/3devices-f32-aie2p-bf16 에서 이식; 프로브 형상별 conv 타일 1 개의 기대 출력 생성; 바(D10: corr ≥ 0.9999 + top-1/top-5 일치 + 층별 최대 절대 오차 vs CPU)를 `docs/WORKING_AGREEMENTS.md` 에. 골든 파일에 bias fold 의 bf16 반올림(VGG corr 0.99998 → 0.99996, `CHANGES.md:89`)과 NPU bf16 변환의 RTZ 를 넣는다 | G15; 하드웨어 없이 유일한 수치 기대값 | T4.1 | 에뮬레이터 자체 테스트(CPU 만) | S |

### 5.6 P5 — epilogue 와 모델 (M6)

| ID | 작업 | 내용 | 왜 | 의존 | 검증 | 크기 |
|---|---|---|---|---|---|---|
| T5.1 | **네이티브 conv 의 bias: origin/dev fold 패스의 conv 형태**(세 번째 DMA 피연산자 없음) | origin/dev 의 Flow 패스 `AMDAIEFoldBiasIntoContractionDispatches`(affinity 와 pad 사이, 무조건 실행)에 conv 형태를 더한다. upstream detach 가 만드는 `fill(0) → conv generic → generic(addf(conv, bias), …)` 를 매칭하고, 가중치 `[3,3,IC,OC]` 에 채널 하나를 덧붙여 bias 를 한 탭에 둔다(`appendBiasRow` 의 4-D 판 + `createReplacementConstant`), 입력에는 1.0 채널을 붙인다(`createPaddingDispatch(…, padValue = 1.0)`). bias 탭은 중심 (kh,kw) = (1,1) 을 권고한다(항상 내부 위치를 읽어 SAME pad 순서와 무관). IC+1 의 패딩 배수는 8 이 아니라 `chooseConvTiling` 의 IC_c·IC_L2 와 같아야 한다(`07d117d1`: K+1 을 작은 배수로만 패딩하자 GEMV 가 작은 K 타일로 떨어졌다). 정적 산술(13 층 MAC 가중): IC+1 → 8 배수 +4.8 %, 16 배수 +9.7 %, 64 배수 +39 %. 지금 패스는 명명 `linalg.matmul` 하나만 찾으므로(`:69-81`) 네이티브 conv dispatch 는 오류 없이 3 입력으로 남는다. `502af3ab` 의 Preprocessing fold 와 옵션은 팀이 넣지 않기로 했다(`CHANGES.md:46`) | G10: 코어 S2MM = A + B 라 bias 스트림 불가 | T2.2 | lit `fold_bias_into_contraction_dispatches.mlir` 에 conv 케이스(6-D generic, 7-D N=4); VGG Flow 덤프에서 conv dispatch 13 개에 bias 바인딩 없음; 이종 컴파일에서만 확인(단일 장치에서는 fold 가 돌지 않는다, C13) | M |
| T5.2 | **코어 내 ReLU + f32→bf16 epilogue**(C 타일) | (i) dispatch 형성 예외는 필요 없다: origin/dev 레시피는 no-fuse 를 쓰지 않아 IREE 기본 fusion 이 epilogue 를 conv dispatch 에 붙인다. 판정은 `isUnaryElementwiseEpilogue`(`28bdc408`; conv 결과 `[OH,OW,OC]` 에도 그대로 맞음)를 Flow pad·root config·affinity 가 공유. (ii) 모델은 origin/dev `33a7112e`: `AMDAIEInsertCopyOps` 가 epilogue 를 contraction 결과 버퍼에서 바로 읽고 epilogue 결과만 L1→L2→L3 로 옮긴다. 단 `isContractionEpilogue` 는 `linalg::ContractionOpInterface` producer 만 보므로 conv generic 으로 넓히고, `f938dd78` 의 '옵션 게이트 없음' 근거(InsertCopyOps 를 도는 파이프라인 중 contraction 은 GEMV 뿐)는 conv 파이프라인이 그 패스를 쓰는 순간 깨지므로 같은 변경에서 고친다. (iii) 8×8 f32 타일의 벡터화 max/truncf(AIE2P aievec 커버리지 먼저 프로브; truncf/max 용 ukernel 은 없음 — 필요하면 aie_api 로). L1 은 §5.0 의 epilogue 항(누산기 depth 1 + bf16 출력 depth 2)으로 계산. 출력이 bf16(쓰기 51.7 → 25.8 MiB) | G10/G13: 커널 속도 외에 네이티브 conv 가 줄 수 있는 구조적 e2e 이득 | T5.1, T4.2 | lit `conv_pipeline_epilogue.mlir`; VGG 컴파일 전용: bf16 결과의 conv NPU dispatch 13 개, conv 사이 elementwise CPU dispatch 0; 감사 채널 수 불변 | L |
| T5.3 | ~~im2col 경로의 NPU 측 bias fold + ReLU~~ — **완료(팀, origin/dev)** | bias 는 k+1 fold(필수: `f3079605` FC, `dd05ed3e` im2col), ReLU·truncf 는 pack-peel 이 같은 dispatch 에서 처리(`f4cedbee`), 플래그 삭제(`38613bbb`). 실측(VGG bf16, 같은 브랜치 on/off): 1.60 → 1.68 s(NPU 113 → 120 ms: K+1 → 다음 32 배수; CPU 44 → 62 ms: LHS host pad 두 번), corr 0.99998 → 0.99996. 예상한 '이득 ≤ 0.15 s' 가 아니라 약 0.08 s 손해이고, 팀 남은 일은 LHS 이중 pad 합치기(`CHANGES.md:83, :91, :97`) | — | — | — | — |
| T5.4 | **네이티브 VGG-16 e2e 레시피, 트래픽 회계, 층별 native/im2col 선택** | `scripts/onnx` 에 `--conv native`(T1.3)를 쓴다: origin/dev `options.py` 고정 플래그에서 `--iree-global-opt-use-im2col-for-convs` 만 빼고 격자 정책 플래그를 더한다. aievec 경로면 `--iree-amdaie-enable-vectorization-passes=true`(T2.1 로 GEMV/pool 은 제외). `--iree-amdaie-pooling-channels-last` 는 pooling 위치와 무관하게 유지한다(`05c7f6a3`: epilogue 가 contraction 에 들어가면 NCHW CPU pool 이 transpose 를 끌어안아 pool1 0.5 → 46 ms). Flow executable 목록, im2col 대비 이동 바이트 표. 층별 선택은 IREE `populateConvToIm2colOpPatterns(patterns, controlFn)` 의 controlFn 으로. bring-up 중 실패 conv 는 `--iree-amdaie-placement-map` 으로 CPU 고정 | 모델 수준 산출물; 전환은 증거 기반·가역(층별로 native 가 이길 때까지 im2col 기본) | T3.5, T4.2, T5.1 | 두 레시피 모두 vmfb 빌드; 모든 NPU dispatch 감사 green; `Unhandled pass pipeline` 0; GEMV/pool dispatch 가 im2col 레시피와 같은 구조 | M |
| T5.5a | **(승인) NPU 배치 B2** | 단일 dispatch 모듈 conv1_2·conv4_2·conv5_1 vs llvm-cpu f32 기준(bias 가 있으면 CPU+NPU 이종으로 컴파일 — 단일 장치에서는 fold 가 돌지 않는다)(corr, 최대 절대 오차, `scripts/repro/aie2p_bf16/matmul.sh` 의 손상된 64 행 블록 검출기), vmfb 당 1 회 = 총 3 회 | lock/동기화 동작과 bfp16 수치는 IR 로 증명 불가 | T3.5, T4.2, T4.5 | 소유자 승인(D9) 시에만; 아니면 M5 는 컴파일 전용으로 닫고 "하드웨어 미검증" | S |
| T5.5b | **(승인) NPU 배치 B3** | 네이티브 VGG corr/top-1 과 dispatch 별 npu/cpu/hwctx 시간 vs B1, 두 전원 모드 = 2 회(+ 필요 시 im2col 재측정 2 회). 첫 hang 에서 중단 | 층별 native/im2col 정책의 근거 | T5.4, T5.5a | 승인 시에만; 아니면 M6 의 시간 기준은 보류 | S |
| T5.6 | 선택: OC=64/128 층의 **G1 격자**(OH 를 열, OC 를 행) | `chooseConvTiling` 에 `ConvGridPolicy::OhOnColumns`: halo pre-split 의 열 청크를 `swapThreadXY`(pool 매핑)로 DimX 에, OC 를 행에 **distinct 슬라이스**로(`6ef3dc1f` 의 연속·오름차순 조건을 만족하는 L2 가중치 레이아웃 필요, T4.3 감사), 가중치를 열마다 복제(shim 바이트 ×8). B3 가 conv1_x/conv2_x 를 G2 에서 DMA 바운드로 보일 때만 | 초기 층은 G2 에서 산술 강도가 상대적으로 낮음 | T4.3(감사), T5.5b | conv1_2/conv2_2 컴파일 전용; 감사(채널, lock); NPU 는 승인 시 | L |
| T5.7 | 선택: **conv + ReLU + 2×2/s2 maxpool 융합** | conv1_2·2_2·3_3·4_3·5_3 의 ReLU + maxpool 을 conv 코어 epilogue 로(T5.2 확장). origin/dev 기본(`--pool cpu`)에서는 NPU dispatch 가 이미 conv 13 + FC 3 = 16 이라 hwctx 절감은 0 이다. 이득은 CPU pool dispatch(4 개 합 약 1.7 ms)와 conv 출력의 DDR 왕복 제거뿐이고, hwctx 레버가 되는 것은 `--pool accel`(NPU 20 dispatch, VGG 2.04 s) 대비일 때만이다(`CHANGES.md:80, :184, :186`). 조건: 행 타일·행 청크가 짝수, 비-elementwise consumer 용 dispatch 형성 예외. window = stride 라 halo 는 불필요 | conv→pool 경계의 DDR 왕복 제거(우선순위 낮음) | T5.2 | 컴파일 전용: pool CPU dispatch 감소; lit | L |

### 5.7 P6 — 통합 (M7)

| ID | 작업 | 내용 | 왜 | 의존 | 검증 | 크기 |
|---|---|---|---|---|---|---|
| T6.1 | **npu4 conv 의 lit·CI·회귀 커버리지** | lit: `lowering_strategy_conv_npu4.mlir`(13 형상), `lowering_strategy_conv_routing`, `conv_pipeline`, `conv_pipeline_columns`, `prepare_conv_for_npu`, conv pad 케이스, 인식 부정 케이스, split 가드, lock fan-out 케이스. CI `run.py`: `ConvolutionFromTemplate` bf16→f32(N=1 과 N=4 청크, IC 32/64, OC 64, 14×14/28×28) `run_on_target ['npu4']`, 네이티브 라우팅, 여기서는 컴파일 전용(`--do_not_run_aie`). 기존 npu1_4col conv CI 3 종·lit·샘플이 그대로인지 확인(공존 규칙). `tests/regression`(origin/dev 형식)에 `vgg16-native-bf16` 케이스(기대 파일은 T5.5b 승인 아래 생성) | G12/G20 | T3.5, T4.2, T5.5b | `ctest -R amd-aie`; lit; CI 설정 dry run | M |
| T6.2 | **리포 관례대로 문서** | 마일스톤마다 `docs_Conv/MMDD_<name>.md` — 변경 전/후 코드 블록과 추가/삭제/변경/유지 표([CLAUDE.md](../CLAUDE.md) §4); 살아있는 문서: 제약 표(T0.1), 층 행렬(T0.4/T3.5), 스파이크 결과(T0.6/T0.7/T2.6), aievec 검사, 반복 금지 목록; dev-conv 의 `docs_Conv/`·`docs_BD/` 를 errata 와 함께 conv-npu 로; 영어 커밋 메시지(원인·측정값·검증 범위); 경로 명시 `git add` | CLAUDE.md §2, §4; dev-conv 지식이 그 브랜치에만 있음 | — | 검토; 본문에 날짜 없음 | S |
| T6.3 | **origin/dev 로 가는 선별 PR 시리즈** | 순서: (0) 락 스크립트(T1.0), (1) conv 인식/라우팅/함수별 분기 + `855c3168`, (2) conv bias fold(`AMDAIEFoldBiasIntoContractionDispatches` 확장) + conv Flow 준비, (3) `chooseConvTiling` + 가드, (4) 패스 수정 + conv 파이프라인 + 열, (5) aievec cherry-pick(-x) + 반올림 또는 conv ukernel, (6) 코어 내 epilogue(T5.2), (7) 테스트/문서/스크립트(`--conv`). pool·GEMV 머지·lock fan-out·im2col fold 는 팀이 이미 dev 에 넣었다. wip 패치·스크래치·aie shim 헤더·전역 `l1BufferDepth=1`·softmax 라우팅 제외; 영어 PR 본문에 컴파일 전용 vs NPU 검증 명시 | 원인 + 수정 + 검증이 끝난 것만 머지(CLAUDE.md §4); bisect 가능한 역사 | T6.1, T6.2 | PR 별 빌드 + lit 게이트; tip: ctest, 두 VGG 레시피 컴파일, 감사 green | M |

### 5.8 실행 순서

각 작업의 선행 작업은 위 표의 "의존" 열이 정본이다. 완료(11-03): T1.2(`conv-npu` 분기·실행 환경·빌드), T0.5. 생략: T0.8, T1.1. 팀이 끝냄: T5.3. 남은 순서는 다음과 같다.

| 단계 | 작업 | 재빌드 | 목적 |
|---|---|---|---|
| S0 | T0.6 (a), T0.7, T3.0 사전 확인(iree-opt), T0.4 의 Flow 덤프(origin/dev 레시피에서 im2col 만 뺌) | 없음 | 가장 위험한 가정 반증, conv 가 Flow 에 오는 모양(bias 융합) 확인 |
| S1 | T1.0, T1.4 | 없음 | 락·스크래치 정리 커밋(락은 지금 로컬 수정) |
| S2 | T4.1 1 단계 = T0.6 (b), T2.6(IGEMM) — `conv-npu` 곁가지 | 증분 | 커널 경로(D5)와 설계 대안(D13)의 근거 |
| S3 | T0.1–T0.3 → T1.3 → (승인) T1.5 | 증분 | 기준선 재현(M1) |
| S4 | T0.4 → T3.0 → T3.1(conv5_1, conv1_2) → T3.2 → (승인, 선택) B1.5 | 증분 | 네이티브 구조 증명(M3) |
| S5 | T2.2 → T2.1 → T2.3 → T5.1 + T2.4(같은 Flow 단계: fold → conv pad) → T2.5 | 증분 | 일반화(M2) |
| S6 | T3.3 → T3.5, T4.1 2 단계 → T4.2 → T4.5 → (승인) T5.5a; 병행 T4.3(감사) | 증분 | 전 배열·벡터 커널(M4, M5) |
| S7 | T5.2 → T5.4 → (승인) T5.5b; 선택 T5.6, T5.7 | 증분 | 모델(M6) |
| S8 | T6.1 → T6.2 → T6.3 | — | 통합(M7) |

남은 작업 37 개의 크기(상대 크기, 일정 추정 아님): S 15 · M 15 · L 4 · XL 3(조건부·선택 T3.4/T4.4/T5.6/T5.7 포함). 완료·생략 5 개(T0.5, T0.8, T1.1, T1.2, T5.3).

### 5.9 범위 밖이지만 e2e 목표에 필요한 일

| 항목 | 이유 | 상태 |
|---|---|---|
| hwctx 재사용(런타임) | default 전원 모드에서 dispatch 당 약 75–95 ms, origin/dev VGG(NPU dispatch 16 개)에서 약 1.2 s. conv 를 아무리 빠르게 해도 남는다. 컴파일러 쪽 레버는 NPU dispatch 수뿐이고 기본 레시피의 하한은 16(conv 13 + FC 3) | 팀 backlog(`CHANGES.md:197, :353`). 3devices 의 opt-in hwctx 캐시 `IREE_AMDXDNA_CACHE_HWCTX` 는 dev 통합 때 제거(`06aa6260`, `CHANGES.md:234`). `cef320fe` 는 origin/3devices-f32-aie2p-bf16 에만 |
| control-packet 모드의 shim 채널 | 오버레이가 shim MM2S 1 개를 선점해 npu4 전 배열 dispatch 가 모두 컴파일 실패(VGG 21/21). G2 는 shim MM2S 2 개를 다 쓴다 | 미해결(docs/1030_vgg_ctrlpkt.md). dev-POOL 에서는 CI matmul 도 실패 |
| NPU 전원 모드 | performance 모드면 VGG 2.29 → 0.44 s(STAGES, 23 dispatch 시절 값; origin/dev 는 미측정). root 필요 | 측정 시 모드를 항상 기록 |
| run 뒤 host sync | 읽기 전용 binding(상수 가중치·입력)은 건너뛴다(`3d454250`, VGG sync 41 → 1.3 ms). 남은 비용은 쓰기 binding(출력)에 비례한다. T5.2 의 bf16 출력이 이것도 절반으로 줄인다 | 해결(origin/dev) |

---

## 6. 마일스톤

| M | 이름 | 포함 | 완료 기준 | NPU 실행 |
|---|---|---|---|---|
| M0 | 설계 전 데이터와 반증 | T0.1–T0.7 | 제약 표, `aie_audit.py`, `conv_tile_calc.py`(정렬·epilogue 포함), 13 층 기준 행렬과 origin/dev 레시피의 Flow 덤프, aievec 64 B 스파이크(T0.6)·G2 objectFifo 스파이크(T0.7) 결과. VGG e2e 불변(1.71 s) | 없음 |
| M1 | 재현 가능한 빠른 기준선 | T1.0–T1.5 | (완료) `conv-npu` 분기·실행 환경·빌드(T1.2). 남은 것: 락 커밋, VGG im2col + bf16 ukernel 컴파일(matmul 13 + matvec 3 호출), `--conv` 스위치. 승인 시 B1(2 회): ~1.7 s / NPU ~0.12 s / corr ≥ 0.9999 / top-1 892 | B1 |
| M2 | 일반화된 dispatch 형성 | T2.1–T2.6, T5.1 | 인식·라우팅·공존 규칙, `chooseConvTiling`(R1–R12), conv bias fold, Flow 준비(사전 pack·halo split·청크 우선 레이아웃), 가드, IGEMM 스파이크와 D13 결정. VGG e2e 불변(im2col 기본) | 없음 |
| M3 | 단일 열 네이티브 하강 | T3.0, T3.1, T3.2 | 두 형상 슬라이스(conv5_1, conv1_2) 컴파일·감사 green. 승인 시 B1.5 로 누산기 상주·peel·Sequential alloc 확인 | B1.5(선택) |
| M4 | 전 배열 | T3.3, T3.4, T3.5 | 13 층 모두 32 코어 / 8 memtile / 8 열; broadcast 와 4-producer join 뿐; 채널 ≤ 6/6, BD ≤ 48, 차원 ≤ 3/4, stride 1, L3 연속 구간·반복 한도 충족 | 없음 |
| M5 | 벡터화·lock 안전·수치 명세 코어 | T4.1–T4.5, T5.5a | `aievec.matmul`(bfp16 MAC, conv_even)과 64 B 정렬 확인, 또는 문서화된 no-go + conv ukernel(aie_api)/IGEMM 결정; conv 레이아웃의 lock 조건 감사; 에뮬레이터 골든 파일(bias 반올림 포함). 승인 시 B2(3 회) | B2 |
| M6 | 모델 | T5.2, T5.4, T5.5b, T5.6, T5.7 | 코어 내 ReLU/truncf + bf16 출력, 네이티브 VGG 레시피 e2e 컴파일. 승인 시 B3 로 층별 native/im2col 정책(corr ≥ 0.9999, top-1 892, 어느 층도 im2col 버전보다 느리지 않음) | B3 |
| M7 | Landed | T6.1–T6.3 | npu4 lit/CI/회귀 커버리지, 기존 conv 테스트 불변, 관례대로 문서, 선별 PR 시리즈가 origin/dev 에 머지(검증 범위를 정직히 명시) | 없음 |

승인이 없으면 하드웨어 항목(M1·M3·M5·M6)은 컴파일 전용으로 닫고 "하드웨어 미검증" 으로 표기한다.

---

## 7. 소유자가 결정할 사항

| D | 결정 | 선택지 | 권고 |
|---|---|---|---|
| D1 | ~~베이스 브랜치와 머지 방식~~ | 결정됨(11-03): origin/dev `786d1d19` 에서 `conv-npu` 분기. dev-POOL 은 팀 `7060d5f3` 로 dev 에 포함 | 종결. dev-POOL 로 돌아가 측정할 때는 반드시 재빌드 |
| D2 | ~~submodule 패치~~ | origin/dev 레시피는 no-fuse 를 쓰지 않고, 팀은 그 fork 옵션을 지울 예정이다(`CHANGES.md:98, :100`) | 종결: 올리지 않는다. 비교 실험 동안만 로컬 적용 |
| D3 | **ukernel 아래 conv1_x** | 짝수 블록 규칙의 N 64→128 패딩 수용(origin/dev 에 그대로 남음)(peano-v22 실측 dispatch 당 6.65/7.69 ms) vs N=64 dispatch 의 numCols 축소를 `getPaddingMultiples` 에도 반영 | 절대 시간이 작으므로 **수용**으로 시작, 측정 뒤 재검토 |
| D4 | **성공 지표** | (a) e2e wall(hwctx 약 1.2 s/1.71 s 가 지배, 팀 목표는 CPU 단독 0.34 s, 네이티브 conv 가 움직일 수 있는 폭은 dispatch 수가 그대로면 ≤ ~0.15 s(추정)); (b) NPU 연산 + DMA 효율 + host dispatch 수(DDR 97 vs 236 MiB, CPU gather·LHS pad 제거, 피크 대비 효율 0.4 % → 두 자릿수 %; NPU 시간 = submit + wait + 쓰기 binding sync, `3d454250` 이후 정의); (c) 연구 산출물 | **(b)+(c)** 를 명시하고 e2e 는 항상 hwctx 비중과 두 전원 모드로 병기. CPU 단독 0.34 s 를 넘으려면 hwctx 재사용(§5.9)이 선행돼야 한다 |
| D5 | **커널 경로** | M0 의 T0.6 결과로 결정: aievec 가 512-bit 이상 읽기를 모두 64 B 정렬로 만들면 aievec(`9b4f831a` + `17986af8`, origin/dev 에 충돌 없이 적용); 아니면 kw 복제 레이아웃(L1 비용), conv ukernel(T4.4 — origin/dev 의 aie_api 인프라와 팀 원칙 'upstream aie_kernels 우선, 없으면 aie_api 로 직접'에 맞음), IGEMM(D13) 중 선택. im2col matmul 은 ukernel. join lock 은 origin/dev 계열 유지 | T0.6 결과 후 결정. 64 B 기준 go 가 아니면 T4.4(aie_api)를 먼저 검토 |
| D6 | ~~im2col 경로 epilogue(T5.3)~~ | 팀이 결정: fold 필수·플래그 삭제(`38613bbb`, `CHANGES.md:16`) | 종결 |
| D7 | **네이티브 bias** | (a) origin/dev fold 패스의 conv 형태(입력 채널 +1, 중심 탭) + 코어 내 ReLU/truncf; (b) FHWC 상수의 pseudo-tap(IC+1 패딩 비용 +4.8–39 % 와 비교해 재평가); (c) 코어 메모리 상수(`4de9cf99`, float 미확인); (d) CPU epilogue — origin/dev 가 지운 플래그를 되살리거나 새 분리 패스가 필요 | **(a)** — 코어 S2MM = A + B 라 bias 스트림 불가, 팀 방향(fold 필수)과 일치 |
| D8 | ~~lock fan-out 수정 소유~~ | 곽준호 님 `6ef3dc1f` 로 origin/dev 에 반영(연속 슬라이스) | 종결. conv 레이아웃 감사만 T4.3 |
| D9 | **NPU 실행 승인** | B1(2 회), B1.5(선택, 1–2 회), B2(3 회), B3(2 회 + 필요 시 기준 재측정 2 회). `--verbose` 를 주면 trace 실행이 1 회씩 더 붙는다. 모두 npu.lock 아래, TDR 60 s, 첫 hang 에서 중단. M0·M2·M4 는 없음 | 배치마다 승인. 승인 요청에는 배치별 총 실행 횟수를 적는다 |
| D10 | **정확도 바** | corr ≥ 0.9999 + top-1/top-5 일치 + 층별 최대 절대 오차 vs CPU(bfp16 MAC 은 1.00000 불가; bias fold 는 bias 를 bf16 으로 반올림한다, `CHANGES.md:89`) vs 스칼라 전용 정확성(성능 목표 없음) | **전자** |
| D11 | 범위 | VGG-16 3×3/s1/pad1 bf16, batch 1, NHWC. 새 경로 밖의 conv(npu1_4col, i8/i32, depthwise, NCHW, stride, dilation)는 기존 conv-decompose 를 그대로 타고, 새 플래그가 켜졌을 때만 거부 진단. 격자 정책은 층별 데이터 기반(G2 기본, G1 선택) | 그대로 |
| D12 | 브랜치 수명 | `conv-npu`(로컬, 11-03 생성)를 push 해 팀 가시성 확보; `dev-GEMV`(== dev-conv tip) 폐기; `dev-conv` 는 선별 커밋·문서의 읽기 전용 출처로 유지; dev-POOL 은 dev 에 포함됐으므로 보존만 | 그대로 |
| D13 | **네이티브 G2 vs IGEMM** | T2.6 스파이크 결과로 일반화(S5) 전에 결정. IGEMM: 엔지니어링이 작음(pack-peel·ukernel 재사용, 정렬 문제 없음, stride/1×1 커버), 온칩 강도 낮음. G2: 강도 높음, 새 파이프라인·정렬·채널 위험. 둘 다 유지하고 층별로 고르는 안도 가능 | 스파이크 후 결정. pack-peel + bf16 ukernel + 융합 ReLU 는 origin/dev 에서 하드웨어 검증됐다 |
| D14 | **층 간 활성값·가중치 레이아웃** | (a) NHWC 유지 + L2 2 단계 청크(IC_L2 ≥ 64 로 128 B 연속, 청크 반복 ≤ 63); (b) host pad·truncf dispatch 가 청크 우선 레이아웃을 씀(추가 dispatch 0, pool·elementwise 영향 평가 필요). 가중치는 어느 쪽이든 const-eval 사전 pack | T0.3 계산 후 결정 |

---

## 8. 리스크

| 리스크 | 완화 |
|---|---|
| **64 B 벡터 정렬**(C12, G17): AIE2P 의 512-bit 이상 접근은 64 B 정렬이 아니면 오류 없이 다른 주소를 접근한다(`33c20094` 실측: corr ~0.92). aievec 는 16 B 어긋남에 경고만 내고 32 B 어긋남은 경고도 없다 | T0.6 의 64 B 기준; R11; kw 복제 레이아웃, ukernel 정렬 로드 + 레지스터 시프트(T4.4), IGEMM(T2.6); 감사에서 버퍼 주소 정렬 확인 |
| **L3 연속 구간·반복 한도**(C11, G16): 16–32 B DDR 읽기, 청크 반복 > 63 이면 제어 코드 폭증·stall | R9/R10, 2 단계 청크, 청크 우선 레이아웃·가중치 사전 pack(D14), 감사에 `npu_instructions` 크기 |
| **빌드 락 미커밋**(G18): origin/dev 의 build.sh 에는 락이 없다. `conv-npu` 로 옮기면서 로컬 수정으로 다시 얹었지만 다른 사용자·다른 체크아웃에는 없다 | T1.0(커밋 + origin/dev PR) |
| **N=64 패딩**(D3): 머지 자체는 팀이 끝냈고 conv1_x 의 N 64→128 만 남는다 | D3; T1.3 에서 pad dispatch 와 matmul ukernel 호출을 확인 |
| **aievec 경로의 Peano v22 이식**: `9b4f831a` 는 v19 에서만 빌드·측정됐고 Peano intrinsic 을 직접 묶는다 | T4.1 2 단계(ELF 까지 컴파일, 바인딩 포팅) |
| **전역 벡터화 스위치**(G19): conv 를 위해 켜면 GEMV/pool 코어도 벡터화 | T2.1 에서 함수별 분기, on/off IR 비교 |
| **기존 conv 테스트**(G20): 새 override·거부 진단이 npu1 conv CI·lit·샘플을 깨뜨림 | 공존 규칙(T2.2), 불변 검증(T2.1, T6.1) |
| **채널 예산 한계**(memtile 6/6, shim MM2S 2)에 G2 격자가 붙어 있음; connection 하나만 더 생겨도 깨짐 | T0.7 로 M0 에서 확인; 파이프라인 변경마다 감사; connection 병합 |
| **열 간 multicast 형성**은 GEMV A multicast 와 pack-peel 에서 가정. 가중치 읽기의 OC 축 위치가 pack-peel 의 B 와 다름 | T0.7(손 입력) → T3.3 첫 단계에서 실제 파이프라인으로 재확인 |
| **IC 루프 재구성**: dev-conv 가 같은 구조에서 실패 사슬을 문서화함 | T3.0 을 T3.1 보다 먼저; iree-opt 사전 확인 |
| **copy 기반 파이프라인의 원인 미상 오답**: GEMV 는 코어당 출력 1 개(C [1,1], 가중치·활성값 L1 타일이 같은 memref 타입)일 때 npu4 에서 전부 0 을 낸다. 정적 검사로는 안 보였고 NPU 로만 발견됐다(`33b86238`) | T3.1 은 `addGemvPassPipeline` 을 본뜨므로 R12(코어 타일 퇴화 금지)를 T0.3/T2.3 에 넣는다 |
| **하드웨어 피드백 없음**(정책): 정적 검사를 다 통과한 vmfb 도 hang 하거나 틀릴 수 있음(GEMV 교착은 PDI/제어 코드가 바이트 동일했음) | lock fan-out 감사, split 가드, 에뮬레이터 골든, 선택 배치 B1.5 를 전 배열 작업 전에 |
| **팀 분기**: 3devices-f32-aie2p-bf16(`d2846c0d`, `822f0fed`, `cef320fe`)·이기영 님의 `feature/k1-fold` 가 동시에 landing 하면 ObjFifoBufferization/LowerToAIE/CoreToStandard/pad·fold 패스가 다시 충돌한다. 팀은 구조 정리(문자열 override → enum, pool 정책 경로, 실험 옵션)를 예고했다(`CHANGES.md:200-203`) | -x cherry-pick, conv-npu 조기 push, PR 순서와 override 형식 합의 |
| **이득 상한**: 완벽한 네이티브 conv 도 hwctx ~1.2 s 를 남기고 컨텍스트 재사용 없이는 CPU 단독 0.34 s 를 못 이긴다. origin/dev 의 NPU 연산 합(0.120 s)도 CPU 단독 dispatch 합(0.085 s)보다 길다 | D4 로 지표 고정, §5.9 |
| **수치**: bfp16 MAC 이 결과를 바꾼다(corr 0.99997). bias fold 는 bias 를 bf16 으로 반올림한다(VGG corr 0.99998 → 0.99996). conv 의 Sequential alloc(#1192, '수치 오류를 피하려고')은 `33c20094` 가 고친 정렬 버그였을 수 있다(미검증). memtile 측 unpack 은 새 레이아웃에서 미검증 | B2 에서 에뮬레이터 비교; conv 함수에 Sequential alloc 유지, B1.5 에서 비교 가능 |
| **빌드 예산·스테일 바이너리**: `conv-npu` 로 옮긴 뒤 빌드는 ccache 덕에 약 9 분이었다(11-03). ccache 디렉터리가 root 소유면 실패한다([EXPERIMENT_ENV.md](../docs/EXPERIMENT_ENV.md) §9). 브랜치를 바꾸거나 되돌린 뒤 결론 전 반드시 재빌드한다(오진 2 회) | 빌드 락, 결론 전 재빌드 체크리스트 |
| **업스트림 휴면 + mlir-aie 에 bf16 conv 커널 없음**: 모든 conv 부품이 fork 소유, 외부 리뷰 없음 | PR 시리즈를 작게, 팀 리뷰. conv ukernel 은 origin/dev 의 aie_api 인프라 위에서 작성 |

---

## 9. 검증 · 측정 방법 (정책 준수)

### 9.1 컴파일 전용 지표 (모든 작업에 기본)

| 지표 | 얻는 방법 | 기준 |
|---|---|---|
| dispatch 지도 | `--compile-to=flow` + `scripts/debug/condense_flow.py` | NPU dispatch 수(origin/dev 기본 16), host pad/crop 수, affinity |
| 코어·memtile·열 수 | `--mlir-print-ir-after=iree-amdaie-lower-to-aie` 의 `aie.core`/`aie.tile` | 네이티브 전 배열 = 32/8/8 |
| DMA 채널 | `--mlir-print-ir-after=iree-amdaie-assign-channels` | memtile ≤ 6/6, shim ≤ 2 MM2S(overlay 사용 시 1), 코어 ≤ 2/2 |
| memtile·코어 BD 수·차원·stride | `--mlir-print-ir-before=amdaie-assign-bd-ids`(aie 방언 패스, 배정이 실패해도 덤프). `iree-amdaie-assign-bd-ids` 는 `--iree-amdaie-reprogram-dmas` 일 때만 돈다 | memtile 짝/홀 각 ≤ 24(합 48), 차원 3/4, 16-bit innermost stride 1, 크기 < 1023, 전송 길이·주소 4 B 배수 |
| shim BD·반복 | `--mlir-print-ir-after=iree-amdaie-assign-npu-dma-bd-ids` 와 제어 코드 | inter 반복 ≤ 63, repeat ≤ 256(stride 0 만), `npu_instructions` 크기 |
| L1 바이트 | `--mlir-print-ir-after=amdaie-assign-buffer-addresses` 의 코어 타일별 max(address + 크기)(스택·정렬 여유 포함). `-after=iree-amdaie-lower-to-aie` 의 `aie.buffer` 크기 합은 `33c20094` 이후 정렬 여유만큼 작게 나온다. 64 B 이상 코어 버퍼의 `address % 64 == 0` 도 확인(어긋나면 조용한 오답). 최종 판정은 `allocated buffers exceeded available memory` 진단 | ≤ 65,536; `conv_tile_calc.py`(정렬·epilogue 포함) 예측과 일치 |
| L3 연속 구간 | shim `npu.dma_cpy_nd` 의 innermost size·stride | ≥ 128 B 목표(GEMV 전례: 64 B → 512 B) |
| lock 구조 | `aie.lock` 수 vs objFifo 의 producer·consumer 수(`-after=iree-amdaie-objfifo-bufferization`) | 1→N distinct fan-out 은 lock 쌍 수 = consumer 수(공유 쌍 + distinct 슬라이스 조합 0 건) |
| 벡터 정렬 | `--mlir-print-ir-after=align-transfer-reads` 의 경고 + IR 의 load 오프셋 식 + 코어 ELF `llvm-objdump -d` | 경고 0 건 + 512-bit 이상 load 오프셋 64 B 배수 |
| 코어 명령 | `--iree-hal-dump-executable-intermediates-to=<dir>` → `<dir>/<entry>/input.ll`, `input.opt.ll`, `input.o`, `core_<col>_<row>.elf`. asm 파일은 생성되지 않으므로 Peano `llvm-objdump -d`, 크기는 `llvm-size`. ukernel 을 쓰면 같은 디렉터리에 `matmul.cc`·`matmul.o`(zero_fill·matvec 동일)와 `ukernel_include/`(embed 된 aie_api·aie_kernels 헤더)가 남는다. `UK_INSTANCES` 는 명령행에만 있으므로 손으로 재컴파일할 때는 IR 의 `ukernel_instance` 로 목록을 만든다 | `aievec.matmul` 수 = `W/8 · OC_t/8 · 9`/청크, `vector.contract` 0, `.text` < 16 KB, spill 없음 |
| 불변성 | 같은 툴체인 안에서는 vmfb md5. 툴체인이 바뀌면(Peano v22, crRnd) `-after=iree-amdaie-lower-to-aie` IR·제어 코드 비교 | conv 변경이 다른 파이프라인을 바꾸지 않음 |
| lit / ctest | 파일의 `// RUN:` 라인 그대로, `ctest -R amd-aie`(NPU 를 열지 않음) | 전부 green |

모든 지표를 `scripts/debug/aie_audit.py`(T0.2) 하나로 뽑고, 위반은 비정상 종료로 게이트한다. 기대한 `IR Dump Before/After <pass>` 헤더가 없으면 감사 실패로 처리한다(존재하지 않는 패스 이름으로 빈 덤프를 읽은 전례가 있다).

### 9.2 NPU 실행 (승인 배치만)

| 배치 | 시점 | 내용 | 실행 횟수 | 산출 |
|---|---|---|---|---|
| B1 | M1 | VGG im2col + ukernel(origin/dev 레시피), `--runs 1 --warmup 0`, 전원 모드 2 | 2(+ `--verbose` trace 실행 2) | dispatch 별 npu/cpu/hwctx(ms), corr, top-1/5 → 기준선 표 |
| B1.5 (선택) | M3 | 단일 열 네이티브 conv 1 dispatch(conv5_1 또는 CI 형상) vs llvm-cpu | 1–2 | 누산기 상주·peel·Sequential alloc 확인 |
| B2 | M5 | conv1_2·conv4_2·conv5_1 단일 dispatch vs llvm-cpu f32 | 3 | corr, 최대 절대 오차, 64 행 블록 손상, hang 여부 |
| B3 | M6 | 네이티브 VGG vs B1, 두 전원 모드 | 2(+ 기준 재측정 2) | 층별 native/im2col 승자, 최종 정책 |

GEMV zero-row 재현은 팀이 `6ef3dc1f`·`0c5bc58f` 에서 NPU 로 확인했으므로(f32 GEMV hang 케이스 통과, VGG GEMV-on corr 1.0) 뺐다.

규칙: vmfb 당 1 회, 첫 hang 에서 중단(TDR 60 s, 다른 컨텍스트까지 막힘), npu.lock 래퍼(run-debug/deploy) 아래에서만, sha·플래그·전원 모드 기록, 결과는 같은 날 docs 에. 승인 요청에는 배치별 총 실행 횟수를 적는다.

### 9.3 측정 도구 (origin/dev 에서 가져옴)

`scripts/onnx/{compile,run,e2e,dispatch_times}.py` + `build-trace/`(런타임 전용 console tracing)가 `conv-npu` 에 들어와 있고 11-03 에 빌드했다(사용법은 [EXPERIMENT_ENV.md](../docs/EXPERIMENT_ENV.md)). `run.py` 기본값은 `--runs 3 --warmup 1` 이라 승인 배치에서는 `--runs 1 --warmup 0` 을 명시한다. `compile.py` 는 onnx 파일 경로를 받는다. `--verbose` 의 npu 시간은 submit + wait 이고, `3d454250` 이후 읽기 전용 binding 의 host sync 가 빠졌으므로(VGG NPU 142 → 105 ms) STAGES.md 의 dispatch 별 값과 직접 비교하지 않는다.

---

## 10. 출처와 조사 방법

### 10.1 브랜치·커밋 (읽기 전용 확인, 2026-11-03)

| 참조 | SHA | 비고 |
|---|---|---|
| `dev-POOL` == `origin/dev-POOL` | `9c6ea263` | origin/dev 에 포함(`7060d5f3`) |
| `dev-conv` == `dev-GEMV` | `56768a52` | 24 커밋, 공통 조상 `f7a5c86f` |
| `origin/dev` == `origin/k1-fold-integration` | `786d1d19` | dev-POOL 포함, dev-POOL 보다 124 커밋 앞, submodule iree `32dad766`·aie_api `b0422227`, Peano v22 |
| `conv-npu`(로컬) | `786d1d19` | 11-03 생성, 실행 환경 구성·빌드 |
| `origin/3devices-integration` | `2c5763b8` | 이전 dev(이 문서 초판의 기준) |
| `origin/pool-integration` | `bdd7d47e` | dev-POOL 통합, origin/dev 의 조상 |
| `origin/feature/k1-fold` | `3bad9591` | 이기영, 미병합(gecv-k1fold 의 후속) |
| `origin/aie2p-bf16-vectorized` | `3e7af2a1` | `9b4f831a`, `d2846c0d`, `17986af8` 포함 — origin/dev 조상 아님, Peano v19 |
| `origin/feature/gecv-k1fold` | `a66400ad` | bias fold·epilogue |
| `origin/peano-v22` | `b16240ae` | STAGES.md(이전 판; 최신 판은 origin/dev) |
| 업스트림 fork 지점 | `fddfec1b` | nod-ai/iree-amd-aie 2026-06-03 |

### 10.2 핵심 파일

`compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/{KernelDispatch.cpp, Passes.cpp, AMDAIELowerExecutableTarget.cpp, AMDAIEPadContractionDispatches.cpp, AMDAIEConvertPoolingToChannelsLast.cpp, AMDAIEAssignDeviceAffinities.cpp, AMDAIEAssignLogicalObjectFifoDepth.cpp, AMDAIEObjFifoBufferization.cpp, AMDAIESplitLogicalObjFifos.cpp, AMDAIELowerToUKernels.cpp, AMDAIEGenerateControlOverlay.cpp, Utils/AMDAIEUtils.cpp, Utils/AMDAIELogicalObjFifoSplittingUtils.cpp, Utils/AMDAIEDmaUtils.cpp}`, `aievec/{VectorToAIEVecConversions.cpp, VectorToVectorConversions.cpp, AIEVecToLLVM.cpp}`, `aie/AIEDialect.cpp`, `Target/{AIETarget.h, AIETarget.cpp, XCLBinGen.cpp}`, `runtime/src/iree-amd-aie/aie_runtime/iree_aie_runtime.h`, `third_party/aie-rt/driver/src/global/xaie2pgbl_reginit.c`, `third_party/iree/compiler/src/iree/compiler/{DispatchCreation/FormDispatchRegions.cpp(미커밋), Codegen/Common/ConvolutionToIGEMM.cpp, Dialect/LinalgExt/Transforms/Transforms.h, Preprocessing/Common/ConvertConvToChannelsLast.cpp}`, `build_tools/ci/cpu_comparison/run.py`, `models/vgg16/vgg.mlir`, `log_shjo/poolpad/compile_vgg.sh`, `scripts/build/build.sh`, `scripts/lock/lock-common.sh`.

### 10.3 문서

docs/0908_im2col_e2e.md, docs/1030_vgg_ctrlpkt.md, docs_POOL/{0921_stage1_pool_pipeline, 0930_pool_halo_split, 1012_full_array_padding_design, 1019_pool_max_columns_policy, 1022_pool_flow_padding}.md, docs_GEMV/{0909_gemv_pipeline_design, 0909_gemv_solution_catalog, 0910_perf_methodology, 0914_matvec_kernel_design, 0930_gemv_padding_plan}.md, dev-conv `docs_Conv/{0825_ci_vs_vgg, 0901_blockers_l1_bd, 0903_restructure_plan, 0907_tiling_impl_results, 0907_solved_cases, 0908_merge_plan}.md`·`docs_Conv/wip/*.patch`·`docs_BD/MEMTILE_BD_COUNT.md`, origin/dev `docs/2026-10-21_peano_v22/STAGES.md`·`docs/CHANGES.md`·`models/vgg16/README.md`·`tests/regression/README.md`, origin/aie2p-bf16-vectorized `docs/{patch,repro}_aie2p_bf16_vectorized.md`.

### 10.4 조사 방법과 검증 상태

1. 7 갈래 병렬 읽기(네이티브 파이프라인 · dev-conv · POOL/GEMV 인프라 · im2col e2e · 팀 브랜치 · 커널 수준 · 업스트림/외부).
2. 계획 3 안(점진 / 네이티브 우선 / 리스크 감사) → 심사·통합(점수 7 / 7.5 / 7.5).
3. 통합안의 핵심 주장 14 건(V1–V14) 검증. 반박 검증 에이전트는 사용량 한도로 일부만 끝나(V1 3 건, V4 1 건) 나머지는 git·grep·산술·Flow 컴파일로 직접 확인했다(§10.5).
4. 조립된 문서에 대해 비평 에이전트 3 개(사실 정확성 / 누락 작업 / 순서·실행 가능성)를 돌렸다. 지적 31 건을 모두 코드로 재확인했고, 반박된 것은 없다. 전처리 IR 에 `linalg.pack` 이 남아 있다는 세부 하나만 실제 덤프(§11)와 달랐다.
5. VGG 를 im2col 없이 Flow 와 전처리 끝까지 컴파일해 conv 의 실제 형태를 확인했다(§11).
6. 같은 날 origin/dev 가 `786d1d19` 로 올라가, 새 커밋 71 개를 주제별 에이전트 4 개(통합·lock·GEMV / ukernel·툴체인·메모리 / bias·pad·epilogue / 레시피·측정)로 읽고 문서 전체를 갱신했다(§10.6). 에이전트 결론 가운데 핵심 사실(머지 포함 여부, `855c3168` 적용성, 64 B 정렬 근거, IGEMM 인프라, 빌드 락 부재)은 git·grep 로 다시 확인했다.

본문 중 **"미검증"/"미확인"** 으로 표시한 수치·주장은 컴파일이나 실행으로 확인되지 않은 것이다.

### 10.5 핵심 주장 검증 결과

| 주장 | 내용 | 판정 | 확인 방법 |
|---|---|---|---|
| V1 | dev-POOL ⊕ origin/dev 머지는 5 파일 11 곳 충돌, "양쪽 변경" 14 파일 | 확인 | 검증 에이전트 3 건: legacy `git merge-tree` 재실행이 바이트 동일, ORT `--write-tree` 도 같은 5 파일, `comm -12` 로 14 파일. 메모: 현재 작업 트리에서 바로 머지하면 로컬 수정 6 파일 때문에 시작 전에 중단된다(T1.0 커밋으로 해소). 팀이 같은 5 파일로 머지했다(`7060d5f3`) |
| V2 | origin/dev 는 `AMDAIEDmaUtils.cpp` 를 건드리지 않아 `855c3168` 이 그대로 적용된다 | 확인 | `git log f7a5c86f..origin/dev -- …/AMDAIEDmaUtils.cpp` 가 비어 있음; `855c3168` 은 그 파일만 수정(+57/−15). 메모: aie2p `d2846c0d` 는 같은 파일을 +80 수정(T4.1 에서 제외) |
| V3 | ukernel 을 켜면 N 이 numCols·2·8 = 128 의 배수로 패딩돼 conv1_x(N=64)가 128 이 된다 | 확인 | origin/dev `getPaddingMultiples`(`perCore = useUKernel ? 2 : 1`); lit `pad_contraction_dispatches.mlir:428-470`(머지 뒤에도 유지); STAGES.md 층별 표 conv1_1 6.65 ms, conv1_2 7.69 ms(bf16 `all`) |
| V4 | 플러그인 전처리 확장은 사용자 전처리 파이프라인 뒤에 돌고, `conv_2d_nhwc_fhwc` 는 명명 op 이며 demote 대상이다 | 확인 | 검증 에이전트: `Preprocessing/Passes.cpp:103` → `:109`; `LinalgNamedStructuredOps.yaml:2135`; `DemoteContractionInputs.cpp:126-127`. 전처리 끝 IR 덤프로 conv init = bias broadcast, 입력 체인 pad → collapse_shape → truncf 확인(§11) |
| V5 | VGG 가중치는 상수이고 레이아웃 transpose 는 const-fold 된다 | 확인 | `vgg.mlir` 상수 34 개; `vgg_s1_flow.log` 의 transpose dispatch 는 모두 활성값 형상, 가중치 형상 0; 전처리 끝 IR 에서 가중치 transpose 는 상수 위의 `linalg.transpose` |
| V6 | FHWC L2 슬랩 → L1 `[9][IC_c/8][OC_t][8]` 전송은 4-D, innermost stride 1 | 확인(산술) | 주소 = oc·9·IC_c + tap·IC_c + ic_o·8 + ic_i. IR 확인은 T0.7 |
| V7 | 여러 코어가 읽는 같은 L2 조각은 다중 타겟 connection 하나(broadcast)가 된다 | 부분 확인 | GEMV fc2 flow 분류(A 는 memtile 하나에서 32 코어로 broadcast, docs_GEMV/0909 §8); pack-peel 32 코어 matmul. conv 레이아웃에서는 T0.7·T3.3 |
| V8 | conv-decompose 코어 타일은 약 2.8 MAC/B 라 벡터화 뒤 DMA 바운드다 | 산술 확인, 바운드 판단은 추정 | 576·IC / (204·IC + 256) ≈ 2.82; 교정된 L1 식이 두 실측값 재현. 채널당 4 B/cycle 은 하드웨어 미검증 |
| V9 | G2 는 broadcast 와 4-producer join 만 만들고, join 은 origin/dev 가 처리하며, 1→N 은 연속·오름차순 서로소 슬라이스면 origin/dev `6ef3dc1f` 의 consumer 별 쌍, 그 밖(인터리브)은 단일 쌍이다(11-03 갱신) | 확인 | origin/dev `AMDAIEObjFifoBufferization.cpp:144-160`; aie2p `d2846c0d` 는 join 만; dev-POOL `:55-71` |
| V10 | 하드웨어 한계: 채널 6/2/2, BD 차원 3/4, 크기 < 1023, stride < 2^20, 16-bit innermost stride 1, 스택 1024 | 확인 | aie-rt `xaie2pgbl_reginit.c` MemTile `:1673`=6, Tile `:1911`=2, Shim `:2164`=2, Shim IterWrapMax `:2066`=63; `AIEDialect.cpp:474-477, :501, :504, :521`; `AIETarget.h:142` |
| V11 | batch-1 VGG conv 는 6-iterator generic 으로 도착한다 | 확인(컴파일) | §11 Flow 컴파일: conv 13 개 모두 `[P,P,P,R,R,R]` |
| V12 | dev-POOL·origin/dev 에서 bf16 contract 는 AIE2P 로 못 내려가고, `9b4f831a`·`d2846c0d`·`17986af8` 은 origin/dev 의 조상이 아니다 | 확인 | `git merge-base --is-ancestor` 3 건 모두 아님; `VectorToAIEVecConversions.cpp:118-126` |
| V13 | 미커밋 submodule 패치는 어느 커밋에도 없고 origin/dev 의 `32dad766` 에도 같은 변경이 없다 | 확인(로컬 범위) | `git -C third_party/iree diff`; `git -C third_party/iree log c73928f0..32dad766 -- …/FormDispatchRegions.cpp` 비어 있음; ace-knu/iree 원격의 다른 브랜치는 확인 못 함. origin/dev 레시피는 no-fuse 를 쓰지 않아 패치는 실험 전용이 됐다(G02) |
| V14 | 팀 측정 74.5 → 8.8 → 2.29 s, NPU 72.30 → 6.65 → 0.124 s, hwctx 1.80 s, CPU 단독 0.34 s, performance 모드 0.44 s | 확인 | origin/dev `STAGES.md:26-28, :37-40, :311, :356-360, :379-387`. hwctx 는 dispatch 당 68–81 ms. 이후 1.62 s(dev-POOL 머지)·1.71 s(k+1 fold)로 바뀌었다(`CHANGES.md:77, :173`) |

비평 반영으로 바뀐 것(모두 코드로 재확인):

- **L1 식**: 누산기도 depth 배가 된다(오늘 하강은 L1 objFifo 전체에 같은 depth). 교정된 식이 105,984 B·105,728 B 두 실측값을 재현한다. 예비 산술 표를 두 안(A 오늘 하강, B 누산기 depth 1)으로 다시 계산했다.
- **측정 지점**: BD 는 `amdaie-assign-bd-ids`(aie 방언)로, shim BD 는 `iree-amdaie-assign-npu-dma-bd-ids` 로 본다. L1 은 depth 가 반영된 `lower-to-aie` 뒤에서 잰다. 존재하지 않는 `convert-vector-to-aievec` 는 `align-transfer-reads`·`test-lower-vector-to-aievec` 로, 자리표시 플래그는 `--iree-hal-dump-executable-intermediates-to` 로 바꿨다.
- **새 결함 G16–G21**: L3 연속 구간·반복 한도, kw 시프트 정렬, 빌드 락 미커밋, 전역 벡터화 스위치, 기존 conv 테스트 공존, IGEMM 미검토.
- **새 작업**: T0.6·T0.7(반증 스파이크), T0.8(기준 산출물), T1.0(락 커밋), T2.6(IGEMM 스파이크), T3.0(패스 선수정), T5.7(conv+pool 융합). 하드웨어 배치 B2·B3 를 T5.5a·T5.5b 로 나눴다.
- **순서**: 제자리 머지·증분 빌드, T3.0 → T3.1 수직 슬라이스 → 일반화, T2.2 를 T2.1 앞으로.
- **실행 가능성**: B1 을 총 4 회로 줄이고 회귀 스위트를 뺐다. 툴체인이 바뀌는 단계의 md5 게이트를 IR 비교와 ELF 한도로 바꿨다.
- **사소한 교정**: §1.1 단위를 MiB 로 통일, G05 의 층 수(8 / 5), 몇 개의 줄 번호·파일명·명령 인자.


### 10.6 dev `786d1d19` 반영 (11-03)

- **팀이 끝낸 일**: dev-POOL 머지(`7060d5f3`) → T1.2·G03·D1; consumer 별 lock 쌍(`6ef3dc1f`, 연속 슬라이스) → G09·T4.3·D8; im2col 경로의 bias fold + ReLU epilogue(`f3079605`·`dd05ed3e`·`38613bbb`) → T5.3·D6; GEMV matvec ukernel(`69a5e8b7`); mlir-aie aie_api 기반 ukernel(`a6874c90` 등).
- **이 세션에서 끝낸 일**: `conv-npu` 분기, 실행 환경(dev 이미지 재빌드, Peano v22, `build-trace`, Vulkan), 빌드, NPU 없는 확인(MLP CPU·CPU+GPU e2e, `--verbose`, MLP·VGG 의 CPU+NPU 컴파일). 기록은 [docs/1103_experiment_env_setup.md](../docs/1103_experiment_env_setup.md).
- **바뀐 수치**: VGG 기준 1.71 s(NPU dispatch 16, hwctx 약 1.2 s, NPU 연산 0.120 s), STAGES 의 2.29 s 는 이전 값.
- **새 제약**: 64 B 벡터 정렬(`33c20094`; C12·G17·R11), BD 전송 4 B 배수(`5eb5a583`; C2), bias fold 전제(C13), L1 의 정렬·epilogue 항(§5.0).
- **바뀐 설계**: conv bias fold(T5.1)를 origin/dev Flow fold 확장으로 앞당겼다. Flow 준비(T2.4)는 상수 재작성 helper 로 한다. IC+1 패딩 배수는 청크 크기에 맞춘다. 코어 epilogue 는 `33a7112e` 를 본뜨고 누산기 depth 1 이 필수다. conv ukernel 은 aie_api 로 작성한다.
- **바뀐 이름·위치**: `--iree-amdaie-enable-pooling` → `--iree-amdaie-accelerate-pooling`; GEMV 파이프라인 스위치·N-split 패스·`createConcatDispatch` 삭제; pad helper 는 `AMDAIEDispatchRewriteUtils`; dispatch 별 override 위치 `KernelDispatch.h:141-143`.
- **그대로인 것**: 네이티브 conv 의 G04–G08·G12·G19–G21, conv-decompose 코드, `855c3168` 의 적용성(충돌 0).

---

## 11. 이번에 실행한 확인 (컴파일 전용)

dev-POOL 의 VGG 레시피(`log_shjo/poolpad/compile_vgg.sh`)에서 `--iree-global-opt-use-im2col-for-convs` 만 빼고 두 지점까지 컴파일했다. 빌드 트리는 10-22 빌드(미커밋 submodule 패치 포함)다. NPU 는 쓰지 않았다.

```bash
docker run --rm --user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  --cpus=4 --memory=12g --memory-swap=12g -v "$PWD:/workspace" -w /workspace -e HOME=/workspace \
  -e LD_LIBRARY_PATH=/workspace/build/lib iree-amd-aie:dev-shjo \
  build/tools/iree-compile models/vgg16/vgg.mlir \
  --iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-amdaie-demote-contraction-inputs-to-bf16 --iree-amdaie-enable-vectorization-passes=false \
  "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-convert-conv-to-channels-last)" \
  --iree-dispatch-creation-no-fuse-into-contraction-conv-roots --iree-global-opt-detach-elementwise-through-reshape \
  --iree-flow-enable-executable-deduplication=false --iree-amdaie-pooling-channels-last --iree-amdaie-enable-pooling \
  --compile-to=flow --mlir-elide-elementsattrs-if-larger=8 --mlir-elide-resource-strings-if-larger=8 \
  -o vgg_flow_noim2col.mlir
```

전처리 끝 덤프는 같은 명령에서 `--compile-to=flow` 를 `--compile-to=preprocessing` 으로 바꿔 얻었다.

※ 이 확인은 dev-POOL 의 10-22 빌드·레시피로 했다. origin/dev 에는 `--iree-amdaie-enable-pooling` 이 없고(`--iree-amdaie-accelerate-pooling`), 레시피에서 no-fuse·detach 가 빠져 im2col 만 빼면 bias+ReLU 가 conv dispatch 에 융합된다(코드 추론). `conv-npu` 에서 다시 보는 일은 T0.4 다.

**Flow 결과**(rc 0, `flow.executable` 58 개 = 모델 dispatch 56 + host pad/crop 2):

| 항목 | 관찰 |
|---|---|
| conv dispatch | 13 개 `conv_<OH>x<OW>x<OC>x3x3x<IC>_bf16xbf16xf32`. 전부 iterator `[P,P,P,R,R,R]` |
| conv 맵 | 입력 `(d0+d3, d1+d4, d5)`, 가중치 `(d3, d4, d5, d2)`, 출력 `(d0, d1, d2)` → batch 접힌 NHWC · HWCF |
| conv 타입 | 입력 `tensor<(OH+2)x(OW+2)xICxbf16>`(이미 패딩됨), 가중치 `tensor<3x3xICxOCxbf16>`, 출력 f32, init 은 `linalg.fill` 0 |
| conv 앞 host dispatch | `elementwise_<N>_f32xbf16` 1 개: SAME pad 와 truncf 를 합쳐 패딩된 bf16 입력을 씀(예: 153228 = 226·226·3) |
| conv 뒤 host dispatch | `elementwise_<OH>x<OW>x<OC>_f32` 1 개: bias + ReLU |
| pooling · FC | pooling 5 개는 `[P,P,P,R,R]` generic, FC 3 개는 `matmul_1x…`(GEMV) |

**전처리 끝 결과**(rc 0; T2.4 가 플러그인 전처리 확장의 끝에서 보게 될 IR):

| 항목 | 관찰 |
|---|---|
| op 수 | `linalg.conv_2d_nhwc_hwcf` 13, `tensor.pad` 13, `linalg.broadcast` 13, `linalg.pooling_nhwc_max` 5, `linalg.pack` 0 |
| 입력 체인 | `tensor.pad`(5-D, 이미 NHWC, H·W 에 1 패딩) → `tensor.collapse_shape` → truncf `linalg.generic` → conv |
| 가중치 체인 | 상수(OIHW f32) → `linalg.transpose`(→ HWCF) → truncf `linalg.generic` → conv |
| init | bias `linalg.broadcast`(NCHW) → `linalg.transpose`(→ NHWC) → conv 의 outs. zero fill + bias elementwise 로 갈라지는 것은 뒤의 GlobalOptimization 이다 |

비교: im2col 경로(`log_shjo/pool/vgg_s1_flow.log`)는 층마다 host dispatch 가 3 개(gather + truncf, bias + ReLU, 다음 층 패딩 버퍼로의 `slow_memcpy`)이고 conv4_x·conv5_x·conv1_1 에는 pad/crop 이 더 붙는다. 네이티브 형태는 층당 2 개이며, NPU 후보 dispatch 수(dev-POOL 레시피 기준 conv 13 + FC 3 + pool 5 = 21)는 같다. origin/dev 기본은 pool 이 CPU 라 16 이다. 지금 컴파일러로는 affinity 가드가 이 conv 들을 NPU 로 보내지만(origin/dev `AMDAIEAssignDeviceAffinities.cpp:108-132`) 6-D generic 을 conv 로 인식하지 못하므로, native vmfb 는 T2.x 이후에만 가능하다.

**`conv-npu` 컴파일 확인**(11-03, origin/dev 레시피 그대로, NPU 실행 없음):

```bash
./scripts/docker/run-onnx.sh compile models/vgg16/vgg16-12.onnx --numeric bf16
```

rc 0, import 4.4 s, compile 112.2 s, vmfb 285,387,057 B, commit `786d1d19`. 집계는 `iree-dump-module --output=metadata` 로 실행 파일 구간을 찾고, 그 구간의 문자열에서 entry point 이름을 셌다.

| 실행 파일 | entry point |
|---|---|
| NPU(`amdaie-pdi-fb`, 3.3 MB) | 16 개: dispatch 3·7·11·15·19·23·27·31·35·39·42·45·48 의 conv matmul 13 + 50·51·52 의 FC 3 |
| CPU(`embedded-elf-x86_64`, 63 KB) | transpose 1, im2col gather 13, max pool 5, `slow_memcpy` 8, host pad 32(matmul 16 개 × 2 와 수가 맞음), crop 9 |
