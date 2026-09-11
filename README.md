[![CI Linux](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-linux.yml)
[![CI Windows](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-windows.yml)
[![CI MacOS](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-macos.yml)

# AMD AIE Plugin for IREE

This repository contains an early-phase IREE compiler and runtime plugin for targeting AMD NPUs with IREE.

## Developer Setup

**Strong recommendation**: check the CI scripts @ [.github/workflows](.github/workflows) - they do a fresh checkout and build on every commit and are written to be read by a non-CI expert.

### Getting the repository

Either

```
# ssh
git clone --recursive git@github.com:nod-ai/iree-amd-aie.git
# https
git clone --recursive https://github.com/nod-ai/iree-amd-aie.git
```

or, if you want a faster checkout,

```
git \
  -c submodule."third_party/torch-mlir".update=none \
  -c submodule."third_party/stablehlo".update=none \
  -c submodule."third_party/XRT".update=none \
  clone \
  --recursive \
  --shallow-submodules \
  git@github.com:nod-ai/iree-amd-aie.git # https://github.com/nod-ai/iree-amd-aie.git
```

The above avoids cloning entire repo histories for submodules, and skips a few, currently, unused,
submodules that are nested in IREE.

## Dependencies

### For Linux

#### Driver

Checkout `xdna-driver`, using commit `20e1f74`:
```
git clone git@github.com:amd/xdna-driver.git
cd <root-of-source-tree>
# get code for submodules
git checkout 20e1f74
git submodule update --init --recursive
```

Remove any previously installed drivers, if applicable.
```
packages=$(dpkg -l | awk '/^ii/ && $2 ~ /^xrt/ { print $2 }')
sudo apt-get remove -y $packages
cd <root-of-source-tree>
rm xrt/build/Release/*.deb
rm build/Release/*.deb
```

Follow the instructions to build and install the driver module: [xdna-driver](https://github.com/amd/xdna-driver/tree/20e1f747887c5889600e91f41e6812a98d285349).

#### LLVM-AIE (Peano)

You will need at least Peano/llvm-aie to be installed in your system to run e2e examples as it's needed for compiling AIE core code. For best performance (but slower compilation times), you will also need Chess.

To install llvm-aie in the current working directory:

```
bash <path-to-iree-amd-aie>/build_tools/download_peano.sh
```

Now, you should see a directory named `llvm-aie` in your current working directory.

After building IREE, you can then run e2e tests by passing `--peano_dir=<path-to-llvm-aie>` to tests, see [Testing](#testing).

#### Chess

For best performance and to run all tests, you can install Chess in the following way:

1. Install Vitis™ AIE Essentials from [Ryzen AI Software 1.3 Early Accesss](https://account.amd.com/en/member/ryzenai-sw-ea.html#tabs-a5e122f973-item-4757898120-tab).
   ``` bash
      tar -xzvf ryzen_ai_1.3.1-ea-lnx64-20250116.tgz
      cd ryzen_ai_1.3.1-ea-lnx64-20250116
      mkdir vitis_aie_essentials
      mv vitis_aie_essentials*.whl vitis_aie_essentials
      cd vitis_aie_essentials
      unzip vitis_aie_essentials*.whl
   ```
2. Set up an AI Engine license.
    1. Get a local license for AI Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense).
    2. Copy your license file (Xilinx.lic) to your preferred location, e.g. `/opt/Xilinx.lic`.

After building IREE, you can then run e2e tests by passing `--vitis_dir=<path-to-vitis-aie-essentials>` to tests, see [Testing](#testing). Note however that you need to export the path to the AI Engine license for successful compilation:
```
export XILINXD_LICENSE_FILE=<path-to-Xilinx.lic>
```

## Building (along with IREE)

### Just show me the CMake

```
cd iree-amd-aie
cmake \
  -B <WHERE_YOU_WOULD_LIKE_TO_BUILD> \
  -S third_party/iree \
  -DIREE_CMAKE_PLUGIN_PATHS=$PWD \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_INPUT_STABLEHLO=OFF \
  -DIREE_INPUT_TORCH=OFF \
  -DIREE_INPUT_TOSA=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_EXTERNAL_HAL_DRIVERS=amdxdna \
  -DCMAKE_INSTALL_PREFIX=<WHERE_YOU_WOULD_LIKE_TO_INSTALL>
cmake --build <WHERE_YOU_WOULD_LIKE_TO_BUILD>
```

### Instructions

The bare minimum configure command for IREE with the amd-aie plugin

```
cmake \
  -B <WHERE_YOU_WOULD_LIKE_TO_BUILD> \
  -S <IREE_REPO_SRC_DIR> \
  -DIREE_CMAKE_PLUGIN_PATHS=<IREE_AMD_AIE_REPO_SRC_DIR> \
  -DIREE_BUILD_PYTHON_BINDINGS=ON
```

Very likely, you will want to use `ccache` and `lld` (or some other modern linker like [mold](https://github.com/rui314/mold))

```
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
```

If you don't plan on using any of IREE's frontends or backends/targets (e.g., you're doing work on this code base itself),
you can opt-out of everything (except the `llvm-cpu` backend) with

```
  -DIREE_INPUT_STABLEHLO=OFF \
  -DIREE_INPUT_TORCH=OFF \
  -DIREE_INPUT_TOSA=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON
```

With the above you can also skip cloning the `stablehlo` and `torch-mlir` submodules/repos but in this case you will need to add

```
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
```

If you're "bringing your own LLVM", i.e., you have a prebuilt/compiled distribution of LLVM you'd like to use, you can add

```
  -DIREE_BUILD_BUNDLED_LLVM=OFF
```

In this case you will need `lit` somewhere in your environment and you will need to add to CMake `-DLLVM_EXTERNAL_LIT=<SOMEWHERE>`
(e.g., `pip install lit; SOMEWHERE=$(which lit)`).

See [Bringing your own LLVM](#bringing-your-own-llvm) below for more information on using prebuilt/compiled distributions of LLVM.

## Testing

Lit tests (i.e., compiler tests) specific to AIE can be run with something like

```
cd <WHERE_YOU_WOULD_LIKE_TO_BUILD>
ctest -R amd-aie --output-on-failure -j 10
```

(the `-j 10` runs `10` tests in parallel)

Other tests, which run on device, are in the `build_tools` subdirectory.
See [build_tools/ci/run_all_runtime_tests.sh](build_tools/ci/run_all_runtime_tests.sh) for an example script that shows how to run all the runtime tests.

## Pro-tips

### Bringing your own LLVM

When using a pre-built distribution of LLVM, getting the right/matching build, that works with IREE, is tough (besides the commit hash, there are various flags to set).
To enable adventurous users to avail themselves of `-DIREE_BUILD_BUNDLED_LLVM=OFF` we cache/store/save the LLVM distribution for every successful CI run.
These can then be downloaded by checking the artifacts section of any recent CI run's [Summary page](https://github.com/nod-ai/iree-amd-aie/actions/runs/10713474448):

<p align="center">
<img src="https://github.com/user-attachments/assets/97fdeff2-41af-4a6d-a072-6ef0a1ec5695" width="500">
</p>


### Debugging HAL

You can turn on HAL API tracing by adding to CMake:

```
-DIREE_ENABLE_RUNTIME_TRACING=ON
-DIREE_TRACING_PROVIDER=console
// optional but recommended
-DIREE_TRACING_CONSOLE_FLUSH=1
```

This will you show you all the HAL APIs that have `IREE_TRACE_ZONE_BEGIN ... IREE_TRACE_ZONE_END` that are hit during a run/execution (of, e.g., `iree-run-module`).

You can turn on VM tracing by adding to CMake:

```
-DIREE_VM_EXECUTION_TRACING_ENABLE=1
-DIREE_VM_EXECUTION_TRACING_FORCE_ENABLE=1
// optional
-DIREE_VM_EXECUTION_TRACING_SRC_LOC_ENABLE=1
```

This will show you all of the [VM dispatches](https://github.com/iree-org/iree/blob/0e8a5737dfe49a48a4e9c15ba7a7d24dd2fd7623/runtime/src/iree/vm/bytecode/dispatch.c#L661) that actually occur during a run/execution.
Note, this is roughly equivalent to [passing](https://github.com/nod-ai/iree-amd-aie/blob/737092791dc2428ad71bc172f69804c583b0f60e/build_tools/ci/run_matmul_test.sh#L420) `--compile-to=vm` to `iree-compile`.

## Architectural overview (out of date)

![image](https://github.com/nod-ai/iree-amd-aie/assets/74956/3fa73139-5fdf-4658-86c3-0705352c4ea0)

---

## `bert-onnx` branch

BERT를 IREE-AMD-AIE 백엔드(NPU)에서 이기종(CPU+NPU)으로 end-to-end 실행하기 위한 작업 브랜치.
`vgg16-onnx`에서 갈라져 나왔으며, 검증 완료된 항목만 포함합니다 (진행 중인 작업은 별도 브랜치에서 관리).

- BERT-tiny / BERT-base CPU+NPU 이기종 e2e 실행 (`models/bert_tiny/`, `models/bert_base/`)
- batch matmul의 tile-multiple padding 누락 수정 (row-overflow 버그 근본 수정)
- int8 batched matmul에서 **batch 0 출력이 통째로 0**이 되던 문제 근본 수정
  (objectFifo lock을 producer마다 하나씩 — 자세한 내용은 아래)
- ONNX → dispatch 프론트엔드 lowering 과정 설명 문서 (`docs/2026-08-16_frontend_lowering_passes.md`)

자세한 내용/알려진 한계는 각 모델 README 참고.

### batch-0 all-zero 수정 (2026-09-11)

AIE lock은 카운팅 세마포어라 `AcquireGreaterEqual(N)`이 "릴리즈가 N번 있었다"만 보장하고
**누가 했는지는 담지 못합니다.** 독립적인 DMA producer 여럿이 한 버퍼의 겹치지 않는 구역을
나눠 쓰고 consumer 하나가 전체를 꺼내가는 구조에서, lock 쌍을 하나만 두고 크레딧을
`numProducers * depth`로 부풀리면 앞서 나간 producer의 릴리즈가 아직 아무것도 쓰지 않은
producer의 몫까지 열어버립니다. consumer는 초기화되지 않은 구역(=0)을 그대로 실어가고,
그 자리가 batch 0이면 **배치 하나가 통째로 0**이 됩니다.

producer마다 lock 쌍을 주어 고쳤습니다. AIE2의 DMA BD는 acquire/release lock을 각각
하나씩만 들 수 있으므로 consumer 전송을 producer 슬라이스별 BD로 분할하는 것이 함께
따라오며, 분할이 원본과 같은 바이트를 같은 순서로 옮기는지는 컴파일 타임에 검사합니다.

이전에 있던 우회책(코어마다 첫 lock release 앞 busy-wait)은 제거했습니다. 증상을 가리기만
했고 규모가 커지면 스스로 오답을 만들었습니다 — 그 delay가 켜져 있으면 12층 attention의
배치 matmul 12개가 **전부** 틀렸습니다(0.12~0.45, `col % 8 == 0`에 집중).

검증: int8 모델 27개 · 320회 실행. 같은 양자화 ONNX를 onnxruntime(CPU)로 돌린 값 대비
**26개가 비트 단위 일치**(max abs error 0), 전부 실행간 출력 1종. 나머지 1개(16층 체인)는
결정론적이며 차이가 모두 출력 양자화 step의 정수배 = 반올림 tie.

### ⚠️ int8 batched matmul을 쓰려면 별도 패치가 필요합니다

torch-mlir에 `aten.bmm`의 int8 양자화 경로가 없습니다(2D `aten.mm`은 이미 있음). ONNX의
배치 MatMul은 `aten.matmul`이 아니라 `aten.bmm`으로 임포트되므로 양자화가 조용히 건너뛰어집니다.
커밋 1개(2개 파일, +66줄)로 해결되지만, 해당 서브모듈의 리모트가 우리 포크가 아니라 upstream
`iree-org/torch-mlir`이라 푸쉬할 수 없어 **패치 파일로 따로 공유**합니다.

`third_party/iree` 포인터는 `origin/vgg16-onnx`/`origin/dev`와 같은 공유 커밋을 가리키므로
그대로 체크아웃됩니다. 위 두 컴파일러 수정은 플러그인 전용이라 서브모듈 리비전과 무관하게
빌드됩니다.

**알려진 한계**: per-channel(축별) 가중치 스케일은 아직 지원되지 않습니다
(`linalg.quantized_matmul`이 zero-point를 스칼라로만 받아 표현 불가 → f32 폴백 후 백엔드에서
컴파일 실패). 양자화는 per-tensor로 하십시오 (`quantize_static(..., per_channel=False)`, 기본값).
