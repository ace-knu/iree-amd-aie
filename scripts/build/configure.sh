#!/usr/bin/env bash
# [container] cmake configure for IREE + amd-aie (dev flags: frontends ON, assertions ON, bindings ON).
# Run from /workspace. Extra args pass through to cmake (main build only).
# Also configures build-trace/: the runtime alone with console tracing, whose
# iree-run-module scripts/onnx/run.py --verbose uses for per-dispatch times.
set -euo pipefail
cmake -B build -S third_party/iree -G Ninja \
  -DIREE_CMAKE_PLUGIN_PATHS="$PWD" -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_INPUT_TORCH=ON -DIREE_INPUT_STABLEHLO=ON -DIREE_INPUT_TOSA=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_EXTERNAL_HAL_DRIVERS=amdxdna \
  -DIREE_HAL_DRIVER_VULKAN=ON -DIREE_TARGET_BACKEND_VULKAN_SPIRV=ON \
  -DIREE_BUILD_TESTS=ON -DIREE_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  "$@"
cmake -B build-trace -S third_party/iree -G Ninja \
  -DIREE_CMAKE_PLUGIN_PATHS="$PWD" -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_PYTHON_BINDINGS=OFF \
  -DIREE_ENABLE_ASSERTIONS=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_TASK=ON -DIREE_HAL_DRIVER_VULKAN=ON \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_EXTERNAL_HAL_DRIVERS=amdxdna \
  -DIREE_ENABLE_RUNTIME_TRACING=ON -DIREE_TRACING_PROVIDER=console \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
