# iree-amd-aie dev/deploy container (Option B, multi-stage)
# Phase 1 (this iteration): base-deps + dev only.
# Phase 2 (later): builder + runtime for a self-contained deploy image.
ARG UBUNTU_BASE=ubuntu:24.04

# ============================================================
# base-deps : common build environment (no source baked in)
#   Rationale for the dependency set:
#     - cmake: IREE requires 3.26...3.29; Ubuntu 24.04 apt cmake (3.28.3) satisfies it.
#     - python3 + python3-numpy: IREE_BUILD_PYTHON_BINDINGS=ON needs NumPy headers
#       (FindPython3 NumPy component). apt python3-numpy avoids PEP 668 pip issues.
#     - libudev-dev / uuid-dev: build deps for third_party/XRT (amdxdna SHIM),
#       which is vendored as a submodule -> no separate amd/xdna-driver clone needed.
#     - ccache: optional (speeds up rebuilds). Remove it together with the
#       CMAKE_*_COMPILER_LAUNCHER=ccache flags if a truly minimal image is wanted.
# ============================================================
FROM ${UBUNTU_BASE} AS base-deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git \
      build-essential cmake ninja-build clang lld \
      libudev-dev uuid-dev \
      python3 python3-dev python3-venv python3-pip python3-numpy \
      ccache \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# dev : development environment (source is bind-mounted, not baked)
#   Source + submodules are prepared on the HOST (recursive clone) and mounted
#   at /workspace. The container only builds/runs.
# ============================================================
FROM base-deps AS dev
# Model-import deps so dev can also exercise the full ONNX/PyTorch -> NPU flow
# (superset of the deploy image). Kept in a venv (PEP 668) at /opt/venv; the build
# itself still uses the system python3, so validated build behavior is unchanged.
# torch pinned to the version validated end-to-end (deploy runtime uses the same).
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir onnx sympy \
 && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.1+cpu
# To test model import in dev (after building with IREE_BUILD_PYTHON_BINDINGS=ON):
#   PYTHONPATH=/workspace/build/compiler/bindings/python /opt/venv/bin/python -m iree.compiler.tools.import_onnx ...

# ubuntu:24.04 ships a default 'ubuntu' user at uid/gid 1000. Reuse it so files
# created through the bind-mount are owned by the host user (typically uid 1000),
# not root. VS Code Dev Containers remaps this uid to the host's automatically
# (updateRemoteUserUID defaults to true); for plain `docker run` on a host whose
# uid is not 1000, pass `--user $(id -u):$(id -g)`.
USER ubuntu
WORKDIR /workspace

# ============================================================
# builder : deploy build (Release, from a pinned commit)  <- Phase 2
#   Hermetic: clones the fork at a pinned commit (not the local tree).
#   For local verification, override the repo to the local clone:
#     --build-arg IREE_AMD_AIE_REPO=file:///home/ace/Projects/iree-amd-aie
#     --build-arg IREE_AMD_AIE_COMMIT=<sha with the v22 peano pin>
#   BUILD_JOBS caps parallelism on constrained hosts (laptops); servers can raise it.
# ============================================================
FROM base-deps AS builder
ARG IREE_AMD_AIE_REPO=https://github.com/ace-knu/iree-amd-aie.git
ARG IREE_AMD_AIE_COMMIT
ARG BUILD_JOBS=6

RUN git clone --recursive "${IREE_AMD_AIE_REPO}" /src/iree-amd-aie \
 && cd /src/iree-amd-aie \
 && git checkout "${IREE_AMD_AIE_COMMIT}" \
 && git submodule update --init --recursive \
 && bash build_tools/download_peano.sh          # Peano v22 per peano_commit_linux.txt (fork release mirror)

# Release (assertions OFF), frontends ON, amdxdna.
# PYTHON_BINDINGS=ON so the version-matched model importers ship with the compiler:
#   iree.compiler.tools.import_onnx (ONNX->MLIR) and iree.compiler.extras.fx_importer (PyTorch->MLIR).
RUN cd /src/iree-amd-aie && cmake -B /build -S third_party/iree -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DIREE_CMAKE_PLUGIN_PATHS=$PWD -DIREE_BUILD_PYTHON_BINDINGS=ON \
      -DIREE_INPUT_TORCH=ON -DIREE_INPUT_STABLEHLO=ON -DIREE_INPUT_TOSA=ON \
      -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
      -DIREE_TARGET_BACKEND_LLVM_CPU=ON -DIREE_EXTERNAL_HAL_DRIVERS=amdxdna \
      -DIREE_BUILD_TESTS=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
 && cmake --build /build -j "${BUILD_JOBS}" \
      --target iree-compile iree-run-module IREECompilerPythonModules \
 && cp -rL /build/compiler/bindings/python /opt/iree-python   # self-contained (deref symlinks)

# ============================================================
# runtime : final deploy image (artifacts + model importers)  <- Phase 2
#   No source, no C/C++ toolchain, no mount. Recipient feeds ONNX/PyTorch/MLIR models.
#   amdxdna XRT SHIM is statically linked into the tools.
# ============================================================
FROM ${UBUNTU_BASE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# python3 for the importers; venv avoids PEP 668. libstdc++6 for the native tools/.so.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libstdc++6 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# Model-import deps (version-matched importer is our own bindings; these are the runtime pkgs).
#   onnx: ONNX import.  torch(+sympy): PyTorch fx import (CPU wheel to keep size down).
#   torch pinned to 2.12.1+cpu = the version validated end-to-end (ONNX/PyTorch -> NPU).
#   (torch-mlir declares a nightly torch 2.12.0.dev*, but nightlies get pruned; the stable
#    2.12.1 is durable and verified compatible with our fx_importer.)
RUN /opt/venv/bin/pip install --no-cache-dir onnx sympy \
 && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.1+cpu

# Native tools + Peano
COPY --from=builder /build/tools/iree-compile      /opt/iree-amd-aie/bin/iree-compile
COPY --from=builder /build/tools/iree-run-module   /opt/iree-amd-aie/bin/iree-run-module
COPY --from=builder /build/lib/libIREECompiler.so  /opt/iree-amd-aie/lib/libIREECompiler.so
COPY --from=builder /src/iree-amd-aie/llvm-aie     /opt/llvm-aie
# Version-matched python compiler package (provides import_onnx + fx_importer)
COPY --from=builder /opt/iree-python               /opt/iree-python

ENV PATH="/opt/venv/bin:/opt/iree-amd-aie/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/iree-amd-aie/lib"
ENV PEANO_INSTALL_DIR="/opt/llvm-aie"
ENV PYTHONPATH="/opt/iree-python"
# NPU 실행: docker run --device=/dev/accel/accelN (호스트 KMD 준비는 HOST_PREREQUISITES.md)
