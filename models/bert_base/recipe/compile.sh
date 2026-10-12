#!/usr/bin/env bash
# Compile a deploy model from build_models.sh for NPU4 (+ CPU for what stays there).
#   compile.sh <bertx_int8.onnx> <out.vmfb> [build_dir]
# build_dir defaults to <repo>/build. Uses the peano install at $PEANO (default
# /workspace/llvm-aie, the dev container's). About 6 minutes.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../../.." && pwd)
src=$(realpath "$1"); out=$(realpath -m "$2"); build=$(realpath "${3:-$root/build}")
# The imported MLIR's path and file name end up in the vmfb (module and executable names,
# source locations), so the same model gives a byte-identical vmfb only when imported to
# the same path. Default: next to the output; set MLIR=<path> to pin it.
mlir=$(realpath -m "${MLIR:-${out%.vmfb}.mlir}"); mkdir -p "$(dirname "$mlir")"
trap 'rm -f "$mlir"' EXIT
PYTHONPATH="$build/compiler/bindings/python${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m iree.compiler.tools.import_onnx "$src" -o "$mlir"
# --iree-flow-inline-constants-max-byte-length=16384: the per-channel constant tables
# (up to 3072 x f32) have to be inlined so they can become core-local constants.
"$build/tools/iree-compile" "$mlir" -o "$out" \
  --iree-flow-inline-constants-max-byte-length=16384 \
  --iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir="${PEANO:-/workspace/llvm-aie}" \
  --iree-flow-enable-executable-deduplication=false \
  --iree-amd-aie-enable-chess-for-ukernel=false \
  --iree-amdaie-enable-ukernels=softmax,layernorm
ls -l "$out"
