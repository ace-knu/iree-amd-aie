#!/usr/bin/env bash
# Apply the one third_party change that cannot live in a submodule commit:
# torch-mlir's remote is upstream iree-org/torch-mlir, which we cannot push to.
# Everything IREE needs is already in the pinned third_party/iree commit
# (ace-knu/iree, branch bert-onnx). Idempotent; rebuild IREE afterwards.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)

apply() {  # <dir> <patch>
  local dir=$1 patch=$here/$2
  if git -C "$dir" apply --check --reverse "$patch" 2>/dev/null; then
    echo "already applied: $2"
  else
    git -C "$dir" apply "$patch"
    echo "applied:         $2"
  fi
}

apply "$root/third_party/iree/third_party/torch-mlir" 0001-torch-mlir-int8-aten-bmm.patch
