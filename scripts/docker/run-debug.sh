#!/usr/bin/env bash
# [host] Run the pipeline-dump debug tool inside the dev container.
# Activates the venv + PYTHONPATH, then runs scripts/debug/pipeline_dump.py "$@".
# NPU passthrough + repo mounted at /workspace (same as run-dev.sh), non-interactive.
# Serialized cluster-wide via npu.lock — this always runs a real NPU workload, so it waits
# its turn behind any other user's npu.lock holder (docs/2026-07-06_env_setup/DEV_CONTAINER.md
# §5). Check who's using it without waiting: scripts/lock/status.sh.
#   e.g. ./scripts/docker/run-debug.sh \
#          --model models/mlp_2layer/mlp_2layer.onnx --function mlp_2layer \
#          --input x.npy --input w1.npy --input w2.npy --label t1
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
. "$here/../lock/lock-common.sh"
root="$(git -C "$here" rev-parse --show-toplevel)"
lock_ensure_dir

args=(--rm --user "$(id -u):$(id -g)")
# resolve host uid/gid to names inside the container (avoids "I have no name!" /
# "groups: cannot find name" when the host uid/gid isn't the image's ubuntu 1000)
args+=(-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro)
npu="$(npu_device)"
if [ -n "$npu" ]; then
  args+=(--device="/dev/accel/$npu")
  # add the device's owning group so the non-root container user can open it
  # even when the node is 0660 root:render (not just the amdxdna-udev 0666 case)
  gid="$(device_gid "$npu")"
  [ -n "$gid" ] && args+=(--group-add "$gid")
  # amdxdna HOST_ONLY BOs pin memory (RLIMIT_MEMLOCK); the default container cap
  # (often 8MB) makes large im2col/activation BOs fail with errno 11 (EAGAIN).
  # Unlock it so full-resolution models (e.g. VGG at 224x224) can run.
  args+=(--ulimit memlock=-1)
fi
args+=(-v "$LOCK_ROOT:$LOCK_ROOT")
args+=(-v "$root:/workspace" -w /workspace -e HOME=/workspace -e PEANO_INSTALL_DIR=/workspace/llvm-aie)

lock_run "$NPU_LOCK_FILE" "npu" -- docker run "${args[@]}" "$IMAGE_DEV" bash -lc '
  source /opt/venv/bin/activate
  export PYTHONPATH=/workspace/build/compiler/bindings/python
  exec python3 scripts/debug/pipeline_dump.py "$@"
' _ "$@"
