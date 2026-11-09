#!/usr/bin/env bash
# [host, 받는 측] Run the deploy container with NPU + GPU passthrough.
# The current directory is mounted at /work (the working directory; ONNX script
# outputs go to /work/onnx). The ONNX file and output directory must be inside it.
#   run-deploy.sh <compile|run|e2e> <args...>   ONNX script (scripts/onnx/README.md)
#   run-deploy.sh [command...]                  command (default bash)
#   e.g. ./run-deploy.sh e2e vgg16-12.onnx --numeric bf16
# Serialized cluster-wide via npu.lock: the NPU is one physical device, and concurrent real
# runs from different users can race/hang it (docs/2026-07-06_env_setup/DEV_CONTAINER.md §5).
# This whole session holds the lock, so keep it short-lived (run, don't idle in the shell).
# Check who's using it without waiting: scripts/lock/status.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
. "$here/../lock/lock-common.sh"
lock_ensure_dir

args=(--rm)
[ -t 0 ] && args+=(-it)
# host uid/gid, so files under /work stay owned by the host user (names resolved
# through the host passwd/group)
args+=(--user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro)
npu="$(npu_device)"
# --ulimit memlock=-1: amdxdna HOST_ONLY BOs pin memory (RLIMIT_MEMLOCK); the
# default container cap (often 8MB) makes large im2col/activation BOs fail with
# errno 11 (EAGAIN). Unlock it so full-resolution models (e.g. VGG) can run.
# --group-add: the non-root container user can open a 0660 root:render node.
if [ -n "$npu" ]; then
  args+=(--device="/dev/accel/$npu" --ulimit memlock=-1)
  gid="$(device_gid "$npu")"
  [ -n "$gid" ] && args+=(--group-add "$gid")
fi
gpu="$(gpu_device)"
if [ -n "$gpu" ]; then
  args+=(--device="$gpu")   # GPU (Vulkan) render node
  gid="$(gpu_gid "$gpu")"
  [ -n "$gid" ] && args+=(--group-add "$gid")
fi
args+=(-v "$LOCK_ROOT:$LOCK_ROOT")
args+=(-v "$PWD:/work" -w /work -e HOME=/work)

case "${1:-}" in
  compile|run|e2e)
    cmd="$1"; shift
    lock_run "$NPU_LOCK_FILE" "npu" -- docker run "${args[@]}" "$IMAGE_DEPLOY" \
      python3 "/opt/iree-amd-aie/scripts/onnx/$cmd.py" "$@" ;;
  *)
    lock_run "$NPU_LOCK_FILE" "npu" -- docker run "${args[@]}" "$IMAGE_DEPLOY" "${@:-bash}" ;;
esac
