#!/usr/bin/env bash
# [host] Run an ONNX model script (scripts/onnx/<cmd>.py) inside the dev container.
# Activates the venv + PYTHONPATH, then runs scripts/onnx/<cmd>.py "$@".
# NPU passthrough + repo mounted at /workspace (same as run-dev.sh), non-interactive.
# The ONNX file and output directory must be inside the repo (only it is mounted).
#   e.g. ./scripts/docker/run-onnx.sh e2e models/vgg16/vgg16-12.onnx --numeric bf16
# run/e2e that put work on the NPU are serialized cluster-wide via npu.lock, like
# run-debug.sh and run-deploy.sh (docs/2026-07-06_env_setup/DEV_CONTAINER.md §5);
# compile and CPU/GPU-only runs do not take it. Check the holder: scripts/lock/status.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
. "$here/../lock/lock-common.sh"
root="$(git -C "$here" rev-parse --show-toplevel)"

cmd="${1:-}"
case "$cmd" in
  compile|run|e2e) shift ;;
  *) echo "usage: $0 <compile|run|e2e> <args...>" >&2; exit 2 ;;
esac

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
gpu="$(gpu_device)"
if [ -n "$gpu" ]; then
  # GPU (Vulkan): render node plus its owning group, as for the NPU above
  args+=(--device="$gpu")
  gid="$(gpu_gid "$gpu")"
  [ -n "$gid" ] && args+=(--group-add "$gid")
fi
args+=(-v "$root:/workspace" -w /workspace -e HOME=/workspace -e PEANO_INSTALL_DIR=/workspace/llvm-aie)

# Does this invocation run anything on the NPU? compile never does. e2e uses the
# devices of --devices (default local-host,amdxdna-npu4); run uses the devices
# recorded in <outdir>/config.json.
uses_npu=0
if [ "$cmd" = e2e ]; then
  uses_npu=1; prev=""
  for a in "$@"; do
    case "$a" in --devices=*) case "${a#--devices=}" in *amdxdna*) uses_npu=1 ;; *) uses_npu=0 ;; esac ;; esac
    if [ "$prev" = "--devices" ]; then case "$a" in *amdxdna*) uses_npu=1 ;; *) uses_npu=0 ;; esac; fi
    prev="$a"
  done
elif [ "$cmd" = run ]; then
  uses_npu=1
  cfg="${1:-}/config.json"
  if [ -f "$cfg" ] && ! grep -q amdxdna "$cfg"; then uses_npu=0; fi
fi

run=(docker run "${args[@]}" "$IMAGE_DEV" bash -lc '
  source /opt/venv/bin/activate
  export PYTHONPATH=/workspace/build/compiler/bindings/python
  cmd="$1"; shift
  exec python3 "scripts/onnx/$cmd.py" "$@"
' _ "$cmd" "$@")
if [ "$uses_npu" = 1 ]; then
  lock_run "$NPU_LOCK_FILE" "npu" -- "${run[@]}"
else
  exec "${run[@]}"
fi
