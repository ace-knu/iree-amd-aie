#!/usr/bin/env bash
# [host] Run the dev container: host uid/gid, NPU + GPU passthrough, repo mounted at /workspace.
# Extra args run as the command (default: bash).  e.g. ./run-dev.sh ./scripts/build/build.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
root="$(git -C "$here" rev-parse --show-toplevel)"

args=(--rm -it --user "$(id -u):$(id -g)")
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

exec docker run "${args[@]}" "$IMAGE_DEV" "${@:-bash}"
