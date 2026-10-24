#!/usr/bin/env bash
# [host, 받는 측] Run the deploy container with NPU + GPU passthrough. Extra args = command (default bash).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"

args=(--rm -it)
npu="$(npu_device)"
# --ulimit memlock=-1: amdxdna HOST_ONLY BOs pin memory (RLIMIT_MEMLOCK); the
# default container cap (often 8MB) makes large im2col/activation BOs fail with
# errno 11 (EAGAIN). Unlock it so full-resolution models (e.g. VGG) can run.
[ -n "$npu" ] && args+=(--device="/dev/accel/$npu" --ulimit memlock=-1)
gpu="$(gpu_device)"
[ -n "$gpu" ] && args+=(--device="$gpu")   # GPU (Vulkan) render node

exec docker run "${args[@]}" "$IMAGE_DEPLOY" "${@:-bash}"
