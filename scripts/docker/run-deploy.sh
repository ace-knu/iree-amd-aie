#!/usr/bin/env bash
# [host, 받는 측] Run the deploy container with NPU passthrough. Extra args = command (default bash).
# Serialized cluster-wide via npu.lock: the NPU is one physical device, and concurrent real
# runs from different users can race/hang it (docs/2026-07-06_env_setup/DEV_CONTAINER.md §5).
# This whole session holds the lock, so keep it short-lived (run, don't idle in the shell).
# Check who's using it without waiting: scripts/lock/status.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
. "$here/../lock/lock-common.sh"
lock_ensure_dir

args=(--rm -it)
npu="$(npu_device)"
# --ulimit memlock=-1: amdxdna HOST_ONLY BOs pin memory (RLIMIT_MEMLOCK); the
# default container cap (often 8MB) makes large im2col/activation BOs fail with
# errno 11 (EAGAIN). Unlock it so full-resolution models (e.g. VGG) can run.
[ -n "$npu" ] && args+=(--device="/dev/accel/$npu" --ulimit memlock=-1)
args+=(-v "$LOCK_ROOT:$LOCK_ROOT")

lock_run "$NPU_LOCK_FILE" "npu" -- docker run "${args[@]}" "$IMAGE_DEPLOY" "${@:-bash}"
