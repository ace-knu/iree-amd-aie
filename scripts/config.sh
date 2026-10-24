#!/usr/bin/env bash
# Shared config, sourced by the other scripts. Every value is overridable via env var.
# (No `set -e` here — this file is sourced.)

# dev image tag is per-user ($(id -un) suffix): the Docker daemon is shared
# system-wide (rootful), so a fixed tag would let one user's `build-dev.sh`
# silently reassign `iree-amd-aie:dev` for everyone on the host. Overridable.
IMAGE_DEV="${IMAGE_DEV:-iree-amd-aie:dev-$(id -un)}"
# IMAGE_DEPLOY stays shared: it is a lab-wide canonical deploy image built from a
# single agreed commit, so one shared tag per machine is intended (not a conflict).
IMAGE_DEPLOY="${IMAGE_DEPLOY:-iree-amd-aie:deploy}"
REPO_URL="${REPO_URL:-https://github.com/ace-knu/iree-amd-aie.git}"
BRANCH="${BRANCH:-dev}"
DEPLOY_TAR="${DEPLOY_TAR:-iree-amd-aie-deploy.tar.gz}"

# Adaptive build parallelism, bounded by BOTH cpu and RAM.
# LLVM/MLIR C++ compiles peak ~4 GB each; over-subscribing RAM freezes/OOMs the host.
# Default caps at 6 for stability; scales down on smaller hosts. Override: JOBS=N ./build.sh
if [ -z "${JOBS:-}" ]; then
  mem_gb=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
  JOBS=$(nproc); JOBS=$(( JOBS > 2 ? JOBS - 2 : 1 ))
  [ $(( mem_gb / 4 )) -lt "$JOBS" ] && JOBS=$(( mem_gb / 4 ))
  [ "$JOBS" -gt 6 ] && JOBS=6
  [ "$JOBS" -lt 1 ] && JOBS=1
fi

# First NPU accel device name (e.g. accel0); empty if none present.
npu_device() { ls /dev/accel/ 2>/dev/null | head -1; }

# Owning-group gid of an NPU device node, for `docker run --group-add`. Lets the
# container user open a 0660 root:render node without host-side group changes
# (docker --user drops supplementary groups). Arg defaults to npu_device().
device_gid() { local n="${1:-$(npu_device)}"; [ -n "$n" ] && stat -c '%g' "/dev/accel/$n"; }

# First GPU render node path (e.g. /dev/dri/renderD128); empty if none present.
gpu_device() { ls /dev/dri/renderD* 2>/dev/null | head -1; }
# Owning-group gid of the GPU render node (see device_gid).
gpu_gid() { local n="${1:-$(gpu_device)}"; [ -n "$n" ] && stat -c '%g' "$n"; }
