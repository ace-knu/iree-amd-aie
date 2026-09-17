#!/usr/bin/env bash
# [container] Incremental build with adaptive -j (bounded by cores AND RAM). Run from /workspace.
# Serialized cluster-wide via build.lock: a full LLVM build's RAM budget isn't sized against
# what OTHER users' concurrent builds are using, so overlapping builds can overcommit RAM and
# freeze/crash the host (docs/2026-07-06_env_setup/DEV_CONTAINER.md §5). One build runs at a
# time on this host; others wait (and are told who they're waiting on).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
. "$here/../lock/lock-common.sh"
lock_run "$BUILD_LOCK_FILE" "build" -- cmake --build build -j "$JOBS" "$@"
