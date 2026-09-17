#!/usr/bin/env bash
# Check NPU/build lock status WITHOUT blocking or waiting — run this before a build or an
# NPU run to see whether someone else on this shared host is already using it.
# Works from the host or from inside any of the dev/deploy/debug containers (they all
# bind-mount the same lock directory). See scripts/lock/lock-common.sh for how the locks work.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/lock-common.sh"

lock_status "$NPU_LOCK_FILE" "NPU"
lock_status "$BUILD_LOCK_FILE" "build"
