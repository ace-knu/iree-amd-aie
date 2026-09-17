#!/usr/bin/env bash
# Wrap any NPU-touching command with the cluster-wide NPU lock (scripts/lock/lock-common.sh).
# Use this for ad-hoc iree-run-module / xrt-smi runs inside a run-dev.sh shell.
# run-deploy.sh and run-debug.sh already wrap themselves automatically — no need to nest this
# around them.
#   e.g. ./scripts/lock/with-npu-lock.sh build/tools/iree-run-module --device=amdxdna \
#          --module=model.vmfb --function=main --input=...
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/lock-common.sh"

[ "$#" -ge 1 ] || { echo "usage: $0 <command...>" >&2; exit 2; }
lock_run "$NPU_LOCK_FILE" "npu" -- "$@"
