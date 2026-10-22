#!/usr/bin/env bash
# [container] Incremental build with adaptive -j (bounded by cores AND RAM). Run from /workspace.
# Builds build/ (extra args pass through), then iree-run-module in build-trace/
# so the console-tracing runtime always matches the sources of the main build.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$here/../config.sh"
cmake --build build -j "$JOBS" "$@"
cmake --build build-trace -j "$JOBS" --target iree-run-module
