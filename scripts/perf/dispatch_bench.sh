#!/usr/bin/env bash
# [host] Per-dispatch NPU timing via IREE's executable benchmark dump. Compiles <src.mlir> for
# the NPU only (the dump pass does not support multi-device modules), emits one benchmark module
# per NPU executable, compiles each, and runs it with iree-benchmark-module --batch_size=N.
#
#   usage: scripts/perf/dispatch_bench.sh <src.mlir> <outdir> [--batch N] [--reps R] [extra iree-compile flags...]
#
# Caveat: the dumped benchmark feeds each dispatch one uninitialised/zero buffer, so a
# data-dependent kernel (the scalar bf16 FMA is: zero operands run faster) is timed on that
# data, not on real activations. Use it for relative comparisons of the same dispatch.
set -uo pipefail
root="$(git rev-parse --show-toplevel)"; cd "$root"
. scripts/config.sh; . scripts/lock/lock-common.sh; lock_ensure_dir
src=$1; out=$2; shift 2
batch=4; reps=1
while [ $# -gt 0 ]; do case "$1" in --batch) batch=$2; shift 2;; --reps) reps=$2; shift 2;; *) break;; esac; done
mkdir -p "$out"
F="--iree-hal-target-device=npu=amdxdna --iree-hal-default-device=npu --iree-amdaie-target-device=npu4 \
   --iree-amd-aie-peano-install-dir=/workspace/llvm-aie --iree-amdaie-enable-vectorization-passes=false $*"
dkc=(docker run --rm --user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro --cpus=4 --memory=12g
     -v "$root:/workspace" -w /workspace -e HOME=/workspace -e LD_LIBRARY_PATH=/workspace/build/lib "$IMAGE_DEV")
"${dkc[@]}" bash -c "build/tools/iree-compile $src -o $out/full.vmfb $F --iree-hal-dump-executable-benchmarks-to=$out 2>&1 | grep -iE 'error|warning: Executable' " 
ls "$out"/*_benchmark.mlir >/dev/null 2>&1 || { echo "no benchmark modules were dumped"; exit 1; }
npu="$(npu_device)"; gid="$(device_gid "$npu")"
dkn=("${dkc[@]:0:$((${#dkc[@]}-1))}" --device="/dev/accel/$npu" --group-add "$gid" --ulimit memlock=-1 -v "$LOCK_ROOT:$LOCK_ROOT" "$IMAGE_DEV")
printf '%-70s %10s %6s\n' dispatch "us/dispatch" reps
for m in "$out"/*_benchmark.mlir; do
  v="${m%.mlir}.vmfb"
  "${dkc[@]}" bash -c "build/tools/iree-compile $m -o $v $F 2>&1 | grep -iE 'error'" 
  for i in $(seq 1 "$reps"); do
    t=$("${dkn[@]}" ./scripts/lock/with-npu-lock.sh build/tools/iree-benchmark-module --device=amdxdna --module="$v" \
        --batch_size="$batch" --benchmark_min_time=1x --benchmark_repetitions=1 2>&1 | awk '/^BM_/{print $2, $3; exit}')
    printf '%-70s %17s %6s\n' "$(basename "$m" _benchmark.mlir | sed 's/^module_//')" "${t:-FAIL}" "$i/$reps"
  done
done
