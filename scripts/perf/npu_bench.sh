#!/usr/bin/env bash
# [host] Standard NPU timing for one vmfb function: iree-benchmark-module, N repetitions,
# under the cluster NPU lock, with provenance (git hash, vmfb sha256, flags) appended to a CSV.
#
#   usage: scripts/perf/npu_bench.sh <vmfb> <function> <label> [--reps N] [--csv FILE] -- <iree-benchmark-module --input=... args>
#   e.g.   scripts/perf/npu_bench.sh out/vgg.vmfb mxnet_converted_model vgg16_after --reps 3 -- --input=@input.npy
#
# Prints the per-repetition wall times and their median (real_time, ms; host wall clock around
# iree_vm_invoke + fence wait, see docs_GEMV/0910_perf_methodology.md). Each repetition is a
# separate process, so the first-iteration device/buffer setup is included in every sample.
set -uo pipefail
root="$(git rev-parse --show-toplevel)"; cd "$root"
. scripts/config.sh; . scripts/lock/lock-common.sh; lock_ensure_dir
vmfb=$1; fn=$2; label=$3; shift 3
reps=3; csv=log_shjo/perf/npu_bench.csv
while [ $# -gt 0 ]; do case "$1" in
  --reps) reps=$2; shift 2;; --csv) csv=$2; shift 2;; --) shift; break;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
mkdir -p "$(dirname "$csv")"
npu="$(npu_device)"; gid="$(device_gid "$npu")"
hash=$(git rev-parse --short HEAD); dirty=$([ -n "$(git status --porcelain --untracked-files=no -- compiler runtime 2>/dev/null)" ] && echo "+dirty" || echo "")
sha=$(sha256sum "$vmfb" | cut -c1-12); stamp=$(date -Is)
dk=(docker run --rm --user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
    --device="/dev/accel/$npu" --group-add "$gid" --ulimit memlock=-1 --cpus=4 --memory=14g
    -v "$LOCK_ROOT:$LOCK_ROOT" -v "$root:/workspace" -w /workspace -e HOME=/workspace
    -e LD_LIBRARY_PATH=/workspace/build/lib "$IMAGE_DEV")
times=()
for i in $(seq 1 "$reps"); do
  out=$("${dk[@]}" ./scripts/lock/with-npu-lock.sh build/tools/iree-benchmark-module --device=amdxdna --device=local-task \
        --module="$vmfb" --function="$fn" --benchmark_min_time=1x --benchmark_repetitions=1 "$@" 2>&1)
  t=$(echo "$out" | awk '/^BM_/{print $2; exit}')
  if [ -z "$t" ]; then echo "rep $i FAILED:"; echo "$out" | grep -iE "error|state" | head -3; echo "$stamp,$hash$dirty,$sha,$label,$fn,$i,FAIL,,\"$*\"" >> "$csv"; continue; fi
  times+=("$t"); echo "rep $i: $t ms"
  echo "$stamp,$hash$dirty,$sha,$label,$fn,$i,$t,ms,\"$*\"" >> "$csv"
done
[ ${#times[@]} -gt 0 ] && echo "median: $(printf '%s\n' "${times[@]}" | sort -n | awk '{a[NR]=$1} END{print (NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}') ms  (n=${#times[@]}, label=$label, git=$hash$dirty, vmfb=$sha)"
