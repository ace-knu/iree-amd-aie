#!/usr/bin/env bash
# [container] Vectorized MxKxN matmul (K=32, N=64) on npu4, checked against numpy.
#
#   bash scripts/repro/aie2p_bf16/matmul.sh <bf16|i8> <M> [runs] [extra iree-compile flags...]
#
#   bf16 256   -> one L2 wave: checks the bf16 -> bfp16 MAC lowering itself.
#   bf16 4096  -> 16 waves: reproduces the memtile join lock race (commit d2846c0d).
#   i8   4096  -> same race on the pre-existing int8 path (no bf16 involved).
#
# The output is split into 64-row slices (one per core row and wave). A slice that
# differs from numpy is "corrupted"; if it instead equals numpy's result for the
# same slice two waves later, the race overwrote it. See
# docs/patch_aie2p_bf16_vectorized.md.
#
# Timing: prints the compile time and, per run, the wall time of iree-run-module
# (process start to exit: loading, NPU setup and the dispatch, not just the
# kernel). BENCH=1 also runs iree-benchmark-module for 10 repetitions.
set -euo pipefail
source /opt/venv/bin/activate
cd /workspace
now_ms() { date +%s%3N; }

KIND=${1:?usage: matmul.sh <bf16|i8> <M> [runs] [flags...]}
M=${2:?usage: matmul.sh <bf16|i8> <M> [runs] [flags...]}
RUNS=${3:-5}
shift $(( $# > 3 ? 3 : $# ))
EXTRA=("$@")
K=32
N=64
OUT=/workspace/build/repro_aie2p_bf16/mm_${KIND}_${M}
mkdir -p "$OUT"

case "$KIND" in
  bf16) IN=bf16; ACC=f32; ZERO="0.0 : f32" ;;
  i8)   IN=i8;   ACC=i32; ZERO="0 : i32" ;;
  *) echo "kind must be bf16 or i8" >&2; exit 1 ;;
esac

cat > "$OUT/mm.mlir" <<EOF
func.func @mm(%x: tensor<${M}x${K}x${IN}>, %w: tensor<${K}x${N}x${IN}>) -> tensor<${M}x${N}x${ACC}> {
  %zero = arith.constant ${ZERO}
  %init = tensor.empty() : tensor<${M}x${N}x${ACC}>
  %fill = linalg.fill ins(%zero : ${ACC}) outs(%init : tensor<${M}x${N}x${ACC}>) -> tensor<${M}x${N}x${ACC}>
  %0 = linalg.matmul ins(%x, %w : tensor<${M}x${K}x${IN}>, tensor<${K}x${N}x${IN}>)
                     outs(%fill : tensor<${M}x${N}x${ACC}>) -> tensor<${M}x${N}x${ACC}>
  return %0 : tensor<${M}x${N}x${ACC}>
}
EOF

python3 - "$KIND" "$M" "$K" "$N" "$OUT" <<'PY'
import sys
import numpy as np
import ml_dtypes
kind, m, k, n, out = sys.argv[1], *map(int, sys.argv[2:5]), sys.argv[5]
rng = np.random.default_rng(9)
if kind == "i8":
    x = rng.integers(-50, 50, size=(m, k), dtype=np.int8)
    w = rng.integers(-50, 50, size=(k, n), dtype=np.int8)
else:
    x = (rng.standard_normal((m, k)) * 0.5).astype(ml_dtypes.bfloat16)
    w = (rng.standard_normal((k, n)) * 0.5).astype(ml_dtypes.bfloat16)
np.save(f"{out}/x.npy", x)
np.save(f"{out}/w.npy", w)
# numpy cannot load bfloat16 .npy files back, so keep exact wide copies to check with.
np.save(f"{out}/x_ref.npy", x.astype(np.int64 if kind == "i8" else np.float64))
np.save(f"{out}/w_ref.npy", w.astype(np.int64 if kind == "i8" else np.float64))
PY

t0=$(now_ms)
build/tools/iree-compile "$OUT/mm.mlir" -o "$OUT/mm.vmfb" \
  --iree-hal-target-backends=amd-aie \
  --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-amdaie-enable-vectorization-passes=true "${EXTRA[@]}"
echo "compile: $(( $(now_ms) - t0 )) ms"

for i in $(seq 1 "$RUNS"); do
  t0=$(now_ms)
  build/tools/iree-run-module --device=amdxdna --module="$OUT/mm.vmfb" --function=mm \
    --input=@"$OUT/x.npy" --input=@"$OUT/w.npy" --output=@"$OUT/out_$i.npy" > /dev/null
  echo $(( $(now_ms) - t0 )) > "$OUT/time_ms_$i.txt"
done

python3 - "$KIND" "$RUNS" "$OUT" <<'PY'
import sys
import numpy as np
kind, runs, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
gold = np.load(f"{out}/x_ref.npy") @ np.load(f"{out}/w_ref.npy")
if kind == "i8":
    same = lambda a, b: np.array_equal(a, b)
else:
    # bfp16 rounding stays far below 0.5 on these inputs; a wrong slice does not.
    same = lambda a, b: np.abs(a - b).max() < 0.5
WAVE, SLICE = 256, 64
total_bad = 0
for r in range(1, runs + 1):
    got = np.load(f"{out}/out_{r}.npy").astype(gold.dtype)
    bad = stale = 0
    for wave in range(gold.shape[0] // WAVE):
        for s in range(WAVE // SLICE):
            r0 = wave * WAVE + s * SLICE
            if same(got[r0:r0 + SLICE], gold[r0:r0 + SLICE]):
                continue
            bad += 1
            l0 = (wave + 2) * WAVE + s * SLICE
            if l0 < gold.shape[0] and same(got[r0:r0 + SLICE], gold[l0:l0 + SLICE]):
                stale += 1
    total_bad += bad
    corr = np.corrcoef(gold.ravel().astype(np.float64), got.ravel().astype(np.float64))[0, 1]
    ms = int(open(f"{out}/time_ms_{r}.txt").read())
    print(f"run {r}: {ms:6d} ms  corr {corr:.5f}  max|err| {np.abs(got - gold).max():.4g}  "
          f"corrupted 64-row slices {bad:3d} (equal to wave+2 result: {stale})")
print("RESULT:", "PASS (no corrupted slice)" if total_bad == 0 else "FAIL (race reproduced)")
PY

if [ "${BENCH:-0}" = 1 ]; then
  build/tools/iree-benchmark-module --device=amdxdna --module="$OUT/mm.vmfb" --function=mm \
    --input=@"$OUT/x.npy" --input=@"$OUT/w.npy" --benchmark_repetitions=10 2>&1 \
    | grep -E "real_time_(mean|median|stddev|cv)"
fi
