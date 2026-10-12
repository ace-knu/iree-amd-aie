#!/usr/bin/env bash
# Run the 64 held-out sentences (quant/eval_ids.npy) through a vmfb on the NPU, under the
# shared NPU lock, 16 sentences per lock hold.
#   run_npu.sh <model.vmfb> <out_dir> [build_dir]
# Writes <out_dir>/in_NN.npy and npu_NN.npy; then score with npu_eval.py.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../../.." && pwd)
vmfb=$(realpath "$1"); out=$(realpath -m "$2"); build=$(realpath "${3:-$root/build}")
mkdir -p "$out"
python3 - "$here/../quant/eval_ids.npy" "$out" <<'PY'
import sys, numpy as np
ids = np.load(sys.argv[1])
for i, r in enumerate(ids): np.save(f"{sys.argv[2]}/in_{i:02d}.npy", r[None].astype(np.int64))
PY
for s in 0 16 32 48; do
  "$root/scripts/lock/with-npu-lock.sh" bash -c "
    for i in \$(seq -w $s $((s + 15))); do
      timeout 300 '$build/tools/iree-run-module' --device=amdxdna --device=local-task \
        --module='$vmfb' --function=main_graph --input=@'$out'/in_\$i.npy \
        --output=@'$out'/npu_\$i.npy > /dev/null || echo \"sentence \$i failed\"
    done"
  echo "sentences $s-$((s + 15)) done"
done
