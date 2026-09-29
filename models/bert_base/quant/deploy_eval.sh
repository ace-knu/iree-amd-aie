#!/usr/bin/env bash
# Accuracy of int8 QDQ models, before and after the deployment folds (QKV K-fold,
# shape consts, every other bias into K) -- the folds change numerics (+0.07 on the
# real-sentence baseline). head_split / hoist do not change numerics, so stop here.
# Usage: quant/deploy_eval.sh <model.onnx>...   (writes <model>_fold.onnx next to each)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); B=$here/..
fold=()
for m in "$@"; do
  m=$(realpath "$m"); v=${m%.onnx}
  python3 "$B/fold_bias_into_k.py" "$m" "${v}_kp.onnx" \
    --include attention/self/query,attention/self/key,attention/self/value --pad-rank3 > /dev/null
  python3 "$B/fold_shape_consts.py" "${v}_kp.onnx" "${v}_s.onnx" > /dev/null
  python3 "$B/fold_bias_into_k_general.py" "${v}_s.onnx" "${v}_fold.onnx" \
    --pad-value auto --max-bias-lsb 128 > /dev/null
  rm -f "${v}_kp.onnx" "${v}_s.onnx"
  fold+=("$m" "${v}_fold.onnx")
done
cd "$here" && python3 eval.py "${fold[@]}"
