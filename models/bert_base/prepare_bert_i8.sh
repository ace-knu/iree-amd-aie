#!/usr/bin/env bash
# Build the whole int8 BERT-base (input_ids -> last_hidden_state) for NPU+CPU:
# embeddings on the CPU, the 12-layer encoder on the NPU.
#
#   bert_base.onnx                     fp32, export_bert_base.py
#   -> bert_base_int8.onnx             per-tensor int8 QDQ on the 96 MatMuls
#   -> bert_base_kpadq_int8.onnx       Q/K/V biases folded into K (Pad)
#   -> bert_s_int8.onnx                batch pinned to 1, shape-only subgraphs
#                                      (position/token-type ids, mask) folded
#   -> bert_f_int8.onnx                every other bias folded into K
#   -> bert_hs_int8.onnx               per-head Q/K/V weights, K as transpose_b
#   -> bertx_int8.onnx                 residuals read slices of the widened
#                                      activations
#
# Usage: models/bert_base/prepare_bert_i8.sh [fp32.onnx] [out-dir]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=${1:-$here/bert_base.onnx}
out=${2:-$here/out}
mkdir -p "$out"

python3 "$here/quantize_bert_base.py" "$src" "$out/bert_base_int8.onnx"
python3 "$here/fold_bias_into_k.py" "$out/bert_base_int8.onnx" "$out/bert_base_kpadq_int8.onnx" \
  --include attention/self/query,attention/self/key,attention/self/value --pad-rank3
python3 "$here/fold_shape_consts.py" "$out/bert_base_kpadq_int8.onnx" "$out/bert_s_int8.onnx"
python3 "$here/fold_bias_into_k_general.py" "$out/bert_s_int8.onnx" "$out/bert_f_int8.onnx" \
  --pad-value auto --max-bias-lsb 64
python3 "$here/head_split.py" "$out/bert_f_int8.onnx" "$out/bert_hs_int8.onnx" --k
python3 "$here/hoist_kfold_pad.py" "$out/bert_hs_int8.onnx" "$out/bertx_int8.onnx"

echo
echo "done: $out/bertx_int8.onnx  (input input_ids int64 [1,32], output f32 [1,32,768]; function main_graph)"
