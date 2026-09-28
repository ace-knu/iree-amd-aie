#!/usr/bin/env bash
# Build the int8 12-layer BERT-base encoder that runs NPU+CPU with LayerNorm,
# softmax and GELU on the NPU, from the fp32 export.
#
#   bert_base.onnx                     fp32, export_bert_base.py
#   -> bert_base_int8.onnx             per-tensor int8 QDQ on the 96 MatMuls
#   -> bert_base_kpadq_int8.onnx       Q/K/V biases folded into K (Pad)
#   -> enc12_i8.onnx                   the encoder alone: int8 in, f32 out
#   -> enc12a64_k_i8.onnx              every other bias folded into K; column value
#                                      picked per matmul (--pad-value auto)
#   -> enc12hsk_k_i8.onnx              per-head Q/K/V weights; K's transpose left
#                                      on int8 for IREE to fold into QK^T
#   -> enc12x_k_i8.onnx                residuals read a slice of the widened
#                                      activations, so every widening is in place
#
# Usage: models/bert_base/prepare_encoder12_i8.sh [fp32.onnx] [out-dir]
# Needs onnx + onnxruntime (1.29.0 reproduces the reference models exactly).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=${1:-$here/bert_base.onnx}
out=${2:-$here/out}
mkdir -p "$out"

python3 "$here/quantize_bert_base.py" "$src" "$out/bert_base_int8.onnx"
python3 "$here/fold_bias_into_k.py" "$out/bert_base_int8.onnx" "$out/bert_base_kpadq_int8.onnx" \
  --include attention/self/query,attention/self/key,attention/self/value --pad-rank3
python3 "$here/extract_encoder_i8.py" "$out/bert_base_kpadq_int8.onnx" "$out/enc12_i8.onnx"
python3 "$here/fold_bias_into_k_general.py" "$out/enc12_i8.onnx" "$out/enc12a64_k_i8.onnx" \
  --pad-value auto --max-bias-lsb 64
python3 "$here/head_split.py" "$out/enc12a64_k_i8.onnx" "$out/enc12hsk_k_i8.onnx" --k
python3 "$here/hoist_kfold_pad.py" "$out/enc12hsk_k_i8.onnx" "$out/enc12x_k_i8.onnx"

echo
echo "done: $out/enc12x_k_i8.onnx  (input int8 [1,32,768], output f32 [1,32,768])"
