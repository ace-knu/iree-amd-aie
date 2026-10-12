#!/usr/bin/env bash
# Build the four int8 BERT-base deploy models of the accuracy work, from the fp32
# export, and the ORT reference models used to check the NPU against.
#
#   build_models.sh [fp32.onnx] [out_dir]
#
#   stage  NPU (64 sentences)  what it adds
#   A      0.9503              SmoothQuant a=0.6, FC1 + FC2 weights per-channel
#   B      0.9690              A + asymmetric GELU output (FC2 input)
#   C      0.9773              rule-based recipe (structural SmoothQuant sites, per-site a
#                              from calibration, all weights per-channel except head-split
#                              Q/K/V, asymmetric where the scale shrinks >= 1.25x)
#   D      0.9845              C + GELU lower bound (FC1 output clipped below at -4)
#
# Outputs, in out_dir:
#   {A,B,C,D}/bertx_int8.onnx  deploy model (compile this)
#   {A,B,C,D}_ref.onnx         same model with the attention-output and FC2-output
#                              requantization as int16 fake quantization, which is what the
#                              compiler does on the NPU (--iree-amdaie-widen-residual-
#                              requantization, on by default). Evaluate this on ORT.
#
# Everything is deterministic with onnxruntime 1.29.0: the deploy models' md5 must match
# EXPECTED_MD5 below (checked at the end). Takes a few minutes on 2 cores.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
B=$(cd "$here/.." && pwd)
F=$(realpath "${1:-$B/bert_base.onnx}")
O=$(realpath -m "${2:-$PWD/recipe_out}")
C=$B/quant/calib_ids.npy
PY=${PYTHON:-python3}
mkdir -p "$O"; cd "$O"
log() { echo "[$(date +%T)] $*"; }
run() {  # run a step; show its output without onnx warnings; stop on failure
  local out; out=$("$PY" "$@" 2>&1) || { echo "$out"; log "FAILED: $*"; exit 1; }
  echo "$out" | grep -v "^W:onnx\|WARNING" | tail -3 || true
}

# The deploy folds of models/bert_base/prepare_bert_i8.sh, with the general bias fold
# swapped for the per-channel ($2 = pc) or per-channel + activation zero point ($2 = pcz)
# copy.
deploy() {  # <in.onnx> <pc|pcz> <out_dir>
  mkdir -p "$3"
  "$PY" "$B/fold_bias_into_k.py" "$1" "$3/kpadq.onnx" \
      --include attention/self/query,attention/self/key,attention/self/value --pad-rank3 > "$3/log.txt" 2>&1
  "$PY" "$B/fold_shape_consts.py" "$3/kpadq.onnx" "$3/s.onnx" >> "$3/log.txt" 2>&1
  "$PY" "$here/fold_bias_into_k_general_$2.py" "$3/s.onnx" "$3/f.onnx" \
      --pad-value auto --max-bias-lsb 128 >> "$3/log.txt" 2>&1
  "$PY" "$B/head_split.py" "$3/f.onnx" "$3/hs.onnx" --k >> "$3/log.txt" 2>&1
  "$PY" "$B/hoist_kfold_pad.py" "$3/hs.onnx" "$3/bertx_int8.onnx" >> "$3/log.txt" 2>&1
  rm -f "$3/kpadq.onnx" "$3/s.onnx" "$3/f.onnx" "$3/hs.onnx"
}

log "fp32 model: $F"
log "A: SmoothQuant a=0.6 (quant/smooth.py)"
run "$B/quant/smooth.py" "$F" s06.onnx "$C" 0.6
log "A: quantize, per-tensor and per-channel weights"
run "$B/quantize_bert_base.py" s06.onnx q_pt.onnx --calib-ids "$C" --method percentile --percentile 99.999
run "$here/quantize_pc.py" s06.onnx sq_pc.onnx --calib-ids "$C" --method percentile
run "$B/quant/fc2_int16.py" sq_pc.onnx c_pc.onnx
log "A: FC2, then FC1 weights per-channel"
run "$here/splice_fc2.py" q_pt.onnx c_pc.onnx q_hy.onnx
run "$here/splice_mix.py" q_hy.onnx c_pc.onnx q_mix.onnx
log "A: deploy folds"
deploy q_mix.onnx pc A

log "B: asymmetric GELU output"
run "$here/asym_q.py" q_mix.onnx s06.onnx "$C" q_asym.onnx
log "B: deploy folds"
deploy q_asym.onnx pcz B
rm -f s06.onnx q_pt.onnx sq_pc.onnx c_pc.onnx q_hy.onnx q_mix.onnx q_asym.onnx

log "C: per-site a from calibration"
run "$here/alpha_rule.py" "$F" "$C" alpha.json
log "C: structural SmoothQuant with per-site a"
run "$here/smooth_rule.py" "$F" s_rule.onnx "$C" alpha.json
log "C: quantize, per-tensor and per-channel weights"
run "$B/quantize_bert_base.py" s_rule.onnx pt_rule.onnx --calib-ids "$C" --method percentile
run "$here/quantize_pc.py" s_rule.onnx pc_rule.onnx --calib-ids "$C" --method percentile
log "C: per-channel weights except head-split Q/K/V"
run "$here/splice_pc.py" pt_rule.onnx pc_rule.onnx q_rule.onnx --skip-head-split
log "C: statistical asymmetric rule"
run "$here/asym_rule.py" q_rule.onnx s_rule.onnx "$C" q_rule_as.onnx
log "C: deploy folds"
deploy q_rule_as.onnx pcz C
rm -f s_rule.onnx pt_rule.onnx pc_rule.onnx q_rule.onnx q_rule_as.onnx

log "D: GELU lower bound"
mkdir -p D
run "$here/gelu_clip.py" C/bertx_int8.onnx D/bertx_int8.onnx 4

log "ORT reference models"
for s in A B C; do run "$here/both_int16.py" $s/bertx_int8.onnx ${s}_ref.onnx; done
run "$here/gelu_clip.py" C_ref.onnx D_ref.onnx 4

EXPECTED_MD5="
78916bc048cf3f7fd31b43208ae39021  A/bertx_int8.onnx
6ea02878a4a6e3eb927d24f0ee991eb2  B/bertx_int8.onnx
1d777342bb9685e378c1ce06ef8d5444  C/bertx_int8.onnx
9d62a97100536476b8785e1920ceba14  D/bertx_int8.onnx
"
log "check against the models measured on the NPU"
if echo "$EXPECTED_MD5" | sed '/^$/d' | md5sum -c -; then
  log "all four deploy models are byte-identical to the measured ones"
else
  log "MISMATCH: see above"; exit 1
fi
