#!/usr/bin/env bash
# [container] VGG16 (or only its first conv+relu) on npu4, checked against torch.
#
#   bash scripts/repro/aie2p_bf16/vgg16.sh <full|layer1> <on|off> [runs]
#   BENCH=1 bash scripts/repro/aie2p_bf16/vgg16.sh full on      # also benchmark
#
# on/off is --iree-amdaie-enable-vectorization-passes. Every other flag is the
# recipe in models/vgg16/README.md. The input is the README's fixed random image
# (np.random.seed(1)); the reference interprets the same ONNX graph with torch.
# Needs models/vgg16/vgg16-12.onnx (see models/vgg16/README.md section 1).
#
# Timing: prints the compile time and, per run, the wall time of iree-run-module
# (process start to exit, so it includes loading and NPU setup). BENCH=1 adds
# iree-benchmark-module (10 repetitions), the number to compare builds with.
set -euo pipefail
source /opt/venv/bin/activate
export PYTHONPATH=/workspace/build/compiler/bindings/python
cd /workspace
now_ms() { date +%s%3N; }

PART=${1:?usage: vgg16.sh <full|layer1> <on|off> [runs]}
VEC=${2:?usage: vgg16.sh <full|layer1> <on|off> [runs]}
RUNS=${3:-3}
case "$VEC" in on) VECFLAG=true ;; off) VECFLAG=false ;;
  *) echo "second argument must be on or off" >&2; exit 1 ;; esac
OUT=/workspace/build/repro_aie2p_bf16/vgg16_${PART}
mkdir -p "$OUT"

if [ "$PART" = layer1 ]; then
  # First Conv + Relu only, output vgg0_relu0_fwd [1,64,224,224].
  [ -f "$OUT/model.onnx" ] || python3 - "$OUT/model.onnx" <<'PY'
import sys
import onnx
from onnx import helper as h
m = onnx.load("models/vgg16/vgg16-12.onnx")
g = m.graph
keep = {"vgg0_conv0_weight", "vgg0_conv0_bias"}
out = h.make_tensor_value_info("vgg0_relu0_fwd", onnx.TensorProto.FLOAT, [1, 64, 224, 224])
graph = h.make_graph(list(g.node[:2]), "vgg_layer1", list(g.input), [out],
                     [i for i in g.initializer if i.name in keep])
model = h.make_model(graph, opset_imports=m.opset_import)
onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
PY
  ONNX=$OUT/model.onnx
elif [ "$PART" = full ]; then
  ONNX=models/vgg16/vgg16-12.onnx
else
  echo "first argument must be full or layer1" >&2; exit 1
fi

[ -f "$OUT/model.mlir" ] || python3 -m iree.compiler.tools.import_onnx "$ONNX" -o "$OUT/model.mlir"
[ -f "$OUT/input.npy" ] || python3 -c "import numpy as np; np.random.seed(1); \
  np.save('$OUT/input.npy', (np.random.randn(1,3,224,224)*0.2).astype(np.float32))"
FUNC=$(grep -oE "func.func @[A-Za-z0-9_]+" "$OUT/model.mlir" | head -1 | sed 's/func.func @//')
VMFB=$OUT/vec_${VEC}.vmfb

t0=$(now_ms)
build/tools/iree-compile "$OUT/model.mlir" -o "$VMFB" \
  --iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-global-opt-use-im2col-for-convs --iree-amdaie-demote-contraction-inputs-to-bf16 \
  --iree-amdaie-enable-vectorization-passes=$VECFLAG \
  --iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-convert-conv-to-channels-last)' \
  --iree-dispatch-creation-no-fuse-into-contraction-conv-roots \
  --iree-global-opt-detach-elementwise-through-reshape \
  --iree-flow-enable-executable-deduplication=false 2>&1 \
  | { grep -iE "error" || true; }  # hides the long generic-CPU-target warning
echo "compile: $(( $(now_ms) - t0 )) ms"

for i in $(seq 1 "$RUNS"); do
  t0=$(now_ms)
  build/tools/iree-run-module --device=amdxdna --device=local-task \
    --module="$VMFB" --function="$FUNC" \
    --input=@"$OUT/input.npy" --output=@"$OUT/out_${VEC}_$i.npy" > /dev/null
  echo $(( $(now_ms) - t0 )) > "$OUT/time_ms_${VEC}_$i.txt"
done

python3 - "$ONNX" "$OUT" "$VEC" "$RUNS" <<'PY'
import sys
import numpy as np
import onnx
import torch
import torch.nn.functional as F
onnx_path, out, vec, runs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
g = onnx.load(onnx_path).graph
init = {i.name: torch.tensor(onnx.numpy_helper.to_array(i).copy()) for i in g.initializer}
def attr(n, k, d=None):
    for a in n.attribute:
        if a.name == k:
            return list(a.ints) if a.ints else a.i
    return d
env = {g.input[0].name: torch.tensor(np.load(f"{out}/input.npy"))}
for n in g.node:
    t, x = n.op_type, env[n.input[0]]
    if t == "Conv":
        w = init[n.input[1]]; b = init[n.input[2]] if len(n.input) > 2 else None
        env[n.output[0]] = F.conv2d(x, w, b, stride=attr(n, "strides", [1, 1])[0],
                                    padding=attr(n, "pads", [0, 0, 0, 0])[0])
    elif t == "Relu":    env[n.output[0]] = F.relu(x)
    elif t == "MaxPool": env[n.output[0]] = F.max_pool2d(x, attr(n, "kernel_shape", [2, 2])[0],
                                                         attr(n, "strides", [2, 2])[0])
    elif t == "Flatten": env[n.output[0]] = x.reshape(x.shape[0], -1)
    elif t == "Gemm":    env[n.output[0]] = x @ init[n.input[1]].t() + (
                             init[n.input[2]] if len(n.input) > 2 else 0)
    elif t == "Dropout": env[n.output[0]] = x
    else: raise SystemExit("unhandled op " + t)
gold = env[g.output[0].name].numpy()
outs = [np.load(f"{out}/out_{vec}_{i}.npy") for i in range(1, runs + 1)]
classify = gold.size == 1000
for i, got in enumerate(outs, 1):
    ms = int(open(f"{out}/time_ms_{vec}_{i}.txt").read())
    line = f"run {i}: {ms:6d} ms  corr {np.corrcoef(gold.ravel(), got.ravel())[0, 1]:.5f}"
    if classify:
        line += f"  top-1 gold={gold.argmax()} got={got.argmax()} match={gold.argmax() == got.argmax()}"
    print(line)
print("runs bit-identical:", all(np.array_equal(outs[0], o) for o in outs[1:]))
PY

if [ "${BENCH:-0}" = 1 ]; then
  build/tools/iree-benchmark-module --device=amdxdna --device=local-task \
    --module="$VMFB" --function="$FUNC" \
    --input=@"$OUT/input.npy" --benchmark_repetitions=10 2>&1 \
    | grep -E "real_time_(mean|median|stddev|cv)"
fi
