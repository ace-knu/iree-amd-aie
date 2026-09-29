"""Leave-one-out sensitivity of a QDQ BERT: remove the Q/DQ pairs of one tensor kind
(keeping it fp32) and measure accuracy against fp32 BERT on held-out sentences.
Usage: loo.py <int8 QDQ model.onnx> [--combos]
"layer_in" also matches the post-attention LayerNorm, so read it as "every LayerNorm output"."""
import re, sys, numpy as np, onnx, onnxruntime as ort
from common import fp32_reference, corr_per_sentence
from onnx import numpy_helper as nh
SRC = sys.argv[1]
GROUPS = {
  "layer_in":  r"(embeddings/LayerNorm|output/LayerNorm)/LayerNormalization_output_0$",
  "qkv_out":   r"self/(query|key|value)/MatMul_output_0$",
  "heads":     r"self/Transpose(_1|_2)?_output_0$",
  "scores":    r"self/MatMul_output_0$",
  "probs":     r"self/Softmax_output_0$",
  "context":   r"self/(MatMul_1|Reshape_3)_output_0$",
  "attn_out":  r"attention/output/dense/MatMul_output_0$",
  "ln1_out":   r"attention/output/LayerNorm/LayerNormalization_output_0$",
  "fc1_out":   r"intermediate/dense/MatMul_output_0$",
  "gelu_out":  r"intermediate_act_fn/Mul_1_output_0$",
  "fc2_out":   r"layer\.\d+/output/dense/MatMul_output_0$",
}
base = onnx.load(SRC)
init_names = {i.name for i in base.graph.initializer}

def strip(model, pred):
    """Remove Q->DQ pairs whose quantized tensor matches pred(name)."""
    g = model.graph
    cons = {}
    for n in g.node:
        for i in n.input: cons.setdefault(i, []).append(n)
    drop, rewire = set(), {}
    for q in [n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] not in init_names]:
        if not pred(q.input[0]): continue
        dqs = [c for c in cons.get(q.output[0], []) if c.op_type == "DequantizeLinear"]
        if len(dqs) != len(cons.get(q.output[0], [])): continue
        drop.add(q.output[0]); [drop.add(d.output[0]) for d in dqs]
        for d in dqs: rewire[d.output[0]] = q.input[0]
    keep = [n for n in g.node if not (n.op_type in ("QuantizeLinear", "DequantizeLinear") and n.output[0] in drop)]
    for n in keep:
        for k, i in enumerate(n.input):
            if i in rewire: n.input[k] = rewire[i]
    del g.node[:]; g.node.extend(keep)
    return len(rewire)

def dequant_weights(model):
    """W32: replace int8 weight initializers + their DQ with float initializers."""
    g = model.graph; init = {i.name: i for i in g.initializer}; drop = []; new = []
    for n in g.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in init and nh.to_array(init[n.input[0]]).ndim >= 2:
            w = nh.to_array(init[n.input[0]]).astype(np.float32)
            s = nh.to_array(init[n.input[1]]).astype(np.float32)
            z = nh.to_array(init[n.input[2]]).astype(np.float32) if len(n.input) > 2 else 0
            new.append(nh.from_array((w - z) * s, n.output[0])); drop.append(n.output[0])
    drop = set(drop)
    keep = [n for n in g.node if not (n.op_type == "DequantizeLinear" and n.output[0] in drop)]
    del g.node[:]; g.node.extend(keep)
    g.initializer.extend(new); del g.value_info[:]; return len(new)

ref = fp32_reference()
def score(model):
    s = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return float(corr_per_sentence(s, ref).mean())

def variant(fn):
    m = onnx.ModelProto(); m.CopyFrom(base); k = fn(m); return score(m), k
if "--combos" not in sys.argv:
    rows = [("all quantized (W8A8)", *variant(lambda m: 0))]
    rows.append(("weights only (W8A32)", *variant(lambda m: strip(m, lambda t: True))))
    rows.append(("activations only (W32A8)", *variant(dequant_weights)))
    for name, pat in GROUPS.items():
        rows.append((f"all except {name}", *variant(lambda m, p=pat: strip(m, lambda t: re.search(p, t) is not None))))
    for label, v, k in rows: print(f"{label:30s} corr vs fp32 {v:.4f}   ({k} tensors kept fp32)")
else:
    P = GROUPS
    def pat(*names, layers=None):
        rx = [re.compile(P[n]) for n in names]
        def f(t):
            if not any(r.search(t) for r in rx): return False
            if layers is None: return True
            m = re.search(r"layer\.(\d+)/", t)
            return m is not None and int(m.group(1)) in layers
        return f
    tests = [
      ("fc2_out + layer_in",                 pat("fc2_out", "layer_in")),
      ("fc2_out + layer_in + ln1_out",       pat("fc2_out", "layer_in", "ln1_out")),
      ("  + gelu_out",                       pat("fc2_out", "layer_in", "ln1_out", "gelu_out")),
      ("  + attn_out",                       pat("fc2_out", "layer_in", "ln1_out", "gelu_out", "attn_out")),
      ("fc2_out, layers 10-11 only",         pat("fc2_out", layers={10, 11})),
      ("fc2_out, layers 6-11",               pat("fc2_out", layers=set(range(6, 12)))),
      ("fc2_out, layers 0-5",                pat("fc2_out", layers=set(range(0, 6)))),
      ("fc2+layer_in+ln1, layers 10-11",     pat("fc2_out", "layer_in", "ln1_out", layers={10, 11})),
      ("fc2+layer_in+ln1, layers 6-11",      pat("fc2_out", "layer_in", "ln1_out", layers=set(range(6, 12)))),
    ]
    for label, f in tests:
        m = onnx.ModelProto(); m.CopyFrom(base); k = strip(m, f)
        print(f"all except {label:34s} corr vs fp32 {score(m):.4f}   ({k} tensors kept fp32)")
