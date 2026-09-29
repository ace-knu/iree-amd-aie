"""SmoothQuant for BERT's matmul inputs: LayerNorm outputs (-> Q/K/V or FC1, and the
residual) and GELU outputs (-> FC2). Per input channel j:
    s_j = max|X_j|^alpha / max|W_j|^(1-alpha),   X' = X / s,   W' = diag(s) W
LayerNorm: gamma, beta /= s (free); the residual gets X' * s back.
GELU: X' = GELU(.) / s (an elementwise mul the FC1 dispatch would carry).
Usage: smooth.py <fp32.onnx> <out.onnx> <calib_ids.npy> [alpha=0.5] [all|ln|gelu]
Then quantize the output with ../quantize_bert_base.py --calib-ids (same sentences)."""
import sys, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh, helper
src, dst, calib = sys.argv[1:4]; alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
only = sys.argv[5] if len(sys.argv) > 5 else "all"      # all | ln | gelu
m = onnx.load(src); g = m.graph
init = {i.name: i for i in g.initializer}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
sites = []   # (tensor, kind, matmul consumers, other consumers, ln node or None)
for n in g.node:
    if n.op_type == "LayerNormalization":
        mms = [c for c in cons.get(n.output[0], []) if c.op_type == "MatMul"]
        if mms and only in ("all", "ln"): sites.append((n.output[0], "ln", mms, [c for c in cons[n.output[0]] if c.op_type != "MatMul"], n))
    if n.op_type == "Mul" and n.name.endswith("intermediate_act_fn/Mul_1") and only in ("all", "gelu"):
        sites.append((n.output[0], "gelu", [c for c in cons[n.output[0]] if c.op_type == "MatMul"], [], None))
# per-channel activation max over the calibration sentences
tap = onnx.ModelProto(); tap.CopyFrom(m)
for t, *_ in sites: tap.graph.output.append(helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
s = ort.InferenceSession(tap.SerializeToString(), providers=["CPUExecutionProvider"])
names = [t for t, *_ in sites]; amax = {t: 0 for t in names}
for r in np.load(calib):
    for t, v in zip(names, s.run(names, {"input_ids": r[None]})):
        amax[t] = np.maximum(amax[t], np.abs(v).reshape(-1, v.shape[-1]).max(0))
new_nodes, k = [], 0
for t, kind, mms, others, ln in sites:
    W = [nh.to_array(init[mm.input[1]]) for mm in mms]
    wmax = np.max(np.stack([np.abs(w).max(1) for w in W]), 0)
    sc = (np.power(np.maximum(amax[t], 1e-5), alpha) / np.power(np.maximum(wmax, 1e-5), 1 - alpha)).astype(np.float32)
    for mm, w in zip(mms, W):
        init[mm.input[1]].CopyFrom(nh.from_array((w * sc[:, None]).astype(np.float32), mm.input[1]))
    k += 1; sname = f"sq_s_{k}"; g.initializer.append(nh.from_array(sc, sname))
    if kind == "ln":
        for idx in (1, 2):
            v = nh.to_array(init[ln.input[idx]]); init[ln.input[idx]].CopyFrom(nh.from_array((v / sc).astype(np.float32), ln.input[idx]))
        for c in others:                       # residual: undo the smoothing
            out = f"sq_res_{k}_{c.name.split('/')[-1]}"
            new_nodes.append(helper.make_node("Mul", [t, sname], [out], name=f"sq_res_{k}"))
            for j, i in enumerate(c.input):
                if i == t: c.input[j] = out
    else:                                      # GELU output: divide before FC2
        inv = f"sq_inv_{k}"; g.initializer.append(nh.from_array((1.0 / sc).astype(np.float32), inv))
        out = f"sq_x_{k}"
        new_nodes.append(helper.make_node("Mul", [t, inv], [out], name=f"sq_div_{k}"))
        for mm in mms:
            for j, i in enumerate(mm.input):
                if i == t: mm.input[j] = out
g.node.extend(new_nodes)
# topological order
ready = {i.name for i in g.initializer} | {i.name for i in g.input}; order, pend = [], list(g.node)
while pend:
    rest = []
    for n in pend:
        if all(i in ready or not i for i in n.input): order.append(n); ready.update(n.output)
        else: rest.append(n)
    assert len(rest) < len(pend); pend = rest
del g.node[:]; g.node.extend(order); del g.value_info[:]
onnx.checker.check_model(m); onnx.save(m, dst)
print(f"smoothed {len(sites)} sites (alpha {alpha}) -> {dst}")
