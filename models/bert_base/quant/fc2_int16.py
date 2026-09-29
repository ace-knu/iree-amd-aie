"""Simulate FC2's output requantized to int16 instead of int8: replace its Q/DQ pair
with fake quantization on the int16 grid over the same calibrated range."""
import sys, re, numpy as np, onnx
from onnx import numpy_helper as nh, helper
m = onnx.load(sys.argv[1]); g = m.graph
init = {i.name: i for i in g.initializer}; cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
new, dropq, k = [], set(), 0
for q in [n for n in g.node if n.op_type == "QuantizeLinear" and re.search(r"layer\.\d+/output/dense/MatMul_output_0$", n.input[0])]:
    s8 = float(nh.to_array(init[q.input[1]])); s16 = s8 * 127.0 / 32767.0; k += 1
    names = [f"fc2i16_{k}_{t}" for t in ("s", "lo", "hi")]
    g.initializer.extend([nh.from_array(np.array(s16, np.float32), names[0]),
                          nh.from_array(np.array(-32768.0, np.float32), names[1]),
                          nh.from_array(np.array(32767.0, np.float32), names[2])])
    x = q.input[0]; a, b, c = f"fc2i16_{k}_div", f"fc2i16_{k}_rnd", f"fc2i16_{k}_clp"
    for dq in cons[q.output[0]]:
        out = dq.output[0]; dropq.add(dq.output[0])
        new += [helper.make_node("Div", [x, names[0]], [a + out[-6:]]),
                helper.make_node("Round", [a + out[-6:]], [b + out[-6:]]),
                helper.make_node("Clip", [b + out[-6:], names[1], names[2]], [c + out[-6:]]),
                helper.make_node("Mul", [c + out[-6:], names[0]], [out])]
    dropq.add(q.output[0])
keep = [n for n in g.node if not (n.op_type in ("QuantizeLinear", "DequantizeLinear") and n.output[0] in dropq)]
del g.node[:]; g.node.extend(keep + new)
ready = {i.name for i in g.initializer} | {i.name for i in g.input}; order, pend = [], list(g.node)
while pend:
    rest = []
    for n in pend:
        if all(i in ready or not i for i in n.input): order.append(n); ready.update(n.output)
        else: rest.append(n)
    assert len(rest) < len(pend); pend = rest
del g.node[:]; g.node.extend(order); del g.value_info[:]
onnx.save(m, sys.argv[2]); print(f"FC2 output int16 on {k} layers -> {sys.argv[2]}")
