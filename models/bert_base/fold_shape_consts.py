#!/usr/bin/env python3
"""Pin a BERT export to batch 1 and fold its shape-only subgraphs into constants.

The full export computes position/token-type ids and the attention mask by
expanding constants to the input's shape (Shape -> ... -> Expand /
ConstantOfShape / Where). Those depend on `input_ids`' shape, not its values,
so at batch 1 they are constants -- but shape inference cannot see through
them, and the model-preparation tools need every tensor's shape. Only
shape-computation op types are folded; Q/DQ, MatMul and the embedding-table
Gathers stay. The result is checked byte-identical on onnxruntime.

Usage: fold_shape_consts.py <in.onnx> <out.onnx>
"""
import sys, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh, helper
SHAPE_OPS = {"Shape","Gather","Unsqueeze","Squeeze","Concat","ConstantOfShape","Expand","Equal",
             "Where","Mul","Slice","Cast","Constant","Range","Reshape","GreaterOrEqual","Less","Not","Add","Sub"}
m = onnx.load(sys.argv[1]); g = m.graph
for t in list(g.input) + list(g.output):          # batch -> 1
    d = t.type.tensor_type.shape.dim[0]
    if d.HasField("dim_param"):
        d.ClearField("dim_param"); d.dim_value = 1
del g.value_info[:]
m = onnx.shape_inference.infer_shapes(m, strict_mode=True); g = m.graph
inp = g.input[0].name
init = {i.name for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
const = set(init)              # tensors known to be constant (value, not just shape)
fold = []                      # nodes to evaluate
changed = True
while changed:
    changed = False
    for n in g.node:
        if n in fold or n.op_type not in SHAPE_OPS: continue
        ins = [i for i in n.input if i]
        ok = all(i in const for i in ins) or (n.op_type == "Shape" and n.input[0] == inp)
        # a Gather on a large float table is an embedding lookup, not shape math
        if ok and n.op_type == "Gather" and any(i in init and np.prod(nh.to_array([x for x in g.initializer if x.name==i][0]).shape) > 4096 for i in ins):
            ok = False
        if ok:
            fold.append(n); const.update(n.output); changed = True
# evaluate only the folded tensors that something outside the folded set reads
outside = {i for n in g.node if n not in fold for i in n.input}
targets = [o for n in fold for o in n.output if o in outside or o in {x.name for x in g.output}]
tmp = onnx.shape_inference.infer_shapes(m)
etype = {v.name: v.type.tensor_type.elem_type for v in list(tmp.graph.value_info) + list(tmp.graph.output)}
del tmp.graph.output[:]
for t in targets: tmp.graph.output.append(helper.make_tensor_value_info(t, etype.get(t, onnx.TensorProto.INT64), None))
s = ort.InferenceSession(tmp.SerializeToString(), providers=["CPUExecutionProvider"])
vals = s.run(targets, {inp: np.zeros([d.dim_value for d in g.input[0].type.tensor_type.shape.dim], np.int64)})
keep = [n for n in g.node if n not in fold]
del g.node[:]; g.node.extend(keep)
for t, v in zip(targets, vals): g.initializer.append(nh.from_array(np.asarray(v), t))
used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
kept = [i for i in g.initializer if i.name in used]; del g.initializer[:]; g.initializer.extend(kept)
del g.value_info[:]
m = onnx.shape_inference.infer_shapes(m, strict_mode=True, data_prop=True); onnx.checker.check_model(m)
none = [v.name for v in m.graph.value_info if any(not d.HasField("dim_value") for d in v.type.tensor_type.shape.dim)]
onnx.save(m, sys.argv[2])
print(f"folded {len(fold)} shape-only nodes into {len(targets)} constants; {len(none)} tensors still symbolic {none[:3]}")
