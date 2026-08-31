"""Two independent matmuls joined by an Add, for multi-device placement tests.

  a = MatMul(x, w1)   branch 1
  b = MatMul(x, w2)   branch 2  (independent of a)
  y = Add(a, b)       join

The branches share x but not each other, so nothing forces them onto one
device. The Add is elementwise, so it should land on the host rather than
fusing into either matmul. Weights are runtime inputs, matching mlp_2layer
(two constant weights fail on the amd-aie backend).
"""

import onnx
from onnx import TensorProto as T
from onnx import helper as h

D = 128


def vi(name):
    return h.make_tensor_value_info(name, T.FLOAT, [D, D])


graph = h.make_graph(
    [
        h.make_node("MatMul", ["x", "w1"], ["a"]),
        h.make_node("MatMul", ["x", "w2"], ["b"]),
        h.make_node("Add", ["a", "b"], ["y"]),
    ],
    "mlp_parallel",
    [vi("x"), vi("w1"), vi("w2")],
    [vi("y")],
)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 20)])
onnx.checker.check_model(model)
onnx.save(model, "mlp_parallel.onnx")
print("saved mlp_parallel.onnx")
