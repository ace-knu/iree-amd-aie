"""mlp_2layer 상수 가중치판 (docs/2026-09-28/05). 기존 models/mlp_2layer/ 는 건드리지 않는다.

git HEAD 의 models/mlp_2layer/export_mlp_2layer.py(03550187, 상수판)를 참고했다.
차이: 기대값을 손으로 계산할 수 있게 가중치를 난수 대신 규칙적인 값으로 뒀다.

  layer 1: h0 = MatMul(x, w1); h1 = Relu(h0)
  layer 2: y  = MatMul(h1, w2)

기본(비-splat): w1[i, j] = 2.0 (i 짝수) / 0.0 (i 홀수), w2[i, j] = 2.0 (i 홀수) / 0.0 (i 짝수).
  열 합 = 64 * 2 = 128. (w1 == w2 이면 IREE 가 상수 하나로 합치므로 행 패턴을 달리 했다)
  x = 1 -> h0 = 128, h1 = 128, y = 128 * 128 = 16384
--ones (splat): w = 1.0 전부. 결과는 같은 16384 이지만, IREE 가 splat 상수를
  dispatch 안에 arith.constant dense<1.0> 로 인라인해서 상수 버퍼가 생기지 않는다
  (docs/2026-09-28/05 README 2절). 그 사례를 재현하려고 남겨 둔 옵션.
--shared: w2 = w1 (짝수 행 2.0). 기대값 16384. IREE 가 같은 값의 상수를 하나로 합쳐서
  NPU matmul 과 GPU matmul 이 상수 하나를 함께 쓰는 경우를 만든다.

컨테이너 안에서: source /opt/venv/bin/activate && python3 export_mlp_2layer_const.py [--ones | --shared]
  -> mlp_2layer_const.onnx (기본) / mlp_2layer_const_ones.onnx (--ones)
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

D = 128


def vi(name):
    return h.make_tensor_value_info(name, T.FLOAT, [D, D])


ONES = "--ones" in sys.argv
SHARED = "--shared" in sys.argv  # w2 = w1 -> IREE 가 상수 하나로 합쳐 NPU/GPU 가 공유
if ONES:
    w1 = np.ones((D, D), dtype=np.float32)
else:
    w1 = np.zeros((D, D), dtype=np.float32)
    w1[0::2, :] = 2.0
w2 = np.ones((D, D), dtype=np.float32) if ONES else np.zeros((D, D), dtype=np.float32)
if SHARED:
    w2 = w1.copy()
elif not ONES:
    w2[1::2, :] = 2.0  # w1 과 값이 같으면 IREE 가 상수 하나로 합쳐서, 행 패턴을 다르게 둔다
OUT = ("mlp_2layer_const_ones.onnx" if ONES else
       "mlp_2layer_const_shared.onnx" if SHARED else "mlp_2layer_const.onnx")

graph = h.make_graph(
    [
        h.make_node("MatMul", ["x", "w1"], ["h0"]),
        h.make_node("Relu", ["h0"], ["h1"]),
        h.make_node("MatMul", ["h1", "w2"], ["y"]),
    ],
    "mlp_2layer_const",
    [vi("x")],  # 런타임 입력은 x 하나
    [vi("y")],
    [nh.from_array(w1, "w1"), nh.from_array(w2, "w2")],  # 상수(initializer)
)
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 20)])
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"saved {OUT} (weights baked as initializers, ones={ONES}, shared={SHARED})")
