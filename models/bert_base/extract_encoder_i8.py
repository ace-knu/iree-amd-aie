#!/usr/bin/env python3
"""Cut the 12-layer encoder out of the quantized BERT at its int8 boundary.

Layer N's int8 output is layer N+1's int8 input: every LayerNorm between layers
has a `QuantizeLinear` with zero zero-point, and both of its consumers -- the
K-fold `Pad` into the next QKV and the `DequantizeLinear` into the next
residual -- read that same int8 tensor. So the stack is one subgraph, no glue.

The two ends are not symmetric, and that is the model's own doing:

  in   the embedding LayerNorm's `QuantizeLinear` output, int8 [1, S, 768].
  out  `last_hidden_state`, f32. Layer 11's output LayerNorm is the only one
       with no `QuantizeLinear` after it -- it feeds the pooler, nothing
       downstream reads it quantized -- and inventing a scale there would be
       making up a calibration value. So 23 of the 24 LayerNorms get the int8
       tail and the last one stays f32.

The attention mask is all zeros at this sequence length; it is baked in as a
constant so every softmax dispatch stays 2-operand.

Input: the QKV-folded model from `fold_bias_into_k.py`.

Usage:
  python3 models/bert_base/extract_encoder_i8.py bert_base_kpadq_int8.onnx enc12_i8.onnx
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper as nh

MASK = "/m/Where_1_output_0"
IN_Q = "/m/embeddings/LayerNorm/LayerNormalization_output_0_QuantizeLinear_Output"
OUT = "last_hidden_state"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="QKV-folded quantized BERT (fold_bias_into_k.py)")
    p.add_argument("output", help="encoder subgraph to write")
    a = p.parse_args()

    onnx.utils.extract_model(a.input, a.output, [IN_Q, MASK], [OUT])
    m = onnx.load(a.output)
    g = m.graph
    g.name = "encoder12"

    mi = [i for i in g.input if i.name == MASK]
    assert len(mi) == 1, f"expected the attention mask {MASK} as an input"
    shape = [d.dim_value or 1 for d in mi[0].type.tensor_type.shape.dim]
    g.initializer.append(nh.from_array(np.zeros(shape, np.float32), MASK))
    g.input.remove(mi[0])

    for t in list(g.input) + list(g.output):
        d = t.type.tensor_type.shape.dim[0]
        if d.HasField("dim_param"):
            d.ClearField("dim_param")
            d.dim_value = 1
    del g.value_info[:]
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.checker.check_model(m)
    onnx.save(m, a.output)

    print(f"wrote {a.output}: {len(m.graph.node)} nodes")
    for io, label in ((m.graph.input, "input"), (m.graph.output, "output")):
        for t in io:
            dims = [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
            print(f"  {label} {t.name} {dims}")


if __name__ == "__main__":
    main()
