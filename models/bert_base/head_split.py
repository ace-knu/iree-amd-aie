"""Fold a projection's `Reshape -> Transpose` into the weight, per attention head.

A Q/K/V projection is written as one `[K, H*D]` matmul followed by a reshape and
a transpose that pull the head dimension to the front:

    MatMul(x[B,S,K], W[K,H*D]) -> [B,S,H*D] -> Reshape [B,S,H,D]
                                            -> Transpose perm=[0,2,1,3] -> [B,H,S,D]

The same values come out of a per-head weight with no reshape and no transpose:

    MatMul(x[B,S,K], W'[H,K,D]) -> [H,S,D] -> Unsqueeze(0) -> [1,H,S,D]
      where W'[h,k,d] = W[k, h*D + d]

The weight is a compile-time constant, so rearranging it costs nothing at run
time, while the transpose it replaces is a real elementwise dispatch on the
activation -- and one this backend places on the host, because a transpose is
not a contraction. The quantization is per tensor, so moving the weight's
elements around leaves its scale and zero point exactly as they were.

Refuses, leaving the projection alone, unless every step is checkable:
  - the weight is a constant feeding a `DequantizeLinear`, rank 2;
  - the matmul's only consumer is a `Reshape` splitting the last axis into
    `[H, D]` with `H*D` equal to the weight's output width;
  - that reshape's only consumer is a `Transpose` with perm `[0,2,1,3]`.

K is consumed transposed, as `[B,H,D,S]` (perm `[0,2,3,1]`), which is not a
plain matmul of the same operands. With `--k` it is split the same way and one
transpose is kept -- but moved onto the *quantized* tensor, between its
QuantizeLinear and DequantizeLinear:

    MatMul(x, W'[H,K,D]) -> [H,S,D] -> Unsqueeze(0) -> QuantizeLinear
      -> Transpose [0,1,3,2] -> [1,H,D,S] -> DequantizeLinear -> QK^T

With per-tensor quantization that is numerically the same, and it leaves IREE
a plain int8 permutation (a `[0,2,1]` once the unit batch is dropped) feeding
the QK^T batch matmul's RHS, which
`PropagateLinalgTranspose` folds into `batch_matmul_transpose_b` -- the
transpose dispatch disappears instead of running on the host. This is the K
handling of `gen_attn_headsplit.py` (2026-09-16), used for the single-layer
models. The DequantizeLinear must stay the matmul's direct operand: an
Unsqueeze between them (tried first) leaves QK^T as an f32 matmul. It
additionally requires:
  - that transpose's only consumer is a `QuantizeLinear` with a scalar scale,
    whose only consumer is a `DequantizeLinear`.
Without `--k`, K is left alone as before.
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh


PERM_QV = (0, 2, 1, 3)
PERM_K = (0, 2, 3, 1)


def run(src, dst, verbose=True, include_k=False):
    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}

    def shape(t):
        v = vi.get(t)
        if v is None:
            return None
        out = []
        for d in v.type.tensor_type.shape.dim:
            if not d.HasField("dim_value"):
                return None
            out.append(d.dim_value)
        return out

    def only(t):
        c = cons.get(t, [])
        return c[0] if len(c) == 1 else None

    split, ksplit, skipped = 0, 0, {}
    new_nodes, new_inits, drop = [], [], set()
    for mm in [n for n in g.node if n.op_type == "MatMul"]:
        wdq = prod.get(mm.input[1])
        if wdq is None or wdq.op_type != "DequantizeLinear" or wdq.input[0] not in init:
            continue
        w = nh.to_array(init[wdq.input[0]])
        if w.ndim != 2:
            continue
        lhs_sh = shape(mm.input[0])
        if lhs_sh is None or len(lhs_sh) != 3:
            continue
        rs = only(mm.output[0])
        if rs is None or rs.op_type != "Reshape":
            continue
        out_sh = shape(rs.output[0])
        if out_sh is None or len(out_sh) != 4:
            continue
        B, S, H, D = out_sh
        if H * D != w.shape[1] or B != lhs_sh[0] or S != lhs_sh[1]:
            continue
        tr = only(rs.output[0])
        if tr is None or tr.op_type != "Transpose":
            continue
        perm = tuple([a.ints for a in tr.attribute if a.name == "perm"][0])
        if perm == PERM_K and include_k:
            q = only(tr.output[0])
            dq = only(q.output[0]) if q is not None and q.op_type == "QuantizeLinear" else None
            if (dq is None or dq.op_type != "DequantizeLinear" or len(q.input) < 2
                    or q.input[1] not in init or nh.to_array(init[q.input[1]]).size != 1):
                skipped[perm] = skipped.get(perm, 0) + 1
                continue
            ksplit += 1
            tag = f"ksplit{ksplit}"
            wp = np.ascontiguousarray(
                w.reshape(w.shape[0], H, D).transpose(1, 0, 2))
            new_inits.append(nh.from_array(wp, f"{tag}_w"))
            new_nodes.append(helper.make_node(
                "DequantizeLinear", [f"{tag}_w"] + list(wdq.input[1:]), [f"{tag}_wdq"],
                name=f"{tag}_dq", **{a.name: helper.get_attribute_value(a)
                                     for a in wdq.attribute if a.name != "axis"}))
            # [H,S,D] -> [1,H,S,D] into the existing QuantizeLinear, as for Q/V.
            new_nodes.append(helper.make_node(
                "MatMul", [mm.input[0], f"{tag}_wdq"], [f"{tag}_mm"], name=f"{tag}_mm"))
            new_inits.append(nh.from_array(np.asarray([0], np.int64), f"{tag}_ax"))
            new_nodes.append(helper.make_node(
                "Unsqueeze", [f"{tag}_mm", f"{tag}_ax"], [f"{tag}_us"], name=f"{tag}_us"))
            q.input[0] = f"{tag}_us"
            # int8 [1,H,S,D] -> [1,H,D,S], between QuantizeLinear and
            # DequantizeLinear, so the DequantizeLinear still feeds QK^T directly
            # (anything between them keeps QK^T from becoming an i8 x i8 -> i32
            # matmul).
            new_nodes.append(helper.make_node(
                "Transpose", [q.output[0]], [f"{tag}_qT"], name=f"{tag}_qT", perm=[0, 1, 3, 2]))
            dq.input[0] = f"{tag}_qT"
            drop.update({id(mm), id(rs), id(tr)})
            continue
        if perm != PERM_QV:
            skipped[perm] = skipped.get(perm, 0) + 1
            continue

        split += 1
        tag = f"hsplit{split}"
        # W[k, h*D + d] -> W'[h, k, d]
        wp = np.ascontiguousarray(
            w.reshape(w.shape[0], H, D).transpose(1, 0, 2))
        new_inits.append(nh.from_array(wp, f"{tag}_w"))
        new_nodes.append(helper.make_node(
            "DequantizeLinear", [f"{tag}_w"] + list(wdq.input[1:]), [f"{tag}_wdq"],
            name=f"{tag}_dq", **{a.name: helper.get_attribute_value(a)
                                 for a in wdq.attribute if a.name != "axis"}))
        new_nodes.append(helper.make_node(
            "MatMul", [mm.input[0], f"{tag}_wdq"], [f"{tag}_mm"], name=f"{tag}_mm"))
        new_inits.append(nh.from_array(np.asarray([0], np.int64), f"{tag}_ax"))
        new_nodes.append(helper.make_node(
            "Unsqueeze", [f"{tag}_mm", f"{tag}_ax"], [tr.output[0]], name=f"{tag}_us"))
        drop.update({id(mm), id(rs), id(tr)})

    if split or ksplit:
        keep = [n for n in g.node if id(n) not in drop]
        del g.node[:]
        g.node.extend(keep + new_nodes)
        g.initializer.extend(new_inits)
        ready = {i.name for i in g.initializer} | {i.name for i in g.input}
        order, pending = [], list(g.node)
        while pending:
            rest, moved = [], False
            for n in pending:
                if all(i in ready or i == "" for i in n.input):
                    order.append(n); ready.update(n.output); moved = True
                else:
                    rest.append(n)
            if not moved:
                raise SystemExit(f"topological sort stuck, e.g. {rest[0].op_type} "
                                 f"{list(rest[0].input)}")
            pending = rest
        del g.node[:]
        g.node.extend(order)
        used = set()
        for n in g.node:
            used.update(n.input)
        kept = [i for i in g.initializer if i.name in used]
        del g.initializer[:]
        g.initializer.extend(kept)

    del g.value_info[:]
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    if verbose:
        print(f"wrote {dst}: {split} projection(s) head-split, "
              f"{ksplit} K projection(s) head-split to transpose_b form, "
              f"{len(m.graph.node)} nodes")
        for perm, k in skipped.items():
            print(f"  left alone: {k} with transpose perm {perm}")
    return split + ksplit


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0], args[1], include_k="--k" in sys.argv[1:])
