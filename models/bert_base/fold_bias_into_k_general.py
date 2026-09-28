"""Fold a constant bias into a quantized MatMul's K axis, as a general graph rewrite.

`fold_bias_into_k.py` does this for BERT projections picked by node name.  This one
keeps the same arithmetic but states the conditions structurally, so it applies to
any static `DequantizeLinear(x) x DequantizeLinear(const W) -> [Q -> DQ ->] Add(const b)`
in an int8 QDQ graph -- the output projection included.

The arithmetic, unchanged from the QKV work:

    x_aug = [x_i8 | v_1 ... v_L]              L extra columns of *constant* int8 values
    W_aug = [W_i8 ; b_1 ; ... ; b_L]          L extra rows, one per column
    sum_i v_i * b_i[n] = round(bias[n] / (s_x * s_w))

so the i32 accumulator already carries the bias and the float `Add` disappears.  Both
`v_i` and `b_i` are int8, so a large bias dynamic range needs more columns; the solver
stacks columns until the residual is under half an accumulator LSB, which is the floor
of this method (and of an int32 bias term).

Two augmentation shapes, because they cost different amounts downstream:

  --aug=pad     every column carries the same value (default 1), so the augmentation is
                an ONNX `Pad`.  IREE lowers that to a buffer splat plus a producer that
                writes the unpadded region in place -- no extra dispatch when the
                producer fuses.  Needs L = ceil(max|T| / 127) columns.
  --aug=concat  columns may carry different values, which is the `B = c1*w1 + c2*w2`
                form and needs far fewer columns (2 is usually enough), but ONNX
                `Concat` lowers to real copies.

Where the augmentation is placed matters as much as its shape.  A `Pad` only costs
nothing when its producer can write the widened buffer in place, and anything sitting
between the two -- a flatten, for instance -- stops that.  When the quantized activation
comes from a `Reshape` that merges trailing dimensions, `[..., A, C] -> [..., A*C]`, and
the padding is a whole number of `C`, the same bytes can be added by padding `A`
instead, with the quantization hoisted above the flatten (exact: it is per-tensor and
elementwise).  Then producer and Pad are adjacent.  That is what `--pad-in-producer-
layout` does, and for BERT's output projection it is the difference between the fold
costing an extra dispatch and costing nothing: the PV matmul grows a 13th head of
constant columns and writes its 12 real heads into it.

Conditions checked per candidate; any failure leaves that matmul alone:

  1. the weight is a constant initializer feeding a DequantizeLinear;
  2. the matmul (through its requantization, if any) is consumed by a single Add whose
     other operand is a constant of shape [N];
  3. both quantization scales are constants and both zero points are zero -- a non-zero
     zero point makes the augmented columns contribute an input-dependent term;
  4. the bias converts to the accumulator domain, T = bias / (s_x * s_w);
  5. T is representable within `--max-columns` int8 columns;
  6. the augmented K rounds up to `--align` (a target property: on npu4 a K/32 that is
     odd has been seen to break `aie.memtile_dma` at N=3072, hence the 64 default);
  7. when the requantization is kept, the bias must fit the requantization's headroom.

Usage:
  python fold_bias_into_k_general.py <in.onnx> <out.onnx> [options]
    --align N          K alignment of the augmented axis (default 64)
    --aug pad|concat   augmentation shape (default pad)
    --pad-value V      constant column value for --aug=pad (default 1)
    --max-columns N    refuse a candidate needing more columns (default = align)
    --drop-requant     remove the matmul's Q/DQ pair instead of keeping it
    --max-bias-lsb X   refuse if max|bias| exceeds X requantization steps (default 48)
    --select RE        only fold matmuls whose name matches (debugging selector)
    --skip RE          never fold matmuls whose name matches (debugging selector)
    --dry-run          report what would happen and write nothing
"""
import argparse
import collections
import re
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper as nh


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--align", type=int, default=64)
    p.add_argument("--aug", choices=("pad", "concat"), default="pad")
    p.add_argument("--pad-value", type=int, default=1)
    p.add_argument("--max-columns", type=int, default=None)
    p.add_argument("--pad-in-producer-layout", dest="producer_layout",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-requant", action="store_true")
    p.add_argument("--max-bias-lsb", type=float, default=48.0)
    p.add_argument("--select", default=None)
    p.add_argument("--skip", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.max_columns is None:
        a.max_columns = a.align
    return a


class Graph:
    def __init__(self, g):
        self.g = g
        self.init = {i.name: nh.to_array(i) for i in g.initializer}
        self.prod = {o: n for n in g.node for o in n.output}
        self.cons = collections.defaultdict(list)
        for n in g.node:
            for i in n.input:
                if i:
                    self.cons[i].append(n)
        self.shapes = {}
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            d = vi.type.tensor_type.shape.dim
            self.shapes[vi.name] = [x.dim_value if x.HasField("dim_value") else None for x in d]

    def only(self, tensor, op_type):
        c = self.cons.get(tensor, [])
        if len(c) == 1 and c[0].op_type == op_type:
            return c[0]
        return None

    def scalar(self, name):
        if name not in self.init:
            return None
        a = np.asarray(self.init[name], dtype=np.float64)
        return float(a.reshape(-1)[0]) if a.size == 1 else None

    def zero_point_is_zero(self, node):
        if len(node.input) < 3 or not node.input[2]:
            return True
        z = self.init.get(node.input[2])
        return z is not None and not np.any(np.asarray(z) != 0)


def find_candidates(G, args):
    """Structural match.  Returns (accepted, rejected) where each entry says why."""
    accepted, rejected = [], []
    for mm in G.g.node:
        if mm.op_type != "MatMul":
            continue
        name = mm.name or mm.output[0]

        def reject(why):
            rejected.append((name, why))

        if args.select and not re.search(args.select, name):
            reject("not selected"); continue
        if args.skip and re.search(args.skip, name):
            reject("skipped by --skip"); continue

        xdq = G.prod.get(mm.input[0])
        wdq = G.prod.get(mm.input[1])
        if xdq is None or xdq.op_type != "DequantizeLinear":
            reject("lhs is not a DequantizeLinear"); continue
        # (1) constant weight
        if wdq is None or wdq.op_type != "DequantizeLinear" or wdq.input[0] not in G.init:
            reject("rhs is not a DequantizeLinear of a constant"); continue
        w = np.asarray(G.init[wdq.input[0]])
        if w.ndim != 2 or w.dtype != np.int8:
            reject(f"weight is not 2-D int8 (got {w.ndim}-D {w.dtype})"); continue

        # (2) the only consumer chain ends in a constant-bias Add
        requant_q = G.only(mm.output[0], "QuantizeLinear")
        requant_d = G.only(requant_q.output[0], "DequantizeLinear") if requant_q else None
        tail = requant_d.output[0] if requant_d else mm.output[0]
        if requant_q is not None and requant_d is None:
            reject("requantization Q has no single DQ"); continue
        add = G.only(tail, "Add")
        if add is None:
            reject("no single Add consuming the matmul"); continue
        bias_names = [i for i in add.input if i != tail and i in G.init]
        if len(bias_names) != 1:
            reject("Add's other operand is not a single constant"); continue
        bias = np.asarray(G.init[bias_names[0]], dtype=np.float64).reshape(-1)
        if bias.size != w.shape[1]:
            reject(f"bias size {bias.size} != N {w.shape[1]}"); continue

        # (3) constant scales, zero zero-points
        xs, ws = G.scalar(xdq.input[1]), G.scalar(wdq.input[1])
        if xs is None or ws is None:
            reject("input/weight scale is not a compile-time scalar"); continue
        if not (G.zero_point_is_zero(xdq) and G.zero_point_is_zero(wdq)):
            reject("non-zero zero point"); continue

        # (4) accumulator-domain bias
        T = bias / (xs * ws)

        # (7) headroom, only when the requantization survives the rewrite
        keep_requant = requant_q is not None and not args.drop_requant
        if keep_requant:
            qs = G.scalar(requant_q.input[1])
            if qs is None:
                reject("requantization scale is not a compile-time scalar"); continue
            if not G.zero_point_is_zero(requant_q):
                reject("requantization has a non-zero zero point"); continue
            bias_lsb = float(np.abs(bias).max() / qs)
            if bias_lsb > args.max_bias_lsb:
                reject(f"bias is {bias_lsb:.1f} requantization LSB > --max-bias-lsb"); continue
        else:
            bias_lsb = None

        accepted.append(dict(mm=mm, name=name, xdq=xdq, wdq=wdq, q=requant_q, d=requant_d,
                             add=add, bias_name=bias_names[0], T=T, w=w, xs=xs, ws=ws,
                             keep_requant=keep_requant, bias_lsb=bias_lsb))
    return accepted, rejected


def solve_columns(Ts, args):
    """Column values V and per-matmul coefficients so that sum_i V[i]*b[i] == round(T).

    With --aug=pad every V[i] is the same constant, so the range has to be covered by
    column count alone.  With --aug=concat the column value adapts, which is the
    `c1*w1 + c2*w2` form and needs far fewer columns."""
    R = [T.copy() for T in Ts]
    V, B = [], [[] for _ in Ts]
    while max(float(np.abs(r).max()) for r in R) > 0.5:
        if len(V) >= args.max_columns:
            return None, None
        if args.aug == "pad":
            v = args.pad_value
        else:
            worst = max(float(np.abs(r).max()) for r in R)
            v = int(min(127, max(1, np.ceil(worst / 127))))
        V.append(v)
        for j, r in enumerate(R):
            b = np.clip(np.rint(r / v), -128, 127)
            B[j].append(b.astype(np.int8))
            R[j] = r - v * b
    if not V:  # bias rounds to zero everywhere; still needs one row to stay well-formed
        V = [args.pad_value]
        for j in range(len(Ts)):
            B[j].append(np.zeros(Ts[j].shape, np.int8))
    worst = max(float(np.abs(r).max()) for r in R)
    return (V, B, worst)


def activation_shape(G, tensor, K):
    """Static shape of the quantized activation.  A Reshape with a -1 leaves the last
    dimension symbolic in some saved models; the weight pins K, so fill it back in."""
    s = G.shapes.get(tensor)
    if not s or len(s) not in (2, 3):
        return None
    s = list(s)
    if s[-1] is None:
        s[-1] = K
    elif s[-1] != K:
        return None
    # a symbolic leading (batch) dimension is fine -- the rewrite emits -1 for it
    if any(d is None for d in s[1:]):
        return None
    if s[0] is None and len(s) == 2:
        return None
    return s


def main(argv):
    args = parse_args(argv)
    model = onnx.load(args.src)
    # The rule needs the activation's static shape; not every saved model carries one.
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception as e:                      # oversized models, unsupported ops
        print(f"shape inference unavailable ({e}); relying on stored value_info")
    G = Graph(model.graph)
    accepted, rejected = find_candidates(G, args)

    print(f"matmul candidates: {len(accepted)} accepted, {len(rejected)} rejected")
    for name, why in rejected:
        print(f"  - {name}: {why}")

    # One augmented activation per quantized-activation tensor, shared by its matmuls.
    groups = collections.defaultdict(list)
    for c in accepted:
        groups[c["xdq"].input[0]].append(c)

    plans = []
    for xi8, members in sorted(groups.items()):
        K_weight = members[0]["w"].shape[0]
        shape = activation_shape(G, xi8, K_weight)
        if shape is None or (len(shape) == 3 and shape[0] not in (1, None)):
            for c in members:
                print(f"  - {c['name']}: activation {xi8} has unsupported shape {shape}")
            continue
        K0 = shape[-1]
        if any(c["w"].shape[0] != K0 for c in members):
            for c in members:
                print(f"  - {c['name']}: weight K {c['w'].shape[0]} != activation K {K0}")
            continue
        solved = solve_columns([c["T"] for c in members], args)
        if solved[0] is None:
            for c in members:
                print(f"  - {c['name']}: bias needs more than --max-columns={args.max_columns} columns")
            continue
        V, B, worst = solved
        # (6) round the augmented K up to the target alignment; the extra rows are zero.
        K_target = int(np.ceil((K0 + len(V)) / args.align) * args.align)
        plans.append(dict(xi8=xi8, members=members, shape=shape, K0=K0, V=V, B=B,
                          worst=worst, K_target=K_target))
        names = ", ".join(c["name"].split("/")[-2] for c in members)
        print(f"  group {xi8}")
        print(f"    matmuls {len(members)} ({names}); columns {len(V)} value(s) "
              f"{sorted(set(V))}; K {K0} -> {K_target} (K/32 = {K_target / 32:g})")
        print(f"    bias error <= {worst:.4f} accumulator LSB; "
              f"requantization {'kept' if members[0]['keep_requant'] else 'dropped'}")
        for c in members:
            if c["bias_lsb"] is not None:
                print(f"      {c['name']}: bias = {c['bias_lsb']:.1f} requantization LSB")

    if not plans:
        print("nothing to fold; graph left unchanged")
        if not args.dry_run:
            onnx.save(model, args.dst)
            print("wrote (unchanged):", args.dst)
        return 0
    if args.dry_run:
        print("--dry-run: no output written")
        return 0

    g = model.graph
    new_nodes, drop, add_init, renames = [], set(), [], {}
    for gi, p in enumerate(plans):
        K0, K_target, V = p["K0"], p["K_target"], p["V"]
        pad_total = K_target - K0
        xaug = build_augmented_activation(g, G, gi, p, args, new_nodes, add_init)
        xaug_dq = f"kfold_xdq_{gi}"
        new_nodes.append(helper.make_node(
            "DequantizeLinear", [xaug] + list(p["members"][0]["xdq"].input[1:]),
            [xaug_dq], name=f"kfold_xdq_{gi}"))
        for j, c in enumerate(p["members"]):
            rows = np.zeros((pad_total, c["w"].shape[1]), np.int8)
            for i, b in enumerate(p["B"][j]):
                rows[i] = b
            wname = f"kfold_w_{gi}_{j}"
            add_init.append(nh.from_array(np.concatenate([c["w"], rows], axis=0), wname))
            wdq_out = f"{wname}_dq"
            new_nodes.append(helper.make_node(
                "DequantizeLinear", [wname] + list(c["wdq"].input[1:]), [wdq_out],
                name=f"kfold_wdq_{gi}_{j}"))
            # Rewrite in place: the matmul keeps its output name, so the requantization
            # chain behind it is untouched when we keep it.
            c["mm"].input[0] = xaug_dq
            c["mm"].input[1] = wdq_out
            drop.add(id(c["wdq"]))
            # The float bias add is now inside the accumulator; drop it and let its
            # producer take over its output name (this preserves graph outputs).
            tail_producer = c["d"] if c["keep_requant"] else c["mm"]
            if not c["keep_requant"] and c["q"] is not None:
                drop.add(id(c["q"])); drop.add(id(c["d"]))
            renames[tail_producer.output[0]] = c["add"].output[0]
            drop.add(id(c["add"]))

    finalize(model, g, new_nodes, drop, add_init, renames, args.dst)
    return 0


def merged_trailing_dims(G, xi8, pad_total):
    """If the quantized activation is Quantize(Reshape(t)) where the Reshape only merges
    the trailing dimensions of `t`, return (quantize, reshape, shape_of_t, inner) so the
    caller can pad the outer merged dimension instead.  `inner` is the innermost merged
    extent; padding that axis is only the same bytes when it divides the pad width."""
    q = G.prod.get(xi8)
    if q is None or q.op_type != "QuantizeLinear":
        return None
    r = G.prod.get(q.input[0])
    if r is None or r.op_type != "Reshape":
        return None
    src_shape = G.shapes.get(r.input[0])
    out_shape = G.shapes.get(r.output[0])
    if not src_shape or not out_shape or any(d is None for d in src_shape + out_shape):
        return None
    # a pure trailing merge: same leading dims, and the tail multiplies out
    if len(src_shape) != len(out_shape) + 1:
        return None
    if src_shape[:-2] != out_shape[:-1]:
        return None
    if src_shape[-2] * src_shape[-1] != out_shape[-1]:
        return None
    inner = src_shape[-1]
    if inner == 0 or pad_total % inner != 0:
        return None
    return q, r, src_shape, inner


def build_augmented_activation(g, G, gi, p, args, new_nodes, add_init):
    """Emit the K-axis augmentation of the quantized activation and return its name.

    The Pad is emitted on a 2-D [rows, K] view: with a 3-D [1, S, K] pad IREE keeps a
    `flow.tensor.reshape` between the producer and the pad, which blocks the producer
    from writing the widened buffer in place.  When the activation itself comes from a
    trailing-dimension flatten, `--pad-in-producer-layout` goes one better and pads
    above that flatten, where nothing separates producer and Pad at all."""
    xi8, K0, K_target, V = p["xi8"], p["K0"], p["K_target"], p["V"]
    pad_total = K_target - K0
    merged = merged_trailing_dims(G, xi8, pad_total) if (
        args.producer_layout and args.aug == "pad") else None
    if merged is not None:
        return build_in_producer_layout(gi, p, args, new_nodes, add_init, merged, pad_total)
    two_d = len(p["shape"]) == 2
    src = xi8
    if not two_d:
        sh2 = f"kfold_sh2_{gi}"
        add_init.append(nh.from_array(np.array([-1, K0], np.int64), sh2))
        src = f"kfold_x2_{gi}"
        new_nodes.append(helper.make_node("Reshape", [xi8, sh2], [src], name=f"kfold_r2_{gi}"))

    out = f"kfold_xaug2_{gi}"
    if args.aug == "pad":
        pads = f"kfold_pads_{gi}"
        add_init.append(nh.from_array(np.array([0, 0, 0, pad_total], np.int64), pads))
        val = f"kfold_padval_{args.pad_value}"
        if not any(i.name == val for i in add_init):
            add_init.append(nh.from_array(np.array(args.pad_value, np.int8), val))
        new_nodes.append(helper.make_node("Pad", [src, pads, val], [out],
                                          mode="constant", name=f"kfold_pad_{gi}"))
    else:
        cols = np.zeros((p["shape"][-2] if two_d else p["shape"][1], pad_total), np.int8)
        cols[:, :len(V)] = np.asarray(V, np.int8)
        cname = f"kfold_cols_{gi}"
        add_init.append(nh.from_array(cols, cname))
        new_nodes.append(helper.make_node("Concat", [src, cname], [out], axis=1,
                                          name=f"kfold_concat_{gi}"))

    if two_d:
        return out
    sh3 = f"kfold_sh3_{gi}"
    lead = p["shape"][0] if p["shape"][0] is not None else -1
    add_init.append(nh.from_array(np.array([lead, p["shape"][1], K_target], np.int64), sh3))
    xaug = f"kfold_xaug_{gi}"
    new_nodes.append(helper.make_node("Reshape", [out, sh3], [xaug], name=f"kfold_r3_{gi}"))
    return xaug


def build_in_producer_layout(gi, p, args, new_nodes, add_init, merged, pad_total):
    """Pad the outer merged dimension, above the flatten, then flatten to K_target."""
    q, reshape, src_shape, inner = merged
    extra = pad_total // inner
    rank = len(src_shape)
    pads = [0] * (2 * rank)
    pads[rank + rank - 2] = extra            # high padding of the outer merged axis
    pads_name = f"kfold_pads4_{gi}"
    add_init.append(nh.from_array(np.array(pads, np.int64), pads_name))
    val = f"kfold_padval_{args.pad_value}"
    if not any(i.name == val for i in add_init):
        add_init.append(nh.from_array(np.array(args.pad_value, np.int8), val))
    shape = [d if d is not None else -1 for d in src_shape[:-2]] + [p["K_target"]]
    shape_name = f"kfold_shape4_{gi}"
    add_init.append(nh.from_array(np.array(shape, np.int64), shape_name))

    qn = f"kfold_q4_{gi}"
    new_nodes.append(helper.make_node("QuantizeLinear",
                                      [reshape.input[0]] + list(q.input[1:]), [qn],
                                      name=qn))
    pn = f"kfold_pad4_{gi}"
    new_nodes.append(helper.make_node("Pad", [qn, pads_name, val], [pn],
                                      mode="constant", name=pn))
    xaug = f"kfold_xaug4_{gi}"
    new_nodes.append(helper.make_node("Reshape", [pn, shape_name], [xaug],
                                      name=f"kfold_r4_{gi}"))
    print(f"    padding in producer layout: {src_shape} axis {rank - 2} "
          f"{src_shape[-2]} -> {src_shape[-2] + extra} (inner {inner})")
    return xaug


def finalize(model, g, new_nodes, drop, add_init, renames, dst):
    kept = [n for n in g.node if id(n) not in drop]
    for n in kept:
        for i, o in enumerate(n.output):
            if o in renames:
                n.output[i] = renames[o]
    for n in new_nodes:
        for i, o in enumerate(n.output):
            if o in renames:
                n.output[i] = renames[o]
    nodes = kept + new_nodes

    outs = {o.name for o in g.output}
    while True:
        used = set(outs)
        for n in nodes:
            used.update(i for i in n.input if i)
        dead = [n for n in nodes if n.output and not any(o in used for o in n.output)]
        if not dead:
            break
        ids = {id(n) for n in dead}
        nodes = [n for n in nodes if id(n) not in ids]
        print(f"  removed {len(dead)} dead node(s)")

    g.initializer.extend(add_init)
    ready = {i.name for i in g.initializer} | {i.name for i in g.input}
    order, pending = [], list(nodes)
    while pending:
        rest, progressed = [], False
        for n in pending:
            if all(i == "" or i in ready for i in n.input):
                order.append(n); ready.update(n.output); progressed = True
            else:
                rest.append(n)
        if not progressed:
            raise SystemExit(f"topological sort stuck on {len(rest)} nodes, "
                             f"e.g. {rest[0].op_type} {list(rest[0].input)}")
        pending = rest
    del g.node[:]
    g.node.extend(order)

    used = set()
    for n in g.node:
        used.update(n.input)
    keep = [i for i in g.initializer if i.name in used]
    dropped = len(g.initializer) - len(keep)
    del g.initializer[:]
    g.initializer.extend(keep)

    del g.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"wrote {dst}: {len(model.graph.node)} nodes, dropped {dropped} unused initializer(s)")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
