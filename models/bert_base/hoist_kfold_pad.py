"""Move a K-fold activation Pad up into its producer's layout.

The K-fold rewrite widens an activation `[M, K] -> [M, K+p]` with a constant
column block. As a standalone `Pad` that is a real copy: IREE gives it its own
dispatch, and on this backend that dispatch lands on the host (`slow_memcpy`).
It costs nothing only when the producer can write the widened buffer in place,
and two things stop that:

  1. a flatten between the producer and the Pad. The producer writes
     `[1, M, K]`, the matmul wants `[M, K+p]`, and the `Reshape` in between means
     the Pad is no longer sitting on the producer's own result.
  2. the producer's result having a second reader. A residual path reads the
     same activation unpadded, and dispatch formation only fuses a producer that
     has a single consumer.

This fixes both, structurally and without naming any node:

  before   P -> [1,M,K] -> Reshape -> [M,K] -> Pad -> [M,K+p] -> matmul
                        \\-> other consumers

  after    P -> [1,M,K] -> Pad -> [1,M,K+p] -> Reshape -> [M,K+p] -> matmul
                                            \\-> Slice[..., :K] -> other consumers

Padding the last axis and then merging leading axes moves exactly the same bytes
as merging first and padding after, so the augmented columns keep their value and
their position. The other readers get a slice of the widened buffer, which is the
original activation element for element, so their scale and values are unchanged.
Nothing here is constant-folded: the activation stays runtime data, only the
shape of the buffer its producer writes into changes.

When the Pad already sits on the producer's layout (BERT's Q/K/V fold pads
[1, M, K] directly) only step 2 applies: the other readers are moved onto a slice
of the Pad's output.

For the copies to actually disappear the compiler has to see these slices and
widenings as in-place accesses; that needs third_party/iree at this branch's
pointer (the slice clone and the unit-collapse sink in DispatchCreation).

Refuses (leaving the Pad alone) unless every step is checkable:
  - the Pad pads only the last axis, only at the end, with a constant value;
  - the ops between producer and Pad are `Reshape`s that keep the last dimension;
  - the producer is a real node, not a graph input (nothing to write in place).
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh

VIEWS = ("Reshape",)


def shapes_of(g):
    out = {}
    for v in list(g.value_info) + list(g.input) + list(g.output):
        d = v.type.tensor_type.shape.dim
        out[v.name] = [x.dim_value if x.HasField("dim_value") else None for x in d]
    return out


def run(src, dst, verbose=True):
    m = onnx.load(src)
    g = m.graph
    init = {i.name: nh.to_array(i) for i in g.initializer}
    shape = shapes_of(g)
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def report(pad, why):
        if verbose:
            print(f"  - {pad.name or pad.output[0]}: {why}")

    new_nodes, new_inits, hoisted = [], [], 0
    uid = 0
    for pad in [n for n in g.node if n.op_type == "Pad"]:
        pads = init.get(pad.input[1])
        if pads is None:
            report(pad, "pad amounts are not a constant"); continue
        rank = len(pads) // 2
        begin, end = list(pads[:rank]), list(pads[rank:])
        if any(begin) or any(end[:-1]) or end[-1] <= 0:
            report(pad, "does not pad only the end of the last axis"); continue
        amount = int(end[-1])
        if len(pad.input) > 2 and pad.input[2] not in init:
            report(pad, "pad value is not a constant"); continue

        chain, t = [], pad.input[0]
        while True:
            p = prod.get(t)
            if p is None or p.op_type not in VIEWS:
                break
            src_sh, dst_sh = shape.get(p.input[0]), shape.get(p.output[0])
            if not src_sh or not dst_sh or src_sh[-1] != dst_sh[-1]:
                chain = None
                break
            chain.append(p)
            t = p.input[0]
        if chain is None:
            report(pad, "a reshape on the way changes the last dimension"); continue
        if prod.get(t) is None:
            report(pad, "its producer is a graph input -- nothing to write in place")
            continue
        if not chain:
            # Already on the producer's layout (the QKV fold pads [1, M, K]
            # directly). What still stops the producer from writing the widened
            # buffer in place is anyone else reading the unpadded value -- the
            # residual. Give them the same slice of the widened one instead.
            others = [c for c in cons.get(t, []) if c is not pad]
            if not others:
                report(pad, "already sits on its producer's layout, sole reader"); continue
            base = shape.get(t)
            if not base or base[-1] is None:
                report(pad, "producer's shape is not static"); continue
            uid += 1
            nm = f"kfhoist{uid}"
            new_inits += [
                nh.from_array(np.asarray([0], np.int64), f"{nm}_st"),
                nh.from_array(np.asarray([base[-1]], np.int64), f"{nm}_en"),
                nh.from_array(np.asarray([len(base) - 1], np.int64), f"{nm}_ax"),
            ]
            new_nodes.append(helper.make_node(
                "Slice", [pad.output[0], f"{nm}_st", f"{nm}_en", f"{nm}_ax"],
                [f"{nm}_slice"], name=f"{nm}_sl"))
            for c in others:
                for j, i in enumerate(c.input):
                    if i == t:
                        c.input[j] = f"{nm}_slice"
            hoisted += 1
            if verbose:
                print(f"  + {pad.name}: already on {prod[t].op_type} {base}; "
                      f"{len(others)} other reader(s) now read a slice of it")
            continue

        base = shape.get(t)
        if not base or base[-1] is None:
            report(pad, "producer's shape is not static"); continue
        orig_last = base[-1]

        uid += 1
        # (1) the Pad, now on the producer's own result
        prank = len(base)
        pad_amounts = [0] * prank + [0] * (prank - 1) + [amount]
        nm = f"kfhoist{uid}"
        new_inits.append(nh.from_array(np.asarray(pad_amounts, np.int64), f"{nm}_pads"))
        pad_in = [t, f"{nm}_pads"] + ([pad.input[2]] if len(pad.input) > 2 else [])
        new_nodes.append(helper.make_node("Pad", pad_in, [f"{nm}_padded"],
                                          name=f"{nm}_pad", mode="constant"))

        # (2) the same view chain, on the widened tensor
        cur = f"{nm}_padded"
        for k, view in enumerate(reversed(chain)):
            tgt = list(shape[view.output[0]])
            tgt[-1] = tgt[-1] + amount
            new_inits.append(nh.from_array(np.asarray(tgt, np.int64), f"{nm}_sh{k}"))
            out = pad.output[0] if k == len(chain) - 1 else f"{nm}_v{k}"
            new_nodes.append(helper.make_node("Reshape", [cur, f"{nm}_sh{k}"], [out],
                                              name=f"{nm}_r{k}"))
            cur = out

        # (3) everyone else who read the unpadded value now reads a slice of the
        #     widened one, so the producer is left with a single consumer.
        others = [c for c in cons.get(t, []) if c is not chain[-1]]
        if others:
            new_inits += [
                nh.from_array(np.asarray([0], np.int64), f"{nm}_st"),
                nh.from_array(np.asarray([orig_last], np.int64), f"{nm}_en"),
                nh.from_array(np.asarray([prank - 1], np.int64), f"{nm}_ax"),
            ]
            new_nodes.append(helper.make_node(
                "Slice", [f"{nm}_padded", f"{nm}_st", f"{nm}_en", f"{nm}_ax"],
                [f"{nm}_slice"], name=f"{nm}_sl"))
            for c in others:
                for j, i in enumerate(c.input):
                    if i == t:
                        c.input[j] = f"{nm}_slice"
        hoisted += 1
        if verbose:
            print(f"  + {pad.name}: pad {orig_last}->{orig_last + amount} moved onto "
                  f"{prod[t].op_type} {base}, {len(chain)} reshape(s) after it"
                  + (f", {len(others)} other reader(s) via Slice" if others else ""))
        pad.op_type = "Identity"  # marked for removal below
        del pad.input[1:]
        for view in chain:
            view.op_type = "Identity"
            del view.input[1:]

    if hoisted:
        keep = [n for n in g.node if n.op_type != "Identity" or n.output[0] not in
                {x.output[0] for x in g.node if x.op_type == "Identity"}]
        keep = [n for n in g.node if n.op_type != "Identity"]
        del g.node[:]
        g.node.extend(keep + new_nodes)
        g.initializer.extend(new_inits)
        # topological order
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
                raise SystemExit(f"topological sort stuck on {len(rest)} node(s), "
                                 f"e.g. {rest[0].op_type} {list(rest[0].input)}")
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
    print(f"wrote {dst}: {hoisted} pad(s) hoisted, {len(m.graph.node)} nodes")
    return hoisted


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
