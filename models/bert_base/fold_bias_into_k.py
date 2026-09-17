#!/usr/bin/env python3
"""Selectively fold a quantized BERT MatMul bias into its K dimension.

This is an ONNX *model preparation* tool, not a compiler pass. It rewrites
only the requested `layer.<n>/<kind>/MatMul` nodes from

  DQ(x_i8) @ DQ(w_i8) -> Q -> DQ -> Add(constant_bias)

into a two-input matmul whose activation and weight each have extra K rows.
The padded activation rows are one and the matching weight rows encode the
bias in accumulator units. `Pad` is used rather than `Concat` so IREE can
fold the extension into the producer dispatch rather than creating memcpy
dispatches. Zero-points must be zero.

Example: fold FC1 only (BERT-base sequence length 32):
  python3 models/bert_base/fold_bias_into_k.py input_int8.onnx fc1_kfold.onnx \
    --include intermediate/dense

The default 64-column alignment deliberately turns K=768 into K=832. The
N=3072 FC1 path cannot currently handle a K/32 odd value such as K=800.
"""

import argparse
import collections
import re

import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper as nh


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="quantized BERT ONNX model")
    parser.add_argument("output", help="rewritten ONNX model")
    parser.add_argument(
        "--include",
        required=True,
        help=("comma-separated MatMul kinds to rewrite; e.g. "
              "intermediate/dense or attention/self/query,attention/self/key"),
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--align", type=int, default=64)
    return parser.parse_args()


def only_consumer(consumers, value, op_type):
    matches = [node for node in consumers.get(value, []) if node.op_type == op_type]
    return matches[0] if len(matches) == 1 else None


def split_int8(values, count):
    """Splits each integer into `count` signed-i8 values with the same sum."""
    result = np.zeros((count, values.size), dtype=np.int64)
    remaining = values.astype(np.int64).copy()
    for i in range(count):
        part = np.clip(remaining, -128, 127)
        result[i] = part
        remaining -= part
    if np.any(remaining):
        raise ValueError("bias needs more K columns than were allocated")
    return result.astype(np.int8)


def main():
    args = parse_args()
    if args.align <= 0 or args.align % 32:
        raise ValueError("--align must be a positive multiple of 32")
    included = set(filter(None, args.include.split(",")))
    model = onnx.load(args.input)
    graph = model.graph
    initializers = {item.name: nh.to_array(item) for item in graph.initializer}
    initializer_names = set(initializers)
    producers = {output: node for node in graph.node for output in node.output}
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for value in node.input:
            consumers[value].append(node)

    groups = collections.defaultdict(list)
    for matmul in graph.node:
        if matmul.op_type != "MatMul":
            continue
        match = re.search(r"layer\.(\d+)/(.+?)/MatMul", matmul.output[0])
        if not match or match.group(2) not in included:
            continue
        x_dq, w_dq = (producers.get(matmul.input[0]), producers.get(matmul.input[1]))
        if not x_dq or not w_dq or x_dq.op_type != "DequantizeLinear" or w_dq.op_type != "DequantizeLinear":
            continue
        if w_dq.input[0] not in initializer_names:
            continue
        quant = only_consumer(consumers, matmul.output[0], "QuantizeLinear")
        dequant = only_consumer(consumers, quant.output[0], "DequantizeLinear") if quant else None
        if not dequant:
            continue
        add = next(
            (node for node in consumers.get(dequant.output[0], [])
             if node.op_type == "Add" and any(value in initializer_names
                                                for value in node.input
                                                if value != dequant.output[0])),
            None,
        )
        if not add:
            continue
        if any(len(dq.input) > 2 and np.any(initializers[dq.input[2]] != 0)
               for dq in (x_dq, w_dq)):
            raise ValueError(f"{matmul.name}: non-zero quantization zero point")
        bias_name = next(value for value in add.input if value != dequant.output[0])
        x_scale = float(np.asarray(initializers[x_dq.input[1]], dtype=np.float64))
        w_scale = float(np.asarray(initializers[w_dq.input[1]], dtype=np.float64))
        groups[x_dq.input[0]].append({
            "matmul": matmul, "x_dq": x_dq, "w_dq": w_dq, "quant": quant,
            "dequant": dequant, "add": add, "kind": match.group(2),
            "tag": re.sub(r"[^0-9A-Za-z]+", "_", matmul.output[0]),
            "bias_units": np.asarray(initializers[bias_name], dtype=np.float64).ravel() / (x_scale * w_scale),
        })

    selected = sum(len(group) for group in groups.values())
    if not selected:
        raise ValueError(f"no eligible MatMul+bias nodes found for: {args.include}")
    print(f"selected {selected} MatMul+bias nodes in {len(groups)} activation groups")

    new_nodes, added_initializers, rewires, dropped = [], [], {}, set()
    seen_constants = set()
    max_error = 0.0
    for group_id, (x_i8, records) in enumerate(sorted(groups.items())):
        max_units = max(np.abs(record["bias_units"]).max() for record in records)
        columns = max(args.align, int(np.ceil(np.ceil(max_units / 127) / args.align) * args.align))
        weight = np.asarray(initializers[records[0]["w_dq"].input[0]])
        k = weight.shape[0]
        pads_name = f"kfold_pads_{columns}"
        one_name = "kfold_one_i8"
        shape_2d_name = f"kfold_shape2_{k}"
        shape_3d_name = f"kfold_shape3_{group_id}"
        for name, value in (
            (pads_name, np.array([0, 0, 0, columns], dtype=np.int64)),
            (one_name, np.array(1, dtype=np.int8)),
            (shape_2d_name, np.array([-1, k], dtype=np.int64)),
        ):
            if name not in seen_constants:
                added_initializers.append(nh.from_array(value, name))
                seen_constants.add(name)
        added_initializers.append(nh.from_array(
            np.array([1, args.sequence_length, k + columns], dtype=np.int64), shape_3d_name))
        x2, xp2, x_aug = (f"kfold_x2_{group_id}", f"kfold_xp2_{group_id}", f"kfold_x_{group_id}")
        new_nodes.extend([
            helper.make_node("Reshape", [x_i8, shape_2d_name], [x2], name=f"kfold_reshape2_{group_id}"),
            helper.make_node("Pad", [x2, pads_name, one_name], [xp2], mode="constant", name=f"kfold_pad_{group_id}"),
            helper.make_node("Reshape", [xp2, shape_3d_name], [x_aug], name=f"kfold_reshape3_{group_id}"),
        ])
        x_aug_dq = f"{x_aug}_dq"
        new_nodes.append(helper.make_node(
            "DequantizeLinear", [x_aug] + list(records[0]["x_dq"].input[1:]), [x_aug_dq],
            name=f"kfold_xdq_{group_id}"))
        for record in records:
            quantized_bias = np.rint(record["bias_units"])
            max_error = max(max_error, float(np.abs(record["bias_units"] - quantized_bias).max()))
            extension = split_int8(quantized_bias, columns)
            old_weight = np.asarray(initializers[record["w_dq"].input[0]])
            new_weight_name = f"kfold_weight_{group_id}_{record['tag']}"
            added_initializers.append(nh.from_array(np.concatenate([old_weight, extension], axis=0), new_weight_name))
            weight_dq = f"{new_weight_name}_dq"
            matmul_output = f"kfold_matmul_{new_weight_name}"
            new_nodes.append(helper.make_node(
                "DequantizeLinear", [new_weight_name] + list(record["w_dq"].input[1:]), [weight_dq],
                name=f"kfold_wdq_{new_weight_name}"))
            new_nodes.append(helper.make_node(
                "MatMul", [x_aug_dq, weight_dq], [matmul_output], name=f"kfold_matmul_{new_weight_name}"))
            for node in (record["matmul"], record["quant"], record["dequant"], record["add"], record["w_dq"]):
                dropped.add(id(node))
            rewires[record["add"].output[0]] = matmul_output

    kept = []
    for node in graph.node:
        if id(node) in dropped:
            continue
        for index, value in enumerate(node.input):
            if value in rewires:
                node.input[index] = rewires[value]
        kept.append(node)
    nodes = kept + new_nodes
    outputs = {value.name for value in graph.output}
    while True:
        used = set(outputs)
        for node in nodes:
            used.update(value for value in node.input if value)
        dead = [node for node in nodes if node.output and not any(value in used for value in node.output)]
        if not dead:
            break
        dead_ids = {id(node) for node in dead}
        nodes = [node for node in nodes if id(node) not in dead_ids]
    graph.initializer.extend(added_initializers)
    ready = {item.name for item in graph.initializer} | {item.name for item in graph.input}
    ordered, pending = [], list(nodes)
    while pending:
        next_pending = []
        for node in pending:
            if all(value in ready or not value for value in node.input):
                ordered.append(node)
                ready.update(node.output)
            else:
                next_pending.append(node)
        if len(next_pending) == len(pending):
            raise ValueError(f"cannot topologically order graph near {pending[0].name}")
        pending = next_pending
    del graph.node[:]
    graph.node.extend(ordered)
    used = {value for node in graph.node for value in node.input}
    kept_initializers = [item for item in graph.initializer if item.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, args.output)
    print(f"wrote {args.output}; max bias representation error {max_error:.4f} accumulator LSB")


if __name__ == "__main__":
    main()
