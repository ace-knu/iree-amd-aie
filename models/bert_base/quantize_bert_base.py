#!/usr/bin/env python3
"""Quantize bert_base.onnx's 96 MatMuls to int8 QDQ (per-tensor, symmetric).

The Q/K/V/output/FFN projections and the batched QK^T / attention x V matmuls
(batch = 12 heads) are quantized; nothing else is.

Calibration:
  default         16 random `input_ids` rows from a fixed seed, MinMax. Reproducible
                  (byte-identical with onnxruntime 1.29.0) but a poor fit for real
                  text: corr vs fp32 on real sentences ~0.40.
  --calib-ids     real sentences (quant/calib_ids.npy, 90 rows). With
                  --method percentile (99.999) this is the accuracy baseline
                  (docs/2026-09-28_real_sentence_calibration.md). Lower
                  percentiles and entropy are worse -- the outliers carry signal.

Per-tensor on purpose: per-channel is the larger accuracy lever but the NPU
backend cannot compile it yet (docs/2026-09-10_per_channel_int8_blocked_in_npu_backend.md).

Usage:
  python3 models/bert_base/quantize_bert_base.py models/bert_base/bert_base.onnx bert_base_int8.onnx
  python3 models/bert_base/quantize_bert_base.py models/bert_base/bert_base.onnx bert_base_int8.onnx \
      --calib-ids models/bert_base/quant/calib_ids.npy --method percentile
"""
import argparse
from collections import Counter

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

VOCAB = 30522


class TokenRows(CalibrationDataReader):
    """Real sentences, one tokenized row of `input_ids` per calibration sample."""
    def __init__(self, ids):
        self.rows = iter([{"input_ids": r[None, :].astype(np.int64)} for r in ids])

    def get_next(self):
        return next(self.rows, None)


class RandomIds(CalibrationDataReader):
    def __init__(self, rows, seq_len, seed):
        rng = np.random.default_rng(seed)
        self.rows = iter([
            {"input_ids": rng.integers(0, VOCAB, size=(1, seq_len)).astype(np.int64)}
            for _ in range(rows)
        ])

    def get_next(self):
        return next(self.rows, None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="fp32 bert_base.onnx (export_bert_base.py)")
    p.add_argument("output", help="int8 QDQ model to write")
    p.add_argument("--sequence-length", type=int, default=32)
    p.add_argument("--calibration-rows", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--calib-ids", help="npy [N, seq] of token ids from real text; "
                                        "default: random ids")
    p.add_argument("--method", choices=("minmax", "percentile", "entropy"), default="minmax")
    p.add_argument("--percentile", type=float, default=99.999)
    a = p.parse_args()

    quantize_static(
        a.input, a.output,
        calibration_data_reader=(TokenRows(np.load(a.calib_ids)) if a.calib_ids else
                                 RandomIds(a.calibration_rows, a.sequence_length, a.seed)),
        calibrate_method={"minmax": CalibrationMethod.MinMax,
                          "percentile": CalibrationMethod.Percentile,
                          "entropy": CalibrationMethod.Entropy}[a.method],
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        extra_options={"ActivationSymmetric": True,
                       **({"CalibPercentile": a.percentile} if a.method == "percentile" else {})},
    )
    ops = Counter(n.op_type for n in onnx.load(a.output).graph.node)
    print(f"wrote {a.output}: {ops['MatMul']} MatMul, {ops['QuantizeLinear']} QuantizeLinear")


if __name__ == "__main__":
    main()
