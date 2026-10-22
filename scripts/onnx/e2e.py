#!/usr/bin/env python3
"""Compile an ONNX model for amd-aie, run it and check it (compile.py + run.py).

Runs inside the dev container. Takes the options of both scripts and prints
one summary line: compile time, run time, accuracy.
"""

import argparse
from pathlib import Path

import compile as compile_
import options
import run


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="ONNX model file")
    compile_.add_compile_args(ap)
    run.add_run_args(ap)
    args = ap.parse_args()

    selected = {name: getattr(args, name) for name in options.user_options()}
    outdir = args.outdir or compile_.default_outdir(args.onnx.stem, selected)
    compile_.compile_model(args.onnx, selected, outdir, args.extra_flags)
    config, result, report = run.run_model(outdir, args.inputs, args.runs, args.warmup,
                                           args.verbose)
    run.print_summary(config, result, args.runs, report)


if __name__ == "__main__":
    main()
