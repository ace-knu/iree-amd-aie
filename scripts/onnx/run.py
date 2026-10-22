#!/usr/bin/env python3
"""Run a model compiled by compile.py on amd-aie and check it.

Runs inside the dev container. Reads config.json and model.vmfb from the
compile output directory, runs the module on the NPU (+ CPU) and prints one
summary line: compile time, run time, accuracy.

Inputs: the --input .npy files, or (by default) fixed-seed random inputs of the
ONNX input shapes. They are passed to iree-run-module WITHOUT a shape prefix
(`--input=@x.npy`): a prefix makes iree read the file as raw bytes and swallow
the 128-byte npy header, yielding garbage.

Accuracy reference: the same imported model compiled with the reference flags
that compile.py wrote to config.json (same options on llvm-cpu, numeric f32),
run on the same inputs. The reference module is cached per model and flags
under build/onnx/ref/, its outputs per input set.

Run time: wall time of iree-run-module (module load + execution), the median
of --runs runs after --warmup runs. Writes result.json next to config.json.
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

import compile as compile_

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "build" / "tools"
REF_ROOT = REPO / "build" / "onnx" / "ref"


def add_run_args(ap):
    ap.add_argument("--input", action="append", default=[], dest="inputs", type=Path,
                    help=".npy input, repeat in ONNX input order; "
                         "default: fixed-seed random inputs")
    ap.add_argument("--runs", type=int, default=3, help="timed runs (default: 3)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed runs before (default: 1)")


def sha(paths, extra=(), n=12):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    for s in extra:
        h.update(s.encode())
    return h.hexdigest()[:n]


def make_inputs(onnx_path, outdir):
    """Writes fixed-seed random inputs of the ONNX graph input shapes."""
    graph = onnx.load(onnx_path, load_external_data=False).graph
    initializers = {i.name for i in graph.initializer}
    rng = np.random.default_rng(1)
    paths = []
    for i, value in enumerate(v for v in graph.input if v.name not in initializers):
        t = value.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        if any(d.HasField("dim_param") or d.dim_value <= 0 for d in t.shape.dim):
            sys.exit(f"input '{value.name}' has a dynamic shape; pass --input")
        dtype = helper.tensor_dtype_to_np_dtype(t.elem_type)
        if np.issubdtype(dtype, np.floating):
            x = (rng.standard_normal(dims) * 0.2).astype(dtype)
        else:
            x = rng.integers(0, 10, dims).astype(dtype)
        paths.append(outdir / f"input{i}.npy")
        np.save(paths[-1], x)
    return paths


def run_module(vmfb, function, run_flags, inputs, outputs, log):
    """Runs `function` of `vmfb`; returns the wall time in seconds."""
    return compile_.run_step(
        [str(TOOLS / "iree-run-module"), *run_flags, f"--module={vmfb}",
         f"--function={function}", *[f"--input=@{p}" for p in inputs],
         *[f"--output=@{p}" for p in outputs]], log)


def reference_outputs(config, mlir, inputs, num_outputs):
    """Returns the reference outputs for `inputs`, compiling and running the
    reference module when they are not cached."""
    reference = config["reference_device"]
    refdir = REF_ROOT / f"{config['model']}-{sha([mlir], reference['compile'])}"
    refdir.mkdir(parents=True, exist_ok=True)
    vmfb = refdir / "model.vmfb"
    if not vmfb.exists():
        compile_.build_vmfb(mlir, reference["compile"], vmfb, refdir / "compile.log")
    key = sha(inputs)
    outputs = [refdir / f"out{i}-{key}.npy" for i in range(num_outputs)]
    if not all(p.exists() for p in outputs):
        run_module(vmfb, config["function"], reference["run"], inputs, outputs,
                   refdir / "run.log")
    return outputs


def compare(got, ref):
    """Returns accuracy metrics of one output against its reference."""
    g, r = np.load(got).astype(np.float64), np.load(ref).astype(np.float64)
    m = {"corr": float(np.corrcoef(g.ravel(), r.ravel())[0, 1]) if g.size > 1 else None,
         "maxabs": float(np.abs(g - r).max())}
    # Logits of one sample ([N] or [1, N]): also check the top-1 class.
    if g.ndim == 1 or (g.ndim == 2 and g.shape[0] == 1):
        m["top1_match"] = bool(g.argmax() == r.argmax())
    return m


def format_metrics(m):
    s = [f"maxabs {m['maxabs']:.3g}"]
    if m["corr"] is not None:
        s.insert(0, f"corr {m['corr']:.5f}")
    if "top1_match" in m:
        s.append("top-1 " + ("match" if m["top1_match"] else "MISMATCH"))
    return ", ".join(s)


def run_model(outdir, inputs, runs, warmup):
    """Runs and checks the module in `outdir`; returns the result dict written
    to outdir/result.json."""
    config = json.loads((outdir / "config.json").read_text())
    vmfb, mlir = outdir / "model.vmfb", outdir / "model.mlir"
    num_outputs = len(onnx.load(config["onnx"], load_external_data=False).graph.output)
    for p in inputs:
        if not p.is_file():
            sys.exit(f"input {p} not found (only the repo is mounted in the container)")
    inputs = inputs or make_inputs(config["onnx"], outdir)

    outputs = [outdir / f"out{i}.npy" for i in range(num_outputs)]
    run_flags = config["device"]["run"]
    for _ in range(warmup):
        run_module(vmfb, config["function"], run_flags, inputs, [], outdir / "run.log")
    times = [run_module(vmfb, config["function"], run_flags, inputs, outputs,
                        outdir / "run.log") for _ in range(runs)]

    refs = reference_outputs(config, mlir, inputs, num_outputs)
    result = {
        "run_s": round(statistics.median(times), 3),
        "run_times_s": [round(t, 3) for t in times],
        "inputs": [str(p) for p in inputs],
        "accuracy": [compare(o, r) for o, r in zip(outputs, refs)],
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return config, result


def print_summary(config, result, runs):
    accuracy = " | ".join(format_metrics(m) for m in result["accuracy"])
    print(f"{config['model']}: compile {config['compile_s']:.1f} s"
          f" | run {result['run_s']:.2f} s (median of {runs})"
          f" | {accuracy}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="compile.py output directory")
    add_run_args(ap)
    args = ap.parse_args()
    config, result = run_model(args.dir, args.inputs, args.runs, args.warmup)
    print_summary(config, result, args.runs)


if __name__ == "__main__":
    main()
