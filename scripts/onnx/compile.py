#!/usr/bin/env python3
"""Compile an ONNX model for amd-aie (NPU + CPU, optionally GPU).

Runs inside the dev container. Imports the ONNX file to MLIR, then runs
iree-compile with the flags from options.py for the selected options. Writes
into the output directory:

  model.mlir       imported model
  model.vmfb       compiled module
  compile.log      iree-compile output
  compile_cmd.sh   the exact commands, to reproduce by hand
  config.json      model, entry function, devices, selected options, the
                   compile and run flags of the model and of its accuracy
                   reference, commit, times

run.py reads config.json, so a directory produced here can be run as is.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import options

REPO = Path(__file__).resolve().parents[2]
IREE_COMPILE = REPO / "build" / "tools" / "iree-compile"


def device_list(value):
    devices = value.split(",")
    for d in devices:
        if d not in options.DEVICES:
            raise argparse.ArgumentTypeError(
                f"unknown device '{d}' (known: {', '.join(options.DEVICES)})")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError(f"device listed twice: {value}")
    return devices


def add_compile_args(ap):
    """Adds --devices, one --<option> argument per user option in options.py,
    plus --extra-flag and --outdir."""
    ap.add_argument("--devices", type=device_list, default=options.DEFAULT_DEVICES,
                    help="comma-separated devices from options.py, local-host "
                         "(the lead device) first; default: "
                         + ",".join(options.DEFAULT_DEVICES))
    for name, choices in options.user_options().items():
        ap.add_argument(f"--{name}", choices=choices,
                        default=options.DEFAULTS[name],
                        help=f"default: {options.DEFAULTS[name]}")
    ap.add_argument("--extra-flag", action="append", default=[], dest="extra_flags",
                    help="flag passed to iree-compile as is, repeatable; write it "
                         "as --extra-flag=<flag> (a separate value that starts "
                         "with '-' is read as an option)")
    ap.add_argument("--outdir", type=Path,
                    help="default: build/onnx/<model>-<options>")


def default_outdir(model, devices, selected):
    parts = [model]
    if devices != options.DEFAULT_DEVICES:
        parts.append("devices_" + "+".join(devices))
    label = "-".join(parts + [f"{k}_{v}" for k, v in selected.items()])
    return REPO / "build" / "onnx" / label


def run_step(cmd, log):
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = log.read_text().splitlines()[-20:]
        sys.exit(f"[FAIL] {cmd[0]} (rc={rc}), see {log}\n" + "\n".join(tail))
    return time.time() - t0


def entry_function(mlir):
    """Returns the first public func.func name of the imported module."""
    with open(mlir) as f:
        for line in f:
            m = re.match(r"\s*func\.func @([\w$.-]+)", line)
            if m:
                return m[1]
    sys.exit(f"no func.func found in {mlir}")


def compile_cmd(mlir, flags, vmfb):
    return [str(IREE_COMPILE), str(mlir), "-o", str(vmfb), *flags]


def build_vmfb(mlir, flags, vmfb, log):
    """Compiles `mlir` with `flags` into `vmfb`; returns the compile time."""
    return run_step(compile_cmd(mlir, flags, vmfb), log)


def device_flags(selected, extra_flags, devices):
    """Returns the compile and run flags of `selected` on `devices`."""
    return {"compile": options.compile_flags(selected, devices) + extra_flags,
            "run": options.run_flags(devices)}


def compile_model(onnx, devices, selected, outdir, extra_flags):
    """Imports and compiles `onnx`; returns the config dict written to
    outdir/config.json. The config also holds the flags of the accuracy
    reference (same options on options.REFERENCE_DEVICES, numeric f32), which
    run.py compiles when it needs it."""
    error = options.check(selected, devices)
    if error:
        sys.exit(error)
    outdir.mkdir(parents=True, exist_ok=True)
    mlir, vmfb = outdir / "model.mlir", outdir / "model.vmfb"
    device = device_flags(selected, extra_flags, devices)
    reference = device_flags({**selected, "numeric": "f32"}, extra_flags,
                             options.REFERENCE_DEVICES)
    import_cmd = [sys.executable, "-m", "iree.compiler.tools.import_onnx",
                  str(onnx), "-o", str(mlir)]
    (outdir / "compile_cmd.sh").write_text(
        "#!/bin/bash\nset -e\n" + "\n".join(
            " \\\n  ".join(shlex.quote(a) for a in c)
            for c in (import_cmd, compile_cmd(mlir, device["compile"], vmfb))) + "\n")

    import_s = run_step(import_cmd, outdir / "import.log")
    compile_s = build_vmfb(mlir, device["compile"], vmfb, outdir / "compile.log")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    config = {
        "model": onnx.stem,
        "onnx": str(onnx.resolve()),
        "function": entry_function(mlir),
        "devices": devices,
        "options": selected,
        "extra_flags": extra_flags,
        "device": device,
        "reference_device": reference,
        "commit": commit,
        "import_s": round(import_s, 2),
        "compile_s": round(compile_s, 2),
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="ONNX model file")
    add_compile_args(ap)
    args = ap.parse_args()

    selected = {name: getattr(args, name) for name in options.user_options()}
    outdir = args.outdir or default_outdir(args.onnx.stem, args.devices, selected)
    config = compile_model(args.onnx, args.devices, selected, outdir, args.extra_flags)
    print(f"compile {config['compile_s']:.1f} s (import {config['import_s']:.1f} s)"
          f" -> {outdir / 'model.vmfb'}")


if __name__ == "__main__":
    main()
