# ONNX model scripts

Compile an ONNX model for amd-aie (CPU + NPU, optionally GPU), run it and check
it against a CPU reference. One set of options for everyone, kept in
[`options.py`](options.py).

| Script | Does |
|---|---|
| `compile.py <model.onnx>` | import to MLIR, compile, write the output directory |
| `run.py <outdir>` | run a compiled directory, print time and accuracy |
| `e2e.py <model.onnx>` | `compile.py` then `run.py`, same options as both |

## Run

From the host, through the dev container (the ONNX file and the output directory
must be inside the repo; only the repo is mounted):

```bash
./scripts/docker/run-onnx.sh e2e models/vgg16/vgg16-12.onnx --numeric bf16
./scripts/docker/run-onnx.sh run build/onnx/vgg16-12-ukernel_all-vectorize_off-numeric_bf16 --verbose
```

Inside the dev container (`./scripts/docker/run-dev.sh`), from `/workspace`,
after `source /opt/venv/bin/activate` and
`export PYTHONPATH=/workspace/build/compiler/bindings/python`:

```bash
python3 scripts/onnx/e2e.py models/mlp_2layer/mlp_2layer.onnx
```

`--devices` picks the devices (default `local-host,amdxdna-npu4`). `local-host`
must come first: the CPU runs the host code and is the default and lead device.
The accelerators after it take contraction/conv dispatches round-robin in the
listed order. The GPU supports `--numeric f32` only.

```bash
./scripts/docker/run-onnx.sh e2e models/vgg16/vgg16-12.onnx --devices local-host,amdxdna-npu4,vulkan-gfx1150
```

Prerequisites: the build (`scripts/build/configure.sh`, `scripts/build/build.sh`)
and, for large models such as VGG, the host settings in
[`models/vgg16/README.md`](../../models/vgg16/README.md) section 2.

## Options

`DEVICES` in `options.py` holds, per device (`<driver>-<target>`), the
`compile` flags that declare it and the `run` flags that open it.
`DEFAULT_DEVICES` and `REFERENCE_DEVICES` (llvm-cpu only, for the accuracy
reference) list the devices used without `--devices`.

`OPTIONS` groups the other iree-compile flags by function:

| Group | Holds |
|---|---|
| `backend` | NPU generation, core compiler (Peano) |
| `graph` | model-level transforms before dispatch formation |
| `dispatch` | how ops are grouped into dispatches |
| `vector` | user options `--ukernel {none,all}`, `--vectorize {off,on}` |
| `numeric` | user option `--numeric {f32,bf16}`: what f32 contraction inputs become |
| `placement` | user option `--pool {cpu,accel}`: whether max pooling runs on the CPU or on the accelerators |

Defaults: `--ukernel all --vectorize off --numeric f32 --pool cpu`. `--pool accel`
places each pooling dispatch like a contraction, round-robin over the
accelerators in `--devices` order (with CPU + NPU, all go to the NPU). It works
but is slower on VGG (CPU + NPU bf16 2.04 s instead of 1.71 s): each pooling
becomes its own dispatch, and its hardware-context and DMA cost exceed the
pooling itself on the CPU (under 1 ms). A pooling fused with a layout transpose
(VGG's last one) stays on the CPU. With `--numeric f32`,
`--ukernel none` compiles faster (VGG 65 s instead of 88 s) and runs the same:
f32 matmuls have no ukernel, only their zero fill does, which costs a ukernel
compile per NPU dispatch without a run-time gain. Fixed groups are always
passed. A flag that only works around a current gap is marked `# workaround:`;
remove it once the gap is fixed. To add, remove or change an option, edit
`options.py` only: each user option becomes a `--<name>` argument by itself.

To add a device, add a `DEVICES` entry with its `compile` and `run` flags.
In the deploy image, through `./scripts/docker/run-deploy.sh` (the current
directory is mounted at `/work`; outputs go to `onnx/` in it). The image has
the same scripts; `--verbose` is not available there (no traced runtime):

```bash
./scripts/docker/run-deploy.sh e2e vgg16-12.onnx --numeric bf16
```

`--extra-flag=<flag>` (repeatable) passes one more flag to iree-compile without
editing the file, e.g. an experimental `--iree-amdaie-placement-map=...` (see
[`docs/device_placement.md`](../../docs/device_placement.md)).

## Output

`build/onnx/<model>-<option>_<choice>...` unless `--outdir` is given
(`<model>-devices_<a>+<b>...-...` when `--devices` is not the default).
The environment can move the paths; the deploy image sets them:
`ONNX_OUT` for `build/onnx`, `IREE_TOOLS` for `build/tools`, and
`IREE_AMD_AIE_COMMIT` for the commit when there is no git checkout.

| File | Content |
|---|---|
| `model.mlir`, `model.vmfb` | imported model, compiled module |
| `compile_cmd.sh` | import and compile commands, to reproduce by hand |
| `config.json` | devices, options, compile/run flags of the model and its reference, commit, times |
| `result.json` | run times, inputs, accuracy per output |
| `*.log` | tool output of each step; `trace.log` with `--verbose` |

`run.py` reads only `config.json`, so any compiled directory can be run again.

## Summary line

```
vgg16-12: compile 102.7 s | run 2.28 s (median of 3) | corr 0.99997, maxabs 0.0333, top-1 match
```

- **compile**: iree-compile time.
- **run**: iree-run-module wall time (module load + execution), the median of
  `--runs` (3) runs after `--warmup` (1).
- **accuracy**: against the same model compiled with the same options for
  `reference_device` with `--numeric f32`. Top-1 is shown for logits (`[N]` or
  `[1, N]`). The reference is cached under `build/onnx/ref/`.

Inputs are fixed-seed random tensors of the ONNX input shapes unless `--input
x.npy` is given (repeat in input order).

## Per-dispatch times (`--verbose`)

One more run with the iree-run-module of `build-trace/`, a runtime built with
console tracing, then per-dispatch times from its log:

```
#0   mxnet_converted_model$async_dispatch_0_transpose_3x224x224_f32  -> cpu      0.839 ms
#4   mxnet_converted_modelasync_dispatch_2_mat_0                     -> npu      7.225 ms
...
---
npu    23 dispatches     0.194 s
cpu    69 dispatches     0.067 s
host                     1.839 s  (hwctx create/destroy 1.752 s)
total                    2.100 s
```

- **npu**: submit + wait of each NPU run, without hardware context
  create/destroy.
- **cpu**: first start to last end of the dispatch's workgroups over the worker
  threads.
- **total**: the function call. **host**: total minus the dispatch times; the
  hardware context create/destroy part is shown in parentheses.

`scripts/build/build.sh` builds `build-trace/` along with `build/`. `--verbose`
stops if the traced iree-run-module is missing or older than the main one, so
the traced runtime matches the one that produced the summary line.

Limits:
- NPU names are the kernel names, which drop the `$` and the shape suffix of
  the CPU-style export names; the `dispatch_<n>` index still matches.
- Tracing every zone adds a little time, more on CPU dispatches (many small
  workgroup zones). Use the summary line for run time, the verbose part for
  the breakdown.
