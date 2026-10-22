# ONNX model scripts

Compile an ONNX model for amd-aie (NPU + CPU), run it on the NPU and check it
against a CPU reference. One set of options for everyone, kept in
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

Prerequisites: the build (`scripts/build/configure.sh`, `scripts/build/build.sh`)
and, for large models such as VGG, the host settings in
[`models/vgg16/README.md`](../../models/vgg16/README.md) section 2.

## Options

`options.py` groups the iree-compile flags by function:

| Group | Holds |
|---|---|
| `device` | `compile` flags that declare the devices, `run` flags that open them |
| `reference_device` | the same pair for the accuracy reference (llvm-cpu only) |
| `backend` | NPU generation, core compiler (Peano) |
| `graph` | model-level transforms before dispatch formation |
| `dispatch` | how ops are grouped into dispatches |
| `vector` | user options `--ukernel {none,all}`, `--vectorize {off,on}` |
| `numeric` | user option `--numeric {f32,bf16}`: what f32 contraction inputs become |

Defaults: `--ukernel all --vectorize off --numeric f32`. With `--numeric f32`,
`--ukernel none` compiles faster (VGG 65 s instead of 83 s) and runs the same:
f32 matmuls have no ukernel, only their zero fill does, which costs a ukernel
compile per NPU dispatch without a run-time gain. Fixed groups are always
passed. A flag that only works around a current gap is marked `# workaround:`;
remove it once the gap is fixed. To add, remove or change an option, edit
`options.py` only: each user option becomes a `--<name>` argument by itself.

To target other devices (e.g. CPU + GPU), fill `device.compile` and
`device.run` as a matching pair. `--extra-flag=<flag>` (repeatable) passes one
more flag to iree-compile without editing the file.

## Output

`build/onnx/<model>-<option>_<choice>...` unless `--outdir` is given:

| File | Content |
|---|---|
| `model.mlir`, `model.vmfb` | imported model, compiled module |
| `compile_cmd.sh` | import and compile commands, to reproduce by hand |
| `config.json` | options, compile/run flags of the model and its reference, commit, times |
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
