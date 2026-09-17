# VGG-16 (ONNX) end-to-end on npu4

Full `vgg16-12.onnx` — 13 conv + 15 relu + 5 maxpool + 3 flatten + 3 FC + 2 dropout,
input `[1,3,224,224]` -> `[1,1000]` logits — compiled and run heterogeneously:
conv/matmul on the NPU, pooling/flatten/layout ops on the CPU.

This is the largest model the amd-aie plugin is known to run end to end, so the
recipe below doubles as the reference for which compile flags a real f32 CNN
currently needs. Every flag is listed with the reason it is required; none of
them is optional today.

## 1. Get the model

Not committed (528 MB). From the ONNX Model Zoo:

```bash
curl -fSL -o models/vgg16/vgg16-12.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-12.onnx
```

## 2. Host prerequisites

Two host-side settings are required; without them the run fails at runtime, not
at compile time.

- **NPU dispatch timeout.** The default 2 s TDR aborts VGG's large dispatches
  with `ert state 6`. Raise it once (survives reboot) — see
  [`USER_GUIDE.md` 1-3](../../docs/2026-07-06_env_setup/USER_GUIDE.md).
- **`RLIMIT_MEMLOCK`.** amdxdna `HOST_ONLY` BOs pin memory; the default 8 MB
  container cap makes VGG's im2col/activation BOs fail with `errno 11`. The
  `scripts/docker/run-*.sh` wrappers already pass `--ulimit memlock=-1`.

## 3. Compile and run

Inside the dev container (`./scripts/docker/run-dev.sh`), from `/workspace` (not
from `models/vgg16` — every path below is relative to the repo root):

```bash
# import_onnx and the checker in 3b both live in the container's venv.
source /opt/venv/bin/activate
export PYTHONPATH=/workspace/build/compiler/bindings/python

python3 -m iree.compiler.tools.import_onnx models/vgg16/vgg16-12.onnx -o /tmp/vgg.mlir

# The run below reads one [1,3,224,224] float32 image. Any input exercises the
# pipeline; a fixed random one keeps the numbers reproducible. Pass a real
# preprocessed image instead when you care about the predicted class rather
# than about the compiler.
python3 -c "import numpy as np; np.random.seed(1); \
  np.save('input.npy', (np.random.randn(1,3,224,224)*0.2).astype(np.float32))"

build/tools/iree-compile /tmp/vgg.mlir -o /tmp/vgg.vmfb \
  --iree-hal-target-device=npu=amdxdna \
  --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-global-opt-use-im2col-for-convs \
  --iree-amdaie-demote-contraction-inputs-to-bf16 \
  --iree-amdaie-enable-vectorization-passes=false \
  --iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-convert-conv-to-channels-last)' \
  --iree-dispatch-creation-no-fuse-into-contraction-conv-roots \
  --iree-global-opt-detach-elementwise-through-reshape \
  --iree-flow-enable-executable-deduplication=false

./scripts/lock/with-npu-lock.sh build/tools/iree-run-module --device=amdxdna --device=local-task \
  --module=/tmp/vgg.vmfb --function=mxnet_converted_model \
  --input=@input.npy --output=@out.npy
```

The run above touches the shared host's one physical NPU, so it's wrapped in
`scripts/lock/with-npu-lock.sh` — see `docs/2026-07-06_env_setup/DEV_CONTAINER.md`
§5 for why. `iree-compile` above it doesn't touch the NPU and doesn't need this.

The entry function name comes from the ONNX graph name; read it back with
`grep -oE "func.func @[A-Za-z0-9_]+" /tmp/vgg.mlir | head -1`.

Measured on npu4: compile ~130 s, run ~75 s, output `[1,1000]` correlates
1.00000 with a torch reference and the argmax matches.

## 3b. Check the result

A run that produces numbers is not a run that produces *correct* numbers, and
nothing in the flow above will tell you the difference. Build the reference by
interpreting the same ONNX graph with torch and compare:

```bash
python3 - <<'PY'
import numpy as np, onnx, torch, torch.nn.functional as F
ONNX, X, OUT = "models/vgg16/vgg16-12.onnx", "input.npy", "out.npy"
g = onnx.load(ONNX).graph
init = {i.name: torch.tensor(onnx.numpy_helper.to_array(i).copy()) for i in g.initializer}
def attr(n, k, d=None):
    for a in n.attribute:
        if a.name == k:
            return list(a.ints) if a.ints else a.i
    return d
env = {g.input[0].name: torch.tensor(np.load(X))}
for n in g.node:
    t, x = n.op_type, env[n.input[0]]
    if t == "Conv":
        w = init[n.input[1]]; b = init[n.input[2]] if len(n.input) > 2 else None
        env[n.output[0]] = F.conv2d(x, w, b, stride=attr(n, "strides", [1, 1])[0],
                                    padding=attr(n, "pads", [0, 0, 0, 0])[0])
    elif t == "Relu":     env[n.output[0]] = F.relu(x)
    elif t == "MaxPool":  env[n.output[0]] = F.max_pool2d(
                              x, attr(n, "kernel_shape", [2, 2])[0],
                              attr(n, "strides", [2, 2])[0])
    elif t == "Flatten":  env[n.output[0]] = x.reshape(x.shape[0], -1)
    elif t == "Gemm":     env[n.output[0]] = x @ init[n.input[1]].t() + (
                              init[n.input[2]] if len(n.input) > 2 else 0)
    elif t == "Dropout":  env[n.output[0]] = x
    else: raise SystemExit("unhandled op " + t)
gold = env[g.output[0].name].numpy(); got = np.load(OUT)
print("corr %.5f  argmax gold=%d got=%d  top-1 match=%s" % (
    np.corrcoef(gold.ravel(), got.ravel())[0, 1],
    gold.argmax(), got.argmax(), gold.argmax() == got.argmax()))
PY
```

Expected: `corr 1.00000` and a top-1 match. Correlation well below 1.0, or a
mismatched argmax, means a dispatch is reading the wrong data — not a rounding
difference. Note bf16 demotion (see below) makes exact equality the wrong thing
to check; correlation and top-1 are the meaningful signals.

This is also the check to run after changing anything in the DMA, placement or
codegen paths: the compiler's own test suite is IR-level and cannot observe a
DMA that reads from the wrong address.

## 4. Why each flag

| Flag | Why |
| --- | --- |
| `--iree-hal-target-device=npu=amdxdna` + `cpu=local` + `--iree-hal-local-target-device-backends=llvm-cpu` + `--iree-hal-default-device=npu` | Declares both devices. Without a CPU device the heterogeneous placement pass is a no-op (`AMDAIEAssignDeviceAffinities` requires a host device), so pooling/flatten have nowhere to go and the model does not compile. |
| `--iree-amdaie-target-device=npu4` | Strix. Phoenix would be `npu1_4col`. |
| `--iree-amd-aie-peano-install-dir` | `PEANO_INSTALL_DIR` alone is not read by `iree-compile`. |
| `--iree-global-opt-use-im2col-for-convs` | The only conv path that reaches the NPU: convs are lowered to matmuls. Standard NCHW conv codegen is not supported. |
| `--iree-amdaie-demote-contraction-inputs-to-bf16` | npu4 has no f32 vector path. f32 conv aborts and f32 matmul only runs scalar; demoting inputs (accumulation stays f32) makes them run as bf16. This is also the precondition for turning vectorization on later — npu4's matmul intrinsics are bf16 — so it stays even though today, paired with vectorization off, it only costs precision. |
| `--iree-amdaie-enable-vectorization-passes=false` | aievec currently only handles i8 `vector.contract`; bf16 fails to vectorize, so the vectorization passes must be off. |
| `--iree-preprocessing-pass-pipeline=...convert-conv-to-channels-last` | ONNX convs are NCHW; the im2col path expects NHWC. |
| `--iree-dispatch-creation-no-fuse-into-contraction-conv-roots` | Works around a codegen gap: amd-aie cannot tile a contraction dispatch that has elementwise ops fused into it, so fusion across contraction/conv group boundaries is switched off. The real fix is on the codegen side, not here. |
| `--iree-global-opt-detach-elementwise-through-reshape` | Detaches the bias add from the conv init through the im2col reshape, so the contraction dispatch has a plain `linalg.fill` init. |
| `--iree-flow-enable-executable-deduplication=false` | Deduplicating two convs that differ only in constant-weight offset turns that offset into a runtime push constant, which `amdaie.npu.address_patch` cannot carry (it is static-only). The N-split pass also relies on its clones not being merged. |

The last three are flags of the `ace-knu/iree` fork (see `.gitmodules`); they do
not exist in upstream IREE.

## 5. Known limitations

- Batch size is fixed at 1 by the ONNX file; other batch sizes are untested.
- The flag list is not a stable interface, and passing it by hand is a stopgap.
  Three of the flags exist only in the `ace-knu/iree` fork, so nothing in
  `--help` or in upstream documentation leads a reader to this combination —
  this file is the only place it is written down. Two more work around current
  gaps (aievec has no bf16 path; `no-fuse` above) and should disappear as those
  are fixed.
- Omitting a flag does not produce a diagnostic naming it; the failure is a
  compile crash deep in the pipeline or an `ert state 6` at runtime. Follow the
  recipe exactly rather than bisecting it.
- The flags encode facts about npu4, not user preferences. A second NPU target
  would need its own combination, and this document would have to fork with it.
  Moving these decisions behind a per-target capability query is tracked as
  future work, not done here.
