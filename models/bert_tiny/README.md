# BERT-tiny (ONNX) end-to-end on npu4

`prajjwal1/bert-tiny` — 2-layer, hidden=128, 2-head BERT encoder, input
`[1,32]` token ids -> `[1,32,128]` last_hidden_state — compiled and run
heterogeneously: every matmul (Q/K/V/output/FFN projections, the QK^T and
Attn@V attention matmuls) on the NPU, everything else (embeddings, LayerNorm,
Softmax, GELU, reshapes/transposes) on the CPU. Same "matmul on NPU, rest on
CPU" pattern as `models/vgg16`, this time for a transformer instead of a CNN.

No tokenizer is used or needed: `input_ids` are random integers in
`[0, vocab_size)`, the same way vgg16's recipe feeds a random image. This
tests the compute graph, not language understanding.

## 1. Get the model

Not committed (17 MB, regenerated from a HF Hub download rather than checked
in — see `.gitignore`). Inside the dev container:

```bash
source /opt/venv/bin/activate
pip install --target=/tmp/bert_deps transformers   # not in the base venv
PYTHONPATH=/tmp/bert_deps python3 models/bert_tiny/export_bert_tiny.py
```

This downloads `prajjwal1/bert-tiny` from the HF Hub and writes
`models/bert_tiny/bert_tiny.onnx`, `input_ids.npy`, and
`ref_last_hidden_state.npy` (a torch reference for the correctness check
below).

## 2. Host prerequisites

Same as `models/vgg16`: raise the NPU dispatch timeout and pass
`--ulimit memlock=-1` (see `USER_GUIDE.md` 1-3). BERT-tiny is far smaller than
VGG16, so in practice these matter less here, but nothing below assumes a
default 2s TDR or 8MB memlock cap.

## 3. Compile and run

From `/workspace` inside the dev container:

```bash
source /opt/venv/bin/activate
export PYTHONPATH=/workspace/build/compiler/bindings/python

python3 -m iree.compiler.tools.import_onnx models/bert_tiny/bert_tiny.onnx -o /tmp/bert_tiny.mlir

build/tools/iree-compile /tmp/bert_tiny.mlir -o /tmp/bert_tiny.vmfb \
  --iree-hal-target-device=npu=amdxdna \
  --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 \
  --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-amdaie-demote-contraction-inputs-to-bf16 \
  --iree-amdaie-enable-vectorization-passes=false \
  --iree-dispatch-creation-no-fuse-into-contraction-conv-roots \
  --iree-flow-enable-executable-deduplication=false

build/tools/iree-run-module --device=amdxdna --device=local-task \
  --module=/tmp/bert_tiny.vmfb --function=main_graph \
  --input=@models/bert_tiny/input_ids.npy --output=@models/bert_tiny/out.npy
```

The entry function name comes from the traced `torch.onnx.export` graph and is
always `main_graph` for this export (unlike vgg16, which keeps the ONNX
graph's original name).

Measured on npu4: compile ~15s, run <1s, output `[1,32,128]` correlates
0.99998 with a torch reference (see 3b) — the same bf16-demotion precision
loss as vgg16, not a correctness bug.

Note what's *not* here compared to vgg16: no im2col, no channels-last
preprocessing, no `--iree-global-opt-detach-elementwise-through-reshape`.
Those three exist specifically to get convolutions into matmul form; BERT has
no convs; its matmuls (`Gemm`/`MatMul` ops, including the batched
attention ones) already hit the plain-matmul path directly.

## 3b. Check the result

```bash
python3 -c "
import numpy as np
gold = np.load('models/bert_tiny/ref_last_hidden_state.npy')
got = np.load('models/bert_tiny/out.npy')
print('corr %.5f' % np.corrcoef(gold.ravel(), got.ravel())[0,1])
print('max abs diff', np.abs(gold - got).max())
"
```

Expected: `corr` at or above ~0.9999. As with vgg16, bf16 demotion means exact
equality is the wrong thing to check.

## 4. Why each flag

Same rationale as `models/vgg16/README.md` §4 for the flags they share
(device declaration, target device, peano dir, bf16 demotion, vectorization
off, no-fuse-into-contraction-roots, executable dedup off — all apply
identically here since they're about the NPU/host split and matmul codegen,
not about convs). The three vgg16 flags that are conv/im2col-specific
(`--iree-global-opt-use-im2col-for-convs`, the channels-last preprocessing
pipeline, `--iree-global-opt-detach-elementwise-through-reshape`) are dropped
here since BERT has no convolutions.

Device placement needed no BERT-specific work: `AMDAIEAssignDeviceAffinities`
routes a `flow.executable` to the NPU if it contains any
`linalg::isaContractionOpInterface` op (batched or not) or a
multiply-accumulate conv, and to the host otherwise — this already covers
embeddings (Gather), LayerNorm, Softmax, and GELU without the pooling-style
special-casing vgg16 needed.

## 5. Known limitations

- **`seq_len` must currently be >= 32** (`SEQ_LEN` in `export_bert_tiny.py`).
  `seq_len=16` reliably crashes `iree-compile` with:
  ```
  [AIE ERROR] _XAie2p_GetTTypefromLoc():61: Cannot find Tile Type
  Assertion `tt < XAIEGBL_TILE_TYPE_MAX && "expected valid tile type"' failed.
  ```
  Root-caused (2026-08-19) by bisecting down to a minimal two-line ONNX repro:
  compiling `linalg.batch_matmul` with shape `M=16,K=16..32,N=64` (batch=2)
  crashes in `AMDAIEDeviceModel::getTileType`, called from
  `AMDAIEGenerateControlOverlayPass` while walking tile ops assigned earlier
  by `AMDAIEAssignTiles`; the same shape with `M=32` compiles and runs
  correctly. This is exactly BERT's Attn@V matmul shape (`M=N=seq_len` for
  the attention scores, contracted against `V`), so it hits every
  attention-using transformer, not just this model.

  Confirmed cause, from comparing the two dispatches' IR after
  `AMDAIEFlattenLogicalObjectFifo` (see `docs/2026-08-19_bert_tiny_e2e.md`
  §5): with `N=64` the packing config splits N into 8 tiles (`N/8`), and the
  tile allocator places all 8 results as 8 *rows* within a single column
  (`amdaie.tile(%c0, %c2)` through `%c9`) instead of spilling into additional
  columns once rows run out — but npu4's target config caps `num_rows` at 4,
  so rows 6-9 don't exist and `getTileType` aborts. The `N=16` case (1 N-tile)
  never needs more than 3 rows, so it never hits the limit. Frame
  identification (`AMDAIEGenerateControlOverlayPass::runOnOperation` →
  `generateControlOverlay` → `getTileType`) was done by hand: this build has
  no working `llvm-symbolizer`/`addr2line` path for `iree-compile`'s crash
  handler, so the crashing process's own `/proc/<pid>/maps` was used to get
  the ASLR load base, subtracted from the printed addresses, and matched
  against a sorted `nm -C build/lib/libIREECompiler.so` dump by nearest
  preceding symbol.

  **Fixed 2026-08-20, at both layers** — see
  `docs/2026-08-20_batch_matmul_row_overflow_fix.md` for the full root-cause
  chain and both fixes. `AMDAIEPadContractionDispatches` now pads
  `linalg::BatchMatmulOp`'s M/K the same way it already padded plain
  `linalg::MatmulOp` (the actual upstream cause — batch matmuls previously
  never got this padding, which is why M ended up untiled downstream).
  `AMDAIEInsertCores.cpp` also spills any excess row-iterations into new
  columns (`col += row / numRows; row %= numRows`) as a general safety net,
  kept in place alongside the padding fix. `seq_len=16` now compiles and
  runs correctly (corr 0.99998, re-verified after both fixes).
  `export_bert_tiny.py` still defaults to `SEQ_LEN=32` from before the fix;
  either value works now.
- Batch size fixed at 1 by the ONNX export; other values are untested.
- Inherits every vgg16 limitation that isn't conv-specific: no f32 vector
  path (bf16 demotion required), flags are not a stable interface, an
  omitted flag fails as an unlabeled crash rather than a diagnostic.
