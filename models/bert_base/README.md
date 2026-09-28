# BERT-base (ONNX) end-to-end on npu4

`bert-base-uncased` — 12-layer, hidden=768, 12-head BERT encoder, input
`[1,32]` token ids -> `[1,32,768]` last_hidden_state. Same recipe as
`models/bert_tiny`, scaled up to the real 110M-parameter checkpoint. See
`models/bert_tiny/README.md` and `docs/2026-08-19_bert_tiny_e2e.md` first —
this file only calls out what's different at this size.

No tokenizer: random `input_ids` in `[0, vocab_size)`, same rationale as
bert_tiny (this tests the compute graph, not language understanding).
Unlike bert-tiny, `bert-base-uncased` *does* ship a real tokenizer, and §5
below builds an actual sentence-in/word-out demo on top of it.

## 1. Get the model

Not committed (416 MB onnx). Same pattern as bert_tiny:

```bash
source /opt/venv/bin/activate
pip install --target=/tmp/bert_deps transformers
PYTHONPATH=/tmp/bert_deps python3 models/bert_base/export_bert_base.py
```

## 2. Compile and run

Identical flags to `models/bert_tiny/README.md` §3 (same function name,
`main_graph`). Nothing model-size-specific was needed.

Measured on npu4: import_onnx ~6s (871 MB intermediate .mlir — regenerate,
don't commit), compile ~8 min, run ~20s, output `[1,32,768]` correlates
**0.99998** with a torch reference (re-measured 2026-08-20 after the
row-overflow fix below; matches bert-tiny's correlation almost exactly).

**Originally measured at 0.99252** (max abs diff 1.37), before
`docs/2026-08-20_batch_matmul_row_overflow_fix.md`'s fix existed. At the
time this was attributed to bf16-demotion rounding compounding through 12
transformer layers instead of 2 — plausible on its own, but re-measuring
after the fix shows most of that gap actually came from bert-base's
batch=12 attention matmuls running *unpadded* (M left as a single untiled
block, per that doc), not just depth-related bf16 accumulation. The fix
doesn't touch bf16 rounding at all, only which physical cores/tiles a
batch matmul's M dimension gets spread across — so the ~750x improvement
in error (1.37 -> ~0.04 max abs diff) means the untiled-M path was
itself less numerically accurate, not merely a performance/placement
concern. Filed away as an open question, not further investigated: *why*
would untiled-vs-tiled M change the result at all for a mathematically
exact operation like matmul? Bf16 accumulation order shouldn't be
sensitive to how work is spatially distributed across cores in a way that
would explain a 30x correlation-gap improvement, so there may have been a
second, subtler numerical issue in the unpadded-M path beyond "just"
placement -- not root-caused.

## 3. What this confirms beyond bert-tiny

bert-tiny's attention batch dimension is 2 (2 heads); this checkpoint's is
12. The row-overflow tile-allocator bug documented in
`docs/2026-08-19_bert_tiny_e2e.md` §5 (small `seq_len` -> too many N-tiles in
one column -> invalid row) was only confirmed at batch=2. `seq_len=32` (this
export's setting) compiled and ran cleanly at batch=12 too, so whatever this
bug's exact row/column math is, it does not scale with batch/head count at
this seq_len — batch count was the untested variable going into this run.
**Update 2026-08-20:** the row-overflow bug is now fixed at the source (see
`docs/2026-08-20_batch_matmul_row_overflow_fix.md`) — `seq_len=16` was
re-tested on bert-tiny after the fix and works. Not re-verified on bert-base
specifically (would need an ~8 min recompile), but the fix is in
`AMDAIEInsertCores.cpp`, which every dispatch in this pipeline goes through
regardless of batch/head count, so there's no reason to expect bert-base to
behave differently.

## 4. Known limitations

Same list as `models/bert_tiny/README.md` §5, plus: batch sizes/attention
head counts other than 2 and 12 are untested.

## 5. Sentence-in, word-out demo (masked language modeling)

`export_bert_base.py` only outputs a raw hidden-state tensor. This section
adds the actual pretrained MLM head on top, so you can type a sentence with
a `[MASK]` and get a real predicted word back.

```bash
source /opt/venv/bin/activate
pip install --target=/tmp/bert_deps transformers
PYTHONPATH=/tmp/bert_deps python3 models/bert_base/export_bert_mlm_trunk.py

python3 -m iree.compiler.tools.import_onnx models/bert_base/bert_mlm_trunk.onnx \
  -o /tmp/bert_mlm_trunk.mlir
build/tools/iree-compile /tmp/bert_mlm_trunk.mlir -o models/bert_base/bert_mlm_trunk.vmfb \
  --iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=/workspace/llvm-aie \
  --iree-amdaie-demote-contraction-inputs-to-bf16 --iree-amdaie-enable-vectorization-passes=false \
  --iree-dispatch-creation-no-fuse-into-contraction-conv-roots \
  --iree-flow-enable-executable-deduplication=false

./scripts/lock/with-npu-lock.sh env PYTHONPATH=/tmp/bert_deps \
  python3 models/bert_base/demo_mlm.py "The capital of France is [MASK]."
```

`demo_mlm.py` runs the trunk on the shared host's one physical NPU, so it's
wrapped in `scripts/lock/with-npu-lock.sh` — see
`docs/2026-07-06_env_setup/DEV_CONTAINER.md` §5 for why. The `export`/`import_onnx`/
`iree-compile` steps above don't touch the NPU and don't need this.

```
Input:  The capital of France is [MASK].
Top-5 predictions for [MASK]:
  1. paris
  2. lyon
  3. marseille
  4. toulouse
  5. lille
Output: The capital of France is paris.
```

**Why "trunk" and not the full model:** `export_bert_mlm.py` exports the
*complete* MLM head, including the final `[768 x vocab_size]`
vocab-projection matmul (`vocab_size=30522` padded to 30528, the next
multiple of 8 — amd-aie requires M/N/K divisible by its 8x8x8 instruction
size, so the unpadded size fails to compile with a clear diagnostic, same
class of check as the `seq_len=17/20` repros in
`docs/2026-08-19_bert_tiny_e2e.md`). That full model **compiles successfully**
but **times out at runtime**: `iree-run-module` fails with
```
INTERNAL; amdxdna dispatch did not complete: ert state 8
```
`ert state 8` is XRT's `ERT_CMD_STATE_TIMEOUT` — the same class of problem as
`models/vgg16/README.md`'s documented "ert state 6" 2-second NPU dispatch
watchdog, just a different resulting state code. The `[32,768]x[768,30528]`
projection is by far the largest single matmul tried in this repo, and with
vectorization off (bf16 has no aievec path, same reason as everywhere else
in this recipe) it runs as scalar AIE code, apparently slow enough to miss
the default 2s TDR.

**The timeout was raised and retried — the full model runs, but is broken in
a worse way.** The fix `models/vgg16/README.md` documents uses a stale
parameter name for this driver (`timeout_in_sec` doesn't exist here and is
silently ignored by `modprobe`); the real parameter, confirmed via
`modinfo amdxdna`, is `tdr_timeout_ms` (milliseconds, `0` = disable):
```
echo "options amdxdna tdr_timeout_ms=60000" | sudo tee /etc/modprobe.d/amdxdna.conf
sudo modprobe -r amdxdna && sudo modprobe amdxdna
```
With that in place, `bert_mlm.vmfb` (the full model, decoder matmul included)
ran without timing out — but produced a **different wrong answer on every
run of the identical model against the identical input**. Two consecutive
runs of the exact same `.vmfb`/input pair differed by up to 21.5 in raw logit
values, with completely different top-5 words each time (`strasbourg/moines/
temps/salle/monde` on one run, `alexia/expressions/hottest/animated/barked`
on another — neither is `paris`). This is not a precision issue: simulating
bf16 rounding for that exact matmul on the host, using the real
`transformed` tensor from the NPU, reproduces `paris` as the clear top
prediction (logit margin ~2.2 over the runner-up, untouched by bf16-scale
noise). It's a genuine non-determinism/correctness bug specific to this
matmul's scale — `N=30528` is roughly 10x larger than anything else compiled
in this repo (the next-largest is the FFN's `N=3072`), so this is very
plausibly a buffer-synchronization or DMA-descriptor issue that only
manifests once a dispatch needs enough tiles/BDs to exceed some resource
this codebase has never exercised before. Not root-caused further.

**Conclusion: use `bert_mlm_trunk.vmfb` (via `demo_mlm.py`), not
`bert_mlm.vmfb`.** The trunk approach stops the compiled graph at
`transformed`, the `[1,32,768]` output of `BertPredictionHeadTransform`
(`Linear(768,768) -> GELU -> LayerNorm`), which contains nothing bigger than
the FFN's already-reliable `768x3072` matmuls, and does the final
`transformed @ decoder_weight.T + decoder_bias` projection in plain numpy on
the host — trivial cost for a CPU BLAS call, and it's the version that
actually gives correct, reproducible answers. `bert_mlm.onnx`/`.vmfb` (the
full, compiles-and-runs-but-wrong version) are left in place as a known-bad
reference for whoever wants to root-cause the non-determinism, not as
something to build a demo on.

## 6. int8 12-layer encoder with softmax / LayerNorm / GELU on the NPU

§2 runs bert-base in bf16 with only the matmuls on the NPU. This section builds
the **int8** encoder that also puts softmax, LayerNorm and GELU on the NPU:
134 dispatches, **131 on the NPU / 3 on the CPU**.

### 6.1 Build the model

```bash
PYTHONPATH=<deps with onnx, onnxruntime, ml_dtypes> \
  models/bert_base/prepare_encoder12_i8.sh models/bert_base/bert_base.onnx models/bert_base/out
```

It chains six steps, each also usable alone:

| Step | Script | What it does |
|---|---|---|
| 1 | `quantize_bert_base.py` | per-tensor int8 QDQ on the 96 MatMuls (fixed-seed calibration) |
| 2 | `fold_bias_into_k.py --pad-rank3` | folds the Q/K/V biases into K, so each projection is a 2-input matmul |
| 3 | `extract_encoder_i8.py` | keeps only the encoder: int8 `[1,32,768]` in, f32 `[1,32,768]` out |
| 4 | `fold_bias_into_k_general.py --pad-value auto --max-bias-lsb 64` | folds every other bias into K |
| 5 | `head_split.py --k` | per-head Q/K/V weights; K's transpose is left on int8 so IREE folds it into QK^T |
| 6 | `hoist_kfold_pad.py` | residuals read a slice of the widened activation, so each widening is written in place |

Why the folds: a bias makes a matmul dispatch a 3-input one, and a third input
forces a DMA pattern that hangs on hardware. Folding it into K keeps two inputs.
The extra activation columns all carry one value v (it has to be one `Pad`);
`--pad-value auto` picks the smallest v whose columns fit, which keeps the bias
rounding under 0.003 requantization LSB. `--max-bias-lsb 64` lets layer 10's FC2
fold: on real activations it changes 2 of 122,880 elements, both towards the fp32
value (docs/2026-09-28_all_biases_folded_and_ukernel_int8_wrap.md).

Why head split: it removes the per-head transposes that would otherwise be CPU
dispatches (36 of them). Why step 6: without it the widening is a copy dispatch
(35 of them); with it and this branch's IREE, none remain.

Checked 2026-09-28 by running the script from `bert_base.onnx` with
onnxruntime 1.29.0: the final model's ORT output is byte-identical to the one
the docs report on.

### 6.2 Compile and run

Needs `third_party/iree` at this branch's pointer and
`patches/third_party/apply.sh` applied (see
`docs/2026-09-28_bert_onnx_submodule_and_patches.md`).

```bash
python3 -m iree.compiler.tools.import_onnx models/bert_base/out/enc12x_k_i8.onnx -o enc12.mlir
iree-compile enc12.mlir -o enc12.vmfb \
  --iree-hal-target-device=npu=amdxdna --iree-hal-target-device=cpu=local \
  --iree-hal-local-target-device-backends=llvm-cpu --iree-hal-default-device=npu \
  --iree-amdaie-target-device=npu4 --iree-amd-aie-peano-install-dir=<llvm-aie> \
  --iree-flow-enable-executable-deduplication=false \
  --iree-amd-aie-enable-chess-for-ukernel=false \
  --iree-amdaie-enable-ukernels=softmax,layernorm
iree-run-module --device=amdxdna --device=local-task --module=enc12.vmfb \
  --function=encoder12 --input=@<int8 [1,32,768] .npy>
```

What stays on the CPU (3): the last LayerNorm (2 dispatches; its output is f32
and the kernel only produces int8) and layer 0's Q/K/V widening (1; its producer
is the model input).

## 7. Whole BERT: input_ids -> last_hidden_state (embeddings on the CPU)

```bash
models/bert_base/prepare_bert_i8.sh models/bert_base/bert_base.onnx models/bert_base/out
python3 -m iree.compiler.tools.import_onnx models/bert_base/out/bertx_int8.onnx -o bert.mlir
iree-compile bert.mlir -o bert.vmfb <same flags as 6.2>
iree-run-module --device=amdxdna --device=local-task --module=bert.vmfb \
  --function=main_graph --input=@<input_ids int64 [1,32] .npy>
```

The same steps as §6 on the whole model, plus `fold_shape_consts.py`: it pins the
batch to 1 and folds the position/token-type id and attention-mask subgraphs
(they depend on the input's shape only) into constants, without which the other
tools cannot see static shapes. 136 dispatches, **131 on the NPU / 5 on the CPU**
(the embedding lookup-and-add, and the embedding and last LayerNorms, whose input
resp. output is f32). No widening copies remain: layer 0's producer is now the
embedding LayerNorm.

Accuracy (2026-09-28): the NPU output matches the same int8 model run on
onnxruntime to the same distance from fp32 BERT -- 0.92 / 0.92 / 0.90 on random
token ids. **On real sentences the int8 model itself reaches only 0.33-0.50**
against fp32 (NPU and onnxruntime alike): step 1 calibrates on random token ids,
not on text. Recalibrating on real sentences is the next step; nothing in the
compiler needs to change for it.
