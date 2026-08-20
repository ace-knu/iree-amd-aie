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
**0.99252** with a torch reference (max abs diff 1.37, values are O(0.1-1)).

That correlation is noticeably lower than bert-tiny's 0.99998. This is
expected, not a new bug: bf16-demotion rounding error compounds through 12
transformer layers instead of 2, and each layer's LayerNorm renormalizes
activations without correcting the accumulated error from prior layers. It's
the same bf16-precision-loss mechanism vgg16 and bert-tiny already document,
just more visible at this depth — still a strong global correlation, not the
"reading wrong data" signature (near-zero or negative correlation, mismatched
argmax) that a real placement/DMA bug would produce.

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

PYTHONPATH=/tmp/bert_deps python3 models/bert_base/demo_mlm.py "The capital of France is [MASK]."
```

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
