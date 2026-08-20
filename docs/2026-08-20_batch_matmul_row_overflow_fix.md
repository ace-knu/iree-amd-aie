# Fixing the batch-matmul row-overflow crash (2026-08-20)

Follow-up to `docs/2026-08-19_bert_tiny_e2e.md` §5-6, which root-caused (but
didn't fix) the `seq_len=16` crash: a batched matmul whose N dimension needs
more physical AIE core tiles than the device has rows would abort in
`AMDAIEDeviceModel::getTileType` instead of compiling. This document covers
finding the actual upstream cause and the fix.

## Recap: where the previous session left off

- Crash: `AMDAIEGenerateControlOverlayPass` calling `getTileType(col, row)`
  on an invalid `row` (e.g. row 9 on a device with 4 core rows).
- Immediate cause: `AMDAIE::CoreOp::build` (`AMDAIEOps.cpp:107-116`) computes
  `row = rowOffset + coreRow` with a plain, unconditional `arith.addi` — no
  bound check against the device's actual row count, no spill into another
  column.
- `coreRow` comes from `AMDAIEInsertCores.cpp:139-144`, which passes the
  `scf.forall`'s Y-mapped induction variable straight through.

That much was known. What wasn't known: *why* the forall ends up with more
than 4 iterations on that axis in the first place for a batched matmul, when
the identical (M, K, N) shape without a batch dimension works fine.

## Finding the actual divergence

Added a temporary debug print to `KernelDispatch.cpp`'s
`ParameterSetting::create` (removed again afterward) and rebuilt
`iree-compile` to compare, for the identical `M=16, K=16, N=64` shape:

```
2D matmul   (works):    M=32 N=64 K=32  M0=32  m0Pack=8   -- (32/4) % 8 == 0
batch matmul (crashes): M=16 N=64 K=16  M0=16  m0Pack=16  -- (16/4) % 8 == 4
```

The M-tiling decision is this ternary (`KernelDispatch.cpp:296`):
```cpp
uint32_t m0Pack = (M0 / numRows) % m1Pack == 0 ? (M0 / numRows) : M0;
```
For the 2D case, `M0` had already been padded from 16 to 32 by an earlier
pass, which happens to make `32/4 == m1Pack(8)` exactly, so the modulo is
zero and M gets tiled. For the batch case, no such padding happened, so
`16/4=4`, `4 % 8 = 4 != 0`, and M is left whole (1 tile) — which is what
forces all 8 of N's tiles onto the row axis instead of splitting across both
axes.

Confirmed with `--mlir-print-ir-before=iree-amdaie-lowering-strategy`: the
2D dispatch's `linalg.matmul` operands really are `tensor<32x32xbf16>` /
`tensor<32x64xbf16>` by that point (padded); the batch dispatch's
`linalg.batch_matmul` operands are still `tensor<2x16x16xbf16>` /
`tensor<2x16x64xbf16>` (unpadded).

**The padding pass — `AMDAIEPadContractionDispatches.cpp:176` — only walks
`linalg::MatmulOp`:**
```cpp
SmallVector<linalg::MatmulOp> matmuls;
func.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
```
`linalg::BatchMatmulOp` never matches this walk. There's no batch-aware
branch that skips padding on purpose — the pass simply never sees batched
contractions. This lines up exactly with its own code comment ("They were
validated on exactly one dispatch: VGG-16's dense0") — dense0 is a plain
`linalg.matmul`.

So the full chain, three independent gaps in a row, each one silently
assuming "this is a plain 2D matmul":
1. `AMDAIEPadContractionDispatches` only pads `linalg::MatmulOp` → batch
   matmuls never get the M-padding that (as a side effect) keeps the later
   tiling formula's modulo check happy.
2. `KernelDispatch.cpp`'s `m0Pack` ternary depends on that padding having
   happened; without it, M stays untiled for shapes where the unpadded M
   doesn't already divide evenly.
3. `AMDAIEInsertCores.cpp` / `CoreOp::build` never bound-checks the resulting
   row index against the device's actual row count, so an untiled M forcing
   all of N onto one axis produces an out-of-range tile instead of a
   diagnostic.

## Two fixes, at two layers

This ended up as two separate changes, done in this order:

### 1. Defensive fix at the placement layer (`AMDAIEInsertCores.cpp`)

First attempt: rather than teaching `AMDAIEPadContractionDispatches` about
batch matmuls, or touching the shared `ParameterSetting::create` heuristic
used by every matmul in this pipeline (both higher regression risk), patch
the layer where the crash actually originates: `AMDAIEInsertCores.cpp`,
right before the `CoreOp` gets created:

```cpp
Value col = threadX;
Value row = threadY;
std::optional<int64_t> maybeNumRows = getConfigNumRows(targetAttr);
if (maybeNumRows && *maybeNumRows > 0) {
  auto numRowsVal = rewriter.create<arith::ConstantIndexOp>(
      rewriter.getUnknownLoc(), *maybeNumRows);
  Value colOffset = rewriter.create<arith::DivUIOp>(
      rewriter.getUnknownLoc(), threadY, numRowsVal);
  row = rewriter.create<arith::RemUIOp>(rewriter.getUnknownLoc(), threadY,
                                        numRowsVal);
  col = rewriter.create<arith::AddIOp>(rewriter.getUnknownLoc(), threadX,
                                       colOffset);
}
```

Instead of `row = threadY` unconditionally, this spills excess row-iterations
into new columns: `col = threadX + threadY / numRows`, `row = threadY %
numRows`. `getConfigNumRows`/`getConfigNumColumns` (`Utils/AMDAIEUtils.h`)
already existed and are used elsewhere in this same file's neighborhood
(`AMDAIEPadContractionDispatches.cpp`), so no new device-query plumbing was
needed.

**Why this is safe for every case that already worked:** when `threadY <
numRows` (true for every dispatch that compiled before this change,
including all of VGG16 and every other BERT dispatch), `threadY / numRows =
0` and `threadY % numRows = threadY` — identical to the old behavior,
by construction, not by testing. The new arithmetic only does something
different once `threadY` would have been out of range anyway.

This alone fixes the crash: it compiles and runs `mm3d_b`/`bert-tiny
seq_len=16` correctly (see below). But it's a fix at the *symptom* layer —
it makes an out-of-range placement land somewhere valid, without addressing
*why* M was left untiled in the first place. Asked directly, this isn't the
fix that was originally agreed on as "the real one": the loop-nest shape
itself is still wrong for batch_matmul, this just papers over the physical
consequence. Batch matmuls still don't get the parallelism VGG16-style 2D
matmuls get from proper M-tiling.

### 2. Root-layer fix at the padding pass (`AMDAIEPadContractionDispatches.cpp`)

Second pass: give `linalg::BatchMatmulOp` the same M/K padding
`linalg::MatmulOp` already gets, so batch matmuls hit the exact same
"padded enough to tile cleanly" path 2D matmuls do — fixing gap #1 in the
chain above, which is what caused gap #2 (untiled M) in the first place.

Rather than generalizing the existing 2D helpers in place (`MatmulInfo`,
`growMatmulInit`, `growOutputStore`, and the `runOnOperation` loop body all
hardcode rank-2 shapes/positions throughout — `SmallVector<int64_t, 2>
lhsPad(2)`, `RankedTensorType::get({mPad, nPad}, ...)`, M/N/K assumed at
fixed loop positions 0/1/2), added a parallel, batch-specific path that
touches zero lines of the existing 2D code:

- `BatchMatmulInfo` (mirrors `MatmulInfo`, holds `linalg::BatchMatmulOp`)
  and `getBatchMatmulInfo` (mirrors `getMatmulInfo`, walks
  `linalg::BatchMatmulOp` instead).
- `growBatchMatmulInit`/`growBatchOutputStore` (mirror the 2D versions,
  building `[batch, mPad, nPad]` shapes instead of `[mPad, nPad]`).
- A new branch at the top of `runOnOperation`'s dispatch loop: try
  `getBatchMatmulInfo` first; if it matches, pad using loop-dim constants
  shifted by the batch dim (`linalg.batch_matmul`'s iterator order is
  `(batch, M, N, K)` vs `linalg.matmul`'s `(M, N, K)`, so `M=1, N=2, K=3`
  instead of `M=0, N=1, K=2`), with the batch dim carried through every
  padded/cropped shape unchanged (never itself padded); `continue` before
  reaching the existing 2D code path.

Three existing helpers needed **no changes at all** because they were
already rank-agnostic: `createPaddingDispatch`/`createCropDispatch` (build
the host pad/crop dispatches from a generic `ArrayRef<int64_t>` shape) and
`getPaddingMultiples`/`operandDimForLoop` (compute target sizes and dimension
positions without referencing rank). Only the shape-construction and
loop-position-constant parts needed a batch-aware counterpart.

Confirmed working: `mm3d_b`'s dispatch now gets padded from `tensor<2x16x16xbf16>
x tensor<2x16x64xbf16>` to `tensor<2x32x32xbf16> x tensor<2x32x64xbf16>` —
the *same* M: 16→32, K: 16→32 padding the plain-2D case gets — verified via
`--mlir-print-ir-before=iree-amdaie-lowering-strategy`.

**Both fixes were kept.** The padding fix addresses the actual root cause and
restores proper M-tiling (and the parallelism that comes with it) for batch
matmuls specifically. The `AMDAIEInsertCores.cpp` placement-layer fix stays
in as well, as a general safety net: it protects against row overflow for
*any* dispatch that ends up with more forall iterations than physical rows,
for whatever reason — not just this specific padding gap. Since it's a
no-op whenever `threadY < numRows` (true whenever tiling was correct to
begin with), keeping both costs nothing and adds defense in depth.

## Verification

- `mm3d_b.mlir` (the minimal repro, `[2,16,16] x [2,16,64]` batched matmul)
  — compiles under both fixes (previously aborted with only the padding gap
  present; the `AMDAIEInsertCores.cpp` fix alone also made it compile, but
  now the operands are also properly padded/tiled). Run on real npu4
  hardware against a random-input reference: **corr 0.999997**, identical
  accuracy to the placement-only fix (expected — padding changes which
  physical cores get used, not the numerics).
- Regenerated `prajjwal1/bert-tiny` at the *original* `seq_len=16` (the
  shape that started this whole investigation, before the `seq_len=32`
  workaround) — compiles, runs on npu4, **corr 0.99998** against the torch
  reference, matching the placement-only-fix run exactly. `seq_len=32` is no
  longer required.
- Regression-compiled every smaller repro from the previous session
  (`two_bmm`, `mm2d`, `mm2d_n64`, `mm2d_k16`, `qk`, `qkv`) and the full
  `bert_tiny` (`seq_len=32`) model, under both fixes together — all still
  compile cleanly.
- Did not re-run the full `bert_base` (12-layer) compile (~8 min) or the MLM
  head under either fix; the row-overflow mechanism these fixes address is
  independent of what caused `bert_mlm.vmfb`'s separate non-determinism bug
  (`models/bert_base/README.md` §5), so that issue is expected to remain
  open regardless.

## What's still open

- `seq_len=16` no longer crashes, but `models/bert_tiny/export_bert_tiny.py`
  and `models/bert_base/export_bert_base.py` still default to `SEQ_LEN=32`
  from before this fix — harmless to leave, but could be reverted to 16 now
  if a smaller/faster example is wanted.
- The padding fix only covers `AMDAIEPadContractionDispatches`
  (`iree-amdaie-pad-contraction-dispatches`). Its sibling pass in the same
  file, `AMDAIESplitLargeContractionDispatchesPass` (splits a large-(K,N)
  transpose_b matmul into N-chunks), still only handles plain
  `linalg::MatmulOp` and shares a rank-2-hardcoded `createConcatDispatch`
  helper — not touched here since nothing in this session's testing
  exercised it for a batched contraction. If a batched matmul ever needs
  that split (very large N with a batch dim), it would need the same
  treatment.
- Performance of the now-properly-tiled batch matmul path (does it actually
  use the physical grid as efficiently as the 2D path?) was not benchmarked
  — only correctness and successful compilation were verified.
- This code change lives directly in this repo (`compiler/plugins/target/
  AMD-AIE/...` is not a submodule) but has not been committed — it's sitting
  as an uncommitted working-tree change (two files:
  `AMDAIEInsertCores.cpp` and `AMDAIEPadContractionDispatches.cpp`) pending
  the user's decision on whether/when to commit it.
