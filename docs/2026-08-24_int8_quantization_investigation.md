# int8 quantization investigation (in progress, paused)

Goal: get `vector.contract` actually vectorized on npu4 (AIE2P/Strix). aievec's
npu4 lowering (`getSuportedAie2PTypes()` in
`compiler/plugins/target/AMD-AIE/aievec/VectorToAIEVecConversions.cpp:118-126`)
only has an int8×int8→i32 matmul intrinsic — no bf16 entry. (npu1/AIE2/Phoenix
does have both, `getSupportedAie2Types()` lines 107-115 — this is an
npu4-backend-specific gap, not a general aievec limitation.) So the plan is:
quantize a model to int8 instead of chasing bf16 vectorization support.

This picks up after the BERT e2e work (`bert` branch, see
`docs/2026-08-20_batch_matmul_row_overflow_fix.md` and the BERT model READMEs).
Everything below happened in one session and is **not committed** — see
"Current state / what's uncommitted" at the bottom before doing anything else.

## 1. Does int8 vectorization even work at all? — YES, confirmed on real hardware

Built a minimal single `[32,128]x[128,64]` matmul, quantized with
`onnxruntime.quantization.quantize_static` (NOT `quantize_dynamic` — that
function's signature in the installed onnxruntime 1.29.0 has no `quant_format`
param and always emits `QuantizationMode.IntegerOps`/`MatMulInteger`-style
output, not QDQ. `quantize_static` with `quant_format=QuantFormat.QDQ`,
`weight_type`/`activation_type=QInt8`, `extra_options={"ActivationSymmetric":
True}`, and a trivial random-data `CalibrationDataReader` does produce the
expected `DequantizeLinear`/`QuantizeLinear`/`MatMul`/... QDQ node sequence).

Confirmed the **entire compiler chain from ONNX QDQ down to `aievec.matmul` is
already wired up unconditionally, with zero new passes needed**:

1. ONNX `QuantizeLinear`/`DequantizeLinear` → torch-mlir's ONNX importer
   already lowers these (`third_party/iree/third_party/torch-mlir/lib/Conversion/TorchOnnxToTorch/DefaultDomainAtoF.cpp:2339`)
   to `torch.aten._make_per_tensor_quantized_tensor` + `aten.dequantize`.
2. IREE's Torch input pipeline (`createTorchToIREEPipeline`,
   `compiler/plugins/input/Torch/InputConversion/Passes.cpp:60`) unconditionally
   runs torch-mlir's `FuseQuantizedOps` pass
   (`third_party/iree/third_party/torch-mlir/lib/Dialect/Torch/Transforms/FuseQuantizedOps.cpp`),
   fusing the Q/DQ chain into real int8 arithmetic before `aten.mm`/`aten.matmul`.
3. Lowered to linalg this becomes `linalg.quantized_matmul` (4-input: lhs, rhs,
   lhsZp, rhsZp). IREE's GlobalOptimization pipeline unconditionally runs
   `LinalgQuantizedMatmulToMatmulPass`
   (`third_party/iree/compiler/src/iree/compiler/GlobalOptimization/QuantizedMatmulToMatmul.cpp`),
   turning it into a plain int8 `linalg.matmul` (with a zero-point correction
   term, or none at all — the "easy case" — when both zero points are
   compile-time-constant zero, i.e. symmetric quantization).
4. From there it hits the existing, already-working AMD-AIE int8 path
   (`AMDAIEPadContractionDispatches`, `getSuportedAie2PTypes`).

Compiled with the vgg16/bert_tiny flag recipe **minus**
`--iree-amdaie-demote-contraction-inputs-to-bf16` (irrelevant — inputs are
already int8) and **minus** `--iree-amdaie-enable-vectorization-passes=false`
(this flag defaults to `true` — `AIETarget.h:58` — just omit it).

**Verified on real npu4 hardware**: `aievec.matmul %_, %_, %_ : vector<8x8xi8>,
vector<8x8xi8> into vector<8x8xi32>` appears in the `--mlir-print-ir-after-all`
dump (not a scalar fallback), `iree-run-module` exits 0, output correlates
0.9998 with the fp32 reference — with zero calibration tuning effort.

A red herring ruled out along the way: `FuseDequantizationMatmul`
(mentioned in `docs/2026-08-16_frontend_lowering_passes.md`) is NOT relevant
here — it's a weight-only-dequant "group dequant" reassociation that produces
*float* arithmetic (memory-saving, not compute), and sits behind a
default-off flag (`clEnableQuantizedMatmulReassociation`).

Repro scratch (gitignored, `_local/int8_debug/`, not committed):
`gen_and_quant.py` (2D matmul gen+quantize), outputs in `out/mm_int8.*`.

## 2. VGG16 attempt — blocked on a real upstream `third_party/iree` bug

Tried quantizing VGG16's 13 Conv + 3 Gemm layers
(`quant_vgg16.py`, `op_types_to_quantize=["Conv","Gemm"]`, static QDQ,
8 random `[1,3,224,224]` calibration samples). Gemm alone would work fine (see
§1) but Conv fails to compile:

```
error: 'linalg.generic' op operand #2 must be variadic of shaped of any type
values, but got 'i32'
```

**Root cause** (confirmed via a research pass, file:line-cited):
`third_party/iree/compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp:358-359`
(`transposeConvLikeLinalgOp`) hardcodes operand index 2 as the DPS "output":

```cpp
Value input = convOp->getOperand(0);
Value filter = convOp->getOperand(1);
Value output = convOp->getOperand(2);   // <-- hardcoded index
```

This is only valid for the standard 2-input+1-output conv layout. A
**quantized** conv is `linalg::Conv2DNchwFchwQOp` (built by torch-mlir's
`ConvertAtenConvolutionOp`,
`third_party/iree/third_party/torch-mlir/lib/Conversion/TorchToLinalg/Linear.cpp:1239`),
which has 5 operands: `input=0, filter=1, inputZp=2, weightZp=3, output=4`.
So the channels-last pass grabs `inputZp` (a bare scalar `i32`) as "output"
instead of the real output tensor at index 4 — the real output silently
disappears and a malformed `linalg.generic` with a scalar wired as the DPS
output is built (exactly the error above). Channels-last conversion is
mandatory for AIE conv codegen (standard NCHW conv codegen isn't supported on
this backend at all), so there's no flag to route around this.

**This needs a real patch to vendored `third_party/iree`** (not this repo's
own AMD-AIE plugin code) — `getDpsInputOperand()`/`getDpsInits()` instead of
fixed indices, threading zero-point operands through the transpose/pack logic
and `defaultConvBuilderFn`/`namedConvBuilderFn`, likely also a dedicated
`Conv2DNchwFchwQOp`→`Conv2DNhwcHwcfQOp` named-op channels-last pattern
(paralleling the existing non-quantized `ConvertLinalgConvNchwFchw` around
line 451 of the same file). **Not started.** `LinalgQuantizedConvToConv`
(the pass that would eventually turn a valid `linalg.quantized_conv` into a
plain conv) never even gets a chance to run — the IR is already malformed
and rejected by the verifier before reaching that stage.

Given this, the user chose to switch focus to BERT (no conv at all — purely
matmul/batch_matmul) rather than sink time into this upstream conv fix right
now. VGG16 int8 is parked at "Gemm-only would work, Conv is a real
multi-hour third_party/iree patch, not attempted."

## 3. BERT batch matmul (attention QK^T / Attn@V) — found and fixed a second gap

Built a minimal batched matmul repro matching BERT attention's shape: two
**dynamic** (non-constant, non-weight) activations, `x1:[2,16,16]`,
`x2:[2,16,64]`, `y = x1 @ x2` (`gen_and_quant_bmm.py`). Confirmed
`onnxruntime.quantize_static` happily QDQ-quantizes a MatMul between two
dynamic activations (both get their own Q/DQ pairs) — this is realistic for
attention, not just weight-quantization.

First compile attempt crashed in this repo's own
`AMDAIEBufferizeToAllocation.cpp:210` ("expected only one target op, found 2
target ops"). Digging into *why* revealed the real, deeper problem — dumping
IR after `iree-global-opt-quantized-matmul-to-matmul` showed the batched
matmul was being computed in **plain float**, sandwiched between a quantize
and dequantize:  `i8 → dequant(f32) → linalg.batch_matmul (f32!) → quant(i8) →
dequant(f32)`. Quantization was happening, but the actual arithmetic wasn't —
zero vectorization benefit even if it hadn't crashed.

**Root cause, precisely pinned down** (unlike VGG's Conv bug, this one is
*not* an IREE-level gap):

- `LinalgQuantizedMatmulToMatmulPass`
  (`third_party/iree/compiler/src/iree/compiler/GlobalOptimization/QuantizedMatmulToMatmul.cpp:44`)
  already fully supports the batch case —
  `isa<linalg::QuantizedMatmulOp, linalg::QuantizedBatchMatmulOp>(op)`, with
  batch-aware affine maps throughout. **Not the bottleneck.**
- Torch-mlir's `ConvertAtenMatmulOp`
  (`third_party/iree/third_party/torch-mlir/lib/Conversion/TorchToLinalg/Linear.cpp:241`)
  already emits `linalg::QuantizedBatchMatmulOp` for its batch branches
  (lines ~545, ~614). **Also not the bottleneck.**
- The real gap: ONNX's batched `MatMul` (two rank-3 operands) imports to
  **`torch.aten.bmm`**, not `torch.aten.matmul`. Confirmed by dumping IR
  before/after the `torch-fuse-quantized-ops` pass — completely unchanged,
  because:
  - `FuseQuantizedOps.cpp`'s pattern list
    (`third_party/iree/third_party/torch-mlir/lib/Dialect/Torch/Transforms/FuseQuantizedOps.cpp:452-465`)
    had `QuantizeOperandsPastCommutingOps<AtenMatmulOp,2>` and
    `<AtenMmOp,4>` — but nothing for `AtenBmmOp`.
  - `ConvertAtenBmmOp`
    (`.../Conversion/TorchToLinalg/Linear.cpp:696`, original code) never
    called `getZeroPoint` at all and unconditionally built a plain float
    `linalg::BatchMatmulOp`.

### Fix applied (uncommitted — see bottom)

Two edits, both in the vendored `third_party/iree/third_party/torch-mlir`
submodule:

1. `FuseQuantizedOps.cpp` — added
   `QuantizeOperandsPastCommutingOps<AtenBmmOp, 2>,` next to the
   `AtenMatmulOp` entry. (`AtenBmmOp` needs no `QuantInfo` specialization —
   the default `operandsToQuantize = {0, 1}` is already correct for its
   `(self, mat2)` operand pair.)
2. `Linear.cpp`'s `ConvertAtenBmmOp` — added zero-point detection
   (`getZeroPoint(op.getSelf(), lhsZeroPoint)` /
   `getZeroPoint(op.getMat2(), rhsZeroPoint)`, mixed-quantization mismatch
   check) and, when present, the same truncate/`signShift`/
   `linalg::QuantizedBatchMatmulOp`-construction dance
   `ConvertAtenMatmulOp`'s `maxRank==3` branch already does (lines ~538-565
   of the same file) — simpler here since `aten.bmm`'s operands are always
   exactly rank 3 on both sides, no broadcasting/collapsing needed.

Rebuilt `iree-compile` incrementally (only **14 seconds**, ccache warm —
`cmake --build build -j 6 --target iree-compile`).

**Confirmed working at the IR level**: recompiling the bmm repro now produces
a real `linalg.batch_matmul ins(%3, %5 : tensor<2x16x16xi8>,
tensor<2x16x64xi8>) outs(... : tensor<2x16x64xi32>)` (no float sandwich), and
`aievec.matmul` appears 20 times in the full IR dump — it vectorizes.

**But wrong on real hardware — this is the open bug, see §4.**

Repro scratch: `gen_and_quant_bmm.py`, outputs in
`out/bmm_int8.*`, `out/bmm_int8_v2.vmfb` (post-patch compile),
`out/bmm_out*.npy`.

## 4. Update: root-caused and mostly fixed — see §4a for what's still open

**The correctness bug described below was root-caused and fixed.** Root
cause: the §3 patch only added half of what `AtenMatmulOp`/`AtenMmOp` already
had. `FuseQuantizedOps.cpp`'s pattern list also has `QuantizeAccumulator<AtenMmOp>`/
`<AtenMatmulOp>` (line ~464) — a *second* pattern, independent from
`QuantizeOperandsPastCommutingOps`, that handles the **output** side: it
redeclares the op's result as `!torch.qint32` (raw accumulator), wraps it with
`Aten_MakePerTensorQuantizedTensorOp` using `scale = lhsScale * rhsScale`, then
`AtenDequantizeTensorOp`s it back to the original float type — this is where
the actual "multiply the raw i32 accumulator by the combined input scale"
math happens. I had only added `QuantizeOperandsPastCommutingOps<AtenBmmOp,2>`
(input side) and missed `QuantizeAccumulator<AtenBmmOp>` (output side).
Without it, `aten.bmm`'s declared result type stayed plain `f32`, so my
`ConvertAtenBmmOp` patch's `accumulatorDType(i32) != resultElementType(f32)`
branch fired and called `torch_to_linalg::convertTensorToElementType` — which
does a **naive `sitofp` type cast, not a scaled dequantization**
(`third_party/iree/third_party/torch-mlir/lib/Conversion/TorchToLinalg/Utils.cpp:631-642`,
just `convertScalarToDtype`). So the raw int32 accumulator (e.g. `14411`) got
reinterpreted as `14411.0f` with no scale multiply, and the downstream
(unfused, unmodified) `aten.quantize_per_tensor(14411.0, y_scale=0.0015,...)`
naturally saturated to the int8 extremes on nearly every element — exactly
matching the observed `0.19019213`/`-0.19168971` (`≈ ±127/-128 * y_scale`)
pattern.

Confirmed by hand-simulating the *intended* correct math in plain numpy with
the real calibrated scales (`x1_scale=0.0031067322`, `x2_scale=0.0035386747`,
`y_scale=0.0014975759`, all zero-points 0): quantize→int8-batch-matmul→
dequantize gives corr 0.9998 against `bmm_ref.npy`, with **zero** saturation —
proving the scales/calibration were always fine, and the bug was purely in
the missing accumulator-rescaling pattern.

**Fix**: added `QuantizeAccumulator<AtenBmmOp>,` next to
`QuantizeAccumulator<AtenMmOp>, QuantizeAccumulator<AtenMatmulOp>` in
`FuseQuantizedOps.cpp`'s pattern list (line ~464-465). Rebuilt (10s
incremental). Confirmed:
- **CPU-only** (`--iree-hal-target-device=local`, no NPU at all): corr
  **0.9998**.
- **npu4, vectorization OFF**: corr **0.9998**, both batches correct.
- **npu4, vectorization ON** (the actual goal): batch index 1 is now
  correct, but **batch index 0 is still exactly all-zero**. See §4a.

This also means the earlier "compare against bf16" instinct and the initial
"maybe it's an NPU DMA/packing bug" hypothesis were both wrong — the bug was
100% in the torch-mlir patch from §3, reproducible on plain CPU codegen, with
nothing AMD-AIE-specific about it. Good general lesson: when a *newly added*
code path misbehaves identically on a completely different backend (CPU),
suspect the new code/pass itself before suspecting the specialized backend.

## 4a. OPEN, narrower bug: batch index 0 is all-zero, but only with vectorization ON

Shape: batch=2, M=16, K=16, N=64 (padded to M=32,K=32 by the existing
`AMDAIEPadContractionDispatches` batch-matmul padding path from
`docs/2026-08-20_batch_matmul_row_overflow_fix.md` — **confirmed still
working correctly** for int8 via
`--mlir-print-ir-before/after=iree-amdaie-pad-contraction-dispatches`: pads
via `pad_dispatch_0`/`pad_dispatch_1`, crops via `crop_dispatch_2`, same
structure as the bf16 case). This is deliberately the exact shape that
originally triggered the bf16 row-overflow bug — that bf16 case is long since
fixed and verified (corr 0.999997).

**Before the §4 accumulator fix**, the symptom looked like a plausible NPU
DMA/packing bug (all elements saturated at int8 extremes, batch 0 all-zero
only with vectorization on). **After the §4 fix**, most of that turned out
to be downstream noise from the accumulator bug — the "both batches show the
saturation pattern" symptom (vectorization off, pre-fix) is fully explained
by §4's root cause alone and has nothing to do with the NPU or batching.

**What's confirmed genuinely open, post-fix** — the CPU-only sanity check
from the original plan *was run* (see §4, "Confirmed" bullets) and gives a
clean, narrow answer:

| Target | Vectorization | Correctness |
| --- | --- | --- |
| CPU only (`--iree-hal-target-device=local`, no NPU) | n/a | **corr 0.9998**, both batches correct |
| npu4 | off | **corr 0.9998**, both batches correct |
| npu4 | **on** (the actual goal) | batch 1 correct; **batch 0 exactly all-zero** |

So this residual bug is real, narrow, and **specific to
`iree-amdaie-vectorization`/`aievec` lowering interacting with the batch
dimension** — not the packing/DMA/tiling machinery shared with the
non-vectorized path (that's proven correct by the vectorization-off row
above), and not a general aievec-batch problem either, since `aievec.matmul`
itself has no notion of "batch" (it's a fixed single-tile 8×8×8 int8 GEMM
intrinsic called repeatedly by the surrounding loop nest regardless of batch,
M-tile, or N-tile index — and that repeated-calling machinery already works
correctly for the non-batched 2D case in §1, which also needs multiple tile
iterations). The bug is most likely in whatever code decides *how many times*
and *with what tile/core assignment* to iterate over the **batch** dimension
specifically when vectorization is on — something that doesn't get exercised
at all by the already-working non-batched-vectorized (§1) or
batched-non-vectorized (this table's middle row) cases individually. This is
the intersection of "batch" and "vectorized" that's never been exercised
before this session.

A forked deep-dive (pass-by-pass IR bisection, mirroring the original
row-overflow investigation's methodology) was started but interrupted/killed
by the user in favor of first reasoning about the likely cause more
efficiently — that reasoning is what produced the table above.

**Progress so far, going deeper (same session, continued):**

1. **Diffed IR around `iree-amdaie-vectorization` directly** (before/after,
   for the dispatch function and for the outlined per-tile compute function
   `generic_matmul_0_outlined` separately). Finding: the **outer batch loop
   (`scf.forall (%arg0,...) in (2,1,1)`) and every surrounding DMA copy are
   byte-identical** whether vectorization is on or off — vectorization only
   changes what's *inside* `generic_matmul_0_outlined` (`linalg.fill` →
   `vector.transfer_write`, `linalg.generic` → `vector.contract`). The output
   indexing map on the vectorized `vector.contract` has two dims transposed
   relative to the original `linalg.generic`'s output map, but every affected
   dim has extent 1 at this point (already sliced down by the caller's
   subview), so this looks like a harmless relabeling, not a real bug.
2. **Diffed the control-code stage** (`iree-amdaie-controlcode-loop-unroll`,
   which lowers to the actual DMA/lock/wait instruction sequence sent to the
   NPU) between vectorized and non-vectorized compiles: op-count diff showed
   **exactly one difference, an attribute string** (`amdaie.packing_config`
   present in one, absent in the other) — the actual DMA/sync/lock schedule
   is identical. This rules out a lock-count or DMA-scheduling mismatch at
   the MLIR level.
3. **Went one level below MLIR: dumped and disassembled the actual AIE core
   machine code** peano generates, via `--iree-hal-dump-executable-intermediates-to=<dir>`
   (a generic IREE flag; the AMD-AIE backend's `serializeExecutable` in
   `AIETarget.cpp` honors `serOptions.dumpIntermediatesPath`) → `input.ll`
   (pre-opt LLVM IR), `input.opt.ll` (post-peano-opt), per-core `.elf`, and
   `.stacksizes`; disassembled with `/workspace/llvm-aie/bin/llvm-objdump -d`.
   Artifacts saved at `_local/int8_debug/out/intermediates{,_novec}/` and
   `_local/int8_debug/out/core_7_5_{vec,novec}.dis`.

**Important correction to the mental model from earlier**: the batch loop is
fully unrolled by the time it reaches the core program — **each physical core
runs BOTH batch iterations sequentially inside ONE program invocation**, not
two separate host-triggered runs. `core_7_5`'s disassembled body is two
near-identical blocks (acquire lock → zero-fill accumulator → acquire input
locks → matmul-accumulate loop → store result → release 3 locks, repeated for
batch 1 with different buffer addresses: batch 0 uses `0x70400`/`0x70500`/
`0x78000`, batch 1 uses `0x74000`/`0x74100`/`0x7c000`, non-overlapping,
consistent double-buffer stride).

**Hypotheses tested directly against the disassembly and ruled out:**
- Dead-store elimination removing batch 0's result store — not present; the
  `vst` to the accumulator and the 3 `rel` (lock release) calls appear in the
  correct order as separate, non-elided instructions.
- Acquire/release lock sequence divergence between vec/novec — identical.
- Buffer address collision between batch 0 and batch 1 — no overlap.
- Stack overflow/spill corrupting a buffer (vectorized code using more
  stack) — refuted by `.stacksizes`: vectorized `core_7_5`/`main` use **zero**
  bytes of stack; non-vectorized uses 64.
- Missing pipeline-latency NOPs between the last `vmac` and the store — NOP
  bundles are present.

One structural difference WAS found — vectorization causes
`generic_matmul_0_outlined` to get **inlined** into the caller (non-vectorized
keeps it as an opaque `call`, a natural optimization barrier) — but the two
batch blocks remain structurally isomorphic even inlined, so this alone
doesn't explain a batch-0-vs-1 asymmetry.

**Where this stands now**: no code-level asymmetry between the batch 0 and
batch 1 blocks was found anywhere from MLIR down through disassembly — they
are structurally identical, differing only in buffer addresses (which don't
overlap). The remaining, unconfirmed hypothesis is a genuine **runtime
hazard** — e.g. batch 0's output DMA racing the core's completion/lock-release
signal — which is invisible to static IR/disassembly reading and would need
actual hardware trace/timing tooling to confirm.

**Cheap follow-up test that DOES support the runtime-hazard theory**: reran
the identical repro at **batch=4** instead of batch=2 (`gen_and_quant_bmm_b4.py`,
same M=16,K=16,N=64 shape, quantized/compiled/run the same way, vectorization
on). Result: **only batch index 0 is wrong (all-zero); batches 1, 2, and 3
are all correct** (corr 0.9998 each). This generalizes the symptom from
"batch=2 specifically" to "the first of N sequential batch iterations is
always wrong, regardless of N" — consistent with something not being
properly warmed up/synchronized on the very first pass through the per-core
loop (e.g. a lock/semaphore initial-state issue, or a pipeline that needs one
iteration to fill before results are safe to trust), and inconsistent with
anything specific to the number 2 or to address offset 0 being special in
some address-computation sense (since batch 0 is still the offset-0 case at
batch=4 too, so this doesn't distinguish those two theories on its own — but
it does rule out "something special about exactly 2 batches").

**Two more targeted experiments tried (same session), both inconclusive/negative:**

1. **`llc -O0` instead of `-O2`** (one-line change to the hardcoded `llcArgs`
   in this repo's own `compiler/plugins/target/AMD-AIE/iree-amd-aie/Target/XCLBinGen.cpp:1417`,
   also needed `--iree-amdaie-stack-size=4096` since `-O0` codegen blew the
   default 1024-byte stack). **Did not fix batch 0** — still all-zero — and
   batch 1's correlation actually dropped to 0.90 (from 0.9998). Disassembling
   the `-O0` output showed the store→release gap did NOT actually grow (if
   anything it looked tighter, buried in heavy stack-spill code) — so this
   experiment didn't cleanly test the timing-margin hypothesis at all;
   optimization level is not a reliable lever for the store/release gap.
   **Reverted** (not committed).
2. **Explicit 16-NOP inline-asm injection immediately before every lock
   release**, added to this repo's own
   `compiler/plugins/target/AMD-AIE/aie/AMDAIECoreToStandard.cpp`'s
   `lockToStd` (right before the `func::CallOp` to `llvm.<arch>.release`,
   guarded on `useLock.getAction() == LockAction::Release`), via
   `rewriter.create<LLVM::InlineAsmOp>(...)` with a `"nop\n\t"`-repeated
   asm string, `has_side_effects=true` (to survive optimization). Compiled
   this MLIR-level change fine, but **`llc` itself crashed**:
   `LLVM ERROR: unable to translate instruction: call (in function: core_7_5)`
   during GlobalISel's `IRTranslator` pass. Root cause: inline assembly
   lowers to a `call asm sideeffect ...` construct in LLVM IR, and **peano's
   AIE2P backend's GlobalISel implementation does not support translating
   inline assembly at all** — this is a hard limitation of the vendored,
   prebuilt `/workspace/llvm-aie` peano install (binaries only, no source,
   so it can't be patched/rebuilt to add support). **Reverted** (not
   committed) — this avenue is a dead end without a peano/LLVM-AIE source
   checkout, which isn't available in this environment.

**Third experiment — this one worked as intended, and is a real negative
result.** Instead of inline asm (blocked by peano's GlobalISel), injected 16
**volatile** scalar stores (`LLVM::StoreOp` with `isVolatile=true`, storing to
a stack `LLVM::AllocaOp`) immediately before every lock release in the same
`lockToStd` location. Volatile stores can't be eliminated or reordered by the
optimizer (unlike a plain dead store to an unread local, which is what a
naive `scf.for`/`memref.store` loop would have been — also considered and
rejected before trying, since nothing reads the loop's result) and are
ordinary, universally-supported instructions (no GlobalISel gap). Compiled
clean, ran without crashing, and **disassembly confirmed the injection
worked exactly as intended**: 16 real `st r0, [p3, #0]` instructions appear
immediately before each `rel` call (verified in
`_local/int8_debug/out/core_7_5_volstore.dis`).

**Result: batch 0 is still exactly all-zero** (batch 1 still correct, corr
0.9999). This is a much stronger negative result than the previous two
attempts — the store→release timing margin was genuinely, substantially
widened (16+ real cycles guaranteed, not "maybe" like the `-O0` attempt) and
it made no difference. **This is fairly strong evidence that the root cause
is NOT simply insufficient time between the compute's final store and the
lock release** — the write-buffer-drain-time hypothesis, while plausible on
its face, does not survive this test. The bug more likely lives in something
structural about the *very first* use of a particular lock/DMA
channel/buffer-descriptor in the program's execution (e.g. a genuinely
different code path or hardware state on a "cold" first acquire/release of a
given resource, not simply "not enough wait time on every release") — but
this is still speculation; no further experiment has isolated it. Reverted
(not committed).

**Fourth experiment, testing the OTHER side of the hazard.** A key
observation motivated this: batch 0's output isn't garbage, it's *exactly*
all-zero — matching the accumulator/buffer's pre-computation zero-filled
state, not a corrupted nonzero value. That's more consistent with "the core
computed 0×0=0 because its input buffer hadn't been written yet by the
producing DMA" than with "the compute/register-aliasing corrupted a real
result into zero." So the same verified-real volatile-store delay technique
was moved to the **acquire** side instead of release: inject the 16 volatile
stores immediately *after* every `LockAction::Acquire`/`AcquireGreaterEqual`
call in `lockToStd` (i.e. give the core extra guaranteed time after
successfully acquiring an input-ready lock, before it starts reading that
buffer — testing whether the core is racing ahead of its own *input* DMA,
the mirror image of the release-side test). Compiled and ran fine.

**Batch 0 is still exactly all-zero.** So delay on neither side of the
hazard (before signaling "output ready", after confirming "input ready")
changes anything. Reverted (not committed).

**Taken together, all three delay experiments (release-side NOP, release-side
volatile-store, acquire-side volatile-store) failed to change the outcome at
all, despite the second and third being independently verified to inject
real, non-optimizable delay at exactly the intended points.** This is now
fairly strong evidence against any theory of the form "there's insufficient
wait time somewhere in the core's own instruction stream." Two directions
this leaves open: (a) the "exactly zero, not garbage" observation could still
mean the input DMA for batch 0 never completes at all (not just "completes
late") — a different bug than a timing margin, e.g. a BD/DMA-configuration
error specific to the first use of a given channel — which no amount of
delay in the CORE's program could ever fix, since the core isn't what's
broken; or (b) the bug isn't on the DMA/lock/timing axis at all, and lies
somewhere neither disassembly-reading nor delay-injection can reach (e.g. an
actual hardware erratum, or something in the amdxdna runtime/driver's BD
programming that's specific to this exact shape/configuration).

**Fifth angle: decoded the raw NPU DMA transaction format itself.** All four
delay experiments above only touched the AIE *core's* own program
(acquire/release lock timing inside `generic_matmul_0_outlined`/`main`) — none
of them looked at the DMA controller/BD (buffer descriptor) side, which is
where the write-back copy from the core's local result to the real output
buffer actually gets programmed, and where the batch index genuinely is used
as a runtime offset multiplier (`amdaie.dma_cpy_nd(%lof_16[%arg0, 0, 0], ...)`)
— unlike inside the core, where every address was already a resolved
compile-time constant. This is exactly the kind of place a "batch × stride
collapsing to something degenerate at batch=0" bug (a plausible, cheap-to-
suspect class of bug) could live, and it's a completely different mechanism
from anything the delay experiments could have caught or fixed.

Decoded `_local/int8_debug/out/intermediates/bmm_int8_srcasync_dispatch_2_batch_matmul_0.npu_inst.txt`
(the raw hex "TXN" transaction IREE's AMD-AIE backend emits) using struct
layouts from `third_party/aie-rt/driver/src/global/xaiegbl.h`
(`XAie_TxnHeader`, `XAie_Write32Hdr`, `XAie_BlockWrite32Hdr`) and
`third_party/XRT`'s documented `patch_op_t` fields. The decode is
**validated**: walking all 63 ops lands exactly on the declared `TxnSize`
(2464 bytes) with zero bytes left over, and the header's `NumCols=8` matches
the compile target — strong confidence the byte layout/field decode is
correct, not guessed.

**Finding: this per-dispatch transaction file does not contain the batch=2
loop at all.** It has 3 groups of `BLOCKWRITE` (BD config) +
`CUSTOM_OP_DDR_PATCH` + kick-off `WRITE`, corresponding to input-tile loads
and an output-side transfer for ONE dispatch invocation — no field anywhere
resembles a `batch_index * stride` computation, and no literal
duplicated-with-different-offset pair exists that would represent "batch 0's
BD" vs "batch 1's BD". The batch loop is realized ABOVE this artifact, most
likely at the HAL command-buffer / control-code level, which invokes this
same per-dispatch BD sequence twice with **different HAL buffer argument
bindings** supplied from outside — meaning if a "batch × constant" address
computation exists and degenerates at batch=0, it lives in
`runtime/src/iree-amd-aie`'s HAL driver / command-buffer dispatch sequencing
code, which **has not been inspected in this investigation** and is a
genuinely new, unexplored layer (distinct from everything else in this doc,
which stayed within the compiler's IR/codegen/disassembly).

**Sixth experiment — the "bypass test", and the single most decisive result
of this whole investigation.** Motivated by wanting to separate "input DMA
never arrives" from "compute corrupts a good value" from "the write-back
path itself drops batch 0's write regardless of content." Implemented a
compute bypass at the correct point in the pipeline (`AMDAIEVectorization.cpp`,
right after `AMDAIEVectorizationPass`'s main vectorize loop — NOT in
`AMDAIECoreToStandard.cpp`, where a first attempt at this failed silently
because by that late stage `vector.contract` has already been converted to
`aievec.matmul`, a different op type, so the walk found nothing to replace).
Two variants tried:
- Replace `vector.contract` with its own (already sign-extended) LHS operand
  — i.e. write the real input straight to the output. This got as far as
  compiling at the MLIR level (confirmed via disassembly: the RHS input
  became provably dead and dropped from the function signature, proving the
  rewrite fired) but **crashed `llc`** with a different GlobalISel failure
  (`unable to legalize instruction: G_SEXT <64 x s8> to <64 x s32>`) — the
  aievec.matmul lowering pattern normally absorbs/elides this exact
  sign-extension as part of building the matmul intrinsic; with the
  contract removed, the bare sign-extend has to go through generic codegen,
  which AIE2P's backend can't legalize at this width. Abandoned in favor of
  a cleaner test.
- Replace `vector.contract` with a **hardcoded constant vector** (broadcast
  value 7, later 7000 once the first value turned out too small and got
  rounded to zero by the output requantization scale) of the exact same
  type/shape as the contract's result. No load-dependent arithmetic at all,
  so no legalization surprises — this is the purest possible test of "does
  ANYTHING the core tries to store into batch 0's slot survive," with zero
  dependency on input correctness or compute correctness.

**Result: batch 1 correctly shows the constant (`0.07637637`, matching
`7000 * combined_scale` through the output requantization) — but batch 0 is
STILL exactly all-zero, even for a value that has nothing to do with input
data or matmul arithmetic at all.**

This is decisive: **it rules out both "input DMA for batch 0 never arrives"
and "the vectorized compute corrupts a real result into zero."** Neither of
those could produce this outcome — a value with zero dependency on either
mechanism still fails to reach the final output for batch 0. The bug is
conclusively in the **write-back path** for batch 0's result specifically
(core-local buffer → real output buffer), independent of what's being
written. Combined with the earlier finding that the control-code/DMA
sequencing IR is identical between vectorized and non-vectorized compiles,
this leaves a genuinely narrow, strange target: something about batch 0's
write-back fails only when the *compute path preceding it* is the vectorized
one — even though the writes/DMA/locks around it are structurally identical
to the (working) non-vectorized case. Reverted (not committed).

**Seventh experiment — a MACRO-scale delay, and the biggest breakthrough of
the day.** All four earlier delay experiments were micro-scale (a handful of
extra cycles at a specific point). This one tests something categorically
different: a large, real, runtime-looped delay (tens of thousands to
millions of iterations of a volatile store) inserted at the very *start* of
the whole core program, before anything else — testing a coarse host-driver
vs. core-startup race rather than a few-cycle instruction-level one.

Implementation note: a plain `scf.for` doesn't work here — by the time
`AMDAIECoreToStandardPass` runs (in `XCLBinGen.cpp`'s separate
`generateUnifiedObject` pass manager), SCF-to-CF lowering has already run
once earlier in the pipeline and never runs again, so a freshly-inserted
`scf.for` is left unconverted and later crashes the final LLVM-IR
translation (`LLVM Translation failed for operation:
builtin.unrealized_conversion_cast`). Built the loop by hand instead, out of
`cf.br`/`cf.cond_br` blocks (`rewriter.splitBlock` to carve the rest of the
entry block into an "exit" block, then a header block with the loop
condition and a body block with one volatile store) — this lowers cleanly
since `cf` ops are already proven to work throughout this exact pipeline.

**Results, escalating the iteration count:**

| Delay (iterations) | Result |
| --- | --- |
| 0 (no delay, baseline) | batch 0 exactly all-zero, 100% deterministic across every prior run this session |
| 50,000 | **non-deterministic** — reran 3×: two runs all-zero, one run byte-identical to the original single 50k run (rows 0-7 correct, rows 8-15 zero) |
| 500,000 | deterministic again, but now correct: rows 0-7 exactly match the fp32 reference (corr ~0.9998), rows 8-15 still exactly zero — reran 3×, byte-identical every time |
| 2,000,000 | **identical to 500,000** — no further improvement |
| 10,000,000 (20x more than 500k) | **identical to 500,000 and 2,000,000** — confirms a genuine plateau, not "just needs more time" |

**This proves two separate things conclusively:**

1. **A real, macro-scale timing race exists and is fixable by delay** — the
   jump from "always deterministically zero" to "non-deterministic" at
   50,000 iterations, and then to "deterministically better" at 500,000, is
   the classic signature of perturbing a genuine race window. This does NOT
   contradict the four earlier micro-scale (few-cycle) experiments failing —
   it explains why they failed: the actual race window is apparently much
   larger than a few cycles (something at the scale of host-driver
   kickoff/DMA-queue setup, not core-instruction spacing), so a 16-cycle
   nudge was noise relative to it, while a 500,000-cycle nudge is not.
2. **The residual failure (rows 8-15, all-zero, unaffected by 20x more
   delay) is a SEPARATE bug, unrelated to timing.** M is padded 16→32 and
   tiled into multiple M-sub-tiles, each handled by a *different physical
   core* (tile assignment is a function of the M-tile/N-tile loop indices
   only, not the batch index — recall `amdaie.tile(%11, %12)` computed from
   `%arg3/%arg4`, the M/N-tile loop variables). Rows 0-7 and rows 8-15 are
   different M-tiles running on different physical cores. The delay fully
   fixed whichever core(s) handle the row-0-7 tile, but had zero effect —
   not even at 20x the iteration count — on whichever core(s) handle the
   row-8-15 tile. So "batch 0 is broken" was never quite the right framing:
   it's closer to "certain specific physical cores have a first-use problem,
   and at least one flavor of that problem is a fixable macro-timing race,
   while another flavor (on a different core) is not fixed by delay at all."

All delay values tested via the same hand-built `cf.br`/`cf.cond_br` loop
inserted at the very start of `coreToStd`'s output function; reverted after
each test (not committed).

**Eighth: compared the delay-fixable tile (physical row 2) against the
delay-immune tile (physical row 3) at every level the toolchain exposes —
found total symmetry.** Disassembled `core_0_2.elf` (row 2, M-tile 0, rows
0-7 of output — fixed by the macro delay) and `core_0_3.elf` (row 3, M-tile
1, rows 8-15 — delay-immune) from the ORIGINAL, unpatched compile: **byte-
for-byte identical instructions**, differing only in meaningless internal
LLVM debug label numbers (`.LBB32_1` vs `.LBB24_1`). Went one level up and
compared the amdaie-dialect MLIR (`--mlir-print-ir-before=iree-amdaie-controlcode-to-transaction`):
`tile_1_2`/`tile_1_3` (and every other column pair checked) have identical
buffer/lock declarations (6 buffers, 6 locks, same IDs 0-5, same initial
values); the `amdaie.core` ops for both have the same 2-input/1-output
shape, correctly sharing the RHS operand (which doesn't depend on M-tile)
and differing only in which LHS-slice SSA value feeds each. Went one level
further and checked the actual NPU transaction's explicit BD/queue
programming (`amdaie.npu.write_bd`/`push_to_queue`/`address_patch`/`tct_sync`,
lines 1786-1941 of that dump): **every single one of these ops has `row =
0`** — this block only programs the shim row (the DDR↔memtile DMA hop) and
never references row 2, 3, 4, or 5 at all. The actual row-to-row data
movement is handled entirely inside each core's own compiled program via
its object-FIFO/lock mechanism — which is the thing already proven
byte-identical between row 2 and row 3.

**Conclusion: no structural difference exists between row 2's and row 3's
configuration anywhere IREE/the compiler generates** — not in the core ELF,
not in the MLIR-level tile/lock/buffer setup, not in the explicit NPU
transaction programming (which doesn't even address these rows). Combined
with the delay result (500k iterations fixes row 2 fully; 10M iterations —
20x more — still doesn't touch row 3 at all), the only remaining explanation
is a genuine **physical/hardware asymmetry between AIE tile rows**
(e.g. a real difference in NoC distance/arbitration to the memtile or shim
DMA engine between row 2 and row 3, with row 3 having some non-delay-fixable
startup behavior) that lives below anything this toolchain — or static
analysis of its output — can express. Confirming this further needs actual
hardware-level tooling (JTAG/AIE trace), not more compiler-side comparison;
there is nothing left to diff at the compiler/IR/disassembly level.

**Hypothesis considered and refuted with direct evidence: AIE "checkerboard"
row-parity memory mirroring.** AMD/Xilinx AIE hardware has a real, documented
concept where even/odd tile rows have their local data memory positioned
asymmetrically (East vs. West of the core), which can mirror the
core-to-neighbor-memory address mapping between even and odd rows — a
plausible-sounding explanation for a row-parity-specific bug that survives
identical compiled code (row 2 and row 3 could read a fixed address
differently in hardware even with byte-identical instructions). Checked this
against real vendored driver source rather than accepting it on general AIE
knowledge:

- The mechanism is real and present in `third_party/aie-rt`'s
  `_XAie_GetTargetTileLoc` (`driver/src/core/xaie_elfloader.c:145-179`):
  data memory addresses are split into 4 directional 64KB windows
  (South/West/North/East, at 0x40000/0x50000/0x60000/0x70000), and for
  East/West, `RowParity = Loc.Row % 2` determines whether a given address
  window resolves to the same tile or an adjacent-column tile.
- Our kernel's actual addresses (`0x70400`/`0x70500`/`0x78000`) land exactly
  in the **East** window (0x70000-0x7FFFF) — DataMemSize is 64KB on AIE2P
  (`xaie2pgbl_reginit.c:185`), so this is precisely the address range the
  mechanism operates on. The theory's target was a real match.
- **But**: AIE2P's device config sets `IsCheckerBoard = 0`
  (`xaie2pgbl_reginit.c:180`) — only original AIE1 sets it to `1`
  (`xaiegbl_reginit.c:1140`). The code forces `RowParity = 1` unconditionally
  whenever `IsCheckerBoard == 0`, meaning **on AIE2P (what npu4/Strix uses),
  row-parity mirroring never happens at all** — it's an AIE1-generation
  behavior explicitly disabled at the vendor-driver level for our hardware
  generation. (Separately, re-deriving the mechanism's direction by hand
  also gave the opposite of the observed asymmetry — row 2 working, row 3
  not — a second, independent reason this specific theory doesn't fit even
  if checkerboard behavior were somehow active.)

**Verdict: refuted for this hardware, not confirmed.** A good, well-reasoned
hypothesis that turned out not to apply to AIE2P specifically — recorded here
so it isn't re-derived and re-investigated from scratch in a future session.
(One loose end not chased further: `_XAie_GetTargetTileLoc` is the ELF
*loader's* host-side logic for deciding which physical tile to load
initialization data into, not necessarily the exact same code path a
*running* core's own load/store address decoding uses — confirming those are
governed by the identical `IsCheckerBoard` gate was not verified beyond what
this pass established.)

**Ninth: the decisive confirmation — a single-shot (non-batched) sanity
check, then delay moved to the batch BOUNDARY instead of the start.**

First, a cheap but critical check using data already in hand: does physical
tile row 3 work correctly at all when there's no batch loop (the original §1
2D matmul, M=32, no batching, so every physical core in rows 2-5 runs its
tile exactly once)? Checked row-wise correlation of that repro's output by
M-tile:

| M-tile (rows) | Physical row | corr |
| --- | --- | --- |
| 0 (0-7) | 2 | 0.9999 |
| 1 (8-15) | 3 | 0.9998 |
| 2 (16-23) | 4 | 0.9998 |
| 3 (24-31) | 5 | 0.9999 |

**Every row works perfectly when each core only runs once.** Row 3 is not
defective — it only fails when the SAME physical core is asked to execute
the same kernel body twice in a row (once per batch), and only on that
core's first execution among the two. This reframes the whole bug: it was
never "row 3 is broken," it's "a given physical core's first-of-N-repeated-
executions has a problem, and the specific flavor of that problem differs by
row (row 2's is a fixable timing race near the very start; row 3's looked
delay-immune when delay was only inserted before the first iteration)."

That "before the first iteration only" caveat turned out to be the missing
piece. All delay experiments so far inserted the busy-wait at the very
*start* of the whole core program (before batch 0 begins). None had tried
inserting it at the *boundary between batch iterations* — right after batch
0's locks release, before batch 1's locks acquire. Moved the same
`cf.br`/`cf.cond_br` busy-wait loop (500,000 volatile stores) to fire
immediately after **every** lock-release call in `lockToStd`'s output
(implemented in `coreToStd` instead, walking for `func.call`s to the
`*.release` runtime function after cloning the region into the new
`func.func`, splitting the block right after each one) — this naturally
lands between batch 0's last release and batch 1's first acquire (and
harmlessly again after batch 1's own final release).

**Result: fully correct.** `iree-run-module`, 4 runs: batch 0 corr 0.9998,
batch 1 corr 0.9999, every time, no zero rows anywhere. Also verified at
**batch=4**: all 4 batches corr 0.9998. **This is a complete, reproducible
fix for the bug this entire investigation has been chasing** — not by
patching the real root cause, but by proving conclusively that the root
cause is a timing race specifically at the *inter-iteration boundary* on a
per-physical-core basis, and that a sufficiently large delay placed exactly
there eliminates it for every affected row, not just row 2.

**This is a workaround, not a real fix.** Inserting a ~500,000-cycle
busy-wait after every one of the ~3 locks released per batch (≈1.5M cycles
between each pair of iterations, plus another ≈1.5M wasted at the very end)
burns most of the performance benefit vectorization was supposed to provide.
The actual fix belongs at the level of whatever is racing here — most likely
host-driver/DMA-kickoff sequencing relative to core-to-core lock reuse
timing (see the still-unexplored `runtime/src/iree-amd-aie` HAL driver
angle from earlier) — not a permanent busy-wait baked into every kernel.
Reverted after verification (not committed); the exact patch (in
`coreToStd`, walking `func.call`s to `*.release` and inserting the
`cf`-based delay loop after each) is preserved in git history of this
conversation/session if needed again — see "Suggested next steps" below for
what to try to shrink or properly fix this.

**Tenth: pinning down WHY it's always batch 0 specifically, and finding a
much cheaper fix shape as a result.** A sharp follow-up question: if the
race is simply "release doesn't wait for the real hardware handshake before
the same lock gets reused," that race should apply to *every* iteration
boundary (0→1, 1→2, 2→3, ...), not just the very first one. But every
experiment this session, at both batch=2 and batch=4, found the bug on
*exactly* the first iteration and nowhere else. Why would only the very
first use of a lock be exposed to this, when the lock gets reused
identically every time?

**Hypothesis: AIE locks are counting semaphores with a pre-charged initial
value**, and this is exactly the double-buffering bootstrap idiom (a lock
representing "buffer slots free to write" starts pre-loaded with credit so
the *very first* use doesn't need to wait for anything — correctly, since
nothing has been produced yet, there's genuinely nothing to wait for). The
theory: this pre-charged credit is also what lets the first
acquire-after-release cycle skip the REAL hardware confirmation that gates
every subsequent reuse of the same lock. From the second use onward, the
credit is exhausted and every acquire genuinely blocks on real hardware
confirmation — which is why iterations 1, 2, 3, ... are always safe
regardless of batch count, and only iteration 0 ever races.

**This is directly testable**: if true, inserting the delay ONLY after the
very first release of each distinct lock (skipping every subsequent
release of that same lock) should still fully fix correctness, at any batch
count. Implemented this precisely: tagged each acquire/release
`func.call` in `lockToStd` with a stable per-lock integer id (derived from
`useLock.getLock()`'s defining op via a `DenseMap`, since the call's actual
value operand is just the acquire/release count — usually a useless
constant `1` for every call, not a lock identity — a first attempt at this
filtered on that value by mistake and inserted the delay only once total,
coincidentally reproducing the original all-zero bug and initially looking
like a refutation before the bug in the filter was found and fixed). Then
in `coreToStd`, grouped release calls by that id and inserted the delay
loop only after each id's *first* occurrence.

**Result: still fully correct, at both batch counts, with far less
delay:**
- batch=2: batch 0 corr 0.9998, batch 1 corr 0.9999 — reproduced across 4
  runs, byte-consistent.
- batch=4: all 4 batches corr 0.9998.
- Only **3 delay insertions total per core** (one per distinct lock),
  regardless of batch count — versus 6 (batch=2) or 12 (batch=4) under the
  "every release" version from the Ninth experiment. Confirmed via
  disassembly that peano/llc even compiles the delay loop into AIE's
  zero-overhead hardware loop feature (`lc`/`ls`/`le` registers) rather than
  a chain of compare-and-branch instructions, so the actual code-size cost
  is small (~150-170 bytes) — the real cost is only the wall-clock time the
  loop burns, and critically, that time is now paid ONCE per core rather
  than once per batch iteration.

**This is strong, direct confirmation of the pre-charged-lock hypothesis**,
and changes the shape of the eventual real fix: it should target
whatever governs a lock's INITIAL credit value / the very first
acquire-release cycle specifically, not something that needs to run on
every iteration boundary. It also gives a dramatically cheaper *workaround*
shape in the meantime — fixed one-time cost per core instead of a cost that
scales with the number of batches/repeated dispatches, which matters a lot
for something like BERT with many repeated attention-batch dispatches.
Still not committed; reverted after verification, same as every other
experiment this session.

**Where this genuinely stands now (updated after the "Tenth" experiment
above): root cause fully understood down to the mechanism, working
workaround confirmed and made cheap, real fix not yet implemented.** The bug
is a timing race specific to the very *first* acquire-release cycle of each
lock on a reused physical core — consistent with AIE locks being
pre-charged counting semaphores where the first use skips the real hardware
confirmation that gates every later reuse. Not a per-row hardware defect,
not a code-generation asymmetry, not the earlier checkerboard-memory theory.
A delay inserted only after each lock's first-ever release reliably fixes
every affected row at both batch=2 and batch=4, at a FIXED one-time cost per
core (not one that scales with batch count). What's still missing is a real
fix — something that doesn't need a busy-wait at all — which most likely
means finding and correcting the actual initial-credit/lock-configuration
value itself (or the real hardware confirmation path being skipped for that
first cycle) rather than working around its absence with a delay. See
"Suggested next steps" for where that fix likely lives.

## Runtime hardware debugging tooling — surveyed, mostly unavailable

Before the two experiments above, surveyed what's actually available in this
environment for RUNTIME (not static IR/disassembly) visibility into the AIE
core during the buggy run:

- **AIE hardware trace units exist but aren't wired into our pipeline.**
  `third_party/mlir-air` has full documented trace support (`docs/trace.md`,
  `air-to-aie`'s `insert-trace-packet-flow=true`, `airrt-to-npu`'s
  `trace-size`/`trace-offset`, a `parse_trace.py` for Chrome Trace output) —
  but this lives entirely in the separate mlir-air/`aircc` toolchain, not the
  `compiler/plugins/target/AMD-AIE` plugin `iree-compile` actually uses.
  Porting it over is substantial, multi-day-scale infrastructure work, not a
  flag.
- **`xrt-smi` is installed on the host** (`/opt/xilinx/xrt/bin/xrt-smi`,
  v2.25.0) and works (`xrt-smi examine` confirms device presence: "NPU Gorgon
  Point 1", aie2p, 6x8, firmware 1.1.2.64) but only exposes
  `aie-partitions`/`all`/`host`/`platform` reports for this device class — no
  error-register, timeline, or trace report. `-r aie-partitions` right after
  a run just says "No hardware contexts running on device" — no history.
- **`dmesg` and `/sys/kernel/debug/accel/`** need root; not accessible with
  current permissions. `/sys/class/accel/accel0/device/` has only generic PCI
  attributes, nothing amdxdna-specific.
- **A real Vitis 2025.2 (and 2023.2) install exists on the HOST** (not in the
  lightweight dev container) at `/tools/Xilinx/2025.2/Vitis`, including a real
  `aiesimulator` binary (`/tools/Xilinx/2025.2/Vitis/aietools/bin/aiesimulator`)
  — a cycle-accurate AIE simulator that could in principle show the exact
  relative timing of "core writes SRAM" vs "DMA reads it." **Not actually
  tried**: `aiesimulator` consumes artifacts from the `aiecc`/mlir-air graph
  compilation flow, not the `.vmfb`/HAL-executable format `iree-compile`
  produces for the amdxdna runtime — compatibility between the two is
  unconfirmed and would need its own investigation before this is usable for
  our specific repro.

Bottom line at the time: no real runtime visibility without either porting
mlir-air's trace infrastructure into the AMD-AIE plugin, or confirming/wiring
up `aiesimulator` compatibility with our `iree-compile` output — both
non-trivial, unstarted.

## Current state / what's uncommitted — READ BEFORE CONTINUING

- **`third_party/iree/third_party/torch-mlir` has an uncommitted, unpinned
  local patch** (the `AtenBmmOp` quantization support from §3 AND §4 — three
  edits total now: `QuantizeOperandsPastCommutingOps<AtenBmmOp,2>` and
  `QuantizeAccumulator<AtenBmmOp>` in `FuseQuantizedOps.cpp`, plus the
  zero-point-aware `ConvertAtenBmmOp` rewrite in `Linear.cpp`). This
  submodule points at plain upstream `iree-org/torch-mlir` (not a fork this
  team controls, unlike `third_party/iree` itself which is `ace-knu/iree`).
  The patch exists **only in this working tree's checkout** — it is NOT
  committed anywhere, not even locally in that nested repo (checked: `git -C
  third_party/iree/third_party/torch-mlir status --short` shows the two files
  modified, uncommitted). It is at real risk of being silently lost on any
  `git submodule update`/reset/fresh clone. Before doing anything that could
  discard it, either commit it locally in that nested repo, or copy the two
  diffs somewhere durable. The two files: `lib/Dialect/Torch/Transforms/FuseQuantizedOps.cpp`
  and `lib/Conversion/TorchToLinalg/Linear.cpp`. **This patch is now
  correctness-verified** (§4) for the non-vectorized/CPU paths — it's good,
  not speculative.
- `models/vgg16/vgg16-12.onnx` was downloaded (553MB, gitignored, not
  committed — re-download via the curl command in `models/vgg16/README.md`
  if needed again).
- All scratch/repro work is in `_local/int8_debug/` (gitignored, not
  committed): quantization gen scripts, ONNX/MLIR/vmfb intermediates, and
  `.npy` input/output/reference comparisons for every experiment above. Latest
  batch-matmul vmfbs: `bmm_int8_v3.vmfb` (npu4, vectorization on — the one
  with the remaining §4a bug), `bmm_cpu_v2.vmfb` (CPU, correct),
  `bmm_out_npu_novec_v2.npy`/etc. (npu4 vectorization-off, correct).
- No source changes in this repo's own `compiler/plugins/target/AMD-AIE`
  were needed for §1-§4 — all of that was either "already works" (§1) or a
  fix inside vendored `third_party/iree`/`torch-mlir` (§2 not-yet-attempted,
  §3+§4 attempted+uncommitted+verified). The remaining §4a bug, being
  specific to `iree-amdaie-vectorization`/aievec lowering, will most likely
  need a fix in this repo's own AMD-AIE plugin — but not yet confirmed. Two
  exploratory patches to this repo's own AMD-AIE plugin (`llc -O0` in
  `XCLBinGen.cpp`, inline-asm NOP injection in `AMDAIECoreToStandard.cpp`)
  were tried for §4a and **reverted** — `git status`/`git diff` on
  `compiler/` should show clean at the end of this session.
- Nothing from this session is committed to git. `git status` on the main
  repo only shows the two pre-existing untracked VGG docs from a prior
  session; the torch-mlir submodule's dirty state doesn't even surface there
  (nested submodule, two levels down) — don't rely on top-level `git status`
  to notice if that patch goes missing.

## Eleventh experiment: shrinking the delay (binary search) + a correction

Shrank the 500,000-iteration bound at batch=2, confirming 100,000 → 10,000
all still passed reliably, but 1,000 failed consistently (batch 0 all-zero)
and 5,000-6,000 was flaky (occasional failures). This first pass used `sed`
with a *value-specific literal* replacement (old exact number → new exact
number), which is safe since each old value was unique in the file.

Then, to re-test the same range at **batch=4** in one shot, the loop switched
to a `sed` regex `getI32IntegerAttr([0-9]*)` intended to only touch the delay
`bound` constant. This regex is unanchored to which constant it's replacing —
it matched **every** numeric `getI32IntegerAttr(...)` call in the whole
delay-loop block, clobbering `oneI32` (alloca size, was `1`), `dummy` (stored
value, was `0`), `zero` (loop init, was `0`), and `stepOne` (increment, was
`1`) down to the *same* value as `bound`. Two effects: (a) `zero == bound`
made the loop's entry condition `iv < bound` false immediately, so the "delay"
executed **zero iterations** regardless of the value being tested, and (b)
the alloca's size operand became the same huge number, allocating that many
`i32`s on the stack. This produced a batch=4 result of **100% failure at
every tested value including the previously-verified 500,000** — which looked
like a serious regression/contradiction of the earlier finding, but was
actually an artifact of a broken loop, not a real hardware result. Caught by
reading the file back and noticing all five constants had become identical.

**Lesson recorded for future edits to this file: never use a value-agnostic
regex across multiple sibling constants that happen to share a token prefix
— anchor by line number or by the old constant's known distinct value.**

After restoring the correct constants (`oneI32=1`, `dummy=0`, `zero=0`,
`stepOne=1`, only `bound` varying) and re-verifying 500,000 was genuinely
clean again (0/10 at batch=2, 0/10 at batch=4), the real binary search at
**batch=4** (the more sensitive case — see below) found:

| delay (iterations) | batch=4 result | batch=2 result |
|---|---|---|
| 500,000 | 0/10 failed | 0/10 failed |
| 100,000 | 0/10 failed | (not re-tested) |
| 20,000 | 0/10 failed | (not re-tested) |
| 12,000 | 0/10 failed | (not re-tested) |
| 10,000 | **0/20 failed** | **0/10 failed** |
| 8,000 | 0/12 failed | (not re-tested) |
| 6,000 | 1/12 failed | 0/15 failed (smaller sample, less sensitive) |
| 5,500 | — | 1/10 failed |
| 5,000 | — | 3/5 failed |
| 1,000 | — | 5/5 failed (full batch-0-zero, the original bug) |

Two findings beyond "what's the minimum number":

- **The failure boundary is sharper/appears earlier when tested at batch=4
  than batch=2**, even though the underlying mechanism (each lock's first
  release racing) doesn't itself depend on batch count. This is consistent
  with the race being an *independent per-core* probability: batch=4 doesn't
  change how risky any single core's first release is, but with 16 physical
  cores all doing this once per invocation, the probability that *at least
  one* core loses the race is higher than when testing with fewer
  observations (batch=2's `sed`-literal 6,000 test, which showed 0/15, was
  simply a smaller/luckier sample — batch=4 is the more reliable stress test
  precisely because it multiplies the number of independent per-core trials
  per single hardware invocation).
- At delay=6,000 the one observed failure showed **two** batches corrupted
  (batch 0 *and* batch 2), not just batch 0 — confirming the corruption
  isn't special-cased to "batch 0" per se, just to "whichever core's first
  release happens to lose the race," which is overwhelmingly batch 0 in
  practice (it's genuinely the first use) but not exclusively.

**Chosen final value: 10,000** (down from 500,000 — a 50x reduction),
verified with 0 failures across 20 runs at batch=4 and 10 runs at batch=2.
This is now the value live in `AMDAIECoreToStandard.cpp`'s first-release-only
delay loop (uncommitted, per the note above).

## Twelfth experiment: applying the fix to real BERT-tiny (not just the minimal repro)

Quantized all 16 `MatMul` nodes in `models/bert_tiny/bert_tiny.onnx` (2-layer,
hidden=128, seq_len=32 — Q/K/V/output/FFN projections plus the batched
QK^T/Attn@V attention matmuls) to int8 QDQ via `onnxruntime.quantization.
quantize_static` (`op_types_to_quantize=["MatMul"]`, symmetric activations,
16-sample random-`input_ids` calibration — script: `_local/int8_debug/
quant_bert_tiny.py`, output `_local/int8_debug/out/bert_tiny_int8.onnx`).
Imported and compiled with vectorization **on** (no
`--iree-amdaie-enable-vectorization-passes=false`, unlike the existing bf16
BERT-tiny recipe) and the current 10,000-iteration first-release-only delay
fix live in `AMDAIECoreToStandard.cpp`. Confirmed via
`--mlir-print-ir-after=iree-amdaie-vectorization` that all 16 matmuls
actually vectorize (16 `vector.contract` ops in the dump, matching the 16
quantized `MatMul` nodes one-to-one).

Ran on real npu4 hardware, **20/20 runs correlate 0.99678 against the fp32
torch reference, fully deterministic (identical correlation every run,
`maxabsdiff`=0.347)** — no sign of the batch-0-zero corruption pattern
(which would show near-zero or negative correlation, or run-to-run
variance). This is the first time the delay-based workaround has been
verified on an actual model rather than the minimal 2D/batched matmul repro,
and it holds up: the fix generalizes past the isolated batch-matmul case to
a full transformer encoder with LayerNorm/Softmax/GELU/reshapes interleaved
between the vectorized int8 matmuls.

0.99678 is a real (if modest) drop from the 0.9998-0.9999 seen on the
isolated single/batched matmul cases — expected, since this model chains 16
quantized matmuls' worth of rounding error through 2 encoder layers, rather
than checking one matmul in isolation. Not yet compared against an
int8-but-non-vectorized or CPU-backend run of the same quantized ONNX to
isolate how much of that 0.3% gap is unavoidable quantization noise
(inherent to 8-bit weights/activations with only 16-sample calibration) vs.
something specific to the vectorized NPU path — worth doing before treating
0.99678 as a hard ceiling.

## Thirteenth experiment: bert_base reveals a SECOND, delay-immune bug

Applying the exact same recipe (quantize all `MatMul` nodes, vectorization
on, current delay fix) to `models/bert_base/bert_base.onnx` (12-layer,
hidden=768, 12-head, batch=12 attention — a scaled-up sibling of BERT-tiny,
which worked perfectly) did **not** work: 8 runs gave non-deterministic
corr in **0.70-0.90**, with the delay bumped from 20,000 → 500,000 →
5,000,000 (10x, then another 10x) making **no improvement at all** — a flat
plateau, unlike the clean monotonic improvement seen tuning the original
batch-matmul bug. This is the signature of a second, different bug, not
just "not enough margin for the known one."

**Error localization (per-position, per-head breakdown of the wrong
outputs)** showed two distinct, superimposed patterns:
- A small, present-in-every-run baseline error concentrated in the **last
  M-tile** (sequence positions 24-31, the 4th/outermost of M=32's four
  8-row tiles) and gently increasing toward the last attention head
  (head 11) — this looks structural/deterministic, not a race.
- On top of that, in roughly 30-40% of runs, **one additional head spikes
  ~2x** above the baseline — and *which* head spikes varies randomly run to
  run (seen: heads 3, 6, 8, 9, 11 across different runs) — this part *is* a
  genuine race, just not the same one already fixed.

**Ruled out via targeted isolation tests** (each a fast, cheap repro,
minutes not the ~3-8 min full bert_base compile):
1. **Not tensor size at the last row/tile**: a plain *non-batched* 2D
   matmul at bert_base's actual FFN scale (`M=32,K=768,N=3072`, all 4
   M-tiles/rows exercised, vectorization on) gave **0/8 error across every
   single row, deterministic, corr=0.9998** — completely clean. Rules out
   "the last physical row (row 5, farthest from the shim DMA) has a timing
   margin that only breaks down once tensors are big enough" — size alone,
   without batching, is fine at any scale tested.
2. **Not batch size or layer/dispatch count in isolation, nor their
   product**: a synthetic chain of `N_LAYERS` sequential batched matmuls
   (`M=32,K=32,N=64`, matching bert's attention shape) at every
   combination of `{layers, batch} = {2,12}×{2,12}` — including the full
   `L=12,B=12` combination matching bert_base's actual scale — gave
   **fully deterministic** results at every setting (0.9998 / 0.9985 /
   0.9968 / 0.9814 respectively), the degradation pattern of ordinary
   accumulating quantization noise, not a race. So neither axis alone, nor
   their combination, reproduces the bug in a *synthetic* back-to-back
   NPU-dispatch chain with no CPU work interleaved.
3. **Does reproduce with the REAL bert_base architecture, truncated to N
   encoder layers** (`_local/int8_debug/export_bert_base_ntrunc.py`, using
   `model.encoder.layer = model.encoder.layer[:N]` on the actual
   `bert-base-uncased` checkpoint, not a hand-built approximation — full 8
   matmuls/layer: Q/K/V/output/intermediate/output-dense projections plus
   QK^T/Attn@V, with real LayerNorm/Softmax/GELU/reshape CPU ops between
   them). Ran 8 hardware trials at each of N=1,2,4,6,8,12:

   | layers | corr (8 runs) | deterministic? |
   |---|---|---|
   | 1 | 0.9859 | yes |
   | 2 | 0.9673 | yes |
   | 4 | 0.9509-0.9510 | almost (±0.0001) |
   | 6 | 0.9291-0.9399 | **no** — splits into exactly two clustered values |
   | 8 | 0.9314-0.9401 | **no** — same two-cluster pattern, similar spread |
   | 12 | 0.70-0.90 | **no** — much wider, qualitatively different spread |

   Confirmed the non-determinism is real (not just "needs more delay")
   by recompiling L=6 at delay=500,000 (50x the working baseline): **the
   exact same two-value split persists, unchanged** (0.9291/0.9292 vs
   0.9399), unlike the original bug which improved monotonically and then
   fully resolved with enough delay.

**Interpretation**: the discrete (not continuously-varying) two-cluster
pattern at L=6/8 — the same two correlation values recur across many runs,
rather than a smooth spread — suggests something closer to "one of two
possible scheduling/ordering outcomes" than a smooth timing skew. This only
emerges once enough real encoder layers (and therefore real interleaved
CPU↔NPU dispatch round-trips, each with genuine wall-clock-variable host
compute in between) accumulate — a synthetic NPU-only back-to-back dispatch
chain of the same nominal size never triggers it, pointing at the
CPU/NPU interleaving itself (variable real-time gaps between dispatches
from actual host-side LayerNorm/Softmax/GELU compute, absent in the
synthetic chain) as a plausible differentiator, though this is not yet
directly confirmed — only inferred from what does and doesn't reproduce it.

**Status: unresolved, needs a different investigation approach than more
delay-tuning.** The delay fix (10,000, restored as the working value for
the *original* bug) stays in `AMDAIECoreToStandard.cpp`, uncommitted. BERT-tiny
(2 layers) is unaffected by this second bug (deterministic, corr 0.99678,
confirmed extensively in the Twelfth experiment) — it's specifically
scale/layer-count-gated, likely starting around 5-6 real encoder layers.
BERT-base int8+vectorization is NOT currently usable end-to-end.

## Fourteenth experiment: found a cheap, reliable repro of the second bug

Tested two more isolation hypotheses for the second (delay-immune) bug,
both scripts in `_local/int8_debug/`:

1. **CPU-NPU interleaving with real host compute, refuted.** Took the
   already-clean synthetic chain (`gen_chained_bmm.py`, repeated SAME-TYPE
   batched matmuls, deterministic even at L=12,B=12) and inserted a real
   `LayerNormalization` (genuine CPU-scheduled op, same op bert uses after
   every block) between every matmul
   (`gen_chained_bmm_cpu_interleave.py`). At L=12,B=12: **fully
   deterministic, 10/10 runs identical (corr=0.97928)**. So real CPU work
   interleaved between NPU dispatches, by itself, is not the trigger —
   refutes the "host-side wall-clock jitter between dispatches" theory in
   this form.

2. **Alternating batched and non-batched matmul DISPATCH TYPES (with tile
   reuse), confirmed as a reproducer.** Real BERT alternates non-batched
   (Q/K/V/output/FFN projections) and batched (QK^T/Attn@V) matmul
   dispatches within every layer, connected by `Reshape`/`Transpose` to
   split/merge the per-head batch dimension — tile assignment being a pure
   function of M/N-tile index (not batch or dispatch-type) means these two
   dispatch types can land on the *same* physical AIE tiles.
   `gen_chained_mixed.py` builds exactly this per-layer shape (non-batched
   `MatMul(M=32,K=64,N=64)` → `Reshape`+`Transpose` → batched
   `MatMul(batch=12,M=32,K=64,N=64)` → `Transpose`+`Reshape` back),
   repeated 12 times. **Result: non-deterministic** — 10 runs gave 3
   distinct correlation values (0.96992/0.97074/0.97103), a small but
   real, non-zero spread (compare: every earlier same-type-only chain gave
   *exactly* one value across every run, always). Confirmed non-delay-fixable
   by recompiling at delay=500,000 (50x): **still non-deterministic, 4
   distinct values across 15 runs (0.96522-0.97074), if anything a wider
   spread than at 10,000** — matching the exact "no improvement, sometimes
   worse" signature seen with the real bert_base model's second bug.

**This is now the best available minimal repro of the second bug**: a
~40-second compile (vs. bert_base's 3+ minutes), 12-"layer" synthetic model
with no BERT-specific semantics at all — just alternating
batched/non-batched matmul dispatch types with shared physical tiles. Two
prior hypotheses are now cleanly separated: repeating ONE dispatch type
(batched-only, any batch/layer count, with or without CPU ops between them)
never reproduces it; introducing a SECOND, alternating dispatch type does.
The likely mechanism: a non-batched dispatch's use of a physical tile
between two batched dispatches' uses of that same tile (or vice versa)
creates a genuinely different lock/timing pattern than the "reuse the same
dispatch type repeatedly" case the original fix was built and tested
against — plausibly a *different* lock (or the same lock in a different
acquire/release sequence position) gets its "first use" moment at a point
the current fix's "first release of each lock, once per core" tagging
doesn't correctly single out, once two different dispatches' worth of
lock/buffer setup interleave on the same tile. Not yet root-caused further
than this — the next step would be comparing this repro's compiled
core-level IR/disassembly for the tiles that are shared between the two
dispatch types against a matching tile that's used by only one dispatch
type, the same way the original bug's row 2/row 3 comparison was done.

## Fifteenth: four more hypotheses checked and refuted for the second bug (session pause point)

Continued narrowing the second (delay-immune) bug using the cheap
`chainmix_L12` repro. Each of the following was checked with real evidence
(file:line + actual IR/hardware dumps, not just plausibility) and refuted:

1. **Stale lock hardware values across dispatches sharing a tile** —
   refuted. `AMDAIEControlCodeToTransaction.cpp:73-81`'s `appendLockOp` →
   `initializeLock` (`runtime/src/iree-amd-aie/aie_runtime/
   iree_aie_configure.cc:306-313`, `XAie_LockSetValue`) unconditionally
   force-resets every lock a dispatch uses via a real MMIO write, every
   single dispatch, regardless of what dispatch type previously touched
   that physical tile. (Tile-sharing itself IS real, though: dumped
   `--mlir-print-ir-after=iree-amdaie-assign-tiles` for `chainmix_L12` and
   confirmed both the non-batched and batched dispatch use the identical
   full tile set `tile_{0..7}_{0..5}`.)
2. **Stale BD (buffer descriptor) register content** — refuted.
   `configureDMABD`/`initDMADesc` (`iree_aie_configure.cc:25,54`) always
   starts from `XAie_DmaDescInit`'s full `memset`
   (`third_party/aie-rt/driver/src/dma/xaie_dma.c:69-97`), and
   `_XAieMl_TileDmaWriteBd` (`xaie_dma_aieml.c:867-1035`) does one full
   6-word `XAie_BlockWrite32` every time a BD is configured — never a
   read-modify-write, so no field can carry over from a prior dispatch.
3. **Cross-dispatch DMA start-queue drain race** (a genuinely different
   *timing/occupancy*-based hazard, unlike 1-2's value-staleness) —
   refuted for our test configuration. Dispatches are fully
   host-synchronous by default:
   `iree_hal_amdxdna_native_queue_submit_and_wait`
   (`runtime/src/iree-amd-aie/driver/amdxdna/native_linux_kmq.cc:574-610`)
   blocks on `wait_command` until `ERT_CMD_STATE_COMPLETED` before the next
   dispatch is even submitted (`direct_command_buffer.cc:771-855`,
   `:841-843`). (An opt-in `ERT_CMD_CHAIN` batching path exists,
   `--amdxdna_cmd_chain=1`, that *would* remove this guarantee — but it's
   off by default and wasn't used in any test run so far.)
4. **`AMDAIEFoldDmaWaits.cpp` dropping a needed wait for one dispatch shape
   but not the other** — refuted. Its fold logic
   (`foldDmaWaitsByQueue`/`foldDmaWaitsByBatch`) only ever *merges*
   consecutive wait ops into fewer waits; `canFoldByQueue` (line 103-129)
   and `canFoldByBatch` (line 245-268) both force a real, unfoldable wait
   whenever a per-(tile,connection) BD-tracking set is empty or the real
   hardware queue depth (`getDmaMaxQueueSize`, backed by genuine aie-rt
   hardware introspection, not a hardcoded constant) is hit. Empirically
   confirmed on `chainmix_L12`'s actual compiled IR
   (`--mlir-print-ir-before/after=iree-amdaie-fold-dma-waits`): both the
   non-batched and batched dispatch fold 20 wait ops down to 1, and in
   both cases every one of the 20 DMA-producing tokens is still covered by
   a wait occurring later in program order — no gap in either shape.

**Where this leaves things**: every layer of "is the compile-time IR/host
synchronization logically correct" has now checked out clean — lock values,
BD register content, inter-dispatch host synchronization, and wait-folding
are all individually provably correct by direct code+dump inspection, not
just argued from plausibility. The remaining unaudited layer is the actual
lowering of `NpuDmaWaitOp` into whatever primitive really blocks on
hardware completion (a task-completion-token read) — i.e. whether that
*specific* translation has an unconditional, correct semantics, rather than
auditing further which waits the IR keeps (already confirmed correct).
Given the original bug also turned out to be a genuine, undocumented
hardware quirk (a lock's pre-charged first-use credit) rather than
anything visible from source alone, it's plausible this second bug is
similarly something no amount of source reading will surface — worth
weighing real hardware trace tooling against continuing the source audit
if this pattern of clean-but-inconclusive audits continues.

**Sixteenth: isolated dispatch-TYPE change as the active ingredient, separate from the CPU op that mediates it.** `gen_chained_mixed.py`'s repro changed two things at once between the non-batched and batched matmul: the dispatch TYPE (non-batched vs batched) AND the specific CPU op (`Reshape`+`Transpose`, vs. the earlier LayerNorm-interleave test's `LayerNormalization`). To separate these, `gen_chained_bmm_transpose.py` keeps dispatch type FIXED at batched throughout (matching the already-clean LayerNorm test) but swaps in a real `Transpose`+`Transpose`-back pair (genuine data movement, permuting within the `[B,M,N]` tensor, no rank/batch-count change) between every matmul. Result: **fully deterministic, 15/15 identical (corr=0.98135)**. So neither LayerNorm nor Transpose, by itself, triggers anything when dispatch type stays constant — confirms the trigger is specifically the *dispatch-type transition* (non-batched↔batched), independent of which CPU op is sandwiched in between it.

## Seventeenth: PDI/hardware-context reload confirmed real; host-side settling delay tried and found inconclusive due to environmental noise

Following up on the "PDI reconfiguration" theory: confirmed via
`direct_command_buffer.cc:970-990` that **every** dispatch (not just
type-transitions) calls `iree_hal_amdxdna_native_device_create_context` →
a real `create_hw_context` ioctl passing PDI bytes — so a fresh hardware
context genuinely is (re)established every single dispatch, same-type or
not. Since same-type dispatch chains (which also reload every time) stay
clean, the refined theory is: reloading with **identical** PDI content
(same-type dispatches) is safe/idempotent, while reloading with
**different** PDI content (a real type-transition, non-batched↔batched) is
a genuine fabric/switchbox reconfiguration — a qualitatively bigger,
riskier event, consistent with everything observed and with why it's
delay-immune at the core level (the reload is host-confirmed-complete
*before* the core's own program starts).

Tried a host-side settling delay (`std::this_thread::sleep_for`, gated
behind `getenv("AMDXDNA_PDI_SETTLE_US")` for fast iteration without
rebuilding) inserted in `direct_command_buffer.cc` right after
`open_cu()` returns, before the first dispatch against the freshly-loaded
context is submitted — the same idea as the original core-internal delay
fix, but relocated to the correct (host/driver) level this time. Result:
**inconclusive, not a clean fix.** At 5,000us: 2/15 runs still wrong. At
50,000us (10x): first sample 8/20 wrong (worse!), but a back-to-back
30-run comparison of baseline (no delay) vs. 50,000us gave **0/30 wrong
for BOTH** — then a further 40-run baseline batch immediately after
that showed the expected ~20-30% failure rate again (5 distinct
correlation values, not just the earlier two).

**This reveals an important confound that likely affects every
non-determinism measurement in this investigation so far**: the race's
observed failure rate is not stable over time even with the exact same
code and delay value — it appears to depend on some external, uncontrolled
system condition (thermal state, host/NPU load, driver internal state,
etc.) that fluctuates between test batches, sometimes suppressing the race
almost entirely (0/30) and sometimes exposing it at a high rate (~20-30%).
Getting 30/30 clean by chance if the true rate were a stable ~20-30% has
probability ~0.001-0.02% — far too unlikely, so the *true* per-batch rate
itself must be moving around, not just sampling noise around a fixed rate.

**Practical implication**: small-sample (10-20 run) A/B comparisons used
throughout this investigation to accept/reject a hypothesis are less
reliable than they appeared — a clean result in a small sample could
reflect a temporarily-quiet system state rather than an actual fix, and
apparent differences between two conditions tested at different times
could reflect this drift rather than the code change under test. The
*qualitative*/structural findings (dispatch-type-alternation is necessary,
tensor size alone isn't, batch/layer count alone isn't, Transpose alone
isn't) are probably still sound since each was checked via a same-type vs.
different-structure comparison, but exact failure-rate numbers and
"which delay value is safest" conclusions should be treated as
directional, not precise, until re-verified with much larger sample
sizes or a way to control/monitor whatever this environmental variable
is.

The experimental `direct_command_buffer.cc` host-side-delay patch was
reverted (`git checkout --`) after this inconclusive result — no changes
left in that file. `AMDAIECoreToStandard.cpp`'s delay value remains at the
validated 10,000 (unrelated, unchanged).

## Eighteenth: re-tested the host-side PDI settling delay properly (interleaved, real bert_base) — negative result

The Seventeenth experiment's host-side delay test used block-sampling
(N runs of condition A, then N runs of condition B), which the discovered
environmental-noise confound makes unreliable — a time-drift in the
"true" failure rate would masquerade as a condition difference. Redid it
properly: **interleaved** A/B/A/B/... sampling (alternating baseline and
delay on every single run, not in blocks) against the real
`bert_base_L6_int8.vmfb` (layer-truncated real bert_base, the same one
that showed a clean two-cluster non-determinism earlier), using the
`AMDXDNA_PDI_SETTLE_US` env var for fast toggling without rebuilding.

- 50,000us delay, 20 interleaved pairs: baseline 12/20 clean (60%) vs.
  delay 6/20 clean (30%) — delay noticeably *worse*.
- 5,000us delay, 20 interleaved pairs: baseline 11/20 clean (55%) vs.
  delay 9/20 clean (45%) — delay slightly worse again.

**Verdict: the host-side settling-delay approach does not help and may
actively hurt, across two different delay magnitudes, with the more
noise-resistant interleaved design.** Plausible explanation (not
confirmed): `std::this_thread::sleep_for` puts the host thread to sleep,
which on real hardware (this machine is a shared laptop) likely allows
the CPU to drop into a deeper idle/power-saving state during the sleep,
and coming back out of that state to issue the next dispatch may
introduce its OWN new timing jitter — potentially trading one source of
timing variance for another, worse one, rather than adding clean
"settling time." A busy-wait (spinning instead of sleeping) might avoid
this specific confound but wasn't tried.

Reverted the experimental change again
(`git checkout -- runtime/src/iree-amd-aie/driver/amdxdna/
direct_command_buffer.cc`) — no changes left in that file.

**Practical implication going forward**: any future test of a *timing*-
related fix for this bug should (a) use interleaved A/B sampling, not
block sampling, and (b) use large sample sizes (20-30+ per condition),
given how easily this machine's environmental noise can produce a
misleading block-sampled result in either direction.

## Nineteenth: busy-wait (spin) instead of sleep — promising signal at 500ms, session ended here

Retried the host-side settling delay using a busy-wait spin
(`std::chrono::steady_clock`-based, no `sleep_for`) instead of sleeping,
to rule out the "sleep lets the host CPU drop into a power-saving state
and reintroduces jitter on wake" explanation for the Eighteenth
experiment's negative result. Env-var gated (`AMDXDNA_PDI_SETTLE_US`)
in the same spot in `direct_command_buffer.cc`, tested interleaved
against real `bert_base_L6_int8.vmfb` (48 dispatches per run, so the
settle cost applies per-dispatch — expensive to test at scale: 500us ×
48 ≈ 24s of pure spin added per model run at the largest value tried).

- 50,000us spin, 20 interleaved pairs: baseline 60% clean vs. spin 50%
  clean — still no help, similar to sleep_for's result.
- 500,000us spin (10x more), 10 interleaved pairs (session ended before
  reaching the planned 20 — a background continuation was started then
  stopped early at the user's request): **baseline 3/10 clean (30%,
  including two unusually-bad outlier correlations 0.914-0.915 not seen
  at other delay values) vs. spin 6/10 clean (60%), with NO outliers
  that bad** — a real, if modest-sample, positive signal, unlike every
  delay value tried before it (5k/50k via sleep_for, 5k/50k via spin).

**Not conclusively proven** (n=10 is small, and this investigation has
already demonstrated how noisy small samples can be) but the FIRST
positive-direction result after four consecutive negative/neutral
attempts at nearby smaller values — suggests a possible threshold
effect where the needed settling time is much larger than initially
guessed (500us-50ms didn't help; 500ms showed promise). Worth prioritizing
as the next thing to properly re-verify (larger interleaved sample, e.g.
20-30 pairs) before drawing a firm conclusion either way.

Reverted the experimental change again (`git checkout --
runtime/src/iree-amd-aie/driver/amdxdna/direct_command_buffer.cc`) — no
changes left in that file. `AMDAIECoreToStandard.cpp`'s delay value
remains at the validated 10,000 (unrelated to this experiment, unchanged).

**Session paused here at the user's request, having reached a genuinely
promising (not yet confirmed) lead: a ~500ms host-side busy-wait after
PDI/context reload, right before the first dispatch against it, may
substantially reduce the second bug's failure rate.** To resume:
re-implement the same env-var-gated busy-wait patch (see this entry for
the exact code), and run a proper 20-30-pair interleaved A/B test at
500,000us (and perhaps try a couple of nearby values, e.g. 200,000 and
1,000,000, to bracket it) against `bert_base_L6_int8.vmfb` before
deciding whether to make this a real (non-experimental) fix. Delay fix value confirmed
restored to the validated 10,000 in `AMDAIECoreToStandard.cpp` before
stopping (still uncommitted). No compiler source is left in a
half-edited/experimental state. To resume: continue at "TCT/NpuDmaWaitOp
lowering audit" above, using the `chainmix_L12` repro
(`_local/int8_debug/gen_chained_mixed.py`, ~40s compile) as the cheap test
case rather than full bert_base.

## Suggested next steps, in order

**§4a is root-caused down to the mechanism (pre-charged lock credit exposes
only the first acquire-release cycle to a real race) and has a confirmed,
cheap (fixed one-time cost per core) working workaround, now shrunk from
500,000 to 10,000 iterations — see the "Tenth" and "Eleventh" experiments
above.** Remaining work is about turning "a workaround exists" into "a real,
cheap fix exists":

1. ~~Try to shrink the delay further.~~ Done — 10,000 iterations, verified
   with 0/20 failures at batch=4 and 0/10 at batch=2. Note the failure
   boundary was found using batch=4 as the stress test, not batch=2 — batch=4
   surfaces the race more often since it has the same number of independent
   per-core trials as batch=2 (16 physical cores either way) but the boundary
   region (5,000-8,000) showed batch=4 failing before batch=2 did in smaller
   samples, so any future re-tuning of this constant should stress-test with
   the highest batch count available, not just batch=2.
2. **Find the real fix at the driver/hardware/lock-configuration level**, so
   no busy-wait is needed at all. Given the mechanism now points
   specifically at each lock's *initial credit value* (or whatever skips
   the real hardware confirmation on a lock's first use), look at how lock
   initial values get chosen/emitted (the buffer/lock declarations
   compared earlier showed values like `2,0,2,0,2,0` — worth understanding
   exactly what those encode and whether adjusting them removes the race
   without any busy-wait), and/or read `runtime/src/iree-amd-aie`'s HAL
   driver / command-buffer dispatch sequencing code (genuinely unexplored
   all session) for anything related to how the first use of a
   double-buffered lock gets bootstrapped.
3. Two hypotheses considered and refuted with direct evidence along the way
   — recorded so they aren't re-derived: AIE "checkerboard" row-parity
   memory mirroring (real AIE1 concept, confirmed disabled on AIE2P via
   `IsCheckerBoard=0` in `third_party/aie-rt`) and "inner vs outer row
   generates different neighbor-routing code" (refuted by md5-identical
   `.text` sections across every row/column checked — the file-size
   difference that prompted the theory is entirely debug-symbol-table
   metadata, not executable code).
4. If a lighter software mitigation is wanted before the real driver fix
   lands: the Tenth experiment's patch is the one to build on — tag each
   acquire/release `func.call` in `lockToStd` with a stable per-lock id
   (the call's own value operand is useless for this, it's just the
   acquire/release count, normally a constant `1` for every call), then in
   `coreToStd` (after cloning into the `func.func`) group release calls by
   that id and insert the `cf.br`/`cf.cond_br`-based busy-wait loop only
   after each id's first occurrence. Fixed, batch-count-independent cost.
   Not committed; reverted after verification each time it was tried this
   session.
5. Once §4a has a real (non-busy-wait) fix: decide whether to commit the
   torch-mlir patch (§3/§4's `AtenBmmOp` quantization support) somewhere
   durable (a fork? a local patch file applied post-checkout?) before it's
   lost — it isn't sitting in this repo's own git history even as an
   uncommitted diff, unlike everything else touched today.
6. VGG16 Conv int8 (§2) remains a separate, larger piece of upstream-iree
   work, deliberately parked in favor of BERT.
