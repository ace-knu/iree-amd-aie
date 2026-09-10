// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Two Flow-phase passes that adapt NPU contraction dispatches for the pack-peel
// backend: AMDAIEPadContractionDispatches (pad to tile multiples) and
// AMDAIESplitLargeContractionDispatches (split a large-(K,N) transpose_b matmul
// into N-chunk sub-dispatches, at the bottom of this file). They share the same
// dispatch/matmul helpers and host slice/pad/crop/concat dispatch builders.
//
// --- AMDAIEPadContractionDispatches ---
// Pads the operands of NPU contraction dispatches up to the target's pack-peel
// tile multiples, at the Flow dispatch boundary.
//
// Runs right after `AMDAIEAssignDeviceAffinities`, when every `flow.dispatch`
// already carries a `stream.affinity` and dispatch regions are outlined. This
// pass assumes the amd-aie contraction dispatch is a matmul whose accumulator
// (`outs`) is *either* a fresh zero-init (`empty -> fill(0) -> matmul`), *or* a
// full-tensor load of a [M, N] bias operand straight from a binding
// (`load -> matmul`, e.g. a conv2d+bias that never went through
// `DetachElementwiseFromNamedOpsPass`'s zero-fill split) -- see `MatmulInfo`'s
// comment. The matmul's result is *either* stored directly, *or* consumed by
// one or more fused elementwise ops (e.g. a ReLU or bias-add fused onto the
// same dispatch by normal IREE fusion) before the store. For such a dispatch
// pinned to an amd-aie device whose M/K dims are not multiples of the target
// tiles, it:
//   * zero-pads the matmul operands to the padded shape *outside* the dispatch
//     (a host `tensor.pad` dispatch), and
//   * rewrites the executable so its bindings/loads/matmul see the padded shape.
//
// K is a reduction dim: the padded elements are zero, so the extra products are
// zero and the output is unchanged (no cropping) -- this holds regardless of
// what (if anything) consumes the matmul's result, so K-only padding is always
// safe to apply. The output dims M (outer) and N (inner) are different: when
// either is padded, the matmul output, the output binding and the store must
// grow too, and the padded dispatch result is cropped back to the logical
// shape by a `tensor.extract_slice` *dispatch* placed on the host (CPU). Using
// a dispatch (rather than `flow.tensor.slice`, which only crops a contiguous
// outer-dim range) lets the inner N dim be cropped with a strided extraction,
// so a dispatch whose N is non-divisible (e.g. a 1000-class classifier matmul)
// is padded and cropped like M.
//
// Growing the output through an arbitrary fused consumer is still not
// implemented in general, but one common shape *is* handled: a single fused
// elementwise consumer (e.g. a ReLU) that is a pure shape-preserving unary map
// over the matmul's result (one input == the matmul result, one output,
// identity indexing maps, all-parallel iterators -- see
// `isShapePreservingUnaryMapOf`). Such an op cannot depend on absolute tensor
// size or mix across positions, so it is grown to [mPad, nPad] right alongside
// the matmul and its (discarded-by-cropping) output on the padded region is
// irrelevant to correctness. A dispatch that needs M/N padding and has any
// other shape of fused consumer (a reduction, more than one real input, or a
// chain of more than one intervening op) is still conservatively skipped
// entirely -- see `MatmulInfo`'s comment and the `padOut` check in
// `runOnOperation`.
//
// The padding multiples are read from each dispatch's own device affinity
// (affinity -> device global -> executable target attr), never hardcoded, so the
// logic is portable if device placement ever moves to a later (Stream) stage.

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDevicePlacementUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-amdaie-pad-contraction-dispatches"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static int64_t roundUpToMultiple(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

/// Tile multiples a matmul's M, N, K must be padded up to on a given target.
struct PaddingMultiples {
  int64_t m, n, k;
};

/// Resolves the amd-aie executable target a dispatch is pinned to via its
/// `stream.affinity` (affinity -> device global -> executable target attr).
/// Returns null if the dispatch is not pinned to an amd-aie device.
static IREE::HAL::ExecutableTargetAttr getDispatchTarget(
    IREE::Flow::DispatchOp dispatch, ModuleOp module) {
  auto affinity =
      dispatch->getAttrOfType<IREE::HAL::DeviceAffinityAttr>("stream.affinity");
  if (!affinity) return {};
  Operation *globalOp =
      SymbolTable::lookupNearestSymbolFrom(module, affinity.getDevice());
  auto global = dyn_cast_or_null<IREE::Util::GlobalOpInterface>(globalOp);
  if (!global) return {};
  auto deviceTarget = dyn_cast_or_null<IREE::HAL::DeviceTargetAttr>(
      global.getGlobalInitialValue());
  if (!deviceTarget) return {};
  ArrayRef<IREE::HAL::ExecutableTargetAttr> targets =
      deviceTarget.getExecutableTargets();
  if (targets.empty()) return {};
  IREE::HAL::ExecutableTargetAttr exec = targets.front();
  if (exec.getBackend().getValue() != "amd-aie") return {};
  return exec;
}

/// Computes the M/N/K pad multiples for `target` and the matmul element types,
/// entirely from the device info tables (no hardcoded geometry): M/N from the
/// core-array shape (num_rows/num_cols) times the vector instruction size, K
/// from the shared pack-peel reduction tile. Returns nullopt if `target` is not
/// a recognized amd-aie target.
static std::optional<PaddingMultiples> getPaddingMultiples(
    IREE::HAL::ExecutableTargetAttr target, Type lhsElemType, Type rhsElemType,
    Type accElemType) {
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(target);
  std::optional<int64_t> numRows = getConfigNumRows(target);
  std::optional<int64_t> numCols = getConfigNumColumns(target);
  if (!device || !numRows || !numCols) return std::nullopt;
  AMDAIEDeviceModel deviceModel = getDeviceModel(*device);
  FailureOr<std::array<uint32_t, 3>> instr =
      deviceModel.getAIEMatmulInstructionSize(lhsElemType, rhsElemType,
                                              accElemType);
  if (failed(instr)) return std::nullopt;
  // M and N come from the device tables. K does not: the reduction tile is
  // pinned to the default scalar pack-peel pipeline's 2-level tiling
  // (kPackScaleL1 = 1), because the tile pipeline is a global option and is not
  // on the executable target attr, so it cannot be read per dispatch here. Two
  // consequences, neither reachable in the configuration that is validated
  // today but both real once it widens:
  //   * pack-peel-4-level-tiling uses kPackScaleL1 = 2 (KernelDispatch.cpp),
  //     i.e. a K tile of 64, so padding to 32 is too little and the
  //     divisibility failure this pass exists to prevent comes back.
  //   * with ukernels enabled KernelDispatch picks a factor of K instead, so
  //     padding K at all is unnecessary work. `ukernels` *is* on the target
  //     attr, so that half could be handled without any new plumbing.
  //
  // Do NOT widen the K multiple here to `reductionTile * numRows` (128 on
  // npu4). That was tried, on the theory that a K-tile count not divisible by
  // numRows hangs -- a theory an isolated, single-dispatch matmul_transpose_b
  // reproduction did seem to support (K=25120 and K=25152 hung, while
  // K=25088, K=25216 and K=25344 ran in ~2s). Applied *here* it is
  // nonetheless wrong: this multiple governs every NPU contraction, conv
  // dispatches included, and widening it made VGG16 hang on real hardware
  // (`ert state 8`) in configurations that pass with the multiple left at
  // the plain reduction tile -- confirmed by flipping just this one value
  // back and forth on an otherwise identical build and model (58s success vs
  // 2min timeout). Whatever the isolated repro was measuring, it does not
  // generalize to the padded conv shapes this pass also feeds.
  //
  // A contraction that genuinely needs its K aligned beyond the reduction
  // tile should arrange that at its source instead, where it can also keep a
  // constant operand constant -- see `AMDAIEFoldBiasIntoContractionPass`,
  // which aligns its own K+1 augmentation so that no padding dispatch is
  // needed here at all (padding a hoisted constant weight through a dispatch
  // is itself what broke FC1: it turns the NPU dispatch's weight operand
  // into a runtime-produced buffer).
  return PaddingMultiples{/*m=*/*numRows * (*instr)[0],
                          /*n=*/*numCols * (*instr)[1],
                          /*k=*/getPackPeelReductionTile(/*kPackScaleL1=*/1)};
}

/// A matmul (`[empty -> fill | load] -> matmul -> [elementwise consumer(s)]
/// -> store`) inside a dispatch executable, with the ops/bindings needed to
/// grow its M/K dims. Sizes are intentionally not cached here -- they are read
/// from each caller dispatch's operands so a shared executable can be handled
/// per caller.
///
/// The accumulator (matmul's `outs`) is recognized in exactly one of two
/// mutually-exclusive forms, populating exactly one of these field pairs:
///   * `fill`/`empty`: a fresh zero-init (`matmul outs(fill(empty))`).
///   * `accLoad`/`accArg`: a full-tensor load of a [M, N] bias operand
///     straight from a binding (`matmul outs(load)`) -- e.g. a conv2d+bias
///     dispatch whose bias was already broadcast to the output shape by an
///     earlier op, so the accumulator itself is a plain external input
///     rather than a zero-init. Growing this form pads the bias binding's
///     *value* the same way LHS/RHS are padded (see `runOnOperation`'s
///     accArg handling): the extra M/N rows/cols it introduces are never
///     read back, since the grown result is cropped to the original [m, n]
///     afterward, so their padded value is immaterial.
///
/// `store`/`outArg` are populated when the single store in the function
/// consumes the matmul result either *directly*, or *indirectly* through
/// exactly one fused `elementwiseConsumer` (see below). When the store's value
/// is produced by anything else (a reduction-shaped consumer, a consumer with
/// more than one real input, or a chain of more than one intervening op),
/// `store`/`outArg` are left null: K-only padding (a reduction dim -- the
/// extra products are zero, so the matmul's result value and shape are
/// provably unchanged) can still proceed using only
/// lhsArg/rhsArg/lhsLoad/rhsLoad, but growing the *output* side through an
/// unrecognized intervening consumer is not handled, so callers must check
/// `store`/`outArg` before attempting to pad M or N.
struct MatmulInfo {
  linalg::MatmulOp matmul;
  IREE::TensorExt::DispatchTensorLoadOp lhsLoad, rhsLoad;
  BlockArgument lhsArg, rhsArg;  // executable bindings feeding LHS/RHS
  linalg::FillOp fill;           // output init: matmul outs = fill(empty)
  tensor::EmptyOp empty;
  // Accumulator-as-bias form (see struct comment); null unless fill/empty are
  // both null.
  IREE::TensorExt::DispatchTensorLoadOp accLoad;
  BlockArgument accArg;
  // Set (non-null) when exactly one elementwise op sits between the matmul
  // and the store, and it is a pure shape-preserving unary map over the
  // matmul's result (see `isShapePreservingUnaryMapOf`) -- e.g. a fused ReLU.
  // Null when the store consumes the matmul result directly.
  linalg::GenericOp elementwiseConsumer;
  IREE::TensorExt::DispatchTensorStoreOp store;  // writes the output binding
                                                  // (null if the store's value
                                                  // isn't the matmul result or
                                                  // a recognized single
                                                  // elementwise consumer of it)
  BlockArgument outArg;  // the output binding (null under the same condition)
};

/// Returns the func of a dispatch's single entry point, or null.
static func::FuncOp getDispatchFunc(IREE::Flow::DispatchOp dispatch,
                                    ModuleOp module) {
  SmallVector<SymbolRefAttr> entryPoints =
      llvm::to_vector(dispatch.getEntryPointRefs());
  if (entryPoints.size() != 1) return nullptr;
  SymbolRefAttr entryPoint = entryPoints.front();
  auto exe = module.lookupSymbol<IREE::Flow::ExecutableOp>(
      entryPoint.getRootReference());
  if (!exe) return nullptr;
  ModuleOp inner = exe.getInnerModule();
  if (!inner) return nullptr;
  return inner.lookupSymbol<func::FuncOp>(entryPoint.getLeafReference());
}

/// Traces a matmul input value to the executable binding (BlockArgument) it is
/// loaded from via a full-tensor `dispatch.tensor.load`.
static IREE::TensorExt::DispatchTensorLoadOp getFullTensorLoad(Value operand) {
  auto load = operand.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
  if (!load) return nullptr;
  // Only a plain full-tensor load (unit strides, zero offsets) is supported.
  for (OpFoldResult o : load.getMixedOffsets())
    if (!isConstantIntValue(o, 0)) return nullptr;
  for (OpFoldResult s : load.getMixedStrides())
    if (!isConstantIntValue(s, 1)) return nullptr;
  return load;
}

/// True iff `generic` is a pure per-position (shape-preserving) unary map
/// whose one real input is exactly `input`: exactly one DPS input, exactly one
/// DPS init/output, every indexing map is the 2-D identity map (no
/// permutation, no broadcast, no projection), and every iterator is
/// `parallel` (no `reduction`). The op's body is deliberately not inspected --
/// these structural checks alone guarantee the op cannot depend on absolute
/// tensor size or mix values across positions, so padding `input` with zeros
/// and then cropping the result later is safe regardless of what arithmetic
/// the body performs (e.g. a fused ReLU's `cmpf`+`select`).
static bool isShapePreservingUnaryMapOf(linalg::GenericOp generic,
                                        Value input) {
  if (!generic) return false;
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;
  if (generic->getNumResults() != 1) return false;
  if (generic.getDpsInputOperand(0)->get() != input) return false;
  if (generic.getNumLoops() != 2) return false;
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(2, generic.getContext());
  for (AffineMap map : generic.getIndexingMapsArray())
    if (map != identity) return false;
  for (utils::IteratorType it : generic.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel) return false;
  return true;
}

/// Inspects the func for a single matmul: LHS/RHS from full-tensor loads of
/// bindings, an `empty`+`fill` output init, and exactly one store in the
/// function. The store may consume the matmul result directly (a plain
/// matmul dispatch), or indirectly through exactly one fused elementwise op
/// that is a shape-preserving unary map of the matmul result (e.g. a ReLU or
/// bias-add fused onto the same dispatch by normal IREE fusion) -- a
/// completely ordinary outcome that is unrelated to whether the matmul's own
/// M/K dims happen to need padding. Anything else feeding the store (a
/// reduction-shaped consumer, a consumer with more than one real input, or a
/// chain of more than one intervening op) is not recognized. In the
/// unrecognized case `MatmulInfo::store`/`outArg` are left null (see the
/// struct comment); the caller is responsible for only using them when
/// non-null.
static std::optional<MatmulInfo> getMatmulInfo(func::FuncOp func) {
  SmallVector<linalg::MatmulOp> matmuls;
  func.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  if (matmuls.size() != 1) return std::nullopt;
  linalg::MatmulOp matmul = matmuls.front();
  auto lhsLoad = getFullTensorLoad(matmul.getInputs()[0]);
  auto rhsLoad = getFullTensorLoad(matmul.getInputs()[1]);
  if (!lhsLoad || !rhsLoad) return std::nullopt;
  auto lhsArg = dyn_cast<BlockArgument>(lhsLoad.getSource());
  auto rhsArg = dyn_cast<BlockArgument>(rhsLoad.getSource());
  if (!lhsArg || !rhsArg) return std::nullopt;
  // The accumulator (matmul's own dps-init operand) is resolved in one of two
  // mutually-exclusive forms -- see MatmulInfo's comment -- entirely
  // independent of what (if anything) consumes the matmul's result
  // downstream, so this is unaffected by the direct-vs-indirect store
  // distinction below.
  auto fill = matmul.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  tensor::EmptyOp empty;
  IREE::TensorExt::DispatchTensorLoadOp accLoad;
  BlockArgument accArg;
  if (fill) {
    empty = fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
    if (!empty) return std::nullopt;
  } else {
    accLoad = getFullTensorLoad(matmul.getDpsInits()[0]);
    if (!accLoad) return std::nullopt;
    accArg = dyn_cast<BlockArgument>(accLoad.getSource());
    if (!accArg) return std::nullopt;
  }
  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> stores;
  func.walk(
      [&](IREE::TensorExt::DispatchTensorStoreOp s) { stores.push_back(s); });
  if (stores.size() != 1) return std::nullopt;
  IREE::TensorExt::DispatchTensorStoreOp store = stores.front();
  IREE::TensorExt::DispatchTensorStoreOp resolvedStore;
  BlockArgument outArg;
  linalg::GenericOp elementwiseConsumer;
  Value storedValue = store.getValue();
  if (storedValue == matmul.getResult(0)) {
    // Plain matmul: the matmul result feeds the store directly.
    if (auto arg = dyn_cast<BlockArgument>(store.getTarget())) {
      resolvedStore = store;
      outArg = arg;
    }
  } else if (auto generic = storedValue.getDefiningOp<linalg::GenericOp>();
            isShapePreservingUnaryMapOf(generic, matmul.getResult(0))) {
    // Exactly one recognized elementwise consumer (e.g. a fused ReLU) sits
    // between the matmul and the store -- resolve store/outArg the same way,
    // the store target is unchanged, only what feeds it differs. A longer
    // chain (the consumer's own input not being the matmul result directly)
    // falls through this `else if` and leaves everything null.
    elementwiseConsumer = generic;
    if (auto arg = dyn_cast<BlockArgument>(store.getTarget())) {
      resolvedStore = store;
      outArg = arg;
    }
  }
  return MatmulInfo{matmul,      lhsLoad,      rhsLoad,       lhsArg,
                    rhsArg,      fill,         empty,         accLoad,
                    accArg,      elementwiseConsumer, resolvedStore, outArg};
}

/// Grows executable binding `arg` and its full-tensor `load` to `newShape`
/// (the matmul revalidates from the loaded operand types).
static void growBinding(IRRewriter &rewriter, BlockArgument arg,
                        IREE::TensorExt::DispatchTensorLoadOp load,
                        ArrayRef<int64_t> newShape) {
  auto dtt = cast<IREE::TensorExt::DispatchTensorType>(arg.getType());
  auto tensorType = dtt.asRankedTensorType();
  auto newTensorType =
      RankedTensorType::get(newShape, tensorType.getElementType());
  arg.setType(IREE::TensorExt::DispatchTensorType::get(dtt.getAccess(),
                                                       newTensorType));
  rewriter.setInsertionPoint(load);
  auto newLoad = rewriter.create<IREE::TensorExt::DispatchTensorLoadOp>(
      load.getLoc(), newTensorType, arg, /*sourceDynamicDims=*/ValueRange{});
  rewriter.replaceOp(load, newLoad.getResult());
}

/// Grows the fused elementwise consumer's own init to a *fresh* `[mPad,
/// nPad]` `tensor.empty` (its values are irrelevant -- the consumer is a pure
/// per-position map, verified by `isShapePreservingUnaryMapOf`, so nothing
/// ever reads the old init's data) and widens its result type to match.
///
/// Must run *before* `growMatmulInit`. The trap this ordering avoids: in the
/// real IR this pass targets (e.g. VGG16's FC1/FC2, "FC -> ReLU"), the
/// consumer's `outs` operand is literally the *same* `tensor.empty` the
/// matmul's own accumulator is built from (`%8 = tensor.empty(); %9 =
/// fill(%8); ... matmul outs(%9) ...; generic ... outs(%8) ...`) -- i.e.
/// `info.empty`'s result has two independent uses, one feeding `info.fill`
/// and one feeding this consumer directly. `growMatmulInit` erases
/// `info.fill`/`info.empty` assuming they have no remaining uses once the
/// matmul is rewired away from them; MLIR cannot erase an op that still has
/// uses. Rewiring the consumer's `outs` away from the shared empty *here*,
/// before `growMatmulInit` runs, severs that second use unconditionally
/// (whether or not it was actually the shared empty), so the later erase is
/// always safe.
static void growElementwiseConsumerInit(IRRewriter &rewriter, MatmulInfo &info,
                                        int64_t mPad, int64_t nPad) {
  linalg::GenericOp consumer = info.elementwiseConsumer;
  Location loc = consumer.getLoc();
  Type elemType =
      cast<RankedTensorType>(consumer.getResult(0).getType()).getElementType();
  rewriter.setInsertionPoint(consumer);
  Value newInit = rewriter.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{mPad, nPad}, elemType);
  consumer.setDpsInitOperand(0, newInit);
  consumer.getResult(0).setType(RankedTensorType::get({mPad, nPad}, elemType));
}

/// Grows the matmul's accumulator and result to [mPad, nPad]. Handles both
/// accumulator forms (see MatmulInfo's comment):
///   * fill/empty: replaces the old `fill(empty)` zero-init with a fresh one
///     of the padded shape (the old fill's constant is preserved), the same
///     way `growBinding` grows LHS/RHS.
///   * accLoad/accArg (accumulator-as-bias): grows the binding/load exactly
///     like `growBinding` does for LHS/RHS. The caller is responsible for
///     zero-padding this argument's host value to match (see the `accArg`
///     handling in `runOnOperation`) -- the extra M/N rows/cols are never
///     read back (the grown result is cropped to the original [m, n]
///     afterward), so their padded value is immaterial.
/// When a fused elementwise consumer is present, the caller must have already
/// called `growElementwiseConsumerInit` first (see its comment) -- by the time
/// this function erases the old `fill`/`empty` (fill/empty form; the accLoad
/// form's `growBinding` call replaces *all* uses of the old load instead, so
/// it has no equivalent ordering requirement), any use of them from the
/// consumer must already be severed, or the erase fails.
static void growMatmulInit(IRRewriter &rewriter, MatmulInfo &info, int64_t mPad,
                           int64_t nPad) {
  Type elemType =
      cast<RankedTensorType>(info.matmul.getResult(0).getType()).getElementType();
  if (info.fill) {
    Location loc = info.fill.getLoc();
    rewriter.setInsertionPoint(info.empty);
    Value newEmpty = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{mPad, nPad}, elemType);
    rewriter.setInsertionPoint(info.fill);
    Value cst = info.fill.getInputs()[0];
    auto newFill = rewriter.create<linalg::FillOp>(loc, ValueRange{cst},
                                                    ValueRange{newEmpty});
    info.matmul.setDpsInitOperand(0, newFill.getResult(0));
    rewriter.eraseOp(info.fill);
    rewriter.eraseOp(info.empty);
  } else {
    growBinding(rewriter, info.accArg, info.accLoad, {mPad, nPad});
  }
  info.matmul.getResult(0).setType(RankedTensorType::get({mPad, nPad}, elemType));
}

/// Grows the output binding and rewrites the store to write the full padded
/// [mPad, nPad] result -- the elementwise consumer's grown result when one is
/// present (it must already have been grown by `growElementwiseConsumerInit`
/// + the automatic type update of the matmul result it reads from), or the
/// matmul's own grown result otherwise.
static void growOutputStore(IRRewriter &rewriter, MatmulInfo &info,
                            int64_t mPad, int64_t nPad) {
  auto dtt = cast<IREE::TensorExt::DispatchTensorType>(info.outArg.getType());
  auto newTensorType = RankedTensorType::get(
      {mPad, nPad}, dtt.asRankedTensorType().getElementType());
  info.outArg.setType(IREE::TensorExt::DispatchTensorType::get(dtt.getAccess(),
                                                               newTensorType));
  Value grownResult = info.elementwiseConsumer
                          ? info.elementwiseConsumer.getResult(0)
                          : info.matmul.getResult(0);
  rewriter.setInsertionPoint(info.store);
  rewriter.create<IREE::TensorExt::DispatchTensorStoreOp>(
      info.store.getLoc(), grownResult, info.outArg,
      /*targetDynamicDims=*/ValueRange{});
  rewriter.eraseOp(info.store);
}

/// For a `linalg.matmul` operand `map`, returns the operand-dim position that the
/// given contraction loop dim (M=0, N=1, K=2 -- fixed by linalg.matmul's iterator
/// order; transpose variants only permute the operand maps, not the loop order)
/// occupies. Lets us read each operand's M/N/K extents from its own layout, so a
/// transpose_b matmul (RHS = [N, K], as an ONNX Gemm lowers to) is handled, not
/// just the plain RHS = [K, N] layout.
static unsigned operandDimForLoop(AffineMap map, unsigned loopDim) {
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i)
    if (cast<AffineDimExpr>(map.getResult(i)).getPosition() == loopDim) return i;
  llvm_unreachable("matmul operand map missing expected loop dim");
}

/// Returns a device affinity pinned to a host (non-amd-aie) device global, or
/// null if none is declared. The padding dispatch runs here (the CPU), where the
/// strided insert is trivially codegen-able.
static IREE::HAL::DeviceAffinityAttr getHostAffinity(ModuleOp module) {
  // Ask the shared capability table rather than testing for "not amd-aie":
  // with a third backend declared, the first non-amd-aie device need not be a
  // host at all (an accelerator would be picked, and the padding dispatch would
  // land on a device that cannot codegen it).
  //
  // The table currently recognizes only llvm-cpu as a host, so a host declared
  // with any other backend yields null here and both callers below return
  // having done nothing. That is silent, and the two of them fail differently:
  // the padding pass leaves a dispatch that later fails divisibility, while the
  // split pass leaves a weight DMA that degrades into a serial BD chain and
  // stalls, i.e. a runtime timeout. The right shape is to move the check after
  // the dispatch walk, so "nothing to do" and "something to do but nowhere to
  // put it" can be told apart and only the latter is reported. Not done here
  // because in this build llvm-cpu is the only host backend that can be
  // declared at all (configure.sh builds no other), so the case is currently
  // unreachable rather than merely unlikely.
  return getFirstDeviceWithRole(module, DeviceRole::Host);
}

/// Creates a `flow.executable` + `flow.dispatch` that zero-pads `v` up to
/// `dstShape` (high padding via `tensor.pad`), placed on `hostAffinity`. Returns
/// the padded value, or `v` unchanged if `dstShape` already matches. A dispatch
/// (unlike `flow.tensor.update`, which only copies contiguous outer-dim
/// sub-ranges) performs the strided placement required to pad an inner
/// dimension. `counter` uniquifies the symbol names.
static Value createPaddingDispatch(IRRewriter &rewriter, ModuleOp module,
                                   Value v, ArrayRef<int64_t> dstShape,
                                   IREE::HAL::DeviceAffinityAttr hostAffinity,
                                   int &counter) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  if (dstShape == srcType.getShape()) return v;
  auto dstType = RankedTensorType::get(dstShape, srcType.getElementType());
  auto inBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, srcType);
  auto outBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, dstType);
  std::string idx = std::to_string(counter++);

  // Worker func: load the full input, high-pad it with zeros, store.
  auto funcOp = func::FuncOp::create(
      loc, "pad_dispatch_" + idx,
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value zero = fb.create<arith::ConstantOp>(
      loc, fb.getZeroAttr(srcType.getElementType()));
  Value loaded = fb.create<IREE::TensorExt::DispatchTensorLoadOp>(
      loc, srcType, entry->getArgument(0), /*sourceDynamicDims=*/ValueRange{});
  Value padded = tensor::createPadHighOp(dstType, loaded, zero,
                                         /*nofold=*/false, loc, fb);
  fb.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, padded, entry->getArgument(1), /*targetDynamicDims=*/ValueRange{});
  fb.create<func::ReturnOp>(loc);

  // Executable wrapping the func, with a default workgroup-count region.
  OpBuilder mb(&module.getBody()->back());
  auto exeOp =
      IREE::Flow::ExecutableOp::create(mb, loc, "pad_executable_" + idx);
  exeOp.setPrivate();
  OpBuilder exeBuilder = OpBuilder::atBlockBegin(&exeOp.getBlock());
  auto innerModule = mlir::ModuleOp::create(exeBuilder, loc);
  innerModule.push_back(funcOp);
  OpBuilder exportBuilder(exeOp.getBody());
  auto exportOp = IREE::Flow::ExecutableExportOp::create(
      exportBuilder, loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *wcBlock = &exportOp.getWorkgroupCount().emplaceBlock();
  OpBuilder wb = OpBuilder::atBlockBegin(wcBlock);
  Type indexTy = wb.getIndexType();
  auto countOp = wb.create<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
      loc, TypeRange{indexTy, indexTy, indexTy}, /*ordinal_operands=*/ValueRange{});
  wb.create<IREE::Flow::ReturnOp>(loc, countOp.getResults());

  // Host dispatch, pinned to the host device.
  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{}, /*arguments=*/ValueRange{v},
      /*argument_dims=*/ValueRange{}, /*tied_operands=*/ArrayAttr{});
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}

/// Zero-pads the raw contents of `srcAttr` -- read as a 2-D row-major
/// `srcType`, which is what a matmul operand is by the time it gets here --
/// up to `dstType`. Returns {} when the attribute is not in a form this can
/// safely rewrite, in which case the caller falls back to a padding dispatch.
static DenseElementsAttr padConstantRowMajor(DenseElementsAttr srcAttr,
                                             RankedTensorType srcType,
                                             RankedTensorType dstType) {
  // A splat's raw data is a single element, so it cannot be copied row-wise;
  // it is also not a shape real weights take.
  if (srcAttr.isSplat()) return {};
  Type elemType = srcType.getElementType();
  if (!elemType.isIntOrFloat()) return {};
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  // Sub-byte types are bit-packed, so a row is not a whole number of bytes.
  if (bitWidth == 0 || bitWidth % 8 != 0) return {};
  int64_t elemBytes = bitWidth / 8;

  int64_t srcRows = srcType.getDimSize(0), srcCols = srcType.getDimSize(1);
  int64_t dstRows = dstType.getDimSize(0), dstCols = dstType.getDimSize(1);
  ArrayRef<char> src = srcAttr.getRawData();
  // Guard the layout assumption rather than trusting it: if the raw buffer is
  // not exactly the dense row-major extent, this is some encoding we do not
  // understand and must not reinterpret.
  if (src.size() != static_cast<size_t>(srcRows * srcCols * elemBytes))
    return {};

  // Zero bytes are +0.0 for f32/bf16 and 0 for integers, i.e. exactly what
  // `createPaddingDispatch`'s `tensor.pad` writes.
  SmallVector<char> dst(dstRows * dstCols * elemBytes, 0);
  for (int64_t r = 0; r < srcRows; ++r) {
    std::memcpy(dst.data() + r * dstCols * elemBytes,
                src.data() + r * srcCols * elemBytes, srcCols * elemBytes);
  }
  return DenseElementsAttr::getFromRawBuffer(dstType, dst);
}

/// If `v` is a weight -- a constant already hoisted into a global by
/// const-eval, reached only through reshapes -- materializes its zero-padded
/// form as a new constant global here, at compile time, and returns a load of
/// it. Returns {} when that is not possible.
///
/// This is what keeps weight padding off the per-inference path. Const-eval
/// cannot do it for us: it runs inside GlobalOptimization while these paddings
/// are decided two phases later, in the Flow pipeline, and nothing re-runs it.
/// Deferring the work to module load instead (a `util.initializer` that runs
/// the padding dispatch once) does not work either: whether the global is
/// declared mutable or not, stream lifetime refinement settles on
/// `!stream.resource<constant>` for a value written once and read thereafter,
/// and the HAL will not let a host dispatch fill a constant-pool buffer
/// (`PERMISSION_DENIED ... requires MAPPING_PERSISTENT`). Computing the padded
/// bytes here sidesteps all of that -- no initializer, no dispatch, and the
/// unpadded global is left dead for SymbolDCE.
static Value createPaddedConstantGlobal(IRRewriter &rewriter, ModuleOp module,
                                        Value v, ArrayRef<int64_t> dstShape,
                                        int &counter) {
  auto srcType = cast<RankedTensorType>(v.getType());
  if (srcType.getRank() != 2 || dstShape.size() != 2) return {};

  // Look through reshapes: a hoisted conv weight arrives as a 4-D global
  // reshaped to the 2-D matmul operand, and a reshape leaves row-major order
  // (and the raw buffer) untouched.
  Value root = v;
  while (auto reshapeOp = root.getDefiningOp<IREE::Flow::TensorReshapeOp>())
    root = reshapeOp.getSource();
  auto loadOp = root.getDefiningOp<IREE::Util::GlobalLoadOp>();
  if (!loadOp || !loadOp.getIsImmutable()) return {};
  auto globalOp = SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
      loadOp, loadOp.getGlobalAttr());
  if (!globalOp) return {};
  auto srcAttr = dyn_cast_or_null<DenseElementsAttr>(
      globalOp.getInitialValueAttr());
  if (!srcAttr) return {};

  auto dstType = RankedTensorType::get(dstShape, srcType.getElementType());
  DenseElementsAttr paddedAttr = padConstantRowMajor(srcAttr, srcType, dstType);
  if (!paddedAttr) return {};

  // Declared at the top of the module, where const-eval puts the weight
  // globals: `UtilToStream` expands a tensor global into a resource + size
  // pair and records it in a map its load pattern then looks up, so a global
  // declared after its uses is converted too late for them.
  Location loc = v.getLoc();
  OpBuilder moduleBuilder = OpBuilder::atBlockBegin(module.getBody());
  auto paddedGlobalOp = IREE::Util::GlobalOp::create(
      moduleBuilder, loc, "__padded_const_" + std::to_string(counter++),
      /*isMutable=*/false, dstType, std::optional<TypedAttr>{paddedAttr});
  SymbolTable::setSymbolVisibility(paddedGlobalOp,
                                   SymbolTable::Visibility::Private);
  if (auto affinity = globalOp->getAttr("stream.affinity.default"))
    paddedGlobalOp->setAttr("stream.affinity.default", affinity);

  return paddedGlobalOp.createLoadOp(loc, rewriter)->getResult(0);
}

/// Pads `v` up to `dstShape`, preferring to fold the padding into a new
/// constant when `v` is a weight (see `createPaddedConstantGlobal`) and
/// falling back to a host dispatch for real runtime values.
static Value createPaddingDispatchOrConstant(
    IRRewriter &rewriter, ModuleOp module, Value v, ArrayRef<int64_t> dstShape,
    IREE::HAL::DeviceAffinityAttr hostAffinity, int &counter) {
  auto srcType = cast<RankedTensorType>(v.getType());
  if (dstShape == srcType.getShape()) return v;
  if (Value folded =
          createPaddedConstantGlobal(rewriter, module, v, dstShape, counter))
    return folded;
  return createPaddingDispatch(rewriter, module, v, dstShape, hostAffinity,
                               counter);
}

/// Creates a `flow.executable` + `flow.dispatch` that crops `v` down to
/// `dstShape` (low corner via `tensor.extract_slice`), placed on `hostAffinity`.
/// Returns the cropped value, or `v` unchanged if `dstShape` already matches.
/// This mirrors `createPaddingDispatch`: a dispatch (unlike `flow.tensor.slice`,
/// which only copies contiguous outer-dim sub-ranges) performs the strided
/// extraction required to crop an inner dimension, so both M (outer) and N
/// (inner) output dims are handled uniformly. `counter` uniquifies the symbol
/// names.
static Value createCropDispatch(IRRewriter &rewriter, ModuleOp module, Value v,
                                ArrayRef<int64_t> dstShape,
                                IREE::HAL::DeviceAffinityAttr hostAffinity,
                                int &counter) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  if (dstShape == srcType.getShape()) return v;
  auto dstType = RankedTensorType::get(dstShape, srcType.getElementType());
  auto inBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, srcType);
  auto outBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, dstType);
  std::string idx = std::to_string(counter++);

  // Worker func: load the full padded input, extract the low-corner slice,
  // store.
  auto funcOp = func::FuncOp::create(
      loc, "crop_dispatch_" + idx,
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value loaded = fb.create<IREE::TensorExt::DispatchTensorLoadOp>(
      loc, srcType, entry->getArgument(0), /*sourceDynamicDims=*/ValueRange{});
  SmallVector<OpFoldResult> offsets(dstShape.size(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(dstShape.size(), rewriter.getIndexAttr(1));
  for (int64_t s : dstShape) sizes.push_back(rewriter.getIndexAttr(s));
  Value cropped = fb.create<tensor::ExtractSliceOp>(loc, dstType, loaded,
                                                    offsets, sizes, strides);
  fb.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, cropped, entry->getArgument(1), /*targetDynamicDims=*/ValueRange{});
  fb.create<func::ReturnOp>(loc);

  // Executable wrapping the func, with a default workgroup-count region.
  OpBuilder mb(&module.getBody()->back());
  auto exeOp =
      IREE::Flow::ExecutableOp::create(mb, loc, "crop_executable_" + idx);
  exeOp.setPrivate();
  OpBuilder exeBuilder = OpBuilder::atBlockBegin(&exeOp.getBlock());
  auto innerModule = mlir::ModuleOp::create(exeBuilder, loc);
  innerModule.push_back(funcOp);
  OpBuilder exportBuilder(exeOp.getBody());
  auto exportOp = IREE::Flow::ExecutableExportOp::create(
      exportBuilder, loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *wcBlock = &exportOp.getWorkgroupCount().emplaceBlock();
  OpBuilder wb = OpBuilder::atBlockBegin(wcBlock);
  Type indexTy = wb.getIndexType();
  auto countOp = wb.create<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
      loc, TypeRange{indexTy, indexTy, indexTy}, /*ordinal_operands=*/ValueRange{});
  wb.create<IREE::Flow::ReturnOp>(loc, countOp.getResults());

  // Host dispatch, pinned to the host device.
  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{}, /*arguments=*/ValueRange{v},
      /*argument_dims=*/ValueRange{}, /*tied_operands=*/ArrayAttr{});
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}

/// Creates a `flow.executable` + `flow.dispatch` on `hostAffinity` that
/// concatenates `chunks` (each `[M, chunkN]`) along the inner N dim into a single
/// `[M, N]` result (chunk i placed at column offset `i*chunkN`). Inverse of
/// `createCropDispatch`: assembles via `tensor.insert_slice` into a
/// `tensor.empty`. All chunks must share the static shape `[M, chunkN]`.
static Value createConcatDispatch(IRRewriter &rewriter, ModuleOp module,
                                  ArrayRef<Value> chunks,
                                  ArrayRef<int64_t> dstShape, int64_t chunkN,
                                  IREE::HAL::DeviceAffinityAttr hostAffinity,
                                  int &counter) {
  Location loc = chunks.front().getLoc();
  Type elemType = cast<RankedTensorType>(chunks.front().getType()).getElementType();
  auto dstType = RankedTensorType::get(dstShape, elemType);
  std::string idx = std::to_string(counter++);

  // Bindings: one ReadOnly per chunk + one WriteOnly output.
  SmallVector<Type> bindingTypes;
  for (Value c : chunks)
    bindingTypes.push_back(IREE::TensorExt::DispatchTensorType::get(
        IREE::TensorExt::TensorAccess::ReadOnly,
        cast<RankedTensorType>(c.getType())));
  bindingTypes.push_back(IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, dstType));

  // Worker func: load each chunk, insert it into an empty [M,N] at its column
  // offset, store the assembled result.
  auto funcOp = func::FuncOp::create(loc, "concat_dispatch_" + idx,
                                     rewriter.getFunctionType(bindingTypes, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value acc = fb.create<tensor::EmptyOp>(loc, dstShape, elemType);
  int64_t M = dstShape[0];
  SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(M),
                                  rewriter.getIndexAttr(chunkN)};
  SmallVector<OpFoldResult> strides(2, rewriter.getIndexAttr(1));
  for (unsigned i = 0, e = chunks.size(); i < e; ++i) {
    Value loaded = fb.create<IREE::TensorExt::DispatchTensorLoadOp>(
        loc, cast<RankedTensorType>(chunks[i].getType()), entry->getArgument(i),
        /*sourceDynamicDims=*/ValueRange{});
    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0),
                                      rewriter.getIndexAttr(i * chunkN)};
    acc = fb.create<tensor::InsertSliceOp>(loc, loaded, acc, offsets, sizes,
                                           strides);
  }
  fb.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, acc, entry->getArgument(chunks.size()),
      /*targetDynamicDims=*/ValueRange{});
  fb.create<func::ReturnOp>(loc);

  // Executable wrapping the func, with a default workgroup-count region.
  OpBuilder mb(&module.getBody()->back());
  auto exeOp =
      IREE::Flow::ExecutableOp::create(mb, loc, "concat_executable_" + idx);
  exeOp.setPrivate();
  OpBuilder exeBuilder = OpBuilder::atBlockBegin(&exeOp.getBlock());
  auto innerModule = mlir::ModuleOp::create(exeBuilder, loc);
  innerModule.push_back(funcOp);
  OpBuilder exportBuilder(exeOp.getBody());
  auto exportOp = IREE::Flow::ExecutableExportOp::create(
      exportBuilder, loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *wcBlock = &exportOp.getWorkgroupCount().emplaceBlock();
  OpBuilder wb = OpBuilder::atBlockBegin(wcBlock);
  Type indexTy = wb.getIndexType();
  auto countOp = wb.create<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
      loc, TypeRange{indexTy, indexTy, indexTy}, /*ordinal_operands=*/ValueRange{});
  wb.create<IREE::Flow::ReturnOp>(loc, countOp.getResults());

  // Host dispatch consuming all chunk values.
  SmallVector<Value> operands(chunks.begin(), chunks.end());
  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{}, /*arguments=*/operands,
      /*argument_dims=*/ValueRange{}, /*tied_operands=*/ArrayAttr{});
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}

/// Largest chunk size to split N into: the largest multiple of `nTile` that
/// divides `n` and is <= `nMax`. `n` is already a multiple of `nTile` (padded),
/// so `nTile` itself always qualifies -> a valid even split always exists.
static int64_t chooseChunkN(int64_t n, int64_t nTile, int64_t nMax) {
  int64_t start = std::min(nMax, n) / nTile * nTile;
  for (int64_t c = start; c >= nTile; c -= nTile)
    if (n % c == 0) return c;
  return nTile;
}

class AMDAIEPadContractionDispatchesPass
    : public impl::AMDAIEPadContractionDispatchesBase<
          AMDAIEPadContractionDispatchesPass> {
 public:
  void runOnOperation() override;
};

void AMDAIEPadContractionDispatchesPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());

  SmallVector<IREE::Flow::DispatchOp> dispatches;
  module.walk([&](IREE::Flow::DispatchOp d) { dispatches.push_back(d); });

  IREE::HAL::DeviceAffinityAttr hostAffinity = getHostAffinity(module);
  if (!hostAffinity) return;  // need a host device to place the padding dispatch
  int counter = 0;

  // Executables rewritten already (shared by multiple dispatches): rewrite the
  // executable once, but pad/crop each caller's operands/result.
  DenseSet<Operation *> paddedExecutables;

  for (IREE::Flow::DispatchOp dispatch : dispatches) {
    IREE::HAL::ExecutableTargetAttr target = getDispatchTarget(dispatch, module);
    if (!target) continue;

    func::FuncOp func = getDispatchFunc(dispatch, module);
    if (!func) continue;
    std::optional<MatmulInfo> info = getMatmulInfo(func);
    if (!info) continue;

    int64_t lhsArgNo = info->lhsArg.getArgNumber();
    int64_t rhsArgNo = info->rhsArg.getArgNumber();

    // Read the logical M/N/K from this caller's operands (robust to a shared
    // executable whose bindings a previous caller already padded). Locate each
    // operand's M/N/K dim from the matmul's indexing maps rather than assuming
    // LHS=[M,K]/RHS=[K,N], so transpose_b (RHS=[N,K], from an ONNX Gemm) reads N
    // from the correct dim instead of mistaking K for N.
    auto lhsType =
        cast<RankedTensorType>(dispatch.getArguments()[lhsArgNo].getType());
    auto rhsType =
        cast<RankedTensorType>(dispatch.getArguments()[rhsArgNo].getType());
    SmallVector<AffineMap> maps = info->matmul.getIndexingMapsArray();
    unsigned lhsMPos = operandDimForLoop(maps[0], /*M=*/0);
    unsigned lhsKPos = operandDimForLoop(maps[0], /*K=*/2);
    unsigned rhsNPos = operandDimForLoop(maps[1], /*N=*/1);
    unsigned rhsKPos = operandDimForLoop(maps[1], /*K=*/2);
    int64_t m = lhsType.getShape()[lhsMPos];
    int64_t k = lhsType.getShape()[lhsKPos];
    int64_t n = rhsType.getShape()[rhsNPos];
    // The matmul's own (accumulator) element type -- used only to pick
    // instruction-size-driven padding multiples below. Kept distinct from
    // `outElemType` (the dispatch's actual output-binding element type,
    // resolved below the `padOut` check): a fused elementwise consumer may
    // legitimately narrow the type between the two (e.g. an f32 accumulator
    // truncated to bf16 for the next layer), and using the accumulator's
    // type for the dispatch's own result type in that case would be wrong.
    Type accElemType = cast<RankedTensorType>(info->matmul.getResult(0).getType())
                            .getElementType();

    std::optional<PaddingMultiples> mult = getPaddingMultiples(
        target, lhsType.getElementType(), rhsType.getElementType(),
        accElemType);
    if (!mult) continue;

    int64_t mPad = roundUpToMultiple(m, mult->m);
    int64_t nPad = roundUpToMultiple(n, mult->n);
    int64_t kPad = roundUpToMultiple(k, mult->k);
    if (mPad == m && nPad == n && kPad == k) continue;  // already divisible
    // Any output dim (M outer and/or N inner) may be padded; the grown result is
    // cropped back on the host by a single extract_slice dispatch.
    bool padOut = mPad != m || nPad != n;
    // Growing the output side requires the matmul result to be stored either
    // directly, or through exactly one recognized shape-preserving unary
    // elementwise consumer (see MatmulInfo's comment): store/outArg are only
    // resolved in those two cases. When the store's value is anything else
    // (a reduction-shaped consumer, a consumer with more than one real
    // input, or a chain of more than one intervening op) and M and/or N also
    // need padding, this pass doesn't know how to grow the output through
    // that unrecognized consumer, so skip the dispatch entirely -- same
    // conservative behavior as before for that harder case. K-only padding
    // (padOut == false) never touches the matmul's output at all, so it
    // remains safe and is handled below regardless of store/outArg.
    if (padOut && (!info->store || !info->outArg)) continue;

    // Padded operand shapes placed at each operand's own M/N/K dim positions, so
    // the RHS layout (plain [K,N] or transpose_b [N,K]) is preserved.
    SmallVector<int64_t, 2> lhsPad(2), rhsPad(2);
    lhsPad[lhsMPos] = mPad;
    lhsPad[lhsKPos] = kPad;
    rhsPad[rhsNPos] = nPad;
    rhsPad[rhsKPos] = kPad;

    // Rewrite the executable once per shared executable: grow the LHS and RHS
    // bindings; when an output dim is padded grow the output init, matmul result
    // and store binding too (the dispatch stays fully divisible, cropped on
    // host). The output is always [M,N], so its grown shape is [mPad, nPad].
    if (paddedExecutables.insert(func.getOperation()).second) {
      growBinding(rewriter, info->lhsArg, info->lhsLoad, lhsPad);
      growBinding(rewriter, info->rhsArg, info->rhsLoad, rhsPad);
      if (padOut) {
        // Order matters: sever the elementwise consumer's use of the
        // (possibly shared) old empty/fill *before* growMatmulInit erases
        // them -- see growElementwiseConsumerInit's comment.
        if (info->elementwiseConsumer)
          growElementwiseConsumerInit(rewriter, *info, mPad, nPad);
        growMatmulInit(rewriter, *info, mPad, nPad);
        growOutputStore(rewriter, *info, mPad, nPad);
      }
      SmallVector<Type> argTypes(llvm::map_range(
          func.getArguments(), [](BlockArgument a) { return a.getType(); }));
      func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
    }

    // Pad this caller's host operands to the padded shapes, then rewire. A
    // weight operand is a constant, so its padding is folded into a new
    // constant here instead of costing a dispatch on every inference; the
    // activation LHS is a runtime value and keeps its dispatch.
    rewriter.setInsertionPoint(dispatch);
    Value lhsPadded = createPaddingDispatchOrConstant(
        rewriter, module, dispatch.getArguments()[lhsArgNo], lhsPad,
        hostAffinity, counter);
    Value rhsPadded = createPaddingDispatchOrConstant(
        rewriter, module, dispatch.getArguments()[rhsArgNo], rhsPad,
        hostAffinity, counter);
    dispatch.getArgumentsMutable().slice(lhsArgNo, 1).assign(lhsPadded);
    dispatch.getArgumentsMutable().slice(rhsArgNo, 1).assign(rhsPadded);

    // Grow this caller's result to [mPad, nPad] and crop it back to [m, n] on
    // the host with an extract_slice dispatch (handles the inner N dim too).
    if (padOut) {
      // Accumulator-as-bias form: the executable rewrite above already grew
      // the accArg binding itself (via growMatmulInit -> growBinding), so
      // this caller's host-side bias value must be grown to match, the same
      // way LHS/RHS were above. Only reachable when padOut (M and/or N need
      // padding) -- when padOut is false, growMatmulInit was never called
      // above and this argument's binding is untouched, so it must not be
      // padded here either.
      if (info->accArg) {
        int64_t accArgNo = info->accArg.getArgNumber();
        Value accPadded = createPaddingDispatch(
            rewriter, module, dispatch.getArguments()[accArgNo], {mPad, nPad},
            hostAffinity, counter);
        dispatch.getArgumentsMutable().slice(accArgNo, 1).assign(accPadded);
      }
      // The dispatch's own result type: the output binding's element type
      // (`info->outArg`, non-null here since `padOut` was checked above),
      // which is `accElemType` in the common direct-store case but may
      // differ when a fused consumer narrows the type (see the comment
      // where `accElemType` is computed above).
      Type outElemType = cast<IREE::TensorExt::DispatchTensorType>(
                              info->outArg.getType())
                              .asRankedTensorType()
                              .getElementType();
      Value result = dispatch.getResult(0);
      result.setType(RankedTensorType::get({mPad, nPad}, outElemType));
      rewriter.setInsertionPointAfter(dispatch);
      Value cropped =
          createCropDispatch(rewriter, module, result, {m, n}, hostAffinity,
                             counter);
      result.replaceAllUsesExcept(cropped, cropped.getDefiningOp());
    }
  }
}

/// Splits a large-(K,N) transpose_b NPU matmul dispatch (RHS = [N,K], N outer)
/// into `N/C` separate dispatches, each computing an N-slice [C,K] of the weight
/// into [M,C], concatenated back to [M,N] on the host. Runs after
/// `AMDAIEPadContractionDispatches` so operands are already tile-divisible.
/// A single large-N weight L3->L2 shim DMA degrades into a serial BD chain
/// (exceeding the shim's 3-intra-dim / 63-iteration limits) and stalls; the
/// smaller per-chunk weight DMA folds within those limits, so each sub-dispatch
/// runs fast.
class AMDAIESplitLargeContractionDispatchesPass
    : public impl::AMDAIESplitLargeContractionDispatchesBase<
          AMDAIESplitLargeContractionDispatchesPass> {
 public:
  using AMDAIESplitLargeContractionDispatchesBase::
      AMDAIESplitLargeContractionDispatchesBase;
  void runOnOperation() override;
};

void AMDAIESplitLargeContractionDispatchesPass::runOnOperation() {
  // Empirical trigger: the cliff is combinatorial, so split only large FC
  // weights (large reduction K and large output N) and leave conv / small
  // matmuls untouched.
  //
  // These two numbers were found by experiment, not derived, and that has three
  // consequences worth knowing before reusing them:
  //
  //  - kKThreshold was validated on exactly one dispatch: VGG-16's dense0
  //    (K=25088, N=4096) in bf16 on npu4. No other element type or device was
  //    measured. `nChunkMax`'s default has since been bisected on real
  //    hardware (docs/2026-09-09_split-n-chunk-max-experimental-bisection.md):
  //    1024 is safe (7/7 correct runs), 2048 hangs reproducibly (2/2), so the
  //    default is 1024 and 2048 is the known upper wall. Note that the
  //    "K == kKThreshold, so it does not split" case
  //    the threshold was originally calibrated against no longer occurs in
  //    VGG-16: AMDAIEFoldBiasIntoContraction absorbs the bias as a K+1
  //    reduction element, so dense1/dense2 arrive here at K=4224 (4096 -> 4097,
  //    then alignment padding) rather than K=4096, and both now split (8-way
  //    and 2-way) even though they ran fine unsplit before the fold. That is an
  //    accident of this empirical threshold, not a DMA requirement -- see
  //    docs/2026-09-07_dev-vs-fold-dispatch-dump-comparison.md section 1.
  //  - kKThreshold is compared in ELEMENTS, while the shim limit it stands in
  //    for is a limit on words. An f32 weight at K=2048 has the same byte
  //    footprint as a bf16 weight at the K=4096 boundary, but sits even further
  //    below the threshold, so element types other than bf16 are compared
  //    against the wrong number.
  //  - Getting it wrong does not produce wrong numbers. The weight DMA degrades
  //    into a serial BD chain and stalls, so the symptom is a runtime timeout
  //    (ert state 6), which is easy to misattribute.
  //
  // The principled replacement is available today and should be preferred when
  // this is revisited: DmaDimConfig (Utils/AMDAIEDmaUtils.h) exposes the real
  // per-device limits (nbIntraDims from AMDAIEDmaProp::NumAddrDim, getMaxSizes,
  // getMaxStrides), so the trigger can ask "does this weight's L3->L2 access
  // pattern still fold within maxNbDims?" in bytes, for whatever device and
  // element type the dispatch actually targets -- `target` is already resolved
  // per dispatch below.
  constexpr int64_t kKThreshold = 4096;  // split only when reduction K exceeds this
  // max N per chunk (validated good chunk at the 1024 default; see the
  // `nChunkMax` option's own doc in Passes.td for why any override needs
  // real-hardware re-verification before being trusted).
  int64_t kNChunkMax = nChunkMax;

  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());
  IREE::HAL::DeviceAffinityAttr hostAffinity = getHostAffinity(module);
  if (!hostAffinity) return;  // need a host device for the concat dispatch
  int counter = 0;

  SmallVector<IREE::Flow::DispatchOp> dispatches;
  module.walk([&](IREE::Flow::DispatchOp d) { dispatches.push_back(d); });

  for (IREE::Flow::DispatchOp dispatch : dispatches) {
    IREE::HAL::ExecutableTargetAttr target = getDispatchTarget(dispatch, module);
    if (!target) continue;
    func::FuncOp func = getDispatchFunc(dispatch, module);
    if (!func) continue;
    std::optional<MatmulInfo> info = getMatmulInfo(func);
    if (!info) continue;
    // This pass always splits/grows the output (each N-chunk gets its own
    // output binding via growOutputStore below), which requires the matmul
    // result to be stored either directly, or through exactly one recognized
    // shape-preserving unary elementwise consumer (see MatmulInfo's
    // comment) -- e.g. a fused ReLU, which is exactly the VGG16 FC-layer
    // shape this pass exists for once elementwise fusion into contraction
    // roots isn't disabled by a workaround flag. Skip a dispatch whose store
    // is indirect through anything else (a reduction-shaped consumer, a
    // consumer with more than one real input, or a chain of more than one
    // intervening op) -- unlike AMDAIEPadContractionDispatches, there is no
    // K-only-safe fallback here.
    //
    // When a recognized consumer IS present, each per-chunk clone below
    // grows the consumer's own init/result alongside the matmul (via
    // growElementwiseConsumerInit, the same ordering
    // AMDAIEPadContractionDispatches uses), and growOutputStore wires the
    // consumer's (not the matmul's) chunk-sized result into the chunk output
    // binding -- `isShapePreservingUnaryMapOf` structurally guarantees the
    // consumer is a per-position map, so it commutes with the N-split:
    // `f(concat(chunks)) == concat(f(chunk))`.
    //
    // `growMatmulInit` below is always called (every N-chunk gets its own
    // [M, C] accumulator), and for the accumulator-as-bias form
    // (`!info->fill`, see MatmulInfo's comment) that means growing (padding)
    // a *slice* of the bias to [M, C] per chunk -- this pass has no logic for
    // that (unlike AMDAIEPadContractionDispatches's accArg handling, which
    // only ever pads the *whole* accumulator once), so `!info->fill` is
    // excluded here too.
    if (!info->store || !info->outArg || !info->fill) continue;

    // Only split when N is the RHS *outer* dim (transpose_b [N,K]), so the weight
    // N-slice is a contiguous outer-dim slice.
    SmallVector<AffineMap> maps = info->matmul.getIndexingMapsArray();
    unsigned rhsNPos = operandDimForLoop(maps[1], /*N=*/1);
    unsigned rhsKPos = operandDimForLoop(maps[1], /*K=*/2);
    if (rhsNPos != 0) continue;

    int64_t lhsArgNo = info->lhsArg.getArgNumber();
    int64_t rhsArgNo = info->rhsArg.getArgNumber();
    auto rhsType =
        cast<RankedTensorType>(dispatch.getArguments()[rhsArgNo].getType());
    int64_t N = rhsType.getShape()[rhsNPos];
    int64_t K = rhsType.getShape()[rhsKPos];
    if (K <= kKThreshold || N <= kNChunkMax) continue;

    auto resultType = cast<RankedTensorType>(info->matmul.getResult(0).getType());
    int64_t M = resultType.getShape()[0];  // matmul output is [M, N]
    // The matmul's own (accumulator) element type -- used only to pick the
    // instruction-size-driven chunk size below. Kept distinct from
    // `outElemType` (the actual output-binding element type, used for the
    // chunk/concat result types further down): a fused elementwise consumer
    // may legitimately narrow the type between the two (e.g. an f32
    // accumulator truncated to bf16 for the next layer).
    Type accElemType = resultType.getElementType();
    Type outElemType = cast<IREE::TensorExt::DispatchTensorType>(
                            info->outArg.getType())
                            .asRankedTensorType()
                            .getElementType();

    // Even, tile-aligned chunk size (N is already a multiple of the N tile).
    std::optional<PaddingMultiples> mult = getPaddingMultiples(
        target,
        cast<RankedTensorType>(dispatch.getArguments()[lhsArgNo].getType())
            .getElementType(),
        rhsType.getElementType(), accElemType);
    if (!mult) continue;
    int64_t C = chooseChunkN(N, mult->n, kNChunkMax);
    int64_t nChunks = N / C;
    if (nChunks < 2) continue;

    // Resolve the original matmul executable to clone per chunk. Each chunk gets
    // its OWN executable (not shared): with a shared executable the per-chunk
    // weight offset would be hoisted to a dynamic push-constant that the NPU DMA
    // lowering cannot recover. (Relies on executable dedup being off, which the
    // amd-aie recipe already requires for the same offset reason.)
    SymbolRefAttr entryPoint = *dispatch.getEntryPointRefs().begin();
    auto origExe = module.lookupSymbol<IREE::Flow::ExecutableOp>(
        entryPoint.getRootReference());
    if (!origExe) continue;
    StringAttr funcName = entryPoint.getLeafReference();
    Attribute npuAffinity = dispatch->getAttr("stream.affinity");
    Value weight = dispatch.getArguments()[rhsArgNo];
    Location loc = dispatch.getLoc();
    auto chunkResultType = RankedTensorType::get({M, C}, outElemType);
    SmallVector<int64_t, 2> rhsChunk(2);
    rhsChunk[rhsNPos] = C;
    rhsChunk[rhsKPos] = K;

    SmallVector<Value> chunkOutputs;
    bool allChunksOk = true;
    for (int64_t i = 0; i < nChunks; ++i) {
      rewriter.setInsertionPoint(dispatch);
      // View the weight N-slice [i*C:(i+1)*C, :] (an outer-dim contiguous slice).
      // Each chunk feeds its OWN cloned executable (below), so this slice's
      // static base offset stays a per-executable constant that the NPU DMA
      // lowering recovers -- no host materialization needed.
      Value cOff = rewriter.create<arith::ConstantIndexOp>(loc, i * C);
      Value cZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value cC = rewriter.create<arith::ConstantIndexOp>(loc, C);
      Value cK = rewriter.create<arith::ConstantIndexOp>(loc, K);
      Value wSlice = rewriter.create<IREE::Flow::TensorSliceOp>(
          loc, RankedTensorType::get({C, K}, rhsType.getElementType()), weight,
          /*source_dims=*/ValueRange{}, /*start_indices=*/ValueRange{cOff, cZero},
          /*lengths=*/ValueRange{cC, cK}, /*result_dims=*/ValueRange{});
      // Clone the matmul executable and shrink the clone to [C,K] -> [M,C].
      int id = counter++;
      OpBuilder mb(&module.getBody()->back());
      auto cloneExe = cast<IREE::Flow::ExecutableOp>(mb.clone(*origExe));
      cloneExe.setSymName((origExe.getSymName() + "_ns" + std::to_string(id)).str());
      func::FuncOp cloneFunc =
          cloneExe.getInnerModule().lookupSymbol<func::FuncOp>(funcName);
      auto cloneExport =
          *cloneExe.getOps<IREE::Flow::ExecutableExportOp>().begin();
      // Rename the inner export + func uniquely: executable linking flattens the
      // inner funcs into one module, so identical clone symbols would collide.
      std::string newName =
          (funcName.getValue() + "_ns" + std::to_string(id)).str();
      cloneExport.setSymName(newName);
      cloneExport.setFunctionRef(newName);
      cloneFunc.setName(newName);
      std::optional<MatmulInfo> ci = getMatmulInfo(cloneFunc);
      if (!ci) {
        // Should not happen: `cloneFunc` is a structural copy of `func`,
        // which already passed this same `getMatmulInfo` call above. Bail
        // out on the whole dispatch rather than risk assembling a concat
        // from an incomplete set of chunks (silently wrong result). Leaves
        // this iteration's already-created (unused) `wSlice` and `cloneExe`
        // as dead IR; not worth the extra bookkeeping to clean up a path
        // that should be unreachable.
        allChunksOk = false;
        break;
      }
      // Order matters: sever the elementwise consumer's use of the
      // (possibly shared) old empty/fill *before* growMatmulInit erases
      // them -- see growElementwiseConsumerInit's comment.
      if (ci->elementwiseConsumer)
        growElementwiseConsumerInit(rewriter, *ci, M, C);
      growBinding(rewriter, ci->rhsArg, ci->rhsLoad, rhsChunk);
      growMatmulInit(rewriter, *ci, M, C);
      growOutputStore(rewriter, *ci, M, C);
      SmallVector<Type> argTypes(llvm::map_range(
          cloneFunc.getArguments(),
          [](BlockArgument a) { return a.getType(); }));
      cloneFunc.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
      SmallVector<Value> operands(dispatch.getArguments().begin(),
                                  dispatch.getArguments().end());
      operands[rhsArgNo] = wSlice;
      rewriter.setInsertionPoint(dispatch);
      auto sub = IREE::Flow::DispatchOp::create(
          rewriter, loc, cloneExport, /*workload=*/dispatch.getWorkload(),
          TypeRange{chunkResultType}, /*result_dims=*/ValueRange{}, operands,
          /*argument_dims=*/ValueRange{}, /*tied_operands=*/ArrayAttr{});
      if (npuAffinity) sub->setAttr("stream.affinity", npuAffinity);
      chunkOutputs.push_back(sub.getResult(0));
    }
    if (!allChunksOk) continue;
    rewriter.setInsertionPointAfter(dispatch);
    Value concatenated = createConcatDispatch(rewriter, module, chunkOutputs,
                                              {M, N}, C, hostAffinity, counter);
    dispatch.getResult(0).replaceAllUsesWith(concatenated);
    rewriter.eraseOp(dispatch);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPadContractionDispatchesPass() {
  return std::make_unique<AMDAIEPadContractionDispatchesPass>();
}

std::unique_ptr<Pass> createAMDAIESplitLargeContractionDispatchesPass(
    AMDAIESplitLargeContractionDispatchesOptions options) {
  return std::make_unique<AMDAIESplitLargeContractionDispatchesPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
