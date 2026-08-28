// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Two Flow-phase passes that pad NPU dispatches at the dispatch boundary:
// AMDAIEPadContractionDispatches (contractions to tile multiples) and
// AMDAIEPadPoolingDispatches (max pooling OH, at the bottom of this file). They
// share the dispatch helpers and host pad/crop dispatch builders.
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

#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDevicePlacementUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
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
/// from the shared pack-peel reduction tile. If the matmul is lowered to a
/// ukernel M/N are doubled, because the matmul ukernels process 2x2 vector
/// instructions per step and need an even count per core (KernelDispatch picks
/// the per-core tile to match).
///
/// An M == 1 matmul the GEMV pipeline takes (KernelDispatch, same predicate)
/// has its own multiples: M is left untiled, N is spread over every core in
/// whole multiples of `kGemvNPerCoreMultiple` outputs when the target allows
/// it (`gemv_pad_n`), and K is a multiple of the matvec ukernel's vector when
/// the tile is lowered to it. Returns nullopt if `target` is not a recognized
/// amd-aie target.
static std::optional<PaddingMultiples> getPaddingMultiples(
    IREE::HAL::ExecutableTargetAttr target, int64_t m, Type lhsElemType,
    Type rhsElemType, Type accElemType, bool transposeB) {
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(target);
  std::optional<int64_t> numRows = getConfigNumRows(target);
  std::optional<int64_t> numCols = getConfigNumColumns(target);
  if (!device || !numRows || !numCols) return std::nullopt;
  if (isGemvShape(m, transposeB)) {
    bool useMatvec = isAie2P(*device) &&
                     hasUkernel(target.getConfiguration(), "matvec") &&
                     hasUKernelTemplate(*device, "matvec",
                                        {lhsElemType, rhsElemType, accElemType},
                                        /*transposeB=*/true);
    return PaddingMultiples{
        /*m=*/1,
        /*n=*/getConfigGemvPadN(target)
            ? *numRows * *numCols * kGemvNPerCoreMultiple
            : 1,
        /*k=*/useMatvec ? kMatvecUKernelK
                        : getPackPeelReductionTile(/*kPackScaleL1=*/1)};
  }
  AMDAIEDeviceModel deviceModel = getDeviceModel(*device);
  FailureOr<std::array<uint32_t, 3>> instr =
      deviceModel.getAIEMatmulInstructionSize(lhsElemType, rhsElemType,
                                              accElemType);
  // Element types without a vector instruction (e.g. f32) are packed by the
  // scalar pack size instead, so pad to that.
  if (failed(instr)) instr = getScalarMatmulPackSize();
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
  bool useUKernel =
      hasUkernel(target.getConfiguration(), "matmul") &&
      hasUKernelTemplate(*device, "matmul",
                         {lhsElemType, rhsElemType, accElemType}, transposeB);
  int64_t perCore = useUKernel ? 2 : 1;
  return PaddingMultiples{
      /*m=*/*numRows * perCore * (*instr)[0],
      /*n=*/*numCols * perCore * (*instr)[1],
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
  // with any other backend yields null here and the callers return having done
  // nothing. That is silent: the unpadded dispatch later fails divisibility.
  // The right shape is to move the check after
  // the dispatch walk, so "nothing to do" and "something to do but nowhere to
  // put it" can be told apart and only the latter is reported. Not done here
  // because in this build llvm-cpu is the only host backend that can be
  // declared at all (configure.sh builds no other), so the case is currently
  // unreachable rather than merely unlikely.
  return getFirstDeviceWithRole(module, DeviceRole::Host);
}

/// First index `i` such that no `pad_executable_i` / `crop_executable_i` /
/// `concat_executable_i` symbol exists yet in `module`: the helper dispatch
/// builders below number their executables from it, so the passes in this
/// file (which run back to back on the same module) never collide.
static int firstFreeHelperIndex(ModuleOp module) {
  int next = 0;
  for (Operation &op : module.getBody()->getOperations()) {
    auto sym = dyn_cast<SymbolOpInterface>(&op);
    if (!sym) continue;
    StringRef name = sym.getName();
    for (StringRef prefix :
         {"pad_executable_", "crop_executable_", "concat_executable_"}) {
      if (!name.consume_front(prefix)) continue;
      int idx;
      if (!name.getAsInteger(10, idx)) next = std::max(next, idx + 1);
    }
  }
  return next;
}

/// Creates a `flow.executable` + `flow.dispatch` that zero-pads `v` up to
/// `dstShape` (high padding via `tensor.pad`), placed on `hostAffinity`. Returns
/// the padded value, or `v` unchanged if `dstShape` already matches. A dispatch
/// (unlike `flow.tensor.update`, which only copies contiguous outer-dim
/// sub-ranges) performs the strided placement required to pad an inner
/// dimension. `counter` uniquifies the symbol names.
/// `padValue`, when set, is the constant to pad with instead of 0 (the pooling
/// pass passes its dispatch's init fill value, -inf for max).
static Value createPaddingDispatch(IRRewriter &rewriter, ModuleOp module,
                                   Value v, ArrayRef<int64_t> dstShape,
                                   IREE::HAL::DeviceAffinityAttr hostAffinity,
                                   int &counter, TypedAttr padValue = nullptr) {
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
  Type elemType = srcType.getElementType();
  TypedAttr fillAttr = padValue ? padValue : fb.getZeroAttr(elemType);
  Value zero = fb.create<arith::ConstantOp>(loc, fillAttr);
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
  int counter = firstFreeHelperIndex(module);

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
    Type outElemType =
        cast<RankedTensorType>(info->matmul.getResult(0).getType())
            .getElementType();

    std::optional<PaddingMultiples> mult = getPaddingMultiples(
        target, m, lhsType.getElementType(), rhsType.getElementType(),
        outElemType, /*transposeB=*/rhsNPos < rhsKPos);
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

    // Pad this caller's host operands to the padded shapes, then rewire.
    rewriter.setInsertionPoint(dispatch);
    Value lhsPadded =
        createPaddingDispatch(rewriter, module, dispatch.getArguments()[lhsArgNo],
                              lhsPad, hostAffinity, counter);
    Value rhsPadded =
        createPaddingDispatch(rewriter, module, dispatch.getArguments()[rhsArgNo],
                              rhsPad, hostAffinity, counter);
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

// --- AMDAIEPadPoolingDispatches ---
// Pads an NPU max pooling dispatch's leading spatial output dim OH up to what
// the target's pool column policy asks for (`pool_columns_policy`: `full` ->
// a multiple of the column count; `cost` -> when the model finds the padded
// shape cheaper; `odd`/`max` -> never), so the pool pipeline (which puts OH on
// the array columns) can use every column. Same shape as the contraction pass:
// the input is padded on the host by `stride` rows per extra output row with
// the dispatch's own fill value (-inf for max, so padded rows never win), the
// executable's input binding/load, init (`empty`+`fill`), pooling result and
// output binding/store grow in place, and the dispatch result is cropped back
// on the host. The lowering strategy later sees an OH divisible by the column
// count and, under the same policy, spreads it over all columns.
//
// Done at the Flow boundary rather than in the preprocessing because a fill
// created before dispatch formation can be hoisted out of the dispatch as a
// `flow.tensor.splat` (seen on VGG-16 pool4), which the pool pipeline cannot
// take as an external init; here the fill is already inside the dispatch.

/// A plain max pooling (`empty -> fill -> generic(max) -> store`) inside a
/// dispatch executable, with the ops/bindings needed to grow its OH.
struct PoolInfo {
  linalg::GenericOp pool;
  MaxPoolingDims dims;
  IREE::TensorExt::DispatchTensorLoadOp inLoad;
  BlockArgument inArg;
  linalg::FillOp fill;
  tensor::EmptyOp empty;
  IREE::TensorExt::DispatchTensorStoreOp store;
  BlockArgument outArg;
  unsigned ohOut;  // OH's dim in the output
  unsigned ohIn;   // OH's (strided) dim in the input
};

static std::optional<PoolInfo> getPoolInfo(func::FuncOp func) {
  SmallVector<linalg::GenericOp> pools;
  func.walk([&](linalg::GenericOp op) {
    if (isMaxPoolingOp(op)) pools.push_back(op);
  });
  if (pools.size() != 1) return std::nullopt;
  linalg::GenericOp pool = pools.front();
  FailureOr<MaxPoolingDims> dims = getMaxPoolingDims(pool);
  if (failed(dims)) return std::nullopt;
  // Only an un-split pooling: no non-unit plain dim ahead of the spatial dims
  // (a halo pre-split pooling carries its chunk dim there and is already
  // column-split by the preprocessing).
  auto outType = cast<RankedTensorType>(pool.getResult(0).getType());
  if (!outType.hasStaticShape()) return std::nullopt;
  unsigned ohOut = dims->spatialDims.front();
  for (unsigned p : dims->plainDims)
    if (p < ohOut && outType.getDimSize(p) != 1) return std::nullopt;
  auto inLoad = getFullTensorLoad(pool.getDpsInputs()[0]);
  if (!inLoad) return std::nullopt;
  auto inArg = dyn_cast<BlockArgument>(inLoad.getSource());
  if (!inArg) return std::nullopt;
  auto fill = pool.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill) return std::nullopt;
  if (!fill.getInputs()[0].getDefiningOp<arith::ConstantOp>())
    return std::nullopt;
  auto empty = fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
  if (!empty) return std::nullopt;
  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> stores;
  func.walk(
      [&](IREE::TensorExt::DispatchTensorStoreOp s) { stores.push_back(s); });
  if (stores.size() != 1) return std::nullopt;
  auto store = stores.front();
  if (store.getValue() != pool.getResult(0)) return std::nullopt;
  auto outArg = dyn_cast<BlockArgument>(store.getTarget());
  if (!outArg) return std::nullopt;
  // The input dim that OH's loop reads (`oh * stride + kh`).
  AffineMap inMap = pool.getIndexingMapsArray()[0];
  std::optional<unsigned> ohIn;
  for (unsigned i = 0, e = inMap.getNumResults(); i < e; ++i) {
    bool uses = false;
    inMap.getResult(i).walk([&](AffineExpr e) {
      if (auto d = dyn_cast<AffineDimExpr>(e))
        if (d.getPosition() == ohOut) uses = true;
    });
    if (uses) ohIn = i;
  }
  if (!ohIn) return std::nullopt;
  return PoolInfo{pool,  *dims, inLoad, inArg, fill,
                  empty, store, outArg, ohOut, *ohIn};
}

class AMDAIEPadPoolingDispatchesPass
    : public impl::AMDAIEPadPoolingDispatchesBase<
          AMDAIEPadPoolingDispatchesPass> {
 public:
  void runOnOperation() override;
};

void AMDAIEPadPoolingDispatchesPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());

  SmallVector<IREE::Flow::DispatchOp> dispatches;
  module.walk([&](IREE::Flow::DispatchOp d) { dispatches.push_back(d); });

  IREE::HAL::DeviceAffinityAttr hostAffinity = getHostAffinity(module);
  if (!hostAffinity) return;
  int counter = firstFreeHelperIndex(module);
  DenseSet<Operation *> paddedExecutables;

  for (IREE::Flow::DispatchOp dispatch : dispatches) {
    IREE::HAL::ExecutableTargetAttr target = getDispatchTarget(dispatch, module);
    if (!target) continue;
    std::optional<int64_t> numCols = getConfigNumColumns(target);
    if (!numCols || *numCols <= 1) continue;
    PoolColumnsPolicy policy = getConfigPoolColumnsPolicy(target);
    if (policy == PoolColumnsPolicy::Odd || policy == PoolColumnsPolicy::Max)
      continue;

    func::FuncOp func = getDispatchFunc(dispatch, module);
    if (!func) continue;
    std::optional<PoolInfo> info = getPoolInfo(func);
    if (!info) continue;

    // Read the shapes from this caller's operand / result.
    int64_t inArgNo = info->inArg.getArgNumber();
    auto inType =
        cast<RankedTensorType>(dispatch.getArguments()[inArgNo].getType());
    auto outType = cast<RankedTensorType>(dispatch.getResult(0).getType());
    if (!inType.hasStaticShape() || !outType.hasStaticShape()) continue;
    const MaxPoolingDims &dims = info->dims;
    unsigned ohOut = info->ohOut, owOut = dims.spatialDims[1];
    int64_t oh = outType.getDimSize(ohOut);
    int64_t c = 1;
    for (unsigned p : dims.plainDims) c *= outType.getDimSize(p);
    auto bytesOf = [](Type t) { return t.getIntOrFloatBitWidth() / 8; };
    // The window operand is [kh, kw] in the spatial dims' order; dilation is
    // already folded into the generic's `oh * stride + kh` map.
    auto winType =
        cast<RankedTensorType>(info->pool.getDpsInputs()[1].getType());
    PoolShape shape{/*ow=*/outType.getDimSize(owOut),
                    /*c=*/c,
                    /*windowH=*/winType.getDimSize(0),
                    /*windowW=*/winType.getDimSize(1),
                    /*strideH=*/dims.strides[0],
                    /*strideW=*/dims.strides[1],
                    /*bytesIn=*/bytesOf(inType.getElementType()),
                    /*bytesOut=*/bytesOf(outType.getElementType())};
    PoolColumns pc = choosePoolColumns(oh, *numCols, policy, shape,
                                       /*allowPad=*/true);
    if (pc.ohPadded <= oh) continue;
    int64_t padRows = pc.ohPadded - oh;

    SmallVector<int64_t> inPad(inType.getShape());
    inPad[info->ohIn] += padRows * shape.strideH;
    SmallVector<int64_t> outPad(outType.getShape());
    outPad[ohOut] = pc.ohPadded;
    TypedAttr padValue = cast<TypedAttr>(
        info->fill.getInputs()[0].getDefiningOp<arith::ConstantOp>().getValue());

    // Rewrite the executable once: grow the input binding/load, the init
    // (empty + fill), the pooling result and the output binding/store.
    if (paddedExecutables.insert(func.getOperation()).second) {
      growBinding(rewriter, info->inArg, info->inLoad, inPad);
      Location loc = info->fill.getLoc();
      Type elemType = outType.getElementType();
      rewriter.setInsertionPoint(info->empty);
      Value newEmpty = rewriter.create<tensor::EmptyOp>(loc, outPad, elemType);
      rewriter.setInsertionPoint(info->fill);
      auto newFill = rewriter.create<linalg::FillOp>(
          loc, ValueRange{info->fill.getInputs()[0]}, ValueRange{newEmpty});
      info->pool.setDpsInitOperand(0, newFill.getResult(0));
      info->pool.getResult(0).setType(RankedTensorType::get(outPad, elemType));
      rewriter.eraseOp(info->fill);
      rewriter.eraseOp(info->empty);
      auto dtt =
          cast<IREE::TensorExt::DispatchTensorType>(info->outArg.getType());
      info->outArg.setType(IREE::TensorExt::DispatchTensorType::get(
          dtt.getAccess(), RankedTensorType::get(outPad, elemType)));
      rewriter.setInsertionPoint(info->store);
      rewriter.create<IREE::TensorExt::DispatchTensorStoreOp>(
          info->store.getLoc(), info->pool.getResult(0), info->outArg,
          /*targetDynamicDims=*/ValueRange{});
      rewriter.eraseOp(info->store);
      SmallVector<Type> argTypes(llvm::map_range(
          func.getArguments(), [](BlockArgument a) { return a.getType(); }));
      func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
    }

    // Pad this caller's input on the host with the fill value, rewire, grow
    // the result and crop it back.
    rewriter.setInsertionPoint(dispatch);
    Value inPadded = createPaddingDispatch(
        rewriter, module, dispatch.getArguments()[inArgNo], inPad,
        hostAffinity, counter, padValue);
    dispatch.getArgumentsMutable().slice(inArgNo, 1).assign(inPadded);
    Value result = dispatch.getResult(0);
    result.setType(RankedTensorType::get(outPad, outType.getElementType()));
    rewriter.setInsertionPointAfter(dispatch);
    Value cropped = createCropDispatch(rewriter, module, result,
                                       outType.getShape(), hostAffinity,
                                       counter);
    result.replaceAllUsesExcept(cropped, cropped.getDefiningOp());
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEPadContractionDispatchesPass() {
  return std::make_unique<AMDAIEPadContractionDispatchesPass>();
}

std::unique_ptr<Pass> createAMDAIEPadPoolingDispatchesPass() {
  return std::make_unique<AMDAIEPadPoolingDispatchesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
