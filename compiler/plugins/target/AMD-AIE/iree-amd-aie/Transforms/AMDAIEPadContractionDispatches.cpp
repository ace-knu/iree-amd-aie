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
  // constant operand constant (padding a hoisted constant weight through a
  // dispatch is itself what broke FC1: it turns the NPU dispatch's weight
  // operand into a runtime-produced buffer).
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

/// High-pads the 2-D bias-fold LHS `src` to `dstType`, writing 1.0 at K index
/// `k` (the first padded position along `kDim`) of every real row and 0.0
/// everywhere else.
///
/// Only that one column has to be 1.0 -- it is what contracts against the
/// folded weight's bias row. Everything else must be 0.0 even though any value
/// would be exact (the folded weight's rows past `k` are +0.0, and padded rows
/// only produce output rows that are cropped away): the AIE core's scalar
/// matmul runs measurably faster on zero operands, and the M padding all lands
/// on one core row, which then bounds the whole dispatch. Filling the padding
/// with 1.0 made VGG16's conv11~13 (196x512x4608, 28 padded rows) 27~33%
/// slower.
///
/// Built as two chained constant-value `tensor.pad`s -- first the ones column,
/// then zeros for the rest -- rather than one pad or a second write op:
///  - zero-padding and then writing the column with a second op is not
///    distributed by the CPU backend (it picks one root, leaves the other write
///    untiled, and rejects it with "write affecting operations on global
///    resources are restricted to workgroup distributed contexts");
///  - a single `tensor.pad` whose body picks the value per index crashes the
///    CPU backend's `foldPadIntoMapStore`, which assumes a constant padding
///    value without checking.
static Value createBiasOnesPad(OpBuilder &b, Location loc,
                               RankedTensorType dstType, Value src,
                               unsigned kDim) {
  auto srcType = cast<RankedTensorType>(src.getType());
  assert(srcType.getRank() == 2 && kDim < 2 && "bias-fold LHS is 2-D");
  Type elemType = srcType.getElementType();
  SmallVector<int64_t> onesShape(srcType.getShape());
  ++onesShape[kDim];
  Value withOnes = tensor::createPadHighOp(
      RankedTensorType::get(onesShape, elemType), src,
      b.create<arith::ConstantOp>(loc, b.getOneAttr(elemType)),
      /*nofold=*/false, loc, b);
  if (ArrayRef<int64_t>(onesShape) == dstType.getShape()) return withOnes;
  return tensor::createPadHighOp(
      dstType, withOnes,
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(elemType)),
      /*nofold=*/false, loc, b);
}

/// Creates a `flow.executable` + `flow.dispatch` that pads `v` up to
/// `dstShape` (high padding via `tensor.pad`), placed on `hostAffinity`. Returns
/// the padded value, or `v` unchanged if `dstShape` already matches. A dispatch
/// (unlike `flow.tensor.update`, which only copies contiguous outer-dim
/// sub-ranges) performs the strided placement required to pad an inner
/// dimension. `counter` uniquifies the symbol names.
/// `padValue`, when set, is the constant to pad with instead of 0 (the pooling
/// pass passes its dispatch's init fill value, -inf for max).
///
/// `onesKDim`, when set, marks `v` as the LHS of a contraction whose bias has
/// been folded into K (see the "Accumulator-bias K-column fold" section) and
/// names its K dim: the padding then carries the ones the fold contracts
/// against (see `createBiasOnesPad`) instead of being all zero. Requires a
/// float element type, which is the bias fold's own precondition and is
/// checked before it decides to fold.
static Value createPaddingDispatch(IRRewriter &rewriter, ModuleOp module,
                                   Value v, ArrayRef<int64_t> dstShape,
                                   IREE::HAL::DeviceAffinityAttr hostAffinity,
                                   int &counter, TypedAttr padValue = nullptr,
                                   std::optional<unsigned> onesKDim = {}) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  if (dstShape == srcType.getShape()) return v;
  auto dstType = RankedTensorType::get(dstShape, srcType.getElementType());
  auto inBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, srcType);
  auto outBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, dstType);
  std::string idx = std::to_string(counter++);
  std::string prefix = onesKDim ? "pad_ones" : "pad";

  // Worker func: load the full input, high-pad it, store.
  auto funcOp = func::FuncOp::create(
      loc, prefix + "_dispatch_" + idx,
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Type elemType = srcType.getElementType();
  Value loaded = fb.create<IREE::TensorExt::DispatchTensorLoadOp>(
      loc, srcType, entry->getArgument(0), /*sourceDynamicDims=*/ValueRange{});
  Value padded =
      onesKDim
          ? createBiasOnesPad(fb, loc, dstType, loaded, *onesKDim)
          : tensor::createPadHighOp(
                dstType, loaded,
                fb.create<arith::ConstantOp>(
                    loc, padValue ? padValue
                                  : cast<TypedAttr>(fb.getZeroAttr(elemType))),
                /*nofold=*/false, loc, fb);
  fb.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, padded, entry->getArgument(1), /*targetDynamicDims=*/ValueRange{});
  fb.create<func::ReturnOp>(loc);

  // Executable wrapping the func, with a default workgroup-count region.
  OpBuilder mb(&module.getBody()->back());
  auto exeOp = IREE::Flow::ExecutableOp::create(mb, loc,
                                               prefix + "_executable_" + idx);
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

/// Computes the `tensor.collapse_shape` reassociation that turns `srcShape`
/// into the 2-D `dstShape`, or {} when the split does not fall on a dim
/// boundary. Used to fold a producer's own N-D result into the 2-D matmul
/// operand shape the `flow.tensor.reshape` between them was performing.
static std::optional<SmallVector<ReassociationIndices>> collapseTo2D(
    ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape) {
  if (dstShape.size() != 2 || srcShape.size() < 2) return std::nullopt;
  ReassociationIndices outer, inner;
  int64_t acc = 1;
  size_t i = 0;
  // Every group must take at least one dim -- `acc == dstShape[0]` is already
  // true before the first iteration when that extent is 1, and an empty
  // reassociation group is not a legal collapse.
  while (i < srcShape.size() && (outer.empty() || acc < dstShape[0])) {
    acc *= srcShape[i];
    outer.push_back(i++);
  }
  if (acc != dstShape[0] || i == srcShape.size()) return std::nullopt;
  acc = 1;
  for (; i < srcShape.size(); ++i) {
    acc *= srcShape[i];
    inner.push_back(i);
  }
  if (acc != dstShape[1] || inner.empty()) return std::nullopt;
  return SmallVector<ReassociationIndices>{outer, inner};
}

/// Tries to make the host dispatch that *produces* `v` write the padded shape
/// itself, instead of emitting a separate padding dispatch downstream of it.
/// Returns the producer's (re-typed) result on success, or {} to fall back.
///
/// This is what removes the model's largest host write. A convolution's LHS is
/// produced by the im2col dispatch and then padded, so the same 155 MiB is
/// written, read back and written again one tile wider; growing the producer's
/// own output binding and padding inside it makes that one write. The
/// `tensor.pad` fuses into the producer generic -- it is a consumer of the
/// dispatch's existing root, not a second root -- which is exactly why this
/// shape compiles where "pad, then write the ones column with a second op"
/// does not (see `createBiasOnesPad`'s comment).
///
/// Deliberately narrow. The producer must be a host dispatch with one caller,
/// one result, one use, and a single full-tensor store, so that nothing else
/// can observe the shape change; anything else falls back to a standalone
/// padding dispatch, which is always correct, just slower.
static Value growProducerDispatchToPadded(
    IRRewriter &rewriter, ModuleOp module, Value v, ArrayRef<int64_t> dstShape,
    std::optional<unsigned> onesKDim,
    IREE::HAL::DeviceAffinityAttr hostAffinity) {
  auto srcType = cast<RankedTensorType>(v.getType());
  Type elemType = srcType.getElementType();
  if (!isa<FloatType>(elemType)) return {};
  if (srcType.getRank() != 2 || dstShape.size() != 2) return {};
  // Nothing to fold in: this operand is already the padded shape (only the
  // other operand's dims grew). Rewriting the producer anyway would flatten
  // its output binding for no reason.
  if (dstShape == srcType.getShape()) return {};

  // v = [flow.tensor.reshape of] the producer dispatch's single result.
  auto reshapeOp = v.getDefiningOp<IREE::Flow::TensorReshapeOp>();
  Value produced = reshapeOp ? reshapeOp.getSource() : v;
  if (reshapeOp && !reshapeOp.getResult().hasOneUse()) return {};
  auto producer = produced.getDefiningOp<IREE::Flow::DispatchOp>();
  if (!producer || producer->getNumResults() != 1) return {};
  if (!producer.getResult(0).hasOneUse()) return {};
  if (producer->getAttr("stream.affinity") != hostAffinity) return {};

  func::FuncOp func = getDispatchFunc(producer, module);
  if (!func) return {};
  // One caller only: the rewrite changes the executable's signature, so a
  // second caller would be left passing the old shape.
  int callers = 0;
  module.walk([&](IREE::Flow::DispatchOp d) {
    if (getDispatchFunc(d, module) == func) ++callers;
  });
  if (callers != 1) return {};

  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> stores;
  func.walk(
      [&](IREE::TensorExt::DispatchTensorStoreOp s) { stores.push_back(s); });
  if (stores.size() != 1) return {};
  IREE::TensorExt::DispatchTensorStoreOp store = stores.front();
  auto outArg = dyn_cast<BlockArgument>(store.getTarget());
  if (!outArg) return {};
  auto outType = dyn_cast<IREE::TensorExt::DispatchTensorType>(outArg.getType());
  if (!outType || outType.getAccess() != IREE::TensorExt::TensorAccess::WriteOnly)
    return {};
  for (OpFoldResult o : store.getMixedOffsets())
    if (!isConstantIntValue(o, 0)) return {};
  for (OpFoldResult s : store.getMixedStrides())
    if (!isConstantIntValue(s, 1)) return {};
  auto storedType = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!storedType || storedType.getElementType() != elemType) return {};
  if (storedType.getShape() != outType.asRankedTensorType().getShape())
    return {};

  // The reshape this absorbs has to be expressible as a collapse of the
  // producer's own result dims.
  std::optional<SmallVector<ReassociationIndices>> reassoc =
      collapseTo2D(storedType.getShape(), srcType.getShape());
  if (!reassoc) return {};

  auto dstType = RankedTensorType::get(dstShape, elemType);
  Location loc = store.getLoc();
  rewriter.setInsertionPoint(store);
  Value flat = storedType.getRank() == 2
                   ? store.getValue()
                   : rewriter.create<tensor::CollapseShapeOp>(
                         loc, store.getValue(), *reassoc).getResult();
  Value padded =
      onesKDim
          ? createBiasOnesPad(rewriter, loc, dstType, flat, *onesKDim)
          : tensor::createPadHighOp(
                dstType, flat,
                rewriter.create<arith::ConstantOp>(
                    loc, rewriter.getZeroAttr(elemType)),
                /*nofold=*/false, loc, rewriter);
  outArg.setType(IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, dstType));
  rewriter.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, padded, outArg, /*targetDynamicDims=*/ValueRange{});
  rewriter.eraseOp(store);
  SmallVector<Type> argTypes(llvm::map_range(
      func.getArguments(), [](BlockArgument a) { return a.getType(); }));
  func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));

  producer.getResult(0).setType(dstType);
  if (reshapeOp) rewriter.replaceOp(reshapeOp, producer.getResult(0));
  return producer.getResult(0);
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

//===----------------------------------------------------------------------===//
// Accumulator-bias K-column fold
//
// A convolution lowered through im2col arrives here as
//   matmul ins(%im2col : [M,K], %weight : [K,N]) outs(%bias_bcast : [M,N])
// where `%bias_bcast` is a *third* dispatch operand: a host dispatch that
// broadcasts the constant bias vector to the full [M,N] accumulator. The NPU
// backend cannot bind a third L3->L2 operand ("'amdaie.connection' op no
// producer DMA channel available"), and the broadcast itself is one of the
// larger host writes in the model.
//
// Both go away by moving the bias into the contraction's K dimension:
//   LHS' = [M, K+1] with column K set to 1.0
//   RHS' = [K+1, N] with row    K set to the bias
// so that sum_k LHS'[m,k] * RHS'[k,n] == sum_k LHS[m,k] * RHS[k,n] + bias[n],
// and the accumulator becomes a plain zero fill again.
//
// The reason this is worth doing *here*, in the padding pass, rather than on
// the convolution before im2col (folding the bias into an extra input
// *channel*): K = C*kh*kw, so one extra channel
// buys nine extra K positions of which eight are structurally zero, and it
// costs a full copy of the activation into a (C+1)-channel buffer plus a fill
// for the ones plane -- two host dispatches per convolution. Here the ones
// column is written by the K-padding dispatch that has to run anyway (K+1 is
// never tile-divisible, so the LHS is padded either way), and the bias row is
// folded into a compile-time constant, so the fold is free.
//===----------------------------------------------------------------------===//

/// Returns the `DenseElementsAttr` behind `v` when `v` is a compile-time
/// constant reached only through reshapes -- either an inlined `arith.constant`
/// or an immutable `util.global` (which is where const-eval leaves weights and
/// biases). Returns {} otherwise.
static DenseElementsAttr matchConstantAttr(Value v) {
  Value root = v;
  while (auto reshapeOp = root.getDefiningOp<IREE::Flow::TensorReshapeOp>())
    root = reshapeOp.getSource();
  if (auto cstOp = root.getDefiningOp<arith::ConstantOp>())
    return dyn_cast<DenseElementsAttr>(cstOp.getValue());
  auto loadOp = root.getDefiningOp<IREE::Util::GlobalLoadOp>();
  if (!loadOp || !loadOp.getIsImmutable()) return {};
  auto globalOp = SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
      loadOp, loadOp.getGlobalAttr());
  if (!globalOp) return {};
  return dyn_cast_or_null<DenseElementsAttr>(globalOp.getInitialValueAttr());
}

/// Resolves `v` to the immutable weight global it loads from (through
/// reshapes) together with that global's dense initial value, in exactly the
/// form `padConstantRowMajor` can rewrite. Returns false without touching the
/// out-params when `v` is not such a weight -- letting a caller decide whether
/// the constant fold is possible *before* it starts mutating IR.
static bool matchConstantWeight(Value v, IREE::Util::GlobalOp &globalOp,
                                DenseElementsAttr &attr) {
  auto srcType = dyn_cast<RankedTensorType>(v.getType());
  if (!srcType || srcType.getRank() != 2) return false;
  Value root = v;
  while (auto reshapeOp = root.getDefiningOp<IREE::Flow::TensorReshapeOp>())
    root = reshapeOp.getSource();
  auto loadOp = root.getDefiningOp<IREE::Util::GlobalLoadOp>();
  if (!loadOp || !loadOp.getIsImmutable()) return false;
  auto candidate = SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
      loadOp, loadOp.getGlobalAttr());
  if (!candidate) return false;
  auto candidateAttr =
      dyn_cast_or_null<DenseElementsAttr>(candidate.getInitialValueAttr());
  // Same restrictions `padConstantRowMajor` enforces on the raw buffer, so a
  // caller that gets `true` here can rely on the pad succeeding.
  if (!candidateAttr || candidateAttr.isSplat()) return false;
  Type elemType = srcType.getElementType();
  if (!elemType.isIntOrFloat()) return false;
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0) return false;
  int64_t elemBytes = bitWidth / 8;
  if (candidateAttr.getRawData().size() !=
      static_cast<size_t>(srcType.getNumElements() * elemBytes))
    return false;
  globalOp = candidate;
  attr = candidateAttr;
  return true;
}

/// Recognizes the host dispatch that materializes a contraction's
/// accumulator-as-bias operand: a single `linalg.generic` that broadcasts a
/// constant 1-D bias along the outer (M) dim and stores it. Returns the bias
/// values, or {} when `acc` is anything else -- a runtime accumulator, a
/// partial-sum carried in from another dispatch, a bias that is not a
/// compile-time constant, or any broadcast shape other than "constant per
/// output column".
static DenseElementsAttr matchBroadcastBiasDispatch(Value acc, ModuleOp module) {
  // const-eval hoists a broadcast whose result is small enough into a global
  // filled by a `util.initializer`, so the accumulator arrives as a load of
  // that global rather than as the broadcast dispatch itself. Look through it
  // to the value the initializer stores; the recognition below is the same
  // either way, and folding the bias leaves the whole initializer dead.
  if (auto loadOp = acc.getDefiningOp<IREE::Util::GlobalLoadOp>()) {
    if (!loadOp.getIsImmutable()) return {};
    SmallVector<IREE::Util::GlobalStoreOp> stores;
    module.walk([&](IREE::Util::GlobalStoreOp s) {
      if (s.getGlobalAttr() == loadOp.getGlobalAttr()) stores.push_back(s);
    });
    if (stores.size() != 1) return {};
    acc = stores.front().getValue();
  }
  auto dispatch = acc.getDefiningOp<IREE::Flow::DispatchOp>();
  if (!dispatch || dispatch->getNumResults() != 1) return {};
  func::FuncOp func = getDispatchFunc(dispatch, module);
  if (!func) return {};

  SmallVector<linalg::GenericOp> generics;
  func.walk([&](linalg::GenericOp g) { generics.push_back(g); });
  if (generics.size() != 1) return {};
  linalg::GenericOp generic = generics.front();
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1) return {};
  if (generic->getNumResults() != 1) return {};
  if (generic.getNumLoops() != 2) return {};
  for (utils::IteratorType it : generic.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel) return {};
  MLIRContext *ctx = generic.getContext();
  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  // ins: (d0, d1) -> (d1)   -- the same value for every row
  // outs:(d0, d1) -> (d0, d1)
  if (maps[0] != AffineMap::get(2, 0, {getAffineDimExpr(1, ctx)}, ctx)) return {};
  if (maps[1] != AffineMap::getMultiDimIdentityMap(2, ctx)) return {};
  // The body must be a bare copy. Unlike `isShapePreservingUnaryMapOf` (where
  // the structural checks alone make any body safe) the body matters here: we
  // are about to reproduce its result as a constant, so it must not compute
  // anything.
  Block &body = generic.getRegion().front();
  if (!llvm::hasSingleElement(body.getOperations())) return {};
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield->getNumOperands() != 1) return {};
  if (yield->getOperand(0) != body.getArgument(0)) return {};
  // Exactly one store, of the broadcast itself.
  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> stores;
  func.walk(
      [&](IREE::TensorExt::DispatchTensorStoreOp s) { stores.push_back(s); });
  if (stores.size() != 1) return {};
  if (stores.front().getValue() != generic.getResult(0)) return {};

  // The bias itself: either inlined into the executable as a constant, or
  // passed in as an operand that is a constant on the host side.
  Value in = generic.getDpsInputOperand(0)->get();
  DenseElementsAttr biasAttr;
  if (auto cstOp = in.getDefiningOp<arith::ConstantOp>()) {
    biasAttr = dyn_cast<DenseElementsAttr>(cstOp.getValue());
  } else if (auto load = getFullTensorLoad(in)) {
    auto arg = dyn_cast<BlockArgument>(load.getSource());
    if (!arg) return {};
    unsigned argNo = arg.getArgNumber();
    if (argNo >= dispatch.getArguments().size()) return {};
    biasAttr = matchConstantAttr(dispatch.getArguments()[argNo]);
  }
  if (!biasAttr) return {};
  auto biasType = dyn_cast<RankedTensorType>(biasAttr.getType());
  if (!biasType || biasType.getRank() != 1) return {};
  if (!isa<FloatType>(biasType.getElementType())) return {};
  return biasAttr;
}

/// Materializes `weight` zero-padded to `dstShape` *with the bias spliced into
/// K position `kIdx`* as a new compile-time constant global, and returns a
/// load of it. `kPos`/`nPos` are the weight's own K/N dim positions, so both
/// the plain [K,N] and the transpose_b [N,K] layouts are handled.
///
/// The bias arrives in the accumulator's element type (f32) while the weight
/// has usually been demoted to bf16; it is converted through `APFloat` and
/// re-encoded by `DenseElementsAttr` itself, so no assumption is made about
/// the raw layout of either type.
static Value createBiasFoldedWeightGlobal(
    IRRewriter &rewriter, ModuleOp module, Value weight,
    ArrayRef<int64_t> dstShape, DenseElementsAttr biasAttr, unsigned kPos,
    unsigned nPos, int64_t kIdx, int64_t n, int &counter) {
  IREE::Util::GlobalOp globalOp;
  DenseElementsAttr srcAttr;
  if (!matchConstantWeight(weight, globalOp, srcAttr)) return {};
  auto srcType = cast<RankedTensorType>(weight.getType());
  Type elemType = srcType.getElementType();
  auto floatType = dyn_cast<FloatType>(elemType);
  if (!floatType) return {};
  auto dstType = RankedTensorType::get(dstShape, elemType);

  DenseElementsAttr paddedAttr = padConstantRowMajor(srcAttr, srcType, dstType);
  if (!paddedAttr) return {};

  // Convert the bias to the weight's element type and let DenseElementsAttr
  // encode it, then copy those bytes into place.
  SmallVector<APFloat> converted;
  converted.reserve(n);
  for (APFloat v : biasAttr.getValues<APFloat>()) {
    bool losesInfo = false;
    APFloat c = v;
    c.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
              &losesInfo);
    converted.push_back(c);
    if (static_cast<int64_t>(converted.size()) == n) break;
  }
  if (static_cast<int64_t>(converted.size()) != n) return {};
  auto rowAttr = DenseElementsAttr::get(
      RankedTensorType::get({n}, elemType), converted);
  ArrayRef<char> rowRaw = rowAttr.getRawData();
  int64_t elemBytes = floatType.getIntOrFloatBitWidth() / 8;
  if (rowRaw.size() != static_cast<size_t>(n * elemBytes)) return {};

  SmallVector<char> dst(paddedAttr.getRawData().begin(),
                        paddedAttr.getRawData().end());
  int64_t dstCols = dstShape[1];
  for (int64_t j = 0; j < n; ++j) {
    // The bias occupies K index `kIdx` and N index `j`, placed at whichever
    // of the two dims each of those is for this weight's layout.
    int64_t coords[2];
    coords[kPos] = kIdx;
    coords[nPos] = j;
    std::memcpy(dst.data() + (coords[0] * dstCols + coords[1]) * elemBytes,
                rowRaw.data() + j * elemBytes, elemBytes);
  }
  auto foldedAttr = DenseElementsAttr::getFromRawBuffer(dstType, dst);
  if (!foldedAttr) return {};

  // Declared at the top of the module for the same reason
  // `createPaddedConstantGlobal` does -- see its comment.
  Location loc = weight.getLoc();
  OpBuilder moduleBuilder = OpBuilder::atBlockBegin(module.getBody());
  auto foldedGlobalOp = IREE::Util::GlobalOp::create(
      moduleBuilder, loc, "__bias_folded_const_" + std::to_string(counter++),
      /*isMutable=*/false, dstType, std::optional<TypedAttr>{foldedAttr});
  SymbolTable::setSymbolVisibility(foldedGlobalOp,
                                   SymbolTable::Visibility::Private);
  if (auto affinity = globalOp->getAttr("stream.affinity.default"))
    foldedGlobalOp->setAttr("stream.affinity.default", affinity);
  return foldedGlobalOp.createLoadOp(loc, rewriter)->getResult(0);
}

/// Returns the constant bias this caller's accumulator broadcasts, if the
/// accumulator-bias K-column fold applies to `dispatch` on its own: the
/// accumulator is a host broadcast of a constant [N] bias *and* the weight is
/// a constant this pass can rewrite at compile time. Null otherwise, including
/// in the ordinary two-operand case (`info.fill` set). The fold is applied only
/// to an executable with a single caller, see `runOnOperation`.
static DenseElementsAttr matchFoldableBias(IREE::Flow::DispatchOp dispatch,
                                           const MatmulInfo &info,
                                           ModuleOp module) {
  if (!info.accArg) return {};
  Value lhs = dispatch.getArguments()[info.lhsArg.getArgNumber()];
  Value rhs = dispatch.getArguments()[info.rhsArg.getArgNumber()];
  if (!isa<FloatType>(getElementTypeOrSelf(lhs.getType())) ||
      !isa<FloatType>(getElementTypeOrSelf(rhs.getType())))
    return {};
  IREE::Util::GlobalOp weightGlobal;
  DenseElementsAttr weightAttr;
  if (!matchConstantWeight(rhs, weightGlobal, weightAttr)) return {};
  DenseElementsAttr biasAttr = matchBroadcastBiasDispatch(
      dispatch.getArguments()[info.accArg.getArgNumber()], module);
  if (!biasAttr) return {};
  linalg::MatmulOp matmul = info.matmul;
  unsigned rhsNPos =
      operandDimForLoop(matmul.getIndexingMapsArray()[1], /*N=*/1);
  int64_t n = cast<RankedTensorType>(rhs.getType()).getShape()[rhsNPos];
  if (cast<RankedTensorType>(biasAttr.getType()).getDimSize(0) != n) return {};
  return biasAttr;
}

/// Rewrites the matmul's accumulator from the incoming bias binding to a plain
/// zero `fill`, erases the now-unused binding from the executable, and updates
/// `info` to the fill/empty form (so the padding helpers that follow take
/// their fill path). The caller must erase the corresponding host operand.
static void replaceAccWithZeroFill(IRRewriter &rewriter, MatmulInfo &info,
                                   func::FuncOp func) {
  linalg::MatmulOp matmul = info.matmul;
  Location loc = matmul.getLoc();
  auto accType = cast<RankedTensorType>(info.accLoad.getResult().getType());
  Type elemType = accType.getElementType();
  rewriter.setInsertionPoint(matmul);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, accType.getShape(),
                                                  elemType);
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemType));
  auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zero},
                                                ValueRange{emptyOp.getResult()});
  matmul.setDpsInitOperand(0, fillOp.getResult(0));
  unsigned accArgNo = info.accArg.getArgNumber();
  rewriter.eraseOp(info.accLoad);
  (void)func.eraseArgument(accArgNo);
  info.accLoad = nullptr;
  info.accArg = nullptr;
  info.fill = fillOp;
  info.empty = emptyOp;
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
  // Executables whose accumulator-bias operand is folded into K (see the
  // "Accumulator-bias K-column fold" section above): only those with a single
  // caller that can take the fold. The fold changes the executable's
  // *signature* (the bias binding goes away, K grows by one), which the
  // per-caller operand padding below cannot reconcile across callers: a caller
  // that does not fold would pass an operand the executable no longer has,
  // and the executable's `MatmulInfo`, recomputed for a later caller after the
  // rewrite, no longer shows the bias at all. A shared executable therefore
  // keeps the accumulator form and is padded like any other, which is always
  // correct. Decided here, before any IR is touched. Only reachable with
  // executable deduplication on, which this model does not use.
  DenseMap<Operation *, int> numCallers;
  for (IREE::Flow::DispatchOp dispatch : dispatches)
    if (func::FuncOp func = getDispatchFunc(dispatch, module))
      ++numCallers[func.getOperation()];

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
        target, m, lhsType.getElementType(), rhsType.getElementType(),
        accElemType, /*transposeB=*/rhsNPos < rhsKPos);
    if (!mult) continue;

    // Accumulator-bias K-column fold (see `matchFoldableBias`), applied only
    // when this is the executable's only caller (see `numCallers`). Null in
    // the ordinary two-operand case (`info->fill` set), which is every
    // dispatch when the pre-im2col convolution fold is left on.
    DenseElementsAttr biasAttr;
    int64_t accArgNo = -1;
    if (info->accArg) {
      accArgNo = info->accArg.getArgNumber();
      if (numCallers.lookup(func.getOperation()) == 1)
        biasAttr = matchFoldableBias(dispatch, *info, module);
    }

    int64_t mPad = roundUpToMultiple(m, mult->m);
    int64_t nPad = roundUpToMultiple(n, mult->n);
    // The bias lives at K index `k`, so the contraction's logical K grows by
    // one before rounding. K+1 is never a tile multiple, so a folded operand
    // is always padded -- which is precisely what makes the ones column free.
    int64_t kPad = roundUpToMultiple(biasAttr ? k + 1 : k, mult->k);
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
      // Before growing anything: turn the bias binding into a zero fill, so
      // the padding below sees the ordinary two-operand form (and `padOut`
      // takes `growMatmulInit`'s fill path).
      if (biasAttr) replaceAccWithZeroFill(rewriter, *info, func);
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
    // Prefer growing the LHS's own producer dispatch over adding a padding
    // dispatch behind it: for a convolution that turns im2col's 155 MiB
    // write + read + wider re-write into a single write.
    Value lhsPadded = growProducerDispatchToPadded(
        rewriter, module, dispatch.getArguments()[lhsArgNo], lhsPad,
        biasAttr ? std::optional<unsigned>(lhsKPos) : std::nullopt,
        hostAffinity);
    if (!lhsPadded) {
      rewriter.setInsertionPoint(dispatch);
      lhsPadded =
          biasAttr ? createPaddingDispatch(
                         rewriter, module, dispatch.getArguments()[lhsArgNo],
                         lhsPad, hostAffinity, counter, /*padValue=*/nullptr,
                         lhsKPos)
                   : createPaddingDispatchOrConstant(
                         rewriter, module, dispatch.getArguments()[lhsArgNo],
                         lhsPad, hostAffinity, counter);
    }
    rewriter.setInsertionPoint(dispatch);
    Value rhsPadded =
        biasAttr ? createBiasFoldedWeightGlobal(
                       rewriter, module, dispatch.getArguments()[rhsArgNo],
                       rhsPad, biasAttr, rhsKPos, rhsNPos, k, n, counter)
                 : createPaddingDispatchOrConstant(
                       rewriter, module, dispatch.getArguments()[rhsArgNo],
                       rhsPad, hostAffinity, counter);
    // `createBiasFoldedWeightGlobal` re-validates what `matchConstantWeight`
    // requires and returns {} rather than guessing. That cannot happen here
    // (the same conditions were checked above, before the executable was
    // rewritten), and if it ever does the only honest thing left is to fail
    // loudly rather than emit a dispatch reading an unpadded operand.
    if (!lhsPadded || !rhsPadded) {
      dispatch.emitError("failed to materialize padded contraction operands");
      return signalPassFailure();
    }
    dispatch.getArgumentsMutable().slice(lhsArgNo, 1).assign(lhsPadded);
    dispatch.getArgumentsMutable().slice(rhsArgNo, 1).assign(rhsPadded);
    // Drop the bias operand last, so the assignments above still address the
    // original operand positions.
    if (biasAttr) dispatch.getArgumentsMutable().erase(accArgNo);

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
