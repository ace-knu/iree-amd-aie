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
// pass assumes the amd-aie contraction dispatch is a *plain* matmul
// (`empty -> fill(0) -> matmul -> store`, with bias/relu/etc. already split into
// separate dispatches upstream). For such a dispatch pinned to an amd-aie device
// whose M/K dims are not multiples of the target tiles, it:
//   * zero-pads the matmul operands to the padded shape *outside* the dispatch
//     (a host `tensor.pad` dispatch), and
//   * rewrites the executable so its bindings/loads/matmul see the padded shape.
//
// K is a reduction dim: the padded elements are zero, so the extra products are
// zero and the output is unchanged (no cropping). The output dims M (outer) and
// N (inner) are handled uniformly: when either is padded the matmul output, the
// output binding and the store grow too, and the padded dispatch result is
// cropped back to the logical shape by a `tensor.extract_slice` *dispatch*
// placed on the host (CPU). Using a dispatch (rather than `flow.tensor.slice`,
// which only crops a contiguous outer-dim range) lets the inner N dim be cropped
// with a strided extraction, so a dispatch whose N is non-divisible (e.g. a
// 1000-class classifier matmul) is padded and cropped like M.
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
  return PaddingMultiples{/*m=*/*numRows * (*instr)[0],
                          /*n=*/*numCols * (*instr)[1],
                          /*k=*/getPackPeelReductionTile(/*kPackScaleL1=*/1)};
}

/// A plain matmul (`empty -> fill -> matmul -> store`) inside a dispatch
/// executable, with the ops/bindings needed to grow its M/K dims. Sizes are
/// intentionally not cached here -- they are read from each caller dispatch's
/// operands so a shared executable can be handled per caller.
struct MatmulInfo {
  linalg::MatmulOp matmul;
  IREE::TensorExt::DispatchTensorLoadOp lhsLoad, rhsLoad;
  BlockArgument lhsArg, rhsArg;  // executable bindings feeding LHS/RHS
  linalg::FillOp fill;           // output init: matmul outs = fill(empty)
  tensor::EmptyOp empty;
  IREE::TensorExt::DispatchTensorStoreOp store;  // writes the output binding
  BlockArgument outArg;                          // the output binding
};

/// Same as `MatmulInfo`, but for a `linalg.batch_matmul` (an extra leading
/// batch dim on every operand/result, otherwise identical structure). Kept as
/// a separate struct/getter/grow-pair rather than generalizing `MatmulInfo`
/// in place: the 2D helpers below hardcode rank-2 shapes throughout, and
/// duplicating the (small) rank-3 equivalents avoids touching code already
/// validated for the plain-matmul case.
struct BatchMatmulInfo {
  linalg::BatchMatmulOp matmul;
  IREE::TensorExt::DispatchTensorLoadOp lhsLoad, rhsLoad;
  BlockArgument lhsArg, rhsArg;
  linalg::FillOp fill;
  tensor::EmptyOp empty;
  IREE::TensorExt::DispatchTensorStoreOp store;
  BlockArgument outArg;
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

/// Inspects the func for a single plain matmul: LHS/RHS from full-tensor loads
/// of bindings, an `empty`+`fill` output init, and the matmul result stored
/// directly to the output binding.
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
  auto fill = matmul.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill) return std::nullopt;
  auto empty = fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
  if (!empty) return std::nullopt;
  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> stores;
  func.walk(
      [&](IREE::TensorExt::DispatchTensorStoreOp s) { stores.push_back(s); });
  if (stores.size() != 1) return std::nullopt;
  auto store = stores.front();
  // Plain matmul: the matmul result feeds the store directly (no output chain).
  if (store.getValue() != matmul.getResult(0)) return std::nullopt;
  auto outArg = dyn_cast<BlockArgument>(store.getTarget());
  if (!outArg) return std::nullopt;
  return MatmulInfo{matmul, lhsLoad, rhsLoad, lhsArg,
                    rhsArg, fill,    empty,   store,  outArg};
}

/// Batch-matmul counterpart of `getMatmulInfo` -- identical shape of check
/// (full-tensor loads, `empty`+`fill` init, direct store), just matched
/// against `linalg::BatchMatmulOp` instead of `linalg::MatmulOp`.
static std::optional<BatchMatmulInfo> getBatchMatmulInfo(func::FuncOp func) {
  SmallVector<linalg::BatchMatmulOp> matmuls;
  func.walk([&](linalg::BatchMatmulOp op) { matmuls.push_back(op); });
  if (matmuls.size() != 1) return std::nullopt;
  linalg::BatchMatmulOp matmul = matmuls.front();
  auto lhsLoad = getFullTensorLoad(matmul.getInputs()[0]);
  auto rhsLoad = getFullTensorLoad(matmul.getInputs()[1]);
  if (!lhsLoad || !rhsLoad) return std::nullopt;
  auto lhsArg = dyn_cast<BlockArgument>(lhsLoad.getSource());
  auto rhsArg = dyn_cast<BlockArgument>(rhsLoad.getSource());
  if (!lhsArg || !rhsArg) return std::nullopt;
  auto fill = matmul.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill) return std::nullopt;
  auto empty = fill.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
  if (!empty) return std::nullopt;
  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> stores;
  func.walk(
      [&](IREE::TensorExt::DispatchTensorStoreOp s) { stores.push_back(s); });
  if (stores.size() != 1) return std::nullopt;
  auto store = stores.front();
  if (store.getValue() != matmul.getResult(0)) return std::nullopt;
  auto outArg = dyn_cast<BlockArgument>(store.getTarget());
  if (!outArg) return std::nullopt;
  return BatchMatmulInfo{matmul, lhsLoad, rhsLoad, lhsArg,
                         rhsArg, fill,    empty,   store,  outArg};
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

/// Grows the output init (`empty`+`fill`) and the matmul result to [mPad, nPad].
static void growMatmulInit(IRRewriter &rewriter, MatmulInfo &info, int64_t mPad,
                           int64_t nPad) {
  Location loc = info.fill.getLoc();
  Type elemType =
      cast<RankedTensorType>(info.matmul.getResult(0).getType()).getElementType();
  rewriter.setInsertionPoint(info.empty);
  Value newEmpty = rewriter.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{mPad, nPad}, elemType);
  rewriter.setInsertionPoint(info.fill);
  Value cst = info.fill.getInputs()[0];
  auto newFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{cst}, ValueRange{newEmpty});
  info.matmul.setDpsInitOperand(0, newFill.getResult(0));
  info.matmul.getResult(0).setType(RankedTensorType::get({mPad, nPad}, elemType));
  rewriter.eraseOp(info.fill);
  rewriter.eraseOp(info.empty);
}

/// Grows the output binding and rewrites the store to write the full padded
/// [mPad, nPad] matmul result.
static void growOutputStore(IRRewriter &rewriter, MatmulInfo &info,
                            int64_t mPad, int64_t nPad) {
  auto dtt = cast<IREE::TensorExt::DispatchTensorType>(info.outArg.getType());
  auto newTensorType = RankedTensorType::get(
      {mPad, nPad}, dtt.asRankedTensorType().getElementType());
  info.outArg.setType(IREE::TensorExt::DispatchTensorType::get(dtt.getAccess(),
                                                               newTensorType));
  rewriter.setInsertionPoint(info.store);
  rewriter.create<IREE::TensorExt::DispatchTensorStoreOp>(
      info.store.getLoc(), info.matmul.getResult(0), info.outArg,
      /*targetDynamicDims=*/ValueRange{});
  rewriter.eraseOp(info.store);
}

/// Batch-matmul counterpart of `growMatmulInit`: identical, except the output
/// init/result shape carries the (unpadded) leading batch dim, `[batch, mPad,
/// nPad]` instead of `[mPad, nPad]`.
static void growBatchMatmulInit(IRRewriter &rewriter, BatchMatmulInfo &info,
                                int64_t batch, int64_t mPad, int64_t nPad) {
  Location loc = info.fill.getLoc();
  Type elemType =
      cast<RankedTensorType>(info.matmul.getResult(0).getType()).getElementType();
  rewriter.setInsertionPoint(info.empty);
  Value newEmpty = rewriter.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{batch, mPad, nPad}, elemType);
  rewriter.setInsertionPoint(info.fill);
  Value cst = info.fill.getInputs()[0];
  auto newFill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{cst}, ValueRange{newEmpty});
  info.matmul.setDpsInitOperand(0, newFill.getResult(0));
  info.matmul.getResult(0).setType(
      RankedTensorType::get({batch, mPad, nPad}, elemType));
  rewriter.eraseOp(info.fill);
  rewriter.eraseOp(info.empty);
}

/// Batch-matmul counterpart of `growOutputStore`: identical, except the output
/// binding/store shape carries the leading batch dim, `[batch, mPad, nPad]`.
static void growBatchOutputStore(IRRewriter &rewriter, BatchMatmulInfo &info,
                                 int64_t batch, int64_t mPad, int64_t nPad) {
  auto dtt = cast<IREE::TensorExt::DispatchTensorType>(info.outArg.getType());
  auto newTensorType = RankedTensorType::get(
      {batch, mPad, nPad}, dtt.asRankedTensorType().getElementType());
  info.outArg.setType(IREE::TensorExt::DispatchTensorType::get(dtt.getAccess(),
                                                               newTensorType));
  rewriter.setInsertionPoint(info.store);
  rewriter.create<IREE::TensorExt::DispatchTensorStoreOp>(
      info.store.getLoc(), info.matmul.getResult(0), info.outArg,
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

    if (std::optional<BatchMatmulInfo> batchInfo = getBatchMatmulInfo(func)) {
      // Same padding scheme as the plain-matmul path below, generalized to
      // carry a leading (unpadded) batch dim: `linalg.batch_matmul`'s
      // iterator order is (batch, M, N, K) -- one position later than
      // `linalg.matmul`'s (M, N, K) -- so the loop-dim constants passed to
      // `operandDimForLoop` shift by one, and every padded/cropped shape
      // gains a leading `batch` entry that is never itself padded.
      int64_t lhsArgNo = batchInfo->lhsArg.getArgNumber();
      int64_t rhsArgNo = batchInfo->rhsArg.getArgNumber();
      auto lhsType =
          cast<RankedTensorType>(dispatch.getArguments()[lhsArgNo].getType());
      auto rhsType =
          cast<RankedTensorType>(dispatch.getArguments()[rhsArgNo].getType());
      SmallVector<AffineMap> maps = batchInfo->matmul.getIndexingMapsArray();
      unsigned lhsBatchPos = operandDimForLoop(maps[0], /*batch=*/0);
      unsigned lhsMPos = operandDimForLoop(maps[0], /*M=*/1);
      unsigned lhsKPos = operandDimForLoop(maps[0], /*K=*/3);
      unsigned rhsNPos = operandDimForLoop(maps[1], /*N=*/2);
      unsigned rhsKPos = operandDimForLoop(maps[1], /*K=*/3);
      int64_t batch = lhsType.getShape()[lhsBatchPos];
      int64_t m = lhsType.getShape()[lhsMPos];
      int64_t k = lhsType.getShape()[lhsKPos];
      int64_t n = rhsType.getShape()[rhsNPos];
      Type outElemType =
          cast<RankedTensorType>(batchInfo->matmul.getResult(0).getType())
              .getElementType();

      std::optional<PaddingMultiples> mult = getPaddingMultiples(
          target, lhsType.getElementType(), rhsType.getElementType(),
          outElemType);
      if (!mult) continue;

      int64_t mPad = roundUpToMultiple(m, mult->m);
      int64_t nPad = roundUpToMultiple(n, mult->n);
      int64_t kPad = roundUpToMultiple(k, mult->k);
      if (mPad == m && nPad == n && kPad == k) continue;
      bool padOut = mPad != m || nPad != n;

      SmallVector<int64_t, 3> lhsPad(3), rhsPad(3);
      lhsPad[lhsBatchPos] = batch;
      lhsPad[lhsMPos] = mPad;
      lhsPad[lhsKPos] = kPad;
      rhsPad[lhsBatchPos] = batch;  // batch is at the same operand position on both sides
      rhsPad[rhsNPos] = nPad;
      rhsPad[rhsKPos] = kPad;

      if (paddedExecutables.insert(func.getOperation()).second) {
        growBinding(rewriter, batchInfo->lhsArg, batchInfo->lhsLoad, lhsPad);
        growBinding(rewriter, batchInfo->rhsArg, batchInfo->rhsLoad, rhsPad);
        if (padOut) {
          growBatchMatmulInit(rewriter, *batchInfo, batch, mPad, nPad);
          growBatchOutputStore(rewriter, *batchInfo, batch, mPad, nPad);
        }
        SmallVector<Type> argTypes(llvm::map_range(
            func.getArguments(), [](BlockArgument a) { return a.getType(); }));
        func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
      }

      rewriter.setInsertionPoint(dispatch);
      Value lhsPadded = createPaddingDispatch(
          rewriter, module, dispatch.getArguments()[lhsArgNo], lhsPad,
          hostAffinity, counter);
      Value rhsPadded = createPaddingDispatch(
          rewriter, module, dispatch.getArguments()[rhsArgNo], rhsPad,
          hostAffinity, counter);
      dispatch.getArgumentsMutable().slice(lhsArgNo, 1).assign(lhsPadded);
      dispatch.getArgumentsMutable().slice(rhsArgNo, 1).assign(rhsPadded);

      if (padOut) {
        Value result = dispatch.getResult(0);
        result.setType(RankedTensorType::get({batch, mPad, nPad}, outElemType));
        rewriter.setInsertionPointAfter(dispatch);
        Value cropped = createCropDispatch(rewriter, module, result,
                                           {batch, m, n}, hostAffinity, counter);
        result.replaceAllUsesExcept(cropped, cropped.getDefiningOp());
      }
      continue;
    }

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
        target, lhsType.getElementType(), rhsType.getElementType(),
        outElemType);
    if (!mult) continue;

    int64_t mPad = roundUpToMultiple(m, mult->m);
    int64_t nPad = roundUpToMultiple(n, mult->n);
    int64_t kPad = roundUpToMultiple(k, mult->k);
    if (mPad == m && nPad == n && kPad == k) continue;  // already divisible
    // Any output dim (M outer and/or N inner) may be padded; the grown result is
    // cropped back on the host by a single extract_slice dispatch.
    bool padOut = mPad != m || nPad != n;

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
  //  - They were validated on exactly one dispatch: VGG-16's dense0
  //    (K=25088, N=4096) in bf16 on npu4. Its other two fully connected layers
  //    (K=4096, N=4096 and K=4096, N=1000) both have K == kKThreshold and so do
  //    not split at all. No other element type or device was measured.
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
  constexpr int64_t kNChunkMax = 512;    // max N per chunk (validated good chunk)

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
    Type elemType = resultType.getElementType();

    // Even, tile-aligned chunk size (N is already a multiple of the N tile).
    std::optional<PaddingMultiples> mult = getPaddingMultiples(
        target,
        cast<RankedTensorType>(dispatch.getArguments()[lhsArgNo].getType())
            .getElementType(),
        rhsType.getElementType(), elemType);
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
    Value lhs = dispatch.getArguments()[lhsArgNo];
    Value weight = dispatch.getArguments()[rhsArgNo];
    Location loc = dispatch.getLoc();
    auto chunkResultType = RankedTensorType::get({M, C}, elemType);
    SmallVector<int64_t, 2> rhsChunk(2);
    rhsChunk[rhsNPos] = C;
    rhsChunk[rhsKPos] = K;

    SmallVector<Value> chunkOutputs;
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
      if (!ci) continue;
      growBinding(rewriter, ci->rhsArg, ci->rhsLoad, rhsChunk);
      growMatmulInit(rewriter, *ci, M, C);
      growOutputStore(rewriter, *ci, M, C);
      SmallVector<Type> argTypes(llvm::map_range(
          cloneFunc.getArguments(),
          [](BlockArgument a) { return a.getType(); }));
      cloneFunc.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
      SmallVector<Value> operands(2);
      operands[lhsArgNo] = lhs;
      operands[rhsArgNo] = wSlice;
      rewriter.setInsertionPoint(dispatch);
      auto sub = IREE::Flow::DispatchOp::create(
          rewriter, loc, cloneExport, /*workload=*/dispatch.getWorkload(),
          TypeRange{chunkResultType}, /*result_dims=*/ValueRange{}, operands,
          /*argument_dims=*/ValueRange{}, /*tied_operands=*/ArrayAttr{});
      if (npuAffinity) sub->setAttr("stream.affinity", npuAffinity);
      chunkOutputs.push_back(sub.getResult(0));
    }
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

std::unique_ptr<Pass> createAMDAIESplitLargeContractionDispatchesPass() {
  return std::make_unique<AMDAIESplitLargeContractionDispatchesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
