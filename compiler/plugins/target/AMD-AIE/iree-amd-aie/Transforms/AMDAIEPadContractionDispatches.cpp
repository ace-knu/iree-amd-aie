// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Flow-phase passes that pad NPU dispatches at the dispatch boundary:
// AMDAIEPadContractionDispatches (contractions to tile multiples) and
// AMDAIEPadPoolingDispatches (max pooling OH), the latter at the bottom of this
// file. They share the host crop dispatch builder below; the helpers they share
// with AMDAIEFoldBiasIntoContractionDispatches are in
// AMDAIEDispatchRewriteUtils.h.
//
// --- AMDAIEPadContractionDispatches ---
// Pads the operands of NPU contraction dispatches up to the target's pack-peel
// tile multiples, at the Flow dispatch boundary.
//
// Runs right after `AMDAIEAssignDeviceAffinities`, when every `flow.dispatch`
// already carries a `stream.affinity` and dispatch regions are outlined. This
// pass assumes the amd-aie contraction dispatch is a plain matmul
// (`empty -> fill(0) -> matmul -> store`), optionally with a unary elementwise
// epilogue (ReLU, truncf) between the matmul and the store; a bias has been
// folded into K by AMDAIEFoldBiasIntoContractionDispatches or split into its own
// dispatch upstream. For such a dispatch pinned to an amd-aie device
// whose M/K dims are not multiples of the target tiles, it:
//   * zero-pads the matmul operands to the padded shape *outside* the dispatch:
//     a constant operand (a weight) at compile time into a new global, any
//     other on the host by a pad dispatch, and
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

#include "iree-amd-aie/Transforms/AMDAIEDispatchRewriteUtils.h"
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"

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
/// the tile is lowered to it, or of the full GEMV K tile (`kGemvKTile`) once K
/// reaches it: KernelDispatch picks that tile only for a K it divides. Returns
/// nullopt if `target` is not a recognized amd-aie target.
static std::optional<PaddingMultiples> getPaddingMultiples(
    IREE::HAL::ExecutableTargetAttr target, int64_t m, int64_t k,
    Type lhsElemType, Type rhsElemType, Type accElemType, bool transposeB) {
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
        /*k=*/useMatvec ? (k >= kGemvKTile ? kGemvKTile : kMatvecUKernelK)
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

/// A plain matmul (`empty -> fill -> matmul [-> epilogue] -> store`) inside a
/// dispatch executable, with the ops/bindings needed to grow its M/K dims. Sizes are
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
  linalg::GenericOp epilogue;  // unary elementwise op on the result, or null
};

/// Inspects the func for a single plain matmul: LHS/RHS from full-tensor loads
/// of bindings, an `empty`+`fill` output init, and the matmul result stored to
/// the output binding directly or through a unary elementwise epilogue with an
/// `empty` init.
static std::optional<MatmulInfo> getMatmulInfo(func::FuncOp func) {
  SmallVector<linalg::MatmulOp> matmuls;
  func.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  if (matmuls.size() != 1) return std::nullopt;
  linalg::MatmulOp matmul = matmuls.front();
  // Padding computes static padded shapes.
  if (!llvm::all_of(matmul->getOperandTypes(), [](Type t) {
        return cast<ShapedType>(t).hasStaticShape();
      }))
    return std::nullopt;
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
  linalg::GenericOp epilogue;
  if (store.getValue() != matmul.getResult(0)) {
    epilogue = store.getValue().getDefiningOp<linalg::GenericOp>();
    if (!epilogue || !isElementwise(epilogue) ||
        epilogue.getNumDpsInputs() != 1 || epilogue.getNumDpsInits() != 1 ||
        epilogue.getDpsInputs()[0] != matmul.getResult(0) ||
        !matmul.getResult(0).hasOneUse() ||
        !epilogue.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>() ||
        !llvm::all_of(epilogue.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }))
      return std::nullopt;
  }
  auto outArg = dyn_cast<BlockArgument>(store.getTarget());
  if (!outArg) return std::nullopt;
  return MatmulInfo{matmul, lhsLoad, rhsLoad, lhsArg, rhsArg,
                    fill,   empty,   store,   outArg, epilogue};
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
  // The epilogue may share the `empty` as its init; it is grown after this.
  if (info.empty.use_empty()) rewriter.eraseOp(info.empty);
}

/// Grows the epilogue (if any) to [mPad, nPad] after the matmul result has
/// grown, then the output binding, and rewrites the store to write the full
/// padded result.
static void growOutputStore(IRRewriter &rewriter, MatmulInfo &info,
                            int64_t mPad, int64_t nPad) {
  Value result = info.matmul.getResult(0);
  if (linalg::GenericOp epilogue = info.epilogue) {
    auto empty = epilogue.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>();
    rewriter.setInsertionPoint(epilogue);
    auto paddedType = RankedTensorType::get(
        {mPad, nPad}, getElementTypeOrSelf(empty.getType()));
    Value newEmpty = rewriter.create<tensor::EmptyOp>(
        empty.getLoc(), paddedType.getShape(), paddedType.getElementType());
    epilogue.setDpsInitOperand(0, newEmpty);
    epilogue.getResult(0).setType(paddedType);
    if (empty.use_empty()) rewriter.eraseOp(empty);
    result = epilogue.getResult(0);
  }
  auto dtt = cast<IREE::TensorExt::DispatchTensorType>(info.outArg.getType());
  auto newTensorType = RankedTensorType::get(
      {mPad, nPad}, dtt.asRankedTensorType().getElementType());
  info.outArg.setType(IREE::TensorExt::DispatchTensorType::get(dtt.getAccess(),
                                                               newTensorType));
  rewriter.setInsertionPoint(info.store);
  rewriter.create<IREE::TensorExt::DispatchTensorStoreOp>(
      info.store.getLoc(), result, info.outArg,
      /*targetDynamicDims=*/ValueRange{});
  rewriter.eraseOp(info.store);
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

/// Returns `attr` zero-padded at the high end of each dim to `dstShape`. Null
/// if it is splat or its elements are not byte-sized.
static DenseElementsAttr padWithZeros(DenseElementsAttr attr,
                                      ArrayRef<int64_t> dstShape) {
  auto type = cast<RankedTensorType>(attr.getType());
  Type elemType = type.getElementType();
  if (attr.isSplat() || type.getRank() == 0 || !elemType.isIntOrFloat() ||
      elemType.getIntOrFloatBitWidth() % 8 != 0)
    return {};
  int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  ArrayRef<int64_t> srcShape = type.getShape();
  int64_t rank = type.getRank();
  int64_t rowBytes = srcShape.back() * elemBytes;
  int64_t numRows = type.getNumElements() / srcShape.back();
  // Zero bits are a zero of every int and float type; copy each innermost
  // source row to its place in the larger result.
  ArrayRef<char> src = attr.getRawData();
  SmallVector<char> dst(ShapedType::getNumElements(dstShape) * elemBytes, 0);
  for (int64_t r = 0; r < numRows; ++r) {
    int64_t rem = r, dstRow = 0, stride = 1;
    for (int64_t d = rank - 2; d >= 0; --d) {
      dstRow += (rem % srcShape[d]) * stride;
      rem /= srcShape[d];
      stride *= dstShape[d];
    }
    llvm::copy(src.slice(r * rowBytes, rowBytes),
               dst.begin() + dstRow * dstShape.back() * elemBytes);
  }
  return DenseElementsAttr::getFromRawBuffer(
      RankedTensorType::get(dstShape, elemType), dst);
}

/// Returns `v` zero-padded to `dstShape` at compile time, as a load of a new
/// immutable global, when `v` is a dense constant (see `getConstantValue`) read
/// only here; the constant's global, if any, is added to `replacedGlobals`.
/// Null otherwise, for a host pad dispatch to do it at run time instead.
static Value createPaddedConstant(
    IRRewriter &rewriter, ModuleOp module, Value v, ArrayRef<int64_t> dstShape,
    int &counter, SmallVectorImpl<IREE::Util::GlobalOp> &replacedGlobals) {
  for (Value root = v;;) {
    if (!root.hasOneUse()) return {};
    auto reshapeOp = root.getDefiningOp<IREE::Flow::TensorReshapeOp>();
    if (!reshapeOp) break;
    root = reshapeOp.getSource();
  }
  IREE::Util::GlobalOp global;
  DenseElementsAttr attr = getConstantValue(v, &global);
  if (!attr) return {};
  if (global) {
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(global, module);
    if (!uses || !llvm::hasSingleElement(*uses)) return {};
  }
  DenseElementsAttr padded = padWithZeros(attr, dstShape);
  if (!padded) return {};
  OpBuilder moduleBuilder = OpBuilder::atBlockBegin(module.getBody());
  auto paddedGlobal = IREE::Util::GlobalOp::create(
      moduleBuilder, v.getLoc(),
      "__padded_constant_" + std::to_string(counter++), /*isMutable=*/false,
      padded.getType(), std::optional<TypedAttr>(padded));
  paddedGlobal.setPrivate();
  if (global) {
    if (Attribute affinity = global->getAttr("stream.affinity.default"))
      paddedGlobal->setAttr("stream.affinity.default", affinity);
    replacedGlobals.push_back(global);
  }
  auto load = cast<IREE::Util::GlobalLoadOp>(
      paddedGlobal.createLoadOp(v.getLoc(), rewriter).getOperation());
  load.setIsImmutable(true);
  return load.getResult();
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
  SmallVector<IREE::Util::GlobalOp> replacedGlobals;

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
        target, m, k, lhsType.getElementType(), rhsType.getElementType(),
        outElemType, /*transposeB=*/rhsNPos < rhsKPos);
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

    // Pad this caller's operands to the padded shapes, a constant at compile
    // time and anything else on the host, then rewire.
    rewriter.setInsertionPoint(dispatch);
    auto pad = [&](Value v, ArrayRef<int64_t> shape) -> Value {
      if (shape == cast<RankedTensorType>(v.getType()).getShape()) return v;
      if (Value padded = createPaddedConstant(rewriter, module, v, shape,
                                              counter, replacedGlobals))
        return padded;
      return createPaddingDispatch(rewriter, module, v, shape, hostAffinity,
                                   counter);
    };
    Value lhs = dispatch.getArguments()[lhsArgNo];
    Value rhs = dispatch.getArguments()[rhsArgNo];
    dispatch.getArgumentsMutable().slice(lhsArgNo, 1).assign(pad(lhs, lhsPad));
    dispatch.getArgumentsMutable().slice(rhsArgNo, 1).assign(pad(rhs, rhsPad));
    eraseDeadProducers(rewriter, lhs);
    eraseDeadProducers(rewriter, rhs);

    // Grow this caller's result to [mPad, nPad] and crop it back to [m, n] on
    // the host with an extract_slice dispatch (handles the inner N dim too).
    if (padOut) {
      // The stored element type, the epilogue's when there is one.
      Value result = dispatch.getResult(0);
      result.setType(RankedTensorType::get(
          {mPad, nPad}, getElementTypeOrSelf(result.getType())));
      rewriter.setInsertionPointAfter(dispatch);
      Value cropped =
          createCropDispatch(rewriter, module, result, {m, n}, hostAffinity,
                             counter);
      result.replaceAllUsesExcept(cropped, cropped.getDefiningOp());
    }
  }
  eraseDeadGlobals(rewriter, module, replacedGlobals);
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
