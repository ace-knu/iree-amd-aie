// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Folds the constant bias of an NPU contraction dispatch into its reduction
// dim, so the dispatch keeps two inputs: y = x . W + b = [x, 1] . [W; b]. The
// amd-aie backend has no DMA channel for a third input, so a matmul + bias +
// ReLU dispatch fails without it.
//
// Matches a bias holding one value per output column in either of two forms:
//   * post-add, `fill(0) -> matmul -> generic(addf(mm, bias), ...)`, as an
//     ONNX Gemm (FC layer) lowers to once the bias add is fused into the
//     contraction dispatch: the bias binding and its `addf` leave the
//     executable, and a consumer left with nothing to do is removed;
//   * accumulator, `matmul outs(bias broadcast)`, as an im2col convolution
//     lowers to: the accumulator is a host dispatch broadcasting a constant
//     [N] bias over M (or a global a `util.initializer` sets to one, where
//     const-eval hoisted it); the accumulator binding leaves the executable and
//     the matmul starts from `fill(0)`.
// The rewrite grows K by one: the LHS gets a ones column at K index k (a host
// pad dispatch with 1 as the pad value) and the constant weight gets the bias
// as its K row k (a new global, the bias rounded to the weight element type).
// Runs right before AMDAIEPadContractionDispatches, which then pads K+1 like
// any other K.
//
// A matched dispatch the fold cannot rewrite (a bias or weight that is not a
// compile-time constant, an executable with several callers) is an error: the
// backend would fail on its third input later with a less clear message.

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

#define DEBUG_TYPE "iree-amdaie-fold-bias-into-contraction-dispatches"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// A bias in a dispatch executable, loaded from its own binding: added to the
/// matmul value by an elementwise consumer (post-add), or the matmul's
/// accumulator (`consumer` null).
struct BiasFoldInfo {
  linalg::MatmulOp matmul;
  IREE::TensorExt::DispatchTensorLoadOp lhsLoad, rhsLoad, biasLoad;
  BlockArgument lhsArg, rhsArg, biasArg;
  linalg::GenericOp consumer;
  arith::AddFOp biasAdd;
  unsigned mmInput;  // the consumer input the matmul result feeds
};

/// Returns the single `linalg.matmul` of `func` with LHS and RHS from
/// full-tensor loads of bindings, or null.
static linalg::MatmulOp getBindingMatmul(func::FuncOp func) {
  SmallVector<linalg::MatmulOp> matmuls;
  func.walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
  if (matmuls.size() != 1) return {};
  linalg::MatmulOp matmul = matmuls.front();
  for (Value input : matmul.getInputs()) {
    auto load = getFullTensorLoad(input);
    if (!load || !isa<BlockArgument>(load.getSource())) return {};
  }
  return matmul;
}

/// Returns true if `map` reads one value per output column of an [M, N]
/// iteration space whose M is `m`: (d0, d1) -> (d1), or a [1, N] operand read
/// through (d0, d1) -> (0, d1) or, when M == 1, the identity.
static bool isPerColumnMap(AffineMap map, int64_t m) {
  MLIRContext *ctx = map.getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
  if (map.getNumDims() != 2) return false;
  if (map == AffineMap::get(2, 0, {d1}, ctx)) return true;
  if (map == AffineMap::get(2, 0, {getAffineConstantExpr(0, ctx), d1}, ctx))
    return true;
  return m == 1 && map.isIdentity();
}

static std::optional<BiasFoldInfo> getPostAddBiasInfo(func::FuncOp func) {
  linalg::MatmulOp matmul = getBindingMatmul(func);
  if (!matmul) return std::nullopt;
  auto fill = matmul.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
    return std::nullopt;
  auto lhsLoad = getFullTensorLoad(matmul.getInputs()[0]);
  auto rhsLoad = getFullTensorLoad(matmul.getInputs()[1]);
  auto lhsArg = cast<BlockArgument>(lhsLoad.getSource());
  auto rhsArg = cast<BlockArgument>(rhsLoad.getSource());

  Value mmResult = matmul.getResult(0);
  if (!mmResult.hasOneUse()) return std::nullopt;
  auto consumer = dyn_cast<linalg::GenericOp>(*mmResult.getUsers().begin());
  if (!consumer || !isElementwise(consumer) ||
      consumer.getNumDpsInputs() != 2 || consumer.getNumDpsInits() != 1)
    return std::nullopt;
  unsigned mmInput = consumer.getDpsInputs()[0] == mmResult ? 0 : 1;
  OpOperand *biasOperand = consumer.getDpsInputOperand(1 - mmInput);
  auto biasLoad = getFullTensorLoad(biasOperand->get());
  if (!biasLoad) return std::nullopt;
  auto biasArg = dyn_cast<BlockArgument>(biasLoad.getSource());
  if (!biasArg) return std::nullopt;
  int64_t m = cast<RankedTensorType>(mmResult.getType()).getDimSize(0);
  if (!consumer.getMatchingIndexingMap(consumer.getDpsInputOperand(mmInput))
           .isIdentity() ||
      !isPerColumnMap(consumer.getMatchingIndexingMap(biasOperand), m))
    return std::nullopt;

  // The matmul value and the bias are each read by one `addf` of the two.
  Block &body = consumer.getRegion().front();
  BlockArgument mmValue = body.getArgument(mmInput);
  BlockArgument biasValue = body.getArgument(1 - mmInput);
  if (!mmValue.hasOneUse() || !biasValue.hasOneUse()) return std::nullopt;
  auto biasAdd = dyn_cast<arith::AddFOp>(*mmValue.getUsers().begin());
  if (!biasAdd || *biasValue.getUsers().begin() != biasAdd.getOperation())
    return std::nullopt;
  return BiasFoldInfo{matmul, lhsLoad, rhsLoad,  biasLoad, lhsArg,
                      rhsArg, biasArg, consumer, biasAdd,  mmInput};
}

/// An accumulator bias: the matmul's init is loaded from its own binding.
static std::optional<BiasFoldInfo> getAccumulatorBiasInfo(func::FuncOp func) {
  linalg::MatmulOp matmul = getBindingMatmul(func);
  if (!matmul) return std::nullopt;
  auto accLoad = getFullTensorLoad(matmul.getDpsInits()[0]);
  if (!accLoad) return std::nullopt;
  auto accArg = dyn_cast<BlockArgument>(accLoad.getSource());
  if (!accArg) return std::nullopt;
  auto lhsLoad = getFullTensorLoad(matmul.getInputs()[0]);
  auto rhsLoad = getFullTensorLoad(matmul.getInputs()[1]);
  return BiasFoldInfo{matmul,
                      lhsLoad,
                      rhsLoad,
                      accLoad,
                      cast<BlockArgument>(lhsLoad.getSource()),
                      cast<BlockArgument>(rhsLoad.getSource()),
                      accArg,
                      /*consumer=*/{},
                      /*biasAdd=*/{},
                      /*mmInput=*/0};
}

/// Returns the constant [N] bias that accumulator `acc` broadcasts over M:
/// `acc` is the result of a dispatch whose executable only broadcasts a 1-D
/// constant (inline or a constant operand) through (d0, d1) -> (d1), or a load
/// of an immutable global a `util.initializer` sets to such a dispatch's
/// result (const-eval hoists a small broadcast). Null otherwise.
static DenseElementsAttr getBroadcastBias(Value acc, ModuleOp module) {
  if (auto loadOp = acc.getDefiningOp<IREE::Util::GlobalLoadOp>()) {
    auto globalOp = SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
        loadOp, loadOp.getGlobalAttr());
    if (!loadOp.getIsImmutable() || !globalOp || globalOp.getInitialValueAttr())
      return {};
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(globalOp, module);
    if (!uses) return {};
    IREE::Util::GlobalStoreOp store;
    for (SymbolTable::SymbolUse use : *uses) {
      auto candidate = dyn_cast<IREE::Util::GlobalStoreOp>(use.getUser());
      if (!candidate) continue;
      if (store || !candidate->getParentOfType<IREE::Util::InitializerOp>())
        return {};
      store = candidate;
    }
    if (!store) return {};
    acc = store.getValue();
  }
  auto dispatch = acc.getDefiningOp<IREE::Flow::DispatchOp>();
  if (!dispatch || dispatch.getNumResults() != 1) return {};
  func::FuncOp func = getDispatchFunc(dispatch, module);
  if (!func) return {};
  SmallVector<linalg::GenericOp> generics;
  func.walk([&](linalg::GenericOp op) { generics.push_back(op); });
  if (generics.size() != 1) return {};
  linalg::GenericOp broadcast = generics.front();
  MLIRContext *ctx = broadcast.getContext();
  if (broadcast.getNumDpsInputs() != 1 || broadcast.getNumDpsInits() != 1 ||
      broadcast.getNumLoops() != 2 ||
      broadcast.getIndexingMapsArray()[0] !=
          AffineMap::get(2, 0, {getAffineDimExpr(1, ctx)}, ctx) ||
      !broadcast.getIndexingMapsArray()[1].isIdentity())
    return {};
  Block &body = broadcast.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  if (!llvm::hasSingleElement(body) ||
      yield.getOperand(0) != body.getArgument(0))
    return {};
  Value input = broadcast.getDpsInputs()[0];
  if (auto cstOp = input.getDefiningOp<arith::ConstantOp>())
    return dyn_cast<DenseElementsAttr>(cstOp.getValue());
  auto load = getFullTensorLoad(input);
  auto arg = load ? dyn_cast<BlockArgument>(load.getSource()) : BlockArgument();
  if (!arg) return {};
  return getConstantValue(dispatch.getArguments()[arg.getArgNumber()], nullptr);
}

/// Returns `weight` ([K, N] or [N, K], K at `kPos`) with `bias` appended as K
/// row k, the bias converted to the weight element type. Null if the weight
/// is splat or not a byte-sized float.
static DenseElementsAttr appendBiasRow(DenseElementsAttr weight,
                                       DenseElementsAttr bias, unsigned kPos) {
  auto type = cast<RankedTensorType>(weight.getType());
  auto floatType = dyn_cast<FloatType>(type.getElementType());
  if (weight.isSplat() || !floatType || floatType.getWidth() % 8 != 0)
    return {};
  int64_t elemBytes = floatType.getWidth() / 8;
  SmallVector<int64_t> shape(type.getShape());
  int64_t k = shape[kPos], n = shape[1 - kPos];
  SmallVector<APFloat> row;
  for (APFloat v : bias.getValues<APFloat>()) {
    bool losesInfo;
    v.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
              &losesInfo);
    row.push_back(v);
  }
  ArrayRef<char> rowRaw =
      DenseElementsAttr::get(RankedTensorType::get({n}, floatType), row)
          .getRawData();
  ArrayRef<char> src = weight.getRawData();
  SmallVector<char> dst;
  dst.reserve(src.size() + rowRaw.size());
  if (kPos == 0) {
    // [K, N]: the bias is one more row.
    dst.append(src.begin(), src.end());
    dst.append(rowRaw.begin(), rowRaw.end());
  } else {
    // [N, K]: each of the N rows gets its bias value appended.
    for (int64_t j = 0; j < n; ++j) {
      auto rowBegin = src.begin() + j * k * elemBytes;
      dst.append(rowBegin, rowBegin + k * elemBytes);
      dst.append(rowRaw.begin() + j * elemBytes,
                 rowRaw.begin() + (j + 1) * elemBytes);
    }
  }
  shape[kPos] = k + 1;
  return DenseElementsAttr::getFromRawBuffer(
      RankedTensorType::get(shape, floatType), dst);
}

/// Rebuilds the bias consumer without the bias: the matmul value is its only
/// input and the bias `addf` is dropped (its uses read the matmul value). A
/// consumer that then only yields the matmul value is removed. Erases the bias
/// load and binding.
static void removeBias(IRRewriter &rewriter, BiasFoldInfo &info,
                       func::FuncOp func) {
  linalg::GenericOp oldConsumer = info.consumer;
  Value mmResult = info.matmul.getResult(0);
  Location loc = oldConsumer.getLoc();
  Block &oldBody = oldConsumer.getRegion().front();
  BlockArgument oldInit = oldBody.getArgument(2);
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
  rewriter.setInsertionPoint(oldConsumer);
  auto newConsumer = rewriter.create<linalg::GenericOp>(
      loc, oldConsumer->getResultTypes(), ValueRange{mmResult},
      oldConsumer.getDpsInits(), ArrayRef<AffineMap>{identity, identity},
      oldConsumer.getIteratorTypesArray());
  Block *newBody = rewriter.createBlock(
      &newConsumer.getRegion(), {},
      {oldBody.getArgument(info.mmInput).getType(), oldInit.getType()},
      {loc, loc});
  IRMapping mapping;
  mapping.map(oldBody.getArgument(info.mmInput), newBody->getArgument(0));
  mapping.map(oldInit, newBody->getArgument(1));
  mapping.map(info.biasAdd.getResult(), newBody->getArgument(0));
  for (Operation &op : oldBody)
    if (&op != info.biasAdd.getOperation()) rewriter.clone(op, mapping);
  rewriter.replaceOp(oldConsumer, newConsumer->getResults());

  auto yield = cast<linalg::YieldOp>(newBody->getTerminator());
  if (llvm::hasSingleElement(*newBody) &&
      yield.getOperand(0) == newBody->getArgument(0) &&
      newConsumer.getResult(0).getType() == mmResult.getType())
    rewriter.replaceOp(newConsumer, mmResult);

  rewriter.eraseOp(info.biasLoad);
  (void)func.eraseArgument(info.biasArg.getArgNumber());
}

/// Starts the matmul from `fill(0)` instead of its accumulator binding, and
/// erases that binding and its load.
static void replaceAccumulatorWithZeroFill(IRRewriter &rewriter,
                                           BiasFoldInfo &info,
                                           func::FuncOp func) {
  Location loc = info.matmul.getLoc();
  auto accType = cast<RankedTensorType>(info.biasLoad.getResult().getType());
  rewriter.setInsertionPoint(info.matmul);
  Value empty = rewriter.create<tensor::EmptyOp>(loc, accType.getShape(),
                                                 accType.getElementType());
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(accType.getElementType()));
  auto fill =
      rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
  info.matmul.setDpsInitOperand(0, fill.getResult(0));
  rewriter.eraseOp(info.biasLoad);
  (void)func.eraseArgument(info.biasArg.getArgNumber());
}

class AMDAIEFoldBiasIntoContractionDispatchesPass
    : public impl::AMDAIEFoldBiasIntoContractionDispatchesBase<
          AMDAIEFoldBiasIntoContractionDispatchesPass> {
 public:
  void runOnOperation() override;
};

void AMDAIEFoldBiasIntoContractionDispatchesPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());

  SmallVector<IREE::Flow::DispatchOp> dispatches;
  DenseMap<Operation *, int> numCallers;
  module.walk([&](IREE::Flow::DispatchOp d) {
    dispatches.push_back(d);
    if (func::FuncOp func = getDispatchFunc(d, module))
      ++numCallers[func.getOperation()];
  });

  IREE::HAL::DeviceAffinityAttr hostAffinity = getHostAffinity(module);
  if (!hostAffinity) return;  // need a host device for the ones-column dispatch
  int counter = firstFreeHelperIndex(module);
  SmallVector<IREE::Util::GlobalOp> replacedGlobals;

  for (IREE::Flow::DispatchOp dispatch : dispatches) {
    if (!getDispatchTarget(dispatch, module)) continue;
    func::FuncOp func = getDispatchFunc(dispatch, module);
    if (!func) continue;
    std::optional<BiasFoldInfo> info = getPostAddBiasInfo(func);
    if (!info) info = getAccumulatorBiasInfo(func);
    if (!info) continue;

    // From here on the dispatch has a bias as its third input, which the
    // backend cannot take: fold it or fail with the reason.
    auto fail = [&](const Twine &reason) {
      dispatch.emitError(
          "cannot fold the bias of this NPU contraction "
          "dispatch into K: ")
          << reason;
      signalPassFailure();
    };
    if (numCallers[func.getOperation()] != 1)
      return fail("its executable has several callers");
    if (llvm::any_of(llvm::seq<unsigned>(0, dispatch.getNumResults()),
                     [&](unsigned r) {
                       return dispatch.getTiedResultOperandIndex(r).has_value();
                     }))
      return fail("the dispatch has tied results");
    // The ones column, the K+1 LHS and the new weight are built with static
    // shapes.
    auto isStatic = [](Type t) {
      auto shaped = dyn_cast<ShapedType>(t);
      return !shaped || shaped.hasStaticShape();
    };
    if (!llvm::all_of(dispatch.getOperandTypes(), isStatic) ||
        !llvm::all_of(dispatch.getResultTypes(), isStatic))
      return fail("the dispatch has dynamically shaped operands or results");
    unsigned lhsArgNo = info->lhsArg.getArgNumber();
    unsigned rhsArgNo = info->rhsArg.getArgNumber();
    unsigned biasArgNo = info->biasArg.getArgNumber();
    Value lhs = dispatch.getArguments()[lhsArgNo];
    Value rhs = dispatch.getArguments()[rhsArgNo];
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    auto lhsElemType = dyn_cast<FloatType>(lhsType.getElementType());
    if (!lhsElemType || !isa<FloatType>(rhsType.getElementType()))
      return fail("the matmul operands are not floats");
    IREE::Util::GlobalOp weightGlobal;
    DenseElementsAttr weightAttr = getConstantValue(rhs, &weightGlobal);
    if (!weightAttr) return fail("the weight is not a constant");
    Value biasOperand = dispatch.getArguments()[biasArgNo];
    DenseElementsAttr biasAttr = info->consumer
                                     ? getConstantValue(biasOperand, nullptr)
                                     : getBroadcastBias(biasOperand, module);
    if (!biasAttr || !isa<FloatType>(biasAttr.getElementType()))
      return fail(info->consumer ? "the bias is not a float constant"
                                 : "the accumulator is not a broadcast of a "
                                   "float constant bias");

    SmallVector<AffineMap> maps = info->matmul.getIndexingMapsArray();
    unsigned lhsKPos = operandDimForLoop(maps[0], /*K=*/2);
    unsigned rhsKPos = operandDimForLoop(maps[1], /*K=*/2);
    DenseElementsAttr foldedAttr = appendBiasRow(weightAttr, biasAttr, rhsKPos);
    if (!foldedAttr) return fail("the weight is not a dense float constant");
    SmallVector<int64_t> lhsShape(lhsType.getShape());
    ++lhsShape[lhsKPos];
    auto foldedType = cast<RankedTensorType>(foldedAttr.getType());

    // Executable: K + 1 on both operands, no bias.
    growBinding(rewriter, info->lhsArg, info->lhsLoad, lhsShape);
    growBinding(rewriter, info->rhsArg, info->rhsLoad, foldedType.getShape());
    if (info->consumer)
      removeBias(rewriter, *info, func);
    else
      replaceAccumulatorWithZeroFill(rewriter, *info, func);
    SmallVector<Type> argTypes(llvm::map_range(
        func.getArguments(), [](BlockArgument a) { return a.getType(); }));
    func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));

    // Host: the LHS with a ones column, the weight with the bias row.
    rewriter.setInsertionPoint(dispatch);
    Value lhsFolded =
        createPaddingDispatch(rewriter, module, lhs, lhsShape, hostAffinity,
                              counter, rewriter.getFloatAttr(lhsElemType, 1.0));
    OpBuilder moduleBuilder = OpBuilder::atBlockBegin(module.getBody());
    auto foldedGlobal = IREE::Util::GlobalOp::create(
        moduleBuilder, dispatch.getLoc(),
        "__bias_folded_weight_" + std::to_string(counter++),
        /*isMutable=*/false, foldedType, std::optional<TypedAttr>(foldedAttr));
    foldedGlobal.setPrivate();
    if (weightGlobal) {
      if (Attribute affinity = weightGlobal->getAttr("stream.affinity.default"))
        foldedGlobal->setAttr("stream.affinity.default", affinity);
      replacedGlobals.push_back(weightGlobal);
    }
    auto rhsFolded = cast<IREE::Util::GlobalLoadOp>(
        foldedGlobal.createLoadOp(dispatch.getLoc(), rewriter).getOperation());
    rhsFolded.setIsImmutable(true);
    dispatch.getArgumentsMutable().slice(lhsArgNo, 1).assign(lhsFolded);
    dispatch.getArgumentsMutable()
        .slice(rhsArgNo, 1)
        .assign(rhsFolded.getResult());
    dispatch.getArgumentsMutable().erase(biasArgNo);
    // The accumulator's broadcast dispatch (or global load) has no other use,
    // nor the old weight and the reshapes it was read through.
    Operation *biasProducer = biasOperand.getDefiningOp();
    if (biasProducer && biasProducer->use_empty() &&
        isa<IREE::Flow::DispatchOp, IREE::Util::GlobalLoadOp>(biasProducer))
      rewriter.eraseOp(biasProducer);
    eraseDeadProducers(rewriter, rhs);
  }

  eraseDeadGlobals(rewriter, module, replacedGlobals);
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFoldBiasIntoContractionDispatchesPass() {
  return std::make_unique<AMDAIEFoldBiasIntoContractionDispatchesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
