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
//
// With the native conv tile pipeline enabled (`native_conv` in the target
// config), a conv dispatch with its bias add fused, `fill(0) -> conv ->
// generic(addf(conv, bias), ...)`, is folded the same way along its input
// channels: the input gets ones channels after its IC channels (a host pad
// dispatch with 1 as the pad value, IC + 1 rounded up to a multiple of 8), and
// the constant filter gets the bias at input channel IC of its center tap and
// zeros in the other new channels. The bias is a binding or, when IREE inlined
// a small one into the executable, an `arith.constant`. The conv's results may
// be tied to an operand other than its input, filter and bias (a conv writing
// into the next layer's padded input).

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
  // The matmul value is an input of the consumer, not its init.
  if (consumer.getDpsInputs()[mmInput] != mmResult) return std::nullopt;
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

/// Returns the values of `bias` converted to `floatType` (round to nearest
/// even) as raw element bytes, one element per bias value. A DenseElementsAttr
/// whose values are all equal stores a single element, so its raw data cannot
/// be indexed per value: equal values are expanded here.
static SmallVector<char> getConvertedBiasBytes(DenseElementsAttr bias,
                                               FloatType floatType) {
  SmallVector<APFloat> values;
  for (APFloat v : bias.getValues<APFloat>()) {
    bool losesInfo;
    v.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
              &losesInfo);
    values.push_back(v);
  }
  auto attr = DenseElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(values.size())}, floatType),
      values);
  ArrayRef<char> raw = attr.getRawData();
  if (!attr.isSplat()) return SmallVector<char>(raw.begin(), raw.end());
  SmallVector<char> bytes;
  for (size_t i = 0, e = values.size(); i < e; ++i)
    bytes.append(raw.begin(), raw.end());
  return bytes;
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
  SmallVector<char> rowRaw = getConvertedBiasBytes(bias, floatType);
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
/// input and the bias `addf` is dropped (its uses read the matmul value). The
/// init keeps its map, so a consumer that also transposes the result still
/// does. A consumer that then only yields the matmul value through an identity
/// init map is removed. Erases the bias load and binding.
static void removeBias(IRRewriter &rewriter, BiasFoldInfo &info,
                       func::FuncOp func) {
  linalg::GenericOp oldConsumer = info.consumer;
  Value mmResult = info.matmul.getResult(0);
  Location loc = oldConsumer.getLoc();
  Block &oldBody = oldConsumer.getRegion().front();
  BlockArgument oldInit = oldBody.getArgument(2);
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
  AffineMap initMap =
      oldConsumer.getMatchingIndexingMap(oldConsumer.getDpsInitOperand(0));
  rewriter.setInsertionPoint(oldConsumer);
  auto newConsumer = rewriter.create<linalg::GenericOp>(
      loc, oldConsumer->getResultTypes(), ValueRange{mmResult},
      oldConsumer.getDpsInits(), ArrayRef<AffineMap>{identity, initMap},
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
  if (initMap.isIdentity() && llvm::hasSingleElement(*newBody) &&
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

/// A conv's bias in post-add form: `fill(0) -> conv -> generic(addf(conv,
/// bias), ...)`, the bias one value per output channel (the conv's last output
/// dim), from its own binding or an inline constant.
struct ConvBiasFoldInfo {
  linalg::LinalgOp conv;
  ConvDims dims;
  IREE::TensorExt::DispatchTensorLoadOp inLoad, filterLoad, biasLoad;
  BlockArgument inArg, filterArg, biasArg;  // biasLoad/biasArg null if inline
  arith::ConstantOp inlineBias;
  linalg::GenericOp consumer;
  arith::AddFOp biasAdd;
  unsigned convInput;  // the consumer input the conv result feeds
};

static std::optional<ConvBiasFoldInfo> getConvBiasInfo(func::FuncOp func) {
  SmallVector<linalg::LinalgOp> convs;
  func.walk([&](linalg::LinalgOp op) {
    if (succeeded(getConvDims(op))) convs.push_back(op);
  });
  if (convs.size() != 1) return std::nullopt;
  linalg::LinalgOp conv = convs.front();
  auto inLoad = getFullTensorLoad(conv.getDpsInputOperand(0)->get());
  auto filterLoad = getFullTensorLoad(conv.getDpsInputOperand(1)->get());
  if (!inLoad || !filterLoad) return std::nullopt;
  auto inArg = dyn_cast<BlockArgument>(inLoad.getSource());
  auto filterArg = dyn_cast<BlockArgument>(filterLoad.getSource());
  if (!inArg || !filterArg) return std::nullopt;
  auto fill = conv.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
    return std::nullopt;

  Value convResult = conv->getResult(0);
  if (!convResult.hasOneUse()) return std::nullopt;
  auto consumer = dyn_cast<linalg::GenericOp>(*convResult.getUsers().begin());
  if (!consumer || !isElementwise(consumer) ||
      consumer.getNumDpsInputs() != 2 || consumer.getNumDpsInits() != 1)
    return std::nullopt;
  unsigned convInput = consumer.getDpsInputs()[0] == convResult ? 0 : 1;
  // The conv value is an input of the consumer, not its init.
  if (consumer.getDpsInputs()[convInput] != convResult) return std::nullopt;
  OpOperand *biasOperand = consumer.getDpsInputOperand(1 - convInput);
  MLIRContext *ctx = consumer.getContext();
  unsigned rank = consumer.getNumLoops();
  AffineMap perChannel =
      AffineMap::get(rank, 0, {getAffineDimExpr(rank - 1, ctx)}, ctx);
  if (!consumer.getMatchingIndexingMap(consumer.getDpsInputOperand(convInput))
           .isIdentity() ||
      consumer.getMatchingIndexingMap(biasOperand) != perChannel)
    return std::nullopt;

  // The conv value and the bias are each read by one `addf` of the two.
  Block &body = consumer.getRegion().front();
  BlockArgument convValue = body.getArgument(convInput);
  BlockArgument biasValue = body.getArgument(1 - convInput);
  if (!convValue.hasOneUse() || !biasValue.hasOneUse()) return std::nullopt;
  auto biasAdd = dyn_cast<arith::AddFOp>(*convValue.getUsers().begin());
  if (!biasAdd || *biasValue.getUsers().begin() != biasAdd.getOperation())
    return std::nullopt;

  ConvBiasFoldInfo info{conv,    *getConvDims(conv), inLoad,   filterLoad,
                        {},      inArg,              filterArg, {},
                        {},      consumer,           biasAdd,   convInput};
  Value bias = biasOperand->get();
  if (auto biasLoad = getFullTensorLoad(bias)) {
    info.biasLoad = biasLoad;
    info.biasArg = dyn_cast<BlockArgument>(biasLoad.getSource());
    if (!info.biasArg) return std::nullopt;
  } else if (auto cstOp = bias.getDefiningOp<arith::ConstantOp>()) {
    info.inlineBias = cstOp;
  } else {
    return std::nullopt;
  }
  return info;
}

/// Returns conv `filter` (HWCF [KH, KW, IC, OC], or FHWC [OC, KH, KW, IC] when
/// `ocFirst`) with its input channels grown to `newIC`: input channel IC (the
/// input's first ones channel) holds `bias` at the center tap and zero at the
/// other taps, and the channels after it are zero. The bias is converted to
/// the filter element type. Null if the filter is splat or not a byte-sized
/// float.
static DenseElementsAttr appendBiasChannel(DenseElementsAttr filter,
                                           DenseElementsAttr bias,
                                           bool ocFirst, int64_t newIC) {
  auto type = cast<RankedTensorType>(filter.getType());
  auto floatType = dyn_cast<FloatType>(type.getElementType());
  if (filter.isSplat() || !floatType || floatType.getWidth() % 8 != 0)
    return {};
  int64_t eb = floatType.getWidth() / 8;
  ArrayRef<int64_t> s = type.getShape();
  int64_t KH = ocFirst ? s[1] : s[0], KW = ocFirst ? s[2] : s[1];
  int64_t IC = ocFirst ? s[3] : s[2], OC = ocFirst ? s[0] : s[3];
  SmallVector<char> biasRaw = getConvertedBiasBytes(bias, floatType);
  ArrayRef<char> src = filter.getRawData();
  // All-zero bytes are +0.0 for IEEE floats.
  SmallVector<char> dst(KH * KW * newIC * OC * eb, 0);
  auto dstIndex = [&](int64_t kh, int64_t kw, int64_t ic, int64_t oc) {
    return ocFirst ? ((oc * KH + kh) * KW + kw) * newIC + ic
                   : ((kh * KW + kw) * newIC + ic) * OC + oc;
  };
  if (ocFirst) {
    // [OC][KH][KW][IC]: each (oc, kh, kw) row of IC values moves as a whole.
    for (int64_t oc = 0; oc < OC; ++oc)
      for (int64_t kh = 0; kh < KH; ++kh)
        for (int64_t kw = 0; kw < KW; ++kw) {
          int64_t from = ((oc * KH + kh) * KW + kw) * IC;
          std::copy_n(src.begin() + from * eb, IC * eb,
                      dst.begin() + dstIndex(kh, kw, 0, oc) * eb);
        }
  } else {
    // [KH][KW][IC][OC]: each (kh, kw, ic) row of OC values moves as a whole.
    for (int64_t kh = 0; kh < KH; ++kh)
      for (int64_t kw = 0; kw < KW; ++kw)
        for (int64_t ic = 0; ic < IC; ++ic) {
          int64_t from = ((kh * KW + kw) * IC + ic) * OC;
          std::copy_n(src.begin() + from * eb, OC * eb,
                      dst.begin() + dstIndex(kh, kw, ic, 0) * eb);
        }
  }
  for (int64_t oc = 0; oc < OC; ++oc)
    std::copy_n(biasRaw.begin() + oc * eb, eb,
                dst.begin() + dstIndex(KH / 2, KW / 2, IC, oc) * eb);
  SmallVector<int64_t> shape =
      ocFirst ? SmallVector<int64_t>{OC, KH, KW, newIC}
              : SmallVector<int64_t>{KH, KW, newIC, OC};
  return DenseElementsAttr::getFromRawBuffer(
      RankedTensorType::get(shape, floatType), dst);
}

/// Rebuilds the conv's bias consumer without the bias (as `removeBias` does for
/// a matmul): the conv value is its only input and the bias `addf` is dropped.
/// The init keeps its map (IREE moves an NHWC -> NCHW transpose of the conv
/// result there). A consumer that then only yields the conv value through an
/// identity init map is removed. Erases the bias load and binding, or the
/// inline bias constant.
static void removeConvBias(IRRewriter &rewriter, ConvBiasFoldInfo &info,
                           func::FuncOp func) {
  linalg::GenericOp oldConsumer = info.consumer;
  Value convResult = info.conv->getResult(0);
  Location loc = oldConsumer.getLoc();
  Block &oldBody = oldConsumer.getRegion().front();
  BlockArgument oldInit = oldBody.getArgument(2);
  AffineMap identity = AffineMap::getMultiDimIdentityMap(
      oldConsumer.getNumLoops(), rewriter.getContext());
  AffineMap initMap =
      oldConsumer.getMatchingIndexingMap(oldConsumer.getDpsInitOperand(0));
  rewriter.setInsertionPoint(oldConsumer);
  auto newConsumer = rewriter.create<linalg::GenericOp>(
      loc, oldConsumer->getResultTypes(), ValueRange{convResult},
      oldConsumer.getDpsInits(), ArrayRef<AffineMap>{identity, initMap},
      oldConsumer.getIteratorTypesArray());
  Block *newBody = rewriter.createBlock(
      &newConsumer.getRegion(), {},
      {oldBody.getArgument(info.convInput).getType(), oldInit.getType()},
      {loc, loc});
  IRMapping mapping;
  mapping.map(oldBody.getArgument(info.convInput), newBody->getArgument(0));
  mapping.map(oldInit, newBody->getArgument(1));
  mapping.map(info.biasAdd.getResult(), newBody->getArgument(0));
  for (Operation &op : oldBody)
    if (&op != info.biasAdd.getOperation()) rewriter.clone(op, mapping);
  rewriter.replaceOp(oldConsumer, newConsumer->getResults());

  auto yield = cast<linalg::YieldOp>(newBody->getTerminator());
  if (initMap.isIdentity() && llvm::hasSingleElement(*newBody) &&
      yield.getOperand(0) == newBody->getArgument(0) &&
      newConsumer.getResult(0).getType() == convResult.getType())
    rewriter.replaceOp(newConsumer, convResult);

  if (info.biasLoad) {
    rewriter.eraseOp(info.biasLoad);
    (void)func.eraseArgument(info.biasArg.getArgNumber());
  } else if (info.inlineBias->use_empty()) {
    rewriter.eraseOp(info.inlineBias);
  }
}

/// Folds the bias of conv dispatch `dispatch` (see `ConvBiasFoldInfo`) into its
/// input channels. Emits an error and fails if the fold cannot be done.
static LogicalResult foldConvBias(
    IRRewriter &rewriter, ModuleOp module, IREE::Flow::DispatchOp dispatch,
    func::FuncOp func, ConvBiasFoldInfo &info, int numCallers,
    IREE::HAL::DeviceAffinityAttr hostAffinity, int &counter,
    SmallVectorImpl<IREE::Util::GlobalOp> &replacedGlobals) {
  auto fail = [&](const Twine &reason) {
    dispatch.emitError(
        "cannot fold the bias of this NPU conv dispatch into its input "
        "channels: ")
        << reason;
    return failure();
  };
  if (numCallers != 1) return fail("its executable has several callers");
  // The fold grows the input and filter bindings and erases the bias binding,
  // which anything else reading them would see.
  auto onlyFeedsOneOp = [](BlockArgument arg, Operation *load) {
    return arg.hasOneUse() && load->hasOneUse();
  };
  if (!onlyFeedsOneOp(info.inArg, info.inLoad) ||
      !onlyFeedsOneOp(info.filterArg, info.filterLoad) ||
      (info.biasLoad && !onlyFeedsOneOp(info.biasArg, info.biasLoad)))
    return fail(
        "its input, filter or bias is also read by another op of the "
        "dispatch");
  auto isStatic = [](Type t) {
    auto shaped = dyn_cast<ShapedType>(t);
    return !shaped || shaped.hasStaticShape();
  };
  if (!llvm::all_of(dispatch.getOperandTypes(), isStatic) ||
      !llvm::all_of(dispatch.getResultTypes(), isStatic))
    return fail("the dispatch has dynamically shaped operands or results");
  unsigned inArgNo = info.inArg.getArgNumber();
  unsigned filterArgNo = info.filterArg.getArgNumber();
  std::optional<unsigned> biasArgNo;
  if (info.biasArg) biasArgNo = info.biasArg.getArgNumber();
  // Results tied to the operands the fold replaces or removes would change
  // under it; a result tied to its own output operand is fine. The storage
  // holds argument indices.
  auto tiedAttr = dispatch->getAttrOfType<ArrayAttr>(
      IREE::Util::TiedOpInterface::getStorageAttrName());
  SmallVector<int64_t> tied;
  if (tiedAttr) {
    for (Attribute a : tiedAttr) {
      int64_t t = cast<IntegerAttr>(a).getInt();
      if (t == static_cast<int64_t>(inArgNo) ||
          t == static_cast<int64_t>(filterArgNo) ||
          (biasArgNo && t == static_cast<int64_t>(*biasArgNo)))
        return fail("a result is tied to the conv input, filter or bias");
      tied.push_back(t);
    }
  }
  Value in = dispatch.getArguments()[inArgNo];
  Value filter = dispatch.getArguments()[filterArgNo];
  auto inType = cast<RankedTensorType>(in.getType());
  auto inElemType = dyn_cast<FloatType>(inType.getElementType());
  auto filterType = cast<RankedTensorType>(filter.getType());
  if (!inElemType || !isa<FloatType>(filterType.getElementType()))
    return fail("the conv operands are not floats");
  IREE::Util::GlobalOp filterGlobal;
  DenseElementsAttr filterAttr = getConstantValue(filter, &filterGlobal);
  if (!filterAttr) return fail("the filter is not a constant");
  Value biasOperand = biasArgNo ? dispatch.getArguments()[*biasArgNo] : Value();
  DenseElementsAttr biasAttr =
      biasArgNo ? getConstantValue(biasOperand, nullptr)
                : getConstantValue(info.inlineBias.getResult(), nullptr);
  if (!biasAttr || !isa<FloatType>(biasAttr.getElementType()))
    return fail("the bias is not a float constant");

  // The input channels are the last input dim.
  int64_t ic = inType.getShape().back();
  int64_t newIC = llvm::alignTo(ic + 1, 8);
  DenseElementsAttr foldedAttr = appendBiasChannel(
      filterAttr, biasAttr, info.dims.filterOcFirst, newIC);
  if (!foldedAttr) return fail("the filter is not a dense float constant");
  SmallVector<int64_t> inShape(inType.getShape());
  inShape.back() = newIC;
  auto foldedType = cast<RankedTensorType>(foldedAttr.getType());

  // Executable: IC grown on the input and the filter, no bias.
  growBinding(rewriter, info.inArg, info.inLoad, inShape);
  growBinding(rewriter, info.filterArg, info.filterLoad, foldedType.getShape());
  removeConvBias(rewriter, info, func);
  SmallVector<Type> argTypes(llvm::map_range(
      func.getArguments(), [](BlockArgument a) { return a.getType(); }));
  func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));

  // Host: the input with ones channels, the filter with the bias channel.
  rewriter.setInsertionPoint(dispatch);
  Value inFolded =
      createPaddingDispatch(rewriter, module, in, inShape, hostAffinity,
                            counter, rewriter.getFloatAttr(inElemType, 1.0));
  Value filterFolded = createReplacementConstant(
      rewriter, module, dispatch.getLoc(),
      "__bias_folded_weight_" + std::to_string(counter++), foldedAttr,
      filterGlobal, replacedGlobals);
  dispatch.getArgumentsMutable().slice(inArgNo, 1).assign(inFolded);
  dispatch.getArgumentsMutable().slice(filterArgNo, 1).assign(filterFolded);
  if (biasArgNo) {
    dispatch.getArgumentsMutable().erase(*biasArgNo);
    // Results tied after the bias operand move down by one.
    if (tiedAttr) {
      for (int64_t &t : tied)
        if (t != IREE::Util::TiedOpInterface::kUntiedIndex &&
            t > static_cast<int64_t>(*biasArgNo))
          --t;
      dispatch->setAttr(IREE::Util::TiedOpInterface::getStorageAttrName(),
                        rewriter.getIndexArrayAttr(tied));
    }
    eraseDeadProducers(rewriter, biasOperand);
  }
  eraseDeadProducers(rewriter, filter);
  return success();
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
    IREE::HAL::ExecutableTargetAttr target =
        getDispatchTarget(dispatch, module);
    if (!target) continue;
    func::FuncOp func = getDispatchFunc(dispatch, module);
    if (!func) continue;
    std::optional<BiasFoldInfo> info = getPostAddBiasInfo(func);
    if (!info) info = getAccumulatorBiasInfo(func);
    if (!info) {
      // A conv bias (native conv pipeline only; with im2col the conv is a
      // matmul above).
      if (getConfigNativeConv(target)) {
        if (std::optional<ConvBiasFoldInfo> convInfo = getConvBiasInfo(func)) {
          if (failed(foldConvBias(rewriter, module, dispatch, func, *convInfo,
                                  numCallers[func.getOperation()],
                                  hostAffinity, counter, replacedGlobals)))
            return signalPassFailure();
        }
      }
      continue;
    }

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
    // K grows on the LHS and weight bindings and the bias binding goes away,
    // which anything else reading them would see.
    auto onlyFeedsOneOp = [](BlockArgument arg, Operation *load) {
      return arg.hasOneUse() && load->hasOneUse();
    };
    if (!onlyFeedsOneOp(info->lhsArg, info->lhsLoad) ||
        !onlyFeedsOneOp(info->rhsArg, info->rhsLoad) ||
        !onlyFeedsOneOp(info->biasArg, info->biasLoad))
      return fail(
          "its LHS, weight or bias is also read by another op of the "
          "dispatch");
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
                                 : "its accumulator input is not a broadcast "
                                   "of a float constant bias (a residual or "
                                   "partial sum cannot be folded)");

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
    Value rhsFolded = createReplacementConstant(
        rewriter, module, dispatch.getLoc(),
        "__bias_folded_weight_" + std::to_string(counter++), foldedAttr,
        weightGlobal, replacedGlobals);
    dispatch.getArgumentsMutable().slice(lhsArgNo, 1).assign(lhsFolded);
    dispatch.getArgumentsMutable().slice(rhsArgNo, 1).assign(rhsFolded);
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
