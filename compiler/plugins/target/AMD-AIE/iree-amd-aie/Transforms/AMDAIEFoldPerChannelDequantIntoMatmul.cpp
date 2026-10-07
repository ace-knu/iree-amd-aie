// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>
#include <limits>
#include <optional>

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-fold-per-channel-dequant-into-matmul"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// An int8 tensor dequantized by a `linalg.generic`, as the ONNX importer
/// writes it:
///
///   sitofp(x) * s                           -- a per-tensor activation
///   sitofp(extsi(x) - zp) * s               -- the same, asymmetric
///   sitofp(extsi(w) - extsi(zp[n])) * s[n]  -- a per-channel weight
///
/// `scale` is either a scalar f32 constant (captured by the body) or a 1-D
/// f32 constant indexed by the generic's last loop. A per-channel zero point
/// has to be a constant zero; a scalar one captured by the body may be any
/// constant.
struct Dequantized {
  Value quantized;
  /// One entry for a per-tensor scale, `N` for a per-channel one.
  SmallVector<float> scale;
  int64_t zeroPoint = 0;
};

/// The constant `v` is, as floats, or nothing.
std::optional<SmallVector<float>> readF32(Value v) {
  DenseElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr)) ||
      !attr.getElementType().isF32())
    return std::nullopt;
  SmallVector<float> out;
  for (APFloat f : attr.getValues<APFloat>()) out.push_back(f.convertToFloat());
  return out;
}

bool isZeroConstant(Value v) {
  DenseIntElementsAttr attr;
  if (matchPattern(v, m_Constant(&attr)))
    return llvm::all_of(attr.getValues<APInt>(),
                        [](const APInt &i) { return i.isZero(); });
  return matchPattern(v, m_Zero());
}

FailureOr<Dequantized> matchDequantize(Value value) {
  auto op = value.getDefiningOp<linalg::GenericOp>();
  if (!op || op.getNumDpsInits() != 1 || op.getNumReductionLoops() != 0)
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!resultType || !resultType.getElementType().isF32()) return failure();
  Block &body = op.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  auto mul = yield.getValues()[0].getDefiningOp<arith::MulFOp>();
  if (!mul) return failure();
  auto toFloat = mul.getLhs().getDefiningOp<arith::SIToFPOp>();
  if (!toFloat) return failure();
  unsigned rank = op.getNumLoops();

  // The value being converted: the quantized operand itself, or it minus a
  // zero point -- a per-channel operand that has to be zero, or a scalar
  // constant.
  Value converted = toFloat.getIn();
  int64_t zeroPoint = 0;
  if (auto sub = converted.getDefiningOp<arith::SubIOp>()) {
    auto lhs = sub.getLhs().getDefiningOp<arith::ExtSIOp>();
    if (!lhs) return failure();
    APInt zp;
    if (matchPattern(sub.getRhs(), m_ConstantInt(&zp))) {
      zeroPoint = zp.getSExtValue();
    } else {
      auto rhs = sub.getRhs().getDefiningOp<arith::ExtSIOp>();
      if (!rhs) return failure();
      auto zpArg = dyn_cast<BlockArgument>(rhs.getIn());
      if (!zpArg || zpArg.getOwner() != &body) return failure();
      if (!isZeroConstant(op.getDpsInputs()[zpArg.getArgNumber()]))
        return failure();
    }
    converted = lhs.getIn();
  }
  auto quantArg = dyn_cast<BlockArgument>(converted);
  if (!quantArg || quantArg.getOwner() != &body ||
      !quantArg.getType().isInteger(8))
    return failure();
  OpOperand *quantOperand = op.getDpsInputOperand(quantArg.getArgNumber());
  if (!op.getMatchingIndexingMap(quantOperand).isIdentity()) return failure();

  Dequantized result;
  result.quantized = quantOperand->get();
  result.zeroPoint = zeroPoint;
  Value scale = mul.getRhs();
  if (auto scaleArg = dyn_cast<BlockArgument>(scale);
      scaleArg && scaleArg.getOwner() == &body) {
    // Per channel: a 1-D constant along the last (output-channel) loop.
    OpOperand *scaleOperand = op.getDpsInputOperand(scaleArg.getArgNumber());
    AffineMap map = op.getMatchingIndexingMap(scaleOperand);
    if (map.getNumResults() != 1 ||
        map.getResult(0) != getAffineDimExpr(rank - 1, op.getContext()))
      return failure();
    std::optional<SmallVector<float>> values = readF32(scaleOperand->get());
    if (!values) return failure();
    result.scale = *values;
  } else {
    // Per tensor: a scalar captured from above.
    FloatAttr attr;
    if (!matchPattern(scale, m_Constant(&attr)) || !attr.getType().isF32())
      return failure();
    result.scale = {static_cast<float>(attr.getValueAsDouble())};
  }
  return result;
}

/// Rewrites
///
///   %a = dequantize(%x : i8)            -- per tensor, scale sx
///   %b = dequantize(%w : i8)            -- per output channel, scale sw[n]
///   %y = linalg.matmul ins(%a, %b) outs(fill 0.0) -> f32
///
/// into
///
///   %acc = linalg.matmul ins(%x, %w) outs(fill 0) -> i32
///   %y   = sitofp(%acc) * (sx * sw[n])
///
/// which is exact: the scales factor out of the sum over K. An activation
/// with a zero point zx adds a term that does not depend on it,
///
///   sum_k (x[m,k] - zx) * w[k,n] = acc[m,n] - zx * sum_k w[k,n],
///
/// so for a constant weight the tail subtracts a compile-time [N] table:
///
///   %y   = sitofp(%acc - c[n]) * (sx * sw[n]),   c[n] = zx * sum_k w[k,n]
///
/// The per-tensor form already arrives this way (as `linalg.quantized_matmul`); the ONNX
/// importer only falls back to a float matmul because that named op cannot
/// carry a per-channel zero point. Left as it was, both dequantizations and a
/// float matmul would land in one dispatch, which the NPU cannot lower.
///
/// After it, the per-channel scale is an ordinary `[N]` operand of the
/// matmul's elementwise tail, and the requantization that follows fuses into
/// it as it does for any per-tensor matmul.
///
/// The same holds for a `linalg.batch_matmul` with a unit batch (see
/// `matchBatchOperand`) -- the shape BERT's FC2 arrives in.

/// A contraction's two operands, once dequantized.
struct Operands {
  Dequantized lhs, rhs;
};

LogicalResult checkScales(const Operands &ops, RankedTensorType resultType) {
  // Only the case the per-tensor path cannot already take.
  if (ops.lhs.scale.size() != 1 || ops.rhs.scale.size() == 1) return failure();
  // A weight zero point would need the activation's row sums at run time.
  if (ops.rhs.zeroPoint != 0) return failure();
  int64_t n = resultType.getDimSize(resultType.getRank() - 1);
  if (ShapedType::isDynamic(n) || static_cast<int64_t>(ops.rhs.scale.size()) != n)
    return failure();
  return success();
}

/// The int8 values of a constant `[K, N]` weight, row-major, or nothing.
std::optional<SmallVector<int8_t>> readI8(Value v) {
  Attribute attr;
  if (!matchPattern(v, m_Constant(&attr))) return std::nullopt;
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (!dense.getElementType().isInteger(8)) return std::nullopt;
    SmallVector<int8_t> out;
    out.reserve(dense.getNumElements());
    for (APInt i : dense.getValues<APInt>())
      out.push_back(static_cast<int8_t>(i.getSExtValue()));
    return out;
  }
  if (auto resource = dyn_cast<DenseI8ResourceElementsAttr>(attr)) {
    std::optional<ArrayRef<int8_t>> data = resource.tryGetAsArrayRef();
    if (!data) return std::nullopt;
    return SmallVector<int8_t>(data->begin(), data->end());
  }
  return std::nullopt;
}

/// `c[n] = zx * sum_k w[k,n]` for an activation zero point `zx` and the
/// constant weight `w : [K, N]`; empty when `zx` is zero, failure when the
/// weight is not a readable constant or a term leaves i32.
FailureOr<SmallVector<int32_t>> zeroPointCorrection(const Operands &ops) {
  if (ops.lhs.zeroPoint == 0) return SmallVector<int32_t>{};
  auto type = dyn_cast<RankedTensorType>(ops.rhs.quantized.getType());
  if (!type || type.getRank() != 2 || !type.hasStaticShape()) return failure();
  std::optional<SmallVector<int8_t>> w = readI8(ops.rhs.quantized);
  int64_t k = type.getDimSize(0), n = type.getDimSize(1);
  if (!w || static_cast<int64_t>(w->size()) != k * n) return failure();
  SmallVector<int64_t> sums(n, 0);
  for (int64_t r = 0; r < k; ++r)
    for (int64_t c = 0; c < n; ++c) sums[c] += (*w)[r * n + c];
  SmallVector<int32_t> out;
  out.reserve(n);
  for (int64_t s : sums) {
    int64_t c = ops.lhs.zeroPoint * s;
    if (c < std::numeric_limits<int32_t>::min() ||
        c > std::numeric_limits<int32_t>::max())
      return failure();
    out.push_back(static_cast<int32_t>(c));
  }
  return out;
}

/// Build `sitofp(acc [- c[n]]) * (sx * sw[n])` over `acc`, and replace
/// `contraction`.
void replaceWithScaledTail(PatternRewriter &rewriter, Operation *contraction,
                           Value acc, const Operands &ops,
                           ArrayRef<int32_t> correction) {
  Location loc = contraction->getLoc();
  auto accType = cast<RankedTensorType>(acc.getType());
  int64_t rank = accType.getRank();
  int64_t n = accType.getDimSize(rank - 1);
  Type f32 = rewriter.getF32Type();
  SmallVector<float> combined;
  combined.reserve(n);
  for (float s : ops.rhs.scale) combined.push_back(ops.lhs.scale.front() * s);
  auto scaleType = RankedTensorType::get({n}, f32);
  Value scales = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(scaleType, ArrayRef<float>(combined)));

  MLIRContext *ctx = rewriter.getContext();
  AffineMap identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
  AffineMap column =
      AffineMap::get(rank, 0, {getAffineDimExpr(rank - 1, ctx)}, ctx);
  Value outInit = rewriter.create<tensor::EmptyOp>(
      loc, tensor::getMixedSizes(rewriter, loc, acc), f32);
  SmallVector<Value> inputs{acc, scales};
  SmallVector<AffineMap> maps{identity, column};
  if (!correction.empty()) {
    auto corrType = RankedTensorType::get({n}, rewriter.getI32Type());
    inputs.push_back(rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(corrType, correction)));
    maps.push_back(column);
  }
  maps.push_back(identity);
  auto scaled = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{outInit.getType()}, inputs, ValueRange{outInit}, maps,
      SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value a = args[0];
        if (!correction.empty()) a = b.create<arith::SubIOp>(l, a, args[2]);
        Value f = b.create<arith::SIToFPOp>(l, f32, a);
        b.create<linalg::YieldOp>(l, b.create<arith::MulFOp>(l, f, args[1])
                                         .getResult());
      });
  rewriter.replaceOp(contraction, scaled.getResults());
}

/// A zero-filled i32 accumulator shaped like `like`.
Value zeroAccumulator(PatternRewriter &rewriter, Location loc, Value like) {
  Value init = rewriter.create<tensor::EmptyOp>(
      loc, tensor::getMixedSizes(rewriter, loc, like), rewriter.getI32Type());
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  return rewriter.create<linalg::FillOp>(loc, zero, init).getResult(0);
}

bool isZeroFilled(Value init) {
  auto fill = init.getDefiningOp<linalg::FillOp>();
  return fill && matchPattern(fill.getInputs()[0], m_AnyZeroFloat());
}

struct FoldPerChannelDequantIntoMatmul : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    if (!matmul.hasPureTensorSemantics()) return failure();
    auto resultType = cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!resultType.getElementType().isF32() || resultType.getRank() != 2)
      return failure();
    if (matmul.getIndexingMapsArray() !=
        linalg::MatmulOp::getDefaultIndexingMaps(matmul.getContext()))
      return failure();
    if (!isZeroFilled(matmul.getOutputs()[0])) return failure();
    FailureOr<Dequantized> lhs = matchDequantize(matmul.getInputs()[0]);
    FailureOr<Dequantized> rhs = matchDequantize(matmul.getInputs()[1]);
    if (failed(lhs) || failed(rhs)) return failure();
    Operands ops{*lhs, *rhs};
    if (failed(checkScales(ops, resultType))) return failure();
    FailureOr<SmallVector<int32_t>> correction = zeroPointCorrection(ops);
    if (failed(correction)) return failure();

    Location loc = matmul.getLoc();
    Value accZero = zeroAccumulator(rewriter, loc, matmul.getOutputs()[0]);
    Value acc = rewriter
                    .create<linalg::MatmulOp>(
                        loc, TypeRange{accZero.getType()},
                        ValueRange{ops.lhs.quantized, ops.rhs.quantized},
                        ValueRange{accZero})
                    .getResult(0);
    replaceWithScaledTail(rewriter, matmul, acc, ops, *correction);
    return success();
  }
};

/// A unit-batch operand of a `batch_matmul`, traced back to its int8 source.
/// Three shapes reach this pass:
///   - a 3-D dequantization of a 3-D int8 tensor (the activation);
///   - a 2-D dequantization given a unit batch by
///     `expand_shape [[0, 1], [2]]`;
///   - a 2-D dequantization broadcast to the batch by a copy-only
///     `linalg.generic` with map `(d0, d1, d2) -> (d1, d2)` (the weight).
/// `rebuild` gives the int8 source the same batch the float operand had.
struct BatchOperand {
  Dequantized dequant;
  std::function<Value(PatternRewriter &, Location, Value)> rebuild;
};

FailureOr<BatchOperand> matchBatchOperand(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || type.getRank() != 3 || type.getDimSize(0) != 1) return failure();
  if (FailureOr<Dequantized> d = matchDequantize(v); succeeded(d)) {
    BatchOperand op{*d, [](PatternRewriter &, Location, Value q) { return q; }};
    return op;
  }
  if (auto expand = v.getDefiningOp<tensor::ExpandShapeOp>()) {
    SmallVector<ReassociationIndices> reassoc =
        expand.getReassociationIndices();
    if (expand.getSrcType().getRank() != 2 || reassoc.size() != 2 ||
        reassoc[0] != ReassociationIndices{0, 1} ||
        reassoc[1] != ReassociationIndices{2})
      return failure();
    FailureOr<Dequantized> d = matchDequantize(expand.getSrc());
    if (failed(d)) return failure();
    BatchOperand op{*d, [expand](PatternRewriter &rewriter, Location loc,
                                 Value q) mutable {
      auto i8Type = RankedTensorType::get(expand.getResultType().getShape(),
                                          rewriter.getI8Type());
      return rewriter
          .create<tensor::ExpandShapeOp>(loc, i8Type, q,
                                         expand.getReassociationIndices(),
                                         expand.getMixedOutputShape())
          .getResult();
    }};
    return op;
  }
  if (auto bcast = v.getDefiningOp<linalg::GenericOp>()) {
    if (bcast.getNumDpsInputs() != 1 || bcast.getNumDpsInits() != 1 ||
        bcast.getNumReductionLoops() != 0)
      return failure();
    auto yield = cast<linalg::YieldOp>(bcast.getRegion().front().getTerminator());
    if (yield.getValues()[0] != bcast.getRegion().front().getArgument(0))
      return failure();
    MLIRContext *ctx = bcast.getContext();
    AffineMap expected = AffineMap::get(
        3, 0, {getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx);
    SmallVector<AffineMap> maps = bcast.getIndexingMapsArray();
    if (maps[0] != expected || !maps[1].isIdentity()) return failure();
    FailureOr<Dequantized> d = matchDequantize(bcast.getDpsInputs()[0]);
    if (failed(d)) return failure();
    BatchOperand op{*d, [bcast, maps](PatternRewriter &rewriter, Location loc,
                                      Value q) mutable {
      Type i8 = rewriter.getI8Type();
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, cast<RankedTensorType>(bcast.getResult(0).getType()).getShape(),
          i8);
      return rewriter
          .create<linalg::GenericOp>(
              loc, TypeRange{init.getType()}, ValueRange{q}, ValueRange{init},
              maps, bcast.getIteratorTypesArray(),
              [](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(l, args[0]);
              })
          .getResult(0);
    }};
    return op;
  }
  return failure();
}

struct FoldPerChannelDequantIntoBatchMatmul
    : OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    if (!matmul.hasPureTensorSemantics()) return failure();
    auto resultType = cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!resultType.getElementType().isF32() || resultType.getRank() != 3 ||
        resultType.getDimSize(0) != 1)
      return failure();
    if (matmul.getIndexingMapsArray() !=
        linalg::BatchMatmulOp::getDefaultIndexingMaps(matmul.getContext()))
      return failure();
    if (!isZeroFilled(matmul.getOutputs()[0])) return failure();
    FailureOr<BatchOperand> lhs = matchBatchOperand(matmul.getInputs()[0]);
    FailureOr<BatchOperand> rhs = matchBatchOperand(matmul.getInputs()[1]);
    if (failed(lhs) || failed(rhs)) return failure();
    Operands ops{lhs->dequant, rhs->dequant};
    if (failed(checkScales(ops, resultType))) return failure();
    FailureOr<SmallVector<int32_t>> correction = zeroPointCorrection(ops);
    if (failed(correction)) return failure();

    Location loc = matmul.getLoc();
    Value lhsI8 = lhs->rebuild(rewriter, loc, ops.lhs.quantized);
    Value rhsI8 = rhs->rebuild(rewriter, loc, ops.rhs.quantized);
    Value accZero = zeroAccumulator(rewriter, loc, matmul.getOutputs()[0]);
    Value acc = rewriter
                    .create<linalg::BatchMatmulOp>(
                        loc, TypeRange{accZero.getType()},
                        ValueRange{lhsI8, rhsI8}, ValueRange{accZero})
                    .getResult(0);
    replaceWithScaledTail(rewriter, matmul, acc, ops, *correction);
    return success();
  }
};

class AMDAIEFoldPerChannelDequantIntoMatmulPass
    : public impl::AMDAIEFoldPerChannelDequantIntoMatmulBase<
          AMDAIEFoldPerChannelDequantIntoMatmulPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<FoldPerChannelDequantIntoMatmul,
                    FoldPerChannelDequantIntoBatchMatmul>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEFoldPerChannelDequantIntoMatmulPass() {
  return std::make_unique<AMDAIEFoldPerChannelDequantIntoMatmulPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
