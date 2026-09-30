// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-fold-per-channel-dequant-into-matmul"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// An int8 tensor dequantized by a `linalg.generic`, as the ONNX importer
/// writes it:
///
///   sitofp(x) * s                       -- a per-tensor activation
///   sitofp(extsi(w) - extsi(zp[n])) * s[n]  -- a per-channel weight
///
/// `scale` is either a scalar f32 constant (captured by the body) or a 1-D
/// f32 constant indexed by the generic's last loop. A zero point, when
/// present, has to be a constant zero.
struct Dequantized {
  Value quantized;
  /// One entry for a per-tensor scale, `N` for a per-channel one.
  SmallVector<float> scale;
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
  // zero point that has to be zero.
  Value converted = toFloat.getIn();
  if (auto sub = converted.getDefiningOp<arith::SubIOp>()) {
    auto lhs = sub.getLhs().getDefiningOp<arith::ExtSIOp>();
    auto rhs = sub.getRhs().getDefiningOp<arith::ExtSIOp>();
    if (!lhs || !rhs) return failure();
    auto zpArg = dyn_cast<BlockArgument>(rhs.getIn());
    if (!zpArg || zpArg.getOwner() != &body) return failure();
    if (!isZeroConstant(op.getDpsInputs()[zpArg.getArgNumber()]))
      return failure();
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
/// which is exact: the scales factor out of the sum over K. The per-tensor
/// form already arrives this way (as `linalg.quantized_matmul`); the ONNX
/// importer only falls back to a float matmul because that named op cannot
/// carry a per-channel zero point. Left as it was, both dequantizations and a
/// float matmul would land in one dispatch, which the NPU cannot lower.
///
/// After it, the per-channel scale is an ordinary `[N]` operand of the
/// matmul's elementwise tail, and the requantization that follows fuses into
/// it as it does for any per-tensor matmul.
struct FoldPerChannelDequantIntoMatmul : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    if (!matmul.hasPureTensorSemantics()) return failure();
    auto resultType = cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!resultType.getElementType().isF32() || resultType.getRank() != 2)
      return failure();
    if (!matmul.getIndexingMapsArray().empty() &&
        matmul.getIndexingMapsArray() !=
            matmul.getDefaultIndexingMaps(matmul.getContext()))
      return failure();
    auto fill = matmul.getOutputs()[0].getDefiningOp<linalg::FillOp>();
    if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat()))
      return failure();
    FailureOr<Dequantized> lhs = matchDequantize(matmul.getInputs()[0]);
    FailureOr<Dequantized> rhs = matchDequantize(matmul.getInputs()[1]);
    if (failed(lhs) || failed(rhs)) return failure();
    // Only the case the per-tensor path cannot already take.
    if (lhs->scale.size() != 1 || rhs->scale.size() == 1) return failure();
    int64_t n = resultType.getDimSize(1);
    if (ShapedType::isDynamic(n) ||
        static_cast<int64_t>(rhs->scale.size()) != n)
      return failure();

    Location loc = matmul.getLoc();
    Type i32 = rewriter.getI32Type();
    Type f32 = rewriter.getF32Type();
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, matmul.getOutputs()[0]);
    Value accInit = rewriter.create<tensor::EmptyOp>(loc, sizes, i32);
    Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    Value accZero =
        rewriter.create<linalg::FillOp>(loc, zero, accInit).getResult(0);
    Value acc = rewriter
                    .create<linalg::MatmulOp>(
                        loc, TypeRange{accZero.getType()},
                        ValueRange{lhs->quantized, rhs->quantized},
                        ValueRange{accZero})
                    .getResult(0);

    SmallVector<float> combined;
    combined.reserve(n);
    for (float s : rhs->scale) combined.push_back(lhs->scale.front() * s);
    auto scaleType = RankedTensorType::get({n}, f32);
    Value scales = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(scaleType, ArrayRef<float>(combined)));

    MLIRContext *ctx = rewriter.getContext();
    AffineMap identity = AffineMap::getMultiDimIdentityMap(2, ctx);
    AffineMap column = AffineMap::get(2, 0, {getAffineDimExpr(1, ctx)}, ctx);
    Value outInit = rewriter.create<tensor::EmptyOp>(loc, sizes, f32);
    auto scaled = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{outInit.getType()}, ValueRange{acc, scales},
        ValueRange{outInit}, ArrayRef<AffineMap>{identity, column, identity},
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value f = b.create<arith::SIToFPOp>(l, f32, args[0]);
          b.create<linalg::YieldOp>(l, b.create<arith::MulFOp>(l, f, args[1])
                                           .getResult());
        });
    rewriter.replaceOp(matmul, scaled.getResults());
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
    patterns.insert<FoldPerChannelDequantIntoMatmul>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEFoldPerChannelDequantIntoMatmulPass() {
  return std::make_unique<AMDAIEFoldPerChannelDequantIntoMatmulPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
