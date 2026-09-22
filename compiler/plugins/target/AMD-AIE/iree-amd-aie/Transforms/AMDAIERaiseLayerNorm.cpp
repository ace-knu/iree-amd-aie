// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIELayerNormUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-raise-layernorm"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Build a `linalg.generic` whose body is filled by `bodyBuilder`.
Value buildGeneric(OpBuilder &b, Location loc, Type resultType,
                   ValueRange inputs, Value init, ArrayRef<AffineMap> maps,
                   ArrayRef<utils::IteratorType> iterators,
                   function_ref<Value(OpBuilder &, ValueRange)> bodyBuilder) {
  auto op = b.create<linalg::GenericOp>(
      loc, TypeRange{resultType}, inputs, ValueRange{init}, maps, iterators,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value result = bodyBuilder(nested, args);
        nested.create<linalg::YieldOp>(nestedLoc, result);
      });
  return op.getResult(0);
}

/// Read an `[N]`-shaped f32 constant, however it is stored: small weights stay
/// inline as a `DenseElementsAttr`, real ones arrive as a resource blob.
bool readF32Constant(Value v, int64_t n, SmallVectorImpl<double> &out) {
  ElementsAttr attr;
  if (!matchPattern(v, m_Constant(&attr))) return false;
  auto shaped = dyn_cast<ShapedType>(attr.getType());
  if (!shaped || shaped.getNumElements() != n || !shaped.getElementType().isF32())
    return false;
  out.clear();
  out.reserve(n);
  if (auto resource = dyn_cast<DenseF32ResourceElementsAttr>(attr)) {
    std::optional<ArrayRef<float>> data = resource.tryGetAsArrayRef();
    if (!data) return false;
    out.assign(data->begin(), data->end());
    return true;
  }
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense) return false;
  for (APFloat f : dense.getValues<APFloat>()) {
    bool lost = false;
    f.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &lost);
    out.push_back(f.convertToDouble());
  }
  return true;
}

/// Fold the scales into a single `[2, N]` bf16 constant: row 0 is
/// `gamma / outputScale` and row 1 is `beta / outputScale`.
///
/// This is what lets the kernel be free of scalar float work. The output scale
/// then never appears again, and the input scale cancels out of the
/// normalization entirely, surviving only inside `epsilon / inputScale^2`.
///
/// The weights are read here rather than left as ops so that the result is a
/// real `DenseElementsAttr`. It has to be, because it does not travel as an
/// operand: a core tile has two incoming DMA channels and the residual add
/// needs both, so this one is placed in the core's own data memory instead, and
/// that placement needs the bytes at compile time. Returns failure when gamma
/// or beta is not a compile-time constant, in which case nothing is rewritten.
FailureOr<DenseElementsAttr> foldPackedGammaBeta(OpBuilder &b, Value gamma,
                                                 Value beta, double outputScale,
                                                 int64_t featureSize) {
  SmallVector<double> g, bt;
  if (!readF32Constant(gamma, featureSize, g)) return failure();
  if (!readF32Constant(beta, featureSize, bt)) return failure();
  auto bf16 = cast<FloatType>(b.getBF16Type());
  SmallVector<APFloat> packed;
  packed.reserve(2 * featureSize);
  auto toBF16 = [&](double v) {
    APFloat f(v / outputScale);
    bool lost = false;
    f.convert(bf16.getFloatSemantics(), APFloat::rmNearestTiesToEven, &lost);
    return f;
  };
  for (double v : g) packed.push_back(toBF16(v));
  for (double v : bt) packed.push_back(toBF16(v));
  auto type = RankedTensorType::get({2, featureSize}, bf16);
  return cast<DenseElementsAttr>(DenseElementsAttr::get(type, packed));
}

/// Turn a residual add into an integer-domain producer of wider samples.
///
/// The add itself never enters a microkernel ABI. It becomes an ordinary
/// elementwise op computing, entirely in integers,
///
///   h = (a * Ma + b * Mb + 2^(k-1)) >> k
///
/// where `Ma` and `Mb` put both inputs on one common step `s_h`. Choosing that
/// step finer than either input scale is what avoids a second quantization:
/// squeezing the sum back into int8 would round it to 7 bits, whereas int16
/// holds it with several fractional bits to spare. The step is reported back so
/// the caller can scale epsilon by it -- that is the only place the input scale
/// survives.
///
/// `computeResidualMultipliers` reports failure when the two scales are too far
/// apart for the sum to fit int16, in which case nothing is rewritten.
constexpr int kResidualShift = 8;

/// The fixed-point multipliers, decided from the scales alone so that the
/// decision can be made before anything is rewritten.
struct ResidualMultipliers {
  int64_t ma;
  int64_t mb;
  double commonStep;
};

FailureOr<ResidualMultipliers> computeResidualMultipliers(double scaleA,
                                                          double scaleB) {
  constexpr int64_t kMaxI16 = 32767;
  double finer = std::min(scaleA, scaleB);
  if (!(finer > 0.0)) return failure();
  double ratio = std::max(scaleA, scaleB) / finer;
  // As many extra fractional bits as the int16 range allows.
  int64_t extra = 1;
  while (extra < 256 &&
         127.0 * (extra * 2) * (1.0 + ratio) <= static_cast<double>(kMaxI16))
    extra *= 2;
  if (127.0 * extra * (1.0 + ratio) > static_cast<double>(kMaxI16))
    return failure();
  double commonStep = finer / static_cast<double>(extra);
  return ResidualMultipliers{
      std::llround(scaleA / commonStep * (1 << kResidualShift)),
      std::llround(scaleB / commonStep * (1 << kResidualShift)), commonStep};
}

Value buildResidualProducer(OpBuilder &b, Location loc, Value a, Value b8,
                            const ResidualMultipliers &m) {
  constexpr int kShift = kResidualShift;
  int64_t ma = m.ma;
  int64_t mb = m.mb;

  MLIRContext *ctx = b.getContext();
  AffineExpr e0, e1;
  bindDims(ctx, e0, e1);
  AffineMap map2d = AffineMap::get(2, 0, {e0, e1}, ctx);
  const SmallVector<utils::IteratorType> pp = {utils::IteratorType::parallel,
                                               utils::IteratorType::parallel};
  auto aType = cast<RankedTensorType>(a.getType());
  auto outType =
      RankedTensorType::get(aType.getShape(), b.getIntegerType(16));
  Value init = b.create<tensor::EmptyOp>(loc, outType.getShape(),
                                         outType.getElementType());
  Type i32 = b.getI32Type();
  Value cMa = b.create<arith::ConstantOp>(loc, i32, b.getI32IntegerAttr(ma));
  Value cMb = b.create<arith::ConstantOp>(loc, i32, b.getI32IntegerAttr(mb));
  Value cHalf = b.create<arith::ConstantOp>(
      loc, i32, b.getI32IntegerAttr(1 << (kShift - 1)));
  Value cShift =
      b.create<arith::ConstantOp>(loc, i32, b.getI32IntegerAttr(kShift));
  return buildGeneric(
      b, loc, outType, ValueRange{a, b8}, init, {map2d, map2d, map2d}, pp,
      [&](OpBuilder &nb, ValueRange args) -> Value {
        Value ai = nb.create<arith::ExtSIOp>(loc, i32, args[0]);
        Value bi = nb.create<arith::ExtSIOp>(loc, i32, args[1]);
        Value pa = nb.create<arith::MulIOp>(loc, ai, cMa);
        Value pb = nb.create<arith::MulIOp>(loc, bi, cMb);
        Value sum = nb.create<arith::AddIOp>(loc, pa, pb);
        Value biased = nb.create<arith::AddIOp>(loc, sum, cHalf);
        Value shifted = nb.create<arith::ShRSIOp>(loc, biased, cShift);
        return nb.create<arith::TruncIOp>(loc, nb.getIntegerType(16), shifted);
      });
}

/// Rebuild the LayerNorm chain on dynamically shaped tensors.
///
/// `iree_linalg_ext.custom_op` hands its region fully dynamic slices of the
/// operands, so the chain cannot simply be moved inside: every intermediate
/// `tensor.empty` and `linalg.fill` in the original is statically shaped.
/// Rebuilding it is also what keeps the body in one canonical form for the
/// microkernel matcher to recognise later.
Value buildLayerNormBody(OpBuilder &b, Location loc, Value input,
                         Value gammaBeta, Value init,
                         const QuantizedLayerNormDAG &dag) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  AffineExpr d0, d1;
  bindDims(ctx, d0, d1);
  AffineMap map2d = AffineMap::get(2, 0, {d0, d1}, ctx);
  AffineMap mapRow = AffineMap::get(2, 0, {d0}, ctx);
  AffineMap mapCol = AffineMap::get(2, 0, {d1}, ctx);
  AffineMap map1d = AffineMap::get(1, 0, {d0}, ctx);
  const SmallVector<utils::IteratorType> pp = {utils::IteratorType::parallel,
                                               utils::IteratorType::parallel};
  const SmallVector<utils::IteratorType> pr = {utils::IteratorType::parallel,
                                               utils::IteratorType::reduction};
  const SmallVector<utils::IteratorType> par = {utils::IteratorType::parallel};

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value rows = b.create<tensor::DimOp>(loc, input, c0);
  Value cols = b.create<tensor::DimOp>(loc, input, c1);
  auto dyn2dF32 = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32);
  auto dyn1dF32 = RankedTensorType::get({ShapedType::kDynamic}, f32);
  Value empty2d = b.create<tensor::EmptyOp>(loc, dyn2dF32, ValueRange{rows, cols});
  Value empty1d = b.create<tensor::EmptyOp>(loc, dyn1dF32, ValueRange{rows});

  auto constF32 = [&](double v) -> Value {
    return b.create<arith::ConstantOp>(loc, f32, b.getF32FloatAttr(v));
  };
  Value zero = constF32(0.0);
  Value featureSize = constF32(static_cast<double>(dag.featureSize));

  // The samples, as integers: the input scale is not applied here, it lives in
  // the scaled epsilon below.
  Value x = buildGeneric(b, loc, dyn2dF32, ValueRange{input}, empty2d,
                         {map2d, map2d}, pp,
                         [&](OpBuilder &nb, ValueRange args) -> Value {
                           return nb.create<arith::SIToFPOp>(loc, f32, args[0]);
                         });

  // One row of the packed constant, widened back to f32.
  Value emptyCol = b.create<tensor::EmptyOp>(loc, dyn1dF32, ValueRange{cols});
  auto packedRow = [&](int64_t row) -> Value {
    SmallVector<OpFoldResult> offsets = {b.getIndexAttr(row), b.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {b.getIndexAttr(1), cols};
    SmallVector<OpFoldResult> strides = {b.getIndexAttr(1), b.getIndexAttr(1)};
    auto sliceType =
        RankedTensorType::get({ShapedType::kDynamic}, b.getBF16Type());
    Value slice = b.create<tensor::ExtractSliceOp>(loc, sliceType, gammaBeta,
                                                  offsets, sizes, strides);
    return buildGeneric(b, loc, dyn1dF32, ValueRange{slice}, emptyCol,
                        {map1d, map1d}, par,
                        [&](OpBuilder &nb, ValueRange args) -> Value {
                          return nb.create<arith::ExtFOp>(loc, f32, args[0]);
                        });
  };

  auto sumOver = [&](Value src) -> Value {
    Value fill = b.create<linalg::FillOp>(loc, ValueRange{zero},
                                          ValueRange{empty1d}).getResult(0);
    return buildGeneric(b, loc, dyn1dF32, ValueRange{src}, fill,
                        {map2d, mapRow}, pr,
                        [&](OpBuilder &nb, ValueRange args) -> Value {
                          return nb.create<arith::AddFOp>(loc, args[0], args[1]);
                        });
  };
  auto divideByN = [&](Value src) -> Value {
    return buildGeneric(b, loc, dyn1dF32, ValueRange{src}, empty1d,
                        {map1d, map1d}, par,
                        [&](OpBuilder &nb, ValueRange args) -> Value {
                          return nb.create<arith::DivFOp>(loc, args[0], featureSize);
                        });
  };
  auto broadcastRow = [&](Value src) -> Value {
    return buildGeneric(b, loc, dyn2dF32, ValueRange{src}, empty2d,
                        {mapRow, map2d}, pp,
                        [&](OpBuilder &, ValueRange args) -> Value { return args[0]; });
  };

  Value mean = broadcastRow(divideByN(sumOver(x)));
  Value centred = buildGeneric(
      b, loc, dyn2dF32, ValueRange{x, mean}, empty2d, {map2d, map2d, map2d}, pp,
      [&](OpBuilder &nb, ValueRange args) -> Value {
        return nb.create<arith::SubFOp>(loc, args[0], args[1]);
      });
  Value squared = buildGeneric(
      b, loc, dyn2dF32, ValueRange{centred}, empty2d, {map2d, map2d}, pp,
      [&](OpBuilder &nb, ValueRange args) -> Value {
        return nb.create<arith::MulFOp>(loc, args[0], args[0]);
      });
  Value variance = divideByN(sumOver(squared));
  Value epsilon = constF32(dag.epsilon / (dag.sampleStep * dag.sampleStep));
  Value shifted = buildGeneric(
      b, loc, dyn1dF32, ValueRange{variance}, empty1d, {map1d, map1d}, par,
      [&](OpBuilder &nb, ValueRange args) -> Value {
        return nb.create<arith::AddFOp>(loc, args[0], epsilon);
      });
  Value rstd = broadcastRow(buildGeneric(
      b, loc, dyn1dF32, ValueRange{shifted}, empty1d, {map1d, map1d}, par,
      [&](OpBuilder &nb, ValueRange args) -> Value {
        return nb.create<math::RsqrtOp>(loc, args[0]);
      }));

  Value normalized = buildGeneric(
      b, loc, dyn2dF32, ValueRange{centred, rstd}, empty2d,
      {map2d, map2d, map2d}, pp, [&](OpBuilder &nb, ValueRange args) -> Value {
        return nb.create<arith::MulFOp>(loc, args[0], args[1]);
      });
  Value scaled = buildGeneric(
      b, loc, dyn2dF32, ValueRange{normalized, packedRow(0)}, empty2d,
      {map2d, mapCol, map2d}, pp, [&](OpBuilder &nb, ValueRange args) -> Value {
        return nb.create<arith::MulFOp>(loc, args[0], args[1]);
      });
  Value shiftedOut = buildGeneric(
      b, loc, dyn2dF32, ValueRange{scaled, packedRow(1)}, empty2d,
      {map2d, mapCol, map2d}, pp, [&](OpBuilder &nb, ValueRange args) -> Value {
        return nb.create<arith::AddFOp>(loc, args[0], args[1]);
      });

  // Round and saturate. There is no divide: the output scale is already in the
  // packed gamma and beta.
  Value lo = constF32(-128.0);
  Value hi = constF32(127.0);
  // Write into a fresh destination rather than the region's output argument.
  // The result is written in full, and leaving the argument unused is what
  // tells the promotion it does not have to copy the destination in -- which
  // would otherwise put the same L2 buffer on both sides of the core and leave
  // its object fifo without a tile assignment.
  auto dynOutType = RankedTensorType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic},
      cast<RankedTensorType>(init.getType()).getElementType());
  Value outInit =
      b.create<tensor::EmptyOp>(loc, dynOutType, ValueRange{rows, cols});
  return buildGeneric(
      b, loc, dynOutType, ValueRange{shiftedOut}, outInit, {map2d, map2d}, pp,
      [&](OpBuilder &nb, ValueRange args) -> Value {
        Value rounded = nb.create<math::RoundEvenOp>(loc, args[0]);
        Value clampLo = nb.create<arith::MaximumFOp>(loc, rounded, lo);
        Value clampHi = nb.create<arith::MinimumFOp>(loc, clampLo, hi);
        return nb.create<arith::FPToSIOp>(loc, nb.getI8Type(), clampHi);
      });
}

struct RaiseLayerNormPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp quantizeOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<QuantizedLayerNormDAG> maybeDag =
        matchQuantizedLayerNorm(quantizeOp);
    if (failed(maybeDag)) return failure();
    const QuantizedLayerNormDAG &dag = *maybeDag;

    // Only the shapes the microkernel is built for. Anything else keeps the
    // ordinary generic lowering: this raise is an approximate bf16/int8
    // kernel-specific rewrite, not a general one.
    if (dag.featureSize != 768) return failure();

    // Everything that can refuse the rewrite is decided before a single op is
    // created, so a refusal really does leave the generic chain untouched.
    FailureOr<DenseElementsAttr> maybeGammaBeta = foldPackedGammaBeta(
        rewriter, dag.gamma, dag.beta, dag.outputScale, dag.featureSize);
    if (failed(maybeGammaBeta)) return failure();
    ResidualMultipliers multipliers{0, 0, dag.inputScale};
    if (dag.residual) {
      FailureOr<ResidualMultipliers> maybeMultipliers =
          computeResidualMultipliers(dag.inputScale, dag.residualScale);
      if (failed(maybeMultipliers)) return failure();
      multipliers = *maybeMultipliers;
    }

    Location loc = quantizeOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto outType = cast<RankedTensorType>(quantizeOp.getResultTypes()[0]);
    auto inType = cast<RankedTensorType>(dag.input.getType());

    // The chain arrives as `1 x rows x features`. Drop the unit batch
    // dimension around the new op so that the op itself, and the tiling that
    // follows it, only ever deal with rows and features.
    SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}};
    RankedTensorType flatInType = RankedTensorType::get(
        {dag.numRows, dag.featureSize}, inType.getElementType());
    auto flatOutType = RankedTensorType::get({dag.numRows, dag.featureSize},
                                             outType.getElementType());
    rewriter.setInsertionPoint(quantizeOp);
    Value flatInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, flatInType, dag.input, reassoc);
    // A residual add becomes an integer producer of int16 samples, right in
    // front of the normalization and at the same rank, so nothing sits between
    // them. Epsilon then carries that producer's step instead of an input
    // scale.
    double sampleStep = dag.inputScale;
    if (dag.residual) {
      Value flatResidual = rewriter.create<tensor::CollapseShapeOp>(
          loc, flatInType, dag.residual, reassoc);
      flatInput = buildResidualProducer(rewriter, loc, flatInput, flatResidual,
                                        multipliers);
      flatInType = cast<RankedTensorType>(flatInput.getType());
      sampleStep = multipliers.commonStep;
    }
    dag.sampleStep = sampleStep;

    // One parallel loop over rows; the feature dimension rides along as a
    // symbol, which is how `custom_op` spells "this dimension is never tiled".
    AffineExpr d0;
    bindDims(ctx, d0);
    // One symbol, the feature length, which is never tiled. Gamma and beta are
    // not operands -- they ride on the op as `kLayerNormGammaBeta` and end up
    // in the core's own data memory -- so the op moves exactly two tensors.
    AffineExpr s0 = getAffineSymbolExpr(0, ctx);
    AffineMap rowAndFeatures = AffineMap::get(1, 1, {d0, s0}, ctx);
    SmallVector<Attribute> maps = {AffineMapAttr::get(rowAndFeatures),
                                   AffineMapAttr::get(rowAndFeatures)};

    Value init = rewriter.create<tensor::EmptyOp>(loc, flatOutType.getShape(),
                                                  flatOutType.getElementType());
    SmallVector<Value> inputs = {flatInput};
    auto iteratorAttr =
        rewriter.getArrayAttr({IREE::LinalgExt::IteratorTypeAttr::get(
            ctx, utils::IteratorType::parallel)});
    auto customOp = rewriter.create<IREE::LinalgExt::CustomOp>(
        loc, TypeRange{flatOutType}, inputs, ValueRange{init},
        rewriter.getArrayAttr(maps), iteratorAttr);

    // The region sees fully dynamic slices of each operand.
    SmallVector<Type> bbTypes;
    SmallVector<Location> bbLocs;
    for (Value v : {flatInput, init}) {
      auto t = cast<RankedTensorType>(v.getType());
      bbTypes.push_back(RankedTensorType::get(
          SmallVector<int64_t>(t.getRank(), ShapedType::kDynamic),
          t.getElementType()));
      bbLocs.push_back(loc);
    }
    Region &region = customOp.getRegion();
    Block *body = rewriter.createBlock(&region, region.end(), bbTypes, bbLocs);
    rewriter.setInsertionPointToStart(body);
    // The packed constant is materialized inside the region so that the body
    // stays a complete description of the op, but it is not an operand: nothing
    // streams it in.
    Value gammaBeta = rewriter.create<arith::ConstantOp>(
        loc, cast<RankedTensorType>(maybeGammaBeta->getType()),
        *maybeGammaBeta);
    Value result =
        buildLayerNormBody(rewriter, loc, body->getArgument(0), gammaBeta,
                           body->getArgument(1), dag);
    rewriter.create<IREE::LinalgExt::YieldOp>(loc, result);

    // Mark the op, and carry the constants the microkernel lowering needs in
    // the form it needs them: the input scale cancels out of the
    // normalization and survives only inside epsilon, and the output scale
    // divides gamma and beta.
    customOp->setAttr(kLayerNormMarker, rewriter.getUnitAttr());
    customOp->setAttr(
        kLayerNormEpsilonScaled,
        rewriter.getF32FloatAttr(dag.epsilon / (sampleStep * sampleStep)));
    customOp->setAttr(kLayerNormGammaBeta, *maybeGammaBeta);

    rewriter.setInsertionPointAfter(customOp);
    Value expanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, outType, customOp.getResult(0), reassoc);
    rewriter.replaceOp(quantizeOp, expanded);
    return success();
  }
};


class AMDAIERaiseLayerNorm
    : public impl::AMDAIERaiseLayerNormBase<AMDAIERaiseLayerNorm> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect,
                    IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<RaiseLayerNormPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIERaiseLayerNormPass() {
  return std::make_unique<AMDAIERaiseLayerNorm>();
}

}  // namespace mlir::iree_compiler::AMDAIE
