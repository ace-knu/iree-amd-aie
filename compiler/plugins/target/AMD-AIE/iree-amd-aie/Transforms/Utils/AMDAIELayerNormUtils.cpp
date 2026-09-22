// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Utils/AMDAIELayerNormUtils.h"

#include <cmath>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Read `v` as a compile-time float constant, as a double.
bool matchConstFloat(Value v, double &out) {
  APFloat apf(0.0f);
  if (!matchPattern(v, m_ConstantFloat(&apf))) return false;
  bool losesInfo = false;
  apf.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  out = apf.convertToDouble();
  return std::isfinite(out);
}

/// The single-use `linalg.generic` defining `v`, or null.
linalg::GenericOp singleUseProducer(Value v) {
  auto op = v.getDefiningOp<linalg::GenericOp>();
  if (!op || !op->hasOneUse() || op.getNumResults() != 1) return nullptr;
  return op;
}

/// All iterators parallel, and every indexing map is one of `expected`, in
/// order. Keeps the shape checks in the matcher honest without repeating the
/// same three lines at each step.
bool hasMaps(linalg::GenericOp op, ArrayRef<StringRef> iterators,
             ArrayRef<AffineMap> expected) {
  SmallVector<utils::IteratorType> its = op.getIteratorTypesArray();
  if (its.size() != iterators.size()) return false;
  for (auto [it, want] : llvm::zip(its, iterators)) {
    bool isReduction = want == "reduction";
    if ((it == utils::IteratorType::reduction) != isReduction) return false;
  }
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  return maps.size() == expected.size() &&
         llvm::equal(maps, expected);
}

/// The ops of a `linalg.generic` body, from the first block argument to the
/// yielded value, failing if the chain forks.
FailureOr<SmallVector<Operation *>> getLinearBody(linalg::GenericOp op) {
  Region &region = op.getRegion();
  if (!region.hasOneBlock()) return failure();
  Block &block = region.front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1) return failure();

  SmallVector<Operation *> chain;
  Value cur = yieldOp.getValues().front();
  while (Operation *def = cur.getDefiningOp()) {
    if (!cur.hasOneUse()) return failure();
    if (def->getNumOperands() < 1) return failure();
    chain.push_back(def);
    cur = def->getOperand(0);
  }
  if (cur != block.getArgument(0)) return failure();
  std::reverse(chain.begin(), chain.end());
  return chain;
}

/// A one-operand elementwise generic whose body is exactly `OpTy(x, const)`,
/// reporting the constant.
template <typename OpTy>
bool isBinaryWithConstant(linalg::GenericOp op, double &constant) {
  FailureOr<SmallVector<Operation *>> body = getLinearBody(op);
  if (failed(body) || body->size() != 1) return false;
  auto binOp = dyn_cast<OpTy>(body->front());
  return binOp && matchConstFloat(binOp->getOperand(1), constant);
}

/// A one-operand elementwise generic whose body is exactly `OpTy(x)`.
template <typename OpTy>
bool isUnary(linalg::GenericOp op) {
  FailureOr<SmallVector<Operation *>> body = getLinearBody(op);
  return succeeded(body) && body->size() == 1 && isa<OpTy>(body->front());
}

/// A two-operand elementwise generic whose body is exactly `OpTy(a, b)`.
template <typename OpTy>
bool isBinary(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 2) return false;
  Region &region = op.getRegion();
  if (!region.hasOneBlock()) return false;
  Block &block = region.front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1) return false;
  auto binOp = yieldOp.getValues().front().getDefiningOp<OpTy>();
  return binOp && binOp->getOperand(0) == block.getArgument(0) &&
         binOp->getOperand(1) == block.getArgument(1);
}

/// A generic that only copies its input through, i.e. a pure broadcast.
bool isPassThrough(linalg::GenericOp op) {
  Region &region = op.getRegion();
  if (!region.hasOneBlock()) return false;
  Block &block = region.front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
  return yieldOp && yieldOp.getValues().size() == 1 &&
         yieldOp.getValues().front() == block.getArgument(0);
}

/// A reduction generic summing the last dimension into a zero-filled init.
bool isSumReduction(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1) return false;
  FailureOr<SmallVector<Operation *>> body = getLinearBody(op);
  if (failed(body) || body->size() != 1) return false;
  auto addOp = dyn_cast<arith::AddFOp>(body->front());
  if (!addOp) return false;
  // `getLinearBody` walked operand 0 back to the input argument, so the other
  // operand has to be the accumulator.
  if (addOp.getRhs() != op.getRegion().front().getArgument(1)) return false;
  auto fillOp = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  double init;
  return fillOp && matchConstFloat(fillOp.value(), init) && init == 0.0;
}

}  // namespace

bool isQuantizedLayerNormOnly(Operation *op) {
  unsigned numCustom = 0;
  bool marked = false;
  op->walk([&](IREE::LinalgExt::CustomOp customOp) {
    ++numCustom;
    if (customOp->hasAttr(kLayerNormMarker)) marked = true;
  });
  if (numCustom != 1 || !marked) return false;
  // Nothing else with real work: an elementwise producer fused alongside is
  // fine, a second contraction is not.
  bool sawContraction = false;
  op->walk([&](linalg::LinalgOp linalgOp) {
    if (linalg::isaContractionOpInterface(linalgOp)) sawContraction = true;
  });
  return !sawContraction;
}

FailureOr<QuantizedLayerNormDAG> matchQuantizedLayerNorm(
    linalg::GenericOp quantizeOp) {
  QuantizedLayerNormDAG dag;
  dag.quantizeOp = quantizeOp;

  // The shape torch-to-linalg emits, before global optimization folds the unit
  // batch dimension away: `1 x rows x features`, with the reductions keeping
  // the reduced dimension as a 1 and a `tensor.collapse_shape` in front of each
  // broadcast.
  auto outType = dyn_cast<RankedTensorType>(quantizeOp.getResultTypes()[0]);
  if (!outType || !outType.hasStaticShape() || outType.getRank() != 3)
    return failure();
  if (outType.getDimSize(0) != 1) return failure();
  auto outElemTy = dyn_cast<IntegerType>(outType.getElementType());
  if (!outElemTy || outElemTy.getWidth() != 8) return failure();
  dag.numRows = outType.getDimSize(1);
  dag.featureSize = outType.getDimSize(2);
  if (dag.featureSize % 32 != 0) return failure();

  MLIRContext *ctx = quantizeOp.getContext();
  AffineExpr d0, d1, d2;
  bindDims(ctx, d0, d1, d2);
  AffineExpr zeroExpr = getAffineConstantExpr(0, ctx);
  AffineMap mapAll = AffineMap::get(3, 0, {d0, d1, d2}, ctx);
  AffineMap mapRed = AffineMap::get(3, 0, {d0, d1, zeroExpr}, ctx);
  AffineMap mapRow = AffineMap::get(3, 0, {d0, d1}, ctx);
  AffineMap mapCol = AffineMap::get(3, 0, {d2}, ctx);
  const SmallVector<StringRef> ppp = {"parallel", "parallel", "parallel"};
  const SmallVector<StringRef> ppr = {"parallel", "parallel", "reduction"};

  /// The `tensor.collapse_shape` that turns a `1 x rows x 1` reduction result
  /// into the `1 x rows` a broadcast reads.
  auto throughCollapse = [](Value v) -> Value {
    auto collapse = v.getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse || !collapse->hasOneUse()) return nullptr;
    return collapse.getSrc();
  };

  // ---- the quantize tail: f32 -> i8 -------------------------------------
  if (!hasMaps(quantizeOp, ppp, {mapAll, mapAll})) return failure();
  FailureOr<SmallVector<Operation *>> quantBody = getLinearBody(quantizeOp);
  if (failed(quantBody)) return failure();
  // divf(s) -> roundeven -> [addf 0] -> maximumf(lo) -> minimumf(hi) -> fptosi
  auto divOp = dyn_cast<arith::DivFOp>(quantBody->front());
  double outputScale;
  if (!divOp || !matchConstFloat(divOp.getRhs(), outputScale)) return failure();
  if (!(outputScale > 0.0)) return failure();
  unsigned i = 1;
  if (i >= quantBody->size() || !isa<math::RoundEvenOp>((*quantBody)[i]))
    return failure();
  ++i;
  if (i < quantBody->size()) {
    if (auto addOp = dyn_cast<arith::AddFOp>((*quantBody)[i])) {
      double zeroPoint;
      if (!matchConstFloat(addOp.getRhs(), zeroPoint) || zeroPoint != 0.0)
        return failure();
      ++i;
    }
  }
  double lo, hi;
  auto maxOp = i < quantBody->size()
                   ? dyn_cast<arith::MaximumFOp>((*quantBody)[i])
                   : nullptr;
  if (!maxOp || !matchConstFloat(maxOp.getRhs(), lo)) return failure();
  ++i;
  auto minOp = i < quantBody->size()
                   ? dyn_cast<arith::MinimumFOp>((*quantBody)[i])
                   : nullptr;
  if (!minOp || !matchConstFloat(minOp.getRhs(), hi)) return failure();
  ++i;
  if (i != quantBody->size() - 1 || !isa<arith::FPToSIOp>((*quantBody)[i]))
    return failure();
  if (lo != -128.0 || hi != 127.0) return failure();
  dag.outputScale = outputScale;

  // ---- + beta, * gamma --------------------------------------------------
  linalg::GenericOp betaOp = singleUseProducer(quantizeOp.getDpsInputs()[0]);
  if (!betaOp || !isBinary<arith::AddFOp>(betaOp) ||
      !hasMaps(betaOp, ppp, {mapAll, mapCol, mapAll}))
    return failure();
  dag.beta = betaOp.getDpsInputs()[1];

  linalg::GenericOp gammaOp = singleUseProducer(betaOp.getDpsInputs()[0]);
  if (!gammaOp || !isBinary<arith::MulFOp>(gammaOp) ||
      !hasMaps(gammaOp, ppp, {mapAll, mapCol, mapAll}))
    return failure();
  dag.gamma = gammaOp.getDpsInputs()[1];

  // ---- centred * rsqrt --------------------------------------------------
  linalg::GenericOp normOp = singleUseProducer(gammaOp.getDpsInputs()[0]);
  if (!normOp || !isBinary<arith::MulFOp>(normOp) ||
      !hasMaps(normOp, ppp, {mapAll, mapAll, mapAll}))
    return failure();

  linalg::GenericOp rstdBcast = singleUseProducer(normOp.getDpsInputs()[1]);
  if (!rstdBcast || !hasMaps(rstdBcast, ppp, {mapRow, mapAll}) ||
      !isPassThrough(rstdBcast))
    return failure();
  Value rstdSrc = throughCollapse(rstdBcast.getDpsInputs()[0]);
  if (!rstdSrc) return failure();

  linalg::GenericOp rsqrtOp = singleUseProducer(rstdSrc);
  if (!rsqrtOp || !isUnary<math::RsqrtOp>(rsqrtOp) ||
      !hasMaps(rsqrtOp, ppp, {mapAll, mapAll}))
    return failure();

  linalg::GenericOp epsOp = singleUseProducer(rsqrtOp.getDpsInputs()[0]);
  double epsilon;
  if (!epsOp || !isBinaryWithConstant<arith::AddFOp>(epsOp, epsilon) ||
      !hasMaps(epsOp, ppp, {mapAll, mapAll}))
    return failure();

  linalg::GenericOp varMeanOp = singleUseProducer(epsOp.getDpsInputs()[0]);
  double divisor;
  if (!varMeanOp || !isBinaryWithConstant<arith::DivFOp>(varMeanOp, divisor) ||
      !hasMaps(varMeanOp, ppp, {mapAll, mapAll}) ||
      divisor != static_cast<double>(dag.featureSize))
    return failure();

  linalg::GenericOp varSumOp = singleUseProducer(varMeanOp.getDpsInputs()[0]);
  if (!varSumOp || !isSumReduction(varSumOp) ||
      !hasMaps(varSumOp, ppr, {mapAll, mapRed}))
    return failure();

  linalg::GenericOp squareOp = singleUseProducer(varSumOp.getDpsInputs()[0]);
  if (!squareOp || !isBinary<arith::MulFOp>(squareOp) ||
      !hasMaps(squareOp, ppp, {mapAll, mapAll, mapAll}) ||
      squareOp.getDpsInputs()[0] != squareOp.getDpsInputs()[1])
    return failure();

  // ---- x - mean ---------------------------------------------------------
  // The centred value feeds both the square above and the normalize below, so
  // it is the one op in the chain with two uses.
  auto centredOp = squareOp.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!centredOp || centredOp != normOp.getDpsInputs()[0].getDefiningOp())
    return failure();
  if (!isBinary<arith::SubFOp>(centredOp) ||
      !hasMaps(centredOp, ppp, {mapAll, mapAll, mapAll}))
    return failure();

  linalg::GenericOp meanBcast = singleUseProducer(centredOp.getDpsInputs()[1]);
  if (!meanBcast || !hasMaps(meanBcast, ppp, {mapRow, mapAll}) ||
      !isPassThrough(meanBcast))
    return failure();
  Value meanSrc = throughCollapse(meanBcast.getDpsInputs()[0]);
  if (!meanSrc) return failure();

  linalg::GenericOp meanOp = singleUseProducer(meanSrc);
  if (!meanOp || !isBinaryWithConstant<arith::DivFOp>(meanOp, divisor) ||
      !hasMaps(meanOp, ppp, {mapAll, mapAll}) ||
      divisor != static_cast<double>(dag.featureSize))
    return failure();

  linalg::GenericOp sumOp = singleUseProducer(meanOp.getDpsInputs()[0]);
  if (!sumOp || !isSumReduction(sumOp) || !hasMaps(sumOp, ppr, {mapAll, mapRed}))
    return failure();

  // ---- the head ---------------------------------------------------------
  // Whatever feeds the statistics also feeds the subtraction, so it has two
  // uses. It is either a dequantize on its own, or a residual add of two.
  auto headOp = sumOp.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
  if (!headOp || headOp != centredOp.getDpsInputs()[0].getDefiningOp())
    return failure();

  /// A `dequantize(i8 -> f32)` by a constant scale, with a zero zero point.
  auto matchDequantize = [&](linalg::GenericOp op, Value &tensor,
                             double &scale) -> bool {
    if (!hasMaps(op, ppp, {mapAll, mapAll})) return false;
    auto elemTy =
        dyn_cast<IntegerType>(getElementTypeOrSelf(op.getDpsInputs()[0]));
    if (!elemTy || elemTy.getWidth() != 8) return false;
    FailureOr<SmallVector<Operation *>> body = getLinearBody(op);
    if (failed(body) || body->size() != 2) return false;
    if (!isa<arith::SIToFPOp>(body->front())) return false;
    auto mulOp = dyn_cast<arith::MulFOp>((*body)[1]);
    if (!mulOp || !matchConstFloat(mulOp.getRhs(), scale)) return false;
    if (!(scale > 0.0)) return false;
    tensor = op.getDpsInputs()[0];
    return true;
  };

  if (isBinary<arith::AddFOp>(headOp)) {
    // A residual add. Both sides have to be plain int8 dequantizes, and each
    // must feed nothing but this add -- the add itself is what has two uses.
    if (!hasMaps(headOp, ppp, {mapAll, mapAll, mapAll})) return failure();
    linalg::GenericOp lhs = singleUseProducer(headOp.getDpsInputs()[0]);
    linalg::GenericOp rhs = singleUseProducer(headOp.getDpsInputs()[1]);
    if (!lhs || !rhs) return failure();
    if (!matchDequantize(lhs, dag.input, dag.inputScale)) return failure();
    if (!matchDequantize(rhs, dag.residual, dag.residualScale)) return failure();
    dag.chain.insert(lhs.getOperation());
    dag.chain.insert(rhs.getOperation());
  } else if (!matchDequantize(headOp, dag.input, dag.inputScale)) {
    return failure();
  }
  dag.epsilon = epsilon;

  for (Operation *op : {headOp.getOperation(), sumOp.getOperation(),
                        meanOp.getOperation(), meanBcast.getOperation(),
                        centredOp.getOperation(), squareOp.getOperation(),
                        varSumOp.getOperation(), varMeanOp.getOperation(),
                        epsOp.getOperation(), rsqrtOp.getOperation(),
                        rstdBcast.getOperation(), normOp.getOperation(),
                        gammaOp.getOperation(), betaOp.getOperation(),
                        quantizeOp.getOperation()})
    dag.chain.insert(op);
  return dag;
}

}  // namespace mlir::iree_compiler::AMDAIE
