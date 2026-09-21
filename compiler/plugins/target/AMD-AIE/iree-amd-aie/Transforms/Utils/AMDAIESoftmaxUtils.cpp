// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Utils/AMDAIESoftmaxUtils.h"

#include <cmath>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

/// All of the op's indexing maps are the identity over the same iteration
/// space, and every iterator is parallel: a plain elementwise op.
bool isPlainElementwise(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) return false;
  if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;
  return llvm::all_of(op.getIndexingMapsArray(),
                      [](AffineMap m) { return m.isIdentity(); });
}

/// The single block argument-rooted value chain of a single-block region,
/// returned as the ops in order from the block argument to the yielded value.
/// Fails if any value in the chain has more than one use inside the region, so
/// that reading the chain linearly is faithful.
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
  // The chain has to bottom out at the op's input block argument.
  if (cur != block.getArgument(0)) return failure();
  std::reverse(chain.begin(), chain.end());
  return chain;
}

}  // namespace

FailureOr<QuantizedSoftmaxDAG> matchQuantizedSoftmax(
    linalg::GenericOp quantizeOp) {
  QuantizedSoftmaxDAG dag;
  dag.quantizeOp = quantizeOp;

  // ---- the quantize tail: f32 -> i8 -------------------------------------
  if (!isPlainElementwise(quantizeOp)) return failure();
  auto outElemTy = dyn_cast<IntegerType>(
      getElementTypeOrSelf(quantizeOp.getDpsInits()[0].getType()));
  if (!outElemTy || outElemTy.getWidth() != 8) return failure();

  FailureOr<SmallVector<Operation *>> quantBody = getLinearBody(quantizeOp);
  if (failed(quantBody)) return failure();
  // divf(c) -> roundeven -> [addf 0] -> maximumf(lo) -> minimumf(hi) -> fptosi
  auto divOp = dyn_cast<arith::DivFOp>(quantBody->front());
  if (!divOp || !matchConstFloat(divOp.getRhs(), dag.outputScale)) return failure();
  if (!(dag.outputScale > 0.0)) return failure();
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
  if (lo != std::round(lo) || hi != std::round(hi)) return failure();
  dag.clampLo = static_cast<int64_t>(lo);
  dag.clampHi = static_cast<int64_t>(hi);
  // The kernel saturates to int8, so it can only stand in for that clamp.
  if (dag.clampLo != -128 || dag.clampHi != 127) return failure();

  // ---- the softmax ------------------------------------------------------
  auto softmaxOp =
      quantizeOp.getDpsInputs()[0].getDefiningOp<linalg::SoftmaxOp>();
  if (!softmaxOp || !softmaxOp->hasOneUse()) return failure();
  auto softmaxTy = dyn_cast<ShapedType>(softmaxOp.getResult()[0].getType());
  if (!softmaxTy || !softmaxTy.hasStaticShape()) return failure();
  if (softmaxOp.getDimension() != softmaxTy.getRank() - 1) return failure();
  dag.softmaxOp = softmaxOp;

  // ---- the dequantize head: i8 -> f32 -----------------------------------
  auto dequantizeOp =
      softmaxOp.getDpsInputOperand(0)->get().getDefiningOp<linalg::GenericOp>();
  if (!dequantizeOp || !dequantizeOp->hasOneUse()) return failure();
  if (!isPlainElementwise(dequantizeOp)) return failure();
  auto inElemTy = dyn_cast<IntegerType>(
      getElementTypeOrSelf(dequantizeOp.getDpsInputs()[0].getType()));
  if (!inElemTy || inElemTy.getWidth() != 8) return failure();

  FailureOr<SmallVector<Operation *>> dequantBody = getLinearBody(dequantizeOp);
  if (failed(dequantBody)) return failure();
  // sitofp, then any run of constant scales and constant offsets.
  //
  // Offsets need no handling at all: softmax is invariant to adding the same
  // value to every element of a row, and the kernel subtracts the row maximum
  // anyway. A folded-away attention mask (all-zero, so `addf 0.0`) shows up
  // here, and so would any uniform bias. Only the scales, whose product is the
  // coefficient of the int8 input, have to be carried into the kernel.
  if (dequantBody->empty() || !isa<arith::SIToFPOp>(dequantBody->front()))
    return failure();
  dag.inputScale = 1.0;
  for (unsigned j = 1; j < dequantBody->size(); ++j) {
    Operation *step = (*dequantBody)[j];
    double constant;
    if (auto mulOp = dyn_cast<arith::MulFOp>(step)) {
      if (!matchConstFloat(mulOp.getRhs(), constant)) return failure();
      dag.inputScale *= constant;
      continue;
    }
    if (auto addOp = dyn_cast<arith::AddFOp>(step)) {
      if (!matchConstFloat(addOp.getRhs(), constant)) return failure();
      continue;
    }
    return failure();
  }
  if (!(dag.inputScale > 0.0)) return failure();

  dag.dequantizeOp = dequantizeOp;
  dag.input = dequantizeOp.getDpsInputs()[0];
  return dag;
}

bool isQuantizedSoftmaxOnly(Operation *op) {
  // `linalg.softmax` is not a `LinalgOp`, so the two are counted separately.
  unsigned numSoftmax = 0;
  op->walk([&](linalg::SoftmaxOp) { ++numSoftmax; });
  if (numSoftmax != 1) return false;

  unsigned numGeneric = 0;
  bool sawOtherLinalgOp = false;
  op->walk([&](linalg::LinalgOp linalgOp) {
    if (isa<linalg::GenericOp>(linalgOp.getOperation()))
      ++numGeneric;
    else
      sawOtherLinalgOp = true;
  });
  if (sawOtherLinalgOp || numGeneric != 2) return false;

  bool matched = false;
  op->walk([&](linalg::GenericOp genericOp) {
    if (succeeded(matchQuantizedSoftmax(genericOp))) matched = true;
  });
  return matched;
}

}  // namespace mlir::iree_compiler::AMDAIE
