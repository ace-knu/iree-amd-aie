// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-expand-roundeven"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Does every use of `v` flow into a narrowing conversion to an integer,
/// through nothing but the ops a quantization tail puts there?
static bool feedsOnlyNarrowingIntCast(Value v, int depth = 0) {
  // Quantization tails are short: roundeven -> [addf] -> maximumf -> minimumf
  // -> fptosi. The bound keeps a long unrelated chain from being accepted.
  if (depth > 4) return false;
  if (v.use_empty()) return false;
  for (Operation *user : v.getUsers()) {
    if (auto cast = dyn_cast<arith::FPToSIOp>(user)) {
      auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(cast.getType()));
      if (!intTy || intTy.getWidth() > 32) return false;
      continue;
    }
    if (!isa<arith::AddFOp, arith::MaximumFOp, arith::MinimumFOp>(user))
      return false;
    if (!feedsOnlyNarrowingIntCast(user->getResult(0), depth + 1)) return false;
  }
  return true;
}

/// Rewrite `math.roundeven` on f32 into ops that peano's aie2p backend can
/// actually select.
///
/// Every float rounding intrinsic (roundeven, rint, nearbyint, round, floor,
/// ceil, trunc) fails to legalize on aie2p, and so does `copysign`, which is
/// what upstream's `math-expand-ops` lowers roundeven through. What does
/// legalize is fptosi/sitofp, fabs, fcmp and select, so this builds
/// round-half-to-even out of only those.
///
/// For `|x| >= 2^23` an f32 is already integral, so the result is `x` itself.
/// That case also keeps the fptosi below away from values it cannot represent.
/// Otherwise, with `t` the truncation of `x` toward zero and `d = x - t`:
///
///   |d| > 0.5                       -> step one away from zero
///   |d| == 0.5 and trunc(x) is odd  -> step one away from zero (ties to even)
///   otherwise                       -> t
///
/// **This expansion does not handle NaN**, so it is deliberately restricted to
/// roundevens whose result is on its way to `arith.fptosi` on a narrow integer.
/// There a NaN is already undefined in the float form, so nothing is lost. Any
/// other `math.roundeven` is left alone rather than silently given different
/// NaN behaviour.
///
/// In practice most quantization tails never reach here at all:
/// `iree-amdaie-integer-requantization` replaces the ones whose scales are
/// compile-time constants outright. This covers what it declines.
struct ExpandRoundEvenPattern : public OpRewritePattern<math::RoundEvenOp> {
  using OpRewritePattern<math::RoundEvenOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(math::RoundEvenOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x = op.getOperand();
    Type fTy = x.getType();
    Type fETy = getElementTypeOrSelf(fTy);
    if (!fETy.isF32())
      return rewriter.notifyMatchFailure(op, "only f32 is expanded");
    if (!feedsOnlyNarrowingIntCast(op.getResult()))
      return rewriter.notifyMatchFailure(
          op, "result is not on its way to a narrowing integer cast");

    Type iTy = rewriter.getI32Type();
    if (auto shapedTy = dyn_cast<ShapedType>(fTy))
      iTy = shapedTy.clone(iTy);

    auto fConst = [&](double v) -> Value {
      TypedAttr attr = rewriter.getF32FloatAttr(v);
      if (auto shapedTy = dyn_cast<ShapedType>(fTy))
        attr = DenseElementsAttr::get(shapedTy, attr);
      return rewriter.create<arith::ConstantOp>(loc, fTy, attr);
    };
    auto iConst = [&](int32_t v) -> Value {
      TypedAttr attr = rewriter.getI32IntegerAttr(v);
      if (auto shapedTy = dyn_cast<ShapedType>(iTy))
        attr = DenseElementsAttr::get(shapedTy, attr);
      return rewriter.create<arith::ConstantOp>(loc, iTy, attr);
    };

    Value zeroF = fConst(0.0);
    Value halfF = fConst(0.5);
    Value oneF = fConst(1.0);
    Value minusOneF = fConst(-1.0);
    // 2^23: the smallest f32 magnitude at which every value is an integer.
    Value bigF = fConst(8388608.0);

    // t = trunc(x) toward zero, and the same value as an integer.
    Value ti = rewriter.create<arith::FPToSIOp>(loc, iTy, x);
    Value t = rewriter.create<arith::SIToFPOp>(loc, fTy, ti);

    Value d = rewriter.create<arith::SubFOp>(loc, x, t);
    Value ad = rewriter.create<math::AbsFOp>(loc, d);

    // Stepping away from zero means +1 for a positive x and -1 for a negative
    // one. `d` carries x's sign, and is zero exactly when x is already
    // integral (where the step is never selected anyway).
    Value isNeg =
        rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, d, zeroF);
    Value step = rewriter.create<arith::SelectOp>(loc, isNeg, minusOneF, oneF);
    Value away = rewriter.create<arith::AddFOp>(loc, t, step);

    Value pastHalf =
        rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, ad, halfF);
    Value atHalf =
        rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, ad, halfF);
    Value lowBit = rewriter.create<arith::AndIOp>(loc, ti, iConst(1));
    Value isOdd = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                 lowBit, iConst(0));
    Value tieAway = rewriter.create<arith::AndIOp>(loc, atHalf, isOdd);
    Value useAway = rewriter.create<arith::OrIOp>(loc, pastHalf, tieAway);
    Value rounded = rewriter.create<arith::SelectOp>(loc, useAway, away, t);

    Value ax = rewriter.create<math::AbsFOp>(loc, x);
    Value isBig = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGE, ax, bigF);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isBig, x, rounded);
    return success();
  }
};

class AMDAIEExpandRoundEven
    : public impl::AMDAIEExpandRoundEvenBase<AMDAIEExpandRoundEven> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<ExpandRoundEvenPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEExpandRoundEvenPass() {
  return std::make_unique<AMDAIEExpandRoundEven>();
}

}  // namespace mlir::iree_compiler::AMDAIE
