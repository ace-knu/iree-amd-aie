// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIERequantUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-integer-requantization"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Read `v` as a compile-time float constant, in f32 semantics.
static bool matchConstF32(Value v, APFloat &out) {
  APFloat apf(0.0f);
  if (!matchPattern(v, m_ConstantFloat(&apf))) return false;
  bool losesInfo = false;
  apf.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
  if (losesInfo) return false;
  out = apf;
  return true;
}

static double toDouble(const APFloat &apf) {
  APFloat tmp = apf;
  bool losesInfo = false;
  tmp.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  return tmp.convertToDouble();
}

/// Read `v` as a float constant that is a whole number.
static bool matchWholeConstF32(Value v, int64_t &out) {
  APFloat apf(0.0f);
  if (!matchConstF32(v, apf)) return false;
  double d = toDouble(apf);
  if (!std::isfinite(d) || d != std::round(d)) return false;
  if (std::abs(d) > 4.0e18) return false;
  out = static_cast<int64_t>(d);
  return true;
}

/// Replace a constant-scale float requantization tail with integer arithmetic.
///
/// The int8 pipelines emit, per output element,
///
///   sitofp(acc:i32) -> [* S1] -> [/ S2] -> roundeven -> [+ zp] -> clamp -> i8
///
/// On aie2p that is a disaster: the backend has no scalar float arithmetic at
/// all, so each of those ops becomes a soft-float libcall (`__mulsf3`,
/// `__divsf3`, `__addsf3`, the `__cmpsf2` family, ...), which together cost
/// several KB of the core's 16KB program memory.
///
/// Since `S = S1 / S2` is known at compile time it can be written as
/// `M / 2^n`, turning the whole tail into an integer multiply, a shift with
/// explicit round-half-to-even, and an integer clamp. That leaves only integer
/// instructions, which the hardware does have.
///
/// This is *not* an approximation, but it is also not exact by construction:
/// whether the integer form reproduces the float tail depends on the particular
/// scales, the clamp range and the accumulator's width. `integerRequantIsExact`
/// checks it for every accumulator value the type can hold, and the rewrite is
/// declined when the answer is no. So a tail is either left alone or replaced
/// by something verified bit-exact for that tail.
struct RequantToIntegerPattern : public OpRewritePattern<arith::FPToSIOp> {
  using OpRewritePattern<arith::FPToSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::FPToSIOp op,
                                PatternRewriter &rewriter) const override {
    Type outTy = op.getType();
    auto outIntTy = dyn_cast<IntegerType>(getElementTypeOrSelf(outTy));
    if (!outIntTy || outIntTy.getWidth() > 32)
      return rewriter.notifyMatchFailure(op, "result is not a narrow integer");

    RequantTail tail;

    // clamp: minimumf(maximumf(x, lo), hi)
    auto minOp = op.getIn().getDefiningOp<arith::MinimumFOp>();
    if (!minOp) return rewriter.notifyMatchFailure(op, "no upper clamp");
    if (!matchWholeConstF32(minOp.getRhs(), tail.clampHi))
      return rewriter.notifyMatchFailure(op, "upper clamp is not a whole "
                                             "compile-time constant");
    auto maxOp = minOp.getLhs().getDefiningOp<arith::MaximumFOp>();
    if (!maxOp) return rewriter.notifyMatchFailure(op, "no lower clamp");
    if (!matchWholeConstF32(maxOp.getRhs(), tail.clampLo))
      return rewriter.notifyMatchFailure(op, "lower clamp is not a whole "
                                             "compile-time constant");

    // optional zero point, which must be a whole number
    Value cur = maxOp.getLhs();
    if (auto addOp = cur.getDefiningOp<arith::AddFOp>()) {
      if (!matchWholeConstF32(addOp.getRhs(), tail.zeroPoint))
        return rewriter.notifyMatchFailure(op, "zero point is not a whole "
                                               "compile-time constant");
      cur = addOp.getLhs();
    }

    auto roundOp = cur.getDefiningOp<math::RoundEvenOp>();
    if (!roundOp) return rewriter.notifyMatchFailure(op, "no roundeven");

    // scale chain, innermost last: sitofp -> [mulf S1] -> [divf S2]
    cur = roundOp.getOperand();
    if (auto divOp = cur.getDefiningOp<arith::DivFOp>()) {
      APFloat c(0.0f);
      if (!matchConstF32(divOp.getRhs(), c))
        return rewriter.notifyMatchFailure(op, "divisor is not a constant");
      if (!c.isFiniteNonZero() || c.isNegative())
        return rewriter.notifyMatchFailure(op, "divisor is not positive");
      tail.divConst = c;
      cur = divOp.getLhs();
    }
    if (auto mulOp = cur.getDefiningOp<arith::MulFOp>()) {
      APFloat c(0.0f);
      if (!matchConstF32(mulOp.getRhs(), c))
        return rewriter.notifyMatchFailure(op, "multiplier is not a constant");
      if (!c.isFiniteNonZero() || c.isNegative())
        return rewriter.notifyMatchFailure(op, "multiplier is not positive");
      tail.mulConst = c;
      cur = mulOp.getLhs();
    }
    auto siToFp = cur.getDefiningOp<arith::SIToFPOp>();
    if (!siToFp)
      return rewriter.notifyMatchFailure(op, "tail does not start at an "
                                             "integer accumulator");
    Value acc = siToFp.getIn();
    auto accIntTy = dyn_cast<IntegerType>(getElementTypeOrSelf(acc.getType()));
    if (!accIntTy || accIntTy.getWidth() > 32)
      return rewriter.notifyMatchFailure(op, "accumulator is wider than i32");
    tail.accBitWidth = accIntTy.getWidth();

    double scale = 1.0;
    if (tail.mulConst) scale *= toDouble(*tail.mulConst);
    if (tail.divConst) scale /= toDouble(*tail.divConst);

    IntegerRequant req;
    if (!chooseMultiplierAndShift(scale, req))
      return rewriter.notifyMatchFailure(op, "scale is not representable");

    // Only rewrite what can be shown to behave identically. Declining here
    // simply leaves the float tail in place.
    if (!integerRequantIsExact(tail, req))
      return rewriter.notifyMatchFailure(
          op, "integer form is not bit-exact for this tail");

    Location loc = op.getLoc();
    Type wideTy = rewriter.getI64Type();
    if (auto shapedTy = dyn_cast<ShapedType>(acc.getType()))
      wideTy = shapedTy.clone(wideTy);

    auto wideConst = [&](int64_t v) -> Value {
      TypedAttr attr = rewriter.getI64IntegerAttr(v);
      if (auto shapedTy = dyn_cast<ShapedType>(wideTy))
        attr = DenseElementsAttr::get(shapedTy, attr);
      return rewriter.create<arith::ConstantOp>(loc, wideTy, attr);
    };

    Value wideAcc = rewriter.create<arith::ExtSIOp>(loc, wideTy, acc);
    Value product =
        rewriter.create<arith::MulIOp>(loc, wideAcc, wideConst(req.multiplier));

    // An arithmetic shift right floors, so the masked remainder is always in
    // [0, 2^n) and the round-half-to-even decision below reads the same for
    // negative and positive values.
    Value quotient =
        rewriter.create<arith::ShRSIOp>(loc, product, wideConst(req.shift));
    Value remainder = rewriter.create<arith::AndIOp>(
        loc, product, wideConst((int64_t{1} << req.shift) - 1));
    Value half = wideConst(int64_t{1} << (req.shift - 1));

    Value pastHalf = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, remainder, half);
    Value atHalf = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, remainder, half);
    Value lowBit = rewriter.create<arith::AndIOp>(loc, quotient, wideConst(1));
    Value isOdd = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, lowBit, wideConst(0));
    Value tieAway = rewriter.create<arith::AndIOp>(loc, atHalf, isOdd);
    Value roundUp = rewriter.create<arith::OrIOp>(loc, pastHalf, tieAway);
    Value bumped = rewriter.create<arith::AddIOp>(loc, quotient, wideConst(1));
    Value rounded =
        rewriter.create<arith::SelectOp>(loc, roundUp, bumped, quotient);

    if (tail.zeroPoint != 0)
      rounded = rewriter.create<arith::AddIOp>(loc, rounded,
                                               wideConst(tail.zeroPoint));

    rounded = rewriter.create<arith::MaxSIOp>(loc, rounded,
                                              wideConst(tail.clampLo));
    rounded = rewriter.create<arith::MinSIOp>(loc, rounded,
                                              wideConst(tail.clampHi));

    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, outTy, rounded);
    return success();
  }
};

class AMDAIEIntegerRequantization
    : public impl::AMDAIEIntegerRequantizationBase<
          AMDAIEIntegerRequantization> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, arith::ArithDialect, math::MathDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<RequantToIntegerPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEIntegerRequantizationPass() {
  return std::make_unique<AMDAIEIntegerRequantization>();
}

}  // namespace mlir::iree_compiler::AMDAIE
