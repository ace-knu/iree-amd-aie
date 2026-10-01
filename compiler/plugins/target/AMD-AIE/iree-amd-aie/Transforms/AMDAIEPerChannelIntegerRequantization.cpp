// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-per-channel-integer-requantization"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// The float requantization of a matmul with a per-channel weight scale, as
/// `AMDAIEFoldPerChannelDequantIntoMatmul` leaves it and the requantization
/// after it writes it:
///
///   %f = generic ins(%acc : iK, %s : [N] f32) { sitofp(acc) * s[n] }
///   %q = generic ins(%f) { roundeven(f / S2) [+ zp] -> clamp [lo, hi] -> iM }
///
/// The scale is a vector, so `AMDAIEIntegerRequantization` -- which works on
/// the core's scalar code and needs a compile-time constant -- cannot take it,
/// and on aie2p the float tail pulls the soft-float library into the core: on
/// BERT-base's FC2 that overflows the 16 KB program memory. This rewrites the
/// pair, while it is still a tensor op whose scales are visible, into
///
///   %q = generic ins(%acc, %m : [N] i32) {
///          round_half_even(acc * m[n] >> shift) [+ zp] -> clamp -> iM }
///
/// with `m[n] / 2^shift ~= s[n] / S2` and one shift for every channel. The
/// table the core reads becomes integer, and so does all of its arithmetic.
///
/// As with the scalar pass's default policy, the integer form tracks the
/// combined scale as closely as an i32 multiplier allows rather than being
/// proven bit-identical to the float one; the two differ only where the float
/// tail's own f32 roundings land on the other side of a tie.
struct Tail {
  double divisor = 1.0;
  int64_t zeroPoint = 0;
  int64_t lo = 0, hi = 0;
  IntegerType outType;
};

bool matchWholeF32(Value v, int64_t &out) {
  APFloat f(0.0f);
  if (!matchPattern(v, m_ConstantFloat(&f))) return false;
  double d = f.convertToDouble();
  if (d != std::floor(d)) return false;
  out = static_cast<int64_t>(d);
  return true;
}

/// The requantization generic `q`, if it is one.
FailureOr<Tail> matchRequantization(linalg::GenericOp q) {
  if (q.getNumDpsInputs() != 1 || q.getNumDpsInits() != 1 ||
      q.getNumReductionLoops() != 0 ||
      !llvm::all_of(q.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); }))
    return failure();
  Block &body = q.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  auto toInt = yield.getValues()[0].getDefiningOp<arith::FPToSIOp>();
  if (!toInt) return failure();
  Tail tail;
  tail.outType = dyn_cast<IntegerType>(toInt.getType());
  if (!tail.outType || tail.outType.getWidth() > 32) return failure();
  auto upper = toInt.getIn().getDefiningOp<arith::MinimumFOp>();
  if (!upper || !matchWholeF32(upper.getRhs(), tail.hi)) return failure();
  auto lower = upper.getLhs().getDefiningOp<arith::MaximumFOp>();
  if (!lower || !matchWholeF32(lower.getRhs(), tail.lo)) return failure();
  Value cur = lower.getLhs();
  if (auto add = cur.getDefiningOp<arith::AddFOp>()) {
    if (!matchWholeF32(add.getRhs(), tail.zeroPoint)) return failure();
    cur = add.getLhs();
  }
  auto round = cur.getDefiningOp<math::RoundEvenOp>();
  if (!round) return failure();
  auto div = round.getOperand().getDefiningOp<arith::DivFOp>();
  if (!div || div.getLhs() != body.getArgument(0)) return failure();
  APFloat d(0.0f);
  if (!matchPattern(div.getRhs(), m_ConstantFloat(&d))) return failure();
  tail.divisor = d.convertToDouble();
  if (!(tail.divisor > 0.0) || !std::isfinite(tail.divisor)) return failure();
  return tail;
}

/// The scaled accumulator generic `f`: its accumulator and per-channel scales.
LogicalResult matchScaledAccumulator(linalg::GenericOp f, Value &acc,
                                     SmallVector<double> &scales) {
  if (f.getNumDpsInputs() != 2 || f.getNumDpsInits() != 1 ||
      f.getNumReductionLoops() != 0)
    return failure();
  unsigned rank = f.getNumLoops();
  SmallVector<AffineMap> maps = f.getIndexingMapsArray();
  AffineMap column = AffineMap::get(
      rank, 0, {getAffineDimExpr(rank - 1, f.getContext())}, f.getContext());
  if (!maps[0].isIdentity() || maps[1] != column || !maps[2].isIdentity())
    return failure();
  Block &body = f.getRegion().front();
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  auto mul = yield.getValues()[0].getDefiningOp<arith::MulFOp>();
  if (!mul || mul.getRhs() != body.getArgument(1)) return failure();
  auto toFloat = mul.getLhs().getDefiningOp<arith::SIToFPOp>();
  if (!toFloat || toFloat.getIn() != body.getArgument(0) ||
      !body.getArgument(0).getType().isInteger(32))
    return failure();
  DenseElementsAttr attr;
  if (!matchPattern(f.getDpsInputs()[1], m_Constant(&attr))) return failure();
  for (APFloat v : attr.getValues<APFloat>()) scales.push_back(v.convertToDouble());
  acc = f.getDpsInputs()[0];
  return success();
}

struct PerChannelIntegerRequantization
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp q,
                                PatternRewriter &rewriter) const override {
    FailureOr<Tail> tail = matchRequantization(q);
    if (failed(tail)) return failure();
    auto f = q.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();
    if (!f || !f->hasOneUse()) return failure();
    Value acc;
    SmallVector<double> scales;
    if (failed(matchScaledAccumulator(f, acc, scales))) return failure();

    // One shift for every channel: as large as the widest channel's multiplier
    // allows inside i32, so the narrowest keeps the most bits it can.
    double widest = 0.0;
    for (double &s : scales) {
      s /= tail->divisor;
      if (!(s > 0.0) || !std::isfinite(s)) return failure();
      widest = std::max(widest, s);
    }
    int shift = 0;
    while (shift < 62 && widest * std::ldexp(1.0, shift + 1) < 2147483647.0)
      ++shift;
    if (shift < 1) return failure();
    SmallVector<int32_t> multipliers;
    multipliers.reserve(scales.size());
    for (double s : scales) {
      int64_t m = std::llround(s * std::ldexp(1.0, shift));
      if (m <= 0 || m > 2147483647) return failure();
      multipliers.push_back(static_cast<int32_t>(m));
    }

    Location loc = q.getLoc();
    Type i32 = rewriter.getI32Type();
    Type i64 = rewriter.getI64Type();
    auto tableType =
        RankedTensorType::get({static_cast<int64_t>(multipliers.size())}, i32);
    Value table = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(tableType, ArrayRef<int32_t>(multipliers)));
    SmallVector<AffineMap> maps = f.getIndexingMapsArray();
    maps[2] = q.getIndexingMapsArray()[1];
    auto replacement = rewriter.create<linalg::GenericOp>(
        loc, q.getResultTypes(), ValueRange{acc, table}, q.getDpsInits(), maps,
        q.getIteratorTypesArray(),
        [&](OpBuilder &b, Location l, ValueRange args) {
          auto c = [&](int64_t v) -> Value {
            return b.create<arith::ConstantOp>(l, b.getI64IntegerAttr(v));
          };
          Value a = b.create<arith::ExtSIOp>(l, i64, args[0]);
          Value m = b.create<arith::ExtSIOp>(l, i64, args[1]);
          Value product = b.create<arith::MulIOp>(l, a, m);
          // An arithmetic shift floors, so the remainder is in [0, 2^shift)
          // for either sign and round-half-to-even reads the same way.
          Value quotient = b.create<arith::ShRSIOp>(l, product, c(shift));
          Value remainder = b.create<arith::AndIOp>(
              l, product, c((int64_t{1} << shift) - 1));
          Value half = c(int64_t{1} << (shift - 1));
          Value past = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sgt,
                                               remainder, half);
          Value atHalf = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq,
                                                 remainder, half);
          Value odd = b.create<arith::CmpIOp>(
              l, arith::CmpIPredicate::ne,
              b.create<arith::AndIOp>(l, quotient, c(1)), c(0));
          Value up = b.create<arith::OrIOp>(
              l, past, b.create<arith::AndIOp>(l, atHalf, odd));
          Value rounded = b.create<arith::SelectOp>(
              l, up, b.create<arith::AddIOp>(l, quotient, c(1)), quotient);
          if (tail->zeroPoint != 0)
            rounded = b.create<arith::AddIOp>(l, rounded, c(tail->zeroPoint));
          rounded = b.create<arith::MaxSIOp>(l, rounded, c(tail->lo));
          rounded = b.create<arith::MinSIOp>(l, rounded, c(tail->hi));
          b.create<linalg::YieldOp>(
              l, b.create<arith::TruncIOp>(l, tail->outType, rounded)
                     .getResult());
        });
    rewriter.replaceOp(q, replacement.getResults());
    rewriter.eraseOp(f);
    return success();
  }
};

class AMDAIEPerChannelIntegerRequantizationPass
    : public impl::AMDAIEPerChannelIntegerRequantizationBase<
          AMDAIEPerChannelIntegerRequantizationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<PerChannelIntegerRequantization>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEPerChannelIntegerRequantizationPass() {
  return std::make_unique<AMDAIEPerChannelIntegerRequantizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
