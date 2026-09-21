// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Utils/AMDAIERequantUtils.h"

#include <cmath>

#include "gtest/gtest.h"

namespace mlir::iree_compiler::AMDAIE {
namespace {

static APFloat f32(double v) {
  APFloat apf(static_cast<float>(v));
  return apf;
}

/// The float tail, evaluated the way the device does: f32 throughout, one
/// rounding per operation, round-half-to-even at the end.
static int64_t referenceFloatTail(const RequantTail &tail, int64_t acc) {
  float y = static_cast<float>(acc);
  if (tail.mulConst) y = y * tail.mulConst->convertToFloat();
  if (tail.divConst) y = y / tail.divConst->convertToFloat();
  // round half to even
  float r = std::nearbyint(y);  // the default mode is round-to-nearest-even
  double v = static_cast<double>(r) + static_cast<double>(tail.zeroPoint);
  if (v < static_cast<double>(tail.clampLo)) return tail.clampLo;
  if (v > static_cast<double>(tail.clampHi)) return tail.clampHi;
  return static_cast<int64_t>(v);
}

static int64_t referenceIntegerTail(const RequantTail &tail,
                                    const IntegerRequant &req, int64_t acc) {
  int64_t p = acc * req.multiplier;
  int64_t q = p >> req.shift;
  int64_t rem = p & ((int64_t{1} << req.shift) - 1);
  int64_t half = int64_t{1} << (req.shift - 1);
  if (rem > half || (rem == half && (q & 1) != 0)) ++q;
  q += tail.zeroPoint;
  if (q < tail.clampLo) return tail.clampLo;
  if (q > tail.clampHi) return tail.clampHi;
  return q;
}

static RequantTail makeTail(double mul, double div, unsigned accBits = 32) {
  RequantTail t;
  if (mul != 1.0) t.mulConst = f32(mul);
  t.divConst = f32(div);
  t.clampLo = -128;
  t.clampHi = 127;
  t.accBitWidth = accBits;
  return t;
}

TEST(RequantUtils, ChoosesAMultiplierNearTheScale) {
  IntegerRequant req;
  ASSERT_TRUE(chooseMultiplierAndShift(0.00384738599, req));
  EXPECT_GT(req.shift, 0);
  EXPECT_GT(req.multiplier, 0);
  double reconstructed = std::ldexp(static_cast<double>(req.multiplier),
                                    -static_cast<int>(req.shift));
  EXPECT_NEAR(reconstructed, 0.00384738599, 1e-12);
}

TEST(RequantUtils, RejectsNonPositiveOrNonFiniteScales) {
  IntegerRequant req;
  EXPECT_FALSE(chooseMultiplierAndShift(0.0, req));
  EXPECT_FALSE(chooseMultiplierAndShift(-0.5, req));
  EXPECT_FALSE(chooseMultiplierAndShift(
      std::numeric_limits<double>::infinity(), req));
  EXPECT_FALSE(
      chooseMultiplierAndShift(std::numeric_limits<double>::quiet_NaN(), req));
}

/// The three tails that layer 0 of BERT-base's attention actually produces.
/// Sweeping the accumulator range each one can reach is what justifies the
/// rewrite for this model; the exactness check has to agree with that sweep.
TEST(RequantUtils, MatchesTheFloatTailOverTheRealAccumulatorRanges) {
  struct Case {
    const char *name;
    double mul, div;
    int64_t accLimit;  // K * 127 * 127
  };
  const Case cases[] = {
      {"QK^T", 2.280560e-03, 0.592755735, 64LL * 127 * 127},
      {"PV", 1.73504828e-4, 0.01683544, 32LL * 127 * 127},
      {"proj", 7.78110261e-5, 0.0237346236, 768LL * 127 * 127},
  };
  for (const Case &c : cases) {
    SCOPED_TRACE(c.name);
    RequantTail tail = makeTail(c.mul, c.div);
    double scale = static_cast<double>(f32(c.mul).convertToFloat()) /
                   static_cast<double>(f32(c.div).convertToFloat());
    IntegerRequant req;
    ASSERT_TRUE(chooseMultiplierAndShift(scale, req));
    EXPECT_TRUE(integerRequantIsExact(tail, req));

    // Step over the range rather than every value: the interesting part is
    // where the two could disagree, which is near a rounding boundary, and
    // the stride below still lands on plenty of those.
    int64_t mismatches = 0;
    for (int64_t acc = -c.accLimit; acc <= c.accLimit; acc += 7) {
      if (referenceFloatTail(tail, acc) != referenceIntegerTail(tail, req, acc))
        ++mismatches;
    }
    EXPECT_EQ(mismatches, 0);
  }
}

/// Every accumulator value, for a narrow accumulator, so nothing is skipped.
TEST(RequantUtils, MatchesTheFloatTailExhaustivelyForANarrowAccumulator) {
  RequantTail tail = makeTail(1.0, 0.0237346236, /*accBits=*/16);
  double scale = 1.0 / static_cast<double>(f32(0.0237346236).convertToFloat());
  IntegerRequant req;
  ASSERT_TRUE(chooseMultiplierAndShift(scale, req));
  ASSERT_TRUE(integerRequantIsExact(tail, req));
  for (int64_t acc = -32768; acc <= 32767; ++acc) {
    ASSERT_EQ(referenceFloatTail(tail, acc),
              referenceIntegerTail(tail, req, acc))
        << "acc = " << acc;
  }
}

/// Ties must go to even, including on the negative side, which is where a
/// "+0.5 then truncate" implementation would be wrong.
TEST(RequantUtils, RoundsHalvesToEvenOnBothSides) {
  // scale = 1/2, so an odd accumulator lands exactly on a half.
  RequantTail tail;
  tail.divConst = f32(2.0);
  tail.clampLo = -128;
  tail.clampHi = 127;
  tail.accBitWidth = 32;
  IntegerRequant req;
  ASSERT_TRUE(chooseMultiplierAndShift(0.5, req));
  ASSERT_TRUE(integerRequantIsExact(tail, req));

  // 2.5 -> 2, 3.5 -> 4, -2.5 -> -2, -3.5 -> -4
  EXPECT_EQ(referenceIntegerTail(tail, req, 5), 2);
  EXPECT_EQ(referenceIntegerTail(tail, req, 7), 4);
  EXPECT_EQ(referenceIntegerTail(tail, req, -5), -2);
  EXPECT_EQ(referenceIntegerTail(tail, req, -7), -4);
  EXPECT_EQ(referenceIntegerTail(tail, req, 1), 0);
  EXPECT_EQ(referenceIntegerTail(tail, req, -1), 0);
}

TEST(RequantUtils, ZeroPointShiftsTheClampedResult) {
  RequantTail tail = makeTail(1.0, 4.0);
  tail.zeroPoint = 10;
  IntegerRequant req;
  ASSERT_TRUE(chooseMultiplierAndShift(0.25, req));
  ASSERT_TRUE(integerRequantIsExact(tail, req));
  EXPECT_EQ(referenceIntegerTail(tail, req, 8), 12);
  EXPECT_EQ(referenceIntegerTail(tail, req, 0), 10);
}

TEST(RequantUtils, SaturatesLikeTheFloatTail) {
  RequantTail tail = makeTail(1.0, 1.0);
  IntegerRequant req;
  ASSERT_TRUE(chooseMultiplierAndShift(1.0, req));
  ASSERT_TRUE(integerRequantIsExact(tail, req));
  EXPECT_EQ(referenceIntegerTail(tail, req, 1 << 20), 127);
  EXPECT_EQ(referenceIntegerTail(tail, req, -(1 << 20)), -128);
  EXPECT_EQ(referenceFloatTail(tail, 1 << 20), 127);
  EXPECT_EQ(referenceFloatTail(tail, -(1 << 20)), -128);
}

/// A clamp range far wider than a quantization tail's is declined rather than
/// checked, so the pass leaves such a chain alone.
TEST(RequantUtils, DeclinesAnImplausiblyWideClampRange) {
  RequantTail tail = makeTail(1.0, 0.5);
  tail.clampLo = -1000000;
  tail.clampHi = 1000000;
  IntegerRequant req;
  ASSERT_TRUE(chooseMultiplierAndShift(2.0, req));
  EXPECT_FALSE(integerRequantIsExact(tail, req));
}

/// The exactness check and a direct sweep must agree, including when they say
/// "not exact". A deliberately truncated multiplier is the easy way to produce
/// a tail that really does differ.
TEST(RequantUtils, AgreesWithASweepWhenTheFormsDiffer) {
  // A scale small enough that a 16-bit accumulator sweeps the whole clamp
  // range instead of saturating immediately, so a perturbation has somewhere
  // to show up.
  RequantTail tail = makeTail(1.0, 305.0, /*accBits=*/16);
  double scale = 1.0 / static_cast<double>(f32(305.0).convertToFloat());
  IntegerRequant good;
  ASSERT_TRUE(chooseMultiplierAndShift(scale, good));
  ASSERT_TRUE(integerRequantIsExact(tail, good));

  // Shift the scale by about 0.4%, which moves several rounding boundaries.
  IntegerRequant bad = good;
  bad.multiplier = good.multiplier - (good.multiplier >> 8);

  bool sweepFoundDifference = false;
  for (int64_t acc = -32768; acc <= 32767 && !sweepFoundDifference; ++acc) {
    if (referenceFloatTail(tail, acc) != referenceIntegerTail(tail, bad, acc))
      sweepFoundDifference = true;
  }
  ASSERT_TRUE(sweepFoundDifference) << "the perturbation was too small to "
                                       "produce a difference to detect";
  EXPECT_FALSE(integerRequantIsExact(tail, bad));
}

}  // namespace
}  // namespace mlir::iree_compiler::AMDAIE

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
