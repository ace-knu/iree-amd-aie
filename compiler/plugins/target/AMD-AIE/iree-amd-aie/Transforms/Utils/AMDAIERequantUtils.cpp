// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Utils/AMDAIERequantUtils.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::iree_compiler::AMDAIE {

/// The widest clamp range this is willing to check. int8 tails need 256 steps;
/// anything far past that is either not a quantization tail or not worth the
/// compile time, and is declined rather than assumed.
static constexpr int64_t kMaxClampSteps = 4096;

bool chooseMultiplierAndShift(double scale, IntegerRequant &out) {
  if (!(scale > 0.0) || !std::isfinite(scale)) return false;
  int64_t n = 0;
  while (n < 62 && std::ldexp(scale, n) < static_cast<double>(1u << 30)) ++n;
  double scaled = std::ldexp(scale, n);
  if (!std::isfinite(scaled)) return false;
  int64_t m = std::llround(scaled);
  if (m >= (int64_t{1} << 31)) {
    m >>= 1;
    --n;
  }
  if (n < 1 || n > 62 || m <= 0) return false;
  out.multiplier = m;
  out.shift = n;
  return true;
}

bool floatTailReaches(const RequantTail &tail, int64_t acc, int64_t threshold) {
  // int -> f32, exactly as the device converts it.
  APFloat y(APFloat::IEEEsingle());
  y.convertFromAPInt(APInt(64, acc, /*isSigned=*/true), /*IsSigned=*/true,
                     APFloat::rmNearestTiesToEven);
  if (tail.mulConst)
    y.multiply(*tail.mulConst, APFloat::rmNearestTiesToEven);
  if (tail.divConst) y.divide(*tail.divConst, APFloat::rmNearestTiesToEven);
  // math.roundeven
  y.roundToIntegral(APFloat::rmNearestTiesToEven);

  APFloat t(APFloat::IEEEsingle());
  t.convertFromAPInt(APInt(64, threshold, /*isSigned=*/true), /*IsSigned=*/true,
                     APFloat::rmNearestTiesToEven);
  // NaN cannot arise here: the input is an integer and the constants are
  // finite and positive, so the comparison is always ordered.
  return y.compare(t) != APFloat::cmpLessThan;
}

bool integerTailReaches(const IntegerRequant &req, int64_t acc,
                        int64_t threshold) {
  int64_t p = acc * req.multiplier;
  int64_t q = p >> req.shift;
  int64_t rem = p & ((int64_t{1} << req.shift) - 1);
  int64_t half = int64_t{1} << (req.shift - 1);
  bool roundUp = rem > half || (rem == half && (q & 1) != 0);
  if (roundUp) ++q;
  return q >= threshold;
}

/// The first accumulator in [lo, hi] for which `reaches` holds, or `hi + 1`
/// when it never does. `reaches` must be monotone over the interval.
static int64_t firstReaching(int64_t lo, int64_t hi,
                             llvm::function_ref<bool(int64_t)> reaches) {
  if (reaches(lo)) return lo;
  if (!reaches(hi)) return hi + 1;
  // Invariant: !reaches(lo) and reaches(hi).
  while (hi - lo > 1) {
    int64_t mid = lo + (hi - lo) / 2;
    if (reaches(mid))
      hi = mid;
    else
      lo = mid;
  }
  return hi;
}

/// The accumulator's value range. Fails only on a width the rewrite cannot
/// express; the clamp range is not its business.
static bool accBounds(const RequantTail &tail, int64_t &lo, int64_t &hi) {
  if (tail.accBitWidth == 0 || tail.accBitWidth > 32) return false;
  lo = -(int64_t{1} << (tail.accBitWidth - 1));
  hi = (int64_t{1} << (tail.accBitWidth - 1)) - 1;
  return true;
}

/// As above, plus the budget the exactness check needs: it walks one binary
/// search per clamp step, so a very wide clamp is declined rather than swept.
static bool accRange(const RequantTail &tail, int64_t &lo, int64_t &hi) {
  if (tail.clampHi < tail.clampLo) return false;
  if (tail.clampHi - tail.clampLo > kMaxClampSteps) return false;
  return accBounds(tail, lo, hi);
}

bool integerRequantIsSafe(const RequantTail &tail, const IntegerRequant &req) {
  int64_t accLo, accHi;
  if (!accBounds(tail, accLo, accHi)) return false;
  if (req.shift < 1 || req.shift > 62) return false;
  if (req.multiplier <= 0) return false;
  // Guard the i64 multiply the integer form will do.
  const int64_t maxAbsAcc = std::max<int64_t>(-accLo, accHi);
  return maxAbsAcc <= std::numeric_limits<int64_t>::max() / req.multiplier;
}

/// Where the float tail's clamped steps happen. This does not depend on the
/// integer representation, so a search over representations computes it once.
static bool floatStepStarts(const RequantTail &tail, int64_t accLo,
                            int64_t accHi, SmallVectorImpl<int64_t> &starts) {
  starts.clear();
  for (int64_t value = tail.clampLo + 1; value <= tail.clampHi; ++value) {
    // The clamp is applied after the zero point is added, so the tail itself
    // has to reach `value - zeroPoint`.
    const int64_t threshold = value - tail.zeroPoint;
    starts.push_back(firstReaching(accLo, accHi, [&](int64_t acc) {
      return floatTailReaches(tail, acc, threshold);
    }));
  }
  return true;
}

/// Whether `req` puts every step where `starts` says the float tail puts it.
static bool integerMatchesSteps(const RequantTail &tail,
                                const IntegerRequant &req, int64_t accLo,
                                int64_t accHi, ArrayRef<int64_t> starts) {
  if (!integerRequantIsSafe(tail, req)) return false;
  size_t i = 0;
  for (int64_t value = tail.clampLo + 1; value <= tail.clampHi; ++value, ++i) {
    const int64_t threshold = value - tail.zeroPoint;
    int64_t fromInteger = firstReaching(accLo, accHi, [&](int64_t acc) {
      return integerTailReaches(req, acc, threshold);
    });
    if (fromInteger != starts[i]) return false;
  }
  return true;
}

bool integerRequantIsExact(const RequantTail &tail, const IntegerRequant &req) {
  int64_t accLo, accHi;
  if (!accRange(tail, accLo, accHi)) return false;
  // Keep the historical behaviour of accepting a zero multiplier's guard check
  // by failing it the same way `integerMatchesSteps` does.
  SmallVector<int64_t> starts;
  if (!floatStepStarts(tail, accLo, accHi, starts)) return false;
  return integerMatchesSteps(tail, req, accLo, accHi, starts);
}

/// How far the multiplier is allowed to drift from the value that represents
/// `scale` at a given shift. The boundaries move by at most one accumulator per
/// unit of multiplier at these magnitudes, so a handful either side is enough to
/// find a representation that lands them all correctly when one exists.
static constexpr int64_t kMultiplierWindow = 4;

/// The largest multiplier worth trying. Past this the i64 guard rejects every
/// candidate for a 32-bit accumulator anyway.
static constexpr int64_t kMaxMultiplier = int64_t{1} << 33;

bool findExactMultiplierAndShift(const RequantTail &tail, double scale,
                                 IntegerRequant &out) {
  int64_t accLo, accHi;
  if (!accRange(tail, accLo, accHi)) return false;
  if (!(scale > 0.0) || !std::isfinite(scale)) return false;
  SmallVector<int64_t> starts;
  if (!floatStepStarts(tail, accLo, accHi, starts)) return false;

  // The preferred representation first, so a tail that lowers to integers today
  // keeps the multiplier and shift it has today.
  IntegerRequant preferred;
  if (chooseMultiplierAndShift(scale, preferred) &&
      integerMatchesSteps(tail, preferred, accLo, accHi, starts)) {
    out = preferred;
    return true;
  }

  for (int64_t shift = 1; shift <= 62; ++shift) {
    double scaled = std::ldexp(scale, shift);
    if (!std::isfinite(scaled)) break;
    int64_t base = std::llround(scaled);
    if (base <= 0) continue;
    if (base > kMaxMultiplier) break;
    for (int64_t delta = -kMultiplierWindow; delta <= kMultiplierWindow;
         ++delta) {
      IntegerRequant cand{base + delta, shift};
      if (cand.multiplier <= 0) continue;
      if (integerMatchesSteps(tail, cand, accLo, accHi, starts)) {
        out = cand;
        return true;
      }
    }
  }
  return false;
}

}  // namespace mlir::iree_compiler::AMDAIE
