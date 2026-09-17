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

bool integerRequantIsExact(const RequantTail &tail, const IntegerRequant &req) {
  if (tail.clampHi < tail.clampLo) return false;
  if (tail.clampHi - tail.clampLo > kMaxClampSteps) return false;
  if (tail.accBitWidth == 0 || tail.accBitWidth > 32) return false;

  const int64_t accLo = -(int64_t{1} << (tail.accBitWidth - 1));
  const int64_t accHi = (int64_t{1} << (tail.accBitWidth - 1)) - 1;

  // Guard the i64 multiply the integer form will do.
  const int64_t maxAbsAcc = std::max<int64_t>(-accLo, accHi);
  if (req.multiplier != 0 &&
      maxAbsAcc > std::numeric_limits<int64_t>::max() / req.multiplier)
    return false;

  // The clamped result is determined by which of the values in
  // (clampLo, clampHi] it has reached, so comparing the first accumulator that
  // reaches each of them compares the two functions everywhere.
  for (int64_t value = tail.clampLo + 1; value <= tail.clampHi; ++value) {
    // The clamp is applied after the zero point is added, so the tail itself
    // has to reach `value - zeroPoint`.
    const int64_t threshold = value - tail.zeroPoint;
    int64_t fromFloat = firstReaching(accLo, accHi, [&](int64_t acc) {
      return floatTailReaches(tail, acc, threshold);
    });
    int64_t fromInteger = firstReaching(accLo, accHi, [&](int64_t acc) {
      return integerTailReaches(req, acc, threshold);
    });
    if (fromFloat != fromInteger) return false;
  }
  return true;
}

}  // namespace mlir::iree_compiler::AMDAIE
