// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIEREQUANTUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIEREQUANTUTILS_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/APFloat.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::AMDAIE {

/// The shape of an int8-style requantization tail:
///
///   sitofp(acc) -> [* mulConst] -> [/ divConst] -> roundeven -> [+ zeroPoint]
///                -> clamp(clampLo, clampHi) -> fptosi
///
/// where `acc` is a signed integer of `accBitWidth` bits and every constant is
/// known at compile time. `mulConst` and `divConst` are f32 values, and the
/// scaling is evaluated in f32 exactly as written, one rounding per operation.
struct RequantTail {
  std::optional<APFloat> mulConst;
  std::optional<APFloat> divConst;
  int64_t zeroPoint = 0;
  int64_t clampLo = 0;
  int64_t clampHi = 0;
  unsigned accBitWidth = 32;
};

/// A requantization tail rewritten as integer arithmetic: the scale becomes
/// `multiplier / 2^shift`, applied to the accumulator with round-half-to-even.
struct IntegerRequant {
  int64_t multiplier = 0;
  int64_t shift = 0;
};

/// Represent `scale` as `multiplier / 2^shift` with a multiplier of about 31
/// bits, which is the most an i32-bounded accumulator can use while keeping
/// `acc * multiplier` inside an i64. Fails if `scale` is not a positive finite
/// number, or if no usable shift exists.
bool chooseMultiplierAndShift(double scale, IntegerRequant &out);

/// Evaluate the float tail of `tail` for the accumulator value `acc`, and
/// report whether the result reaches `threshold` before clamping. The
/// arithmetic is done in f32 with round-to-nearest-even, matching what the
/// device computes.
bool floatTailReaches(const RequantTail &tail, int64_t acc, int64_t threshold);

/// Same question for the integer form.
bool integerTailReaches(const IntegerRequant &req, int64_t acc,
                        int64_t threshold);

/// Decide whether `req` reproduces `tail` exactly, for *every* accumulator
/// value the tail's integer type can hold.
///
/// This is not a general property of 31-bit multipliers: whether the two agree
/// depends on the specific scales, on the clamp range and on how wide the
/// accumulator is. It is checked rather than assumed.
///
/// Both forms are non-decreasing in the accumulator (f32 multiply and divide by
/// positive constants are monotone under round-to-nearest, and so is rounding
/// to integral), and after clamping both are step functions that only take
/// values in [clampLo, clampHi]. Two monotone step functions with the same
/// range are equal everywhere exactly when each of their steps happens at the
/// same place, so it is enough to locate, for every value the clamped result
/// can take, the first accumulator that reaches it. That is a binary search per
/// step instead of a sweep over the whole integer range.
///
/// Returns false if they differ anywhere, or if the clamp range is too wide to
/// check cheaply.
bool integerRequantIsExact(const RequantTail &tail, const IntegerRequant &req);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIEREQUANTUTILS_H_
