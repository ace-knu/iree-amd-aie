// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIESOFTMAXUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIESOFTMAXUTILS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::AMDAIE {

/// A quantized softmax, as an int8 model's attention emits it:
///
///   dequantize(i8 -> f32, scaling by `inputScale`)
///     -> linalg.softmax over the last dimension
///       -> quantize(f32 -> i8, by `1 / outputScale`, round-to-nearest, clamp)
///
/// The three ops always travel together and are the whole dispatch, so they can
/// be replaced by a single int8-in/int8-out microkernel call that folds both
/// scales in. That is worth doing beyond saving two passes over the data: aie2p
/// has no scalar float arithmetic, so leaving the dequantize/quantize outside
/// the kernel turns them into soft-float libcalls costing several KB of a
/// core's 16KB program memory.
struct QuantizedSoftmaxDAG {
  linalg::GenericOp dequantizeOp;
  linalg::SoftmaxOp softmaxOp;
  linalg::GenericOp quantizeOp;
  /// The int8 tensor the dequantize reads.
  Value input;
  /// Product of the dequantize's scale factors.
  double inputScale = 1.0;
  /// The quantize's divisor.
  double outputScale = 1.0;
  int64_t clampLo = 0;
  int64_t clampHi = 0;
};

/// Match `quantizeOp` as the tail of a quantized softmax, and report the whole
/// chain. The match is deliberately strict: every op is single-use, every map
/// is an identity, the scales are compile-time constants, the softmax reduces
/// the last dimension, and the zero points are zero. Anything else is left for
/// ordinary codegen.
FailureOr<QuantizedSoftmaxDAG> matchQuantizedSoftmax(linalg::GenericOp quantizeOp);

/// Whether `op` region contains exactly one quantized softmax chain and no
/// other linalg op, i.e. the dispatch is nothing but a quantized softmax.
bool isQuantizedSoftmaxOnly(Operation *op);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIESOFTMAXUTILS_H_
