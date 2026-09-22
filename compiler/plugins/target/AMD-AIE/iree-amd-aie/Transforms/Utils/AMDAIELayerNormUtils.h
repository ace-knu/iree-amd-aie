// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIELAYERNORMUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIELAYERNORMUTILS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::AMDAIE {

/// A quantized row-wise LayerNorm, as an int8 model's encoder emits it after
/// the ONNX importer has decomposed `onnx.LayerNormalization`:
///
///   dequantize(i8 -> f32)
///     -> mean   = sum(x) / N
///     -> centred = x - mean
///     -> var     = sum(centred^2) / N
///     -> y       = centred * rsqrt(var + eps) * gamma + beta
///       -> quantize(f32 -> i8)
///
/// There is no named op for this, so it arrives as a chain of thirteen
/// `linalg.generic`s that dispatch formation then splits at the second
/// reduction. Recognising the whole chain lets it be raised to a single tiling
/// root, and lets the int8 scales be folded away: with m and v the mean and
/// variance of the *integer* samples,
///
///   (x*s - m*s) / sqrt(v*s^2 + eps) = (x - m) / sqrt(v + eps/s^2)
///
/// so the input scale only survives inside `epsilonScaled`, and the output
/// scale divides gamma and beta, both compile-time constants.
struct QuantizedLayerNormDAG {
  /// The int8 tensor the leading dequantize reads.
  Value input;
  /// When the normalization is fed by a residual add, the second int8 tensor
  /// and its scale. The add stays outside any microkernel ABI: it becomes an
  /// ordinary integer-domain elementwise producer of the wider samples the
  /// kernel's int16 variant takes.
  Value residual;
  double residualScale = 0.0;
  /// The `[N]`-shaped gamma and beta the chain multiplies and adds.
  Value gamma;
  Value beta;
  /// The quantize at the end of the chain, which the rewrite is rooted on.
  linalg::GenericOp quantizeOp;
  /// Every op of the chain, in no particular order, for moving as a unit.
  SetVector<Operation *> chain;
  /// The chain's constants, as written. A microkernel lowering folds them
  /// (`eps / inputScale^2`, and gamma/beta divided by `outputScale`); the
  /// rebuilt fallback body needs them as they are.
  double inputScale = 1.0;
  /// The step of the samples the kernel actually sees: the input scale, or the
  /// residual producer's common step when there is one. Set by the rewrite.
  mutable double sampleStep = 1.0;
  double epsilon = 0.0;
  double outputScale = 1.0;
  /// Rows and the (untiled) feature length.
  int64_t numRows = 0;
  int64_t featureSize = 0;
};

/// Marks an `iree_linalg_ext.custom_op` that holds a quantized LayerNorm.
/// Device placement, pipeline selection, tiling and the microkernel lowering
/// all key off this rather than re-running the matcher below.
constexpr StringLiteral kLayerNormMarker = "amdaie.layernorm";

/// `epsilon / inputScale^2`, the only trace either quantization scale leaves in
/// the raised op: the output scale is folded into the packed gamma/beta operand
/// and the input scale cancels out of the normalization.
constexpr StringLiteral kLayerNormEpsilonScaled =
    "amdaie.layernorm_epsilon_scaled";

/// The packed `[2, N]` bf16 constant the microkernel reads: row 0 is
/// `gamma / outputScale`, row 1 is `beta / outputScale`.
///
/// It rides on the op as an attribute rather than as an operand because a core
/// tile has only two incoming DMA channels and a residual add already needs
/// both of them for its two int8 activations. The microkernel lowering turns
/// the attribute into a `memref.global` that the per-core linker script places
/// in the core's own data memory, so it costs no channel and no transfer.
constexpr StringLiteral kLayerNormGammaBeta = "amdaie.layernorm_gamma_beta";

/// Whether `op` is an executable holding nothing but one marked LayerNorm.
bool isQuantizedLayerNormOnly(Operation *op);

/// Match `quantizeOp` as the tail of a quantized LayerNorm and report the whole
/// chain. The match is deliberately strict: every op in the chain is
/// single-use, the reductions are over the last dimension, the scales, epsilon
/// and `1/N` are compile-time constants, the zero points are zero, and gamma
/// and beta are `[N]`-shaped. Anything else is left for ordinary codegen.
FailureOr<QuantizedLayerNormDAG> matchQuantizedLayerNorm(
    linalg::GenericOp quantizeOp);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIELAYERNORMUTILS_H_
