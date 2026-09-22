// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-approximate-math-functions"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// The transcendentals a core can be given as polynomials. A core has no libm,
/// so anything left here has no lowering at all: GELU's `math.erf` reaches LLVM
/// translation and fails with "missing LLVMTranslationDialectInterface
/// registration for dialect for op: math.erf".
bool isApproximable(StringRef name) {
  return llvm::is_contained(
      {math::AtanOp::getOperationName(), math::Atan2Op::getOperationName(),
       math::TanhOp::getOperationName(), math::LogOp::getOperationName(),
       math::Log2Op::getOperationName(), math::Log1pOp::getOperationName(),
       math::ErfOp::getOperationName(), math::AsinOp::getOperationName(),
       math::AcosOp::getOperationName(), math::ExpOp::getOperationName(),
       math::ExpM1Op::getOperationName(), math::CbrtOp::getOperationName(),
       math::SinOp::getOperationName(), math::CosOp::getOperationName()},
      name);
}

/// `math.fma` -> `arith.mulf` + `arith.addf`.
///
/// The approximations above are written with `math.fma`, and peano's aie2p
/// backend cannot legalize the scalar form (`G_FMA`) -- unlike plain scalar
/// multiply and add, which at least bottom out on soft-float libcalls, and
/// unlike the vector form, which the core's MAC unit implements directly.
/// Splitting it loses the single-rounding guarantee, which a polynomial
/// approximation does not depend on.
struct ExpandFmaPattern : OpRewritePattern<math::FmaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const override {
    Value mul = rewriter.create<arith::MulFOp>(op.getLoc(), op.getA(),
                                               op.getB(), op.getFastmathAttr());
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, mul, op.getC(),
                                               op.getFastmathAttr());
    return success();
  }
};

class AMDAIEApproximateMathFunctions
    : public impl::AMDAIEApproximateMathFunctionsBase<
          AMDAIEApproximateMathFunctions> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, arith::ArithDialect, math::MathDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Narrower floats go through f32, which is the width the approximations are
    // written for; a bf16 GELU is the common case.
    populateMathF32ExpansionPatterns(patterns, isApproximable);
    populateMathPolynomialApproximationPatterns(patterns, isApproximable);
    patterns.insert<ExpandFmaPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEApproximateMathFunctionsPass() {
  return std::make_unique<AMDAIEApproximateMathFunctions>();
}

}  // namespace mlir::iree_compiler::AMDAIE
