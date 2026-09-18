// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-cleanup"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// A `linalg.generic` that only copies its single input into a bigger result,
/// broadcasting along dimensions that are all unit-extent, computes nothing:
/// it is a rank-expanding reshape written as a compute op.
///
/// This is what a shared matmul operand's batch broadcast turns into once the
/// contraction has been tiled, because the batch tile size is 1. Left as a
/// compute op it forces the whole operand to be packed and staged in L1 before
/// the K loop, instead of being streamed a K-slice at a time. As a
/// `tensor.expand_shape` it folds into the operand's access pattern and the
/// schedule is the same as if the broadcast had never been there.
struct FoldUnitExtentBroadcastToExpandShape
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasPureTensorSemantics() ||
        genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
      return rewriter.notifyMatchFailure(genericOp, "not a one-operand copy");
    }
    if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
      return rewriter.notifyMatchFailure(genericOp, "has a reduction loop");
    }

    // The body must be exactly `linalg.yield <input element>`.
    Block &body = genericOp.getRegion().front();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1 ||
        yieldOp.getOperand(0) != body.getArgument(0) ||
        !body.without_terminator().empty()) {
      return rewriter.notifyMatchFailure(genericOp, "body is not a copy");
    }

    // The result must be written in iteration order, and the input read in the
    // same order with some dimensions simply missing (a broadcast, not a
    // transpose).
    if (!genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0))
             .isIdentity()) {
      return rewriter.notifyMatchFailure(genericOp, "result is not identity");
    }
    AffineMap inputMap =
        genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0));
    if (!inputMap.isProjectedPermutation(/*allowZeroInResults=*/false)) {
      return rewriter.notifyMatchFailure(genericOp, "input is not a broadcast");
    }
    SmallVector<int64_t> keptDims;
    for (AffineExpr expr : inputMap.getResults()) {
      keptDims.push_back(cast<AffineDimExpr>(expr).getPosition());
    }
    if (keptDims.empty() || !llvm::is_sorted(keptDims)) {
      return rewriter.notifyMatchFailure(genericOp, "input dims are permuted");
    }

    auto resultType = cast<RankedTensorType>(genericOp.getResultTypes()[0]);
    int64_t resultRank = resultType.getRank();
    if (resultRank == static_cast<int64_t>(keptDims.size())) {
      return rewriter.notifyMatchFailure(genericOp, "nothing is broadcast");
    }
    // Only a unit-extent broadcast is free. Anything else really does
    // replicate data and has to stay a copy.
    llvm::SmallDenseSet<int64_t> kept(keptDims.begin(), keptDims.end());
    for (int64_t dim = 0; dim < resultRank; ++dim) {
      if (!kept.contains(dim) && resultType.getDimSize(dim) != 1) {
        return rewriter.notifyMatchFailure(genericOp,
                                           "broadcast is not unit-extent");
      }
    }

    // Attach each broadcast (unit) dimension to the neighbouring source
    // dimension's group, so the expansion is 1 x <source dim>.
    SmallVector<ReassociationIndices> reassociation(keptDims.size());
    int64_t seen = 0;
    for (int64_t dim = 0; dim < resultRank; ++dim) {
      int64_t group = std::min<int64_t>(seen, keptDims.size() - 1);
      reassociation[group].push_back(dim);
      if (kept.contains(dim)) ++seen;
    }

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        genericOp, resultType, genericOp.getDpsInputOperand(0)->get(),
        reassociation);
    return success();
  }
};

static void loopIndependentCodeMotion(Operation *funcOp, IRRewriter &rewriter) {
  // This assumes LICM never removes operations so we don't need tracking.
  // TODO: confirm / revisit this assumption and plumb a rewriter through
  // upstream moveLoopInvariantCode if necessary.
  funcOp->walk([](LoopLikeOpInterface loopLike) {
    // Do not hoist from scf.forall ops. These capture isolated computations
    // that will be mapped to a certain level in the GPU hierarchy (e.g.,
    // GPU blocks), so hoisting is not desired.
    if (!isa<scf::ForallOp>(loopLike.getOperation()))
      moveLoopInvariantCode(loopLike);
  });
  // For now, put single loop promotion as part of licm. Underlying
  // implementations perform splice operations which shouldn't need
  // tracking.
  // TODO: confirm / revisit this assumption and plumb a rewriter through
  // upstream moveLoopInvariantCode if necessary.
  funcOp->walk([&](Operation *op) {
    (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<affine::AffineForOp, scf::ForOp>(
            [&](auto loop) { return loop.promoteIfSingleIteration(rewriter); })
        .Default([](Operation *) { return success(); });
  });
}

static void populateCleanupPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  linalg::FillOp::getCanonicalizationPatterns(patterns, context);
  // Pulling in upstream scf.for and affine.min canonicalization patterns.
  // They work on tiled (but not distributed) loops.
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  patterns.add<FoldUnitExtentBroadcastToExpandShape>(context);
}

class AMDAIECleanupPass : public impl::AMDAIECleanupBase<AMDAIECleanupPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect>();
  }

  AMDAIECleanupPass() = default;
  AMDAIECleanupPass(const AMDAIECleanupPass &pass){};

  void runOnOperation() override;
};

void AMDAIECleanupPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  RewritePatternSet patterns(context);
  populateCleanupPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }
  {
    IRRewriter rewriter(context);
    loopIndependentCodeMotion(funcOp, rewriter);
  }
}
}  // namespace

std::unique_ptr<Pass> createAMDAIECleanupPass() {
  return std::make_unique<AMDAIECleanupPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
