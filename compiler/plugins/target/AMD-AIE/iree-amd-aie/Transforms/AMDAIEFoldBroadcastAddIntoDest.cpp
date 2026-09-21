// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-fold-broadcast-add-into-dest"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Return true if `val` is defined to be (broadcast) zero. Mirrors
/// `isDefinedAsZero` in upstream MLIR's
/// `mlir/lib/Dialect/Linalg/Transforms/FoldAddIntoDest.cpp` (a local static
/// there, not reusable from this file).
static bool isDefinedAsZero(Value val) {
  if (!val) return false;
  if (isZeroIntegerOrFloat(val)) return true;
  return TypeSwitch<Operation *, bool>(val.getDefiningOp())
      .Case<linalg::FillOp, linalg::CopyOp>([&](auto op) {
        return op && op.getInputs().size() == 1 &&
               isDefinedAsZero(op.getInputs()[0]);
      })
      .Default([&](auto) { return false; });
}

/// Return true if `map`'s results are a strictly-increasing sequence of
/// `AffineDimExpr`s, i.e. an ordered projection of the iteration space with
/// no permutation and no repeats -- the shape one-hot broadcasting always
/// produces, and the shape a contraction's own (non-transposed) destination
/// always has.
static bool isOrderedProjection(AffineMap map) {
  int64_t prevDimPos = -1;
  for (AffineExpr expr : map.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim || static_cast<int64_t>(dim.getPosition()) <= prevDimPos)
      return false;
    prevDimPos = dim.getPosition();
  }
  return true;
}

/// Fold a broadcasting elementwise add -- e.g. `matmul_result + bias` where
/// `bias` has fewer dims than `matmul_result` -- into the zero-initialized
/// destination of the contraction producing `matmul_result`, eliminating the
/// separate elementwise op. See the pass description in Passes.td for the
/// full before/after example.
///
/// This is the broadcasting counterpart of upstream MLIR's
/// `FoldAddIntoDest` (`mlir/lib/Dialect/Linalg/Transforms/FoldAddIntoDest.cpp`),
/// which only handles a same-shape `linalg.add`. A mismatched-shape bias-add
/// (matmul + an `[N]`-shaped bias) lowers to a broadcasting `linalg.generic`
/// instead, which upstream's pattern doesn't match.
struct FoldBroadcastAddIntoDest final
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasPureTensorSemantics()) {
      llvm::errs() << "DEBUG[refold] no tensor semantics: " << genericOp << "\n";
      return failure();
    }
    if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
      llvm::errs() << "DEBUG[refold] wrong operand counts ins="
                   << genericOp.getNumDpsInputs()
                   << " inits=" << genericOp.getNumDpsInits() << ": "
                   << genericOp << "\n";
      return failure();
    }
    if (!llvm::all_of(genericOp.getIteratorTypesArray(),
                      [](utils::IteratorType it) {
                        return it == utils::IteratorType::parallel;
                      })) {
      llvm::errs() << "DEBUG[refold] non-parallel iterator: " << genericOp << "\n";
      return failure();
    }

    // Body must be a single `arith.addf`/`arith.addi` of the two block
    // arguments, yielded directly -- a plain binary elementwise add.
    Block &body = genericOp.getRegion().front();
    if (body.getOperations().size() != 2) {
      llvm::errs() << "DEBUG[refold] body size != 2 (" << body.getOperations().size()
                   << "): " << genericOp << "\n";
      return failure();
    }
    Operation &addOp = body.front();
    if (!isa<arith::AddFOp, arith::AddIOp>(addOp)) {
      llvm::errs() << "DEBUG[refold] first op not add: " << addOp << "\n";
      return failure();
    }
    Value lhs = addOp.getOperand(0), rhs = addOp.getOperand(1);
    bool usesBothArgs =
        (lhs == body.getArgument(0) && rhs == body.getArgument(1)) ||
        (lhs == body.getArgument(1) && rhs == body.getArgument(0));
    if (!usesBothArgs) {
      llvm::errs() << "DEBUG[refold] add doesn't use both block args: " << genericOp << "\n";
      return failure();
    }
    auto yieldOp = dyn_cast<linalg::YieldOp>(body.back());
    if (!yieldOp || yieldOp.getValues().size() != 1 ||
        yieldOp.getValues()[0] != addOp.getResult(0)) {
      llvm::errs() << "DEBUG[refold] bad yield: " << genericOp << "\n";
      return failure();
    }

    // Split the two inputs into the identity-mapped (full-rank, no
    // broadcast) contraction operand and the broadcasting bias operand.
    OpOperand *in0 = genericOp.getDpsInputOperand(0);
    OpOperand *in1 = genericOp.getDpsInputOperand(1);
    AffineMap map0 = genericOp.getMatchingIndexingMap(in0);
    AffineMap map1 = genericOp.getMatchingIndexingMap(in1);
    OpOperand *contractionOperand, *biasOperand;
    AffineMap biasMap;
    if (map0.isIdentity() && !map1.isIdentity()) {
      contractionOperand = in0;
      biasOperand = in1;
      biasMap = map1;
    } else if (map1.isIdentity() && !map0.isIdentity()) {
      contractionOperand = in1;
      biasOperand = in0;
      biasMap = map0;
    } else {
      llvm::errs() << "DEBUG[refold] map0=" << map0 << " map1=" << map1
                   << " neither/both identity: " << genericOp << "\n";
      return rewriter.notifyMatchFailure(
          genericOp,
          "expected exactly one identity-mapped and one broadcasting input");
    }
    // A genuine broadcast: strictly fewer dims kept than the iteration space,
    // in an ordered (non-transposed) projection.
    if (biasMap.getNumResults() >= biasMap.getNumDims() ||
        !isOrderedProjection(biasMap)) {
      llvm::errs() << "DEBUG[refold] biasMap not a strict ordered broadcast: "
                   << biasMap << ": " << genericOp << "\n";
      return rewriter.notifyMatchFailure(
          genericOp, "expected bias operand to have a strict, ordered "
                     "broadcasting indexing map");
    }

    // The contraction operand must be the single result of a single-result,
    // destination-passing contraction op, and this add must be its only use
    // (mirrors upstream `FoldAddIntoDest`'s dominance/single-use checks).
    auto contractionOp =
        contractionOperand->get().getDefiningOp<linalg::LinalgOp>();
    if (!contractionOp) return failure();
    auto destOp =
        dyn_cast<DestinationStyleOpInterface>(contractionOp.getOperation());
    if (contractionOp->getNumResults() != 1 ||
        !linalg::isaContractionOpInterface(contractionOp) ||
        !destOp || destOp.getNumDpsInits() != 1) {
      return rewriter.notifyMatchFailure(
          contractionOp, "expected a single-result, destination-passing "
                         "contraction");
    }
    if (!contractionOp->getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(
          contractionOp, "expected this add to be the contraction's only "
                         "use");
    }

    // Only safe to replace the contraction's current dest when it's the
    // additive identity (zero) -- otherwise the contraction is already
    // accumulating something else.
    OpOperand *destOperand = destOp.getDpsInitOperand(0);
    if (!isDefinedAsZero(destOperand->get())) {
      return rewriter.notifyMatchFailure(
          contractionOp, "expected contraction's dest to be additive zero");
    }
    // Guard against unusual (e.g. transposed) dest layouts, mirroring
    // upstream `FoldAddIntoDest`.
    AffineMap destMap = contractionOp.getIndexingMapsArray()[destOperand
                                                                  ->getOperandNumber()];
    if (!isOrderedProjection(destMap)) {
      return rewriter.notifyMatchFailure(
          contractionOp, "expected contraction's dest indexing map to be an "
                         "ordered projection");
    }

    // The bias must dominate the contraction so it's valid to feed it in as
    // the contraction's (earlier-running) dest operand.
    DominanceInfo domInfo(contractionOp);
    if (!domInfo.properlyDominates(biasOperand->get(), contractionOp)) {
      return rewriter.notifyMatchFailure(
          contractionOp, "bias operand does not dominate the contraction");
    }

    // Materialize the broadcast of the bias into the contraction dest's
    // shape, and use that as the contraction's new dest -- since a
    // contraction accumulates additively on its dest, this makes its result
    // `bias + contraction(...)`, i.e. exactly this generic op's result.
    auto destType = cast<RankedTensorType>(destOperand->get().getType());
    llvm::SmallDenseSet<int64_t> keptDims;
    for (AffineExpr expr : biasMap.getResults())
      keptDims.insert(cast<AffineDimExpr>(expr).getPosition());
    SmallVector<int64_t> broadcastDims;
    for (int64_t d = 0, e = destType.getRank(); d < e; ++d)
      if (!keptDims.contains(d)) broadcastDims.push_back(d);

    rewriter.setInsertionPoint(contractionOp);
    Value newInit = rewriter.create<tensor::EmptyOp>(
        contractionOp.getLoc(), destType.getShape(),
        destType.getElementType());
    Value broadcast = rewriter.create<linalg::BroadcastOp>(
        contractionOp.getLoc(), biasOperand->get(), newInit, broadcastDims)
                          ->getResult(0);

    rewriter.modifyOpInPlace(contractionOp, [&]() {
      contractionOp->setOperand(destOperand->getOperandNumber(), broadcast);
    });
    rewriter.replaceAllOpUsesWith(genericOp, contractionOp->getResult(0));
    return success();
  }
};

struct AMDAIEFoldBroadcastAddIntoDestPass
    : public impl::AMDAIEFoldBroadcastAddIntoDestBase<
          AMDAIEFoldBroadcastAddIntoDestPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldBroadcastAddIntoDest>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEFoldBroadcastAddIntoDestPass() {
  return std::make_unique<AMDAIEFoldBroadcastAddIntoDestPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
