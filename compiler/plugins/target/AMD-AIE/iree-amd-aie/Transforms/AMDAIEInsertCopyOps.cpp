// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-insert-copy-ops"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Promote a value by allocating a new buffer or using the provided output
// buffer, then copying the value into it.
FailureOr<Value> promoteValue(IRRewriter &rewriter, Location loc, Value v,
                              Value copyDest = Value()) {
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType) {
    llvm::errs() << "expected a ranked tensor type\n";
    return failure();
  }

  // If no output buffer is provided, allocate a new buffer.
  if (!copyDest) {
    SmallVector<Value> dynamicSizes;
    for (auto [idx, size] : llvm::enumerate(tensorType.getShape())) {
      if (ShapedType::isDynamic(size)) {
        dynamicSizes.push_back(rewriter.create<tensor::DimOp>(loc, v, idx));
      }
    }
    auto alloc = rewriter.create<bufferization::AllocTensorOp>(loc, tensorType,
                                                               dynamicSizes);
    copyDest = alloc.getResult();
  }

  auto copy = rewriter.create<linalg::CopyOp>(loc, v, copyDest);
  return copy.getResult(0);
}

/// `initDest`, when set, is the tensor the op's init was read from before its
/// inputs were promoted; the result is copied back into it (the copy level
/// below already writes that tensor out to the forall's output).
LogicalResult promoteResults(IRRewriter &rewriter, Operation *op,
                             Value initDest = Value()) {
  OpBuilder::InsertionGuard g(rewriter);

  // Only a single output is supported.
  if (op->getNumResults() != 1)
    return op->emitError("expected a single output");

  // Only ranked tensor is supported.
  Value result = op->getResults()[0];
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType) return op->emitError("expected ranked tensor result");

  // Handle case where the target op is inside scf.forall and we want to use
  // block arguments as copy destinations.
  Value outputBuffer = initDest;
  if (auto forallOp = op->getParentOfType<scf::ForallOp>();
      forallOp && !outputBuffer) {
    if (forallOp.getNumResults() != 1)
      return forallOp->emitError("expected a single output");

    Block &block = forallOp.getRegion().front();
    unsigned numIvs = forallOp.getInductionVars().size();
    Value blockArg = block.getArgument(numIvs);
    auto blockArgType = dyn_cast<RankedTensorType>(blockArg.getType());
    if (!blockArgType)
      return forallOp->emitError("expected ranked tensor block argument");

    // Create tensor.extract_slice on block arg. The slice the result occupies
    // is whatever the forall's terminator inserts it at: take the offsets from
    // that `tensor.parallel_insert_slice`, so a forall over any dim (e.g. the
    // GEMV pipeline tiles N, dim 1) lands on the right slice. Fall back to
    // "one induction variable per leading dim" when the result is not
    // inserted directly.
    rewriter.setInsertionPoint(op);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    tensor::ParallelInsertSliceOp insertOp;
    for (Operation &termOp : forallOp.getTerminator().getYieldingOps()) {
      auto candidate = dyn_cast<tensor::ParallelInsertSliceOp>(termOp);
      if (candidate && candidate.getSource() == result &&
          candidate.getDest() == blockArg) {
        insertOp = candidate;
        break;
      }
    }
    if (insertOp) {
      offsets = insertOp.getMixedOffsets();
      sizes = insertOp.getMixedSizes();
      strides = insertOp.getMixedStrides();
    } else {
      for (Value iv : forallOp.getInductionVars()) offsets.push_back(iv);
      // Pad offset with zeros if iv size is smaller than the rank.
      for (unsigned i = numIvs; i < resultType.getRank(); ++i)
        offsets.push_back(rewriter.getIndexAttr(0));
      for (int64_t d : resultType.getShape())
        sizes.push_back(rewriter.getIndexAttr(d));
      strides.assign(resultType.getRank(), rewriter.getIndexAttr(1));
    }

    outputBuffer = rewriter.create<tensor::ExtractSliceOp>(
        op->getLoc(), blockArg, offsets, sizes, strides);
  }

  rewriter.setInsertionPointAfter(op);
  FailureOr<Value> maybeReplacement =
      promoteValue(rewriter, op->getLoc(), result, outputBuffer);
  if (failed(maybeReplacement))
    return op->emitError() << "failed to promote result";

  rewriter.replaceUsesWithIf(result, *maybeReplacement, [&](OpOperand &use) {
    // Only replace uses not inside the newly created copy op.
    return use.getOwner() != maybeReplacement->getDefiningOp();
  });
  return success();
}

/// Promotes `op`'s operands. With `onlyInits`, its inputs are left alone and
/// an init the payload never reads gets a fresh, uninitialized buffer rather
/// than a copy of its (dead) contents.
LogicalResult promoteInputs(IRRewriter &rewriter, Operation *op,
                            bool onlyInits = false) {
  OpBuilder::InsertionGuard g(rewriter);
  auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dstStyleOp) return failure();

  Location loc = dstStyleOp.getLoc();
  unsigned numDpsInputs = dstStyleOp.getNumDpsInputs();

  if (onlyInits) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp) return failure();
    for (OpOperand &init : linalgOp.getDpsInitsMutable()) {
      rewriter.setInsertionPoint(op);
      Value replacement;
      if (linalgOp.payloadUsesValueFromOperand(&init)) {
        FailureOr<Value> maybeReplacement =
            promoteValue(rewriter, loc, init.get());
        if (failed(maybeReplacement))
          return dstStyleOp.emitError() << "failed to promote init";
        replacement = *maybeReplacement;
      } else {
        auto tensorType = dyn_cast<RankedTensorType>(init.get().getType());
        if (!tensorType || !tensorType.hasStaticShape())
          return dstStyleOp.emitError() << "expected a static init";
        replacement =
            rewriter.create<bufferization::AllocTensorOp>(loc, tensorType,
                                                          ValueRange{});
      }
      init.set(replacement);
    }
    return success();
  }

  // Promote the input operands.
  for (auto [i, operand] : llvm::enumerate(dstStyleOp.getDpsInputs())) {
    // An input with no defined contents (a pooling window `tensor.empty` that
    // only sizes the reduction loops) has nothing to move.
    if (isUndefinedTensor(operand)) continue;
    rewriter.setInsertionPoint(op);
    FailureOr<Value> maybeReplacement = promoteValue(rewriter, loc, operand);
    if (failed(maybeReplacement))
      return dstStyleOp.emitError() << "failed to promote input " << i;
    op->setOperand(i, *maybeReplacement);
  }

  // Promote the init operands.
  for (auto [i, operand] : llvm::enumerate(dstStyleOp.getDpsInits())) {
    rewriter.setInsertionPoint(op);
    FailureOr<Value> maybeReplacement = promoteValue(rewriter, loc, operand);
    if (failed(maybeReplacement))
      return dstStyleOp.emitError() << "failed to promote init " << i;
    op->setOperand(numDpsInputs + i, *maybeReplacement);
  }

  return success();
}

class AMDAIEInsertCopyOpsPass
    : public impl::AMDAIEInsertCopyOpsBase<AMDAIEInsertCopyOpsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    AMDAIEDialect>();
  }

  AMDAIEInsertCopyOpsPass() = default;
  AMDAIEInsertCopyOpsPass(const AMDAIEInsertCopyOpsPass &pass){};
  AMDAIEInsertCopyOpsPass(const AMDAIEInsertCopyOpsOptions &options)
      : AMDAIEInsertCopyOpsBase(options) {}
  void runOnOperation() override;
};

void AMDAIEInsertCopyOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  mlir::FunctionOpInterface funcOp = getOperation();
  SmallVector<Operation *> targetOps;
  // Named contraction ops are included for the GEMV pipeline, which moves the
  // matmul operands with copies instead of packs (a pack would have turned the
  // named op into a generic before this point).
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (isa<linalg::SoftmaxOp, linalg::GenericOp,
            linalg::ContractionOpInterface>(op))
      targetOps.push_back(op);
  });
  // A contraction with a fused elementwise epilogue (the GEMV pipeline with a
  // fused ReLU, say) is staged the way pack-peel's data layout propagation
  // stages one: the epilogue reads the contraction's result where it is
  // computed (the per-core local tile) and it is the *epilogue's* result that
  // moves out level by level, as a standalone contraction's result would. So
  // the contraction's result gets no copies of its own, and the epilogue's
  // input none either; the epilogue only gets a buffer to write at each level
  // plus the usual result copy. Promoting it like a standalone op would
  // instead chain same-memory-space copies (L1 -> L1, which no DMA can do)
  // around it at every level, and copy the contraction result into a slice
  // guessed from the induction variables, since the forall inserts the
  // epilogue's result, not the contraction's.
  bool hasContraction =
      stageContractionEpilogue && llvm::any_of(targetOps, [](Operation *op) {
        return isa<linalg::ContractionOpInterface>(op);
      });
  auto isEpilogue = [&](Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    return hasContraction && linalgOp && isa<linalg::GenericOp>(op) &&
           isElementwiseWithMatmulProducer(linalgOp);
  };
  for (Operation *targetOp : targetOps) {
    if (isEpilogue(targetOp)) {
      Value initDest;
      if (useInitAsResultDest) {
        auto dstStyleOp = cast<DestinationStyleOpInterface>(targetOp);
        if (dstStyleOp.getNumDpsInits() != 1) {
          targetOp->emitError("expected an epilogue with one init");
          return signalPassFailure();
        }
        initDest = dstStyleOp.getDpsInits()[0];
      }
      if (failed(promoteInputs(rewriter, targetOp, /*onlyInits=*/true)) ||
          failed(promoteResults(rewriter, targetOp, initDest)))
        return signalPassFailure();
      continue;
    }
    bool feedsEpilogue =
        isa<linalg::ContractionOpInterface>(targetOp) &&
        llvm::any_of(targetOp->getUsers(), isEpilogue);
    // Captured before the init is replaced by its promoted copy.
    Value initDest;
    if (useInitAsResultDest) {
      auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(targetOp);
      if (!dstStyleOp || dstStyleOp.getNumDpsInits() != 1) {
        targetOp->emitError("expected a destination style op with one init");
        return signalPassFailure();
      }
      initDest = dstStyleOp.getDpsInits()[0];
    }
    if (failed(promoteInputs(rewriter, targetOp))) return signalPassFailure();
    if (feedsEpilogue) continue;
    if (failed(promoteResults(rewriter, targetOp, initDest)))
      return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertCopyOpsPass(
    AMDAIEInsertCopyOpsOptions options) {
  return std::make_unique<AMDAIEInsertCopyOpsPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
