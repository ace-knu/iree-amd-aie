// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-insert-copy-ops"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Promote a value by allocating a new buffer or using the provided output
// buffer, then copying the value into it.
//
// `copyContents` says whether the new buffer has to start out holding `v`.
// A destination the consuming op overwrites in full does not: only the buffer
// itself is wanted there, and reading the old contents back in is a DMA that
// moves data nobody looks at. Worse, when the value comes from the dispatch's
// own output binding it puts an L3 operand on both sides of the L2 buffer,
// which leaves tile assignment with no tiled neighbour to take a placement
// from (`getUserTiles` in `AMDAIEAssignTiles`).
FailureOr<Value> promoteValue(IRRewriter &rewriter, Location loc, Value v,
                              Value copyDest = Value(),
                              bool copyContents = true) {
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

  if (!copyContents) return copyDest;

  auto copy = rewriter.create<linalg::CopyOp>(loc, v, copyDest);
  return copy.getResult(0);
}

/// Whether `op` reads the destination operand at `initIdx`, rather than only
/// writing it.
///
/// For a `linalg` op the answer is written in its body: the block argument
/// standing for that destination is used only if the computation reads the
/// value already there (an accumulator does, an elementwise result does not).
/// `linalg.softmax` has no body but writes its output in full. Anything else
/// is assumed to read, which just keeps the copy that was made before.
static bool readsInit(DestinationStyleOpInterface dstStyleOp,
                      unsigned initIdx) {
  if (isa<linalg::SoftmaxOp>(dstStyleOp.getOperation())) return false;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(dstStyleOp.getOperation());
  if (!linalgOp || !linalgOp.getBlock()) return true;
  OpOperand *initOperand = linalgOp.getDpsInitOperand(initIdx);
  return !linalgOp.getMatchingBlockArgument(initOperand).use_empty();
}

/// Whether `value` stays inside `op`'s own block, i.e. it is an edge between
/// two compute ops that the tiling already fused into the same tile.
///
/// Those edges must not be promoted. A dispatch is often a chain --
/// `dequantize -> softmax -> quantize` for a quantized softmax, and the same
/// shape recurs in normalization and activation-quantization graphs -- and
/// only its two ends actually cross the tile boundary. Copying the edges in
/// between would round-trip each intermediate through its own buffer and, worse,
/// hand the next tiling level a value that no longer comes from the chain.
static bool isInternalChainEdge(Value value, Operation *op) {
  // Tiling puts a `tensor.extract_slice` on the edge between two fused ops, so
  // the producer has to be looked for through any number of them.
  Operation *producer = value.getDefiningOp();
  while (auto sliceOp = dyn_cast_if_present<tensor::ExtractSliceOp>(producer))
    producer = sliceOp.getSource().getDefiningOp();
  return producer && producer->getBlock() == op->getBlock() &&
         isa<linalg::SoftmaxOp, linalg::GenericOp>(producer);
}

/// Whether every result of `op` is consumed by another compute op in the same
/// block, so nothing it produces leaves the tile here.
static bool valueStaysInBlock(Value value, Block *block, unsigned depth = 0) {
  if (value.use_empty()) return false;
  // Tiling stacks at most a couple of slices on an edge between fused ops.
  if (depth > 4) return false;
  for (Operation *user : value.getUsers()) {
    if (user->getBlock() != block) return false;
    if (isa<linalg::SoftmaxOp, linalg::GenericOp>(user)) continue;
    // Fusion leaves slices on the edge; look through them.
    if (isa<tensor::ExtractSliceOp>(user) &&
        valueStaysInBlock(user->getResult(0), block, depth + 1))
      continue;
    return false;
  }
  return true;
}

static bool resultsStayInBlock(Operation *op) {
  for (OpResult result : op->getResults()) {
    if (result.use_empty()) continue;
    if (!valueStaysInBlock(result, op->getBlock())) return false;
  }
  return true;
}

LogicalResult promoteResults(IRRewriter &rewriter, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);

  // Only a single output is supported.
  if (op->getNumResults() != 1)
    return op->emitError("expected a single output");

  // Only ranked tensor is supported.
  Value result = op->getResults()[0];
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType) return op->emitError("expected ranked tensor result");

  // An edge to the next op of the same tile needs no buffer of its own.
  if (resultsStayInBlock(op)) return success();

  // Handle case where the target op is inside scf.forall and we want to use
  // block arguments as copy destinations.
  //
  // Which block argument, though, has to be read off the op's own
  // `parallel_insert_slice` rather than assumed to be the first one: the loop
  // can carry several shared outputs (fusing an elementwise consumer into it
  // adds one), and an intermediate result is written to none of them. Picking
  // blindly would pair the result with a destination of a different shape or
  // element type -- a quantized softmax has an f32 softmax inside a loop whose
  // output is i8 -- so fall through to a freshly allocated destination when
  // there is no insertion for this result.
  Value outputBuffer;
  if (auto forallOp = op->getParentOfType<scf::ForallOp>()) {
    Value blockArg;
    for (Operation &terminatorOp :
         forallOp.getTerminator().getRegion().front()) {
      auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&terminatorOp);
      if (insertOp && insertOp.getSource() == result) {
        blockArg = insertOp.getDest();
        break;
      }
    }
    if (blockArg) {
      unsigned numIvs = forallOp.getInductionVars().size();
      // Create tensor.extract_slice on block arg.
      rewriter.setInsertionPoint(op);
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (Value iv : forallOp.getInductionVars()) offsets.push_back(iv);
      // Pad offset with zeros if iv size is smaller than the rank.
      for (unsigned i = numIvs; i < resultType.getRank(); ++i)
        offsets.push_back(rewriter.getIndexAttr(0));
      for (int64_t d : resultType.getShape())
        sizes.push_back(rewriter.getIndexAttr(d));
      strides.assign(resultType.getRank(), rewriter.getIndexAttr(1));

      outputBuffer = rewriter.create<tensor::ExtractSliceOp>(
          op->getLoc(), blockArg, offsets, sizes, strides);
    }
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

LogicalResult promoteInputs(IRRewriter &rewriter, Operation *op) {
  OpBuilder::InsertionGuard g(rewriter);
  auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dstStyleOp) return failure();

  Location loc = dstStyleOp.getLoc();
  unsigned numDpsInputs = dstStyleOp.getNumDpsInputs();

  // Promote the input operands.
  for (auto [i, operand] : llvm::enumerate(dstStyleOp.getDpsInputs())) {
    if (isInternalChainEdge(operand, op)) continue;
    rewriter.setInsertionPoint(op);
    FailureOr<Value> maybeReplacement = promoteValue(rewriter, loc, operand);
    if (failed(maybeReplacement))
      return dstStyleOp.emitError() << "failed to promote input " << i;
    op->setOperand(i, *maybeReplacement);
  }

  // Promote the init operands. A destination only needs a buffer of its own if
  // the value it holds leaves the tile: when the next op of the chain is right
  // here, giving it one would put a copy on an edge that never crosses a
  // boundary, and the next tiling level would then read that buffer instead of
  // re-tiling the chain.
  bool resultsLeaveBlock = !resultsStayInBlock(op);
  for (auto [i, operand] : llvm::enumerate(dstStyleOp.getDpsInits())) {
    if (!resultsLeaveBlock) break;
    rewriter.setInsertionPoint(op);
    FailureOr<Value> maybeReplacement =
        promoteValue(rewriter, loc, operand, /*copyDest=*/Value(),
                     /*copyContents=*/readsInit(dstStyleOp, i));
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
  void runOnOperation() override;
};

void AMDAIEInsertCopyOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  mlir::FunctionOpInterface funcOp = getOperation();
  SmallVector<Operation *> targetOps;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    if (isa<linalg::SoftmaxOp>(op) || isa<linalg::GenericOp>(op))
      targetOps.push_back(op);
  });
  // Tiling leaves the original ops behind, replacing their uses; with
  // canonicalization suppressed until bufferization those clones are still
  // here. They form a chain, so each one still has a user -- the next clone --
  // and only the last is trivially dead. Propagate deadness through the chain,
  // otherwise promoting them would build a whole untiled copy chain next to
  // the tiled one, at the untiled shape, which nothing downstream can pair
  // with the loop.
  DenseSet<Operation *> deadOps;
  bool changed = true;
  while (changed) {
    changed = false;
    funcOp->walk([&](Operation *op) {
      if (op->getNumResults() == 0 || deadOps.contains(op)) return;
      if (!isMemoryEffectFree(op)) return;
      bool allUsersDead = llvm::all_of(op->getResults(), [&](OpResult result) {
        return llvm::all_of(result.getUsers(), [&](Operation *user) {
          return deadOps.contains(user);
        });
      });
      if (!allUsersDead) return;
      deadOps.insert(op);
      changed = true;
    });
  }

  for (Operation *targetOp : targetOps) {
    if (deadOps.contains(targetOp)) continue;
    if (failed(promoteInputs(rewriter, targetOp))) return signalPassFailure();
    if (failed(promoteResults(rewriter, targetOp))) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertCopyOpsPass() {
  return std::make_unique<AMDAIEInsertCopyOpsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
