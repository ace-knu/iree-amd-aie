// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-bufferize-to-allocation"

namespace mlir::iree_compiler::AMDAIE {

namespace {

static LogicalResult applyBufferizeToAllocation(RewriterBase &rewriter,
                                                Operation *op,
                                                Attribute memorySpace) {
  linalg::BufferizeToAllocationOptions options;
  options.memcpyOp =
      linalg::BufferizeToAllocationOptions::MemcpyOp::MaterializeInDestination;
  options.allocOp = linalg::BufferizeToAllocationOptions::AllocOp::MemrefAlloc;
  options.bufferizeDestinationOnly = true;
  options.emitDealloc = true;

  // Bufferize ops.
  Value buffer =
      linalg::bufferizeToAllocation(rewriter, options, op, memorySpace);
  if (!buffer) {
    LLVM_DEBUG(llvm::dbgs() << "----- failed to bufferize operation -----\n");
    return failure();
  }
  return success();
}

/// Utility to fetch input and output operands from the LinalgOp (matmul or
/// elementwise op). For matmul-elementwise special case, since one of the
/// elementwise op's input is the output of the matmul op and has already been
/// promoted, there is no need to promote such operand again.
static SmallVector<Value> getInputOutputOperands(
    DestinationStyleOpInterface &dstStyleOp) {
  SmallVector<Value> operands;
  for (Value operand : dstStyleOp->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType())) continue;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(dstStyleOp.getOperation())) {
      if (isElementwise(linalgOp) && isMatmulInDefChain(operand)) continue;
    }
    operands.push_back(operand);
  }
  return operands;
}

/// Utility to fetch input operands from the LinalgOp (matmul or elementwise
/// op). For matmul-elementwise special case, since one of the elementwise op's
/// input is the output of the matmul op and has already been promoted, there is
/// no need to promote such operand again.
static SmallVector<Value> getInputOperands(
    DestinationStyleOpInterface &dstStyleOp) {
  SmallVector<Value> operands;
  for (Value operand : dstStyleOp.getDpsInputs()) {
    if (!isa<RankedTensorType>(operand.getType())) continue;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(dstStyleOp.getOperation())) {
      if (isElementwise(linalgOp) && isMatmulInDefChain(operand)) continue;
    }
    operands.push_back(operand);
  }
  return operands;
}

/// Utility to fetch pack operands at a specified depth from the LinalgOp's
/// input operands.
static FailureOr<SmallVector<Value>> getPackOrCopyOperands(
    DestinationStyleOpInterface &dstStyleOp, uint32_t depthLevel) {
  SmallVector<Value> operands;
  for (auto input : llvm::enumerate(dstStyleOp.getDpsInputs())) {
    uint32_t currentLevel{0};
    Operation *currentOp = input.value().getDefiningOp();
    while (currentLevel < depthLevel && currentOp != nullptr) {
      if (dyn_cast<linalg::PackOp>(currentOp)) {
        currentLevel++;
        if (currentLevel == depthLevel) break;
      } else if (dyn_cast<linalg::CopyOp>(currentOp)) {
        currentLevel++;
        if (currentLevel == depthLevel) break;
      }
      currentOp = currentOp->getOperand(0).getDefiningOp();
    }
    // The defining op has to be a pack or a copy op, fail otherwise.
    if (!currentOp) {
      return dstStyleOp.emitOpError()
             << "operand #" << input.index()
             << " only has pack/copy ops to depth " << currentLevel
             << ", but request is for a depth " << depthLevel
             << " pack/copy op.";
    }
    // We only want to fetch the input operand of the pack op.
    operands.push_back(currentOp->getResult(0));
  }
  return operands;
}

// This function helps to fetch operands of either a LinalgOp or its defining
// ops, based on which operands the caller wants to bufferize via
// `bufferizeOperand` parameter.
/// Whether `operand` is an edge between two compute ops that tiling already
/// put in the same tile.
///
/// Such an edge must not get its own buffer. A dispatch is often a chain --
/// `dequantize -> softmax -> quantize` for a quantized softmax, and the same
/// shape recurs in normalization and activation-quantization graphs -- and only
/// its two ends actually cross the tile boundary. Promoting an edge in between
/// puts a copy there, and the next tiling level then reads that buffer instead
/// of re-tiling the chain, leaving the producer stranded at the untiled shape.
static bool isInternalChainEdge(Value operand, Operation *op) {
  Operation *producer = operand.getDefiningOp();
  // Tiling leaves a `tensor.extract_slice` on the edge between fused ops.
  while (auto sliceOp = dyn_cast_if_present<tensor::ExtractSliceOp>(producer))
    producer = sliceOp.getSource().getDefiningOp();
  if (!producer || producer->getBlock() != op->getBlock()) return false;
  return isa<linalg::SoftmaxOp>(producer) ||
         (isa<linalg::GenericOp>(producer) &&
          isElementwise(cast<linalg::LinalgOp>(producer)));
}

/// Whether every result of `op` is consumed by another compute op in the same
/// block, so nothing it produces leaves the tile.
///
/// The counterpart of `isInternalChainEdge` for the destination side. An
/// operand is reached either as an input, where the producer says whether the
/// edge is internal, or as a destination, where the *result* does: giving a
/// destination its own buffer is what puts a copy between this op and the next
/// one in the chain, and that copy is exactly what stops the next tiling level
/// from fusing them back together.
static bool valueStaysInBlock(Value value, Block *block, unsigned depth = 0) {
  if (value.use_empty()) return false;
  // Tiling stacks at most a couple of slices on an edge between fused ops.
  if (depth > 4) return false;
  for (Operation *user : value.getUsers()) {
    if (user->getBlock() != block) return false;
    if (isa<linalg::SoftmaxOp, linalg::GenericOp, IREE::LinalgExt::CustomOp>(
            user))
      continue;
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

/// Follow the chain `targetOp` belongs to and report the operands that really
/// cross the tile boundary: the first op's inputs and the last op's
/// destination.
///
/// This pass looks at one op's operands, which is right for a dispatch with a
/// single compute op. Once tiling fuses a chain -- `dequantize -> softmax ->
/// quantize` for a quantized softmax, and the same shape in normalization and
/// activation-quantization graphs -- the root's own operands are all internal,
/// and promoting them would put buffers on edges that never leave the tile
/// while leaving the chain's real boundary unbuffered.
static void collectChainBoundaryOperands(Operation *op, bool wantInputs,
                                         SmallVectorImpl<Value> &result,
                                         llvm::SmallPtrSetImpl<Operation *> &seen,
                                         unsigned depth = 0) {
  if (depth > 4 || !seen.insert(op).second) return;
  auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dstStyleOp) return;

  if (wantInputs) {
    for (Value operand : dstStyleOp.getDpsInputs()) {
      Operation *producer = operand.getDefiningOp();
      while (auto sliceOp = dyn_cast_if_present<tensor::ExtractSliceOp>(producer))
        producer = sliceOp.getSource().getDefiningOp();
      if (producer && producer->getBlock() == op->getBlock() &&
          isa<linalg::SoftmaxOp, linalg::GenericOp,
              IREE::LinalgExt::CustomOp>(producer)) {
        collectChainBoundaryOperands(producer, wantInputs, result, seen,
                                     depth + 1);
        continue;
      }
      result.push_back(operand);
    }
    return;
  }

  // Destination side: if what this op produces is read by the next op of the
  // chain, that op's destination is the one that leaves the tile.
  if (!resultsStayInBlock(op)) {
    for (Value init : dstStyleOp.getDpsInits()) result.push_back(init);
    return;
  }
  for (OpResult opResult : op->getResults()) {
    for (Operation *user : opResult.getUsers()) {
      Operation *consumer = user;
      while (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(consumer)) {
        if (!sliceOp.getResult().hasOneUse()) return;
        consumer = *sliceOp.getResult().getUsers().begin();
      }
      collectChainBoundaryOperands(consumer, wantInputs, result, seen,
                                   depth + 1);
    }
  }
}

static FailureOr<SmallVector<Value>> getOperandsToBufferize(
    BufferizeOperand bufferizeOperand, Operation *op, uint32_t inputDepth) {
  auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dstStyleOp) {
    llvm::errs() << "expected a destination style op\n";
    return failure();
  }
  switch (bufferizeOperand) {
    /// Create new allocations for Lhs, Rhs and Out.
    case BufferizeOperand::LinalgInputOutput:
      return getInputOutputOperands(dstStyleOp);
    /// Create new allocation only for Lhs, Rhs.
    case BufferizeOperand::LinalgInput:
      return getInputOperands(dstStyleOp);
    /// Create new allocations only for Out.
    case BufferizeOperand::LinalgOutput:
      return SmallVector<Value>(dstStyleOp.getDpsInits());
    /// Create new allocations for operands from the pack ops.
    case BufferizeOperand::PackOrCopyInput:
      return getPackOrCopyOperands(dstStyleOp, inputDepth);
    default:
      return failure();
  }
}

/// Utility to create and return AMDAIEMemSpaceAttr with a given integer
/// `memorySpace`.
static AMDAIEMemSpaceAttr getMemorySpaceAttr(RewriterBase &rewriter,
                                             int64_t memorySpace) {
  AMDAIEMemSpace memSpace = AMDAIEMemSpace::None;
  switch (memorySpace) {
    case 0:
      memSpace = AMDAIEMemSpace::Global;
      break;
    case 1:
      memSpace = AMDAIEMemSpace::Shared;
      break;
    case 2:
      memSpace = AMDAIEMemSpace::Local;
      break;
    default:
      assert(false && "incorrect memory space");
      break;
  }
  return AMDAIEMemSpaceAttr::get(rewriter.getContext(), memSpace);
}

class AMDAIEBufferizeToAllocationPass
    : public impl::AMDAIEBufferizeToAllocationBase<
          AMDAIEBufferizeToAllocationPass> {
 public:
  AMDAIEBufferizeToAllocationPass() = default;
  AMDAIEBufferizeToAllocationPass(const AMDAIEBufferizeToAllocationPass &pass) {
  }
  AMDAIEBufferizeToAllocationPass(
      const AMDAIEBufferizeToAllocationOptions &options)
      : AMDAIEBufferizeToAllocationBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEBufferizeToAllocationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();
  SmallVector<Operation *> targetOps;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    // A `custom_op` is promoted as a unit. Its body describes what one tile
    // computes, so the ops inside it are not separate promotion targets -- and
    // there are several of them, which would look like several target ops.
    if (op->getParentOfType<IREE::LinalgExt::CustomOp>())
      return WalkResult::advance();
    if (isa<IREE::LinalgExt::CustomOp>(op)) {
      targetOps.push_back(op);
      return WalkResult::advance();
    }
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Skip if the op is not elementwise/reduction/contraction/convolution.
      if (!isElementwise(linalgOp) && !isReductionOp(linalgOp) &&
          !linalg::isaContractionOpInterface(linalgOp) &&
          !linalg::isaConvolutionOpInterface(linalgOp)) {
        return WalkResult::advance();
      }
      // Skip FillOp and CopyOp.
      if (isa<linalg::FillOp, linalg::CopyOp>(linalgOp))
        return WalkResult::advance();
      // Skip an elementwise op already folded into a contraction/conv's own
      // accumulator init (e.g. a broadcasted bias, see
      // AMDAIEFoldBroadcastAddIntoDestPass): it isn't an independent
      // elementwise dispatch tail, and the contraction's own output-promotion
      // step (BufferizeOperand::LinalgOutput) already bufferizes it as a
      // whole, the same way it does a `linalg.fill`. Treating it as its own
      // target here would instead try to bufferize its raw, not-yet-lowered
      // input operands (e.g. a `dispatch.tensor.load`) directly, which fails.
      if (isElementwiseFeedingContractionDest(linalgOp))
        return WalkResult::advance();
      // Skip if the op's elementwise status doesn't match the bufferization
      // mode.
      if (isElementwise(linalgOp) != bufferizeElementwise)
        return WalkResult::advance();
      // Accept as a target op.
      targetOps.push_back(op);
    } else if (isa<linalg::SoftmaxOp>(op)) {
      // Always accept SoftmaxOp as a target op.
      targetOps.push_back(op);
    }
    return WalkResult::advance();
  });

  if (targetOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no linalg op -----\n");
    return;
  }

  if (targetOps.size() > 1) {
    llvm::errs() << "expected only one target op, found " << targetOps.size()
                 << " target ops\n";
    return signalPassFailure();
  }

  IRRewriter rewriter(context);
  for (Operation *targetOp : targetOps) {
    // Find the producer ops for the target op, and bufferizes them in new
    // allocations.
    FailureOr<SmallVector<Value>> operandsToBufferize =
        getOperandsToBufferize(bufferizeOperand, targetOp, inputDepth);
    if (failed(operandsToBufferize)) {
      targetOp->emitOpError("could not fetch operands to bufferize");
      return signalPassFailure();
    }
    // Replace the root's own operands with the chain's boundary ones, but only
    // when the root really is inside a fused chain: an operand of its own is
    // internal, or what it produces is read by the next op right here. An
    // ordinary single-op dispatch is left exactly as it was.
    auto rootDstStyleOp = dyn_cast<DestinationStyleOpInterface>(targetOp);
    bool isInsideChain =
        rootDstStyleOp &&
        (resultsStayInBlock(targetOp) ||
         llvm::any_of(rootDstStyleOp.getDpsInputs(), [&](Value operand) {
           return isInternalChainEdge(operand, targetOp);
         }));
    if (isInsideChain &&
        (bufferizeOperand == BufferizeOperand::LinalgInputOutput ||
         bufferizeOperand == BufferizeOperand::LinalgInput ||
         bufferizeOperand == BufferizeOperand::LinalgOutput)) {
      SmallVector<Value> boundary;
      llvm::SmallPtrSet<Operation *, 4> seen;
      if (bufferizeOperand != BufferizeOperand::LinalgOutput)
        collectChainBoundaryOperands(targetOp, /*wantInputs=*/true, boundary,
                                     seen);
      seen.clear();
      if (bufferizeOperand != BufferizeOperand::LinalgInput)
        collectChainBoundaryOperands(targetOp, /*wantInputs=*/false, boundary,
                                     seen);
      if (!boundary.empty()) *operandsToBufferize = boundary;
    }
    for (auto operand : *operandsToBufferize) {
      AMDAIEMemSpaceAttr memorySpaceAttr =
          getMemorySpaceAttr(rewriter, memorySpace);
      Operation *definingOp = operand.getDefiningOp();
      // `linalg::bufferizeToAllocation` requires the defining op to have
      // exactly one operand that aliases/writes its result (e.g. a
      // `linalg.fill`'s init, for a matmul's zero-initialized accumulator).
      // A bare `tensor.empty()` -- the usual init for a non-reduction
      // elementwise op's own output, which never needs a fill -- has none,
      // so it can't go through that path as-is. Give it one: a harmless
      // zero-fill (the elementwise op unconditionally overwrites every
      // element anyway) makes it structurally identical to the
      // already-supported case.
      if (auto emptyOp = dyn_cast<tensor::EmptyOp>(definingOp)) {
        rewriter.setInsertionPointAfter(emptyOp);
        auto tensorType = cast<RankedTensorType>(emptyOp.getType());
        Value zero = rewriter.create<arith::ConstantOp>(
            emptyOp.getLoc(), rewriter.getZeroAttr(tensorType.getElementType()));
        auto fillOp = rewriter.create<linalg::FillOp>(
            emptyOp.getLoc(), ValueRange{zero}, ValueRange{emptyOp.getResult()});
        rewriter.replaceAllUsesExcept(emptyOp.getResult(),
                                      fillOp.getResult(0), fillOp);
        definingOp = fillOp;
      }
      rewriter.setInsertionPointAfter(definingOp);
      if (failed(
              applyBufferizeToAllocation(rewriter, definingOp, memorySpaceAttr))) {
        targetOp->emitOpError("failed bufferizing to allocations");
        return signalPassFailure();
      }
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEBufferizeToAllocationPass(
    AMDAIEBufferizeToAllocationOptions options) {
  return std::make_unique<AMDAIEBufferizeToAllocationPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
