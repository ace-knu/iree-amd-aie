// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-amdaie-assign-logical-objectfifo-depth"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEAssignLogicalObjectFifoDepthPass
    : public impl::AMDAIEAssignLogicalObjectFifoDepthBase<
          AMDAIEAssignLogicalObjectFifoDepthPass> {
 public:
  AMDAIEAssignLogicalObjectFifoDepthPass() = default;
  AMDAIEAssignLogicalObjectFifoDepthPass(
      const AMDAIEAssignLogicalObjectFifoDepthPass &pass){};
  AMDAIEAssignLogicalObjectFifoDepthPass(
      const AMDAIEAssignLogicalObjectFifoDepthOptions &options)
      : AMDAIEAssignLogicalObjectFifoDepthBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEAssignLogicalObjectFifoDepthPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());
  // Assign buffer depths based on provided options.
  WalkResult res = parentOp->walk(
      [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjectFifo) {
        uint8_t memSpace = logicalObjectFifo.getMemorySpaceAsUInt();
        uint8_t bufferDepth{0};
        if (memSpace == 0) {
          bufferDepth = l3BufferDepth;
        } else if (memSpace == 1) {
          // Debug-only escape hatch: does upstream (L2) buffering add to how far
          // a core can run ahead, or is the core's own L1 credit the only bound?
          static const char *l2Override = std::getenv("AMDAIE_L2_DEPTH");
          bufferDepth = l2Override ? std::atoi(l2Override) : l2BufferDepth;
        } else if (memSpace == 2) {
          // Debug-only escape hatch (not part of any fix): lets experiments vary
          // the L1 depth, which is what bounds how far a core can run ahead.
          // AMDAIE_L1_DEPTH moves both directions at once; the _IN_/_OUT_ pair
          // separates them.  A core's input fifo is the *target* of the DMA that
          // fills it; its output fifo is the *source* of the DMA that drains it.
          static const char *l1All = std::getenv("AMDAIE_L1_DEPTH");
          static const char *l1In = std::getenv("AMDAIE_L1_IN_DEPTH");
          static const char *l1Out = std::getenv("AMDAIE_L1_OUT_DEPTH");
          bufferDepth = l1All ? std::atoi(l1All) : l1BufferDepth;
          if (l1In || l1Out) {
            bool isTarget = false, isSource = false;
            for (Operation *user : logicalObjectFifo->getUsers()) {
              auto dmaOp = dyn_cast<AMDAIE::DmaCpyNdOp>(user);
              if (!dmaOp) continue;
              if (dmaOp.getTarget() == logicalObjectFifo.getResult())
                isTarget = true;
              if (dmaOp.getSource() == logicalObjectFifo.getResult())
                isSource = true;
            }
            if (isTarget && !isSource && l1In) bufferDepth = std::atoi(l1In);
            if (isSource && !isTarget && l1Out) bufferDepth = std::atoi(l1Out);
          }
        } else {
          return WalkResult::advance();
        }
        MemRefType elementType = logicalObjectFifo.getMemrefType();
        rewriter.setInsertionPoint(logicalObjectFifo);
        rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromMemrefOp>(
            logicalObjectFifo,
            LogicalObjectFifoType::get(elementType, bufferDepth),
            logicalObjectFifo.getMemref(), logicalObjectFifo.getTiles());
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignLogicalObjectFifoDepthPass(
    AMDAIEAssignLogicalObjectFifoDepthOptions options) {
  return std::make_unique<AMDAIEAssignLogicalObjectFifoDepthPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
