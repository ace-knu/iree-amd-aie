// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Iterators.h"
#define DEBUG_TYPE "iree-amdaie-fold-dma-waits"

namespace mlir::iree_compiler::AMDAIE {

namespace {

using DmaBdIdKey = std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>;
/// The BD IDs are all those a wait on the half DMA copy operation's token
/// covers, see `retrieveDmaBdIdPair`.
using DmaBdIdPair = std::pair<DmaBdIdKey, SmallVector<uint32_t>>;

/// Utility function to retrieve TileOp, ConnectionOp, and the BD IDs covered by
/// a wait on the token of a given half DMA copy operation. DMA operations on the
/// same channel run in order, so these are the BD IDs of the operation itself
/// and of the preceding half DMA copy operations on the same connection and
/// channel back to the previous one with a token: the rest of its BD chain and
/// any earlier chains that have no token of their own.
FailureOr<DmaBdIdPair> retrieveDmaBdIdPair(
    AMDAIE::NpuHalfDmaCpyNdOp &npuHalfDmaCpyNdOp) {
  // Retrieve the connection op.
  std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
      npuHalfDmaCpyNdOp.getConnectionOp();
  if (!maybeConnectionOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "expected to operate on an `amdaie.connection`";
  }
  AMDAIE::ConnectionOp connectionOp = maybeConnectionOp.value();

  // Retrieve the BD ID op.
  std::optional<AMDAIE::BdIdOp> maybeBdIdOp = npuHalfDmaCpyNdOp.getBdIdOp();
  if (!maybeBdIdOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "must have a BD ID op to lower to "
              "`amdaie.npu.write_bd`";
  }
  AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();
  uint32_t currBdIdVal = getConstantIndexOrAssert(bdIdOp.getValue());

  // Retrieve the tile op.
  AMDAIE::TileOp tileOp =
      dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
  if (!tileOp) {
    return bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
  }

  SmallVector<uint32_t> bdIdVals = {currBdIdVal};
  for (Operation *op = npuHalfDmaCpyNdOp->getPrevNode(); op;
       op = op->getPrevNode()) {
    auto prevOp = dyn_cast<AMDAIE::NpuHalfDmaCpyNdOp>(op);
    if (!prevOp || prevOp.getConnection() != connectionOp.getResult() ||
        prevOp.getChannel() != npuHalfDmaCpyNdOp.getChannel())
      continue;
    // Stop at the previous operation with a token, its wait covers the rest.
    if (prevOp->getNumResults() != 0) break;
    if (std::optional<AMDAIE::BdIdOp> maybePrevBdIdOp = prevOp.getBdIdOp())
      bdIdVals.push_back(getConstantIndexOrAssert(maybePrevBdIdOp->getValue()));
  }

  DmaBdIdKey currBdIdKey = {tileOp, connectionOp};
  return DmaBdIdPair{currBdIdKey, bdIdVals};
}

/// Utility function to check whether any of `bdIdVals` is already in use on
/// `tileOp` according to `dmaBdIdsMap`.
bool isAnyBdIdInUse(const DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap,
                    AMDAIE::TileOp tileOp, ArrayRef<uint32_t> bdIdVals) {
  return llvm::any_of(dmaBdIdsMap, [&](const auto &entry) {
    return entry.first.first == tileOp &&
           llvm::any_of(bdIdVals, [&](uint32_t bdIdVal) {
             return entry.second.contains(bdIdVal);
           });
  });
}

/// Utility function to record the BD ID that a half DMA copy operation
/// programs, if it has one. Folding a wait moves it past the operations
/// recorded since the wait that is kept, so a BD ID of the waited chain among
/// them would be reprogrammed while the chain may still be running.
void recordProgrammedBdId(
    AMDAIE::NpuHalfDmaCpyNdOp npuHalfDmaCpyNdOp,
    DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &programmedBdIdsMap) {
  std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
      npuHalfDmaCpyNdOp.getConnectionOp();
  std::optional<AMDAIE::BdIdOp> maybeBdIdOp = npuHalfDmaCpyNdOp.getBdIdOp();
  if (!maybeConnectionOp || !maybeBdIdOp) return;
  auto tileOp = dyn_cast_if_present<AMDAIE::TileOp>(
      maybeBdIdOp->getTile().getDefiningOp());
  if (!tileOp) return;
  programmedBdIdsMap[{tileOp, *maybeConnectionOp}].insert(
      getConstantIndexOrAssert(maybeBdIdOp->getValue()));
}

/// Utility function to erase the DMA wait operations in the queue, except for
/// the last one.
LogicalResult eraseQueueOperations(IRRewriter &rewriter,
                                   SmallVector<AMDAIE::NpuDmaWaitOp> &waitOps) {
  // Skip if there are less than two DMA wait operations in the queue.
  if (waitOps.size() < 2) return success();

  Operation *parentOp = waitOps.back()->getParentOp();
  // Do not modify the last wait op, it will be kept.
  waitOps.pop_back();

  for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) {
    if (waitOp->getParentOp() != parentOp) {
      return waitOp.emitError(
          "DMA operations to be queued must belong to the same scope");
    }
    // Erase the wait op.
    SmallVector<Value> asyncTokens(waitOp.getAsyncTokens());
    rewriter.eraseOp(waitOp);
    for (Value token : asyncTokens) {
      auto dmaOp =
          dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(token.getDefiningOp());
      if (!dmaOp)
        waitOp.emitError("expected to operate on an `amdaie.half_dma_cpy_nd`");
      if (dmaOp.use_empty()) {
        rewriter.setInsertionPoint(dmaOp);
        TypeRange resultTypeRange = TypeRange{};
        // Nullify the result to avoid issuing a token.
        rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
            dmaOp.getLoc(), resultTypeRange, dmaOp.getConnection(),
            dmaOp.getInput(), dmaOp.getMixedOffsets(), dmaOp.getMixedSizes(),
            dmaOp.getMixedStrides(), dmaOp.getBdId(), dmaOp.getChannel(),
            dmaOp.getNextBd(), dmaOp.getStartBd());
        rewriter.eraseOp(dmaOp);
      }
    }
  }
  return success();
}

/// Utility function to determine whether a DMA wait op can be folded into a
/// queue based on its half DMA copy operation.
/// Can't fold wait op if:
/// (1) the current operation is not in the same scope as the queue, or
/// (2) reaches the maximum queue size, or
/// (3) the queue is empty, or
/// (4) a BD ID of the current BD chain on the same tile already occurs in the
/// queue.
FailureOr<bool> canFoldByQueue(
    const AMDAIE::AMDAIEDeviceModel &deviceModel,
    const Operation *queueParentOp,
    const DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap,
    const DenseMap<DmaBdIdKey, uint32_t> &numQueuedChains,
    AMDAIE::NpuHalfDmaCpyNdOp currHalfDmaCpyNdOp, DmaBdIdPair currBdIdPair) {
  // Not in the same scope? Can't fold.
  if (currHalfDmaCpyNdOp->getParentOp() != queueParentOp) return false;

  // Reached the maximum queue size, or the queue is empty? Can't fold.
  // The queue holds one entry per BD chain, however many BDs it has.
  DmaBdIdKey currBdIdKey = currBdIdPair.first;
  uint32_t numChains = numQueuedChains.lookup(currBdIdKey);
  TileOp tileOp = currBdIdKey.first;
  uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
  uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
  uint32_t maxQueueSize = deviceModel.getDmaMaxQueueSize(col, row);
  if (numChains >= maxQueueSize || numChains == 0) return false;

  // Duplicate BD ID on the same tile? Can't fold.
  if (isAnyBdIdInUse(dmaBdIdsMap, tileOp, currBdIdPair.second)) return false;

  // Can fold.
  return true;
}

/// Traverses the control code in reverse, ensuring that for each connection,
/// only one DMA wait op is retained for every maximum queue size.
///
/// Example Output: assuming a maximum queue size of 4.
///   dma_cpy_nd(connection=0, bd_id=0)
///   %0 = dma_cpy_nd(connection=0, bd_id=1)
///   dma_wait(%0)
///   dma_cpy_nd(connection=0, bd_id=2)
///   dma_cpy_nd(connection=0, bd_id=3)
///   dma_cpy_nd(connection=0, bd_id=4)
///   %1 = dma_cpy_nd(connection=0, bd_id=5)
///   dma_wait(%1)
/// From the bottom up, for every four DMA copy operations, only one DMA wait
/// operation is retained.
///
/// Reverse traversal simplifies handling duplicate BD IDs, preventing
/// the need to revisit and modify earlier operations after processing later
/// ones.
LogicalResult foldDmaWaitsByQueue(const AMDAIE::AMDAIEDeviceModel &deviceModel,
                                  AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());
  SmallVector<SmallVector<AMDAIE::NpuDmaWaitOp>> waitOpQueues;
  DenseMap<DmaBdIdKey, DenseSet<uint32_t>> dmaBdIdsMap;
  DenseMap<DmaBdIdKey, uint32_t> numQueuedChains;
  DenseMap<DmaBdIdKey, DenseSet<uint32_t>> programmedBdIdsMap;

  auto updateWithCurrBdId =
      [&](bool canFold, DmaBdIdPair currBdIdPair,
          DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap) {
        DmaBdIdKey currBdIdKey = currBdIdPair.first;
        if (!canFold) {
          dmaBdIdsMap[currBdIdKey].clear();
          numQueuedChains[currBdIdKey] = 0;
        }
        dmaBdIdsMap[currBdIdKey].insert(currBdIdPair.second.begin(),
                                        currBdIdPair.second.end());
        ++numQueuedChains[currBdIdKey];
      };

  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](Operation *op) {
        if (auto waitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
          bool toQueue = true;
          Operation *queueParentOp =
              waitOpQueues.empty() ? waitOp->getParentOp()
                                   : waitOpQueues.back().front()->getParentOp();
          for (Value token : waitOp.getAsyncTokens()) {
            if (auto npuHalfDmaCpyNdOp =
                    dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                        token.getDefiningOp())) {
              // Retrieve the TileOp, ConnectionOp, and BD ID.
              FailureOr<DmaBdIdPair> currBdIdPair =
                  retrieveDmaBdIdPair(npuHalfDmaCpyNdOp);
              if (failed(currBdIdPair)) return WalkResult::interrupt();
              // Check if the current DMA wait op can be folded into the queue.
              FailureOr<bool> canFold = canFoldByQueue(
                  deviceModel, queueParentOp, dmaBdIdsMap, numQueuedChains,
                  npuHalfDmaCpyNdOp, *currBdIdPair);
              if (failed(canFold)) return WalkResult::interrupt();
              bool fold = *canFold && !isAnyBdIdInUse(programmedBdIdsMap,
                                                      currBdIdPair->first.first,
                                                      currBdIdPair->second);
              // Update the `dmaBdIdsMap`.
              updateWithCurrBdId(fold, *currBdIdPair, dmaBdIdsMap);
              toQueue &= fold;
            }
          }
          // Store all the queues, and modify later to avoid invalidating the
          // iterator.
          if (toQueue) {
            // Append the wait op to the last queue if it can be folded.
            waitOpQueues.back().push_back(waitOp);
          } else {
            // Create a new queue if the wait op cannot be folded.
            waitOpQueues.push_back({waitOp});
            programmedBdIdsMap.clear();
          }
        } else if (auto npuHalfDmaCpyNdOp =
                       dyn_cast<AMDAIE::NpuHalfDmaCpyNdOp>(op)) {
          recordProgrammedBdId(npuHalfDmaCpyNdOp, programmedBdIdsMap);
        } else if (auto npuBarrierOp = dyn_cast<AMDAIE::NpuBarrierOp>(op)) {
          // Clear all queues if a sync barrier is encountered.
          for (auto &[_, bdIds] : dmaBdIdsMap) bdIds.clear();
          numQueuedChains.clear();
          programmedBdIdsMap.clear();
        }
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  for (SmallVector<AMDAIE::NpuDmaWaitOp> &waitOps : waitOpQueues) {
    // Since the controlcode is traversed in reverse order, we need to
    // restore the original order of the DMA operations.
    std::reverse(waitOps.begin(), waitOps.end());
    if (failed(eraseQueueOperations(rewriter, waitOps))) return failure();
  }
  return success();
}

/// For each batch, combine the async tokens into a single NpuDmaWaitOp.
LogicalResult eraseBatchOperations(IRRewriter &rewriter,
                                   SmallVector<AMDAIE::NpuDmaWaitOp> &waitOps) {
  // Skip if there are less than two DMA wait operations.
  if (waitOps.size() < 2) return success();

  SmallVector<Value> asyncTokens;
  Operation *parentOp = waitOps[0]->getParentOp();
  for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) {
    if (waitOp->getParentOp() != parentOp) {
      return waitOp.emitError(
          "DMA operations to be batched must belong to the same scope");
    }
    asyncTokens.append(waitOp.getAsyncTokens().begin(),
                       waitOp.getAsyncTokens().end());
  }

  rewriter.setInsertionPointAfter(waitOps.back());
  rewriter.create<AMDAIE::NpuDmaWaitOp>(waitOps.back().getLoc(), asyncTokens);
  for (AMDAIE::NpuDmaWaitOp waitOp : waitOps) rewriter.eraseOp(waitOp);
  return success();
}

/// Utility function to determine if a DMA wait operation can be folded into a
/// a batch based on its half DMA copy operation.
/// Can't fold wait op if:
/// (1) the current operation is not in the same scope as the batch, or
/// (2) the current connection op already occurs in the batch, or
/// (3) the batch is empty, or
/// (4) a BD ID of the current BD chain on the same tile already occurs in the
/// batch.
FailureOr<bool> canFoldByBatch(
    const Operation *batchParentOp,
    const DenseSet<AMDAIE::ConnectionOp> &connectionOps,
    const DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap,
    AMDAIE::NpuHalfDmaCpyNdOp currHalfDmaCpyNdOp, DmaBdIdPair currBdIdPair) {
  // Not in the same scope? Can't fold.
  if (currHalfDmaCpyNdOp->getParentOp() != batchParentOp) return false;

  // Connection op already in the batch, or an empty batch? Can't fold.
  AMDAIE::ConnectionOp connectionOp = currBdIdPair.first.second;
  if (connectionOps.contains(connectionOp) || connectionOps.empty())
    return false;

  // Duplicate BD ID on the same tile? Can't fold.
  AMDAIE::TileOp tileOp = currBdIdPair.first.first;
  if (isAnyBdIdInUse(dmaBdIdsMap, tileOp, currBdIdPair.second)) return false;

  // Can fold.
  return true;
}

/// Traverses the control code in reverse, ensuring that only one DMA wait op is
/// retained for every batch of DMA copy operations.
///
/// Example Input:
///   %0 = dma_cpy_nd(connection0)
///   dma_wait(%0)
///   %1 = dma_cpy_nd(connection1)
///   %2 = dma_cpy_nd(connection2)
///   %3 = dma_cpy_nd(connection3)
///   dma_wait(%1)
///   dma_wait(%2)
///   dma_wait(%3)
/// Example Output:
///   %0 = dma_cpy_nd(connection0)
///   %1 = dma_cpy_nd(connection1)
///   %2 = dma_cpy_nd(connection2)
///   %3 = dma_cpy_nd(connection3)
///   dma_wait(%0, %1, %2, %3)
/// Reverse traversal simplifies handling duplicate connections, preventing
/// the need to revisit and modify earlier operations after processing later
/// ones.
LogicalResult foldDmaWaitsByBatch(AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());
  SmallVector<AMDAIE::NpuDmaWaitOp> waitOps;
  DenseSet<AMDAIE::ConnectionOp> connectionOps;
  DenseMap<DmaBdIdKey, DenseSet<uint32_t>> dmaBdIdsMap;
  DenseMap<DmaBdIdKey, DenseSet<uint32_t>> programmedBdIdsMap;

  auto updateWithCurrBdId =
      [&](bool canFold, DmaBdIdPair currBdIdPair,
          DenseSet<AMDAIE::ConnectionOp> &connectionOps,
          DenseMap<DmaBdIdKey, DenseSet<uint32_t>> &dmaBdIdsMap) {
        DmaBdIdKey currBdIdKey = currBdIdPair.first;
        if (!canFold) {
          // Clear the BD IDs for all the connections in the batch.
          for (auto &entry : dmaBdIdsMap) {
            ConnectionOp connectionOp = entry.first.second;
            DenseSet<uint32_t> &bdIds = entry.second;
            if (connectionOps.contains(connectionOp)) bdIds.clear();
          }
          connectionOps.clear();
        }
        connectionOps.insert(currBdIdKey.second);
        dmaBdIdsMap[currBdIdKey].insert(currBdIdPair.second.begin(),
                                        currBdIdPair.second.end());
      };

  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](Operation *op) {
        if (auto waitOp = dyn_cast<AMDAIE::NpuDmaWaitOp>(op)) {
          bool toBatch = true;
          Operation *batchParentOp = waitOps.empty()
                                         ? waitOp->getParentOp()
                                         : waitOps[0]->getParentOp();
          for (Value token : waitOp.getAsyncTokens()) {
            if (auto npuHalfDmaCpyNdOp =
                    dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
                        token.getDefiningOp())) {
              // Retrieve the TileOp, ConnectionOp, and BD ID.
              FailureOr<DmaBdIdPair> currBdIdPair =
                  retrieveDmaBdIdPair(npuHalfDmaCpyNdOp);
              if (failed(currBdIdPair)) return WalkResult::interrupt();
              // Check if the current DMA wait op can be folded into the batch.
              FailureOr<bool> canFold =
                  canFoldByBatch(batchParentOp, connectionOps, dmaBdIdsMap,
                                 npuHalfDmaCpyNdOp, *currBdIdPair);
              if (failed(canFold)) return WalkResult::interrupt();
              bool fold = *canFold && !isAnyBdIdInUse(programmedBdIdsMap,
                                                      currBdIdPair->first.first,
                                                      currBdIdPair->second);
              // Update the `connectionOps` and `dmaBdIdsMap`.
              updateWithCurrBdId(fold, *currBdIdPair, connectionOps,
                                 dmaBdIdsMap);
              toBatch &= fold;
            }
          }
          // Process the previous batch of wait ops, and start a new batch.
          if (!toBatch) {
            // Since the controlcode is traversed in reverse order, we need to
            // restore the original order of the DMA operations.
            std::reverse(waitOps.begin(), waitOps.end());
            if (failed(eraseBatchOperations(rewriter, waitOps)))
              return WalkResult::interrupt();
            waitOps.clear();
            programmedBdIdsMap.clear();
          }
          waitOps.push_back(waitOp);
        } else if (auto npuHalfDmaCpyNdOp =
                       dyn_cast<AMDAIE::NpuHalfDmaCpyNdOp>(op)) {
          recordProgrammedBdId(npuHalfDmaCpyNdOp, programmedBdIdsMap);
        } else if (auto npuBarrierOp = dyn_cast<AMDAIE::NpuBarrierOp>(op)) {
          // Clear all batches if a sync barrier is encountered.
          connectionOps.clear();
          for (auto &[_, bdIds] : dmaBdIdsMap) bdIds.clear();
          programmedBdIdsMap.clear();
        }
        return WalkResult::advance();
      });

  if (res.wasInterrupted()) return failure();
  // Process the remaining wait ops.
  std::reverse(waitOps.begin(), waitOps.end());
  if (failed(eraseBatchOperations(rewriter, waitOps))) return failure();
  return success();
}

class AMDAIEFoldDmaWaitsPass
    : public impl::AMDAIEFoldDmaWaitsBase<AMDAIEFoldDmaWaitsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEFoldDmaWaitsPass() = default;
  AMDAIEFoldDmaWaitsPass(const AMDAIEFoldDmaWaitsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFoldDmaWaitsPass::runOnOperation() {
  Operation *parentOp = getOperation();

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to fold DMA wait "
           "ops.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  WalkResult res = parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    AMDAIE::ControlCodeOp controlCodeOp = workgroupOp.getControlCode();
    if (failed(foldDmaWaitsByQueue(deviceModel, controlCodeOp))) {
      return WalkResult::interrupt();
    }
    if (failed(foldDmaWaitsByBatch(controlCodeOp))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFoldDmaWaitsPass() {
  return std::make_unique<AMDAIEFoldDmaWaitsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
