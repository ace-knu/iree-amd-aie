// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/Utils/LockIdGenerator.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-amdaie-obj-fifo-bufferization"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Whether the producers of `logicalObjFifo` write slices that tile its single
/// consumer's transfer contiguously and in increasing offset order: their base
/// offsets are `consumerOffset + i * sliceLength`. This mirrors the check in
/// `connectionToAIE` (AMDAIELowerToAIE.cpp) that guards splitting the
/// consumer's transfer into one BD per producer slice. Layouts that fail it,
/// like the interleaved slices of 4-level tiling, keep the shared lock pair.
static bool producersTileConsumerInOrder(
    AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjFifo,
    ArrayRef<CopyOpInterface> producers, CopyOpInterface consumer) {
  SmallVector<size_t> offsets;
  for (CopyOpInterface producer : producers) {
    auto connectionOp = dyn_cast<AMDAIE::ConnectionOp>(producer.getOperation());
    if (!connectionOp) return false;
    FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> dmaOp =
        connectionOp.getNpuCircularDmaCpyNdUser();
    if (failed(dmaOp)) return false;
    std::optional<size_t> offset = dmaOp->getTargetStaticBaseOffset();
    if (!offset) return false;
    offsets.push_back(offset.value());
  }
  llvm::sort(offsets);
  offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
  if (offsets.size() != producers.size()) return false;

  auto connectionOp = dyn_cast<AMDAIE::ConnectionOp>(consumer.getOperation());
  if (!connectionOp) return false;
  FailureOr<AMDAIE::NpuCircularDmaCpyNdOp> dmaOp =
      connectionOp.getNpuCircularDmaCpyNdUser();
  if (failed(dmaOp)) return false;
  std::optional<size_t> consumerOffset = dmaOp->getSourceStaticBaseOffset();
  if (!consumerOffset) return false;
  FailureOr<size_t> repetitionCount = getRepetitionCount(
      cast<LogicalObjFifoOpInterface>(logicalObjFifo.getOperation()));
  if (failed(repetitionCount)) return false;
  SmallVector<OpFoldResult> sizes = dmaOp->getSourceMixedSizes();
  SmallVector<OpFoldResult> strides = dmaOp->getSourceMixedStrides();
  if (failed(foldRepetitionCount(logicalObjFifo.getContext(), sizes, strides,
                                 repetitionCount.value())))
    return false;
  std::optional<SmallVector<int64_t>> staticSizes = getConstantIntValues(sizes);
  if (!staticSizes) return false;
  int64_t totalLength =
      std::accumulate(staticSizes->begin(), staticSizes->end(), int64_t{1},
                      std::multiplies<>());
  int64_t numProducers = static_cast<int64_t>(producers.size());
  if (totalLength % numProducers != 0) return false;
  size_t sliceLength = static_cast<size_t>(totalLength / numProducers);
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    if (offset != consumerOffset.value() + i * sliceLength) return false;
  }
  return true;
}

LogicalResult bufferize(AMDAIE::WorkgroupOp workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());

  // Get the device model.
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(workgroupOp);
  if (!device)
    return workgroupOp->emitOpError()
           << "No AMDAIEDevice found in the target attribute configuratione";
  AMDAIEDeviceModel deviceModel = AMDAIE::getDeviceModel(device.value());
  LockIdGenerator lockGenerator(deviceModel);

  // Convert `amdaie.logicalobjectfifo.from_memref` to
  // `amdaie.logicalobjectfifo.from_buffers`.
  WalkResult res = workgroupOp->walk(
      [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjFifo) {
        // Skip logical objectFifos on L3/external main memory, else collect
        // tiles.
        SmallVector<TileOp> tiles;
        for (Value tile : logicalObjFifo.getTiles()) {
          auto tileOp =
              dyn_cast_if_present<AMDAIE::TileOp>(tile.getDefiningOp());
          if (!tileOp) {
            logicalObjFifo.emitOpError() << "got non-TileOp tile";
            return WalkResult::interrupt();
          }
          int64_t col = getConstantIndexOrAssert(tileOp.getCol());
          int64_t row = getConstantIndexOrAssert(tileOp.getRow());
          if (deviceModel.isShimTile(col, row)) return WalkResult::advance();
          tiles.push_back(tileOp);
        }

        MemRefType memrefType = logicalObjFifo.getMemrefType();
        unsigned depth = logicalObjFifo.getDepth();

        // TODO(jornt): for now, for consistency with the MLIR-AIE backend
        // expectations, we generate the expected lock init values here based on
        // the number of producer and consumer DMA operations, operating on this
        // logical objectFifo. However, there are some choices being made and
        // assumptions between different passes on them, so we should move the
        // logic to assign lock init values downstream in the pipeline.
        SmallVector<CopyOpInterface> copyLikeConsumers =
            logicalObjFifo.getCopyLikeConsumers();
        SmallVector<CopyOpInterface> copyLikeProducers =
            logicalObjFifo.getCopyLikeProducers();
        if (copyLikeConsumers.size() > 1 && copyLikeProducers.size() > 1) {
          logicalObjFifo.emitOpError()
              << "has a multi-producer, multi-consumer DMA "
                 "pattern, which is currently not supported";
          return WalkResult::interrupt();
        }
        // Several independent DMA producers each writing a disjoint slice of
        // the same buffers, read back by a single DMA consumer, need one lock
        // pair per producer. A single pair cannot express what the consumer
        // waits for: `AcquireGreaterEqual(N)` only asserts that N releases
        // happened, and a counting semaphore carries no producer identity, so
        // N releases from a subset of the producers that have run ahead into
        // the next buffer satisfy it just as well as one release from each. The
        // consumer then ships a buffer whose remaining slices are still stale.
        // Scaling the init value by the producer count (the single-pair case
        // below) makes the total credit count right but does not recover that
        // identity.
        //
        // With one pair of `depth` credits per producer, each can only run
        // `depth` ahead against its own slice. The consumer side of this lives
        // in `createDMABlocks` (AMDAIELowerToAIE.cpp), which splits its
        // transfer into one BD per producer slice -- an AIE2 BD carries exactly
        // one acquire and one release lock, so that split is the only way to
        // express "wait for all N".
        //
        // The mirrored single-producer/multi-consumer case keeps the scaled
        // single pair; it has the same weakness, but is not exercised by the
        // workloads this was written for. So do producers whose slices don't
        // tile the consumer's transfer contiguously (see
        // `producersTileConsumerInOrder`).
        size_t numProducers = copyLikeProducers.size();
        size_t numConsumers = copyLikeConsumers.size();
        bool lockPairPerProducer =
            numProducers > 1 && numConsumers == 1 &&
            producersTileConsumerInOrder(logicalObjFifo, copyLikeProducers,
                                         copyLikeConsumers[0]);
        size_t numLockPairs = lockPairPerProducer ? numProducers : 1;
        int8_t consumerLockInitValue{0};
        int8_t producerLockInitValue =
            lockPairPerProducer
                ? depth
                : (numProducers >= numConsumers ? numProducers * depth
                                                : numConsumers * depth);

        SmallVector<Value> buffers;
        SmallVector<Value> producerLocks;
        SmallVector<Value> consumerLocks;
        for (AMDAIE::TileOp tileOp : tiles) {
          rewriter.setInsertionPointAfter(tileOp);
          for (unsigned i = 0; i < depth; i++) {
            auto bufferOp = rewriter.create<AMDAIE::BufferOp>(
                rewriter.getUnknownLoc(), memrefType, tileOp, nullptr, nullptr,
                nullptr);
            buffers.push_back(bufferOp.getResult());
          }

          // Every set of buffers needs one producer and one consumer lock per
          // synchronizing party (one pair in the common case).
          int64_t col = getConstantIndexOrAssert(tileOp.getCol());
          int64_t row = getConstantIndexOrAssert(tileOp.getRow());
          for (size_t pair = 0; pair < numLockPairs; ++pair) {
            std::optional<uint32_t> producerLock =
                lockGenerator.getAndAssignLockId(col, row);
            if (!producerLock) {
              logicalObjFifo.emitOpError()
                  << "could not find an available producer lock";
              return WalkResult::interrupt();
            }
            auto producerLockOp = rewriter.create<AMDAIE::LockOp>(
                rewriter.getUnknownLoc(), tileOp, producerLock.value(),
                rewriter.getI8IntegerAttr(producerLockInitValue));
            producerLocks.push_back(producerLockOp.getResult());

            std::optional<uint32_t> consumerLock =
                lockGenerator.getAndAssignLockId(col, row);
            if (!consumerLock) {
              logicalObjFifo.emitOpError()
                  << "could not find an available consumer lock";
              return WalkResult::interrupt();
            }
            auto consumerLockOp = rewriter.create<AMDAIE::LockOp>(
                rewriter.getUnknownLoc(), tileOp, consumerLock.value(),
                rewriter.getI8IntegerAttr(consumerLockInitValue));
            consumerLocks.push_back(consumerLockOp.getResult());
          }
        }

        rewriter.setInsertionPoint(logicalObjFifo);
        rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromBuffersOp>(
            logicalObjFifo, logicalObjFifo.getType(), buffers, producerLocks,
            consumerLocks);
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEObjFifoBufferizationPass
    : public impl::AMDAIEObjFifoBufferizationBase<
          AMDAIEObjFifoBufferizationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEObjFifoBufferizationPass::runOnOperation() {
  Operation *parentOp = getOperation();
  SmallVector<AMDAIE::WorkgroupOp> workgroupOps;
  parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    workgroupOps.push_back(workgroupOp);
  });
  for (AMDAIE::WorkgroupOp workgroupOp : workgroupOps) {
    if (failed(bufferize(workgroupOp))) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEObjFifoBufferizationPass() {
  return std::make_unique<AMDAIEObjFifoBufferizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
