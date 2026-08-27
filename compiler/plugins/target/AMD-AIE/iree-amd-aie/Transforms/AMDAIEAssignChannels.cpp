// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "iree-amdaie-assign-channels"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// EXPERIMENTAL. A packet connection whose source or target logical
/// objectFifo spans more than one physical tile needs a hardware
/// packet-flow-style distribution (one source routed to N destination
/// channels, or vice versa) to reach every tile from a single connection.
/// This has been found -- both in this project's own compiled output and
/// independently reproduced with AMD's own mlir-aie/IRON toolchain -- to
/// hang real npu4 hardware, regardless of `repeat_count`. (See
/// docs/2026-08-27_matmul_bias_fusion_hang_root_cause_refined.md.) This is
/// distinct from the case where a *circuit* connection spans multiple tiles,
/// which is common (e.g. a real tiled operand shared identically by every
/// row of one column) and does not hang -- so only `Packet` connections are
/// split here, not every multi-tile connection.
///
/// Splits such a connection into fully independent single-tile connections
/// (and single-tile objectFifos on the multi-tile side), each still fed from
/// the same shared source/target on the other side. Verified on real
/// hardware to avoid the hang (same source doc, mlir-aie repro).
LogicalResult splitPacketConnectionAcrossTiles(
    IRRewriter &rewriter, AMDAIE::ConnectionOp connectionOp,
    AMDAIE::LogicalObjFifoOpInterface multiTileObjFifo) {
  Operation *objFifoOp = multiTileObjFifo.getOperation();
  Value objFifoResult = objFifoOp->getResult(0);
  SmallVector<Value> tiles = multiTileObjFifo.getTiles();

  // Find the `amdaie.core` a use belongs to, whether the use's owner is the
  // core op itself directly (e.g. as an `in`/`out` dependency operand on the
  // core op) or an op nested inside the core's body (e.g. a
  // `logicalobjectfifo.acquire`/`release`/`access`). Declared once here so
  // it's usable both for the transitive non-core chain below and inside the
  // per-tile loop.
  auto findEnclosingCore = [](Operation *user) -> AMDAIE::CoreOp {
    if (auto coreOp = dyn_cast<AMDAIE::CoreOp>(user)) return coreOp;
    return user->getParentOfType<AMDAIE::CoreOp>();
  };

  // Collect the full transitive chain of ops, starting from `connectionOp`,
  // that live *outside* any `amdaie.core` -- e.g. a control-code-level
  // `amdaie.npu.circular_dma_cpy_nd` that kicks off the DMA transfer this
  // connection describes, which isn't scoped to any one tile at all.  These
  // need to be cloned once per new per-tile connection too (not just
  // redirected), and the whole chain has to be followed since such an op's
  // own result can itself be consumed by further control-code-level ops.
  SmallVector<Operation *> nonCoreChain;
  {
    SmallVector<Operation *> worklist = {connectionOp.getOperation()};
    llvm::SmallPtrSet<Operation *, 8> seen;
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (findEnclosingCore(user)) continue;  // handled per-tile below
          if (!seen.insert(user).second) continue;
          nonCoreChain.push_back(user);
          worklist.push_back(user);
        }
      }
    }
  }

  for (Value tileVal : tiles) {
    auto tileOp = dyn_cast_if_present<AMDAIE::TileOp>(tileVal.getDefiningOp());
    if (!tileOp) return connectionOp.emitOpError() << "expected a tile op";
    std::optional<int64_t> column = getConstantIntValue(tileOp.getCol());
    std::optional<int64_t> row = getConstantIntValue(tileOp.getRow());
    if (!column || !row) {
      return connectionOp.emitOpError() << "tile has non-constant location";
    }

    // Clone the objectFifo with just this one tile. `replaceWithNewTiles`
    // erases the op it's called on and returns a *new* op -- the new
    // objectFifo must come from its return value, never from the
    // pre-replace clone (which becomes dangling the moment it runs).
    rewriter.setInsertionPoint(objFifoOp);
    IRMapping objFifoMapper;
    Operation *clonedObjFifoOp = rewriter.clone(*objFifoOp, objFifoMapper);
    auto clonedObjFifo =
        cast<AMDAIE::LogicalObjFifoOpInterface>(clonedObjFifoOp);
    FailureOr<AMDAIE::LogicalObjFifoOpInterface> maybeNewObjFifo =
        clonedObjFifo.replaceWithNewTiles(rewriter, {tileVal});
    if (failed(maybeNewObjFifo)) {
      return objFifoOp->emitOpError()
            << "could not assign a single split-off tile";
    }
    Value newObjFifoResult = maybeNewObjFifo->getOperation()->getResult(0);

    // Clone the connection, redirecting the multi-tile side to this tile's
    // new independent objectFifo. The shared side (source or target,
    // whichever wasn't the multi-tile one) is left as-is -- multiple
    // independent connections reading/writing the same shared logical
    // objectFifo is fine.
    rewriter.setInsertionPoint(connectionOp);
    IRMapping connMapper;
    connMapper.map(objFifoResult, newObjFifoResult);
    Operation *newConnOp =
        rewriter.clone(*connectionOp.getOperation(), connMapper);

    auto tileMatches = [&](AMDAIE::CoreOp coreOp) {
      AMDAIE::TileOp userTileOp = coreOp.getTileOp();
      return getConstantIntValue(userTileOp.getCol()) == column &&
             getConstantIntValue(userTileOp.getRow()) == row;
    };

    // The connection's own result (an async completion token) can be
    // consumed either directly by the `amdaie.core` op(s) it feeds (as an
    // `in`/`out` dependency operand on the core op itself) or by an
    // `acquire`/`release` op nested inside a core's body -- redirect exactly
    // this tile's core from the shared token to this new per-tile
    // connection's own token.
    for (auto [oldResult, newResult] :
         llvm::zip(connectionOp->getResults(), newConnOp->getResults())) {
      for (OpOperand &use : llvm::make_early_inc_range(oldResult.getUses())) {
        AMDAIE::CoreOp coreOp = findEnclosingCore(use.getOwner());
        if (coreOp && tileMatches(coreOp)) use.set(newResult);
      }
    }

    // Redirect any remaining direct uses of the shared objectFifo that live
    // inside this specific tile's `amdaie.core` (e.g. a
    // `logicalobjectfifo.access`) to the new per-tile copy.
    for (OpOperand &use :
         llvm::make_early_inc_range(objFifoResult.getUses())) {
      Operation *user = use.getOwner();
      if (user == connectionOp.getOperation()) continue;  // erased below
      AMDAIE::CoreOp coreOp = findEnclosingCore(user);
      if (coreOp && tileMatches(coreOp)) use.set(newObjFifoResult);
    }

    // Clone the transitive non-core chain (e.g. the control-code-level
    // `amdaie.npu.circular_dma_cpy_nd` that kicks off this connection's DMA
    // transfer, and anything further downstream of it) for this tile too,
    // redirecting it to reference this tile's new connection instead of the
    // shared one. `rewriter.clone` extends `chainMapper` itself as it goes,
    // so cloning in dependency order (the order `nonCoreChain` was
    // discovered in) correctly chains new-op-to-new-op references.
    IRMapping chainMapper;
    for (auto [oldRes, newRes] :
         llvm::zip(connectionOp->getResults(), newConnOp->getResults())) {
      chainMapper.map(oldRes, newRes);
    }
    chainMapper.map(objFifoResult, newObjFifoResult);
    for (Operation *chainOp : nonCoreChain) {
      rewriter.setInsertionPoint(chainOp);
      rewriter.clone(*chainOp, chainMapper);
    }
  }

  // The original shared connection, objectFifo, and non-core chain are now
  // dead -- every use was redirected/cloned to a per-tile copy above. Erase
  // the chain in reverse (a consumer before whatever it depends on).
  for (Operation *chainOp : llvm::reverse(nonCoreChain)) {
    rewriter.eraseOp(chainOp);
  }
  rewriter.eraseOp(connectionOp);
  if (!objFifoResult.use_empty()) {
    return objFifoOp->emitOpError()
          << "still has uses after splitting across tiles that were not "
             "inside an `amdaie.core` on one of its tiles, nor part of the "
             "connection's own non-core use chain";
  }
  rewriter.eraseOp(objFifoOp);
  return success();
}

/// EXPERIMENTAL. Finds every `Packet`-type connection whose source or target
/// spans more than one tile and splits it via
/// `splitPacketConnectionAcrossTiles`. Run before channel assignment, since
/// it changes which (and how many) connections exist.
LogicalResult splitMultiTilePacketConnections(AMDAIE::WorkgroupOp workgroupOp,
                                              IRRewriter &rewriter) {
  SmallVector<AMDAIE::ConnectionOp> packetConnections;
  workgroupOp->walk([&](AMDAIE::ConnectionOp op) {
    if (op.getConnectionType() == AMDAIE::ConnectionType::Packet) {
      packetConnections.push_back(op);
    }
  });
  for (AMDAIE::ConnectionOp connectionOp : packetConnections) {
    auto sourceObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getSource().getDefiningOp());
    auto targetObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getTarget().getDefiningOp());
    if (targetObjFifo && targetObjFifo.getTiles().size() > 1) {
      if (failed(splitPacketConnectionAcrossTiles(rewriter, connectionOp,
                                                  targetObjFifo))) {
        return failure();
      }
      continue;
    }
    if (sourceObjFifo && sourceObjFifo.getTiles().size() > 1) {
      if (failed(splitPacketConnectionAcrossTiles(rewriter, connectionOp,
                                                  sourceObjFifo))) {
        return failure();
      }
    }
  }
  return success();
}

/// Initializes channel generators for tiles by detecting DMA channels
/// previously assigned by other passes (e.g., for control packets) and
/// registering them to prevent conflicts.
LogicalResult initializeChannelsGenerators(
    AMDAIE::WorkgroupOp workgroupOp, const AMDAIEDeviceModel &deviceModel,
    DenseMap<Value, ChannelGenerator> &tileToGeneratorMap) {
  // Get the number of producer and consumer channels for each tile.
  WalkResult res = workgroupOp.walk([&](AMDAIE::TileOp tileOp) {
    uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
    uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
    AMDAIETileType tileType = deviceModel.getTileType(col, row);
    FailureOr<uint8_t> maybeNumDmaChannels =
        deviceModel.getDmaProp<uint8_t>(tileType, AMDAIEDmaProp::NumChannels);
    if (failed(maybeNumDmaChannels) || *maybeNumDmaChannels == 0) {
      tileOp.emitOpError() << "does not have any DMA channels";
      return WalkResult::interrupt();
    }
    tileToGeneratorMap[tileOp.getResult()] =
        ChannelGenerator(*maybeNumDmaChannels, *maybeNumDmaChannels);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();

  res = workgroupOp.walk([&](AMDAIE::ConnectionOp connectionOp) {
    ChannelAssignmentMode mode =
        (connectionOp.getConnectionType() == AMDAIE::ConnectionType::Packet)
            ? ChannelAssignmentMode::RoundRobinPacketFlow
            : ChannelAssignmentMode::FirstAvailableCircuitFlow;
    // Check source DMA channels previously assigned by other passes,
    // and register them in `ChannelGenerator` using `assignProducerDMAChannel`.
    for (Value source : connectionOp.getSourceChannels()) {
      auto channelOp = dyn_cast<AMDAIE::ChannelOp>(source.getDefiningOp());
      if (!channelOp) {
        connectionOp.emitOpError() << "expected a `amdaie.channel` op source";
        return WalkResult::interrupt();
      }
      if (channelOp.getPortType() == StrmSwPortType::DMA) {
        Value tile = channelOp.getTileOp().getResult();
        tileToGeneratorMap[tile].assignProducerDMAChannel(channelOp.getValue(),
                                                          mode);
      }
    }
    // Check target DMA channels previously assigned by other passes,
    // and register them in `ChannelGenerator` using `assignConsumerDMAChannel`.
    for (Value target : connectionOp.getTargetChannels()) {
      auto channelOp = dyn_cast<AMDAIE::ChannelOp>(target.getDefiningOp());
      if (!channelOp) {
        connectionOp.emitOpError() << "expected a `amdaie.channel` op target";
        return WalkResult::interrupt();
      }
      if (channelOp.getPortType() == StrmSwPortType::DMA) {
        Value tile = channelOp.getTileOp().getResult();
        tileToGeneratorMap[tile].assignConsumerDMAChannel(channelOp.getValue(),
                                                          mode);
      }
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

/// Assign channels to `amdaie.connection` ops.
LogicalResult assignChannels(AMDAIE::WorkgroupOp workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());

  // Get the device model.
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(workgroupOp);
  if (!device) {
    return workgroupOp->emitOpError()
           << "could not find an AMDAIEDevice attribute";
  }
  AMDAIEDeviceModel deviceModel = AMDAIE::getDeviceModel(device.value());
  // EXPERIMENTAL: split multi-tile packet connections before anything below
  // looks at connections/tiles, since this changes which connections exist.
  if (failed(splitMultiTilePacketConnections(workgroupOp, rewriter))) {
    return failure();
  }
  // Initialize channel generators for tiles.
  DenseMap<Value, ChannelGenerator> tileToGeneratorMap;
  if (failed(initializeChannelsGenerators(workgroupOp, deviceModel,
                                          tileToGeneratorMap))) {
    return failure();
  }
  // Get all `amdaie.connection` ops.
  SmallVector<AMDAIE::ConnectionOp> circuitConnections, packetConnections;
  workgroupOp->walk([&](AMDAIE::ConnectionOp op) {
    if (op.getConnectionType() == AMDAIE::ConnectionType::Packet) {
      packetConnections.push_back(op);
    } else {
      circuitConnections.push_back(op);
    }
  });
  SmallVector<AMDAIE::ConnectionOp> connectionOps;
  connectionOps.reserve(circuitConnections.size() + packetConnections.size());
  // Append circuit connections first, so that they are also assigned first.
  connectionOps.append(circuitConnections.begin(), circuitConnections.end());
  connectionOps.append(packetConnections.begin(), packetConnections.end());

  for (AMDAIE::ConnectionOp connectionOp : connectionOps) {
    auto sourceLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getSource().getDefiningOp());
    if (!sourceLogicalObjFifo) {
      return connectionOp.emitOpError()
             << "expected a `LogicalObjFifoOpInterface` source";
    }
    auto targetLogicalObjFifo =
        dyn_cast_if_present<AMDAIE::LogicalObjFifoOpInterface>(
            connectionOp.getTarget().getDefiningOp());
    if (!targetLogicalObjFifo) {
      return connectionOp.emitOpError()
             << "expected a `LogicalObjFifoOpInterface` target";
    }
    ChannelAssignmentMode mode =
        (connectionOp.getConnectionType() == AMDAIE::ConnectionType::Packet)
            ? ChannelAssignmentMode::RoundRobinPacketFlow
            : ChannelAssignmentMode::FirstAvailableCircuitFlow;
    rewriter.setInsertionPoint(connectionOp);
    SmallVector<Value> sourceChannels = connectionOp.getSourceChannels();
    // Assign source (producer) DMA channels if not already assigned.
    if (sourceChannels.empty()) {
      for (Value tile : sourceLogicalObjFifo.getTiles()) {
        assert(tileToGeneratorMap.contains(tile) &&
               "no channel generator found for tile");
        std::optional<uint8_t> maybeChannel =
            tileToGeneratorMap[tile].getAndAssignProducerDMAChannel(mode);
        if (!maybeChannel) {
          return connectionOp.emitOpError()
                 << "no producer DMA channel available";
        }
        auto channelOp = rewriter.create<AMDAIE::ChannelOp>(
            rewriter.getUnknownLoc(), tile, maybeChannel.value(),
            StrmSwPortType::DMA, AMDAIE::DMAChannelDir::MM2S);
        sourceChannels.push_back(channelOp.getResult());
      }
    }
    // Assign target (consumer) DMA channels if not already assigned.
    SmallVector<Value> targetChannels = connectionOp.getTargetChannels();
    if (targetChannels.empty()) {
      for (Value tile : targetLogicalObjFifo.getTiles()) {
        assert(tileToGeneratorMap.contains(tile) &&
               "no channel generator found for tile");
        std::optional<uint8_t> maybeChannel =
            tileToGeneratorMap[tile].getAndAssignConsumerDMAChannel(mode);
        if (!maybeChannel) {
          return connectionOp.emitOpError()
                 << "no consumer DMA channel available";
        }
        auto channelOp = rewriter.create<AMDAIE::ChannelOp>(
            rewriter.getUnknownLoc(), tile, maybeChannel.value(),
            StrmSwPortType::DMA, AMDAIE::DMAChannelDir::S2MM);
        targetChannels.push_back(channelOp.getResult());
      }
    }
    // Replace the `amdaie.connection` op with newly assigned `sourceChannels`
    // and `targetChannels`.
    rewriter.replaceOpWithNewOp<AMDAIE::ConnectionOp>(
        connectionOp, connectionOp.getTarget(), targetChannels,
        connectionOp.getSource(), sourceChannels,
        connectionOp.getConnectionTypeAttr(), /*flow*/ nullptr);
  }
  return success();
}

class AMDAIEAssignChannelsPass
    : public impl::AMDAIEAssignChannelsBase<AMDAIEAssignChannelsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEAssignChannelsPass::runOnOperation() {
  Operation *parentOp = getOperation();
  SmallVector<AMDAIE::WorkgroupOp> workgroupOps;
  parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    workgroupOps.push_back(workgroupOp);
  });
  for (AMDAIE::WorkgroupOp workgroupOp : workgroupOps) {
    if (failed(assignChannels(workgroupOp))) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignChannelsPass() {
  return std::make_unique<AMDAIEAssignChannelsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
