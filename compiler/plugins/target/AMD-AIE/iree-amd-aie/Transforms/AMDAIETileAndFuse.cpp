// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-tile-and-fuse"

namespace mlir::iree_compiler::AMDAIE {

namespace {

enum class GPUGroupType { Block, Thread };

/// Assign GPU dialect thread/block mapping attributes to tiled dimensions.
///
/// \param groupType The type of group that the attributes will have.
///
/// \param nbTiles The number of tiles in each dimension of the tiled operation
///                [(ub - lb) / step].
///
/// \param tilingInterfaceOp The TilingInterface operation that is being tiled.
///                          Required only to emit useful errors.
FailureOr<SmallVector<Attribute>> getGPUMappingAttributes(
    ArrayRef<int64_t> nbTiles, GPUGroupType groupType,
    TilingInterface tilingInterfaceOp) {
  MLIRContext *context = tilingInterfaceOp.getContext();

  const uint64_t nbDims = nbTiles.size();
  const uint32_t nbLoopCountsAboveOne = std::count_if(
      nbTiles.begin(), nbTiles.end(), [](int64_t t) { return t != 1; });

  // The mlir::gpu::MappingId enum supports 13 dimensions, see:
  // https://github.com/llvm/llvm-project/blob/main
  //   /mlir/include/mlir/Dialect/GPU/IR/GPUDeviceMappingAttr.td
  if (nbDims > mlir::gpu::getMaxEnumValForMappingId()) {
    return tilingInterfaceOp->emitOpError()
           << "has too many dimensions to tile, there are only "
           << mlir::gpu::getMaxEnumValForMappingId()
           << " dimensions available in the mlir::gpu dialect, but " << nbDims
           << " are required here.";
  }

  // A maximum of 2 dimensions can have 'loop count > 1' in the thread
  // dimensions. This is to target the 2-D AIE array.
  //
  // TODO(newling) if there are 3+ dimensions, they should probably be collapsed
  // into 2. I'm leaving this as a follow-up task. Basically, instead of
  //
  //   ```(i,j,k) in (2,3,5)```
  //
  // it should be
  //   ```(i,l) in (2,15)```
  //
  // with then
  //   j=l/5 and k=l%5.
  //
  // Once the above is implemented, we can safely remove the following check:
  if (nbLoopCountsAboveOne > 2 && groupType == GPUGroupType::Thread) {
    auto tileSizesStr = std::to_string(nbTiles[0]);
    for (unsigned i = 1; i < nbDims; ++i) {
      tileSizesStr += ", " + std::to_string(nbTiles[i]);
    }
    return tilingInterfaceOp->emitOpError()
           << "has requested tiling with loop counts [" << tileSizesStr
           << "]. Currently we only support tiling thread dimensions "
           << "with at most 2 dimensions with loop counts greater than 1, "
           << "there are " << nbLoopCountsAboveOne << " here.";
  }

  auto getMappingAttributeForDimension = [&](uint32_t i) -> Attribute {
    auto id = static_cast<gpu::MappingId>(i);
    if (groupType == GPUGroupType::Block)
      return gpu::GPUBlockMappingAttr::get(context, id);
    else if (groupType == GPUGroupType::Thread)
      return gpu::GPUThreadMappingAttr::get(context, id);
    else {
      assert(false && "Unhandled group type, must be thread or block.");
      return Attribute{nullptr};
    }
  };

  // Map an integer to an Attribute as follows:
  // 0 -> DimY
  // 1 -> DimX
  // 2 -> DimZ
  // 3 -> LinearDim0
  // 4 -> LinearDim1
  // etc.
  //
  // Note that 0 and 1 are effectively swapped, because for AIE we want to
  // map the first dimension to AIE array columns (or something like that).
  auto getAttribute = [&](uint32_t i) -> Attribute {
    if (i == 0)
      return getMappingAttributeForDimension(1);
    else if (i == 1)
      return getMappingAttributeForDimension(0);
    else
      return getMappingAttributeForDimension(i);
  };

  // We give priority to tiling with loop count > 1, so that they
  // preferentially get DimY and DimX.
  SmallVector<Attribute> mapping(nbDims, {});
  uint32_t nAttributes = 0;
  for (uint32_t i = 0; i < nbDims; ++i) {
    if (nbTiles[i] != 1) {
      mapping[i] = getAttribute(nAttributes);
      ++nAttributes;
    }
  }
  for (uint32_t i = 0; i < nbDims; ++i) {
    if (!mapping[i] && nbTiles[i] > 0) {
      mapping[i] = getAttribute(nAttributes);
      ++nAttributes;
    }
  }

  // Squeeze out the empty attributes (corresponding to '0's in nbTiles).
  SmallVector<Attribute> finalMapping;
  finalMapping.reserve(nbDims);
  for (Attribute attr : mapping) {
    if (attr) finalMapping.push_back(attr);
  }

  assert(finalMapping.size() == nbTiles.size() &&
         "There should be one mapping attribute per tiled dimension.");
  return finalMapping;
}

/// Utility function to check if any of the reduction dimension is being tiled.
static bool isTilingReductionDimension(TilingInterface consumerOp,
                                       SmallVector<int64_t> tileSizesVal) {
  SmallVector<utils::IteratorType> loopIteratorTypes =
      consumerOp.getLoopIteratorTypes();
  unsigned totalTileSizes = tileSizesVal.size();
  unsigned totalLoopIteratorTypes = loopIteratorTypes.size();
  // Assume the following cases for [parallel, parallel, reduction, parallel]
  // iter types :-
  // Case 1:
  //    tile_size = [8, 8] - this is essentially -> [8, 8, 0, 0], i.e. we tile
  //    both the parallel iter types and do not tile the last two iter type
  //    (reduction and parallel in this case).
  // Case 2:
  //    tile_size = [8, 0, 8] - this is essentially -> [8, 0, 8, 0], i.e. here
  //    we tile the first iter type (parallel), do not tile the second iter type
  //    (parallel), tile the third iter type (reduction) and do not tile the
  //    last iter type (parallel).
  // Case 3:
  //    tile_size = [0, 0, 8, 8] - here we only tile the last two iter types
  //    (reduction and parallel).
  if (totalTileSizes < totalLoopIteratorTypes) {
    tileSizesVal.append(totalLoopIteratorTypes - totalTileSizes, 0);
  }
  for (auto [tileSize, loopIteratorType] :
       llvm::zip(tileSizesVal, loopIteratorTypes)) {
    if (loopIteratorType == utils::IteratorType::reduction && tileSize != 0)
      return true;
  }
  return false;
}

static bool consumerToSkip(TilingInterface op) {
  if (isa<linalg::CopyOp>(op) || isa<linalg::PackOp>(op) ||
      isa<linalg::UnPackOp>(op))
    return true;
  return false;
}


/// Greedily fuse the elementwise consumers of `loops`' results into the
/// innermost loop, so that a dispatch shaped `elementwise -> root ->
/// elementwise` ends up with all three inside one tile.
///
/// Tiling on its own only pulls producers in: the root's consumer stays
/// outside at the original, untiled shape. Everything downstream then has to
/// reconcile a loop carrying one shape and element type with an op carrying
/// another -- which is how a quantized softmax (`dequantize -> softmax ->
/// quantize`) breaks, and the same shape recurs in normalization and in
/// activation-quantization graphs.
///
/// Only ops that are cheap to recompute per tile are taken: elementwise linalg
/// ops with a single use. Copies, packs and unpacks are left alone, matching
/// `consumerToSkip`.
static void fuseConsumersIntoLoops(RewriterBase &rewriter,
                                   MutableArrayRef<LoopLikeOpInterface> loops) {
  if (loops.empty()) return;

  // Bound the walk: each round fuses one consumer, and a dispatch has only a
  // handful.
  for (unsigned round = 0; round < 8; ++round) {
    // Re-read the innermost loop every round: fusing a consumer rebuilds the
    // loop with an extra result and replaces the old one.
    LoopLikeOpInterface innermost = loops.back();
    Operation *candidateSliceOp = nullptr;
    innermost->walk([&](Operation *op) {
      if (!isa<tensor::InsertSliceOp, tensor::ParallelInsertSliceOp>(op))
        return WalkResult::advance();
      // The value being written back is what a consumer would read.
      Value source = isa<tensor::InsertSliceOp>(op)
                         ? cast<tensor::InsertSliceOp>(op).getSource()
                         : cast<tensor::ParallelInsertSliceOp>(op).getSource();
      Operation *producer = source.getDefiningOp();
      if (!producer) return WalkResult::advance();
      candidateSliceOp = op;
      return WalkResult::interrupt();
    });
    if (!candidateSliceOp) return;

    // Check the loop result actually feeds a single fusable consumer before
    // disturbing the IR.
    bool fusable = false;
    for (OpResult result : innermost->getResults()) {
      if (!result.hasOneUse()) continue;
      Operation *consumer = *result.getUsers().begin();
      auto linalgOp = dyn_cast<linalg::LinalgOp>(consumer);
      if (!linalgOp || !isElementwise(linalgOp)) continue;
      if (auto tilingOp = dyn_cast<TilingInterface>(consumer))
        if (consumerToSkip(tilingOp)) continue;
      fusable = true;
      break;
    }
    if (!fusable) return;

    // Hand over the caller's own array, not a copy of the innermost loop:
    // `tileAndFuseConsumerOfSlices` replaces the loops it is given and writes
    // the new ones back through this reference. Passing a local copy would
    // leave the caller holding erased operations.
    std::optional<scf::SCFFuseConsumerOfSliceResult> fusedConsumer =
        scf::tileAndFuseConsumerOfSlices(rewriter, candidateSliceOp, loops);
    if (!fusedConsumer) return;
    fusedConsumer->origConsumerOperands.front()->getOwner()->erase();
  }
}

/// Whether nothing real reads `value`, treating slice ops that themselves go
/// nowhere as not reading it. Collects the ops to erase, deepest first.
static bool isDeadSliceChain(Value value,
                             llvm::SetVector<Operation *> &toErase,
                             unsigned depth = 0) {
  if (value.use_empty()) return true;
  // Tiling stacks at most a couple of slices on such an edge.
  if (depth > 4) return false;
  for (Operation *user : value.getUsers()) {
    if (!isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(user)) return false;
    if (!isDeadSliceChain(user->getResult(0), toErase, depth + 1)) return false;
    toErase.insert(user);
  }
  return true;
}

/// Free `iterArg` of its remaining uses, if they are all ones a dead loop
/// result can have, and report whether it worked.
///
/// After the write-back for a dead result is dropped, what is usually left is
/// the plumbing tiling built around it: an `insert_slice` nothing reads, and an
/// `extract_slice` serving as the root op's destination. A destination is
/// written in full by the op it belongs to, so a fresh `tensor.empty` does the
/// same job without tying the value to the loop. Any other use means the
/// argument is really live, and nothing is changed.
static bool detachDeadIterArg(RewriterBase &rewriter, BlockArgument iterArg) {
  llvm::SetVector<Operation *> deadInserts;
  SmallVector<tensor::ExtractSliceOp> destinations;
  for (Operation *user : iterArg.getUsers()) {
    if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(user)) {
      if (!isDeadSliceChain(insertOp.getResult(), deadInserts)) return false;
      deadInserts.insert(insertOp);
      continue;
    }
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!sliceOp) return false;
    // Only a destination operand may be replaced: a read would see the
    // initial value, which a fresh buffer does not have.
    for (OpOperand &use : sliceOp.getResult().getUses()) {
      auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
      if (!dstStyleOp || !dstStyleOp.isDpsInit(&use)) return false;
    }
    destinations.push_back(sliceOp);
  }

  OpBuilder::InsertionGuard guard(rewriter);
  for (Operation *deadOp : deadInserts) rewriter.eraseOp(deadOp);
  for (tensor::ExtractSliceOp sliceOp : destinations) {
    auto sliceType = cast<RankedTensorType>(sliceOp.getResult().getType());
    rewriter.setInsertionPoint(sliceOp);
    Value empty = rewriter.create<tensor::EmptyOp>(
        sliceOp.getLoc(), sliceType.getShape(), sliceType.getElementType());
    rewriter.replaceOp(sliceOp, empty);
  }
  return iterArg.use_empty();
}

/// Rebuild `forallOp` without the results nothing outside the loop reads.
///
/// Tiling gives the root its own loop result. Once a consumer is fused in, the
/// root's value is read inside the loop and its escape becomes redundant: no
/// one reads that result, and the only thing keeping it alive is the loop's own
/// interface. Left in place it keeps a `shared_out`, a
/// `tensor.parallel_insert_slice` and, once the copies and promotion go in, a
/// whole buffer for an intermediate that never leaves the tile -- which the
/// next tiling level then reads instead of re-tiling the chain, stranding the
/// producer at the untiled shape.
///
/// This is the general cleanup after `producer -> root -> consumer` fusion, not
/// specific to any one chain. `scf.forall`'s own canonicalization is
/// deliberately not reused for it: those patterns would also fold away a
/// single-iteration loop, and the block-level loop this pass creates for a
/// full-size tile is exactly that.
static void dropDeadForallResults(RewriterBase &rewriter,
                                  scf::ForallOp forallOp) {
  unsigned numOutputs = forallOp.getOutputs().size();
  if (numOutputs == 0) return;

  SmallVector<bool> isDead(numOutputs, false);
  unsigned numDead = 0;
  for (unsigned i = 0; i < numOutputs; ++i) {
    if (!forallOp.getResult(i).use_empty()) continue;
    if (!detachDeadIterArg(rewriter, forallOp.getRegionIterArgs()[i])) continue;
    isDead[i] = true;
    ++numDead;
  }
  if (numDead == 0) return;

  SmallVector<Value> liveOutputs;
  for (unsigned i = 0; i < numOutputs; ++i)
    if (!isDead[i]) liveOutputs.push_back(forallOp.getOutputs()[i]);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  auto newForallOp = rewriter.create<scf::ForallOp>(
      forallOp.getLoc(), forallOp.getMixedLowerBound(),
      forallOp.getMixedUpperBound(), forallOp.getMixedStep(), liveOutputs,
      forallOp.getMapping());

  // Move the old body across. The dead iteration arguments have no uses, so
  // any dominating value of the right type stands in for them; their own
  // initial values are the obvious choice.
  // Bind the range first: `getInductionVars()` returns by value, so calling
  // `begin()` and `end()` on it directly would iterate two different
  // temporaries.
  auto newInductionVars = newForallOp.getInductionVars();
  SmallVector<Value> replacements(newInductionVars.begin(),
                                  newInductionVars.end());
  unsigned liveIdx = 0;
  for (unsigned i = 0; i < numOutputs; ++i) {
    replacements.push_back(isDead[i]
                               ? forallOp.getOutputs()[i]
                               : newForallOp.getRegionIterArgs()[liveIdx++]);
  }
  rewriter.eraseOp(newForallOp.getTerminator());
  rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(), replacements);

  liveIdx = 0;
  for (unsigned i = 0; i < numOutputs; ++i) {
    if (isDead[i]) continue;
    rewriter.replaceAllUsesWith(forallOp.getResult(i),
                                newForallOp.getResult(liveIdx++));
  }
  rewriter.eraseOp(forallOp);
}

FailureOr<scf::SCFTileAndFuseResult> applyTileAndFuse(
    RewriterBase &rewriter, TilingInterface rootOp,
    DominanceInfo &dominanceInfo,
    scf::SCFTileAndFuseOptions &tileAndFuseOptions) {
  FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, rootOp,
                                                tileAndFuseOptions);
  if (failed(tileAndFuseResult)) {
    return rootOp.emitOpError("failed to tile and fuse with op as root");
  }

  for (auto [origVal, replacement] : tileAndFuseResult->replacements) {
    rewriter.replaceUsesWithIf(origVal, replacement, [&](OpOperand &use) {
      return !isa<tensor::DimOp>(use.getOwner());
    });
  }

  return tileAndFuseResult;
}

/// This pass starts with the last TilingInterface operation, tiles the op and
/// fuses its producers recursively. The `tilingLevel` must be specified. It
/// picks the `tilingLevel`-th list as tiling sizes from lowering_config.
class AMDAIETileAndFusePass
    : public impl::AMDAIETileAndFuseBase<AMDAIETileAndFusePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect>();
  }

  AMDAIETileAndFusePass() = default;
  AMDAIETileAndFusePass(const AMDAIETileAndFusePass &pass) {}
  AMDAIETileAndFusePass(const AMDAIETileAndFuseOptions &options)
      : AMDAIETileAndFuseBase(options) {}

  void runOnOperation() override;
};

/// Set the tiling attribute on `loopForAll`, based on the number of loop
/// iterations in each dimension.
static LogicalResult setGpuAttributeOnForall(
    GPUGroupType groupType, scf::ForallOp loopForAll,
    TilingInterface tilingInterfaceOp) {
  MLIRContext *context = tilingInterfaceOp.getContext();

  std::optional<SmallVector<OpFoldResult>> maybeUbs =
      loopForAll.getLoopUpperBounds();
  std::optional<SmallVector<OpFoldResult>> maybeLbs =
      loopForAll.getLoopLowerBounds();
  std::optional<SmallVector<OpFoldResult>> maybeSteps =
      loopForAll.getLoopSteps();

  if (!maybeUbs || !maybeLbs || !maybeSteps) {
    return tilingInterfaceOp->emitOpError(
        "after tiling does not have constant loop upper bounds / "
        "lower bounds / steps.");
  }

  const SmallVector<OpFoldResult> &ubs = maybeUbs.value();
  const SmallVector<OpFoldResult> &lbs = maybeLbs.value();
  const SmallVector<OpFoldResult> &steps = maybeSteps.value();

  SmallVector<int64_t> nbIters;
  for (uint32_t i = 0; i < ubs.size(); ++i) {
    // Try and determine a static loop count in dimension i.
    // Try and get constant values for ubs[i], lbs[i], steps[i]:
    auto maybeUb = getConstantIntValue(ubs[i]);
    auto maybeLb = getConstantIntValue(lbs[i]);
    auto maybeStep = getConstantIntValue(steps[i]);
    if (maybeUb && maybeLb && maybeStep) {
      int64_t ub = maybeUb.value();
      int64_t lb = maybeLb.value();
      int64_t step = maybeStep.value();
      int64_t cnt = (ub - lb) / step;
      nbIters.push_back(cnt);
    } else {
      nbIters.push_back(ShapedType::kDynamic);
    }
  }

  auto maybeMapping =
      getGPUMappingAttributes(nbIters, groupType, tilingInterfaceOp);
  if (failed(maybeMapping)) return failure();
  ArrayAttr mappingArrayAttr = ArrayAttr::get(context, maybeMapping.value());
  loopForAll.setMappingAttr(mappingArrayAttr);
  return success();
}

void AMDAIETileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  TilingInterface consumerOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface op) {
    // Find the next consumer op if it does not have loops OR it is from
    // the skip ops list which currently contains linalg.copy and linalg.unpack.
    if (op.getLoopIteratorTypes().empty() || consumerToSkip(op))
      return WalkResult::advance();

    // For matmul + elementwise dispatch, we use flag `tileElementwise` to
    // indicate whether we want to tile the elementwise op. If flag
    // `tileElementwise == false`, and the linalg op is an elementwise op, it
    // will advance to find the next target op for tiling.
    auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(op.getOperation());
    if (linalgOp && isElementwise(linalgOp) && !tileElementwise)
      return WalkResult::advance();

    consumerOp = op;
    return WalkResult::interrupt();
  });
  if (!consumerOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no consumer op -----\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "consumerOp: " << consumerOp << "\n");
  LLVM_DEBUG(llvm::dbgs() << "tilingLevel: " << tilingLevel << "\n");
  // TODO(avarma): Have a global CONSTANT defining tiling stages and the
  // tiling strategy.
  // If `consumerOp` has its own lowering config, we prefer using it.
  // Otherwise, fallback to find a lowering_config from other operations.
  SmallVector<int64_t> tileSizesVal;
  if (auto loweringConfig =
          getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(consumerOp)) {
    tileSizesVal = loweringConfig.getTileSizeVals(tilingLevel);
  } else {
    FailureOr<IREE::Codegen::LoweringConfigAttr> maybeLoweringConfig =
        getFirstLoweringConfig<IREE::Codegen::LoweringConfigAttr>(
            getComputeOps(funcOp));
    if (failed(maybeLoweringConfig)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "can't find lowering_config, skip TileAndFuse");
      return;
    }
    tileSizesVal = maybeLoweringConfig.value().getTileSizeVals(tilingLevel);
  }

  if (llvm::all_of(tileSizesVal, [&](int64_t size) { return size == 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, all zeros -----\n");
    return;
  }

  SmallVector<OpFoldResult> tileSizes =
      getAsIndexOpFoldResult(context, tileSizesVal);

  auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);

  if (!useSCFFor) {
    options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  }

  IRRewriter rewriter(context);
  DominanceInfo dominanceInfo(funcOp);
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(options);
  // We switch off fusion if any of the reduction dimension is being tiled. We
  // resort to the default fusion control function that eliminates certain ops
  // otherwise.
  if (isTilingReductionDimension(consumerOp, tileSizesVal)) {
    tileAndFuseOptions.setFusionControlFn(
        [&](tensor::ExtractSliceOp sliceOp, OpResult originalProducer,
            bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          // Fusion is disabled here in general to avoid duplicating a
          // reduction-dependent producer's computation across every
          // reduction-tile iteration. A `linalg.broadcast` feeding the
          // *destination*/accumulator operand is a safe, narrow exception:
          // its value never depends on the reduction dimension being tiled
          // here (e.g. a bias vector folded into a contraction's own
          // accumulator init, see `AMDAIEFoldBroadcastAddIntoDest`), so
          // fusing it costs nothing extra per tile. Just as importantly,
          // fusing it here lets each tile get its own per-tile slice of the
          // broadcast -- matching how a genuine `ins` operand (e.g. the
          // contraction's other input) already gets tiled -- instead of the
          // whole broadcast being materialized once as a single buffer
          // shared across every tile, which then has to be redistributed
          // to each tile's consumer after the fact via an actual multi-tile
          // DMA broadcast.
          if (isDestinationOperand &&
              isa<linalg::BroadcastOp>(originalProducer.getOwner())) {
            return scf::SCFTileAndFuseOptions::ControlFnResult{false};
          }
          return std::nullopt;
        });
  } else {
    tileAndFuseOptions.setFusionControlFn(
        [&](tensor::ExtractSliceOp sliceOp, OpResult originalProducer,
            bool isDestinationOperand)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          bool fusableOp =
              TypeSwitch<Operation *, bool>(originalProducer.getOwner())
                  // List ops that shouldnt be fused.
                  .Case<linalg::PackOp, tensor::PadOp, linalg::CopyOp,
                        memref::CopyOp>([](Operation *) { return false; })
                  // Fuse all Linalg ops (can be generalized later)
                  .Default([&](Operation *op) {
                    return op->getDialect() ==
                           context->getLoadedDialect<linalg::LinalgDialect>();
                  });
          if (!fusableOp) return std::nullopt;
          return scf::SCFTileAndFuseOptions::ControlFnResult{false};
        });
  }

  FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
      applyTileAndFuse(rewriter, consumerOp, dominanceInfo, tileAndFuseOptions);

  if (failed(tileAndFuseResult)) return signalPassFailure();

  // Kept across both steps below: consumer fusion replaces the tiled loops, so
  // the mapping has to be set on what it leaves behind, not on the loops tiling
  // originally produced.
  SmallVector<LoopLikeOpInterface> loops(tileAndFuseResult->loops.begin(),
                                         tileAndFuseResult->loops.end());
  if (fuseConsumers) fuseConsumersIntoLoops(rewriter, loops);

  // When tiling using scf.for we do not need to set any mapping.
  if (!useSCFFor) {
    // Currently only thread groups are used in lowering, blocks get unrolled
    // temporally. In theory we should be able to just not add any block group
    // dimensions to the outer scf.forall operation, but currently this results
    // in compilation failure. What happens is
    // 1) without any block group dimensions, the scf.forall operation can be
    //    be canonicalized away if the tile sizes are all 1 (small matmul, for
    //    example). Leaving only the inner thread scf.forall.
    // 2) certain passes expect an outer scf.forall operation, so if it is
    //    canonicalized away, the pass fails.
    // So for now we're keeping the block group dimension here, but should
    // be able to compile without any block group dimensions TODO(newling)

    if (loops.size() != 1) {
      consumerOp.emitOpError() << "expected exactly one scf.forall operation "
                                  "after tiling, but there are "
                               << loops.size() << '.';
      signalPassFailure();
    }
    LoopLikeOpInterface loop = loops[0];
    scf::ForallOp loopForAll = dyn_cast<scf::ForallOp>(loop.getOperation());
    if (!loopForAll) {
      loop.getOperation()->emitOpError(
          "expected to be an scf.forall operation.");
      signalPassFailure();
    }
    if (hardwareMapping == HardwareMapping::Core ||
        hardwareMapping == HardwareMapping::Block) {
      auto groupType = hardwareMapping == HardwareMapping::Core
                           ? GPUGroupType::Thread
                           : GPUGroupType::Block;
      if (failed(setGpuAttributeOnForall(groupType, loopForAll, consumerOp))) {
        return signalPassFailure();
      }
    }
  }
  // Clean up the loop interface that consumer fusion leaves behind: first drop
  // the write-back for any result nothing uses, then rebuild the loop without
  // that result at all. See `dropDeadForallResults`.
  if (fuseConsumers) {
    funcOp->walk([&](scf::ForallOp forallOp) {
      SmallVector<Operation *> deadWriteBacks;
      for (Operation &terminatorOp :
           forallOp.getTerminator().getRegion().front()) {
        auto insertOp = dyn_cast<tensor::ParallelInsertSliceOp>(&terminatorOp);
        if (!insertOp) continue;
        auto blockArg = dyn_cast<BlockArgument>(insertOp.getDest());
        if (!blockArg) continue;
        unsigned outputIdx =
            blockArg.getArgNumber() - forallOp.getInductionVars().size();
        if (outputIdx >= forallOp.getResults().size()) continue;
        if (!forallOp.getResult(outputIdx).use_empty()) continue;
        // The value has to stay live inside the loop, otherwise this is not a
        // redundant escape but the loop's only writer.
        Value source = insertOp.getSource();
        bool usedInLoop = llvm::any_of(source.getUsers(), [&](Operation *user) {
          return user != insertOp.getOperation();
        });
        if (!usedInLoop) continue;
        deadWriteBacks.push_back(insertOp);
      }
      for (Operation *writeBack : deadWriteBacks) rewriter.eraseOp(writeBack);
    });
    // Collect first: the rebuild replaces the op being visited.
    SmallVector<scf::ForallOp> foralls;
    funcOp->walk([&](scf::ForallOp forallOp) { foralls.push_back(forallOp); });
    for (scf::ForallOp forallOp : foralls)
      dropDeadForallResults(rewriter, forallOp);
  }

}

}  // namespace

std::unique_ptr<Pass> createAMDAIETileAndFusePass(
    AMDAIETileAndFuseOptions options) {
  return std::make_unique<AMDAIETileAndFusePass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
