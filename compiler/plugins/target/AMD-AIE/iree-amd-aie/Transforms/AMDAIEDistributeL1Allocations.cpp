// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-distribute-l1-allocations"

namespace mlir::iree_compiler::AMDAIE {

using namespace mlir;

namespace {

/// Find all `scf.forall` ops that are mapped to gpu thread dimensions (as
/// opposed to gpu block dimensions etc) i.e. the ones which will be mapped to
/// cores.
DenseSet<scf::ForallOp> getCoreForallOps(ModuleOp moduleOp) {
  DenseSet<scf::ForallOp> coreForallOps;
  moduleOp.walk([&](scf::ForallOp forallOp) {
    std::optional<ArrayAttr> maybeMapping = forallOp.getMapping();
    if (!maybeMapping) return WalkResult::advance();
    SmallVector<Attribute> mapping = llvm::to_vector(maybeMapping->getValue());
    if (mapping.empty()) return WalkResult::advance();
    if (!isa<gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();
    coreForallOps.insert(forallOp);
    return WalkResult::advance();
  });
  return coreForallOps;
}

/// For a given alloc, the distributed type is fetched by looking at each of its
/// subview users. For each subview, we check if their users are within the
/// scf.forall(s) which are mapped to GPU thread IDs (i.e. will be mapped to
/// core). We return an empty memref type if :-
///   a. Any subview's user is NOT within the innermost scf.forall.
///   b. The result types of two subviews do not match.
MemRefType getDistributedType(memref::AllocOp alloc,
                              DenseSet<scf::ForallOp> &coreForallOps) {
  MemRefType type;
  for (Operation *allocUser : alloc->getUsers()) {
    if (auto subview = dyn_cast<memref::SubViewOp>(allocUser)) {
      for (Operation *subviewUser : subview->getUsers()) {
        auto parentOp = dyn_cast<scf::ForallOp>(subviewUser->getParentOp());
        if (!parentOp || !coreForallOps.contains(parentOp)) return {};
      }
      auto nextType = cast<MemRefType>(subview.getResult().getType());
      if (!type) {
        type = nextType;
      } else if (type != nextType) {
        // This is the case where there are 2+ subview ops which look like
        // they should be distributing, but they have different result types.
        // Bail.
        return {};
      }
    }
  }
  return type;
}

/// Create a copy of `toUpdate` with all values in `toRemove` replaced by
/// `replacement`.
template <typename Container>
SmallVector<Value> substitute(Container toUpdate,
                              const DenseSet<Value> &toRemove,
                              Value replacement) {
  SmallVector<Value> updated(toUpdate.begin(), toUpdate.end());
  for (Value &v : updated) {
    if (toRemove.contains(v)) v = replacement;
  }
  return updated;
}

/// Given a `linalg::PackOp` and, for each of its DEST dims, the affine
/// expression (in terms of some outer iteration domain) that dim's index is
/// drawn from, compute the corresponding expression for each of the pack's
/// SOURCE dims -- i.e. invert the pack's index transformation. An untiled
/// source dim just forwards its dest "outer" position's expression
/// (`outerDimsPerm`, or identity if absent, gives the mapping); a tiled
/// source dim recombines its outer (tile-count) and inner (tile-offset) dest
/// positions as `outer * tileSize + inner`.
SmallVector<AffineExpr> composeThroughPack(linalg::PackOp packOp,
                                           ArrayRef<AffineExpr> destExprs) {
  auto sourceType = cast<MemRefType>(packOp.getSource().getType());
  int64_t sourceRank = sourceType.getRank();
  ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();
  SmallVector<OpFoldResult> mixedTiles = packOp.getMixedTiles();
  SmallVector<int64_t> outerDimsPerm(packOp.getOuterDimsPerm());
  if (outerDimsPerm.empty())
    outerDimsPerm = llvm::to_vector(llvm::seq<int64_t>(0, sourceRank));
  SmallVector<int64_t> destOuterPosOfSourceDim(sourceRank, -1);
  for (auto [pos, srcDim] : llvm::enumerate(outerDimsPerm))
    destOuterPosOfSourceDim[srcDim] = pos;

  SmallVector<AffineExpr> sourceExprs(sourceRank);
  for (int64_t d = 0; d < sourceRank; ++d) {
    AffineExpr outerExpr = destExprs[destOuterPosOfSourceDim[d]];
    auto tiledPosIt = llvm::find(innerDimsPos, d);
    if (tiledPosIt == innerDimsPos.end()) {
      sourceExprs[d] = outerExpr;
      continue;
    }
    int64_t tileIdx = std::distance(innerDimsPos.begin(), tiledPosIt);
    std::optional<int64_t> tileSize = getConstantIntValue(mixedTiles[tileIdx]);
    assert(tileSize && "expected a static inner tile size");
    AffineExpr innerExpr = destExprs[sourceRank + tileIdx];
    sourceExprs[d] = outerExpr * (*tileSize) + innerExpr;
  }
  return sourceExprs;
}

/// Returns the destination-style writer of `value`, if it has exactly one.
/// Memrefs have their allocation as their SSA definition, so their actual
/// writer has to be found through the DPS init operand instead.
static Operation *getUniqueDpsWriter(Value value) {
  Operation *writer = nullptr;
  for (OpOperand &use : value.getUses()) {
    auto dstStyle = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
    if (!dstStyle || !dstStyle.isDpsInit(&use)) continue;
    if (writer && writer != use.getOwner()) return nullptr;
    writer = use.getOwner();
  }
  return writer;
}

/// If `value` is ultimately initialized by a constant fill through only
/// layout-preserving copies and packs, return that fill. A fill has the same
/// value in every element, so the intervening layout changes are irrelevant:
/// emitting the fill at the final packed destination is equivalent and avoids
/// materializing the otherwise dead memtile round trip.
static linalg::FillOp findConstantFillProducer(Value value) {
  DenseSet<Value> visited;
  while (visited.insert(value).second) {
    Operation *writer = getUniqueDpsWriter(value);
    if (!writer) return {};
    if (auto fillOp = dyn_cast<linalg::FillOp>(writer)) return fillOp;
    if (auto packOp = dyn_cast<linalg::PackOp>(writer)) {
      value = packOp.getSource();
      continue;
    }
    auto genericOp = dyn_cast<linalg::GenericOp>(writer);
    if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
        genericOp.getNumDpsInits() != 1 ||
        !genericOp.getRegion().hasOneBlock())
      return {};
    auto yieldOp = dyn_cast<linalg::YieldOp>(
        genericOp.getRegion().front().getTerminator());
    if (!yieldOp || yieldOp.getValues().size() != 1 ||
        yieldOp.getValues().front() !=
            genericOp.getRegion().front().getArgument(0))
      return {};
    value = genericOp.getDpsInputs()[0];
  }
  return {};
}

/// Rebuild, as a single `linalg.generic` writing directly into `finalDest`,
/// the producer chain that computed `topPackOp`'s result (i.e. what
/// `topPackOp` would have packed into `oldAlloc`) -- entirely outside any
/// per-thread `scf.forall`, from data that doesn't itself vary per-thread
/// (e.g. a broadcasted bias) -- skipping every `linalg.pack` in the chain,
/// including `topPackOp` itself.
///
/// This matters, not just simplifies: every intermediate buffer in this
/// chain is L1 (it only ever existed to feed a per-thread slice of an L1
/// accumulator), so a cloned `linalg.pack` between two of them would be
/// purely L1-to-L1. Once cloned into a per-thread `scf.forall`,
/// `AMDAIEConvertToDma` unconditionally rewrites every `linalg.pack` into a
/// DMA -- which `AMDAIEInsertCores` then rejects, since a core can't issue
/// an L1-to-L1 DMA to itself. Composing the packs' index algebra instead
/// (`composeThroughPack`) and emitting one plain `linalg.generic` avoids
/// ever creating that op.
///
/// Walks backward through any number of `linalg::PackOp` producers --
/// tracking, via `composeThroughPack`, the affine expression each producer's
/// own source dims correspond to in `finalDest`'s domain -- until it reaches
/// a non-pack producer, expected to be a `linalg::GenericOp` with exactly
/// one input read through a projected permutation (e.g. our broadcast bias,
/// see AMDAIEFoldBroadcastAddIntoDestPass), which is rebuilt directly
/// against `finalDest`.
void buildDirectComputationNarrowed(RewriterBase &rewriter,
                                    linalg::PackOp topPackOp,
                                    Value finalDest) {
  // transpose_b accumulators are zero-filled before the first pack, then
  // copied through memtile and packed again into L1. Preserve the fill but
  // make it directly at the final (two-stage) packed output instead. Besides
  // removing those unnecessary buffers and DMA candidates, this avoids
  // rebuilding a copy whose input has the undistributed shape while its
  // output is per-core.
  if (linalg::FillOp fillOp =
          findConstantFillProducer(topPackOp.getSource())) {
    rewriter.create<linalg::FillOp>(fillOp.getLoc(), fillOp.value(), finalDest);
    return;
  }

  auto finalType = cast<MemRefType>(finalDest.getType());
  MLIRContext *ctx = rewriter.getContext();
  int64_t finalRank = finalType.getRank();
  SmallVector<AffineExpr> destExprs;
  for (int64_t i = 0; i < finalRank; ++i)
    destExprs.push_back(getAffineDimExpr(i, ctx));
  destExprs = composeThroughPack(topPackOp, destExprs);
  Value cur = topPackOp.getSource();

  while (true) {
    // `cur` is a memref (buffer) SSA value: its own `getDefiningOp()` is
    // just the `memref.alloc()` that reserved the storage, not whatever
    // wrote the data into it (a memref-semantics op like
    // `linalg.pack`/`linalg.generic` has zero results -- it's found by its
    // *use* as a DPS init operand, not as a def).
    Operation *writer = nullptr;
    for (OpOperand &use : cur.getUses()) {
      auto dstStyle = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
      if (dstStyle && dstStyle.isDpsInit(&use)) {
        writer = use.getOwner();
        break;
      }
    }
    assert(writer && "expected `cur` to have a writer");
    if (auto packOp = dyn_cast<linalg::PackOp>(writer)) {
      destExprs = composeThroughPack(packOp, destExprs);
      cur = packOp.getSource();
      continue;
    }

    auto genericOp = cast<linalg::GenericOp>(writer);
    assert(genericOp.getNumDpsInputs() == 1 &&
          genericOp.getNumDpsInits() == 1 &&
          "expected a single-input, single-output broadcast generic");
    AffineMap origInputMap =
        genericOp.getMatchingIndexingMap(&genericOp->getOpOperand(0));
    SmallVector<AffineExpr> newInputExprs;
    for (AffineExpr e : origInputMap.getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(e);
      assert(dimExpr && "expected a plain dim-projection indexing map");
      newInputExprs.push_back(destExprs[dimExpr.getPosition()]);
    }
    AffineMap newInputMap = AffineMap::get(finalRank, 0, newInputExprs, ctx);
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(finalRank);
    SmallVector<utils::IteratorType> iterators(finalRank,
                                               utils::IteratorType::parallel);
    auto newGeneric = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), ValueRange{genericOp.getDpsInputs()[0]},
        ValueRange{finalDest}, ArrayRef<AffineMap>{newInputMap, identityMap},
        iterators);
    rewriter.cloneRegionBefore(genericOp.getRegion(), newGeneric.getRegion(),
                               newGeneric.getRegion().end());
    return;
  }
}

/// `packOp` is now provably dead: `buildDirectComputationNarrowed` has fully
/// replaced the computation it used to feed into `oldAlloc`, whose only
/// other users are `oldAlloc`'s own dealloc and (now redirected away, see
/// the `SubViewOp` case above) dead subviews. Erase it, and recurse into
/// whatever wrote its own source for as long as that's *also* a now-dead
/// `linalg::PackOp`. Unlike a dead `linalg.generic`/`linalg.fill` -- harmless
/// to leave for a later DCE pass, same as this pass already does for every
/// other dead value it creates -- a dead pack/unpack is NOT harmless to
/// leave here: `AMDAIEConvertToDma` unconditionally converts every
/// pack/unpack in the module into a DMA connection regardless of liveness,
/// and an orphaned one fails connection-building later with "no source
/// channel".
void eraseDeadPackChain(RewriterBase &rewriter, linalg::PackOp packOp) {
  Value source = packOp.getSource();
  rewriter.eraseOp(packOp);
  for (OpOperand &use : source.getUses()) {
    auto dstStyle = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
    if (dstStyle && dstStyle.isDpsInit(&use)) {
      if (auto sourcePackOp = dyn_cast<linalg::PackOp>(use.getOwner()))
        eraseDeadPackChain(rewriter, sourcePackOp);
      break;
    }
  }
}

/// Distribute local memory accesses through subviews by allocating a single,
/// smaller memory. This is ultimately needed because cores can't operate on
/// one shared L1 memory.
LogicalResult distributeLocalMemory(ModuleOp moduleOp) {
  DenseSet<scf::ForallOp> coreForallOps = getCoreForallOps(moduleOp);
  DenseSet<Value> indVars;
  for (scf::ForallOp forallOp : coreForallOps) {
    for (Value indVar : forallOp.getInductionVars()) indVars.insert(indVar);
  }
  IRRewriter rewriter(moduleOp.getContext());

  auto allocWalkResult = moduleOp->walk([&](memref::AllocOp oldAlloc)
                                            -> WalkResult {
    // Only consider local memory (L1).
    Attribute maybeMemorySpace = oldAlloc.getType().getMemorySpace();
    if (!maybeMemorySpace) return WalkResult::advance();
    auto memorySpace = cast<IntegerAttr>(maybeMemorySpace);
    if (memorySpace.getInt() != 2) return WalkResult::advance();

    // Don't try and distribute memory if the alloc is inside a scf.for op.
    if (auto scfForOp = oldAlloc->getParentOfType<scf::ForOp>())
      return WalkResult::advance();

    MemRefType memRefType = getDistributedType(oldAlloc, coreForallOps);

    // Failed to find a memref.subview that looks like it is distributing.
    // This doesn't mean that we can't distribute (for example there might be
    // no subviews at all), but this requires further work.
    if (!memRefType) return WalkResult::advance();

    ArrayRef<int64_t> newShape = memRefType.getShape();
    Type elementType = memRefType.getElementType();

    rewriter.setInsertionPoint(oldAlloc);
    MemRefType newAllocType = MemRefType::get(
        newShape, elementType, MemRefLayoutAttrInterface{}, memorySpace);
    auto newAlloc = rewriter.create<memref::AllocOp>(rewriter.getUnknownLoc(),
                                                     newAllocType);

    const SmallVector<Operation *> users(oldAlloc->user_begin(),
                                         oldAlloc->user_end());

    // Replace uses of the old alloc with the new alloc.
    for (Operation *user : users) {
      LogicalResult switchResult =
          llvm::TypeSwitch<Operation *, LogicalResult>(user)
              .Case<memref::SubViewOp>([&](memref::SubViewOp subviewOp) {
                rewriter.replaceAllUsesWith(subviewOp, newAlloc);
                return success();
              })
              .Case<vector::TransferReadOp>([&](vector::TransferReadOp readOp) {
                rewriter.setInsertionPoint(readOp);
                Value c0 =
                    rewriter.create<arith::ConstantIndexOp>(readOp.getLoc(), 0);
                SmallVector<Value> indices =
                    substitute(readOp.getIndices(), indVars, c0);
                rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
                    readOp, readOp.getType(), newAlloc, indices,
                    readOp.getPermutationMapAttr(), readOp.getPadding(),
                    readOp.getMask(), readOp.getInBoundsAttr());
                return success();
              })
              .Case<vector::TransferWriteOp>(
                  [&](vector::TransferWriteOp writeOp) {
                    rewriter.setInsertionPoint(writeOp);
                    Value c0 = rewriter.create<arith::ConstantIndexOp>(
                        writeOp.getLoc(), 0);
                    SmallVector<Value> indices =
                        substitute(writeOp.getIndices(), indVars, c0);
                    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
                        writeOp, writeOp.getVector(), newAlloc, indices,
                        writeOp.getPermutationMapAttr(), writeOp.getMask(),
                        writeOp.getInBoundsAttr());
                    return success();
                  })
              .Case<memref::ExtractStridedMetadataOp>(
                  [&](memref::ExtractStridedMetadataOp
                          extractStridedMetadataOp) {
                    rewriter
                        .replaceOpWithNewOp<memref::ExtractStridedMetadataOp>(
                            extractStridedMetadataOp, newAlloc);
                    return success();
                  })
              .Case<memref::DeallocOp>([&](memref::DeallocOp deallocOp) {
                rewriter.setInsertionPoint(deallocOp);
                rewriter.create<memref::DeallocOp>(rewriter.getUnknownLoc(),
                                                   newAlloc);
                return success();
              })
              .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
                // A plain structured op inserted by bufferization (e.g. an
                // identity-copy `linalg.generic` moving a fused elementwise
                // consumer's result out of local memory, or the `linalg.fill`
                // zero-initializing it beforehand) reads/writes `oldAlloc`
                // directly, with no intervening subview. Just point the
                // matching operand(s) at `newAlloc` instead -- same shape, no
                // index/offset adjustment needed.
                for (OpOperand &operand : linalgOp->getOpOperands()) {
                  if (operand.get() == oldAlloc.getResult())
                    linalgOp->setOperand(operand.getOperandNumber(), newAlloc);
                }
                return success();
              })
              .Case<linalg::PackOp, linalg::UnPackOp>(
                  [&](Operation *packLikeOp) -> LogicalResult {
                // Same reasoning as the `linalg::LinalgOp` case above --
                // `linalg.pack`/`linalg.unpack` don't implement that
                // interface, but a pack-peel dispatch's re-layout of a
                // fused-in accumulator (e.g. a broadcasted bias folded into
                // a contraction's dest, see AMDAIEFoldBroadcastAddIntoDest)
                // can reach this alloc directly the same way.
                auto packOp = dyn_cast<linalg::PackOp>(packLikeOp);
                bool oldAllocIsDest =
                    packOp && packOp.getDest() == oldAlloc.getResult();
                auto enclosingForall =
                    packLikeOp->getParentOfType<scf::ForallOp>();
                bool insideCoreForall =
                    enclosingForall && coreForallOps.contains(enclosingForall);

                if (oldAllocIsDest && !insideCoreForall) {
                  // This pack currently computes the *whole*, undistributed
                  // buffer once, outside any per-thread scope, then hands it
                  // off via a per-thread `memref.subview` (the SubViewOp case
                  // above) -- unlike the other, already-correctly-per-thread
                  // pack chains (e.g. X/Y), which slice first and pack
                  // second. A blind operand swap would break the pack's own
                  // source/dest size relationship, since only the dest would
                  // shrink. Instead, since the data doesn't actually depend
                  // on which thread reads it (true for a broadcasted-bias
                  // accumulator init), recompute it redundantly once per
                  // thread: find the per-thread subview reader that produced
                  // `newAlloc`'s type, then rebuild the whole producer chain
                  // (this pack and everything upstream of it) as a single
                  // `linalg.generic` writing directly into `newAlloc`
                  // (`buildDirectComputationNarrowed`) -- not by cloning the
                  // pack chain itself, since every buffer in it is L1 and a
                  // cloned L1-to-L1 pack would later get rejected as an
                  // unsupported DMA once it's inside a per-thread core. The
                  // original pack
                  // (and its own producer chain) is left where it is, now
                  // dead, for a later DCE pass to clean up -- same as this
                  // pass already does for `oldAlloc` itself in every case.
                  // Find via each subview's own enclosing region, not its
                  // use-list: if the SubViewOp case above already ran for
                  // this same `oldAlloc` (processing order within `users` is
                  // not guaranteed), it already redirected the subview's uses
                  // to `newAlloc` directly, leaving the subview itself with
                  // no remaining users to inspect. There can be more than one
                  // qualifying (subview, forall) pair -- e.g. one per unrolled
                  // K-tile chunk, all reading/accumulating into the same
                  // `newAlloc` in program order. Pick the *earliest* one: this
                  // clone must run before the first chunk accumulates, and
                  // must NOT also be redone before later chunks (which would
                  // reset their already-accumulated partial sums back to just
                  // bias).
                  SmallVector<std::pair<memref::SubViewOp, scf::ForallOp>>
                      candidates;
                  for (Operation *u : oldAlloc->getUsers()) {
                    auto sv = dyn_cast<memref::SubViewOp>(u);
                    if (!sv) continue;
                    // Compare shape/element type only, not the full type --
                    // `newAlloc.getType()` has an identity layout (it's a
                    // fresh alloc) while a distributing subview's type
                    // carries an explicit strided layout even when it
                    // describes the same, contiguous shape; `getDistributedType`
                    // (which chose `newAlloc`'s shape from this same subview)
                    // already discards that layout the same way.
                    auto svType = dyn_cast<MemRefType>(sv.getResult().getType());
                    if (!svType || svType.getShape() != newAlloc.getType().getShape() ||
                        svType.getElementType() !=
                            newAlloc.getType().getElementType())
                      continue;
                    auto parent = sv->getParentOfType<scf::ForallOp>();
                    if (parent && coreForallOps.contains(parent))
                      candidates.push_back({sv, parent});
                  }
                  if (candidates.empty()) {
                    return packLikeOp->emitOpError(
                        "writes a distributed L1 alloc from outside the "
                        "per-thread scf.forall, but no matching per-thread "
                        "subview reader was found to derive an offset from.");
                  }
                  auto earliest = llvm::min_element(
                      candidates, [](const auto &a, const auto &b) {
                        return a.second->isBeforeInBlock(b.second);
                      });
                  scf::ForallOp coreForallOp = earliest->second;
                  rewriter.setInsertionPointToStart(coreForallOp.getBody());
                  buildDirectComputationNarrowed(rewriter, packOp, newAlloc);
                  eraseDeadPackChain(rewriter, packOp);
                  return success();
                }

                for (OpOperand &operand : packLikeOp->getOpOperands()) {
                  if (operand.get() == oldAlloc.getResult())
                    packLikeOp->setOperand(operand.getOperandNumber(), newAlloc);
                }
                return success();
              })
              .Default([&](Operation *user) {
                return user->emitOpError(
                    "needs logic implemented for handling.");
              });

      if (failed(switchResult)) return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (allocWalkResult.wasInterrupted()) return failure();

  return success();
}

class AMDAIEDistributeL1AllocationsPass
    : public impl::AMDAIEDistributeL1AllocationsBase<
          AMDAIEDistributeL1AllocationsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDistributeL1AllocationsPass() = default;
  AMDAIEDistributeL1AllocationsPass(
      const AMDAIEDistributeL1AllocationsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDistributeL1AllocationsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  if (failed(distributeLocalMemory(moduleOp))) return signalPassFailure();
}
}  // namespace

std::unique_ptr<Pass> createAMDAIEDistributeL1AllocationsPass() {
  return std::make_unique<AMDAIEDistributeL1AllocationsPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
