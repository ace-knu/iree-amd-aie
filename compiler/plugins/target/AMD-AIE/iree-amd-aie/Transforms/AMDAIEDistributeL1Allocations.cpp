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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
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

/// Finds the `scf.forall` (from `coreForallOps`) that consumes `alloc`
/// through a `memref.subview` and executes *earliest*, relative to `anchor`'s
/// block -- i.e. the one a whole-buffer initialization pack (see
/// `distributeWholeBufferInitPack`) must be hoisted into, since the buffer it
/// initializes has to hold the initial value before the *first* core compute
/// that accumulates into it, not a later one (multiple candidates arise from
/// loop peeling: the same logical reduction loop's prologue/steady-state/
/// epilogue bodies are separate `scf.forall`s, all consuming the same,
/// already-distributed `alloc`). Each candidate is walked up to whichever
/// ancestor shares `anchor`'s block (loop peeling nests some candidates, e.g.
/// the steady-state body, inside constructs `anchor` is a sibling of, so
/// they aren't directly comparable), then compared with `isBeforeInBlock`.
/// Returns null if no candidate is reachable that way.
scf::ForallOp findEarliestCoreForallOp(
    memref::AllocOp alloc, const DenseSet<scf::ForallOp> &coreForallOps,
    Operation *anchor) {
  Block *anchorBlock = anchor->getBlock();
  scf::ForallOp earliest;
  Operation *earliestAnchor = nullptr;
  for (Operation *user : alloc->getUsers()) {
    auto subview = dyn_cast<memref::SubViewOp>(user);
    if (!subview) continue;
    auto forallOp = subview->getParentOfType<scf::ForallOp>();
    if (!forallOp || !coreForallOps.contains(forallOp)) continue;
    Operation *candidateAnchor = forallOp;
    while (candidateAnchor && candidateAnchor->getBlock() != anchorBlock)
      candidateAnchor = candidateAnchor->getParentOp();
    if (!candidateAnchor) continue;
    if (!earliestAnchor || candidateAnchor->isBeforeInBlock(earliestAnchor)) {
      earliestAnchor = candidateAnchor;
      earliest = forallOp;
    }
  }
  return earliest;
}

/// Rewrites a `linalg.pack` that writes the *entire* (not-yet-distributed)
/// `oldAlloc` in one shot -- e.g. loading a real bias value into an L1
/// accumulator before any core's compute loop touches it, as opposed to
/// relying on the AIE MAC instruction's hardware-supported zero-start, which
/// needs no such write -- into the same per-core-pack idiom every other
/// operand already uses: hoisted into the earliest existing core
/// `scf.forall` that consumes `newAlloc` (found via
/// `findEarliestCoreForallOp`; *reused*, not a freshly created forall --
/// later lowering (tile assignment in particular) infers a buffer's physical
/// tile from which core computation touches it, so a standalone forall
/// containing only a pack and no compute isn't recognized the same way an
/// existing, already-working compute forall is), whose body slices
/// `packOp`'s *source* at that forall's own induction variables (mirroring
/// the leading "core grid" dims that distributing `oldAlloc` down to
/// `newAlloc` shrank) and packs that per-core slice directly into `newAlloc`
/// (already exactly one core's shape, so the destination needs no subview at
/// all). Only rewrites the specific shape this is known to be correct for --
/// see the checks below -- and leaves `packOp` untouched (returning failure)
/// for the caller to report as an unhandled case otherwise, rather than risk
/// silently emitting incorrect IR for a pack shape this hasn't been verified
/// against.
LogicalResult distributeWholeBufferInitPack(
    IRRewriter &rewriter, linalg::PackOp packOp, memref::AllocOp oldAlloc,
    memref::AllocOp newAlloc, const DenseSet<scf::ForallOp> &coreForallOps) {
  ArrayRef<int64_t> oldShape = oldAlloc.getType().getShape();
  ArrayRef<int64_t> newShape = newAlloc.getType().getShape();
  int64_t rank = oldShape.size();
  if (rank != static_cast<int64_t>(newShape.size())) return failure();

  // The leading run of dims the distribution actually shrank (to size 1) --
  // these are the "core grid" dims this pack's whole-buffer write must be
  // split over. Bail (leaving `packOp` for the caller to report) if that
  // shrinking isn't exactly a leading, size-1-everywhere prefix with an
  // otherwise-identical tail: a different shape here is a pack this rewrite
  // hasn't been verified to handle correctly.
  int64_t numLeadDims = 0;
  while (numLeadDims < rank && oldShape[numLeadDims] != newShape[numLeadDims])
    ++numLeadDims;
  if (numLeadDims == 0) return failure();
  for (int64_t i = 0; i < numLeadDims; ++i)
    if (newShape[i] != 1) return failure();
  for (int64_t i = numLeadDims; i < rank; ++i)
    if (oldShape[i] != newShape[i]) return failure();

  Value source = packOp.getSource();
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  if (!sourceType) return failure();
  int64_t sourceRank = sourceType.getRank();
  ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();
  ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();
  if (!outerDimsPerm.empty() &&
      static_cast<int64_t>(outerDimsPerm.size()) != sourceRank)
    return failure();

  // For each distributed (grid) dim, find the *source* dim that lands there:
  // `linalg.pack`'s own shape-inference puts dest's outer position `i` at
  // source's outer contribution `outerDimsPerm[i]` (identity if
  // `outerDimsPerm` is absent). It must be an untiled source dim (not in
  // `innerDimsPos`) -- a tiled dim's outer contribution is
  // `ceil(extent / tile)`, not the raw extent, so slicing the raw source dim
  // to size 1 would not correspond to slicing the packed dest dim to size 1.
  SmallVector<int64_t> gridSourceDims(numLeadDims);
  for (int64_t i = 0; i < numLeadDims; ++i) {
    int64_t srcDim = outerDimsPerm.empty() ? i : outerDimsPerm[i];
    if (srcDim < 0 || srcDim >= sourceRank) return failure();
    if (llvm::is_contained(innerDimsPos, srcDim)) return failure();
    if (sourceType.getShape()[srcDim] != oldShape[i]) return failure();
    gridSourceDims[i] = srcDim;
  }

  scf::ForallOp forallOp =
      findEarliestCoreForallOp(oldAlloc, coreForallOps, packOp);
  if (!forallOp) return failure();
  // The hoisted pack must run exactly once, after `packOp`'s position and
  // before any accumulation into `newAlloc`. That holds only if nothing
  // between `forallOp` and `packOp`'s block repeats it -- e.g. without K-loop
  // peeling the earliest compute forall is inside the reduction `scf.for`,
  // and packing the init there would overwrite the partial sums on every K
  // step, silently. Reject any enclosing loop `packOp` is not itself in, and
  // an anchor that runs before `packOp`.
  Operation *forallAnchor = forallOp;
  while (forallAnchor->getBlock() != packOp->getBlock()) {
    forallAnchor = forallAnchor->getParentOp();
    if (isa<LoopLikeOpInterface>(forallAnchor)) return failure();
  }
  if (!packOp->isBeforeInBlock(forallAnchor)) return failure();
  // `getInductionVars()` returns a SmallVector by value; a `ValueRange` bound
  // to it would dangle as soon as this statement ends.
  SmallVector<Value> ivs = forallOp.getInductionVars();
  if (static_cast<int64_t>(ivs.size()) != numLeadDims) return failure();

  Location loc = packOp.getLoc();
  rewriter.setInsertionPointToStart(forallOp.getBody());
  SmallVector<OpFoldResult> offsets(sourceRank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes(sourceRank);
  SmallVector<OpFoldResult> strides(sourceRank, rewriter.getIndexAttr(1));
  for (int64_t d = 0; d < sourceRank; ++d)
    sizes[d] = rewriter.getIndexAttr(sourceType.getShape()[d]);
  for (int64_t i = 0; i < numLeadDims; ++i) {
    offsets[gridSourceDims[i]] = ivs[i];
    sizes[gridSourceDims[i]] = rewriter.getIndexAttr(1);
  }
  auto sourceSlice =
      rewriter.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);

  // `linalg::PackOp::build` (the ODS-generated, tensor-oriented builder used
  // above via `innerDimsPos`/`outerDimsPerm`/etc.) unconditionally adds a
  // result typed `dest.getType()`, which the verifier rejects for a memref
  // dest (`results = (outs Optional<AnyRankedTensor>:$result)` -- a memref
  // pack must have *no* result at all, matching destination-passing-style
  // buffer semantics). Bufferization's own `PackOpInterface::bufferize`
  // (BufferizableOpInterfaceImpl.cpp) sidesteps this the same way: build with
  // an explicit empty `TypeRange` and the raw operand list, reusing the
  // original (already memref-correct) op's attributes -- `operandSegmentSizes`
  // included -- since only the source/dest *values* change here, not the
  // pack's shape parameters or operand arity.
  SmallVector<Value> newOperands;
  newOperands.push_back(sourceSlice.getResult());
  newOperands.push_back(newAlloc.getResult());
  if (Value pad = packOp.getPaddingValue()) newOperands.push_back(pad);
  llvm::append_range(newOperands, packOp.getInnerTiles());
  rewriter.create<linalg::PackOp>(loc, TypeRange{}, newOperands,
                                  packOp->getAttrs());

  rewriter.eraseOp(packOp);
  return success();
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

  // Collected up front rather than rewritten from inside the walk: a user
  // rewrite may erase ops (`distributeWholeBufferInitPack` erases the pack),
  // and erasing the op right after `oldAlloc` -- which the walk has already
  // captured as the next one to visit -- is a use-after-free.
  SmallVector<memref::AllocOp> allocs;
  moduleOp->walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });
  auto distributeAlloc = [&](memref::AllocOp oldAlloc) -> WalkResult {
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
              .Case<linalg::PackOp>([&](linalg::PackOp packOp) {
                // A `linalg.pack` writing the *entire* old buffer in one
                // shot (as opposed to per-core, via a subview of a
                // *different* buffer, which is how every other operand
                // pack already works -- see AMDAIEUtils/this pass's
                // helpers) shows up when the accumulator this L1 buffer
                // holds starts from a real external value (a bias) rather
                // than the AIE MAC's hardware-supported zero-start: that
                // real value has to be physically written into L1 before
                // any core's compute loop reads it.
                if (packOp.getDest() != oldAlloc.getResult()) {
                  packOp->emitOpError("needs logic implemented for handling.");
                  return failure();
                }
                if (failed(distributeWholeBufferInitPack(
                        rewriter, packOp, oldAlloc, newAlloc,
                        coreForallOps))) {
                  packOp->emitOpError("needs logic implemented for handling.");
                  return failure();
                }
                return success();
              })
              .Case<memref::DeallocOp>([&](memref::DeallocOp deallocOp) {
                rewriter.setInsertionPoint(deallocOp);
                rewriter.create<memref::DeallocOp>(rewriter.getUnknownLoc(),
                                                   newAlloc);
                return success();
              })
              .Default([&](Operation *user) {
                return user->emitOpError(
                    "needs logic implemented for handling.");
              });

      if (failed(switchResult)) return WalkResult::interrupt();
    }

    return WalkResult::advance();
  };
  for (memref::AllocOp oldAlloc : allocs)
    if (distributeAlloc(oldAlloc).wasInterrupted()) return failure();

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
