// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stores the output tiles that the cores of a native conv dispatch write into
// one shared-memory (L2) buffer as contiguous slices in core order, so that
// AMDAIEObjFifoBufferization gives the buffer one lock pair per writing core.
//
// The conv tile pipeline spreads output channels over the cores of a column
// (see `ConvTiling`): each core writes an [oh, ow, ocTile] tile of the
// [n?, oh, ow, oc] block into the column's L2 output buffer, and one DMA writes
// the buffer out to L3. In the NHWC layout the tiles interleave (core r owns
// channels [r * ocTile, (r + 1) * ocTile) of every pixel), so they do not tile
// the write-out transfer contiguously, and the bufferization falls back to a
// single counting lock pair for all writers. That pair is racy: a core that has
// run ahead into the next buffer can supply the releases the write-out waits
// for, which then ships a buffer whose other tiles are still stale.
//
// The buffer keeps its type, but its elements are laid out as
// [n?, slice, oh, ow, ocTile] instead: a core's write addresses its own
// contiguous slice, and the write-out reads the buffer in that order and
// scatters it into the NHWC L3 tile with the same dims permuted. The slices
// then tile the write-out in order. Runs on the DMAs of functions on the `conv`
// tile pipeline, right after AMDAIEConvertToDma.

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-amdaie-conv-output-slices-in-order"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Returns true if `func` runs the `conv` tile pipeline.
static bool isOnConvPipeline(func::FuncOp func) {
  IREE::Codegen::TranslationInfoAttr info = getTranslationInfo(func);
  if (!info) return false;
  DictionaryAttr config = info.getConfiguration();
  auto name =
      config ? config.getAs<StringAttr>(kTilePipelineOverrideName) : nullptr;
  return name && name.getValue() == kConvPipelineName;
}

/// Returns the memref alloc behind logical objectFifo `lof`, or null.
static memref::AllocOp getAlloc(Value lof) {
  auto lofOp = lof.getDefiningOp<AMDAIE::LogicalObjectFifoFromMemrefOp>();
  return lofOp ? lofOp.getMemref().getDefiningOp<memref::AllocOp>()
               : memref::AllocOp();
}

/// Memory space of the memref behind logical objectFifo `lof` (0 for L3).
static std::optional<uint32_t> getMemSpace(Value lof) {
  auto lofOp = lof.getDefiningOp<AMDAIE::LogicalObjectFifoFromMemrefOp>();
  if (!lofOp) return std::nullopt;
  return cast<MemRefType>(lofOp.getMemref().getType()).getMemorySpaceAsInt();
}

static SmallVector<int64_t> rowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

static bool hasConstantValues(ArrayRef<OpFoldResult> values,
                              ArrayRef<int64_t> expected) {
  std::optional<SmallVector<int64_t>> c = getConstantIntValues(values);
  return c && ArrayRef<int64_t>(*c) == expected;
}

/// Rewrites the writes of `writer` (one core's [1?, oh, ow, ocTile] tile into
/// the L2 buffer of shape `shape`) and the read of `reader` (the whole buffer
/// to L3) to the [n?, slice, oh, ow, ocTile] layout. Returns failure, changing
/// nothing, if the DMAs do not have that form.
static LogicalResult putSlicesInOrder(IRRewriter &rewriter,
                                      AMDAIE::DmaCpyNdOp writer,
                                      AMDAIE::DmaCpyNdOp reader,
                                      ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  if (rank != 3 && rank != 4) return failure();
  int64_t lead = rank - 3;  // the chunk dim, if any
  SmallVector<int64_t> strides = rowMajorStrides(shape);

  // Writer: a static-size tile, row-major strides of the buffer, one chunk,
  // full spatial dims at offset 0, and part of the channels.
  std::optional<SmallVector<int64_t>> sizes =
      getConstantIntValues(writer.getTargetMixedSizes());
  SmallVector<OpFoldResult> offsets = writer.getTargetMixedOffsets();
  if (!sizes || static_cast<int64_t>(sizes->size()) != rank ||
      !hasConstantValues(writer.getTargetMixedStrides(), strides))
    return failure();
  int64_t ocTile = sizes->back(), oc = shape.back();
  if (ocTile >= oc || oc % ocTile != 0) return failure();
  if (lead && (*sizes)[0] != 1) return failure();
  for (int64_t d = lead; d < rank - 1; ++d)
    if ((*sizes)[d] != shape[d] || !isConstantIntValue(offsets[d], 0))
      return failure();

  // Reader: the whole buffer in row-major order, into an L3 tile of the same
  // shape.
  SmallVector<int64_t> zeros(rank, 0);
  if (!hasConstantValues(reader.getSourceMixedOffsets(), zeros) ||
      !hasConstantValues(reader.getSourceMixedSizes(), shape) ||
      !hasConstantValues(reader.getSourceMixedStrides(), strides) ||
      !hasConstantValues(reader.getTargetMixedSizes(), shape))
    return failure();
  std::optional<SmallVector<int64_t>> l3Strides =
      getConstantIntValues(reader.getTargetMixedStrides());
  if (!l3Strides) return failure();

  // New layout [n?, slice, oh, ow, ocTile]: `spatial` elements per channel,
  // `spatial * ocTile` per slice.
  int64_t spatial = shape[rank - 3] * shape[rank - 2];
  int64_t numSlices = oc / ocTile;
  SmallVector<int64_t> sliceShape = {shape[rank - 3], shape[rank - 2], ocTile};
  SmallVector<int64_t> sliceStrides = rowMajorStrides(sliceShape);
  auto attrs = [&](ArrayRef<int64_t> v) {
    return getAsIndexOpFoldResult(rewriter.getContext(), v);
  };
  Location loc = writer.getLoc();

  // Writer: slice number = channel offset / ocTile, so the channel offset
  // steps `spatial` elements per channel.
  SmallVector<OpFoldResult> wOffsets, wSizes, wStrides;
  if (lead) {
    wOffsets.push_back(offsets[0]);
    wSizes.push_back(rewriter.getIndexAttr(1));
    wStrides.push_back(rewriter.getIndexAttr(strides[0]));
  }
  wOffsets.push_back(offsets.back());
  wSizes.push_back(rewriter.getIndexAttr(1));
  wStrides.push_back(rewriter.getIndexAttr(spatial));
  llvm::append_range(wOffsets, attrs({0, 0, 0}));
  llvm::append_range(wSizes, attrs(sliceShape));
  llvm::append_range(wStrides, attrs(sliceStrides));

  // Reader: the slices in order on the L2 side, the same dims scattered into
  // the NHWC tile on the L3 side (the slice dim steps ocTile channels).
  SmallVector<OpFoldResult> l3Offsets = reader.getTargetMixedOffsets();
  SmallVector<int64_t> rSizes, rSrcStrides, rDstStrides;
  SmallVector<OpFoldResult> rDstOffsets;
  if (lead) {
    rSizes.push_back(shape[0]);
    rSrcStrides.push_back(strides[0]);
    rDstStrides.push_back((*l3Strides)[0]);
    rDstOffsets.push_back(l3Offsets[0]);
  }
  rSizes.push_back(numSlices);
  rSrcStrides.push_back(spatial * ocTile);
  rDstStrides.push_back(ocTile * l3Strides->back());
  rDstOffsets.push_back(rewriter.getIndexAttr(0));
  llvm::append_range(rSizes, sliceShape);
  llvm::append_range(rSrcStrides, sliceStrides);
  for (int64_t d = lead; d < rank; ++d) {
    rDstStrides.push_back((*l3Strides)[d]);
    rDstOffsets.push_back(l3Offsets[d]);
  }
  SmallVector<int64_t> rSrcOffsets(rSizes.size(), 0);

  rewriter.setInsertionPoint(writer);
  auto newWriter = AMDAIE::DmaCpyNdOp::create(
      rewriter, loc, writer.getTarget(), wOffsets, wSizes, wStrides,
      writer.getSource(), writer.getSourceMixedOffsets(),
      writer.getSourceMixedSizes(), writer.getSourceMixedStrides());
  rewriter.replaceOp(writer, newWriter);
  rewriter.setInsertionPoint(reader);
  auto newReader = AMDAIE::DmaCpyNdOp::create(
      rewriter, reader.getLoc(), reader.getTarget(), rDstOffsets,
      attrs(rSizes), attrs(rDstStrides), reader.getSource(),
      attrs(rSrcOffsets), attrs(rSizes), attrs(rSrcStrides));
  rewriter.replaceOp(reader, newReader);
  return success();
}

class AMDAIEConvOutputSlicesInOrderPass
    : public impl::AMDAIEConvOutputSlicesInOrderBase<
          AMDAIEConvOutputSlicesInOrderPass> {
 public:
  void runOnOperation() override;
};

void AMDAIEConvOutputSlicesInOrderPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!isOnConvPipeline(func)) return;
  IRRewriter rewriter(func.getContext());

  // The DMAs that write each L2 buffer from L1 and read it to L3. A buffer
  // with any other DMA (an L2 -> L1 read, an L3 -> L2 write) is not an output
  // join and is left alone.
  llvm::MapVector<Operation *, SmallVector<AMDAIE::DmaCpyNdOp>> writers,
      readers;
  llvm::SmallPtrSet<Operation *, 8> otherUses;
  func.walk([&](AMDAIE::DmaCpyNdOp dma) {
    memref::AllocOp target = getAlloc(dma.getTarget());
    memref::AllocOp source = getAlloc(dma.getSource());
    std::optional<uint32_t> targetSpace = getMemSpace(dma.getTarget());
    std::optional<uint32_t> sourceSpace = getMemSpace(dma.getSource());
    if (target && targetSpace == 1) {
      if (sourceSpace == 2)
        writers[target].push_back(dma);
      else
        otherUses.insert(target);
    }
    if (source && sourceSpace == 1) {
      if (targetSpace == 0)
        readers[source].push_back(dma);
      else
        otherUses.insert(source);
    }
  });
  for (auto &[alloc, ws] : writers) {
    auto it = readers.find(alloc);
    if (ws.size() != 1 || it == readers.end() || it->second.size() != 1 ||
        otherUses.contains(alloc))
      continue;
    auto type = cast<MemRefType>(cast<memref::AllocOp>(alloc).getType());
    if (!type.hasStaticShape()) continue;
    if (failed(putSlicesInOrder(rewriter, ws.front(), it->second.front(),
                                type.getShape())))
      LLVM_DEBUG(llvm::dbgs() << "left alone: " << *alloc << "\n");
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEConvOutputSlicesInOrderPass() {
  return std::make_unique<AMDAIEConvOutputSlicesInOrderPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
