// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Preprocessing pass that moves NCHW max pooling to the channels-last (NHWC)
// layout: `linalg.pooling_nchw_max` becomes `linalg.pooling_nhwc_max` wrapped
// in `linalg.transpose` ops (input/init NCHW -> NHWC, result NHWC -> NCHW).
//
// Why: `iree-preprocessing-convert-conv-to-channels-last` moves convolutions
// to NHWC but skips pooling (pooling's channel dim is classified as a `depth`
// dim by `inferConvolutionDims`, which that pass rejects). In a conv -> pool ->
// conv chain this leaves the pooling NCHW between two NHWC convs, so a
// transpose is materialised on each side of every pooling (in VGG-16: ten
// transpose dispatches around five pools, all on the host). With the pooling
// also NHWC, the transposes this pass inserts meet the conv-side ones and
// cancel in the upstream transpose propagation / canonicalization, and the
// remaining pooling reads C contiguous elements per window tap, which is the
// layout the AIE pooling pipeline (later stage) needs for vectorized `max`.
//
// Runs in the preprocessing phase, before named ops are generalized, so it
// matches the named `linalg.pooling_nchw_max` only. Gated by the target option
// `--iree-amdaie-pooling-channels-last` (default off) until validated.
//
// With `channel-multiple` > 1 the NHWC pooling's channel dim is padded up to a
// multiple of it (`tensor.pad` on the input, and on the init unless it is a
// plain fill, which is just rebuilt at the padded size) and the padded channels
// are sliced off the result. The pool pipeline splits C over the array rows
// (`PoolCoreSplit::SpatialChannel`), so the multiple is the row count: every
// core row then gets the same channel count. Max pooling never mixes channels,
// so the pad value is irrelevant to the kept channels.
//
// With `halo-split-columns` > 0 an NHWC max pooling whose window overlaps
// along H (effective window != stride, e.g. 3x3/2) is pre-split on the host:
// the output rows are cut into `choosePoolColumns(OH, columns)` chunks, each
// chunk's input rows (its windows' bounding box, so neighbouring chunks share
// `window - stride` halo rows, which are duplicated) are gathered with
// `tensor.extract_slice` + `tensor.concat` onto a leading chunk dim, the
// pooling runs as a batched NHWC pooling over the chunks and the result is
// reshaped back. The pool pipeline then puts the chunk dim on array columns
// (its memtile split needs disjoint pieces, which a leading chunk dim gives
// for free), where OH itself could not be split for an overlapping window.
// Costs one host gather dispatch and `(window - stride) / (stride * t)` extra
// input bytes.

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-convert-pooling-to-channels-last"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// NCHW -> NHWC and its inverse, as `linalg.transpose` permutations.
constexpr std::array<int64_t, 4> kNchwToNhwc = {0, 2, 3, 1};
constexpr std::array<int64_t, 4> kNhwcToNchw = {0, 3, 1, 2};

/// Materialize `linalg.transpose(src, perm)` into a fresh `tensor.empty`.
static Value transposeTensor(RewriterBase &rewriter, Location loc, Value src,
                             ArrayRef<int64_t> perm) {
  auto srcType = cast<RankedTensorType>(src.getType());
  SmallVector<OpFoldResult> srcSizes = tensor::getMixedSizes(rewriter, loc, src);
  SmallVector<OpFoldResult> dstSizes;
  for (int64_t p : perm) dstSizes.push_back(srcSizes[p]);
  Value empty = tensor::EmptyOp::create(rewriter, loc, dstSizes,
                                        srcType.getElementType());
  return linalg::TransposeOp::create(rewriter, loc, src, empty, perm)
      .getResult()[0];
}

/// Pad the channel (last) dim of an NHWC tensor up to `cPad` with `padValue`.
static Value padChannels(RewriterBase &rewriter, Location loc, Value nhwc,
                         int64_t cPad, Value padValue) {
  auto type = cast<RankedTensorType>(nhwc.getType());
  int64_t c = type.getDimSize(3);
  if (c == cPad) return nhwc;
  SmallVector<OpFoldResult> low(4, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> high(4, rewriter.getIndexAttr(0));
  high[3] = rewriter.getIndexAttr(cPad - c);
  SmallVector<int64_t> shape(type.getShape());
  shape[3] = cPad;
  return tensor::PadOp::create(
             rewriter, loc, RankedTensorType::get(shape, type.getElementType()),
             nhwc, low, high, padValue)
      .getResult();
}

/// Same as `transposeTensor`, but a `linalg.fill` init is rebuilt as a fill of
/// the transposed shape instead of being transposed (keeps the init a plain
/// `empty -> fill` so the later pipeline sees the usual pooling init). `cPad`
/// is the (possibly padded) channel extent of the NHWC init; a rebuilt fill
/// takes it directly, any other init is transposed and then padded with
/// `padValue`.
static Value transposeInit(RewriterBase &rewriter, Location loc, Value init,
                           ArrayRef<int64_t> perm, int64_t cPad,
                           Value padValue) {
  if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
    auto initType = cast<RankedTensorType>(init.getType());
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, fill.getOutputs()[0]);
    SmallVector<OpFoldResult> dstSizes;
    for (int64_t p : perm) dstSizes.push_back(sizes[p]);
    if (cPad > 0) dstSizes[3] = rewriter.getIndexAttr(cPad);
    Value empty = tensor::EmptyOp::create(rewriter, loc, dstSizes,
                                          initType.getElementType());
    return linalg::FillOp::create(rewriter, loc, fill.getInputs()[0], empty)
        .getResult(0);
  }
  Value nhwc = transposeTensor(rewriter, loc, init, perm);
  return cPad > 0 ? padChannels(rewriter, loc, nhwc, cPad, padValue) : nhwc;
}

struct PoolingNchwMaxToNhwc : OpRewritePattern<linalg::PoolingNchwMaxOp> {
  PoolingNchwMaxToNhwc(MLIRContext *ctx, int64_t channelMultiple)
      : OpRewritePattern(ctx), channelMultiple(channelMultiple) {}

  LogicalResult matchAndRewrite(linalg::PoolingNchwMaxOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(op, "expected tensor semantics");
    Location loc = op.getLoc();
    Value input = op.getInputs()[0];
    Value window = op.getInputs()[1];
    Value init = op.getOutputs()[0];

    // Channel padding target (NCHW dim 1). 0 = none (static C already a
    // multiple, dynamic C, or padding disabled).
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t cPad = 0;
    if (channelMultiple > 1 && !inputType.isDynamicDim(1)) {
      int64_t c = inputType.getDimSize(1);
      if (c % channelMultiple != 0)
        cPad = (c / channelMultiple + 1) * channelMultiple;
    }
    // Pad value: the fill constant when the init is a plain fill (the usual
    // -inf), else zero. The padded channels are sliced off, so any value is
    // correct for the kept ones.
    Value padValue;
    if (cPad > 0) {
      if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
        padValue = fill.getInputs()[0];
      } else {
        padValue = arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getZeroAttr(inputType.getElementType()));
      }
    }

    Value inputNhwc = transposeTensor(rewriter, loc, input, kNchwToNhwc);
    if (cPad > 0) inputNhwc = padChannels(rewriter, loc, inputNhwc, cPad, padValue);
    Value initNhwc =
        transposeInit(rewriter, loc, init, kNchwToNhwc, cPad, padValue);
    auto pool = linalg::PoolingNhwcMaxOp::create(
        rewriter, loc, TypeRange{initNhwc.getType()},
        ValueRange{inputNhwc, window}, initNhwc, op.getStrides(),
        op.getDilations());
    Value resultNhwc = pool.getResult(0);
    if (cPad > 0) {
      // Slice the padded channels back off: [N, OH, OW, Cp] -> [N, OH, OW, C].
      auto resType = cast<RankedTensorType>(resultNhwc.getType());
      SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes =
          tensor::getMixedSizes(rewriter, loc, resultNhwc);
      sizes[3] = rewriter.getIndexAttr(inputType.getDimSize(1));
      SmallVector<int64_t> shape(resType.getShape());
      shape[3] = inputType.getDimSize(1);
      resultNhwc = tensor::ExtractSliceOp::create(
          rewriter, loc, RankedTensorType::get(shape, resType.getElementType()),
          resultNhwc, offsets, sizes, strides);
    }
    Value resultNchw = transposeTensor(rewriter, loc, resultNhwc, kNhwcToNchw);
    rewriter.replaceOp(op, resultNchw);
    return success();
  }

 private:
  int64_t channelMultiple;
};

/// Pre-split an NHWC max pooling with an overlapping window along H into
/// disjoint-per-chunk batched pooling (see the file comment).
struct PoolingNhwcMaxHaloSplit : OpRewritePattern<linalg::PoolingNhwcMaxOp> {
  PoolingNhwcMaxHaloSplit(MLIRContext *ctx, int64_t columns)
      : OpRewritePattern(ctx), columns(columns) {}

  LogicalResult matchAndRewrite(linalg::PoolingNhwcMaxOp op,
                                PatternRewriter &rewriter) const override {
    if (columns <= 1) return failure();
    if (!op.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(op, "expected tensor semantics");
    Value input = op.getInputs()[0];
    Value window = op.getInputs()[1];
    Value init = op.getOutputs()[0];
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto winType = dyn_cast<RankedTensorType>(window.getType());
    if (!inType || !outType || !winType || !inType.hasStaticShape() ||
        !outType.hasStaticShape() || !winType.hasStaticShape() ||
        inType.getRank() != 4 || winType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected static 4-D NHWC");
    // Already chunked (or a real batch): the chunk dim would collide with N.
    if (inType.getDimSize(0) != 1)
      return rewriter.notifyMatchFailure(op, "expected N == 1");
    int64_t strideH = op.getStrides().getValues<int64_t>()[0];
    int64_t dilH = op.getDilations().getValues<int64_t>()[0];
    int64_t winH = (winType.getDimSize(0) - 1) * dilH + 1;
    if (winH == strideH)
      return rewriter.notifyMatchFailure(op, "windows do not overlap along H");
    int64_t OH = outType.getDimSize(1);
    int64_t cols = choosePoolColumns(OH, columns);
    if (cols <= 1 || OH % cols != 0)
      return rewriter.notifyMatchFailure(op, "no column split for OH");
    int64_t t = OH / cols;                     // output rows per chunk
    int64_t rows = (t - 1) * strideH + winH;   // input rows per chunk (halo incl.)
    int64_t IW = inType.getDimSize(2), C = inType.getDimSize(3);
    int64_t OW = outType.getDimSize(2);
    Type elemType = inType.getElementType();
    Location loc = op.getLoc();

    // Gather the chunks: chunk i takes input rows [i*t*strideH, +rows).
    SmallVector<Value> chunks;
    SmallVector<OpFoldResult> sizes = {
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(rows),
        rewriter.getIndexAttr(IW), rewriter.getIndexAttr(C)};
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    auto chunkType = RankedTensorType::get({1, rows, IW, C}, elemType);
    for (int64_t i = 0; i < cols; ++i) {
      SmallVector<OpFoldResult> offsets = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(i * t * strideH),
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
      chunks.push_back(tensor::ExtractSliceOp::create(
          rewriter, loc, chunkType, input, offsets, sizes, strides));
    }
    Value stacked = tensor::ConcatOp::create(rewriter, loc, /*dim=*/0, chunks);

    // Init: a plain fill is rebuilt at the chunked shape, anything else is
    // reshaped [1, OH, OW, C] -> [OH, OW, C] -> [cols, t, OW, C].
    SmallVector<ReassociationIndices> collapseN = {{0, 1}, {2}, {3}};
    auto chunkedOutType = RankedTensorType::get({cols, t, OW, C}, elemType);
    Value initChunked;
    if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
      Value empty = tensor::EmptyOp::create(rewriter, loc,
                                            chunkedOutType.getShape(), elemType);
      initChunked =
          linalg::FillOp::create(rewriter, loc, fill.getInputs()[0], empty)
              .getResult(0);
    } else {
      Value flat = tensor::CollapseShapeOp::create(rewriter, loc, init, collapseN);
      initChunked = tensor::ExpandShapeOp::create(rewriter, loc, chunkedOutType,
                                                  flat, collapseN);
    }
    auto pool = linalg::PoolingNhwcMaxOp::create(
        rewriter, loc, TypeRange{chunkedOutType}, ValueRange{stacked, window},
        initChunked, op.getStrides(), op.getDilations());
    // [cols, t, OW, C] -> [OH, OW, C] -> [1, OH, OW, C].
    Value flat =
        tensor::CollapseShapeOp::create(rewriter, loc, pool.getResult(0), collapseN);
    Value result =
        tensor::ExpandShapeOp::create(rewriter, loc, outType, flat, collapseN);
    rewriter.replaceOp(op, result);
    return success();
  }

 private:
  int64_t columns;
};

class AMDAIEConvertPoolingToChannelsLastPass
    : public impl::AMDAIEConvertPoolingToChannelsLastBase<
          AMDAIEConvertPoolingToChannelsLastPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }
  AMDAIEConvertPoolingToChannelsLastPass() = default;
  AMDAIEConvertPoolingToChannelsLastPass(
      const AMDAIEConvertPoolingToChannelsLastOptions &options)
      : AMDAIEConvertPoolingToChannelsLastBase(options) {}
  AMDAIEConvertPoolingToChannelsLastPass(
      const AMDAIEConvertPoolingToChannelsLastPass &pass) {}
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<PoolingNchwMaxToNhwc>(ctx, channelMultiple);
    patterns.add<PoolingNhwcMaxHaloSplit>(ctx, haloSplitColumns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEConvertPoolingToChannelsLastPass(
    AMDAIEConvertPoolingToChannelsLastOptions options) {
  return std::make_unique<AMDAIEConvertPoolingToChannelsLastPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
