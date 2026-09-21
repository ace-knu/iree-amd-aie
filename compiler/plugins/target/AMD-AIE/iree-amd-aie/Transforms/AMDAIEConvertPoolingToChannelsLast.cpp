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

#include "iree-amd-aie/Transforms/Passes.h"
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

/// Same as `transposeTensor`, but a `linalg.fill` init is rebuilt as a fill of
/// the transposed shape instead of being transposed (keeps the init a plain
/// `empty -> fill` so the later pipeline sees the usual pooling init).
static Value transposeInit(RewriterBase &rewriter, Location loc, Value init,
                           ArrayRef<int64_t> perm) {
  if (auto fill = init.getDefiningOp<linalg::FillOp>()) {
    auto initType = cast<RankedTensorType>(init.getType());
    SmallVector<OpFoldResult> sizes =
        tensor::getMixedSizes(rewriter, loc, fill.getOutputs()[0]);
    SmallVector<OpFoldResult> dstSizes;
    for (int64_t p : perm) dstSizes.push_back(sizes[p]);
    Value empty = tensor::EmptyOp::create(rewriter, loc, dstSizes,
                                          initType.getElementType());
    return linalg::FillOp::create(rewriter, loc, fill.getInputs()[0], empty)
        .getResult(0);
  }
  return transposeTensor(rewriter, loc, init, perm);
}

struct PoolingNchwMaxToNhwc : OpRewritePattern<linalg::PoolingNchwMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNchwMaxOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(op, "expected tensor semantics");
    Location loc = op.getLoc();
    Value input = op.getInputs()[0];
    Value window = op.getInputs()[1];
    Value init = op.getOutputs()[0];

    Value inputNhwc = transposeTensor(rewriter, loc, input, kNchwToNhwc);
    Value initNhwc = transposeInit(rewriter, loc, init, kNchwToNhwc);
    auto pool = linalg::PoolingNhwcMaxOp::create(
        rewriter, loc, TypeRange{initNhwc.getType()},
        ValueRange{inputNhwc, window}, initNhwc, op.getStrides(),
        op.getDilations());
    Value resultNchw =
        transposeTensor(rewriter, loc, pool.getResult(0), kNhwcToNchw);
    rewriter.replaceOp(op, resultNchw);
    return success();
  }
};

class AMDAIEConvertPoolingToChannelsLastPass
    : public impl::AMDAIEConvertPoolingToChannelsLastBase<
          AMDAIEConvertPoolingToChannelsLastPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<PoolingNchwMaxToNhwc>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEConvertPoolingToChannelsLastPass() {
  return std::make_unique<AMDAIEConvertPoolingToChannelsLastPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
