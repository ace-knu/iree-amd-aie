// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-npu-dma-to-half-dma-cpy-nd"

namespace mlir::iree_compiler::AMDAIE {

struct NpuDmaToHalfDmaCpyNdConverter final
    : OpConversionPattern<AMDAIE::NpuDmaCpyNdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AMDAIE::NpuDmaCpyNdOp dmaOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "matchAndRewrite[AMDAIE::NpuDmaCpyNdOp]\n");
    AMDAIE::ConnectionOp connectionOp = dmaOp.getConnectionOp();
    if (!connectionOp) {
      return dmaOp.emitOpError()
             << "should operate on an `amdaie.connection` op";
    }
    // Convert source half. Only when this op actually describes a source-side
    // transfer: falling back to the connection's own static source endpoint
    // for a `dmaOp` that never had one would synthesize an inert placeholder
    // half (its "input" is just the connection's declared endpoint, with no
    // real per-iteration offsets/sizes/strides and no async token to wait on)
    // -- the real movement for that side, if any, is issued by a different,
    // genuinely source-populated `NpuDmaCpyNdOp` elsewhere, or handled
    // entirely by the connection's own circular/repeat descriptor. Such a
    // placeholder has always been harmless downstream because its memory
    // space (memtile/core) got it filtered out of BD-ID requirements, but a
    // shim-to-shim relay connection's placeholder is also memory space 0 --
    // indistinguishable by that heuristic from a real shim DMA needing a BD
    // ID. Not synthesizing the placeholder at all sidesteps that ambiguity.
    AMDAIE::NpuHalfDmaCpyNdOp sourceDma = nullptr;
    bool hasAsyncSourceToken =
        llvm::any_of(dmaOp.getAsyncTokens(), [](Value token) {
          return isa<AMDAIE::AsyncSourceTokenType>(token.getType());
        });
    SmallVector<Type> resultTypes = {
        rewriter.getType<AMDAIE::AsyncTokenType>()};
    if (dmaOp.getSource()) {
      if (connectionOp.getSourceChannels().size() != 1) {
        return connectionOp.emitOpError()
               << "expected a single source channel";
      }
      auto sourceChannelOp = dyn_cast<AMDAIE::ChannelOp>(
          connectionOp.getSourceChannels()[0].getDefiningOp());
      TypeRange sourceResultTypes =
          hasAsyncSourceToken ? TypeRange{resultTypes} : TypeRange{};
      rewriter.setInsertionPoint(dmaOp);
      sourceDma = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
          dmaOp.getLoc(), sourceResultTypes, connectionOp, dmaOp.getSource(),
          dmaOp.getSourceMixedOffsets(), dmaOp.getSourceMixedSizes(),
          dmaOp.getSourceMixedStrides(), dmaOp.getSourceBdId(),
          sourceChannelOp);
    }

    // Convert target half, under the same "only if genuinely populated" rule
    // as the source half above.
    AMDAIE::NpuHalfDmaCpyNdOp targetDma = nullptr;
    bool hasAsyncTargetToken =
        llvm::any_of(dmaOp.getAsyncTokens(), [](Value token) {
          return isa<AMDAIE::AsyncTargetTokenType>(token.getType());
        });
    if (dmaOp.getTarget()) {
      // Broadcasting is allowed only when the NPU DMA operation does not
      // specify a target LogicalObjectFifo, meaning the data flow is directed
      // into the AIE array. Otherwise, if a target is specified, ensure there
      // is exactly one target channel.
      if (connectionOp.getTargetChannels().size() != 1)
        return connectionOp.emitOpError()
               << "expected a single target channel";
      auto targetChannelOp = dyn_cast<AMDAIE::ChannelOp>(
          connectionOp.getTargetChannels()[0].getDefiningOp());
      TypeRange targetResultTypes =
          hasAsyncTargetToken ? TypeRange{resultTypes} : TypeRange{};
      rewriter.setInsertionPoint(dmaOp);
      targetDma = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
          dmaOp.getLoc(), targetResultTypes, connectionOp, dmaOp.getTarget(),
          dmaOp.getTargetMixedOffsets(), dmaOp.getTargetMixedSizes(),
          dmaOp.getTargetMixedStrides(), dmaOp.getTargetBdId(),
          targetChannelOp);
    }
    // Build a one-to-one replacement vector to satisfy the conversion
    // framework without duplicate replacements.
    SmallVector<Value> replacements;
    replacements.reserve(dmaOp.getNumResults());
    for (auto result : dmaOp.getResults()) {
      Value replacement;
      if (isa<AMDAIE::AsyncSourceTokenType>(result.getType())) {
        if (sourceDma && sourceDma.getNumResults() == 1) {
          replacement = sourceDma.getResult(0);
        }
      } else if (isa<AMDAIE::AsyncTargetTokenType>(result.getType())) {
        if (targetDma && targetDma.getNumResults() == 1) {
          replacement = targetDma.getResult(0);
        }
      }
      if (!replacement) {
        return dmaOp.emitOpError() << "could not find replacement for result";
      }
      replacements.push_back(replacement);
    }
    rewriter.replaceOp(dmaOp, replacements);
    return success();
  }
};

struct NpuDmaWaitOpConverter final : OpConversionPattern<AMDAIE::NpuDmaWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AMDAIE::NpuDmaWaitOp waitOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Rebuild the wait op with converted async tokens to drop any
    // source/target-specific token types in favor of the unified async token.
    rewriter.replaceOpWithNewOp<AMDAIE::NpuDmaWaitOp>(waitOp,
                                                      adaptor.getAsyncTokens());
    return success();
  }
};

namespace {
class AMDAIENpuDmaToHalfDmaCpyNdPass
    : public impl::AMDAIENpuDmaToHalfDmaCpyNdBase<
          AMDAIENpuDmaToHalfDmaCpyNdPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIENpuDmaToHalfDmaCpyNdPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget conversionTarget(*context);

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  // Specific mapping: source/target token -> generic async token.
  typeConverter.addConversion([](AMDAIE::AsyncSourceTokenType type) {
    return AMDAIE::AsyncTokenType::get(type.getContext());
  });
  typeConverter.addConversion([](AMDAIE::AsyncTargetTokenType type) {
    return AMDAIE::AsyncTokenType::get(type.getContext());
  });

  conversionTarget.addLegalDialect<AMDAIEDialect>();
  conversionTarget.addIllegalOp<AMDAIE::NpuDmaCpyNdOp>();
  conversionTarget.addDynamicallyLegalOp<AMDAIE::NpuDmaWaitOp>(
      [&](AMDAIE::NpuDmaWaitOp op) {
        return typeConverter.isLegal(op.getAsyncTokens().getTypes());
      });

  patterns.insert<NpuDmaToHalfDmaCpyNdConverter>(typeConverter, context);
  patterns.insert<NpuDmaWaitOpConverter>(typeConverter, context);

  if (failed(applyPartialConversion(parentOp, conversionTarget,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIENpuDmaToHalfDmaCpyNdPass() {
  return std::make_unique<AMDAIENpuDmaToHalfDmaCpyNdPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
