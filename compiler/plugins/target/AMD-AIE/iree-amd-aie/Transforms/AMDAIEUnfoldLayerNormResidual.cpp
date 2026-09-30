// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIELayerNormUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-amdaie-unfold-layernorm-residual"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Take the residual producer out of a LayerNorm `custom_op` that carries it
/// in its region, leaving
///
///   %h = linalg.generic ins(%a, %b, %mb) ...
///   %y = custom_op ins(%h) { <the normalization> }
///
/// which is exactly what an int8 residual add looks like once dispatch
/// formation has cloned its producer in.
///
/// The producer is the first `linalg.generic` of the region to read the
/// region's first argument -- directly, or through the `tensor.cast` to a
/// static feature size that canonicalization puts in; the raise builds it
/// first, so everything it needs sits in front of it. Of those, whatever the normalization also uses -- an
/// index constant CSE merged, say -- is cloned into the new region too. The
/// one thing that cannot follow is a read of the activations themselves, and
/// the only such read that survives canonicalization is a `tensor.dim`, which
/// is taken of the samples instead: every operand of the op spans the same
/// rows and features.
LogicalResult unfold(RewriterBase &rewriter,
                     IREE::LinalgExt::CustomOp customOp) {
  Block &oldBody = customOp.getRegion().front();
  if (customOp.getInputs().size() != 2 || customOp.getOutputs().size() != 1 ||
      oldBody.getNumArguments() != 3)
    return customOp.emitOpError("expected two inputs and one output");
  BlockArgument argA = oldBody.getArgument(0);
  BlockArgument argB = oldBody.getArgument(1);
  BlockArgument argInit = oldBody.getArgument(2);
  // An activation as the region sees it, possibly behind casts.
  auto activationOf = [&](Value v) -> Value {
    while (auto cast = v.getDefiningOp<tensor::CastOp>()) v = cast.getSource();
    return v == argA || v == argB ? v : Value();
  };

  linalg::GenericOp producer;
  for (Operation &op : oldBody) {
    auto generic = dyn_cast<linalg::GenericOp>(&op);
    if (generic && llvm::any_of(generic.getDpsInputs(), [&](Value v) {
          return activationOf(v) == argA;
        })) {
      producer = generic;
      break;
    }
  }
  if (!producer || producer->getNumResults() != 1)
    return customOp.emitOpError("found no residual producer in the region");

  // The producer's own inputs and what its body captures from the region.
  SetVector<Value> captured;
  getUsedValuesDefinedAbove(producer.getRegion(), captured);
  Location loc = producer.getLoc();
  rewriter.setInsertionPoint(customOp);
  IRMapping outer;
  outer.map(argA, customOp.getInputs()[0]);
  outer.map(argB, customOp.getInputs()[1]);
  auto cloneOutside = [&](Value v) -> LogicalResult {
    if (outer.contains(v)) return success();
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != &oldBody) return success();
    if (!matchPattern(v, m_Constant()))
      return customOp.emitOpError("producer reads a non-constant ")
             << v << " from the region";
    rewriter.clone(*def, outer);
    return success();
  };
  for (Value v : captured)
    if (failed(cloneOutside(v))) return failure();
  SmallVector<Value> inputs;
  for (Value v : producer.getDpsInputs()) {
    // Outside, the activations are already statically shaped: no cast.
    if (Value act = activationOf(v)) {
      inputs.push_back(outer.lookup(act));
      continue;
    }
    if (failed(cloneOutside(v))) return failure();
    inputs.push_back(outer.lookupOrDefault(v));
  }

  // Rebuilt rather than cloned: outside, the activations are statically
  // shaped, and so is the int16 buffer between the two ops.
  Value outerA = customOp.getInputs()[0];
  auto samplesType = cast<RankedTensorType>(producer.getResult(0).getType());
  Value init = rewriter.create<tensor::EmptyOp>(
      loc, tensor::getMixedSizes(rewriter, loc, outerA),
      samplesType.getElementType());
  auto newProducer = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{init.getType()}, inputs, ValueRange{init},
      producer.getIndexingMapsArray(), producer.getIteratorTypesArray());
  rewriter.cloneRegionBefore(producer.getRegion(), newProducer.getRegion(),
                             newProducer.getRegion().end(), outer);
  Value samples = newProducer.getResult(0);

  // The op again, fed the samples.
  SmallVector<Attribute> maps(customOp.getIndexingMaps().getValue());
  maps.erase(maps.begin() + 1);
  auto newOp = rewriter.create<IREE::LinalgExt::CustomOp>(
      customOp.getLoc(), customOp->getResultTypes(), ValueRange{samples},
      customOp.getOutputs(), rewriter.getArrayAttr(maps),
      customOp.getIteratorTypes());
  for (NamedAttribute attr : customOp->getDiscardableAttrs())
    if (attr.getName() != kLayerNormResidualInRegion)
      newOp->setAttr(attr.getName(), attr.getValue());

  // The region sees fully dynamic slices, as the raise built it; the
  // normalization inside was written against the producer's own result type.
  auto dynamicSamplesType = RankedTensorType::get(
      SmallVector<int64_t>(samplesType.getRank(), ShapedType::kDynamic),
      samplesType.getElementType());
  Block *newBody = rewriter.createBlock(
      &newOp.getRegion(), newOp.getRegion().end(),
      TypeRange{dynamicSamplesType, argInit.getType()},
      {loc, argInit.getLoc()});
  BlockArgument newSamples = newBody->getArgument(0);
  IRMapping inner;
  inner.map(argInit, newBody->getArgument(1));
  rewriter.setInsertionPointToStart(newBody);
  Value samplesInside = newSamples;
  if (samplesType != dynamicSamplesType)
    samplesInside =
        rewriter.create<tensor::CastOp>(loc, samplesType, newSamples);
  inner.map(producer.getResult(0), samplesInside);
  // What of the producer's side of the block the normalization still needs,
  // transitively. A `tensor.dim` of an activation needs only its index.
  auto isActivationDim = [&](Operation *op) {
    auto dim = dyn_cast<tensor::DimOp>(op);
    return dim && activationOf(dim.getSource());
  };
  DenseSet<Operation *> needed;
  SmallVector<Operation *> work;
  auto require = [&](Value v) {
    Operation *def = v.getDefiningOp();
    if (def && def->getBlock() == &oldBody && def != producer.getOperation() &&
        def->isBeforeInBlock(producer) && needed.insert(def).second)
      work.push_back(def);
  };
  auto requireOperandsOf = [&](Operation *op) {
    if (isActivationDim(op)) {
      require(cast<tensor::DimOp>(op).getIndex());
      return;
    }
    op->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands()) require(operand);
    });
  };
  for (Operation &op : oldBody)
    if (producer->isBeforeInBlock(&op)) requireOperandsOf(&op);
  while (!work.empty()) requireOperandsOf(work.pop_back_val());

  for (Operation &op : oldBody) {
    if (&op == producer.getOperation()) continue;
    if (op.isBeforeInBlock(producer) && !needed.contains(&op)) continue;
    if (isActivationDim(&op)) {
      auto dim = cast<tensor::DimOp>(&op);
      Value index = inner.lookupOrDefault(dim.getIndex());
      inner.map(dim.getResult(),
                rewriter.create<tensor::DimOp>(dim.getLoc(), newSamples, index)
                    .getResult());
      continue;
    }
    bool readsActivations = false;
    op.walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        if (activationOf(operand)) readsActivations = true;
    });
    if (readsActivations)
      return customOp.emitOpError("normalization reads the activations");
    rewriter.clone(op, inner);
  }

  rewriter.replaceOp(customOp, newOp->getResults());
  return success();
}

class AMDAIEUnfoldLayerNormResidual
    : public impl::AMDAIEUnfoldLayerNormResidualBase<
          AMDAIEUnfoldLayerNormResidual> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    SmallVector<IREE::LinalgExt::CustomOp> ops;
    getOperation()->walk([&](IREE::LinalgExt::CustomOp op) {
      if (op->hasAttr(kLayerNormResidualInRegion)) ops.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for (IREE::LinalgExt::CustomOp op : ops)
      if (failed(unfold(rewriter, op))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEUnfoldLayerNormResidualPass() {
  return std::make_unique<AMDAIEUnfoldLayerNormResidual>();
}

}  // namespace mlir::iree_compiler::AMDAIE
