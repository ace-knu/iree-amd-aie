// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Folds a `linalg.matmul` fused with a broadcast 1-D bias -- either as the
// matmul's own accumulator (`outs(bias_broadcast)`) or as a separate `add`
// generic immediately consuming the matmul's result -- into a single, larger
// matmul with no bias operand or bias-add left at all, using the identity
// `y = W*x + b == W_aug * x_aug` where `W_aug` has one extra reduction (K) row
// equal to `b`, and `x_aug` has one extra K column of the constant 1.0.
//
// Motivation: this exact "contraction with a bias-broadcast accumulator"
// shape is what `GlobalOptimization::DetachElementwiseFromNamedOpsPass`
// (`--iree-global-opt-detach-elementwise-through-reshape`) and dispatch
// creation's contraction-fusion heuristic
// (`--iree-dispatch-creation-no-fuse-into-contraction-conv-roots`) exist to
// work around: left alone, DispatchCreation fuses the bias into the
// contraction's dispatch as a third input (activation, weight, bias), and the
// amd-aie backend cannot give bias its own minimum ~8 independent shim DMA
// producer connections without exceeding the physical 8-column x 2-channel
// shim capacity once weight/activation's own connections are counted too (see
// docs/2026-08-24_3input-dispatch-investigation.md and
// docs/2026-08-25_connection-reuse-below-16.md for the full investigation).
//
// Running this pass in the Preprocessing phase (before GlobalOptimization,
// hence before both the detach pass and dispatch creation ever see the op)
// removes the third input outright: amd-aie's codegen only ever sees an
// ordinary 2-input contraction, so it inherits the weight operand's existing
// (already working) connection-reuse behavior for free, and the two
// workaround flags become unnecessary for this pattern.
//
// `linalg.conv_2d_nchw_fchw` is handled the same way: one extra *input
// channel*, held at the constant 1.0 across the whole (already-padded) input,
// with a filter tap of `bias` at a single (kh=0, kw=0) position (0 elsewhere)
// -- see `foldBiasIntoConv2d`'s doc comment. torch-to-linalg's onnx.Conv
// lowering gives the conv's accumulator as a `linalg.broadcast` (a distinct
// named op, not the `linalg.generic` broadcast `matchBiasBroadcast` targets
// for matmul) immediately followed by the conv; this runs before
// `--iree-preprocessing-convert-conv-to-channels-last` (a user-opt-in
// pipeline flag, not run by default), so only the NCHW/FCHW layout is
// matched -- an already-channels-last conv falls through unmatched.

#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-amdaie-fold-bias-into-contraction"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// For a `linalg.matmul` operand `map` (2-D, non-batched), returns the
/// operand-dim position that the given contraction loop dim (M=0, N=1, K=2 --
/// fixed by linalg.matmul's iterator order; transpose variants only permute
/// the operand maps, not the loop order) occupies. Mirrors
/// `AMDAIEPadContractionDispatches.cpp`'s `operandDimForLoop`  (kept as a
/// separate copy: that one is `static` in its own translation unit).
static std::optional<unsigned> operandDimForLoop(AffineMap map,
                                                  unsigned loopDim) {
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
    if (dimExpr && dimExpr.getPosition() == loopDim) return i;
  }
  return std::nullopt;
}

/// Matches a `linalg.generic` that is a pure broadcast of a 1-D tensor: a
/// single input, a single init, all-parallel iterators, and a body that does
/// nothing but yield the input straight through. This is the shape
/// `DetachElementwiseFromNamedOpsPass` (see
/// `third_party/iree/.../GlobalOptimization/DetachElementwiseFromNamedOps.cpp`)
/// looks for as a contraction/conv accumulator: e.g.
///   %bcast = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d1)>,
///                                              affine_map<(d0,d1)->(d0,d1)>],
///                            iterator_types = ["parallel","parallel"]}
///       ins(%bias : tensor<Nxf32>) outs(%empty : tensor<MxNxf32>) {
///     ^bb0(%in: f32, %out: f32): linalg.yield %in : f32
///   } -> tensor<MxNxf32>
/// On a match, returns the 1-D bias operand and the output-tensor dim it
/// broadcasts into (as a loop dim, via the *consuming* op's own indexing
/// map -- the caller resolves that).
static Value matchBiasBroadcast(linalg::GenericOp generic,
                                unsigned &biasResultDim) {
  if (!generic || !generic.hasPureTensorSemantics()) return nullptr;
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return nullptr;
  if (!generic->getResult(0).hasOneUse()) return nullptr;
  for (utils::IteratorType it : generic.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel) return nullptr;

  Block &body = generic.getRegion().front();
  if (!llvm::hasSingleElement(body)) return nullptr;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1) return nullptr;
  if (yieldOp.getOperand(0) != body.getArgument(0)) return nullptr;

  OpOperand *biasOperand = generic.getDpsInputOperand(0);
  Value bias = biasOperand->get();
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || biasType.getRank() != 1) return nullptr;

  AffineMap biasMap = generic.getMatchingIndexingMap(biasOperand);
  if (biasMap.getNumResults() != 1) return nullptr;
  auto dimExpr = dyn_cast<AffineDimExpr>(biasMap.getResult(0));
  if (!dimExpr) return nullptr;
  biasResultDim = dimExpr.getPosition();
  return bias;
}

/// Matches `linalg.matmul` (with an ordinary accumulator, e.g. a
/// `linalg.fill(0)`) immediately followed by exactly one `linalg.generic`
/// that adds a broadcast 1-D bias onto the matmul's result:
///   %mm = linalg.matmul ...
///   %r = linalg.generic ins(%mm, %bias) ... { ^bb0(%m, %b, %o): yield %m+%b }
/// This is the shape a direct "y = x @ W + b" lowering already produces (e.g.
/// torch-mlir's onnx.Gemm import via torch-to-linalg) without ever going
/// through the accumulator form `matchBiasBroadcast` above targets -- no
/// upstream detach pass is involved. On a match, returns the bias-add
/// `linalg.generic`, the 1-D bias operand, and the dim (in the generic's
/// output tensor, same shape as the matmul's result) it broadcasts into.
static LogicalResult matchPostAddBias(linalg::MatmulOp matmul,
                                      linalg::GenericOp &genericOp, Value &bias,
                                      unsigned &biasResultDim) {
  if (!matmul->hasOneUse()) return failure();
  auto generic = dyn_cast<linalg::GenericOp>(*matmul->getUsers().begin());
  if (!generic || !generic.hasPureTensorSemantics()) return failure();
  if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1)
    return failure();
  for (utils::IteratorType it : generic.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel) return failure();

  OpOperand *matmulOperand = nullptr;
  OpOperand *biasOperand = nullptr;
  for (OpOperand *operand : generic.getDpsInputOperands()) {
    if (operand->get() == matmul.getResult(0))
      matmulOperand = operand;
    else
      biasOperand = operand;
  }
  if (!matmulOperand || !biasOperand) return failure();
  if (!generic.getMatchingIndexingMap(matmulOperand).isIdentity())
    return failure();

  auto biasType = dyn_cast<RankedTensorType>(biasOperand->get().getType());
  if (!biasType || biasType.getRank() != 1) return failure();
  AffineMap biasMap = generic.getMatchingIndexingMap(biasOperand);
  if (biasMap.getNumResults() != 1) return failure();
  auto dimExpr = dyn_cast<AffineDimExpr>(biasMap.getResult(0));
  if (!dimExpr) return failure();

  // Body must be exactly `yield(addf(matmulArg, biasArg))`, either operand
  // order (addf is commutative).
  Block &body = generic.getRegion().front();
  if (!llvm::hasSingleElement(body.without_terminator())) return failure();
  auto addOp = dyn_cast<arith::AddFOp>(&body.front());
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!addOp || !yieldOp || yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0) != addOp.getResult())
    return failure();
  Value matmulArg = body.getArgument(matmulOperand->getOperandNumber());
  Value biasArg = body.getArgument(biasOperand->getOperandNumber());
  bool argsMatch = (addOp.getLhs() == matmulArg && addOp.getRhs() == biasArg) ||
                   (addOp.getLhs() == biasArg && addOp.getRhs() == matmulArg);
  if (!argsMatch) return failure();

  genericOp = generic;
  bias = biasOperand->get();
  biasResultDim = dimExpr.getPosition();
  return success();
}

/// Builds `tensor.empty(dstShape) ; tensor.insert_slice %src into it at
/// `offsets` (full extent of `src`, unit strides)`, returning the insert.
static Value insertFullSlice(RewriterBase &rewriter, Location loc, Value src,
                             Value dst, ArrayRef<int64_t> offsets) {
  int64_t rank = cast<RankedTensorType>(src.getType()).getRank();
  SmallVector<OpFoldResult> offs, sizes, strides;
  for (int64_t i = 0; i < rank; ++i) {
    offs.push_back(rewriter.getIndexAttr(offsets[i]));
    strides.push_back(rewriter.getIndexAttr(1));
  }
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, src);
  return tensor::InsertSliceOp::create(rewriter, loc, src, dst, offs,
                                       mixedSizes, strides);
}

/// Matches a `linalg.broadcast` that broadcasts a 1-D bias into `init`'s
/// shape, keeping exactly one dim of `init` un-broadcast (the one the 1-D
/// bias indexes). Named-op counterpart of `matchBiasBroadcast` above: the
/// torch-to-linalg conv+bias lowering produces a `linalg.broadcast` (a
/// distinct named op, not a `linalg.generic`) as the conv's accumulator.
static Value matchConvBiasBroadcast(linalg::BroadcastOp broadcast,
                                    unsigned &biasResultDim) {
  if (!broadcast || !broadcast.hasPureTensorSemantics()) return nullptr;
  if (!broadcast->getResult(0).hasOneUse()) return nullptr;
  Value bias = broadcast.getInput();
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || biasType.getRank() != 1) return nullptr;
  auto initType = dyn_cast<RankedTensorType>(broadcast.getInit().getType());
  if (!initType) return nullptr;
  ArrayRef<int64_t> dims = broadcast.getDimensions();
  if (static_cast<int64_t>(dims.size()) != initType.getRank() - 1)
    return nullptr;
  llvm::SmallDenseSet<int64_t> broadcastDims(dims.begin(), dims.end());
  int64_t nonBroadcastDim = -1;
  for (int64_t i = 0, e = initType.getRank(); i < e; ++i) {
    if (broadcastDims.contains(i)) continue;
    if (nonBroadcastDim != -1) return nullptr;  // more than one -- ambiguous
    nonBroadcastDim = i;
  }
  if (nonBroadcastDim == -1) return nullptr;
  biasResultDim = nonBroadcastDim;
  return bias;
}

/// Matches a conv's accumulator down to its underlying 1-D bias, handling
/// both forms observed: a direct `linalg.broadcast` (native NCHW/FCHW
/// output, no `--iree-preprocessing-convert-conv-to-channels-last`), and a
/// `linalg.transpose` of one (channels-last: that pass converts an
/// NCHW-shaped bias broadcast into the NHWC-shaped accumulator the
/// now-channels-last conv actually consumes, via a `linalg.transpose` in
/// between). `wantDim` is the conv's own output-channel dim (from
/// `ConvChannelDims::outputFPos`, i.e. the accumulator's shape, which is the
/// same shape as the conv's own result); resolved back through any
/// transpose, the broadcast's non-broadcast dim must land exactly there,
/// same requirement `foldBiasIntoMatmul` places on the N-dim (bias is
/// per-output-channel, not per-spatial-position). On a match, appends the
/// broadcast (and transpose, if present) to `deadOps` for the caller to
/// erase once unused.
static Value matchConvAccumulatorBias(Value initOperand, unsigned wantDim,
                                      SmallVectorImpl<Operation *> &deadOps) {
  Value bcastResult = initOperand;
  if (auto transposeOp = initOperand.getDefiningOp<linalg::TransposeOp>()) {
    if (!transposeOp->getResult(0).hasOneUse()) return nullptr;
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    if (wantDim >= perm.size()) return nullptr;
    // dim(result, wantDim) == dim(input, perm[wantDim]) -- resolve `wantDim`
    // back to the pre-transpose (broadcast's own) dim space.
    wantDim = perm[wantDim];
    bcastResult = transposeOp.getInput();
    deadOps.push_back(transposeOp);
  }
  auto broadcast = bcastResult.getDefiningOp<linalg::BroadcastOp>();
  unsigned biasResultDim = 0;
  Value bias = matchConvBiasBroadcast(broadcast, biasResultDim);
  if (!bias || biasResultDim != wantDim) return nullptr;
  deadOps.push_back(broadcast);
  return bias;
}

/// The operand-dim positions (input-channel and output-channel) a 2-D conv's
/// K+1-style bias fold needs, resolved from the conv's own indexing maps
/// without assuming any fixed physical layout: NCHW/NHWC (and other
/// permutations linalg's named conv variants support) assign the same loop
/// dims to different operand-dim positions, and even different loop-dim
/// *indices* per variant (e.g. `conv_2d_nchw_fchw` uses loop dim 1 for output
/// channel, `conv_2d_nhwc_hwcf` uses loop dim 3) -- nothing here is
/// hardcoded to either.
struct ConvChannelDims {
  unsigned inputCPos, filterCPos, filterFPos, outputFPos;
};

/// Returns the operand-dim position of the unique result in `map` that is a
/// *pure* `AffineDimExpr` equal to `loopDim` -- i.e. a result referencing
/// `loopDim` alone, not combined with any other dim the way a conv's H/W
/// dims always are (`d_oh * stride + d_kh * dilation`). Distinguishes a
/// spatial (kh/kw) reduction dim, which never appears "pure" in the input
/// map, from the input-channel reduction dim, which always does.
static std::optional<unsigned> pureDimPos(AffineMap map, unsigned loopDim) {
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
    if (dimExpr && dimExpr.getPosition() == loopDim) return i;
  }
  return std::nullopt;
}

static std::optional<ConvChannelDims> resolveConvChannelDims(
    linalg::LinalgOp conv) {
  SmallVector<AffineMap> maps = conv.getIndexingMapsArray();
  if (maps.size() != 3) return std::nullopt;
  AffineMap inputMap = maps[0], filterMap = maps[1], outputMap = maps[2];
  SmallVector<utils::IteratorType> iterTypes = conv.getIteratorTypesArray();

  // Output-channel (F) loop dim: the one loop dim the filter map references
  // that is `parallel` -- every other filter-referenced loop dim (spatial
  // kh/kw and input-channel) is `reduction`.
  std::optional<unsigned> fLoopDim, filterFPos;
  for (unsigned i = 0, e = filterMap.getNumResults(); i < e; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(filterMap.getResult(i));
    if (!dimExpr) continue;
    unsigned d = dimExpr.getPosition();
    if (d >= iterTypes.size() ||
        iterTypes[d] != utils::IteratorType::parallel)
      continue;
    if (fLoopDim) return std::nullopt;  // more than one -- not a plain conv
    fLoopDim = d;
    filterFPos = i;
  }
  if (!fLoopDim) return std::nullopt;

  // Input-channel (C) loop dim: the one `reduction` loop dim the filter map
  // references that *also* appears as a pure dim reference in the input map
  // (kh/kw are reduction too, but only ever appear inside the input map's
  // strided+dilated compound exprs, never as a pure reference there).
  std::optional<unsigned> cLoopDim, filterCPos, inputCPos;
  for (unsigned i = 0, e = filterMap.getNumResults(); i < e; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(filterMap.getResult(i));
    if (!dimExpr) continue;
    unsigned d = dimExpr.getPosition();
    if (d >= iterTypes.size() ||
        iterTypes[d] != utils::IteratorType::reduction)
      continue;
    std::optional<unsigned> inPos = pureDimPos(inputMap, d);
    if (!inPos) continue;
    if (cLoopDim) return std::nullopt;  // more than one -- ambiguous
    cLoopDim = d;
    filterCPos = i;
    inputCPos = *inPos;
  }
  if (!cLoopDim) return std::nullopt;

  std::optional<unsigned> outputFPos = pureDimPos(outputMap, *fLoopDim);
  if (!outputFPos) return std::nullopt;

  return ConvChannelDims{*inputCPos, *filterCPos, *filterFPos, *outputFPos};
}

/// Attempts the K+1-style augmentation fold on a single 2-D conv
/// (`linalg.conv_2d_nchw_fchw` or `linalg.conv_2d_nhwc_hwcf` -- the only two
/// variants VGG16's im2col path and channels-last preprocessing produce; see
/// file header). Identical identity to `foldBiasIntoMatmul`, applied to
/// conv's input-channel axis instead of a matmul's K: one extra input
/// channel, held at the constant 1.0 across the *entire* (already-padded)
/// input -- uniform regardless of stride/dilation, since the identity only
/// needs one selected (kh, kw) tap of it to read back 1.0 -- and a filter tap
/// of `bias` at that single (kh=0, kw=0) position (0 elsewhere along kh/kw),
/// contributing to each output position exactly once, unscaled. Returns
/// failure (leaving the IR untouched) if the accumulator is not a bias
/// broadcast in the exact shape handled, so unrelated convs (including plain
/// `fill(0)`-initialized ones, or ones with a per-spatial-position rather
/// than per-output-channel accumulator) are never touched.
static LogicalResult foldBiasIntoConv2d(RewriterBase &rewriter,
                                        linalg::LinalgOp conv) {
  if (!isa<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>(
          conv.getOperation()))
    return failure();
  if (!conv.hasPureTensorSemantics()) return failure();
  if (conv.getDpsInits().size() != 1) return failure();

  std::optional<ConvChannelDims> dims = resolveConvChannelDims(conv);
  if (!dims) return failure();

  SmallVector<Operation *> deadOps;
  Value initOperand = conv.getDpsInits()[0];
  Value bias =
      matchConvAccumulatorBias(initOperand, dims->outputFPos, deadOps);
  if (!bias) return failure();

  Value input = conv.getDpsInputs()[0];
  Value filter = conv.getDpsInputs()[1];
  auto inputType = cast<RankedTensorType>(input.getType());
  auto filterType = cast<RankedTensorType>(filter.getType());
  auto biasType = cast<RankedTensorType>(bias.getType());
  if (inputType.getRank() != 4 || filterType.getRank() != 4) return failure();
  if (inputType.getElementType() != filterType.getElementType() ||
      filterType.getElementType() != biasType.getElementType()) {
    return rewriter.notifyMatchFailure(
        conv, "input/filter/bias element types differ; augmenting would "
              "need a cast, not implemented");
  }
  if (inputType.isDynamicDim(dims->inputCPos) ||
      filterType.isDynamicDim(dims->filterCPos)) {
    return rewriter.notifyMatchFailure(conv,
                                       "dynamic channel dim is not supported");
  }

  Location loc = conv.getLoc();
  Type elemType = filterType.getElementType();

  // --- Augment the input: one more channel, held at the constant 1.0 across
  // the whole (already-padded) spatial extent. ---
  SmallVector<int64_t> augInputShape(inputType.getShape());
  augInputShape[dims->inputCPos] += 1;
  Value augInputEmpty =
      tensor::EmptyOp::create(rewriter, loc, augInputShape, elemType);
  SmallVector<int64_t> zeroOffsets4(4, 0);
  Value augInput =
      insertFullSlice(rewriter, loc, input, augInputEmpty, zeroOffsets4);

  SmallVector<int64_t> onesChannelShape(augInputShape);
  onesChannelShape[dims->inputCPos] = 1;
  Value onesEmpty =
      tensor::EmptyOp::create(rewriter, loc, onesChannelShape, elemType);
  Value one = arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getFloatAttr(elemType, 1.0));
  Value onesSlice =
      linalg::FillOp::create(rewriter, loc, one, onesEmpty).getResult(0);
  SmallVector<int64_t> onesOffsets4(4, 0);
  onesOffsets4[dims->inputCPos] = inputType.getShape()[dims->inputCPos];
  augInput = insertFullSlice(rewriter, loc, onesSlice, augInput, onesOffsets4);

  // --- Augment the filter: one more input-channel plane, zero everywhere
  // except a single (kh=0, kw=0) tap (whichever operand-dim positions those
  // happen to be -- unresolved here, see below) set to `bias`. ---
  SmallVector<int64_t> augFilterShape(filterType.getShape());
  augFilterShape[dims->filterCPos] += 1;
  Value augFilterEmpty =
      tensor::EmptyOp::create(rewriter, loc, augFilterShape, elemType);
  Value augFilter =
      insertFullSlice(rewriter, loc, filter, augFilterEmpty, zeroOffsets4);

  SmallVector<int64_t> tapPlaneShape(augFilterShape);
  tapPlaneShape[dims->filterCPos] = 1;
  Value zeroF =
      arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(elemType));
  Value tapPlaneEmpty =
      tensor::EmptyOp::create(rewriter, loc, tapPlaneShape, elemType);
  Value tapPlane =
      linalg::FillOp::create(rewriter, loc, zeroF, tapPlaneEmpty).getResult(0);

  // `biasTapShape` collapses every dim of the tap plane to 1 except F's own
  // position (kept at full count) -- so inserting it at an all-zero offset
  // lands `bias` at *whichever* single position the plane's kh/kw dims call
  // "0", i.e. the (kh=0, kw=0) tap, without this code ever naming kh/kw
  // explicitly (their operand-dim positions are never resolved above).
  SmallVector<int64_t> biasTapShape(tapPlaneShape.size(), 1);
  biasTapShape[dims->filterFPos] = filterType.getShape()[dims->filterFPos];
  auto biasTapType = RankedTensorType::get(biasTapShape, elemType);
  SmallVector<ReassociationIndices> reassoc = {{0, 1, 2, 3}};
  Value biasReshaped = tensor::ExpandShapeOp::create(
      rewriter, loc, biasTapType, bias, reassoc);
  SmallVector<int64_t> biasTapOffsets4(4, 0);
  tapPlane =
      insertFullSlice(rewriter, loc, biasReshaped, tapPlane, biasTapOffsets4);

  SmallVector<int64_t> tapOffsets4(4, 0);
  tapOffsets4[dims->filterCPos] = filterType.getShape()[dims->filterCPos];
  augFilter = insertFullSlice(rewriter, loc, tapPlane, augFilter, tapOffsets4);

  // --- Replace the conv: same output shape, C+1 reduction, plain zero-fill
  // accumulator (no bias broadcast/transpose survives). ---
  auto outputType = cast<RankedTensorType>(conv->getResult(0).getType());
  Value newEmpty = tensor::EmptyOp::create(
      rewriter, loc,
      tensor::getMixedSizes(rewriter, loc, conv->getResult(0)),
      outputType.getElementType());
  Value zeroOut = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(outputType.getElementType()));
  Value zeroFill =
      linalg::FillOp::create(rewriter, loc, zeroOut, newEmpty).getResult(0);

  Attribute strides = conv->getAttr("strides");
  Attribute dilations = conv->getAttr("dilations");
  Operation *newConv =
      llvm::TypeSwitch<Operation *, Operation *>(conv.getOperation())
          .Case<linalg::Conv2DNchwFchwOp>([&](auto) {
            return linalg::Conv2DNchwFchwOp::create(
                       rewriter, loc, TypeRange{outputType},
                       ValueRange{augInput, augFilter}, ValueRange{zeroFill},
                       strides, dilations)
                .getOperation();
          })
          .Case<linalg::Conv2DNhwcHwcfOp>([&](auto) {
            return linalg::Conv2DNhwcHwcfOp::create(
                       rewriter, loc, TypeRange{outputType},
                       ValueRange{augInput, augFilter}, ValueRange{zeroFill},
                       strides, dilations)
                .getOperation();
          })
          .Default([](auto) { return nullptr; });
  if (!newConv) return failure();  // unreachable: isa-checked above

  rewriter.replaceOp(conv, newConv->getResult(0));
  for (Operation *dead : deadOps)
    if (dead->use_empty()) rewriter.eraseOp(dead);
  return success();
}

/// Attempts the K+1 augmentation fold on a single matmul. Returns failure
/// (leaving the IR untouched) if the accumulator is not a bias broadcast in
/// the exact shape handled, so unrelated matmuls (including plain
/// `fill(0)`-initialized ones) are never touched.
static LogicalResult foldBiasIntoMatmul(RewriterBase &rewriter,
                                        linalg::MatmulOp matmul) {
  if (!matmul.hasPureTensorSemantics()) return failure();
  if (matmul.getNumDpsInits() != 1) return failure();

  // Two source shapes are handled, tried in turn: (a) bias-as-accumulator
  // (`matmul outs(bias_broadcast)`, what the im2col conv path and
  // `DetachElementwiseFromNamedOpsPass` deal with) and (b) bias-as-post-add
  // (`matmul` immediately followed by a separate `add` generic, what a direct
  // "y = x @ W + b" lowering -- e.g. onnx.Gemm via torch-to-linalg -- produces
  // without any detach pass involved). Both reduce to the same K+1
  // augmentation below; they only differ in which op's result gets replaced
  // and which extra op(s) become dead afterwards.
  Operation *resultOp = matmul.getOperation();
  Value bias;
  unsigned biasResultDim = 0;
  Operation *biasGeneric = nullptr;  // accumulator form: erase if now dead.
  bool postAdd = false;

  Value initOperand = matmul.getDpsInits()[0];
  if (auto candidate = initOperand.getDefiningOp<linalg::GenericOp>()) {
    bias = matchBiasBroadcast(candidate, biasResultDim);
    if (bias) biasGeneric = candidate.getOperation();
  }
  if (!bias) {
    linalg::GenericOp postAddGeneric;
    if (succeeded(
            matchPostAddBias(matmul, postAddGeneric, bias, biasResultDim))) {
      resultOp = postAddGeneric.getOperation();
      postAdd = true;
    }
  }
  if (!bias) return failure();

  SmallVector<AffineMap> maps = matmul.getIndexingMapsArray();
  AffineMap lhsMap = maps[0], rhsMap = maps[1], outMap = maps[2];

  // The bias broadcasts into `initOperand` dim `biasResultDim`; resolve which
  // loop dim that tensor position stands for, and require it to be N (bias
  // is per-output-column, the ordinary case -- a per-row bias would need a
  // different augmentation on the M side and is not handled here).
  if (biasResultDim >= outMap.getNumResults()) return failure();
  auto outDimExpr = dyn_cast<AffineDimExpr>(outMap.getResult(biasResultDim));
  if (!outDimExpr || outDimExpr.getPosition() != 1 /* N */) {
    return rewriter.notifyMatchFailure(
        matmul, "bias broadcast is not over the N (output-column) dim");
  }

  auto lhsKPos = operandDimForLoop(lhsMap, /*loopDim=*/2);
  auto lhsMPos = operandDimForLoop(lhsMap, /*loopDim=*/0);
  auto rhsKPos = operandDimForLoop(rhsMap, /*loopDim=*/2);
  auto rhsNPos = operandDimForLoop(rhsMap, /*loopDim=*/1);
  if (!lhsKPos || !lhsMPos || !rhsKPos || !rhsNPos) {
    return rewriter.notifyMatchFailure(
        matmul, "could not resolve M/N/K positions from indexing maps "
                "(batched matmul is not supported)");
  }

  Value lhs = matmul.getDpsInputs()[0];
  Value rhs = matmul.getDpsInputs()[1];
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  auto biasType = cast<RankedTensorType>(bias.getType());
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
    return rewriter.notifyMatchFailure(matmul,
                                       "only rank-2 (non-batched) operands "
                                       "are supported");
  }
  if (lhsType.getElementType() != rhsType.getElementType() ||
      rhsType.getElementType() != biasType.getElementType()) {
    return rewriter.notifyMatchFailure(
        matmul, "lhs/rhs/bias element types differ; augmenting would need a "
                "cast, not implemented");
  }
  if (rhsType.isDynamicDim(*rhsKPos) || lhsType.isDynamicDim(*lhsKPos)) {
    return rewriter.notifyMatchFailure(matmul,
                                       "dynamic K dim is not supported");
  }

  Location loc = matmul.getLoc();
  Type elemType = rhsType.getElementType();
  int64_t k = rhsType.getShape()[*rhsKPos];

  // --- Augment rhs: [K,N] (or [N,K]) -> one more row/col along K, holding
  // bias at the new K index. ---
  SmallVector<int64_t> augRhsShape(rhsType.getShape());
  augRhsShape[*rhsKPos] += 1;
  Value augRhsEmpty =
      tensor::EmptyOp::create(rewriter, loc, augRhsShape, elemType);
  SmallVector<int64_t> zeroOffsets(2, 0);
  Value augRhs = insertFullSlice(rewriter, loc, rhs, augRhsEmpty, zeroOffsets);

  SmallVector<int64_t> biasSliceShape(augRhsShape);
  biasSliceShape[*rhsKPos] = 1;
  auto biasSliceType = RankedTensorType::get(biasSliceShape, elemType);
  SmallVector<ReassociationIndices> reassoc = {{0, 1}};
  Value biasReshaped = tensor::ExpandShapeOp::create(
      rewriter, loc, biasSliceType, bias, reassoc);
  SmallVector<int64_t> biasSliceOffsets(2, 0);
  biasSliceOffsets[*rhsKPos] = k;
  augRhs = insertFullSlice(rewriter, loc, biasReshaped, augRhs,
                           biasSliceOffsets);

  // --- Augment lhs: one more row/col along K, holding the constant 1.0 at
  // the new K index (so the extra rhs row is picked up exactly once, added
  // unscaled, for every M row). ---
  SmallVector<int64_t> augLhsShape(lhsType.getShape());
  augLhsShape[*lhsKPos] += 1;
  Value augLhsEmpty =
      tensor::EmptyOp::create(rewriter, loc, augLhsShape, elemType);
  Value augLhs = insertFullSlice(rewriter, loc, lhs, augLhsEmpty, zeroOffsets);

  SmallVector<int64_t> onesSliceShape(augLhsShape);
  onesSliceShape[*lhsKPos] = 1;
  Value onesEmpty =
      tensor::EmptyOp::create(rewriter, loc, onesSliceShape, elemType);
  Value one = arith::ConstantOp::create(rewriter, loc,
                                        rewriter.getFloatAttr(elemType, 1.0));
  Value onesSlice =
      linalg::FillOp::create(rewriter, loc, one, onesEmpty).getResult(0);
  SmallVector<int64_t> onesOffsets(2, 0);
  onesOffsets[*lhsMPos] = 0;
  onesOffsets[*lhsKPos] = lhsType.getShape()[*lhsKPos];
  augLhs = insertFullSlice(rewriter, loc, onesSlice, augLhs, onesOffsets);

  // --- Replace the matched result (the matmul itself for the accumulator
  // form, or the bias-add generic for the post-add form): same M/N output
  // shape, K+1 reduction, plain zero-fill accumulator (no bias operand or
  // bias-add survives). ---
  auto outputType = cast<RankedTensorType>(resultOp->getResult(0).getType());
  Value newEmpty = tensor::EmptyOp::create(
      rewriter, loc, tensor::getMixedSizes(rewriter, loc, resultOp->getResult(0)),
      outputType.getElementType());
  Value zero = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(outputType.getElementType()));
  Value zeroFill =
      linalg::FillOp::create(rewriter, loc, zero, newEmpty).getResult(0);

  NamedAttribute indexingMapsAttr(matmul.getIndexingMapsAttrName(),
                                  matmul.getIndexingMapsAttr());
  auto newMatmul = linalg::MatmulOp::create(
      rewriter, loc, TypeRange{outputType}, ValueRange{augLhs, augRhs},
      ValueRange{zeroFill}, ArrayRef<NamedAttribute>{indexingMapsAttr});
  rewriter.replaceOp(resultOp, newMatmul.getResult(0));
  if (biasGeneric && biasGeneric->use_empty()) rewriter.eraseOp(biasGeneric);
  if (postAdd && matmul->use_empty()) rewriter.eraseOp(matmul);
  return success();
}

// See file header for the motivation and the K+1 augmentation identity this
// pass rewrites into (`foldBiasIntoMatmul` for `linalg.matmul`,
// `foldBiasIntoConv2d` for `linalg.conv_2d_{nchw_fchw,nhwc_hwcf}`).
struct AMDAIEFoldBiasIntoContractionPass
    : public impl::AMDAIEFoldBiasIntoContractionBase<
          AMDAIEFoldBiasIntoContractionPass> {
 public:
  using AMDAIEFoldBiasIntoContractionBase::AMDAIEFoldBiasIntoContractionBase;

  void runOnOperation() override {
    SmallVector<linalg::MatmulOp> matmuls;
    SmallVector<linalg::LinalgOp> convs;
    getOperation()->walk([&](Operation *op) {
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        matmuls.push_back(matmul);
      } else if (foldConvBias &&
                 isa<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>(op)) {
        convs.push_back(cast<linalg::LinalgOp>(op));
      }
    });
    IRRewriter rewriter(&getContext());
    for (linalg::MatmulOp matmul : matmuls) {
      rewriter.setInsertionPoint(matmul);
      (void)foldBiasIntoMatmul(rewriter, matmul);
    }
    for (linalg::LinalgOp conv : convs) {
      rewriter.setInsertionPoint(conv);
      (void)foldBiasIntoConv2d(rewriter, conv);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEFoldBiasIntoContractionPass(
    AMDAIEFoldBiasIntoContractionOptions options) {
  return std::make_unique<AMDAIEFoldBiasIntoContractionPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
