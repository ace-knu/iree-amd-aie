// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Spreads the native conv dispatches of the NPU (`native_conv` in the target
// config) over the array columns by splitting their output rows into chunks.
// A batch-1 conv reaches the backend with its batch folded away (6 loops on an
// [H, W, IC] input), which the conv tile pipeline can only spread over output
// channels, i.e. one column: the memtile split cuts a shared-memory buffer
// along its first dim, and neighboring output rows share input rows (the
// window's halo), so the input cannot be cut by row in place.
//
// The rewrite makes the chunks explicit at the Flow dispatch boundary, as
// `PoolingNhwcMaxHaloSplit` does for an overlapping pooling before dispatch
// formation:
//   * input: a host dispatch gathers N chunks of S + KH - 1 input rows, chunk
//     n starting at row n * S, into [N, S + KH - 1, W, IC], reading zero rows
//     past the input when N * S > OH (see below). When the input is
//     the result of a host pad dispatch that only grew the channels (the ones
//     channels of AMDAIEFoldBiasIntoContractionDispatches), the gather reads
//     the pad's source and writes into the pad's splat instead, so the layer
//     keeps one host dispatch on its input;
//   * filter: a constant HWCF filter becomes an FHWC constant, the layout the
//     memtile split cuts by output channel, so output channels can go to the
//     array rows next to chunks on the columns (see `ConvTiling`);
//   * executable: the conv gets the chunk loop in front (7 loops) and writes
//     [N, S, OW, OC] to a new writeonly binding, its elementwise epilogue (the
//     ReLU the bias fold leaves) carried along;
//   * output: the host reads the chunks as [N * S, OW, OC] through a
//     `flow.tensor.reshape`, and its first OH rows through a
//     `flow.tensor.slice` when N * S > OH. A conv that wrote into the interior
//     of a larger
//     tensor (the next layer's padded input, as a tied result) writes its own
//     result now, and a host dispatch tied to the larger tensor copies it in,
//     the form IREE gives a chunked conv written into a padded buffer. Inside
//     the executable, that store would collapse the chunks back into rows of
//     a strided interior, which the conv pipeline cannot take (no memtile DMA
//     channel is left for the output).
// Every column computes its chunk at once, so a layer takes as long as one
// chunk: S is the fewest rows per chunk the columns allow, at least two (once
// IREE folds a unit row dim away, the conv is no longer one the pipeline
// recognizes), and N = ceil(OH / S) the fewest chunks that cover OH. When no
// divisor of OH gives that S, N * S passes OH: the extra rows read zeros and
// are cropped (OH = 7 on 8 columns: 4 chunks of 2 rows, the last one half
// padding, instead of 1 chunk of 7). When a divisor does, N is that divisor
// and nothing is padded (OH = 28: 7 chunks of 4; an 8th column would only take
// padding rows). A conv with fewer rows stays on one column but still gets the
// FHWC filter, so the column's rows can take output-channel tiles; a dispatch
// of another shape is left as is.
// Runs right after AMDAIEFoldBiasIntoContractionDispatches.

#include "iree-amd-aie/Transforms/AMDAIEDispatchRewriteUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-amdaie-chunk-conv-dispatches"

namespace mlir::iree_compiler::AMDAIE {

namespace {

using IREE::TensorExt::DispatchTensorLoadOp;
using IREE::TensorExt::DispatchTensorStoreOp;
using IREE::TensorExt::DispatchTensorType;
using IREE::TensorExt::TensorAccess;

/// A native conv executable the pass can chunk: a 6-loop conv (no chunk dim)
/// reading its input and filter from full-tensor binding loads and starting
/// from a fill, followed by at most one unary elementwise generic (none, or the
/// ReLU the bias fold leaves; the conv pipeline takes no chain) whose value is
/// stored into the output binding, whole or an interior of it. The func has no
/// other arguments and nothing else but constants and empties.
struct ConvChunkInfo {
  linalg::LinalgOp conv;
  ConvDims dims;
  DispatchTensorLoadOp inLoad, filterLoad;
  BlockArgument inArg, filterArg, outArg;
  linalg::FillOp fill;
  SmallVector<linalg::GenericOp> epilogue;
  DispatchTensorStoreOp store;
  SmallVector<int64_t> storeOffsets;
};

/// Returns true if `op` is an elementwise generic whose one input is `v` and
/// whose one init its payload does not read, with identity maps and no index
/// ops: a chunk dim can be put in front of it and its init replaced by an
/// empty.
static bool isUnaryEpilogueOf(linalg::GenericOp op, Value v) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getDpsInputOperand(0)->get() != v ||
      op.payloadUsesValueFromOperand(op.getDpsInitOperand(0)) ||
      op.getNumParallelLoops() != op.getNumLoops())
    return false;
  if (!llvm::all_of(op.getIndexingMapsArray(),
                    [](AffineMap m) { return m.isIdentity(); }))
    return false;
  return !op.getBody()
              ->walk([](linalg::IndexOp) { return WalkResult::interrupt(); })
              .wasInterrupted();
}

static std::optional<ConvChunkInfo> getConvChunkInfo(func::FuncOp func) {
  SmallVector<linalg::LinalgOp> convs;
  func.walk([&](linalg::LinalgOp op) {
    if (succeeded(getConvDims(op))) convs.push_back(op);
  });
  if (convs.size() != 1) return std::nullopt;
  ConvChunkInfo info;
  info.conv = convs.front();
  info.dims = *getConvDims(info.conv);
  if (info.dims.n || !info.conv.hasPureTensorSemantics()) return std::nullopt;
  info.inLoad = getFullTensorLoad(info.conv.getDpsInputOperand(0)->get());
  info.filterLoad = getFullTensorLoad(info.conv.getDpsInputOperand(1)->get());
  if (!info.inLoad || !info.filterLoad) return std::nullopt;
  info.inArg = dyn_cast<BlockArgument>(info.inLoad.getSource());
  info.filterArg = dyn_cast<BlockArgument>(info.filterLoad.getSource());
  info.fill =
      info.conv.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
  if (!info.inArg || !info.filterArg || !info.fill) return std::nullopt;

  Value v = info.conv->getResult(0);
  if (v.hasOneUse()) {
    auto generic = dyn_cast<linalg::GenericOp>(*v.getUsers().begin());
    if (generic && isUnaryEpilogueOf(generic, v)) {
      info.epilogue.push_back(generic);
      v = generic->getResult(0);
    }
  }
  if (!v.hasOneUse()) return std::nullopt;
  info.store = dyn_cast<DispatchTensorStoreOp>(*v.getUsers().begin());
  if (!info.store || info.store.getValue() != v) return std::nullopt;
  info.outArg = dyn_cast<BlockArgument>(info.store.getTarget());
  if (!info.outArg) return std::nullopt;
  // A static, unit-stride store of the whole value (not rank-reducing).
  for (OpFoldResult s : info.store.getMixedStrides())
    if (!isConstantIntValue(s, 1)) return std::nullopt;
  for (OpFoldResult o : info.store.getMixedOffsets()) {
    std::optional<int64_t> c = getConstantIntValue(o);
    if (!c) return std::nullopt;
    info.storeOffsets.push_back(*c);
  }
  SmallVector<int64_t> sizes;
  for (OpFoldResult s : info.store.getMixedSizes()) {
    std::optional<int64_t> c = getConstantIntValue(s);
    if (!c) return std::nullopt;
    sizes.push_back(*c);
  }
  if (ArrayRef<int64_t>(sizes) !=
      cast<RankedTensorType>(v.getType()).getShape())
    return std::nullopt;

  if (func.getNumArguments() != 3 || info.inArg == info.filterArg ||
      info.outArg == info.inArg || info.outArg == info.filterArg)
    return std::nullopt;
  DenseSet<Operation *> matched = {info.conv,       info.inLoad,
                                   info.filterLoad, info.fill,
                                   info.store};
  for (linalg::GenericOp g : info.epilogue) matched.insert(g);
  for (Operation &op : func.getBody().front()) {
    if (!matched.contains(&op) &&
        !isa<arith::ConstantOp, tensor::EmptyOp, func::ReturnOp>(op))
      return std::nullopt;
  }
  return info;
}

/// Row chunks of a conv with `oh` output rows on `numCols` array columns (see
/// the file comment): the fewest rows per chunk the columns allow, at least
/// two, and the fewest chunks of that many rows that cover `oh`. Their rows
/// can pass `oh`; one chunk means no chunks.
struct ConvChunks {
  int64_t numChunks;
  int64_t rowsPerChunk;
};
static ConvChunks chooseConvChunks(int64_t oh, int64_t numCols) {
  numCols = std::max<int64_t>(numCols, 1);
  int64_t rows = std::max<int64_t>(2, (oh + numCols - 1) / numCols);
  return {(oh + rows - 1) / rows, rows};
}

/// Returns HWCF filter `filter` ([KH, KW, IC, OC]) as FHWC ([OC, KH, KW, IC]).
/// Null if its elements are not byte-sized.
static DenseElementsAttr transposeFilterToFHWC(DenseElementsAttr filter) {
  auto type = cast<RankedTensorType>(filter.getType());
  Type elemType = type.getElementType();
  if (!elemType.isIntOrFloat() || elemType.getIntOrFloatBitWidth() % 8 != 0)
    return {};
  ArrayRef<int64_t> s = type.getShape();
  int64_t KH = s[0], KW = s[1], IC = s[2], OC = s[3];
  auto dstType = RankedTensorType::get({OC, KH, KW, IC}, elemType);
  if (filter.isSplat()) return filter.reshape(dstType);
  int64_t eb = elemType.getIntOrFloatBitWidth() / 8;
  ArrayRef<char> src = filter.getRawData();
  SmallVector<char> dst(src.size());
  for (int64_t kh = 0; kh < KH; ++kh)
    for (int64_t kw = 0; kw < KW; ++kw)
      for (int64_t ic = 0; ic < IC; ++ic)
        for (int64_t oc = 0; oc < OC; ++oc) {
          int64_t from = ((kh * KW + kw) * IC + ic) * OC + oc;
          int64_t to = ((oc * KH + kh) * KW + kw) * IC + ic;
          std::copy_n(src.begin() + from * eb, eb, dst.begin() + to * eb);
        }
  return DenseElementsAttr::getFromRawBuffer(dstType, dst);
}

/// When `v` is the result of a host pad dispatch (`createPaddingDispatch`) that
/// grew only the last dim and that nothing else reads, returns that dispatch
/// and sets `padValue` to the value of its splat.
static IREE::Flow::DispatchOp getChannelPadDispatch(Value v, ModuleOp module,
                                                    TypedAttr &padValue) {
  auto pad = v.getDefiningOp<IREE::Flow::DispatchOp>();
  if (!pad || !v.hasOneUse() || pad.getNumResults() != 1 ||
      pad.getArguments().size() != 2 || !pad.getWorkload().empty() ||
      pad.getTiedResultOperand(pad.getResult(0)) != pad.getArguments()[1])
    return {};
  auto splat = pad.getArguments()[1].getDefiningOp<IREE::Flow::TensorSplatOp>();
  if (!splat || !matchPattern(splat.getValue(), m_Constant(&padValue)))
    return {};
  auto srcType = dyn_cast<RankedTensorType>(pad.getArguments()[0].getType());
  auto dstType = cast<RankedTensorType>(v.getType());
  if (!srcType || srcType.getRank() != dstType.getRank() ||
      srcType.getShape().drop_back() != dstType.getShape().drop_back())
    return {};
  // Worker func: a full load of the source stored at the low corner.
  func::FuncOp func = getDispatchFunc(pad, module);
  if (!func || func.getNumArguments() != 2) return {};
  Block &body = func.getBody().front();
  if (body.getOperations().size() != 3) return {};
  auto load = dyn_cast<DispatchTensorLoadOp>(body.front());
  auto store = dyn_cast<DispatchTensorStoreOp>(*std::next(body.begin()));
  if (!load || !store || getFullTensorLoad(load.getResult()) != load ||
      load.getSource() != func.getArgument(0) ||
      store.getTarget() != func.getArgument(1) ||
      store.getValue() != load.getResult())
    return {};
  for (OpFoldResult o : store.getMixedOffsets())
    if (!isConstantIntValue(o, 0)) return {};
  for (OpFoldResult s : store.getMixedStrides())
    if (!isConstantIntValue(s, 1)) return {};
  return pad;
}

/// Creates a host dispatch (`halo_executable_<i>`) gathering `numChunks`
/// chunks of `rows` rows of `src` ([H, W, C0]), chunk n starting at row
/// n * `rowStride`, into [numChunks, rows, W, `channels`]. Rows past H, up to
/// `padRows` of them, read zeros. With `channels` > C0 the chunks are written
/// into a splat of `padValue`, which the channels after C0 keep.
static Value createHaloSplitDispatch(IRRewriter &rewriter, ModuleOp module,
                                     Value src, int64_t numChunks,
                                     int64_t rows, int64_t rowStride,
                                     int64_t padRows, int64_t channels,
                                     TypedAttr padValue,
                                     IREE::HAL::DeviceAffinityAttr hostAffinity,
                                     int &counter) {
  Location loc = src.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  auto srcType = cast<RankedTensorType>(src.getType());
  Type elemType = srcType.getElementType();
  int64_t width = srcType.getDimSize(1);
  auto gatherType = RankedTensorType::get(
      {numChunks, rows, width, srcType.getDimSize(2)}, elemType);
  auto dstType =
      RankedTensorType::get({numChunks, rows, width, channels}, elemType);
  bool intoSplat = channels != srcType.getDimSize(2);
  auto inBinding = DispatchTensorType::get(TensorAccess::ReadOnly, srcType);
  auto outBinding = DispatchTensorType::get(
      intoSplat ? TensorAccess::ReadWrite : TensorAccess::WriteOnly, dstType);
  std::string idx = std::to_string(counter++);

  // Worker func: gather the chunks with an `n * rowStride + r` row map and
  // store them at the low corner of the output.
  auto funcOp = func::FuncOp::create(
      loc, "halo_dispatch_" + idx,
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value loaded =
      DispatchTensorLoadOp::create(fb, loc, srcType, entry->getArgument(0),
                                   /*sourceDynamicDims=*/ValueRange{});
  if (padRows > 0) {
    auto paddedType = RankedTensorType::get(
        {srcType.getDimSize(0) + padRows, width, srcType.getDimSize(2)},
        elemType);
    Value zero = arith::ConstantOp::create(fb, loc, fb.getZeroAttr(elemType));
    loaded = tensor::createPadHighOp(paddedType, loaded, zero,
                                     /*nofold=*/false, loc, fb);
  }
  Value empty =
      tensor::EmptyOp::create(fb, loc, gatherType.getShape(), elemType);
  AffineExpr n, r, w, c;
  bindDims(ctx, n, r, w, c);
  SmallVector<AffineMap> maps = {
      AffineMap::get(4, 0, {n * rowStride + r, w, c}, ctx),
      AffineMap::getMultiDimIdentityMap(4, ctx)};
  auto gather = linalg::GenericOp::create(
      fb, loc, TypeRange{gatherType}, ValueRange{loaded}, ValueRange{empty},
      maps,
      SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
      [](OpBuilder &b, Location l, ValueRange args) {
        linalg::YieldOp::create(b, l, args[0]);
      });
  SmallVector<OpFoldResult> offsets(4, fb.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(4, fb.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes;
  for (int64_t d : gatherType.getShape()) sizes.push_back(fb.getIndexAttr(d));
  DispatchTensorStoreOp::create(fb, loc, gather.getResult(0),
                                entry->getArgument(1),
                                /*targetDynamicDims=*/ValueRange{}, offsets,
                                sizes, strides);
  func::ReturnOp::create(fb, loc);
  IREE::Flow::ExecutableExportOp exportOp =
      createExecutable(module, funcOp, "halo_executable_" + idx);

  SmallVector<Value> args = {src};
  ArrayAttr tied;
  if (intoSplat) {
    Value scalar = arith::ConstantOp::create(rewriter, loc, padValue);
    auto splatOp = IREE::Flow::TensorSplatOp::create(
        rewriter, loc, dstType, scalar, /*result_dims=*/ValueRange{});
    splatOp->setAttr("stream.affinity", hostAffinity);
    args.push_back(splatOp.getResult());
    tied = rewriter.getIndexArrayAttr({1});
  }
  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{}, args, /*argument_dims=*/ValueRange{}, tied);
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}

/// Rewrites the executable func of `info` to the chunked conv: the input
/// binding becomes [numChunks, S + KH - 1, W, IC] (S = `rowsPerChunk`), the
/// filter binding FHWC when `toFHWC`, and the output a new writeonly
/// [numChunks, S, OW, OC] binding appended after the others (the old output
/// binding goes away).
static void chunkExecutable(IRRewriter &rewriter, func::FuncOp func,
                            ConvChunkInfo &info, int64_t numChunks,
                            int64_t rowsPerChunk, bool toFHWC) {
  MLIRContext *ctx = func.getContext();
  Location loc = info.conv.getLoc();
  SmallVector<int64_t> bounds = info.conv.getStaticLoopRanges();
  int64_t ow = bounds[info.dims.ow];
  int64_t oc = bounds[info.dims.oc], kh = bounds[info.dims.kh];
  int64_t kw = bounds[info.dims.kw], ic = bounds[info.dims.ic];
  auto inType = cast<RankedTensorType>(info.inLoad.getType());
  growBinding(rewriter, info.inArg, info.inLoad,
              {numChunks, rowsPerChunk + kh - 1, inType.getDimSize(1),
               inType.getDimSize(2)});
  if (toFHWC)
    growBinding(rewriter, info.filterArg, info.filterLoad, {oc, kh, kw, ic});

  // Loops (n, oh, ow, oc, kh, kw, ic).
  rewriter.setInsertionPoint(info.store);
  SmallVector<int64_t> outShape = {numChunks, rowsPerChunk, ow, oc};
  auto newEmpty = [&](Value like) {
    return tensor::EmptyOp::create(
        rewriter, loc, outShape,
        cast<ShapedType>(like.getType()).getElementType());
  };
  Value fill =
      linalg::FillOp::create(rewriter, loc, info.fill.getInputs(),
                             ValueRange{newEmpty(info.conv->getResult(0))})
          .getResult(0);
  AffineExpr dN, dOH, dOW, dOC, dKH, dKW, dIC;
  bindDims(ctx, dN, dOH, dOW, dOC, dKH, dKW, dIC);
  auto map = [&](ArrayRef<AffineExpr> results) {
    return AffineMap::get(7, 0, results, ctx);
  };
  SmallVector<AffineMap> maps = {
      map({dN, dOH + dKH, dOW + dKW, dIC}),
      toFHWC || info.dims.filterOcFirst ? map({dOC, dKH, dKW, dIC})
                                        : map({dKH, dKW, dIC, dOC}),
      map({dN, dOH, dOW, dOC})};
  SmallVector<utils::IteratorType> iterators(4, utils::IteratorType::parallel);
  iterators.append(3, utils::IteratorType::reduction);
  auto conv = linalg::GenericOp::create(
      rewriter, loc, TypeRange{fill.getType()},
      ValueRange{info.conv.getDpsInputOperand(0)->get(),
                 info.conv.getDpsInputOperand(1)->get()},
      ValueRange{fill}, maps, iterators);
  IRMapping convMapping;
  info.conv->getRegion(0).cloneInto(&conv.getRegion(), convMapping);
  Value value = conv.getResult(0);
  AffineMap identity = AffineMap::getMultiDimIdentityMap(4, ctx);
  for (linalg::GenericOp old : info.epilogue) {
    Value init = newEmpty(old->getResult(0));
    auto generic = linalg::GenericOp::create(
        rewriter, old.getLoc(), TypeRange{init.getType()}, ValueRange{value},
        ValueRange{init}, ArrayRef<AffineMap>{identity, identity},
        SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel));
    IRMapping mapping;
    old.getRegion().cloneInto(&generic.getRegion(), mapping);
    value = generic.getResult(0);
  }
  auto outType = cast<RankedTensorType>(value.getType());
  unsigned oldOutArgNo = info.outArg.getArgNumber();
  (void)func.insertArgument(
      func.getNumArguments(),
      DispatchTensorType::get(TensorAccess::WriteOnly, outType),
      DictionaryAttr(), loc);
  DispatchTensorStoreOp::create(rewriter, loc, value,
                                func.getArguments().back(),
                                /*targetDynamicDims=*/ValueRange{});

  // The old chain, its empties and the old output binding go away.
  rewriter.eraseOp(info.store);
  for (linalg::GenericOp old : llvm::reverse(info.epilogue))
    rewriter.eraseOp(old);
  rewriter.eraseOp(info.conv);
  rewriter.eraseOp(info.fill);
  for (Operation &op : llvm::make_early_inc_range(func.getBody().front()))
    if (isa<tensor::EmptyOp>(op) && op.use_empty()) rewriter.eraseOp(&op);
  (void)func.eraseArgument(oldOutArgNo);
  SmallVector<Type> argTypes(llvm::map_range(
      func.getArguments(), [](BlockArgument a) { return a.getType(); }));
  func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
}

/// Gives the unchunked conv of `info` (a generic) an FHWC copy of its constant
/// HWCF filter: a new global as the dispatch operand, and in the executable the
/// filter binding and the conv's filter map (oc, kh, kw, ic). The conv then
/// keeps one column whose rows can take output-channel tiles, each a
/// contiguous slice of the filter. Fails, changing nothing, if the filter is
/// not a constant.
static LogicalResult useFHWCFilter(
    IRRewriter &rewriter, ModuleOp module, IREE::Flow::DispatchOp dispatch,
    func::FuncOp func, ConvChunkInfo &info, int &counter,
    SmallVectorImpl<IREE::Util::GlobalOp> &replacedGlobals) {
  auto conv = dyn_cast<linalg::GenericOp>(info.conv.getOperation());
  if (!conv) return failure();
  unsigned filterArgNo = info.filterArg.getArgNumber();
  Value filter = dispatch.getArguments()[filterArgNo];
  IREE::Util::GlobalOp filterGlobal;
  DenseElementsAttr attr = getConstantValue(filter, &filterGlobal);
  DenseElementsAttr fhwc =
      attr ? transposeFilterToFHWC(attr) : DenseElementsAttr();
  if (!fhwc) return failure();

  rewriter.setInsertionPoint(dispatch);
  Value newFilter = createReplacementConstant(
      rewriter, module, dispatch.getLoc(),
      "__conv_fhwc_weight_" + std::to_string(counter++), fhwc, filterGlobal,
      replacedGlobals);
  SmallVector<int64_t> bounds = conv.getStaticLoopRanges();
  growBinding(rewriter, info.filterArg, info.filterLoad,
              {bounds[info.dims.oc], bounds[info.dims.kh],
               bounds[info.dims.kw], bounds[info.dims.ic]});
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<AffineMap> maps = conv.getIndexingMapsArray();
  maps[1] = AffineMap::get(conv.getNumLoops(), 0,
                           {getAffineDimExpr(info.dims.oc, ctx),
                            getAffineDimExpr(info.dims.kh, ctx),
                            getAffineDimExpr(info.dims.kw, ctx),
                            getAffineDimExpr(info.dims.ic, ctx)},
                           ctx);
  conv.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
  SmallVector<Type> argTypes(llvm::map_range(
      func.getArguments(), [](BlockArgument a) { return a.getType(); }));
  func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
  dispatch.getArgumentsMutable().slice(filterArgNo, 1).assign(newFilter);
  eraseDeadProducers(rewriter, filter);
  return success();
}

class AMDAIEChunkConvDispatchesPass
    : public impl::AMDAIEChunkConvDispatchesBase<
          AMDAIEChunkConvDispatchesPass> {
 public:
  void runOnOperation() override;
};

void AMDAIEChunkConvDispatchesPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());

  SmallVector<IREE::Flow::DispatchOp> dispatches;
  DenseMap<Operation *, int> numCallers;
  module.walk([&](IREE::Flow::DispatchOp d) {
    dispatches.push_back(d);
    if (func::FuncOp func = getDispatchFunc(d, module))
      ++numCallers[func.getOperation()];
  });

  IREE::HAL::DeviceAffinityAttr hostAffinity = getHostAffinity(module);
  if (!hostAffinity) return;  // need a host device for the gather dispatch
  int counter = firstFreeHelperIndex(module);
  SmallVector<IREE::Util::GlobalOp> replacedGlobals;

  for (IREE::Flow::DispatchOp dispatch : dispatches) {
    IREE::HAL::ExecutableTargetAttr target =
        getDispatchTarget(dispatch, module);
    if (!target || !getConfigNativeConv(target)) continue;
    func::FuncOp func = getDispatchFunc(dispatch, module);
    if (!func || numCallers[func.getOperation()] != 1) continue;
    std::optional<ConvChunkInfo> info = getConvChunkInfo(func);
    if (!info) continue;
    SmallVector<int64_t> bounds = info->conv.getStaticLoopRanges();
    if (llvm::any_of(bounds, ShapedType::isDynamic)) continue;
    int64_t oh = bounds[info->dims.oh], ow = bounds[info->dims.ow];
    int64_t oc = bounds[info->dims.oc], kh = bounds[info->dims.kh];
    ConvChunks chunks =
        chooseConvChunks(oh, getConfigNumColumns(target).value_or(1));
    int64_t numChunks = chunks.numChunks, rowsPerChunk = chunks.rowsPerChunk;
    int64_t padRows = numChunks * rowsPerChunk - oh;
    if (numChunks < 2) {
      // Too few rows for chunks: the conv stays on one column, whose rows can
      // only take output-channel tiles of an FHWC filter (see `ConvTiling`).
      if (!info->dims.filterOcFirst)
        (void)useFHWCFilter(rewriter, module, dispatch, func, *info, counter,
                            replacedGlobals);
      continue;
    }

    // Dispatch operands: the input, the filter and, for a result tied to the
    // tensor it is written into, that tensor (the output binding).
    if (dispatch.getNumResults() != 1 || !dispatch.getWorkload().empty())
      continue;
    Value result = dispatch.getResult(0);
    Value tiedDest = dispatch.getTiedResultOperand(result);
    unsigned numArgs = dispatch.getArguments().size();
    unsigned inArgNo = info->inArg.getArgNumber();
    unsigned filterArgNo = info->filterArg.getArgNumber();
    unsigned outArgNo = info->outArg.getArgNumber();
    if (tiedDest ? numArgs != 3 || dispatch.getArguments()[outArgNo] != tiedDest
                 : numArgs != 2 || outArgNo != 2)
      continue;
    auto valueType = cast<RankedTensorType>(info->store.getValue().getType());
    if (!tiedDest &&
        cast<RankedTensorType>(result.getType()).getShape() !=
            valueType.getShape())
      continue;  // a partial write of a writeonly result

    // Host: the chunked input, merged with a ones-channel pad, and the FHWC
    // filter.
    rewriter.setInsertionPoint(dispatch);
    Value in = dispatch.getArguments()[inArgNo];
    TypedAttr padValue;
    IREE::Flow::DispatchOp padDispatch =
        getChannelPadDispatch(in, module, padValue);
    Value src = padDispatch ? padDispatch.getArguments()[0] : in;
    Value chunked = createHaloSplitDispatch(
        rewriter, module, src, numChunks, rowsPerChunk + kh - 1, rowsPerChunk,
        padRows, cast<RankedTensorType>(in.getType()).getDimSize(2), padValue,
        hostAffinity, counter);
    Value filter = dispatch.getArguments()[filterArgNo];
    Value newFilter = filter;
    if (!info->dims.filterOcFirst) {
      IREE::Util::GlobalOp filterGlobal;
      DenseElementsAttr fhwc;
      if (DenseElementsAttr attr = getConstantValue(filter, &filterGlobal))
        fhwc = transposeFilterToFHWC(attr);
      if (fhwc)
        newFilter = createReplacementConstant(
            rewriter, module, dispatch.getLoc(),
            "__conv_fhwc_weight_" + std::to_string(counter++), fhwc,
            filterGlobal, replacedGlobals);
    }
    bool toFHWC = newFilter != filter;

    // Executable, then the chunked dispatch read back as rows.
    chunkExecutable(rewriter, func, *info, numChunks, rowsPerChunk, toFHWC);
    rewriter.setInsertionPoint(dispatch);
    SmallVector<Value> args;
    for (auto [i, arg] : llvm::enumerate(dispatch.getArguments())) {
      if (tiedDest && i == outArgNo) continue;
      args.push_back(i == inArgNo       ? chunked
                     : i == filterArgNo ? newFilter
                                        : arg);
    }
    auto exportOp =
        SymbolTable::lookupNearestSymbolFrom<IREE::Flow::ExecutableExportOp>(
            dispatch, llvm::to_vector(dispatch.getEntryPointRefs()).front());
    auto chunkedType = RankedTensorType::get(
        {numChunks, rowsPerChunk, ow, oc}, valueType.getElementType());
    auto newDispatch = IREE::Flow::DispatchOp::create(
        rewriter, dispatch.getLoc(), exportOp, /*workload=*/ValueRange{},
        TypeRange{chunkedType}, /*result_dims=*/ValueRange{}, args,
        /*argument_dims=*/ValueRange{}, /*tied_operands=*/ArrayAttr{});
    newDispatch->setAttr("stream.affinity",
                         dispatch->getAttr("stream.affinity"));
    auto rowsType = RankedTensorType::get({numChunks * rowsPerChunk, ow, oc},
                                          valueType.getElementType());
    Value rows = IREE::Flow::TensorReshapeOp::create(
        rewriter, dispatch.getLoc(), rowsType, newDispatch.getResult(0),
        /*source_dims=*/ValueRange{}, /*result_dims=*/ValueRange{});
    if (padRows > 0) {
      // The first OH rows: a contiguous outer-dim range of the chunks.
      auto index = [&](int64_t v) -> Value {
        return arith::ConstantIndexOp::create(rewriter, dispatch.getLoc(), v);
      };
      Value zero = index(0);
      rows = IREE::Flow::TensorSliceOp::create(
          rewriter, dispatch.getLoc(), valueType, rows,
          /*source_dims=*/ValueRange{},
          /*start_indices=*/ValueRange{zero, zero, zero},
          /*lengths=*/ValueRange{index(oh), index(ow), index(oc)},
          /*result_dims=*/ValueRange{});
    }
    if (tiedDest)
      rows = createInsertDispatch(rewriter, module, rows, tiedDest,
                                  info->storeOffsets, hostAffinity, counter);
    rewriter.replaceOp(dispatch, rows);

    if (padDispatch) {
      auto splat = padDispatch.getArguments()[1]
                       .getDefiningOp<IREE::Flow::TensorSplatOp>();
      auto padExe = module.lookupSymbol<IREE::Flow::ExecutableOp>(
          llvm::to_vector(padDispatch.getEntryPointRefs())
              .front()
              .getRootReference());
      rewriter.eraseOp(padDispatch);
      if (padExe && SymbolTable::symbolKnownUseEmpty(padExe, module))
        rewriter.eraseOp(padExe);
      if (splat.use_empty()) {
        Value scalar = splat.getValue();
        rewriter.eraseOp(splat);
        if (Operation *def = scalar.getDefiningOp())
          if (def->use_empty()) rewriter.eraseOp(def);
      }
    }
    if (toFHWC) eraseDeadProducers(rewriter, filter);
  }

  eraseDeadGlobals(rewriter, module, replacedGlobals);
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEChunkConvDispatchesPass() {
  return std::make_unique<AMDAIEChunkConvDispatchesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
