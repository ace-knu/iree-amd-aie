// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEUTILS_H_

#include <array>

#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler::AMDAIE {

/// Returns the BlockArgument that leads to `val`, if any. Traverses optional
/// ext* ops.
BlockArgument getBlockArgumentWithOptionalExtOps(Value val);

/// Returns the target AMDAIE device.
std::optional<AMDAIEDevice> getConfigAMDAIEDevice(
    IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns the AMDAIE device from an operation. Looks for an executable target
/// attr in the AST.
std::optional<AMDAIEDevice> getConfigAMDAIEDevice(Operation *op);

/// Starting from operation `op`, consider ancestors until a module op
/// containing an AMDAIE device is found. If no such device is found, return an
/// empty optional.
std::optional<AMDAIEDevice> getConfigAMDAIEDeviceFromAncestor(Operation *op);

/// Returns the number of columns being targeted.
std::optional<int64_t> getConfigNumColumns(
    IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns the number of rows being targeted.
std::optional<int64_t> getConfigNumRows(
    IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns the pack-peel L2 reduction (K) tile size. A matmul K dimension must
/// be a multiple of this for the pack-peel codegen to tile it without a partial
/// reduction tile. Single source of truth shared by `KernelDispatch` (which uses
/// it as `k0Pack`) and the K-padding pass (which pads K up to a multiple of it).
/// `kPackScaleL1` selects the pipeline variant (1 = default 2-level tiling,
/// 2 = 4-level tiling).
int64_t getPackPeelReductionTile(int64_t kPackScaleL1);

/// Returns the M/N/K pack size for a matmul whose element types have no AIE
/// vector instruction (e.g. f32). Shared by `KernelDispatch` (which packs by
/// it, or by the largest factor of each dim below it) and the padding pass
/// (which pads M/N/K up to multiples of it).
std::array<uint32_t, 3> getScalarMatmulPackSize();

/// Returns true if the ukernel sources of `device` have a template for the
/// ukernel `ukernelName` with the element types `elemTypes` (operands, then
/// result), and for a transposed B if `transposeB`. An op without a template is
/// left to the non-ukernel path, so this decides whether an op is lowered to a
/// ukernel (`AMDAIELowerToUKernels`) and whether it is tiled and padded for one
/// (`KernelDispatch`, the padding pass). Only the element types are checked
/// here: the shape conditions of a ukernel are checked when lowering to it.
/// TODO: The goal is every element type combination the hardware supports.
/// The table covers the npu4 peano sources (uKernels/npu4/peano). Other devices
/// and ukernels without an entry keep the previous behavior: a template is
/// assumed. When a template is added, add it here.
bool hasUKernelTemplate(AMDAIEDevice device, StringRef ukernelName,
                        ArrayRef<Type> elemTypes, bool transposeB = false);

/// Returns true if a 2-D matmul with `m` rows takes the GEMV pipeline instead
/// of pack-peel. Today that is M == 1 only (a vector times a matrix). Pack-peel
/// zero-pads any M below numRows * instr_m (32 on npu4 bf16) up to it, so
/// 2 <= M < 32 would also profit, but the GEMV lowering's L2 split
/// (`AMDAIESplitLogicalObjFifos`) still splits along the first non-unit dim and
/// for M >= 2 that is M, not N: every core then writes C into one memtile and
/// channel assignment fails. For the same reason the weight must be [N, K]
/// (`transposeB`, K contiguous): a [K, N] weight block is cut along K, so every
/// core reads from one memtile. Those shapes stay on pack-peel until that is
/// fixed. Shared by `KernelDispatch` and the Flow-phase pad pass so they agree
/// on which dispatches are GEMV.
bool isGemvShape(int64_t m, bool transposeB);

/// Whether the Flow pad pass may pad a GEMV's N up to a multiple of
/// `numRows * numCols * kGemvNPerCoreMultiple` so the GEMV pipeline reaches
/// every core (`gemv_pad_n` in the executable target config; absent means
/// enabled).
bool getConfigGemvPadN(IREE::HAL::ExecutableTargetAttr targetAttr);

/// The GEMV pipeline's per-core N tile and the core count a block reaches, for
/// a (1 x K) x (N x K)^T matmul with `kTile` K elements streamed per step.
/// Over the divisors `n1` of N whose double-buffered A/B/C tiles fit
/// `l1Limit`, take the one whose block covers the most cores. Among the
/// candidates that reach every core, the largest n1 that is <= `n1Max`
/// (`gemv_n1` on the target, default 32), so a padded N (a multiple of every
/// core times `kGemvNPerCoreMultiple`) gets the same per-core tile in every
/// dispatch and only the block count varies. Otherwise (an N the pad pass has
/// not grown, e.g. in NPU-only builds) ties go to the largest tile (fewest
/// blocks). A block must fill whole columns or stay in one
/// (`cores <= numRows || cores % numRows == 0`): the L2 weight split hands
/// every memtile one column.
struct GemvNTiling {
  int64_t n1;
  int64_t cores;
};
FailureOr<GemvNTiling> chooseGemvNTiling(int64_t M, int64_t N, int64_t kTile,
                                         int64_t bytesLhs, int64_t bytesRhs,
                                         int64_t bytesOut, int64_t numRows,
                                         int64_t numCols, int64_t l1Limit,
                                         int64_t n1Max);

/// Per-core N tile cap of the GEMV pipeline (`gemv_n1` in the executable
/// target config; absent means 32).
int64_t getConfigGemvN1(IREE::HAL::ExecutableTargetAttr targetAttr);

/// The pool column policy (`pool_columns_policy` in the executable target
/// config, one of "odd" | "cost" | "full" | "max"; absent means `Max`). Read
/// per dispatch by the Flow pooling pad pass so it pads OH exactly when the
/// lowering strategy, running with the same policy, will spread the padded OH
/// over all columns.
PoolColumnsPolicy getConfigPoolColumnsPolicy(
    IREE::HAL::ExecutableTargetAttr targetAttr);
llvm::StringRef stringifyPoolColumnsPolicy(PoolColumnsPolicy policy);

/// Utility to retrieve a constant index from an OpFoldResult.
int64_t getConstantIndexOrAssert(OpFoldResult ofr);

// Return the number of elements in a vector fma/mul/add instruction that AIE
// supports.
FailureOr<uint32_t> getAIEMacNumElements(Type inputType, Type outputType);

/// Utility function that, given an element type, computes tiling scaling factor
/// as : 64/bitWidth.
/// Therefore, for example, if the element type has :-
///     a. 64 bit width -> scaling factor will be 1.
///     b. 32 bit width -> scaling factor will be 2.
///     c. 16 bit width -> scaling factor will be 4.
/// This can easily be scaled to other element data types by just changing the
/// check in the `if` in the function body.
/// The rationale/need for this is - higher bit width element types can be
/// accomodated using small tile sizes, whereas lower bit width element types
/// can make use of larger tile sizes as they can be easily accomodated within
/// the same. Therefore this tiling scale factor helps dynamically
/// increase/decrease the tiling window depending on the element type's bit
/// width.
FailureOr<unsigned> getTilingScaleFactor(Type elemType);

/// Utility to identify whether a linalg op is a matmul op.
bool isMatmul(linalg::LinalgOp linalgOp);

/// Utility to identify whether a linalg op is a matmul_transpose_a op.
bool isMatmulTransposeA(linalg::LinalgOp linalgOp);

/// Utility to identify whether a linalg op is a matmul_transpose_b op.
bool isMatmulTransposeB(linalg::LinalgOp linalgOp);

/// Utility to identify whether a linalg op is a 2D matmul-like op.
bool is2DMatmulLikeOp(linalg::LinalgOp linalgOp);

/// Utility to identify whether a linalg op is a 4D matmul-like op.
bool is4DMatmulLikeOp(linalg::LinalgOp linalgOp);

/// Utility to identify if the input operand has matmul-like op in its
/// def-chain.
bool isMatmulInDefChain(Value operand);

/// Utility to identify if `linalgOp` is an elementwise operation with a
/// matmul-like op upstream in its computation tree.
bool isElementwiseWithMatmulProducer(linalg::LinalgOp linalgOp);

/// Utility to identify if `linalgOp` is a matmul-like operation with an
/// elementwise op as its consumer.
bool isMatmulWithElementwiseConsumer(linalg::LinalgOp linalgOp);

/// Utility to identify if `linalgOp` is a supported reduction op.
bool isReductionOp(linalg::LinalgOp linalgOp);

bool isConv2DNhwcHwcfOp(Operation *linalgOp);
bool isDepthwiseConv2DNhwcHwcOp(Operation *linalgOp);
bool isConvOp(linalg::GenericOp linalgOp);

/// Loop-dimension structure of a 2-D max pooling `linalg.generic` in a
/// channels-last-like layout: the output indexing map is the identity over the
/// parallel loops, the data input indexes each output dim either directly
/// (`plainDims`: batch, channel) or as `d * stride + k` (`spatialDims`, with
/// the window loop `k` in `windowDims` and the stride in `strides`), and the
/// second input is the window (`windowDims` only). Body: `maximumf(out, in)`.
struct MaxPoolingDims {
  SmallVector<unsigned> plainDims;
  SmallVector<unsigned> spatialDims;
  SmallVector<unsigned> windowDims;
  SmallVector<int64_t> strides;
};
FailureOr<MaxPoolingDims> getMaxPoolingDims(linalg::LinalgOp linalgOp);

/// Utility to identify a 2-D max pooling generic (see `getMaxPoolingDims`).
bool isMaxPoolingOp(linalg::LinalgOp linalgOp);

/// Number of array columns the pool pipeline spreads a pooling's leading
/// spatial output dim `oh` over under `PoolColumnsPolicy::Odd`: its odd factor
/// when 1 < odd <= numCols (a halving pooling chain keeps that factor down to
/// its last layer, so every layer gets the same column count), else the
/// largest divisor of `oh` that is <= numCols.
int64_t choosePoolColumns(int64_t oh, int64_t numCols);

/// Per-column cost model of a pool dispatch, `t(cols) = fixedMs +
/// perColumnMs * (cols - 1) + msPerMBPerColumn * MB / cols`: each column adds
/// a fixed cost (its shim DMA programming / sync) and one shim channel's worth
/// of bandwidth. Fitted on npu4, f32 scalar pooling, VGG-16 pool1-5 at 7
/// columns and pool4/5 at 1 column (docs_POOL/1001_pool_column_cost_model.md).
struct PoolColumnCost {
  double fixedMs = 0.74;
  double perColumnMs = 0.21;
  double msPerMBPerColumn = 1.77;
  // Host side of padding OH (only when a padded candidate is considered): a
  // pad dispatch that copies the input and a crop dispatch that copies the
  // output, at an assumed host copy rate, plus a launch each. Not measured;
  // an estimate (5 GB/s, 50 us) that keeps a padded candidate from winning on
  // a few hundred microseconds of NPU time (pool5: 2 padded columns would be
  // 0.17 ms faster than 1 unpadded column on the NPU alone).
  double hostMsPerMB = 0.2;
  double hostDispatchMs = 0.05;
  double timeMs(int64_t cols, double bytes) const {
    return fixedMs + perColumnMs * (cols - 1) +
           msPerMBPerColumn * (bytes / (1024.0 * 1024.0)) / cols;
  }
  double hostPadMs(double bytesCopied) const {
    return hostMsPerMB * (bytesCopied / (1024.0 * 1024.0)) +
           2 * hostDispatchMs;
  }
};

/// What a 2-D max pooling moves through the array for the cost model, from
/// its output extents and window: `(inExtent(oh) * inExtent(ow) * c) *
/// bytesIn + (oh * ow * c) * bytesOut`, `inExtent(t) = (t - 1) * stride +
/// window`. Computed from the same quantities in the preprocessing pass and
/// the root config so the two choose the same column count.
struct PoolShape {
  int64_t ow, c, windowH, windowW, strideH, strideW, bytesIn, bytesOut;
  double bytesMoved(int64_t oh) const;
};

/// Column count for `oh` under `policy` (see `PoolColumnsPolicy`). With
/// `allowPad` the result may name an `ohPadded > oh` the caller must pad OH up
/// to first (preprocessing); the root config calls with `allowPad = false` on
/// the shape it is given and gets the same `cols` the preprocessing chose for
/// it, since both evaluate the same candidates on the same bytes.
struct PoolColumns {
  int64_t cols;
  int64_t ohPadded;
};
PoolColumns choosePoolColumns(int64_t oh, int64_t numCols,
                              PoolColumnsPolicy policy, const PoolShape &shape,
                              bool allowPad,
                              const PoolColumnCost &cost = PoolColumnCost());

/// Returns true if `v` has no defined contents: it is a `tensor.empty`, or a
/// slice of one. Such an operand (e.g. a pooling window that only sizes the
/// reduction loops) never needs to be copied between memory levels.
bool isUndefinedTensor(Value v);

/// Utility to convert a `uint32_t` value into a hex string.
std::string utohexstr(uint32_t value, size_t width, bool header = true,
                      bool lowercase = false);

/// If `op` is in `block`, then return `op`. Otherwise traverse through parents
/// to the first ancestor of `op` that is in `block`, and return that
/// ancestor. If `op` has no ancestor in `block`, or if `op` is nullptr or
/// `block` is nullptr, return nullptr.
Operation *getAncestorInBlock(Operation *op, Block *block);

namespace detail {

/// Returns the largest number that perfectly divides `num` that
/// is less than or equal to max
int findLargestFactor(int num, int max);

/// A variant where we prefer factors to also be a multiple of `multiple`
int findLargestFactor(int num, int max, int multiple);

}  // namespace detail

/// Convert an array into a string, for example "[1,2,3]".
template <typename T>
std::string getArrayString(ArrayRef<T> vs) {
  return std::string("[")
      .append(llvm::join(
          llvm::map_range(vs, [](T v) { return std::to_string(v); }), ","))
      .append("]");
}

/// If all values in `opFoldResults` are constant, return a string
/// representation of the constant values. Otherwise, return
/// "[not constant integers]".
std::string getConstantIntValuesString(ArrayRef<OpFoldResult> opFoldResults);

/// Consider all operations in the region, recursively. If the operation
/// has an operand that is not in the region, and the `shouldSink` function
/// returns true for that operand's producer, then replace all uses of the
/// operand inside the region with a clone of the operand in the block.
///
/// If `shouldSink` returns true for all operations, then this function will
/// make the region isolated from above.  So this function essentially makes
/// the region isolated from above with respect to the set of operation types
/// defined by `shouldSink`.
///
/// \return true if the region was changed.
bool sinkInto(Region &, IRRewriter &,
              std::function<bool(Operation *)> shouldSink);

/// Create an scf.for operation with an attribute specifying that LLVM
/// must not unroll it. The for loop will iterate from `start` to `end` with
/// `step` increment, and have debug location `loc`. The body of the loop will
/// be empty, and the insertion point of builder will be unchanged by this
/// function.
scf::ForOp createForOpWithUnrollingDisabled(OpBuilder &builder, Location loc,
                                            int start, int end, int step);

/// Find all func.func operations that are nested inside `rootOp`, and all
/// func.call operations inside `rootOp` that call the func.func ops.
SmallVector<std::pair<func::FuncOp, SmallVector<func::CallOp>>>
getFunctionsAndTheirCallers(Operation *rootOp);

/// Get the number of columns used by the cores in the module. Returns an
/// empty optional if the number cannot be determined.
std::optional<int64_t> getNumColumnsUsedByCores(ModuleOp moduleOp);
}  // namespace mlir::iree_compiler::AMDAIE

#endif
