// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEUtils.h"

#include <algorithm>
#include <optional>

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-amdaie-utils"

namespace mlir::iree_compiler::AMDAIE {

std::string getConstantIntValuesString(ArrayRef<OpFoldResult> ofrs) {
  std::optional<SmallVector<int64_t>> maybeValues =
      mlir::getConstantIntValues(ofrs);
  if (maybeValues.has_value())
    return getArrayString<int64_t>(maybeValues.value());
  return "[not all constant integers]";
}

template <typename T>
std::optional<T> getConfigAttr(IREE::HAL::ExecutableTargetAttr targetAttr,
                               StringRef name) {
  if (!targetAttr) return std::nullopt;
  DictionaryAttr config = targetAttr.getConfiguration();
  if (!config) return std::nullopt;
  T attr = config.getAs<T>(name);
  if (!attr) return std::nullopt;
  return attr;
}

std::optional<AMDAIEDevice> getConfigAMDAIEDevice(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<StringAttr> attr =
      getConfigAttr<StringAttr>(targetAttr, "target_device");
  if (!attr) return std::nullopt;
  return AMDAIE::symbolizeEnum<AMDAIEDevice>(attr.value().getValue());
}

std::optional<AMDAIEDevice> getConfigAMDAIEDevice(Operation *op) {
  IREE::HAL::ExecutableTargetAttr targetAttr =
      IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!targetAttr) return std::nullopt;
  return getConfigAMDAIEDevice(targetAttr);
}

std::optional<AMDAIE::AMDAIEDevice> getConfigAMDAIEDeviceFromAncestor(
    Operation *op) {
  while (op) {
    if (ModuleOp moduleOp = dyn_cast<ModuleOp>(op)) {
      IREE::HAL::ExecutableTargetAttr targetAttr =
          IREE::HAL::ExecutableTargetAttr::lookup(moduleOp);
      std::optional<AMDAIEDevice> maybeDevice =
          AMDAIE::getConfigAMDAIEDevice(targetAttr);
      if (maybeDevice.has_value()) return maybeDevice;
    }
    op = op->getParentOp();
  }
  return std::nullopt;
}

/// Utility that returns the number of columns being targeted.
std::optional<int64_t> getConfigNumColumns(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<IntegerAttr> attr =
      getConfigAttr<IntegerAttr>(targetAttr, "num_cols");
  if (!attr) return std::nullopt;
  return attr->getInt();
}

/// Utility that returns the number of rows being targeted.
std::optional<int64_t> getConfigNumRows(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<IntegerAttr> attr =
      getConfigAttr<IntegerAttr>(targetAttr, "num_rows");
  if (!attr) return std::nullopt;
  return attr->getInt();
}

int64_t getPackPeelReductionTile(int64_t kPackScaleL1) {
  // The base L2 reduction (K) tile the pack-peel pipeline packs by. Empirically
  // best at 32 for bf16 (device-independent tuning constant); the 4-level
  // variant scales it (`kPackScaleL1 == 2`). See KernelDispatch `k0Pack`.
  return kPackScaleL1 * 32;
}

std::array<uint32_t, 3> getScalarMatmulPackSize() { return {4, 4, 8}; }

bool hasUKernelTemplate(AMDAIEDevice device, StringRef ukernelName,
                        ArrayRef<Type> elemTypes, bool transposeB) {
  struct UKernelTemplate {
    StringRef ukernelName;
    SmallVector<StringRef, 3> elemTypes;
    bool transposeB;
  };
  // The templates in uKernels/npu4/peano.
  static const UKernelTemplate kNpu4Templates[] = {
      {"matmul", {"bf16", "bf16", "f32"}, /*transposeB=*/true},
      {"matmul", {"i8", "i8", "i32"}, /*transposeB=*/false},
      {"matvec", {"bf16", "bf16", "f32"}, /*transposeB=*/true},
      {"zero_fill", {"f32"}, /*transposeB=*/false},
      {"zero_fill", {"i32"}, /*transposeB=*/false},
      {"trunci", {"i32", "i8"}, /*transposeB=*/false},
  };
  auto isUKernel = [&](const UKernelTemplate &t) {
    return t.ukernelName == ukernelName;
  };
  if (!isAie2P(device) || llvm::none_of(kNpu4Templates, isUKernel)) return true;
  auto typeToString = [](Type type) {
    std::string typeStr;
    llvm::raw_string_ostream rso(typeStr);
    type.print(rso);
    return typeStr;
  };
  return llvm::any_of(kNpu4Templates, [&](const UKernelTemplate &t) {
    return isUKernel(t) && (t.transposeB || !transposeB) &&
           llvm::equal(t.elemTypes, elemTypes, [&](StringRef s, Type type) {
             return s == typeToString(type);
           });
  });
}

bool isGemvShape(int64_t m, bool transposeB) { return m == 1 && transposeB; }

int64_t choosePoolColumns(int64_t oh, int64_t numCols) {
  if (oh <= 0 || numCols <= 0) return 1;
  int64_t odd = oh;
  while (odd % 2 == 0) odd /= 2;
  if (odd > 1 && odd <= numCols) return odd;
  return detail::findLargestFactor(oh, numCols);
}

double PoolShape::bytesMoved(int64_t oh) const {
  auto inExtent = [](int64_t t, int64_t stride, int64_t window) {
    return (t - 1) * stride + window;
  };
  double in = static_cast<double>(inExtent(oh, strideH, windowH)) *
              inExtent(ow, strideW, windowW) * c * bytesIn;
  double out = static_cast<double>(oh) * ow * c * bytesOut;
  return in + out;
}

PoolColumns choosePoolColumns(int64_t oh, int64_t numCols,
                              PoolColumnsPolicy policy, const PoolShape &shape,
                              bool allowPad, const PoolColumnCost &cost) {
  if (oh <= 0 || numCols <= 0) return {1, oh};
  auto roundUp = [](int64_t v, int64_t m) { return ((v + m - 1) / m) * m; };
  switch (policy) {
    case PoolColumnsPolicy::Odd:
      return {choosePoolColumns(oh, numCols), oh};
    case PoolColumnsPolicy::Max: {
      int64_t best = 1;
      for (int64_t d = 1; d <= std::min(oh, numCols); ++d)
        if (oh % d == 0) best = d;
      return {best, oh};
    }
    case PoolColumnsPolicy::Full: {
      if (oh % numCols == 0) return {numCols, oh};
      if (allowPad) return {numCols, roundUp(oh, numCols)};
      return {choosePoolColumns(oh, numCols), oh};
    }
    case PoolColumnsPolicy::Cost: {
      // Cheapest divisor of a given extent under the cost model.
      auto bestFor = [&](int64_t extent) {
        double bytes = shape.bytesMoved(extent);
        int64_t bestCols = 1;
        double bestT = cost.timeMs(1, bytes);
        for (int64_t d = 2; d <= std::min(extent, numCols); ++d) {
          if (extent % d != 0) continue;
          double t = cost.timeMs(d, bytes);
          if (t < bestT) {
            bestT = t;
            bestCols = d;
          }
        }
        return std::make_pair(bestCols, bestT);
      };
      auto [cols, t] = bestFor(oh);
      if (allowPad && oh % numCols != 0) {
        int64_t padded = roundUp(oh, numCols);
        auto [colsP, tP] = bestFor(padded);
        // The padded shape also pays the host pad (input copy) and crop
        // (output copy) dispatches.
        tP += cost.hostPadMs(shape.bytesMoved(padded));
        if (tP < t) return {colsP, padded};
      }
      return {cols, oh};
    }
  }
  return {choosePoolColumns(oh, numCols), oh};
}

bool getConfigGemvPadN(IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<BoolAttr> attr =
      getConfigAttr<BoolAttr>(targetAttr, "gemv_pad_n");
  return !attr || attr->getValue();
}

int64_t getConfigGemvN1(IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<IntegerAttr> attr =
      getConfigAttr<IntegerAttr>(targetAttr, "gemv_n1");
  return attr ? attr->getInt() : 32;
}

llvm::StringRef stringifyPoolColumnsPolicy(PoolColumnsPolicy policy) {
  switch (policy) {
    case PoolColumnsPolicy::Odd:
      return "odd";
    case PoolColumnsPolicy::Cost:
      return "cost";
    case PoolColumnsPolicy::Full:
      return "full";
    case PoolColumnsPolicy::Max:
      return "max";
  }
  return "max";
}

PoolColumnsPolicy getConfigPoolColumnsPolicy(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<StringAttr> attr =
      getConfigAttr<StringAttr>(targetAttr, "pool_columns_policy");
  if (!attr) return PoolColumnsPolicy::Max;
  llvm::StringRef s = attr->getValue();
  if (s == "odd") return PoolColumnsPolicy::Odd;
  if (s == "cost") return PoolColumnsPolicy::Cost;
  if (s == "full") return PoolColumnsPolicy::Full;
  return PoolColumnsPolicy::Max;
}

/// Double-buffered A (M x kTile), B (n1 x kTile) and C (M x n1) tiles, plus a
/// fused epilogue's (M x n1) output tile when there is one.
static int64_t gemvL1Bytes(int64_t M, int64_t n1, int64_t kTile,
                           int64_t bytesLhs, int64_t bytesRhs,
                           int64_t bytesOut, int64_t bytesEpilogueOut) {
  return 2 * (M * kTile * bytesLhs + n1 * kTile * bytesRhs +
              M * n1 * (bytesOut + bytesEpilogueOut));
}

FailureOr<GemvNTiling> chooseGemvNTiling(int64_t M, int64_t N, int64_t kTile,
                                         int64_t bytesLhs, int64_t bytesRhs,
                                         int64_t bytesOut, int64_t numRows,
                                         int64_t numCols, int64_t l1Limit,
                                         int64_t n1Max,
                                         int64_t bytesEpilogueOut) {
  int64_t numCores = numRows * numCols;
  GemvNTiling best{0, 0};
  auto better = [&](int64_t n1, int64_t cores) {
    if (cores != best.cores) return cores > best.cores;
    if (cores == numCores) {
      // Canonical: the largest n1 that is <= n1Max; anything above the cap
      // only if nothing fits under it.
      bool a = n1 <= n1Max, b = best.n1 <= n1Max;
      if (a != b) return a;
      return a ? n1 > best.n1 : n1 < best.n1;
    }
    return n1 > best.n1;
  };
  // n1 = 1 is never taken: a GEMV with one output per core produces wrong
  // results (all zeros on npu4), cause not yet known.
  for (int64_t n1 = 2; n1 <= N; ++n1) {
    if (gemvL1Bytes(M, n1, kTile, bytesLhs, bytesRhs, bytesOut,
                    bytesEpilogueOut) > l1Limit)
      break;
    if (N % n1 != 0) continue;
    int64_t cores = 0;
    for (int64_t c = numCores; c >= 1; --c) {
      if (N % (c * n1) != 0 || (c > numRows && c % numRows != 0)) continue;
      cores = c;
      break;
    }
    if (best.n1 == 0 || better(n1, cores)) best = {n1, cores};
  }
  if (best.n1 == 0) return failure();
  return best;
}

/// Utility to retrieve a constant index from an OpFoldResult.
int64_t getConstantIndexOrAssert(OpFoldResult ofr) {
  std::optional<int64_t> res = getConstantIntValue(ofr);
  assert(res.has_value() && "expect constant index");
  return res.value();
}

// The number of elements in a vector instruction for a given element type.
// Reference:
// https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_ml_intrinsics/intrinsics/group__intr__gpvectorop__mul__bf16xbf16.html
FailureOr<uint32_t> getAIEMacNumElements(Type inputType, Type outputType) {
  if (inputType.isInteger(8) && outputType.isInteger(32)) return 32;
  if (inputType.isInteger(16) && outputType.isInteger(32)) return 32;
  if (inputType.isBF16() && outputType.isF32()) return 16;
  return failure();
}

/// Get the m/n/k dimension of a matmul-like op from its affine map.
static mlir::AffineExpr getAffineMapDim(ArrayAttr indexingMaps,
                                        uint32_t mapIndex, uint32_t mnkIndex) {
  auto affineMap = cast<AffineMapAttr>(indexingMaps[mapIndex]).getValue();
  uint32_t nResults = affineMap.getNumResults();
  return affineMap.getResult(nResults - 2 + mnkIndex);
}

/// Returns the BlockArgument that leads to `val`, if any. Traverses optional
/// ext* ops.
BlockArgument getBlockArgumentWithOptionalExtOps(Value val) {
  BlockArgument blockArg;
  if (!(blockArg = dyn_cast<BlockArgument>(val))) {
    Operation *defOp = val.getDefiningOp();
    if (!dyn_cast_if_present<arith::ExtFOp>(defOp) &&
        !dyn_cast_if_present<arith::ExtSIOp>(defOp) &&
        !dyn_cast_if_present<arith::ExtUIOp>(defOp)) {
      return nullptr;
    }
    blockArg = dyn_cast<BlockArgument>(defOp->getOperand(0));
  }
  return blockArg;
}

/// Utility to match block body for matmul-like ops.
static bool bodyMatcherForMatmulLikeOps(Value yieldVal, Block *body) {
  Operation *addOp = yieldVal.getDefiningOp();
  if (!isa_and_present<arith::AddIOp, arith::AddFOp>(addOp)) return false;

  Operation *mulOp = addOp->getOperand(1).getDefiningOp();
  if (!isa_and_present<arith::MulIOp, arith::MulFOp>(mulOp)) return false;

  BlockArgument lhsBlockArg =
      getBlockArgumentWithOptionalExtOps(mulOp->getOperand(0));
  BlockArgument rhsBlockArg =
      getBlockArgumentWithOptionalExtOps(mulOp->getOperand(1));
  BlockArgument outBlockArg =
      getBlockArgumentWithOptionalExtOps(addOp->getOperand(0));
  if (!lhsBlockArg || !rhsBlockArg || !outBlockArg ||
      lhsBlockArg.getOwner() != body || rhsBlockArg.getOwner() != body ||
      outBlockArg.getOwner() != body || lhsBlockArg.getArgNumber() != 0 ||
      rhsBlockArg.getArgNumber() != 1 || outBlockArg.getArgNumber() != 2) {
    return false;
  }
  return true;
}

/// Utility to check if the input generic op is a 2D matmul-like op.
bool is2DMatmulLikeOp(linalg::LinalgOp linalgOp) {
  // Check iterator types.
  unsigned numParallelLoops = linalgOp.getNumParallelLoops();
  unsigned numReductionLoops = linalgOp.getNumReductionLoops();
  if (numParallelLoops != 2 || numReductionLoops != 1) return false;

  // Check the number of inputs and results from indexing maps.
  ArrayAttr indexingMaps = linalgOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 2 || map1.getNumResults() != 2 ||
      map2.getNumResults() != 2 || map0.getNumInputs() != 3 ||
      map1.getNumInputs() != 3 || map2.getNumInputs() != 3) {
    return false;
  }
  return true;
}

/// Utility to check if the input generic op is a 4D matmul-like op.
bool is4DMatmulLikeOp(linalg::LinalgOp linalgOp) {
  // Check iterator types.
  unsigned numParallelLoops = linalgOp.getNumParallelLoops();
  unsigned numReductionLoops = linalgOp.getNumReductionLoops();
  if (numParallelLoops != 4 || numReductionLoops != 2) return false;

  // Check indexing maps.
  ArrayAttr indexingMaps = linalgOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 4 || map1.getNumResults() != 4 ||
      map2.getNumResults() != 4 || map0.getNumInputs() != 6 ||
      map1.getNumInputs() != 6 || map2.getNumInputs() != 6) {
    return false;
  }
  return true;
}

/// Utility to check if the input generic op is a 6D matmul-like op.
bool is6DMatmulLikeOp(linalg::LinalgOp linalgOp) {
  // Check iterator types.
  unsigned numParallelLoops = linalgOp.getNumParallelLoops();
  unsigned numReductionLoops = linalgOp.getNumReductionLoops();
  if (numParallelLoops != 6 || numReductionLoops != 3) return false;

  // Check indexing maps.
  ArrayAttr indexingMaps = linalgOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 6 || map1.getNumResults() != 6 ||
      map2.getNumResults() != 6 || map0.getNumInputs() != 9 ||
      map1.getNumInputs() != 9 || map2.getNumInputs() != 9) {
    return false;
  }
  return true;
}

/// Utility to identify whether a linalg op is a broad concept matmul op. Here
/// we don't limit the number of operands in the input and outputs, but the
/// innermost 3 dimensions must map exactly to a matmul.
bool isMatmul(linalg::LinalgOp linalgOp) {
  // Step 0. Test if the op itself is a linalg.matmul op.
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(linalgOp)) return true;
  if (!isa<linalg::GenericOp>(linalgOp)) return false;

  // Step 1. Test the body of the generic to indeed be what we expect for a
  //         matmul.
  Block *body = linalgOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  if (!bodyMatcherForMatmulLikeOps(yieldVal, body)) return false;

  // Step 2. Check the innermost 3 dimensions 'parallel, parallel, reduction'
  //         map exactly to a matmul.
  ArrayAttr maps = linalgOp.getIndexingMaps();
  uint32_t A = 0, B = 1, C = 2;
  bool isMatmul =
      getAffineMapDim(maps, A, 0) == getAffineMapDim(maps, C, 0) &&  // M
      getAffineMapDim(maps, B, 1) == getAffineMapDim(maps, C, 1) &&  // N
      getAffineMapDim(maps, A, 1) == getAffineMapDim(maps, B, 0);    // K
  return isMatmul;
}

/// Utility to identify whether a linalg op is a broad concept matmul with
/// lhs matrix transposed. Here we don't limit the number of operands in the
/// input and outputs, but the innermost 3 dimensions must map exactly to a
/// matmul_transpose_a op.
bool isMatmulTransposeA(linalg::LinalgOp linalgOp) {
  // Step 0. Test if the op itself is a linalg.matmul_transpose_a op.
  if (isa<linalg::MatmulTransposeAOp, linalg::BatchMatmulTransposeAOp>(
          linalgOp.getOperation()))
    return true;
  if (!isa<linalg::GenericOp>(linalgOp)) return false;

  // Step 1. Test the body of the generic to indeed be what we expect for a
  //         matmul.
  Block *body = linalgOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  if (!bodyMatcherForMatmulLikeOps(yieldVal, body)) return false;

  // Step 2. Check the innermost 3 dimensions 'parallel, parallel, reduction'
  //         map exactly to a matmul_transpose_a.
  ArrayAttr maps = linalgOp.getIndexingMaps();
  uint32_t A = 0, B = 1, C = 2;
  bool isATransposed =
      getAffineMapDim(maps, A, 1) == getAffineMapDim(maps, C, 0) &&  // M
      getAffineMapDim(maps, B, 1) == getAffineMapDim(maps, C, 1) &&  // N
      getAffineMapDim(maps, A, 0) == getAffineMapDim(maps, B, 0);    // K
  return isATransposed;
}

/// Utility to identify whether a linalg op is a broad concept matmul with
/// rhs matrix transposed. Here we don't limit the number of operands in the
/// input and outputs, but the innermost 3 dimensions must map exactly to a
/// matmul_transpose_b op.
bool isMatmulTransposeB(linalg::LinalgOp linalgOp) {
  // Step 0. Test if the op itself is a linalg.matmul_transpose_b op.
  if (isa<linalg::MatmulTransposeBOp, linalg::BatchMatmulTransposeBOp>(
          linalgOp.getOperation()))
    return true;
  if (!isa<linalg::GenericOp>(linalgOp)) return false;

  // Step 1. Test the body of the generic to indeed be what we expect for a
  //         matmul.
  Block *body = linalgOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  if (!bodyMatcherForMatmulLikeOps(yieldVal, body)) return false;

  // Step 2. Check the innermost 3 dimensions 'parallel, parallel, reduction'
  //         map exactly to a matmul_transpose_a.
  ArrayAttr maps = linalgOp.getIndexingMaps();
  uint32_t A = 0, B = 1, C = 2;
  bool isBTransposed =
      getAffineMapDim(maps, A, 0) == getAffineMapDim(maps, C, 0) &&  // M
      getAffineMapDim(maps, B, 0) == getAffineMapDim(maps, C, 1) &&  // N
      getAffineMapDim(maps, A, 1) == getAffineMapDim(maps, B, 1);    // K
  return isBTransposed;
}

/// Utility to identify if the input operand has matmul-like op in its
/// def-chain.
bool isMatmulInDefChain(Value operand) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp) return false;

  if (isa<arith::ConstantOp>(defOp)) return false;

  // By the time this is called (post first-level packing), a source matmul
  // has already become a packed `linalg.generic` whose indexing maps may be
  // permuted for a transposed operand, so `isMatmul`'s Step 0 op-type
  // short-circuit no longer applies and its Step 2 map check alone would
  // reject a transpose_a/transpose_b matmul. Ask all three so a fused
  // consumer of a transpose_b matmul (e.g. VGG16's FC layers) is still
  // recognized and skipped here, instead of being (wrongly) promoted to a
  // second shared-memory buffer.
  if (auto defLinalgOp = dyn_cast_if_present<linalg::LinalgOp>(defOp)) {
    if (isMatmul(defLinalgOp) || isMatmulTransposeA(defLinalgOp) ||
        isMatmulTransposeB(defLinalgOp))
      return true;
  }

  // If something is being produced from a for/forall loop, we just assume it is
  // some fused computation and do not really need to look at its body to match
  // matmul.
  if (isa<scf::ForOp>(defOp) || isa<scf::ForallOp>(defOp)) {
    return true;
  }

  for (Value operand : defOp->getOperands()) {
    if (isMatmulInDefChain(operand)) return true;
  }
  return false;
}

/// Utility to identify if `linalgOp` is an elementwise operation with a
/// matmul-like op upstream in its computation tree.
bool isElementwiseWithMatmulProducer(linalg::LinalgOp linalgOp) {
  if (!linalg::isElementwise(linalgOp) ||
      isa<linalg::FillOp, linalg::CopyOp>(linalgOp)) {
    return false;
  }
  // Check if any of the defining op is a matmul-like op.
  for (Value operand : linalgOp->getOperands()) {
    if (isMatmulInDefChain(operand)) return true;
  }
  return false;
}

/// Utility to identify if `linalgOp` is a matmul-like operation with an
/// elementwise op as its consumer.
bool isMatmulWithElementwiseConsumer(linalg::LinalgOp linalgOp) {
  // Currently only called (KernelDispatch.cpp) on the unpacked source op,
  // where `isMatmul`'s Step 0 short-circuit still applies for a
  // transpose_a/transpose_b matmul, but ask all three for the same reason as
  // `isMatmulInDefChain` above, in case this is ever called post-packing.
  if (!isMatmul(linalgOp) && !isMatmulTransposeA(linalgOp) &&
      !isMatmulTransposeB(linalgOp))
    return false;
  // Check if the user is an elementwise op but not fill or copy op.
  for (Operation *userOp : linalgOp->getUsers()) {
    if (auto linalgUser = dyn_cast<linalg::LinalgOp>(userOp)) {
      if (isElementwise(linalgUser) &&
          !isa<linalg::FillOp, linalg::CopyOp>(linalgUser))
        return true;
    }
  }
  return false;
}

static bool isIteratorTypesOfConvSame(
    linalg::GenericOp linalgOp,
    SmallVector<utils::IteratorType> expectedIteratorTypes) {
  if (!linalgOp) return false;
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (iteratorTypes.size() != expectedIteratorTypes.size()) return false;
  for (unsigned i = 0, n = expectedIteratorTypes.size(); i < n; i++) {
    if (iteratorTypes[i] != expectedIteratorTypes[i]) return false;
  }
  return true;
}

bool isConv2DNhwcHwcfOp(Operation *linalgOp) {
  if (isa<linalg::Conv2DNhwcHwcfOp>(linalgOp)) return true;
  SmallVector<utils::IteratorType> expectedIteratorTypes = {
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::reduction, utils::IteratorType::reduction,
      utils::IteratorType::reduction};
  return isIteratorTypesOfConvSame(dyn_cast<linalg::GenericOp>(linalgOp),
                                   expectedIteratorTypes);
}

bool isDepthwiseConv2DNhwcHwcOp(Operation *linalgOp) {
  if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(linalgOp)) return true;
  SmallVector<utils::IteratorType> expectedIteratorTypes(
      cast<ShapedType>(linalgOp->getResult(0).getType()).getRank(),
      utils::IteratorType::parallel);
  expectedIteratorTypes.push_back(utils::IteratorType::reduction);
  expectedIteratorTypes.push_back(utils::IteratorType::reduction);
  return isIteratorTypesOfConvSame(dyn_cast<linalg::GenericOp>(linalgOp),
                                   expectedIteratorTypes);
}

/// Utility to identify whether a linalg op is a broad concept conv op.
bool isConvOp(linalg::GenericOp linalgOp) {
  if (!linalgOp) return false;
  // Test the body of the generic to indeed be what we expect for a matmul.
  Block *body = linalgOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  if (!bodyMatcherForMatmulLikeOps(yieldVal, body)) return false;

  return isConv2DNhwcHwcfOp(linalgOp) || isDepthwiseConv2DNhwcHwcOp(linalgOp);
}

/// Utility to identify if `linalgOp` is a supported reduction op. Currently,
/// we are using strict conditions for reduction op matching.
bool isReductionOp(linalg::LinalgOp linalgOp) {
  if (isMatmul(linalgOp) || isMatmulTransposeA(linalgOp) ||
      isMatmulTransposeB(linalgOp))
    return false;

  // Make sure there is only one reduction dimension.
  SmallVector<unsigned> reductionDims;
  linalgOp.getReductionDims(reductionDims);
  if (reductionDims.size() != 1) return false;

  // Make sure the reduction dimension is static and innermost.
  unsigned dim = reductionDims[0];
  SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
  if (ShapedType::isDynamic(bounds[dim])) return false;

  int64_t numParallelDims = linalgOp.getNumParallelLoops();
  if (dim < numParallelDims) return false;

  // Only support projected permutation.
  if (llvm::any_of(linalgOp.getDpsInputOperands(), [&](OpOperand *input) {
        return !linalgOp.getMatchingIndexingMap(input).isProjectedPermutation();
      }))
    return false;

  // Only support a single output.
  if (linalgOp.getNumDpsInits() != 1) return false;

  // Only support single combiner operation.
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps) ||
      combinerOps.size() != 1)
    return false;

  return true;
}

std::string utohexstr(uint32_t value, size_t width, bool header,
                      bool lowercase) {
  std::string res = "";
  if (header) res += "0x";
  std::string hexStr = llvm::utohexstr(value, lowercase);
  std::string prefix(width - hexStr.size(), '0');
  return res + prefix + hexStr;
}

/// Return an ancestor of 'op' in 'block', or nullptr if no such ancestor.
Operation *getAncestorInBlock(Operation *op, Block *block) {
  if (!op || !block) return nullptr;
  while (op && (op->getBlock() != block)) op = op->getParentOp();
  return op;
}

/// Find the largest factor of 'num' which is not larger than 'max'.
int detail::findLargestFactor(int num, int max) {
  assert(max > 0 && "No factors less than or equal to 0 exist");

  // Do O(1) instead of O(sqrt(num)) computation for this common case.
  if (num <= max) {
    return num;
  }

  int largestLowFactor = 1;
  for (int lowFactor = 2; lowFactor <= max; ++lowFactor) {
    const int highFactor = num / lowFactor;

    // This early exit is what makes this O(sqrt(num)) instead of O(num).
    if (highFactor < lowFactor) return largestLowFactor;

    const bool areActuallyFactors = num % lowFactor == 0;
    if (areActuallyFactors) {
      // We're certain that here lowFactor <= highFactor, and highFactor is
      // descending in this loop. So we can return immediately if highFactor is
      // good.
      if (highFactor <= max) return highFactor;
      largestLowFactor = lowFactor;
    }
  }
  return largestLowFactor;
}

/// Find the largest factor of 'num' which is not larger than 'max' and is a
/// multiple of `multiple` if possible.
int detail::findLargestFactor(int num, int max, int multiple) {
  int factor = 0;
  for (int i = multiple; i <= max && i <= num; i += multiple) {
    if (num % i == 0 && i % multiple == 0) {
      factor = i;
    }
  }
  // if we could not find the desired factor then we give up and call the code
  // that doesnt require the multiple constrain.
  return factor ? factor : detail::findLargestFactor(num, max);
}

bool sinkInto(Region &region, IRRewriter &rewriter,
              std::function<bool(Operation *)> shouldSink) {
  Operation *parentOfRegion = region.getParentOp();
  assert(parentOfRegion && "Region has no parent operation");
  if (region.getBlocks().empty()) return false;
  bool regionChanged = false;
  for (Block &block : region.getBlocks()) {
    // Collect all ops in the block.
    SmallVector<Operation *> ops;
    SmallVector<Operation *> nextIterationOps;
    block.walk([&](Operation *op) { ops.push_back(op); });
    while (!ops.empty()) {
      for (Operation *op : ops) {
        for (Value operand : op->getOperands()) {
          if (!operand || !operand.getDefiningOp()) continue;
          Operation *dependencyOp = operand.getDefiningOp();
          // Skip if the dependency is already in the core.
          if (parentOfRegion->isAncestor(dependencyOp)) continue;
          if (!shouldSink(dependencyOp)) continue;
          rewriter.setInsertionPointToStart(&block);
          Operation *sunkOp = rewriter.clone(*dependencyOp);
          nextIterationOps.push_back(sunkOp);
          // Replace uses of the dependency op inside the block. Specifically,
          // if `use` is in `block` then replace its operand with `sunkOp`.
          auto isInBlock = [&block](OpOperand &use) {
            Operation *op = use.getOwner();
            while (op) {
              if (op->getBlock() == &block) return true;
              op = op->getParentOp();
            }
            return false;
          };
          dependencyOp->replaceUsesWithIf(sunkOp, isInBlock);
          regionChanged = true;
        }
      }
      std::swap(ops, nextIterationOps);
      nextIterationOps.clear();
    }
  }
  return regionChanged;
}

scf::ForOp createForOpWithUnrollingDisabled(OpBuilder &builder, Location loc,
                                            int start, int end, int step) {
  // RAII guard to reset the insertion point of the builder when destroyed.
  OpBuilder::InsertionGuard guard(builder);

  auto getConstant = [&](int64_t v) {
    return arith::ConstantIndexOp::create(builder, loc, v);
  };

  Value cStart = getConstant(start);
  Value cEnd = getConstant(end);
  Value cStep = getConstant(step);

  scf::ForOp forOp = scf::ForOp::create(builder, loc, cStart, cEnd, cStep);

  mlir::LLVM::LoopUnrollAttr unrollAttr;
  mlir::LLVM::LoopAnnotationAttr loopAnnotationAttr;
  BoolAttr disableAttr = builder.getBoolAttr(true);
  unrollAttr = mlir::LLVM::LoopUnrollAttr::get(
      builder.getContext(), /*disable=*/disableAttr, /*count=*/{},
      /*runtimeDisable=*/{}, /*full=*/{}, /*followupUnrolled=*/{},
      /*followupRemainder=*/{}, /*followupAll=*/{});

  loopAnnotationAttr = mlir::LLVM::LoopAnnotationAttr::get(
      builder.getContext(), /*disableNonforced=*/{},
      /*vectorize=*/{}, /*interleave=*/{}, /*unroll=*/unrollAttr,
      /*unrollAndJam=*/{}, /*licm=*/{}, /*distribute=*/{},
      /*pipeline=*/{},
      /*peeled=*/{}, /*unswitch=*/{}, /*mustProgress=*/{},
      /*isVectorized=*/{}, /*startLoc=*/{}, /*endLoc=*/{},
      /*parallelAccesses=*/{});

  // Add the llvm.loop_annotation attribute to the loop.
  forOp->setAttr("loop_annotation", loopAnnotationAttr);

  return forOp;
}

SmallVector<std::pair<func::FuncOp, SmallVector<func::CallOp>>>
getFunctionsAndTheirCallers(Operation *rootOp) {
  // A mapping from all the function ops in the root op, to their callers.
  SmallVector<std::pair<func::FuncOp, SmallVector<func::CallOp>>>
      functionsAndCallers;

  // A mapping from function symbol names, to their index in
  // `functionsAndCallers`.
  DenseMap<StringRef, uint32_t> funcOpIndex;

  // Final all the function ops.
  rootOp->walk([&](func::FuncOp funcOp) {
    funcOpIndex.insert({funcOp.getSymName(), functionsAndCallers.size()});
    SmallVector<func::CallOp> callers;
    functionsAndCallers.push_back({funcOp, callers});
  });

  // Find all the call ops.
  rootOp->walk([&](func::CallOp callOp) {
    StringRef callee = callOp.getCallee();
    auto iter = funcOpIndex.find(callee);
    if (iter != funcOpIndex.end()) {
      functionsAndCallers[iter->second].second.push_back(callOp);
    }
  });
  return functionsAndCallers;
}

std::optional<int64_t> getNumColumnsUsedByCores(ModuleOp moduleOp) {
  int64_t minColumn = std::numeric_limits<int64_t>::max();
  int64_t maxColumn = std::numeric_limits<int64_t>::min();
  bool foundCoreOp = false;

  WalkResult res = moduleOp->walk([&](AMDAIE::CoreOp coreOp) {
    foundCoreOp = true;
    // Check if the core op has a constant column location.
    AMDAIE::TileOp tileOp = coreOp.getTileOp();
    std::optional<int64_t> maybeColumn = getConstantIntValue(tileOp.getCol());
    if (!maybeColumn) {
      coreOp.emitOpError() << "has non-constant tile location";
      return WalkResult::interrupt();
    }
    // Update the min and max column values.
    int64_t column = maybeColumn.value();
    minColumn = std::min(minColumn, column);
    maxColumn = std::max(maxColumn, column);
    return WalkResult::advance();
  });

  if (res.wasInterrupted() || !foundCoreOp) return std::nullopt;
  return (maxColumn - minColumn + 1);
}


/// Matches `d_o * stride + d_k` (or `d_o + d_k`) where `d_o` is a parallel
/// loop and `d_k` a reduction loop.
static bool matchStridedDimExpr(AffineExpr e,
                                ArrayRef<utils::IteratorType> iters,
                                unsigned &outDim, unsigned &winDim,
                                int64_t &stride) {
  auto add = dyn_cast<AffineBinaryOpExpr>(e);
  if (!add || add.getKind() != AffineExprKind::Add) return false;
  auto parse = [&](AffineExpr a, AffineExpr b) -> bool {
    auto bDim = dyn_cast<AffineDimExpr>(b);
    if (!bDim) return false;
    unsigned aPos;
    int64_t s = 1;
    if (auto aDim = dyn_cast<AffineDimExpr>(a)) {
      aPos = aDim.getPosition();
    } else {
      auto mul = dyn_cast<AffineBinaryOpExpr>(a);
      if (!mul || mul.getKind() != AffineExprKind::Mul) return false;
      auto d = dyn_cast<AffineDimExpr>(mul.getLHS());
      auto c = dyn_cast<AffineConstantExpr>(mul.getRHS());
      if (!d || !c) return false;
      aPos = d.getPosition();
      s = c.getValue();
    }
    unsigned bPos = bDim.getPosition();
    bool aPar = iters[aPos] == utils::IteratorType::parallel;
    bool bPar = iters[bPos] == utils::IteratorType::parallel;
    if (aPar && !bPar) {
      outDim = aPos;
      winDim = bPos;
      stride = s;
      return true;
    }
    if (!aPar && bPar && s == 1) {
      outDim = bPos;
      winDim = aPos;
      stride = 1;
      return true;
    }
    return false;
  };
  return parse(add.getLHS(), add.getRHS()) || parse(add.getRHS(), add.getLHS());
}

FailureOr<MaxPoolingDims> getMaxPoolingDims(linalg::LinalgOp linalgOp) {
  if (!isa<linalg::GenericOp>(linalgOp.getOperation())) return failure();
  if (linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1)
    return failure();
  if (linalgOp.getNumReductionLoops() != 2) return failure();
  Block *body = linalgOp.getBlock();
  if (!body || body->getNumArguments() != 3) return failure();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1) return failure();
  auto maxOp = yieldOp.getOperand(0).getDefiningOp<arith::MaximumFOp>();
  if (!maxOp) return failure();
  Value in = body->getArgument(0), out = body->getArgument(2);
  if (!((maxOp.getLhs() == out && maxOp.getRhs() == in) ||
        (maxOp.getLhs() == in && maxOp.getRhs() == out)))
    return failure();

  SmallVector<utils::IteratorType> iters = linalgOp.getIteratorTypesArray();
  AffineMap inMap =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  AffineMap winMap =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(1));
  AffineMap outMap =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));

  // Output: identity over the parallel loops, which come first.
  unsigned numPar = linalgOp.getNumParallelLoops();
  if (outMap.getNumResults() != numPar) return failure();
  for (auto [i, e] : llvm::enumerate(outMap.getResults())) {
    auto d = dyn_cast<AffineDimExpr>(e);
    if (!d || d.getPosition() != i || iters[i] != utils::IteratorType::parallel)
      return failure();
  }
  // Window: the two reduction loops.
  if (winMap.getNumResults() != 2) return failure();
  SmallVector<unsigned> winDims;
  for (AffineExpr e : winMap.getResults()) {
    auto d = dyn_cast<AffineDimExpr>(e);
    if (!d || iters[d.getPosition()] != utils::IteratorType::reduction)
      return failure();
    winDims.push_back(d.getPosition());
  }
  // Data input: one result per output dim, in the output's order, either the
  // dim itself or `dim * stride + window`.
  if (inMap.getNumResults() != numPar) return failure();
  MaxPoolingDims dims;
  for (auto [i, e] : llvm::enumerate(inMap.getResults())) {
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      if (d.getPosition() != i) return failure();
      dims.plainDims.push_back(i);
      continue;
    }
    unsigned o, k;
    int64_t s;
    if (!matchStridedDimExpr(e, iters, o, k, s) || o != i) return failure();
    dims.spatialDims.push_back(o);
    dims.windowDims.push_back(k);
    dims.strides.push_back(s);
  }
  if (dims.spatialDims.size() != 2 || dims.windowDims != winDims)
    return failure();
  return dims;
}

bool isMaxPoolingOp(linalg::LinalgOp linalgOp) {
  return succeeded(getMaxPoolingDims(linalgOp));
}

bool isUndefinedTensor(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (isa<tensor::EmptyOp>(def)) return true;
    auto slice = dyn_cast<tensor::ExtractSliceOp>(def);
    if (!slice) return false;
    v = slice.getSource();
  }
  return false;
}

}  // namespace mlir::iree_compiler::AMDAIE
