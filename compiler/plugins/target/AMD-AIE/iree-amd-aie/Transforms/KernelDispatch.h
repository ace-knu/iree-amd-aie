// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_
#define IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_

#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::AMDAIE {

/// Enum for AIE lowering pipelines to pick.
enum class LowerToAIEPassPipeline { AIR, ObjectFifo, None };

/// Enum for tiling pass pipelines to pick. Because of how the pass-pipeline
/// enums are implemented using tablegen in IREE, it isnt extensible.
/// This is an enum to pick different pass pipelines in IREE.
enum class TilePassPipeline {
  PackPeelPipeline,
  PackPeel4LevelTilingPipeline,
  ConvDecomposePipeline,
  GeneralCopyPipeline,
  None
};

/// Enum for types of loop peeling.
enum class PeelingType { First, Last, FirstLast };

/// Enum for operands to be padded.
enum class PadOperand { InputOutput, Input, Output };

/// Enum for operands to be bufferized to allocation.
enum class BufferizeOperand {
  LinalgInputOutput,
  LinalgInput,
  LinalgOutput,
  PackOrCopyInput
};

/// Enum for hardware mapping attributes.
enum class HardwareMapping { Core, Block, None };

/// K elements the GEMV pipeline streams through L1 per step (see
/// `setRootConfigForGemvPipeline`). Shared with the Flow pad pass, which
/// evaluates the GEMV core count on the same K tile.
constexpr int64_t kGemvKTile = 256;

/// How the pool pipeline spreads a max-pooling dispatch over the core array
/// (see `setRootConfigForPoolPipeline`).
enum class PoolCoreSplit {
  // Array columns take the leading spatial output dim (its odd factor: the
  // extent the last layer of a halving pooling chain is left with, 7 for every
  // VGG-16 pooling layer), array rows take the channel dim. The core forall is
  // 2-D and every layer uses the same cols x rows cores.
  SpatialChannel,
  // Exhaustive search over one distributed output dim (1-D core forall),
  // ranked by core count, then channel tile, then L1 bytes.
  Search
};

/// How the pool pipeline (and the preprocessing halo split / row padding that
/// must agree with it) picks the number of array columns a max pooling's
/// leading spatial output dim OH is spread over.
enum class PoolColumnsPolicy {
  // OH's odd factor when 1 < odd <= numCols, else its largest divisor <=
  // numCols. The pre-cost-model rule: every layer of a halving pooling chain
  // gets the same column count (7 for VGG-16). Never pads OH.
  Odd,
  // The divisor of OH (or of OH padded up to a multiple of numCols, when that
  // is cheaper) that minimises the measured per-column cost model
  // `fixed + perColumn * (cols - 1) + bytes / (bandwidth * cols)`.
  Cost,
  // Always numCols: OH is padded up to a multiple of numCols on the host when
  // it does not divide. Every pooling uses the full array.
  Full,
  // The largest divisor of OH that is <= numCols: all columns whenever OH
  // divides by numCols, else as many as divide it (7 for 7 / 14 / 28, 6 for
  // 6 / 12). Never pads OH, never consults the cost model.
  Max
};

enum class OutliningStrategy {
  // No outlining.
  None,
  // Try outlining all ops.
  All,
  // A balanced strategy trying to achieve good performance and low program
  // memory size.
  Balanced,
};

enum class PacketFlowStrategy {
  // No packet flow.
  None,
  // Congestion-aware packet flow assignment.
  Auto,
  // Use packet mode on all input flows.
  Inputs,
  // Use packet mode on all output flows.
  Outputs,
  // Use packet mode on all flows.
  All,
};

enum class AllocScheme { Sequential, BankAware, None };

/// Entry in `translation_info`'s configuration dictionary that overrides the
/// global `--iree-amdaie-tile-pipeline` for one dispatch, and the value that
/// selects the GEMV pipeline. Set by the lowering strategy from the root op's
/// shape (a module mixes conv matmuls and M=1 FC matmuls, so a global option
/// cannot pick per dispatch); read by `AMDAIELowerExecutableTarget`.
constexpr llvm::StringLiteral kTilePipelineOverrideName = "amdaie.tile_pipeline";
constexpr llvm::StringLiteral kGemvPipelineName = "gemv";
constexpr llvm::StringLiteral kPoolPipelineName = "pool";

LogicalResult initAIELaunchConfig(FunctionOpInterface funcOp,
                                  TilePassPipeline useTilePipeline,
                                  LowerToAIEPassPipeline useLowerToAIEPipeline,
                                  AMDAIEDevice targetDevice, uint32_t numRows,
                                  uint32_t numCols, uint32_t stackSize,
                                  std::string enableAMDAIEUkernels,
                                  PoolCoreSplit poolCoreSplit,
                                  PoolColumnsPolicy poolColumnsPolicy);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_
