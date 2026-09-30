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
                                  PoolCoreSplit poolCoreSplit);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_
