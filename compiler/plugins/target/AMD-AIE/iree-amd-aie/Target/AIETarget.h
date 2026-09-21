// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_AIETARGET_H_
#define IREE_AMD_AIE_TARGET_AIETARGET_H_

#include <string>

#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler::AMDAIE {

struct AMDAIEOptions {
  std::string amdAieInstallDir;

  // Path to Peano installation directory.
  std::string peanoInstallDir;

  // Path to Vitis installation directory.
  std::string vitisInstallDir;

  // Dump system commands used during compilation
  bool showInvokedCommands{false};

  // Use the chess compiler. The default is to use peano.
  bool useChess{false};

  // Use the chess compiler for ukernel. The default is to use chess.
  bool useChessForUKernel{true};

  // Additional flags to run peano's opt with (if peano is the backend compiler
  // selected). These are mostly appended on the end of the default flags, but
  // some flags may replace existing flags if they conflict.
  std::string additionalPeanoOptFlags;

  // Print IR after all MLIR passes run in aie2xclbin (to stderr).
  bool aie2xclbinPrintIrAfterAll{false};

  // Print IR before all MLIR passes run in aie2xclbin (to stderr).
  bool aie2xclbinPrintIrBeforeAll{false};

  // Print IR at module scope in MLIR passes in aie2xclbin.
  bool aie2xclbinPrintIrModuleScope{false};

  // Print MLIR timing summary for the MLIR passes in aie2xclbin.
  bool aie2xclbinTiming{false};

  LowerToAIEPassPipeline useLowerToAIEPipeline{
      LowerToAIEPassPipeline::ObjectFifo};
  TilePassPipeline useTilePipeline{TilePassPipeline::PackPeelPipeline};
  bool enableVectorizationPasses{true};

  // Demote the inputs of contraction ops (matmul + convolution) from f32 to
  // bf16 during preprocessing; accumulation stays f32. npu4 (AIE2P) has no f32
  // vector path, so f32 conv aborts and f32 matmul only runs scalar; demoting
  // makes them run as bf16. Applied in preprocessing (before named ops are
  // generalized) because the upstream demote pass only matches named ops.
  // Note this is module-wide and runs before device placement, so in a
  // heterogeneous compile a contraction that ends up on the host is demoted
  // too. Harmless while every contraction is placed on the accelerator; revisit
  // if the placement rules change.
  bool demoteContractionInputsToBf16{false};

  // Fold conv2d + bias here, before im2col, by augmenting the *activation*
  // with a constant-one input channel (K+1 on the convolution). Off by
  // default: `AMDAIEPadContractionDispatches` folds the same bias into a
  // single K column of the post-im2col matmul instead, which is strictly
  // cheaper on a convolution whose weight and bias are compile-time
  // constants. K = C*kh*kw, so one extra input channel buys nine extra K
  // positions of which eight are structurally zero, and it costs a full copy
  // of the activation into a (C+1)-channel buffer plus a fill for the ones
  // plane -- two host dispatches per convolution. On VGG16 that is 26 of
  // 105 dispatches; folding at the K column instead brings the model to 79
  // with the ones written by the K-padding dispatch that has to run anyway
  // (verified on npu4: corr 0.999994, argmax match; see
  // docs/2026-09-10_cpu-dispatch-reduction-plan.md).
  //
  // Turn it back on for a convolution whose weight or bias is *not* a
  // compile-time constant: the K-column fold has to materialize the bias row
  // as a constant, so it does not apply there, and without either fold the
  // bias stays a third dispatch operand -- which the NPU cannot bind
  // ("'amdaie.connection' op no producer DMA channel available", every conv).
  // Matmul + bias is folded here regardless of this flag; it is the
  // convolution path alone that this gates.
  bool foldConvBiasIntoContraction{false};

  // Debug-only escape hatches for bisecting the 2026-09-04 fold-bias change
  // (see AMDAIEFoldBiasIntoContraction.cpp). Defaults reproduce today's
  // verified-working behavior; false / 1 reproduce the pre-2026-09-04
  // behavior.
  bool foldRhsTransposeIntoContraction{true};
  int64_t foldBiasKAlign{128};

  // Max N per AMDAIESplitLargeContractionDispatchesPass split chunk (see its
  // own comment in AMDAIEPadContractionDispatches.cpp). 1024 is the largest
  // value validated on real hardware; 2048 fails reproducibly (`ert state 8`,
  // 2/2). A wrong value does not miscompute, it hangs the NPU run, so any
  // other value needs its own real-hardware verification before being
  // trusted. See docs/2026-09-09c_nchunk1024-measured-benefit-data-movement-
  // and-time.md for the 512-vs-1024 measurement this default is based on.
  int64_t splitLargeContractionNChunkMax{1024};

  bool enableCoalescingLoops{false};
  bool enableCollapsingUnitDims{false};
  OutliningStrategy enableFunctionOutlining{OutliningStrategy::Balanced};
  int callReplication{1};
  bool insertLoopAroundCoreBlock{false};
  bool matmulElementwiseFusion{false};
  // Bound to a single global flag, and this options struct is per plugin
  // session rather than per declared device, so every amd-aie device global in
  // a compile gets the same target_device / num_rows / num_cols. Declaring two
  // amd-aie devices of *different* models (say an npu4 and an npu1_4col in one
  // compile) is therefore not expressible today. A second NPU from another
  // vendor is fine -- that is a separate plugin with its own options struct --
  // so this only binds when two devices of this plugin need to differ.
  AMDAIEDevice AMDAIETargetDevice{AMDAIEDevice::npu1_4col};

  // Enable/Disable reprogramming of DMAs.
  bool reprogramDmas{false};

  // The number of rows for the compiler to target. '0' denotes 'all'.
 private:
  unsigned AMDAIENumRows{0};

 public:
  unsigned getNumRows(const AMDAIEDeviceModel &model) const {
    return AMDAIENumRows == 0 ? model.getNumCoreRows() : AMDAIENumRows;
  }

  // The number of rows for the compiler to target. '0' denotes 'all'.
 private:
  unsigned AMDAIENumCols{0};

 public:
  unsigned getNumCols(const AMDAIEDeviceModel &model) const {
    return AMDAIENumCols == 0 ? model.getNumCoreCols() : AMDAIENumCols;
  }

  std::string enableAMDAIEUkernels{"none"};
  // Route small-M (GEMV-like) matmuls to the GEMV tile pipeline and skip their
  // M/N padding and N-splitting (KernelDispatch, AMDAIEPadContractionDispatches).
  // Off restores the pack-peel-only behaviour, for A/B comparison and rollback.
  bool enableGemvPipeline{true};
  PacketFlowStrategy packetFlowStrategy{PacketFlowStrategy::None};
  bool enableCtrlPkt{false};

  enum class DeviceHAL { XRT, AMDXDNA };
  DeviceHAL deviceHal{DeviceHAL::AMDXDNA};

  // The default stack size for all cores is 1024 bytes.
  uint32_t coreStackSize{1024};

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("AMD AIE Options");

    binder.opt<std::string>(
        "iree-amd-aie-additional-peano-opt-flags", additionalPeanoOptFlags,
        llvm::cl::cat(category),
        llvm::cl::desc("Additional flags for peano's opt. Example: "
                       "\"-O3 --magic-flag\"."));

    binder.opt<std::string>(
        "iree-amd-aie-install-dir", amdAieInstallDir, llvm::cl::cat(category),
        llvm::cl::desc("Path to AMDAIE installation directory (typically the "
                       "IREE install directory)"));

    binder.opt<std::string>(
        "iree-amd-aie-peano-install-dir", peanoInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to Peano installation directory"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-after-all", aie2xclbinPrintIrAfterAll,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, print the IR after all MLIR passes run in aie2xclbin"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-before-all", aie2xclbinPrintIrBeforeAll,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, print the IR before all MLIR passes run in aie2xclbin"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-module-scope", aie2xclbinPrintIrModuleScope,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, when printing the IR do so at the module scope"));

    binder.opt<bool>(
        "aie2xclbin-timing", aie2xclbinTiming, llvm::cl::cat(category),
        llvm::cl::desc("If true, print MLIR timing summary for the MLIR passes "
                       "in aie2xclbin"));

    binder.opt<bool>(
        "iree-amd-aie-show-invoked-commands", showInvokedCommands,
        llvm::cl::cat(category),
        llvm::cl::desc("Show commands invoked during binary generation"));

    binder.opt<std::string>(
        "iree-amd-aie-vitis-install-dir", vitisInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to aietools in Vitis installation"));

    binder.opt<bool>("iree-amd-aie-enable-chess", useChess,
                     llvm::cl::cat(category),
                     llvm::cl::desc("Use the legacy chess compiler"));

    binder.opt<bool>(
        "iree-amdaie-demote-contraction-inputs-to-bf16",
        demoteContractionInputsToBf16, llvm::cl::cat(category),
        llvm::cl::desc(
            "Demote the inputs of contraction ops (matmul + convolution) from "
            "f32 to bf16 during preprocessing (accumulation stays f32). Needed "
            "to run f32 models on npu4, which has no f32 vector path."));

    binder.opt<bool>(
        "iree-amdaie-fold-conv-bias-into-contraction",
        foldConvBiasIntoContraction, llvm::cl::cat(category),
        llvm::cl::desc(
            "Fold conv2d + bias before im2col by augmenting the activation "
            "with a constant-one input channel. Off by default because "
            "AMDAIEPadContractionDispatches folds the same bias into one K "
            "column of the post-im2col matmul at no dispatch cost (VGG16: "
            "105 -> 79 dispatches). Enable for a convolution whose weight or "
            "bias is not a compile-time constant, which the K-column fold "
            "cannot handle. Matmul + bias is folded regardless."));

    binder.opt<bool>(
        "iree-amdaie-fold-rhs-transpose-into-contraction",
        foldRhsTransposeIntoContraction, llvm::cl::cat(category),
        llvm::cl::desc(
            "Debug-only: fold a materialized linalg.transpose feeding the "
            "matmul RHS into the matmul's own indexing maps (transpose_b "
            "form) before the K+1 bias fold. false reproduces the "
            "pre-2026-09-04 behavior. For bisecting the fold-bias change; "
            "not meant for general use."));

    binder.opt<int64_t>(
        "iree-amdaie-fold-bias-k-align", foldBiasKAlign,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Debug-only: grow the K+1 bias-fold augmentation up to the next "
            "multiple of this value instead of stopping at exactly K+1. 1 "
            "reproduces the pre-2026-09-04 behavior. For bisecting the "
            "fold-bias change; not meant for general use."));

    binder.opt<int64_t>(
        "iree-amdaie-split-n-chunk-max", splitLargeContractionNChunkMax,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Max N per AMDAIESplitLargeContractionDispatchesPass split chunk. "
            "The built-in 1024 is the largest value validated on real "
            "hardware (VGG-16, bf16, npu4: 7/7 runs correct); 2048 fails "
            "reproducibly with an NPU timeout. Any other value needs its own "
            "real-hardware re-verification -- a wrong value hangs the NPU run "
            "rather than miscomputing."));

    binder.opt<bool>(
        "iree-amd-aie-enable-chess-for-ukernel", useChessForUKernel,
        llvm::cl::cat(category),
        llvm::cl::desc("Use the chess compiler for compiling ukernels"));

    binder.opt<std::string>(
        "iree-amdaie-enable-ukernels", enableAMDAIEUkernels,
        llvm::cl::cat(category),
        llvm::cl::desc("Enables microkernels in the amdaie backend. May be "
                       "`none`, `all`, or a comma-separated list of specific "
                       "unprefixed microkernels to enable, e.g. `matmul`."));

    binder.opt<bool>(
        "iree-amdaie-enable-gemv-pipeline", enableGemvPipeline,
        llvm::cl::cat(category),
        llvm::cl::desc("Lower small-M (GEMV-like) matmuls with the GEMV tile "
                       "pipeline instead of pack-peel (and skip their M/N "
                       "padding and N-splitting). Default true."));

    /// Command line option for selecting the lowering pipeline to use to
    /// generate AIE DMA configurations, core code and control code.
    binder.opt<LowerToAIEPassPipeline>(
        "iree-amdaie-lower-to-aie-pipeline", useLowerToAIEPipeline,
        llvm::cl::cat(category),
        llvm::cl::desc("Pick the lowering pipeline to use"),
        llvm::cl::values(
            clEnumValN(LowerToAIEPassPipeline::AIR, "air",
                       "Use the IREE lowering through AIR"),
            clEnumValN(LowerToAIEPassPipeline::ObjectFifo, "objectFifo",
                       "Use the IREE lowering to objectFifos")));

    /// Command line option for selecting the lowering pipeline to use tiling
    /// computations and packing data.
    binder.opt<TilePassPipeline>(
        "iree-amdaie-tile-pipeline", useTilePipeline, llvm::cl::cat(category),
        llvm::cl::desc("Pick the lowering pipeline to use"),
        llvm::cl::values(
            clEnumValN(TilePassPipeline::PackPeelPipeline, "pack-peel",
                       "Use the pack-peel based lowering strategy for "
                       "matmul-like ops"),
            clEnumValN(TilePassPipeline::PackPeel4LevelTilingPipeline,
                       "pack-peel-4-level-tiling",
                       "Use the pack-peel based lowering strategy with 4 "
                       "levels of tiling for "
                       "matmul-like ops"),
            clEnumValN(TilePassPipeline::ConvDecomposePipeline,
                       "conv-decompose",
                       "Use the conv-decompose based lowering strategy for "
                       "convolution interface ops"),
            clEnumValN(TilePassPipeline::GeneralCopyPipeline, "general-copy",
                       "Use the copy based lowering strategy for elementwise "
                       "or reduction ops")));

    binder.opt<bool>(
        "iree-amdaie-enable-vectorization-passes", enableVectorizationPasses,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Some pipelines (see iree-amdaie-tile-pipeline) may include "
            "vectorization passes. This option enables or disables "
            "these vectorization passes. It is intended for development "
            "purposes only."));

    binder.opt<bool>(
        "iree-amdaie-enable-coalescing-loops", enableCoalescingLoops,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Pass insert-loops-for-vectorization may disable/enable loop "
            "coalescing depending on this pass flag. It is intended for "
            "development purposes only."));

    binder.opt<bool>(
        "iree-amdaie-enable-collapsing-unit-dims", enableCollapsingUnitDims,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Pass insert-loops-for-vectorization may disable/enable collapsing "
            "unit dims of a tensor/memref depending on this pass flag. It is "
            "intended for development purposes only."));

    binder.opt<OutliningStrategy>(
        "iree-amdaie-enable-function-outlining", enableFunctionOutlining,
        llvm::cl::cat(category),
        llvm::cl::desc("Flag to enable/disable linalg-function-outlining pass."
                       "It is intended for development purposes only."),
        llvm::cl::values(clEnumValN(OutliningStrategy::None, "none",
                                    "No linalg ops will be outlined."),
                         clEnumValN(OutliningStrategy::All, "all",
                                    "All linalg ops will be outlined."),
                         clEnumValN(OutliningStrategy::Balanced, "balanced",
                                    "Will outline some ops, to try to achieve "
                                    "a good balance between "
                                    "performance and program size.")));

    binder.opt<int>(
        "iree-amdaie-call-replication", callReplication,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "The number of calls to outlined function. n!=1 will result "
            "in numerical errors. Useful for benchmarking data movement "
            "(n=0) and core performance (n>>1). For development purposes "
            "only."));

    binder.opt<bool>(
        "iree-amdaie-enable-infinite-loop-around-core-block",
        insertLoopAroundCoreBlock, llvm::cl::cat(category),
        llvm::cl::desc("Flag to enable/disable insertion of loops around the "
                       "core blocks. Typically only used for granular "
                       "peformance measurement purposes."));

    binder.opt<bool>(
        "iree-amdaie-matmul-elementwise-fusion", matmulElementwiseFusion,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "This option enables/disables special passes in MLIR-AIR "
            "for matmul-elementwise fusion. It is currently added for "
            "development purpose and should be removed in the future."));

    binder.opt<bool>(
        "iree-amdaie-reprogram-dmas", reprogramDmas, llvm::cl::cat(category),
        llvm::cl::desc(
            "Flag used to enable/disable reprogramming of DMAs. "
            "By default it'll be disabled, so we would have circular "
            "DMAs for L2/L1 cache."));

    /// Command line option for selecting the target AIE device.
    binder.opt<AMDAIEDevice>(
        "iree-amdaie-target-device", AMDAIETargetDevice,
        llvm::cl::cat(category),
        llvm::cl::desc("Sets the target device architecture."),
        llvm::cl::values(
            clEnumValN(AMDAIEDevice::xcvc1902, "xcvc1902",
                       "The xcvc1902 device"),
            clEnumValN(AMDAIEDevice::xcve2302, "xcve2302",
                       "The xcve2302 device"),
            clEnumValN(AMDAIEDevice::xcve2802, "xcve2802",
                       "The xcve2802 device"),
            clEnumValN(AMDAIEDevice::npu1, "npu1", "Default Phoenix NPU"),
            clEnumValN(AMDAIEDevice::npu1_1col, "npu1_1col",
                       "Phoenix NPU with a single column"),
            clEnumValN(AMDAIEDevice::npu1_2col, "npu1_2col",
                       "Phoenix NPU with two columns"),
            clEnumValN(AMDAIEDevice::npu1_3col, "npu1_3col",
                       "Phoenix NPU with three columns"),
            clEnumValN(AMDAIEDevice::npu1_4col, "npu1_4col",
                       "Phoenix NPU with four columns"),
            clEnumValN(AMDAIEDevice::npu4, "npu4",
                       "Strix B0 NPU with 8 columns and 6 rows")));

    binder.opt<unsigned>(
        "iree-amdaie-num-rows", AMDAIENumRows, llvm::cl::cat(category),
        llvm::cl::desc(
            "Number of rows used in an AIE core array. The compiler will "
            "choose a tiling strategy that uses no more than this number of "
            "rows. However, some workloads (like convolution) currently ignore "
            "this flag, and use a hardcoded number of rows."));

    binder.opt<unsigned>(
        "iree-amdaie-num-cols", AMDAIENumCols, llvm::cl::cat(category),
        llvm::cl::desc(
            "Number of columns used in an AIE core array. The compiler will "
            "choose a tiling strategy that uses no more than this number of "
            "columns. However, some workloads (like convolution) currently "
            "ignore this flag, and use a hardcoded number of cols."));

    binder.opt<PacketFlowStrategy>(
        "iree-amdaie-packet-flow-strategy", packetFlowStrategy,
        llvm::cl::cat(category),
        llvm::cl::desc("Enable packet routing data movements"),
        llvm::cl::values(clEnumValN(PacketFlowStrategy::None, "none",
                                    "No packet flow will be used."),
                         clEnumValN(PacketFlowStrategy::Auto, "auto",
                                    "Congestion-aware packet flow assignment."),
                         clEnumValN(PacketFlowStrategy::Inputs, "inputs",
                                    "Use packet mode on all input flows."),
                         clEnumValN(PacketFlowStrategy::Outputs, "outputs",
                                    "Use packet mode on all output flows."),
                         clEnumValN(PacketFlowStrategy::All, "all",
                                    "Use packet mode on all flows.")));

    binder.opt<DeviceHAL>(
        "iree-amdaie-device-hal", deviceHal, llvm::cl::cat(category),
        llvm::cl::desc("Sets the target device HAL."),
        llvm::cl::values(
            clEnumValN(DeviceHAL::XRT, "xrt", "xrt device HAL"),
            clEnumValN(DeviceHAL::AMDXDNA, "amdxdna", "amdxdna device HAL")));

    binder.opt<bool>(
        "iree-amdaie-enable-control-packet", enableCtrlPkt,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Enable conversion of `aie.device` "
            "operations into control packets for fast reconfiguration."));

    binder.opt<unsigned>(
        "iree-amdaie-stack-size", coreStackSize, llvm::cl::cat(category),
        llvm::cl::desc("The stack size to be used for the AIE cores."));
  }
};

// Creates the default AIE target.
std::shared_ptr<IREE::HAL::TargetDevice> createTarget(
    const AMDAIEOptions &options);

// Creates the default AIE backend.
std::shared_ptr<IREE::HAL::TargetBackend> createBackend(
    const AMDAIEOptions &options);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_AIETARGET_H_
