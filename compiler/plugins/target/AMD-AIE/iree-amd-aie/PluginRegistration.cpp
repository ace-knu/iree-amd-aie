// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "aie/Passes.h"
#include "aievec/AIEVecDialect.h"
#include "aievec/Passes.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Passes.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Target/AIETarget.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"

namespace mlir::iree_compiler {
namespace {

namespace {
#define GEN_PASS_REGISTRATION
#include "aie/Passes.h.inc"
}  // namespace

struct AMDAIESession
    : public PluginSession<AMDAIESession, AMDAIE::AMDAIEOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    AMDAIE::registerAMDAIEPasses();
    registerAMDAIEAssignBufferAddresses();
    AMDAIE::registerAMDAIEAssignBufferDescriptorIDs();
    registerAMDAIECoreToStandard();
    AMDAIE::registerAMDAIELocalizeLocks();
    AMDAIE::registerAMDAIENormalizeAddressSpaces();
    registerAMDAIERouteFlowsWithPathfinder();
    AMDAIE::registerAMDAIEDmaToNpu();
    AMDAIE::registerAMDAIEIncrementRepeatCount();
    AMDAIE::registerAIRConversionPasses();
    AMDAIE::registerAIRTransformPasses();
    aievec::registerConvertAIEVecToLLVMPass();
    aievec::registerAlignTransferReadsPass();
    aievec::registerCanonicalizeVectorForAIEVecPass();
    aievec::registerLowerVectorToAIEVecPass();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<AMDAIE::AMDAIEDialect, xilinx::AIE::AIEDialect,
                    aievec::AIEVecDialect, xilinx::AIEX::AIEXDialect,
                    xilinx::air::airDialect>();
  }

  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    // Fold a broadcasting bias-add (matmul + an [N]-shaped bias, the common
    // nn.Linear/torch.aten.linear shape) into the matmul's own
    // zero-initialized accumulator, eliminating the separate elementwise op
    // entirely. Must run before dispatch region formation (this is the Flow
    // phase, not yet Dispatch Creation) so the fused result forms a single,
    // ordinary-looking matmul dispatch -- avoiding the multi-op dispatch
    // fusion machinery (separate elementwise consumer, its own DMA/tile
    // placement) that a detached bias-add would otherwise require.
    passManager.addPass(AMDAIE::createAMDAIEFoldBroadcastAddIntoDestPass());
    // Demote contraction (matmul + conv) inputs f32 -> bf16 before the named
    // ops are generalized (the upstream demote pass only matches named ops).
    // npu4 has no f32 vector path, so this is required to run f32 models.
    if (options.demoteContractionInputsToBf16) {
      passManager.addPass(GlobalOptimization::createDemoteContractionInputsPass(
          GlobalOptimization::DemoteType::BF16,
          GlobalOptimization::DemoteOperation::All));
    }
    // Inject the device topology now, while both device globals still exist
    // (an unused CPU device global would otherwise be DCE'd before the Flow
    // phase). The topology's symbol references keep the CPU device global
    // alive through SymbolDCE. No dispatches exist yet, so no affinity is set
    // here. No-op unless both an amd-aie and an llvm-cpu device are declared.
    passManager.addPass(AMDAIE::createAMDAIEAssignDeviceAffinitiesPass(
        softmaxUkernelEnabled()));
  }


  /// Whether the softmax microkernel is enabled. `--iree-amdaie-enable-ukernels`
  /// takes `none`, `all`, or a comma-separated list.
  bool softmaxUkernelEnabled() const {
    StringRef list(options.enableAMDAIEUkernels);
    if (list == "all") return true;
    SmallVector<StringRef> names;
    list.split(names, ',');
    return llvm::is_contained(names, "softmax");
  }

  void extendFlowTransformPassPipeline(OpPassManager &passManager) override {
    // Heterogeneous placement: pin contraction/conv dispatches to the amd-aie
    // (NPU) device and everything else (transposes, casts) to the CPU device.
    // No-op unless both an amd-aie and an llvm-cpu device are declared.
    passManager.addPass(AMDAIE::createAMDAIEAssignDeviceAffinitiesPass(
        softmaxUkernelEnabled()));
    // With affinity known per-dispatch, pad NPU contraction operands up to the
    // target's pack-peel tile multiples so divisibility holds inside the
    // dispatch. No-op for dispatches already divisible / not on amd-aie.
    passManager.addPass(AMDAIE::createAMDAIEPadContractionDispatchesPass());
    // Split large-(K,N) transpose_b NPU matmul dispatches (padded above) into
    // N-chunk sub-dispatches so each weight's L3->L2 shim DMA stays within the
    // shim addressing limits (a single large-N weight DMA degrades and stalls).
    // No-op for conv / small matmuls / non-amd-aie dispatches.
    passManager.addPass(
        AMDAIE::createAMDAIESplitLargeContractionDispatchesPass());
  }

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    // #hal.device.target<"xrt", ...
    targets.add("xrt", [=] {
      options.deviceHal = AMDAIE::AMDAIEOptions::DeviceHAL::XRT;
      return AMDAIE::createTarget(options);
    });
    // #hal.device.target<"amdxdna", ...
    targets.add("amdxdna", [=] {
      options.deviceHal = AMDAIE::AMDAIEOptions::DeviceHAL::AMDXDNA;
      return AMDAIE::createTarget(options);
    });
  }

  void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    targets.add("amd-aie", [=]() { return AMDAIE::createBackend(options); });
  }
};

}  // namespace
}  // namespace mlir::iree_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::AMDAIE::AMDAIEOptions);

extern "C" bool iree_register_compiler_plugin_amd_aie(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::AMDAIESession>("amd_aie");
  return true;
}
