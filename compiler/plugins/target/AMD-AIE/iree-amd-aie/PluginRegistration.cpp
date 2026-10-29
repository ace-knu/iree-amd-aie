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
    // Demote contraction (matmul + conv) inputs f32 -> bf16 before the named
    // ops are generalized (the upstream demote pass only matches named ops).
    // npu4 has no f32 vector path, so this is required to run f32 models.
    if (options.demoteContractionInputsToBf16) {
      passManager.addPass(GlobalOptimization::createDemoteContractionInputsPass(
          GlobalOptimization::DemoteType::BF16,
          GlobalOptimization::DemoteOperation::All));
    }
    // Move NCHW max pooling to NHWC so it matches the channels-last convs
    // (the upstream channels-last pass skips pooling); the transposes this
    // inserts cancel against the conv-side ones downstream.
    // The pool pipeline splits C over the array rows, so C is padded up to a
    // multiple of the row count here (channels are independent under max, the
    // extra ones are sliced off the result).
    if (options.poolingChannelsLast) {
      AMDAIE::AMDAIEDeviceModel deviceModel =
          AMDAIE::getDeviceModel(options.AMDAIETargetDevice);
      AMDAIE::AMDAIEConvertPoolingToChannelsLastOptions poolOptions;
      poolOptions.channelMultiple = options.getNumRows(deviceModel);
      poolOptions.haloSplitColumns = options.getNumCols(deviceModel);
      poolOptions.columnsPolicy = options.poolColumnsPolicy;
      passManager.addPass(
          AMDAIE::createAMDAIEConvertPoolingToChannelsLastPass(poolOptions));
    }
    // Inject the device topology now, while both device globals still exist
    // (an unused CPU device global would otherwise be DCE'd before the Flow
    // phase). The topology's symbol references keep the CPU device global
    // alive through SymbolDCE. No dispatches exist yet, so no affinity is set
    // here. Requires a host device; the topology is injected only when two or
    // more devices are used.
    passManager.addPass(AMDAIE::createAMDAIEAssignDeviceAffinitiesPass());
  }

  void extendStreamTransformPassPipeline(OpPassManager &passManager) override {
    // Resolve optimal<[...]> buffer owners from stream.topology (replaces the
    // runtime "amdxdna" rule). No-op without a topology or on ties.
    passManager.addPass(AMDAIE::createAMDAIEResolveBufferOwnerPass());
    // Record per-argument input devices (amdaie.input_devices reflection) from
    // the now-final stream.tensor.import affinities, so tooling can place
    // inputs without --input_device. Attribute-only; no-op without topology.
    passManager.addPass(AMDAIE::createAMDAIEAnnotateInputDevicesPass());
  }

  void extendFlowTransformPassPipeline(OpPassManager &passManager) override {
    // Heterogeneous placement: pin contraction/conv dispatches to the
    // accelerators (round-robin) and everything else (transposes, casts) to the
    // host. Requires a host device.
    // --iree-amdaie-placement-map (experiment, default empty) pins dispatches
    // by number/name rules. --iree-amdaie-accelerate-pooling adds max pooling
    // to the accelerator work.
    AMDAIE::AMDAIEAssignDeviceAffinitiesOptions placementOptions;
    placementOptions.placementMap = options.placementMap;
    placementOptions.acceleratePooling = options.acceleratePooling;
    passManager.addPass(
        AMDAIE::createAMDAIEAssignDeviceAffinitiesPass(placementOptions));
    // Fold a constant bias fused into an NPU contraction dispatch into K, so
    // the dispatch keeps two inputs (then padded like any other K below).
    // No-op without such a dispatch.
    passManager.addPass(
        AMDAIE::createAMDAIEFoldBiasIntoContractionDispatchesPass());
    // With affinity known per-dispatch, pad NPU contraction operands up to the
    // target's pack-peel tile multiples so divisibility holds inside the
    // dispatch. No-op for dispatches already divisible / not on amd-aie.
    passManager.addPass(AMDAIE::createAMDAIEPadContractionDispatchesPass());
    // Likewise pad NPU max pooling dispatches' OH up to what the target's pool
    // column policy asks for (`full`: a multiple of the column count), so the
    // pool pipeline can spread OH over every column. Done here, not in the
    // preprocessing, because the dispatch's init fill is already inside the
    // dispatch and just grows in place (a fill created before dispatch
    // formation gets hoisted out as a splat the pool pipeline cannot take).
    passManager.addPass(AMDAIE::createAMDAIEPadPoolingDispatchesPass());
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
