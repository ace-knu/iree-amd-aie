// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Utils/AMDAIEDevicePlacementUtils.h"

namespace mlir::iree_compiler::AMDAIE {

StringRef getDeviceBackend(IREE::Util::GlobalOpInterface global) {
  auto targetAttr = dyn_cast_or_null<IREE::HAL::DeviceTargetAttr>(
      global.getGlobalInitialValue());
  if (!targetAttr) return {};
  ArrayRef<IREE::HAL::ExecutableTargetAttr> targets =
      targetAttr.getExecutableTargets();
  if (targets.empty()) return {};
  return targets.front().getBackend().getValue();
}

DeviceRole deviceRole(StringRef backend) {
  if (backend == "llvm-cpu") return DeviceRole::Host;
  if (backend == "amd-aie") return DeviceRole::Accelerator;
  return DeviceRole::Unknown;  // not managed by this plugin
}

LinkCapability linkFlags(StringRef srcBackend, StringRef dstBackend) {
  // `transparent_access` on a link (src -> dst) means "src can access dst's
  // memory without staging", so the two directions are asked separately and
  // here they differ:
  //
  //  - llvm-cpu -> amd-aie is true. An amdxdna allocation is a BO that is
  //    host-local and device-visible, so a CPU dispatch host-maps it directly.
  //  - amd-aie -> llvm-cpu is false. An allocation from the CPU allocator is
  //    not a BO, and the amdxdna allocator cannot wrap one (`import_buffer` is
  //    unimplemented), so the NPU cannot bind it at all. This is exactly why a
  //    buffer shared across the two devices has to be allocated on the
  //    accelerator.
  //
  // Declaring both directions true (as this did originally) says the NPU can
  // read CPU-allocated memory, which is the case that does not work.
  //
  // This is read directionally by more than the topology resolver:
  // Stream/Transforms/ElideAsyncCopies.cpp asks hasTransparentAccess(source,
  // result) before dropping a transfer, so an NPU->CPU transfer that the
  // symmetric declaration let it elide is now kept. That is the intended
  // outcome -- the NPU's result has to be copied into a CPU allocation for a
  // host dispatch to read it -- but it is a behavior change, not a no-op.
  if (srcBackend == "llvm-cpu" && dstBackend == "amd-aie")
    return {/*unifiedMemory=*/false, /*transparentAccess=*/true};
  return {};  // conservative default: stage (copy) across the boundary.
}

IREE::HAL::DeviceAffinityAttr getFirstDeviceWithRole(ModuleOp module,
                                                     DeviceRole role) {
  MLIRContext *ctx = module.getContext();
  for (auto global : module.getOps<IREE::Util::GlobalOpInterface>()) {
    if (deviceRole(getDeviceBackend(global)) != role) continue;
    return IREE::HAL::DeviceAffinityAttr::get(
        ctx, FlatSymbolRefAttr::get(ctx, global.getGlobalName()),
        /*queue_mask=*/-1ll);
  }
  return {};
}

static bool deviceAttrHasRole(Attribute attr, DeviceRole role) {
  if (auto target = dyn_cast_if_present<IREE::HAL::DeviceTargetAttr>(attr)) {
    return llvm::any_of(target.getExecutableTargets(),
                        [&](IREE::HAL::ExecutableTargetAttr executable) {
                          return deviceRole(
                                     executable.getBackend().getValue()) ==
                                 role;
                        });
  }
  if (auto select = dyn_cast_if_present<IREE::HAL::DeviceSelectAttr>(attr)) {
    return llvm::any_of(select.getDevices(), [&](Attribute candidate) {
      return deviceAttrHasRole(candidate, role);
    });
  }
  return false;
}

bool anyDeviceHasRole(ModuleOp module, DeviceRole role) {
  return llvm::any_of(
      module.getOps<IREE::Util::GlobalOpInterface>(),
      [&](IREE::Util::GlobalOpInterface global) {
        return deviceAttrHasRole(global.getGlobalInitialValue(), role);
      });
}

}  // namespace mlir::iree_compiler::AMDAIE
