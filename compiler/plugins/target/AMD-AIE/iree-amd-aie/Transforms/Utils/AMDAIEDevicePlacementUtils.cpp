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
  if (backend == "vulkan-spirv") return DeviceRole::Accelerator;  // TEMP: experiment
  return DeviceRole::Unknown;  // not managed by this plugin
}

LinkCapability linkFlags(StringRef srcBackend, StringRef dstBackend) {
  // `transparent_access` on a link (src -> dst) means "src can access memory
  // allocated by dst without a staging copy". The two directions of a pair
  // are independent and are declared separately below.
  //
  // This table is the single source of truth for which device may own a
  // buffer shared across a device boundary: a device X can own such a buffer
  // only if every other user Y has (Y -> X) = true. Change a row here when a
  // driver gains or loses an import/export path; nothing else should need to
  // encode device names.
  //
  // Consumers:
  //  - ResolveTopologyQueries (buffer usage / mapping bits).
  //  - ElideAsyncCopies asks hasTransparentAccess(source, result) before
  //    dropping a transfer, so flipping a row to true can remove a copy that
  //    was previously kept. Treat every change here as a behavior change and
  //    re-run the boundary tests (chain2, mlp_reduce, cpu_gpu_reduce).
  //
  // Unlisted pairs fall through to the conservative default at the bottom.

  // ---- CPU <-> NPU --------------------------------------------------------
  //  - llvm-cpu -> amd-aie is true. An amdxdna allocation is a BO that is
  //    host-local and device-visible, so a CPU dispatch host-maps it directly.
  if (srcBackend == "llvm-cpu" && dstBackend == "amd-aie")
    return {/*unifiedMemory=*/false, /*transparentAccess=*/true};
  //  - amd-aie -> llvm-cpu is false. The amdxdna HAL allocator has no
  //    `import_buffer`, so the NPU cannot bind a CPU (heap) allocation. This
  //    is a software limitation of the HAL driver, not of the hardware: the
  //    kernel driver handles imported dma-buf and userptr BOs.
  if (srcBackend == "amd-aie" && dstBackend == "llvm-cpu")
    return {/*unifiedMemory=*/false, /*transparentAccess=*/false};

  // ---- CPU <-> GPU --------------------------------------------------------
  //  - llvm-cpu -> vulkan-spirv is true. Vulkan dispatch allocations are
  //    selected as DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT on this platform,
  //    so a CPU dispatch host-maps them directly.
  //    Verified: cpu_gpu_reduce, GPU owner (lead=1) -> 1.
  //    NOTE: on a discrete GPU whose dispatch memory is not host-visible this
  //    row must be false.
  if (srcBackend == "llvm-cpu" && dstBackend == "vulkan-spirv")
    return {/*unifiedMemory=*/false, /*transparentAccess=*/true};
  //  - vulkan-spirv -> llvm-cpu is declared true as the target behavior: the
  //    Vulkan allocator can import a host pointer (HOST_ALLOCATION via
  //    VK_EXT_external_memory_host).
  //    TODO: not honored at runtime yet. The dispatch binding bridge in
  //    vulkan/direct_command_buffer.cc only requests OPAQUE_FD from the
  //    owning allocator; the heap allocator cannot export that, so a
  //    CPU-owned buffer bound to a GPU dispatch fails ("export failed",
  //    segfault; cpu_gpu_reduce with lead=0). Add a HOST_ALLOCATION fallback
  //    there before relying on this row.
  if (srcBackend == "vulkan-spirv" && dstBackend == "llvm-cpu")
    return {/*unifiedMemory=*/false, /*transparentAccess=*/true};

  // ---- GPU <-> NPU --------------------------------------------------------
  //  - vulkan-spirv -> amd-aie is true. The amdxdna allocator exports a BO as
  //    a dma-buf fd (OPAQUE_FD), and the Vulkan allocator imports it as
  //    VkDeviceMemory; the dispatch binding bridge does this automatically.
  //    Verified: chain2 -> 32768, "import binding ok fd=N".
  if (srcBackend == "vulkan-spirv" && dstBackend == "amd-aie")
    return {/*unifiedMemory=*/false, /*transparentAccess=*/true};
  //  - amd-aie -> vulkan-spirv is false. amdxdna has no `import_buffer`, and
  //    the Vulkan allocator cannot export, so the NPU cannot bind a
  //    GPU-allocated buffer. A buffer shared by the two must be NPU-owned.
  //    Verified: setting this false still compiles and runs (chain2 -> 32768).
  if (srcBackend == "amd-aie" && dstBackend == "vulkan-spirv")
    return {/*unifiedMemory=*/false, /*transparentAccess=*/false};

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

}  // namespace mlir::iree_compiler::AMDAIE
