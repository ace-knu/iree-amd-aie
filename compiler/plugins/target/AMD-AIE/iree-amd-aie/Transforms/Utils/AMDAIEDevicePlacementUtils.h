// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Device-capability table for heterogeneous accelerator + host placement.
//
// This is the single source of truth for "what role does a device backend play"
// and "what can two backends do across their boundary". Every pass that needs
// to find a host or an accelerator device must ask here rather than testing the
// backend name itself -- otherwise adding a backend to the table does not
// actually reach the code that needs to know about it.
//
// See AMDAIEAssignDeviceAffinities.cpp for what the table does *not* cover.

#ifndef IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIEDEVICEPLACEMENTUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIEDEVICEPLACEMENTUTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler::AMDAIE {

/// Role of a device backend: Accelerator = a backend contraction/convolution is
/// offloaded to; Host = the CPU fallback for everything else.
enum class DeviceRole { Host, Accelerator, Unknown };

/// Directed link capability between two device backends. IREE does not derive
/// the topology (it only consumes it), and `transparent_access` is a per-pair
/// hardware property, so this defaults to staging (no transparent access) and
/// only known-good pairs are marked true.
struct LinkCapability {
  bool unifiedMemory = false;
  bool transparentAccess = false;
};

/// Returns the backend of the device global's target attr (e.g. "amd-aie",
/// "llvm-cpu"), or empty if the global is not a single device target.
StringRef getDeviceBackend(IREE::Util::GlobalOpInterface global);

/// Role of a device backend. Supporting new hardware starts by adding an entry
/// here; see the header comment in AMDAIEAssignDeviceAffinities.cpp for what
/// else is needed.
DeviceRole deviceRole(StringRef backend);

/// Link capability from `srcBackend` to `dstBackend`.
LinkCapability linkFlags(StringRef srcBackend, StringRef dstBackend);

/// Returns an affinity referring to the first device global in `module` whose
/// backend has `role`, or null if there is none.
IREE::HAL::DeviceAffinityAttr getFirstDeviceWithRole(ModuleOp module,
                                                     DeviceRole role);

/// Whether any device global in `module` can run a backend with `role`,
/// looking at every executable target of a `#hal.device.target` and every
/// candidate of a `#hal.device.select` -- unlike `getDeviceBackend`, which
/// reads only a plain target's first executable target. For a module-wide
/// question ("is this compile for amd-aie at all?") rather than placement.
bool anyDeviceHasRole(ModuleOp module, DeviceRole role);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_UTILS_AMDAIEDEVICEPLACEMENTUTILS_H_
