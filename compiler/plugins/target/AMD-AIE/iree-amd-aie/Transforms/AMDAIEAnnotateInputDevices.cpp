// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Records, for every public entry point, the device each input argument should
// be placed on, so tooling can place inputs without --input_device.
//
// Runs at the end of the stream pipeline (after ResolveBufferOwner), where the
// `stream.tensor.import on(<affinity>) %argK` ops are final. An import does not
// move data: it only asserts the buffer is usable by that device. When several
// devices import the same argument (CloneToConsumers makes one import per
// consumer device) the buffer must live where all of them can read it, which
// is the same rule ResolveBufferOwner applies to boundary buffers:
//   Q = { X in importers | every importer Y has (Y -> X) transparent_access }.
//
// Written into the function's `iree.reflection` dictionary:
//   amdaie.input_devices         = "amdxdna,amdxdna,vulkan"  (device target IDs)
//   amdaie.input_device_ordinals = "0,0,1"                    (compile order)
// The runtime reads only amdaie.input_devices; the ordinals let a vmfb's input
// placement be checked against the compile order (--iree-hal-target-device
// indices) without the device IDs. Per argument: one ID when |Q| == 1, "a|b" (compile order) on ties, "?" when
// no importer qualifies (a warning is emitted; tooling must refuse to guess),
// and an empty entry when the argument is never imported (scalars etc.).
// Sync wrappers that only `util.call` an annotated `$async` function receive
// the callee's entries remapped to their own argument numbers.
//
// Attribute-only: the IR is otherwise unchanged. No-op without stream.topology.

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

constexpr StringLiteral kReflectionAttr = "iree.reflection";
constexpr StringLiteral kDevicesKey = "amdaie.input_devices";
constexpr StringLiteral kOrdinalsKey = "amdaie.input_device_ordinals";

struct DeviceInfo {
  int64_t ordinal = -1;
  std::string id;  // #hal.device.target<"id">; empty if not a plain target.
};

static bool isEntryPoint(IREE::Util::FuncOp funcOp) {
  return funcOp.isPublic() && funcOp->hasAttr("iree.abi.stub") &&
         !funcOp.isExternal();
}

static void setReflectionEntries(IREE::Util::FuncOp funcOp,
                                 ArrayRef<std::string> devices,
                                 ArrayRef<std::string> ordinals) {
  MLIRContext *ctx = funcOp.getContext();
  NamedAttrList entries;
  if (auto existing = funcOp->getAttrOfType<DictionaryAttr>(kReflectionAttr))
    entries.append(existing.getValue());
  entries.set(kDevicesKey, StringAttr::get(ctx, llvm::join(devices, ",")));
  entries.set(kOrdinalsKey, StringAttr::get(ctx, llvm::join(ordinals, ",")));
  funcOp->setAttr(kReflectionAttr, entries.getDictionary(ctx));
}

// Splits a comma-separated reflection value keeping empty entries.
static SmallVector<std::string> splitEntries(StringRef value) {
  SmallVector<StringRef> parts;
  value.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  return llvm::map_to_vector(parts, [](StringRef s) { return s.str(); });
}

class AMDAIEAnnotateInputDevicesPass
    : public impl::AMDAIEAnnotateInputDevicesBase<
          AMDAIEAnnotateInputDevicesPass> {
 public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto topology = moduleOp->getAttrOfType<IREE::HAL::DeviceTopologyAttr>(
        "stream.topology");
    if (!topology) return;

    // Device globals in declaration order: @__device_N -> (N, target ID).
    llvm::StringMap<DeviceInfo> devices;
    int64_t nextOrdinal = 0;
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      if (!isa<IREE::HAL::DeviceType>(globalOp.getType())) continue;
      DeviceInfo info;
      info.ordinal = nextOrdinal++;
      if (auto targetAttr = dyn_cast_if_present<IREE::HAL::DeviceTargetAttr>(
              globalOp.getInitialValueAttr())) {
        info.id = targetAttr.getDeviceID().str();
      }
      devices[globalOp.getSymName()] = info;
    }

    auto lookupDevice =
        [&](IREE::Stream::AffinityAttr affinity) -> const DeviceInfo * {
      auto deviceAffinity =
          dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(affinity);
      if (!deviceAffinity) return nullptr;
      auto it = devices.find(deviceAffinity.getDevice().getLeafReference());
      return it == devices.end() ? nullptr : &it->second;
    };

    // 1. Functions that import their own arguments.
    for (auto funcOp : moduleOp.getOps<IREE::Util::FuncOp>()) {
      if (!isEntryPoint(funcOp)) continue;
      Block &entry = funcOp.getFunctionBody().front();
      SmallVector<llvm::SetVector<IREE::Stream::AffinityAttr>> importers(
          entry.getNumArguments());
      bool anyImport = false;
      funcOp.walk([&](IREE::Stream::TensorImportOp importOp) {
        auto arg = dyn_cast<BlockArgument>(importOp.getSource());
        if (!arg || arg.getOwner() != &entry) return;
        if (auto affinity = importOp.getAffinityAttr()) {
          importers[arg.getArgNumber()].insert(affinity);
          anyImport = true;
        }
      });
      if (!anyImport) continue;

      SmallVector<std::string> ids, ordinals;
      for (auto [argIndex, importerSet] : llvm::enumerate(importers)) {
        if (importerSet.empty()) {
          ids.push_back("");
          ordinals.push_back("");
          continue;
        }
        // Qualifiers in compile (device ordinal) order.
        SmallVector<const DeviceInfo *> qualified;
        bool unknownDevice = false;
        for (auto x : importerSet) {
          const DeviceInfo *info = lookupDevice(x);
          if (!info || info->id.empty()) {
            unknownDevice = true;
            break;
          }
          if (llvm::all_of(importerSet, [&](auto y) {
                return topology.hasTransparentAccess(y, x);
              })) {
            qualified.push_back(info);
          }
        }
        if (unknownDevice || qualified.empty()) {
          funcOp.emitWarning()
              << "input " << argIndex
              << (unknownDevice
                      ? " is imported on a device without a plain target ID"
                      : " is imported on devices that cannot all read any "
                        "one of them")
              << "; recorded as '?' in " << kDevicesKey
              << " (tooling requires --input_device for it)";
          ids.push_back("?");
          ordinals.push_back("?");
          continue;
        }
        llvm::sort(qualified, [](const DeviceInfo *a, const DeviceInfo *b) {
          return a->ordinal < b->ordinal;
        });
        ids.push_back(llvm::join(
            llvm::map_range(qualified, [](auto *i) { return i->id; }), "|"));
        ordinals.push_back(llvm::join(
            llvm::map_range(qualified,
                            [](auto *i) { return std::to_string(i->ordinal); }),
            "|"));
      }
      setReflectionEntries(funcOp, ids, ordinals);
    }

    // 2. Sync wrappers that call an annotated function: remap the callee's
    // entries to the caller's own argument numbers.
    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<IREE::Util::FuncOp>()) {
      if (!isEntryPoint(funcOp)) continue;
      if (auto reflection =
              funcOp->getAttrOfType<DictionaryAttr>(kReflectionAttr);
          reflection && reflection.get(kDevicesKey)) {
        continue;
      }
      Block &entry = funcOp.getFunctionBody().front();
      SmallVector<std::string> ids(entry.getNumArguments()),
          ordinals(entry.getNumArguments());
      bool found = false;
      funcOp.walk([&](IREE::Util::CallOp callOp) {
        auto calleeOp =
            symbolTable.lookup<IREE::Util::FuncOp>(callOp.getCallee());
        if (!calleeOp) return;
        auto reflection =
            calleeOp->getAttrOfType<DictionaryAttr>(kReflectionAttr);
        auto devicesAttr =
            reflection ? reflection.getAs<StringAttr>(kDevicesKey) : nullptr;
        auto ordinalsAttr =
            reflection ? reflection.getAs<StringAttr>(kOrdinalsKey) : nullptr;
        if (!devicesAttr || !ordinalsAttr) return;
        SmallVector<std::string> calleeIds = splitEntries(devicesAttr);
        SmallVector<std::string> calleeOrdinals = splitEntries(ordinalsAttr);
        for (auto [j, operand] : llvm::enumerate(callOp.getOperands())) {
          auto arg = dyn_cast<BlockArgument>(operand);
          if (!arg || arg.getOwner() != &entry || j >= calleeIds.size())
            continue;
          ids[arg.getArgNumber()] = calleeIds[j];
          ordinals[arg.getArgNumber()] = calleeOrdinals[j];
          found = true;
        }
      });
      if (found) setReflectionEntries(funcOp, ids, ordinals);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEAnnotateInputDevicesPass() {
  return std::make_unique<AMDAIEAnnotateInputDevicesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
