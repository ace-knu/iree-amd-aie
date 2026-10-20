// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Heterogeneous accelerator + host device placement.
//
// This pass offloads the ops amd-aie can codegen (contraction/convolution) to
// an accelerator device and runs everything else (layout transposes, f32->bf16
// casts introduced by demotion, other elementwise leftovers) on a host device.
// Devices are declared via `--iree-hal-target-device` (util.global with a
// DeviceTargetAttr); a single device-capability table, in
// Transforms/Utils/AMDAIEDevicePlacementUtils.{h,cpp}, maps each backend to a
// role (Host/Accelerator) and each backend pair to its topology link capability
// -- the two seams here (classification and topology) both read it, as does
// AMDAIEPadContractionDispatches when it needs a host to place work on.
//
// Run at the end of the Flow phase (after dispatch outlining, before the Stream
// affinity solver), it pins each `flow.dispatch` via `stream.affinity`
// (contraction/conv -> accelerator, else -> host) and injects a `stream.topology`
// module attribute over the devices it places work on (unless one is already
// provided) so cross-device buffers can be resolved.
//
// Requires at least one host device (accelerators are optional). Choosing among
// multiple same-role devices currently uses the first one (see policy TODO).

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDevicePlacementUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#define DEBUG_TYPE "iree-amdaie-assign-device-affinities"

namespace mlir::iree_compiler::AMDAIE {

namespace {

//===----------------------------------------------------------------------===//
// Device-capability table
//
// The table itself lives in Utils/AMDAIEDevicePlacementUtils.{h,cpp} so the
// other passes that need to find a host or an accelerator resolve it the same
// way (AMDAIEPadContractionDispatches places its padding dispatches on the
// host). Maps a device backend to its role and a backend pair to its topology
// link capability; both the classification and the topology seams below read
// it.
//
// Adding an entry is necessary but not sufficient to support a second
// accelerator backend, for two reasons:
//
//  - Classification is a single shared predicate.
//    `executableIsContractionOrConv` below answers "does an accelerator want
//    this?", not "does *this* accelerator support it". Two accelerators with
//    different op coverage cannot be told apart by editing the table alone.
//  - Placement is unconditional. The walk in `runOnOperation` sets
//    `stream.affinity` on every dispatch without checking for an existing one.
//    `PluginManager` fans the pipeline hooks out to every initialized session,
//    so if a second accelerator plugin registers its own placement pass, the
//    two passes overwrite each other and whichever runs last wins. Skipping
//    dispatches that already carry an affinity is the minimum needed before
//    that configuration can work.
//===----------------------------------------------------------------------===//

/// A classified device global: its symbol reference and backend.
struct DeviceInfo {
  FlatSymbolRefAttr ref;
  StringRef backend;
};

/// Returns true if the linalg op's reduction body is a multiply-accumulate
/// (`out + lhs * rhs`), i.e. a real contraction/convolution rather than a
/// pooling op. `isaConvolutionOpInterface` matches on indexing-map structure
/// only and so also accepts pooling (max/min/sum over a window), which amd-aie
/// cannot codegen; this body check distinguishes the two. Uses upstream MLIR
/// (`linalg::detail::isContractionBody` + `arith`) so the classification stays
/// backend-neutral (reusable for other accelerator backends).
static bool hasMultiplyAccumulateBody(linalg::LinalgOp linalgOp) {
  Block *body = linalgOp.getBlock();
  if (!body) return false;
  return linalg::detail::isContractionBody(
      *body, [](Operation *mul, Operation *add) {
        return isa<arith::MulFOp, arith::MulIOp>(mul) &&
               isa<arith::AddFOp, arith::AddIOp>(add);
      });
}

/// Returns true if `exe` contains a contraction/convolution op (the ops
/// amd-aie can codegen), inspected structurally on the linalg IR. Pooling ops
/// structurally look like convolutions but have a non-multiply-accumulate body
/// (max/min); they are excluded via `hasMultiplyAccumulateBody` so they route
/// to the host instead. (Contractions already imply a multiply-accumulate body,
/// so that branch needs no extra check.)
static bool executableIsContractionOrConv(IREE::Flow::ExecutableOp exe) {
  ModuleOp innerModule = exe.getInnerModule();
  if (!innerModule) return false;
  bool found = false;
  innerModule.walk([&](linalg::LinalgOp linalgOp) {
    if (linalg::isaContractionOpInterface(linalgOp) ||
        (linalg::isaConvolutionOpInterface(linalgOp) &&
         hasMultiplyAccumulateBody(linalgOp))) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Device mark read from an executable for `respect-marks`: a
/// `stream.affinity = #hal.device.affinity<@dev>` attribute on a linalg op
/// inside the executable. The contraction/conv root's mark wins; without one,
/// the first mark in walk order is used. Marks that disagree are reported with
/// a warning (the chosen mark still applies). Returns failure, after emitting
/// an error, for a mark that is not a `#hal.device.affinity`; returns a null
/// attribute when the executable carries no mark.
static FailureOr<SymbolRefAttr> readExecutableMark(
    IREE::Flow::ExecutableOp exe) {
  ModuleOp innerModule = exe.getInnerModule();
  if (!innerModule) return SymbolRefAttr();
  SymbolRefAttr rootMark;
  SmallVector<SymbolRefAttr> marks;
  bool invalid = false;
  innerModule.walk([&](linalg::LinalgOp linalgOp) {
    Attribute attr = linalgOp->getAttr("stream.affinity");
    if (!attr) return WalkResult::advance();
    auto affinity = dyn_cast<IREE::HAL::DeviceAffinityAttr>(attr);
    if (!affinity) {
      linalgOp->emitError("device mark must be a #hal.device.affinity, got ")
          << attr;
      invalid = true;
      return WalkResult::interrupt();
    }
    bool isRoot = linalg::isaContractionOpInterface(linalgOp) ||
                  (linalg::isaConvolutionOpInterface(linalgOp) &&
                   hasMultiplyAccumulateBody(linalgOp));
    if (isRoot && !rootMark) rootMark = affinity.getDevice();
    marks.push_back(affinity.getDevice());
    return WalkResult::advance();
  });
  if (invalid) return failure();
  if (marks.empty()) return SymbolRefAttr();
  SymbolRefAttr chosen = rootMark ? rootMark : marks.front();
  if (!llvm::all_equal(marks)) {
    exe.emitWarning("dispatch has conflicting device marks; using ")
        << chosen << (rootMark ? " (contraction/conv root)" : " (first mark)");
  }
  return chosen;
}

class AMDAIEAssignDeviceAffinitiesPass
    : public impl::AMDAIEAssignDeviceAffinitiesBase<
          AMDAIEAssignDeviceAffinitiesPass> {
 public:
  AMDAIEAssignDeviceAffinitiesPass() = default;
  AMDAIEAssignDeviceAffinitiesPass(
      const AMDAIEAssignDeviceAffinitiesOptions &options)
      : AMDAIEAssignDeviceAffinitiesBase(options) {}
  void runOnOperation() override;
};

void AMDAIEAssignDeviceAffinitiesPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  // Classify device globals by role via the capability table.
  SmallVector<DeviceInfo> accelerators, hosts;
  for (auto global : moduleOp.getOps<IREE::Util::GlobalOpInterface>()) {
    StringRef backend = getDeviceBackend(global);
    DeviceInfo info{FlatSymbolRefAttr::get(ctx, global.getGlobalName()),
                    backend};
    switch (deviceRole(backend)) {
      case DeviceRole::Accelerator:
        accelerators.push_back(info);
        break;
      case DeviceRole::Host:
        hosts.push_back(info);
        break;
      case DeviceRole::Unknown:
        break;  // ignored: not placed, not in topology -> DCE
    }
  }
  // A host device is required; accelerators are optional.
  if (hosts.empty()) return;
  if (accelerators.empty()) return;

  // Placement policy (currently trivial "front"). A performance/config-aware
  // policy would replace this selection (and then also need the two-phase
  // topology keep-alive noted below).
  // TODO: choose among multiple accelerators/hosts by device performance/config.
  DeviceInfo hostTarget = hosts.front();

  auto affinityFor = [&](FlatSymbolRefAttr ref) {
    return IREE::HAL::DeviceAffinityAttr::get(ctx, ref, /*queue_mask=*/-1ll);
  };

  // Pin each dispatch: contraction/conv -> accelerator, everything else -> host.
  // Both are set explicitly so the affinity solver does not propagate the host
  // affinity onto accelerator work (it minimizes transfers otherwise).
  DenseMap<StringRef, bool> executableIsAccel;
  for (auto exe : moduleOp.getOps<IREE::Flow::ExecutableOp>())
    executableIsAccel[exe.getSymName()] = executableIsContractionOrConv(exe);
  unsigned accelIdx = 0;
  DenseSet<Attribute> usedRefs;
  // respect-marks (debug, default off): only consulted inside the
  // `if (respectMarks)` block below, so the default path is unchanged.
  DenseMap<StringRef, IREE::Flow::ExecutableOp> executablesByName;
  if (respectMarks) {
    for (auto exe : moduleOp.getOps<IREE::Flow::ExecutableOp>())
      executablesByName[exe.getSymName()] = exe;
  }
  auto isPlacementDevice = [&](SymbolRefAttr ref) {
    for (const DeviceInfo &d : accelerators)
      if (Attribute(d.ref) == Attribute(ref)) return true;
    for (const DeviceInfo &d : hosts)
      if (Attribute(d.ref) == Attribute(ref)) return true;
    return false;
  };
  bool markFailed = false;
  moduleOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
    if (respectMarks) {
      // (1) An affinity already on the dispatch is kept.
      if (Attribute existing = dispatchOp->getAttr("stream.affinity")) {
        if (auto affinity = dyn_cast<IREE::HAL::DeviceAffinityAttr>(existing))
          usedRefs.insert(affinity.getDevice());
        return;
      }
      // (2) Otherwise a mark inside the (first) entry point's executable pins
      // the dispatch. Marked dispatches do not advance the round-robin.
      SymbolRefAttr mark;
      for (SymbolRefAttr entryPoint : dispatchOp.getEntryPointRefs()) {
        auto it = executablesByName.find(
            entryPoint.getRootReference().getValue());
        if (it == executablesByName.end()) continue;
        FailureOr<SymbolRefAttr> read = readExecutableMark(it->second);
        if (failed(read)) {
          markFailed = true;
          return;
        }
        mark = *read;
        break;
      }
      if (mark) {
        if (!isPlacementDevice(mark)) {
          dispatchOp.emitError("device mark ")
              << mark << " does not name a placement device";
          markFailed = true;
          return;
        }
        usedRefs.insert(mark);
        dispatchOp->setAttr("stream.affinity",
                            IREE::HAL::DeviceAffinityAttr::get(
                                ctx, mark, /*queue_mask=*/-1ll));
        return;
      }
      // (3) No mark: fall through to the default placement below.
    }
    bool onAccel = false;
    for (SymbolRefAttr entryPoint : dispatchOp.getEntryPointRefs()) {
      auto it = executableIsAccel.find(entryPoint.getRootReference().getValue());
      if (it != executableIsAccel.end() && it->second) onAccel = true;
    }
    DeviceInfo target = hostTarget;
    if (onAccel && !accelerators.empty()) {
      target = accelerators[accelIdx % accelerators.size()];
      ++accelIdx;
    }
    usedRefs.insert(target.ref);
    dispatchOp->setAttr("stream.affinity", affinityFor(target.ref));
  });
  if (markFailed) return signalPassFailure();

  // Inject the device topology over the devices we place work on so cross-device
  // boundary buffers resolve (ResolveTopologyQueries). Respect a user-provided
  // topology if present. Link flags come from the capability table.
  // NOTE: the "used" set is derivable here (before dispatches) only because the
  // front policy is dispatch-independent; a per-dispatch policy would instead
  // keep all candidates alive at preprocessing and narrow at the Flow call.
  if (!moduleOp->hasAttr("stream.topology")) {
    SmallVector<DeviceInfo> used;
    // This pass runs twice. The first run happens before any flow.dispatch
    // exists, so the walk sees nothing and usedRefs stays empty; keep every
    // candidate alive there or FoldGlobals removes the accelerator globals
    // before the second run can place work on them. Once dispatches exist,
    // narrow to what was actually used.
    bool sawDispatch = !usedRefs.empty();
    for (const DeviceInfo &d : accelerators)
      if (!sawDispatch || usedRefs.contains(d.ref)) used.push_back(d);
    used.push_back(hostTarget);
    if (used.size() >= 2) {
      SmallVector<IREE::HAL::DeviceLinkAttr> links;
      for (const DeviceInfo &from : used)
        for (const DeviceInfo &to : used) {
          if (from.ref == to.ref) continue;
          LinkCapability cap = linkFlags(from.backend, to.backend);
          links.push_back(IREE::HAL::DeviceLinkAttr::get(
              ctx, from.ref, to.ref, cap.unifiedMemory, cap.transparentAccess,
              /*extra_properties=*/DictionaryAttr()));
        }
      moduleOp->setAttr("stream.topology",
                        IREE::HAL::DeviceTopologyAttr::get(ctx, links));
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAssignDeviceAffinitiesPass(
    AMDAIEAssignDeviceAffinitiesOptions options) {
  return std::make_unique<AMDAIEAssignDeviceAffinitiesPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
