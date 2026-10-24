// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Heterogeneous accelerator + host device placement.
//
// This pass offloads contraction/convolution ops to accelerator devices
// (amd-aie, vulkan-spirv) and runs everything else (layout transposes, f32->bf16
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
// Requires at least one host device (accelerators are optional).
// Contraction/conv dispatches go round-robin across the accelerators in
// declaration order; everything else goes to the first host (see policy TODO).

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDevicePlacementUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

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

//===----------------------------------------------------------------------===//
// placement-map (experiment, default empty = off)
//
// A list of `<selector>=<device>` rules; the first rule matching a dispatch
// pins it. Selectors: `dN`, `dN-dM`, `dA,dB,...` (the N of the executable name
// `<func>_dispatch_N`, assigned by OutlineDispatchRegions before this pass and
// so independent of placement; stable only for the same model and compile
// flags), `~substr` (executable or export name contains substr) and `*`.
// Devices: `npu` / `gpu` / `cpu` (first device global whose backend is amd-aie
// / vulkan-spirv / llvm-cpu; `deviceRole` cannot tell npu from gpu since both
// are accelerators) or `@<device global>`.
//===----------------------------------------------------------------------===//

struct PlacementRule {
  enum class Kind { Numbers, Substring, All };
  std::string text;  // as written, for diagnostics and the summary
  Kind kind = Kind::All;
  SmallVector<std::pair<int64_t, int64_t>> ranges;  // Numbers (inclusive)
  std::string substring;                            // Substring
  FlatSymbolRefAttr device;
  StringRef backend;
  unsigned matched = 0;
};

/// Short device label for the stderr summary: npu / gpu / cpu by backend.
static std::string deviceLabel(StringRef backend) {
  if (backend == "amd-aie") return "npu";
  if (backend == "vulkan-spirv") return "gpu";
  if (backend == "llvm-cpu") return "cpu";
  return backend.str();
}

/// Dispatch number N from a `..._dispatch_N...` name, or -1.
static int64_t dispatchNumber(StringRef name) {
  size_t pos = name.find("_dispatch_");
  if (pos == StringRef::npos) return -1;
  StringRef digits = name.drop_front(pos + strlen("_dispatch_"))
                         .take_while(llvm::isDigit);
  int64_t num;
  if (digits.empty() || digits.getAsInteger(10, num)) return -1;
  return num;
}

/// Parses `dN` (`d` optional on the second end of a range).
static bool parseDispatchRef(StringRef s, bool requireD, int64_t &num) {
  s = s.trim();
  if (s.consume_front("d")) requireD = false;
  if (requireD) return false;
  return !s.empty() && !s.getAsInteger(10, num) && num >= 0;
}

/// Parses `placementMap` (rules separated by ';', or `@<file>` with one rule
/// per line and `#` comments) and resolves each device against `devices`.
/// Emits an error on `moduleOp` and returns failure on any bad rule.
static LogicalResult parsePlacementMap(StringRef placementMap,
                                       ArrayRef<DeviceInfo> devices,
                                       ModuleOp moduleOp,
                                       SmallVectorImpl<PlacementRule> &rules) {
  StringRef spec = placementMap.trim();
  // Pass-pipeline syntax may keep the quotes around the value.
  if (spec.size() >= 2 && (spec.front() == '\'' || spec.front() == '"') &&
      spec.back() == spec.front())
    spec = spec.drop_front().drop_back().trim();
  std::unique_ptr<llvm::MemoryBuffer> file;
  SmallVector<std::pair<StringRef, std::string>> entries;  // rule, where
  if (spec.consume_front("@")) {
    auto fileOrErr = llvm::MemoryBuffer::getFile(spec);
    if (!fileOrErr) {
      return moduleOp.emitError("placement-map: cannot read file '")
             << spec << "': " << fileOrErr.getError().message();
    }
    file = std::move(*fileOrErr);
    SmallVector<StringRef> lines;
    file->getBuffer().split(lines, '\n');
    for (auto [i, line] : llvm::enumerate(lines)) {
      StringRef rule = line.split('#').first.trim();
      if (!rule.empty())
        entries.push_back(
            {rule, (spec + ":" + std::to_string(i + 1)).str()});
    }
  } else {
    SmallVector<StringRef> parts;
    spec.split(parts, ';');
    for (auto [i, part] : llvm::enumerate(parts)) {
      StringRef rule = part.trim();
      if (!rule.empty())
        entries.push_back({rule, "rule " + std::to_string(i + 1)});
    }
  }
  if (entries.empty())
    return moduleOp.emitError("placement-map: no rules in '")
           << placementMap << "'";

  for (auto &entry : entries) {
    StringRef ruleText = entry.first;
    const std::string &where = entry.second;
    auto fail = [&]() {
      return moduleOp.emitError("placement-map ")
             << where << " '" << ruleText << "': ";
    };
    PlacementRule rule;
    rule.text = ruleText.str();
    auto [selector, deviceName] = ruleText.split('=');
    selector = selector.trim();
    deviceName = deviceName.trim();
    if (!ruleText.contains('=') || selector.empty() || deviceName.empty())
      return fail() << "expected '<selector>=<device>'";

    if (selector == "*") {
      rule.kind = PlacementRule::Kind::All;
    } else if (selector.consume_front("~")) {
      if (selector.empty()) return fail() << "empty '~' substring";
      rule.kind = PlacementRule::Kind::Substring;
      rule.substring = selector.str();
    } else {
      rule.kind = PlacementRule::Kind::Numbers;
      SmallVector<StringRef> items;
      selector.split(items, ',');
      for (StringRef item : items) {
        auto [lo, hi] = item.split('-');
        int64_t loNum, hiNum;
        if (!parseDispatchRef(lo, /*requireD=*/true, loNum))
          return fail() << "bad dispatch selector '" << item.trim()
                        << "' (expected dN, dN-dM, dA,dB, ~substr or *)";
        hiNum = loNum;
        if (item.contains('-') &&
            !parseDispatchRef(hi, /*requireD=*/false, hiNum))
          return fail() << "bad dispatch range '" << item.trim() << "'";
        if (hiNum < loNum)
          return fail() << "reversed dispatch range '" << item.trim() << "'";
        rule.ranges.push_back({loNum, hiNum});
      }
    }

    const DeviceInfo *target = nullptr;
    if (deviceName.consume_front("@")) {
      for (const DeviceInfo &d : devices)
        if (d.ref.getValue() == deviceName) target = &d;
      if (!target)
        return fail() << "device @" << deviceName
                      << " is not a placement device of this module";
    } else {
      StringRef backend = llvm::StringSwitch<StringRef>(deviceName)
                              .Case("npu", "amd-aie")
                              .Case("gpu", "vulkan-spirv")
                              .Case("cpu", "llvm-cpu")
                              .Default("");
      if (backend.empty())
        return fail() << "unknown device '" << deviceName
                      << "' (expected npu, gpu, cpu or @<device global>)";
      for (const DeviceInfo &d : devices)
        if (d.backend == backend) {
          target = &d;
          break;
        }
      if (!target)
        return fail() << "no " << backend << " device ('" << deviceName
                      << "') is declared in this module";
    }
    rule.device = target->ref;
    rule.backend = target->backend;
    rules.push_back(std::move(rule));
  }
  return success();
}

/// Returns true if `rule` matches a dispatch with number `num` (-1 if none)
/// whose entry point is `exeName::exportName`.
static bool ruleMatches(const PlacementRule &rule, int64_t num,
                        StringRef exeName, StringRef exportName) {
  switch (rule.kind) {
    case PlacementRule::Kind::All:
      return true;
    case PlacementRule::Kind::Substring:
      return exeName.contains(rule.substring) ||
             exportName.contains(rule.substring);
    case PlacementRule::Kind::Numbers:
      if (num < 0) return false;
      return llvm::any_of(rule.ranges, [&](auto range) {
        return range.first <= num && num <= range.second;
      });
  }
  return false;
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

  // Placement policy: round-robin over the accelerators, first host for the
  // rest. A performance/config-aware policy would replace this selection.
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
  // placement-map (experiment, default empty): only consulted inside the
  // `if (!placementMap.empty())` blocks below, so the default path is
  // unchanged. Rules pin first; unmatched dispatches fall through.
  SmallVector<PlacementRule> mapRules;
  struct MapEntry {
    IREE::Flow::DispatchOp dispatchOp;
    int64_t num;
    StringRef exportName;
    int ruleIdx;  // -1: no rule matched
  };
  SmallVector<MapEntry> mapEntries;
  if (!placementMap.empty()) {
    SmallVector<DeviceInfo> devices(accelerators);
    devices.append(hosts);
    if (failed(parsePlacementMap(placementMap, devices, moduleOp, mapRules)))
      return signalPassFailure();
  }
  moduleOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
    if (!placementMap.empty()) {
      SymbolRefAttr entryPoint = *dispatchOp.getEntryPointRefs().begin();
      StringRef exeName = entryPoint.getRootReference().getValue();
      StringRef exportName = entryPoint.getLeafReference().getValue();
      int64_t num = dispatchNumber(exeName);
      if (num < 0) num = dispatchNumber(exportName);
      mapEntries.push_back({dispatchOp, num, exportName, -1});
      for (auto [idx, rule] : llvm::enumerate(mapRules)) {
        if (!ruleMatches(rule, num, exeName, exportName)) continue;
        ++rule.matched;
        mapEntries.back().ruleIdx = idx;
        auto it = executableIsAccel.find(exeName);
        if (rule.backend == "amd-aie" &&
            (it == executableIsAccel.end() || !it->second)) {
          dispatchOp.emitWarning("placement-map places a dispatch without a "
                                 "contraction/convolution on amd-aie device ")
              << rule.device << " (rule '" << rule.text
              << "'); amd-aie may fail to lower it";
        }
        // Mapped dispatches do not advance the round-robin.
        usedRefs.insert(rule.device);
        dispatchOp->setAttr("stream.affinity", affinityFor(rule.device));
        return;
      }
      // No rule matched: fall through to the default below.
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

  if (!placementMap.empty() && !mapEntries.empty()) {
    // A number rule that matched nothing is a typo or is shadowed by an
    // earlier rule. (Skipped without dispatches: the preprocessing run.)
    bool deadRule = false;
    for (auto [idx, rule] : llvm::enumerate(mapRules)) {
      if (rule.kind != PlacementRule::Kind::Numbers || rule.matched) continue;
      moduleOp.emitError("placement-map rule #")
          << (idx + 1) << " '" << rule.text
          << "' matches no dispatch (no such dispatch number, or every "
             "match is taken by an earlier rule)";
      deadRule = true;
    }
    if (deadRule) return signalPassFailure();

    // Summary and per-dispatch placement on stderr (only with the flag).
    DenseMap<Attribute, std::string> labels;
    for (const DeviceInfo &d : accelerators)
      labels[d.ref] = deviceLabel(d.backend);
    for (const DeviceInfo &d : hosts) labels[d.ref] = deviceLabel(d.backend);
    SmallVector<std::pair<std::string, unsigned>> perLabel = {
        {"npu", 0}, {"gpu", 0}, {"cpu", 0}};
    SmallVector<std::string> lines;
    unsigned unmatched = 0;
    for (const MapEntry &e : mapEntries) {
      auto affinity = dyn_cast_if_present<IREE::HAL::DeviceAffinityAttr>(
          e.dispatchOp->getAttr("stream.affinity"));
      std::string device = "?", label = "?";
      if (affinity) {
        device = "@" + affinity.getDevice().getRootReference().getValue().str();
        auto it = labels.find(affinity.getDevice());
        if (it != labels.end()) label = it->second;
      }
      auto count =
          llvm::find_if(perLabel, [&](auto &p) { return p.first == label; });
      if (count == perLabel.end())
        perLabel.push_back({label, 1});
      else
        ++count->second;
      if (e.ruleIdx < 0) ++unmatched;
      std::string rule =
          e.ruleIdx < 0 ? std::string("(no rule: default placement)")
                        : "rule#" + std::to_string(e.ruleIdx + 1) + " " +
                              mapRules[e.ruleIdx].text;
      std::string num =
          e.num < 0 ? std::string("d?") : "d" + std::to_string(e.num);
      lines.push_back(num + "\t" + e.exportName.str() + "\t" + rule + "\t" +
                      device + " (" + label + ")");
    }
    llvm::raw_ostream &os = llvm::errs();
    os << "[amdaie placement-map] rules=" << mapRules.size()
       << " dispatches=" << mapEntries.size();
    for (auto &[l, n] : perLabel) os << " " << l << "=" << n;
    os << " unmatched=" << unmatched << "\n";
    for (auto [idx, rule] : llvm::enumerate(mapRules))
      os << "[amdaie placement-map] rule#" << (idx + 1) << " " << rule.text
         << ": " << rule.matched << "\n";
    for (const std::string &line : lines)
      os << "[amdaie placement-map] " << line << "\n";
  }

  // Inject the device topology over the devices we place work on so cross-device
  // boundary buffers resolve (ResolveTopologyQueries). Respect a user-provided
  // topology if present. Link flags come from the capability table.
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
    // placement-map may also pin work on a host other than the first one.
    if (!placementMap.empty()) {
      for (const DeviceInfo &d : hosts)
        if (d.ref != hostTarget.ref && usedRefs.contains(d.ref))
          used.push_back(d);
    }
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
