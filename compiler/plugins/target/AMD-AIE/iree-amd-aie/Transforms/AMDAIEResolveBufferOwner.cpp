// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Resolves #hal.device.optimal<[...]> allocation affinities to a single owner
// device using the module's stream.topology table, so the runtime
// hal.allocator.select does not have to pick (and no device name is needed).
//
// Rule: candidate X becomes the owner iff every candidate Y has
// (Y -> X) transparent_access, i.e. all users can read X's allocation without a
// copy. Exactly one qualifier -> resolved. Zero -> left as optimal<> for the
// runtime to decide.
//
// Tie (several qualifiers, e.g. CPU<->GPU when both links are transparent):
// the resource's per-invocation use decides. If, outside initializers, the
// resource is used by stream.cmd.execute ops of exactly one qualifier D (other
// devices only touch it in initializers, which run once), D owns it: every
// invocation then uses the buffer natively instead of reaching it through a
// link on each use. Otherwise (several devices use it per invocation, nobody
// does, or a use is not understood) it stays optimal<> as before. Only the
// topology qualification and the set of using devices are consulted; no device
// kind is named.

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Where a resource is used, following it through stream.timepoint.await
// results and util.global store/load pairs.
struct ResourceUse {
  // Affinities of the stream.cmd.execute ops that capture the resource outside
  // initializers (per-invocation uses).
  llvm::SetVector<Attribute> perInvocationDevices;
  // stream.resource.dealloca ops of the resource; they follow the owner.
  SmallVector<IREE::Stream::ResourceDeallocaOp> deallocas;
  // False when a use is not one of the forms above (e.g. the resource is
  // returned, passed as a block argument or exported); the caller then keeps
  // optimal<>.
  bool understood = true;
};

ResourceUse collectUses(Value root, ModuleOp moduleOp) {
  ResourceUse use;
  SmallVector<Value> worklist = {root};
  llvm::DenseSet<Value> seen;
  while (!worklist.empty() && use.understood) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second) continue;
    for (OpOperand &operand : value.getUses()) {
      Operation *user = operand.getOwner();
      if (auto exec = dyn_cast<IREE::Stream::CmdExecuteOp>(user)) {
        if (!llvm::is_contained(exec.getResourceOperands(), value)) {
          use.understood = false;  // used as something else (e.g. a size)
          break;
        }
        if (!exec->getParentOfType<IREE::Util::InitializerOp>()) {
          Attribute affinity = exec.getAffinityAttr();
          if (!affinity) {
            use.understood = false;  // device of the use unknown
            break;
          }
          use.perInvocationDevices.insert(affinity);
        }
      } else if (auto dealloca =
                     dyn_cast<IREE::Stream::ResourceDeallocaOp>(user)) {
        use.deallocas.push_back(dealloca);
      } else if (auto await = dyn_cast<IREE::Stream::TimepointAwaitOp>(user)) {
        for (auto [in, out] :
             llvm::zip_equal(await.getResourceOperands(), await.getResults())) {
          if (in == value) worklist.push_back(out);
        }
      } else if (auto store =
                     dyn_cast<IREE::Util::GlobalStoreOpInterface>(user)) {
        // The global must hold only this resource: a single store.
        StringRef name = store.getGlobalAttr().getValue();
        int stores = 0;
        SmallVector<Value> loads;
        moduleOp.walk([&](Operation *op) {
          if (auto s = dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
            if (s.getGlobalAttr().getValue() == name) ++stores;
          } else if (auto l = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
            if (l.getGlobalAttr().getValue() == name)
              loads.push_back(l.getLoadedGlobalValue());
          }
        });
        if (stores != 1) {
          use.understood = false;
          break;
        }
        worklist.append(loads.begin(), loads.end());
      } else {
        use.understood = false;  // returned, exported, block argument, ...
        break;
      }
    }
  }
  return use;
}

class AMDAIEResolveBufferOwnerPass
    : public impl::AMDAIEResolveBufferOwnerBase<AMDAIEResolveBufferOwnerPass> {
 public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto topology = moduleOp->getAttrOfType<IREE::HAL::DeviceTopologyAttr>(
        "stream.topology");
    if (!topology) return;

    // Allocations whose candidates tie; resolved below by their use.
    SmallVector<std::pair<IREE::Stream::ResourceAllocaOp,
                          SmallVector<IREE::Stream::AffinityAttr>>>
        ties;
    moduleOp.walk([&](IREE::Stream::AffinityOpInterface op) {
      auto optimal = dyn_cast_if_present<IREE::HAL::DeviceOptimalAttr>(
          op.getAffinityAttr());
      if (!optimal) return;
      SmallVector<IREE::Stream::AffinityAttr> qualified;
      for (auto x : optimal.getAffinities()) {
        if (llvm::all_of(optimal.getAffinities(), [&](auto y) {
              return topology.hasTransparentAccess(y, x);
            })) {
          qualified.push_back(x);
        }
      }
      if (qualified.size() == 1) {
        op.setAffinityAttr(qualified.front());
      } else if (qualified.size() > 1) {
        if (auto alloca = dyn_cast<IREE::Stream::ResourceAllocaOp>(
                op.getOperation()))
          ties.push_back({alloca, std::move(qualified)});
      }
    });

    for (auto &[alloca, qualified] : ties) {
      ResourceUse use = collectUses(alloca.getResult(), moduleOp);
      if (!use.understood || use.perInvocationDevices.size() != 1) continue;
      Attribute device = use.perInvocationDevices.front();
      auto owner = llvm::find_if(
          qualified, [&](IREE::Stream::AffinityAttr x) { return x == device; });
      if (owner == qualified.end()) continue;
      // The deallocas of the resource must carry the same tied optimal<>; any
      // other affinity means they were already decided elsewhere.
      if (!llvm::all_of(use.deallocas, [&](auto dealloca) {
            return dealloca.getAffinityAttr() == alloca.getAffinityAttr();
          }))
        continue;
      alloca.setAffinityAttr(*owner);
      for (auto dealloca : use.deallocas) dealloca.setAffinityAttr(*owner);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEResolveBufferOwnerPass() {
  return std::make_unique<AMDAIEResolveBufferOwnerPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
