// Resolves #hal.device.optimal<[...]> allocation affinities to a single owner
// device using the module's stream.topology table, so the runtime
// hal.allocator.select does not have to pick (and no device name is needed).
//
// Rule: candidate X becomes the owner iff every candidate Y has
// (Y -> X) transparent_access, i.e. all users can read X's allocation without a
// copy. Exactly one qualifier -> resolved. Zero or several (tie, e.g. CPU<->GPU
// on unified memory) -> left as optimal<> for the runtime to decide.

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEResolveBufferOwnerPass
    : public impl::AMDAIEResolveBufferOwnerBase<AMDAIEResolveBufferOwnerPass> {
 public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto topology = moduleOp->getAttrOfType<IREE::HAL::DeviceTopologyAttr>(
        "stream.topology");
    if (!topology) return;

    moduleOp.walk([&](IREE::Stream::AffinityOpInterface op) {
      auto optimal = dyn_cast_if_present<IREE::HAL::DeviceOptimalAttr>(
          op.getAffinityAttr());
      if (!optimal) return;
      IREE::Stream::AffinityAttr owner;
      int qualified = 0;
      for (auto x : optimal.getAffinities()) {
        if (llvm::all_of(optimal.getAffinities(), [&](auto y) {
              return topology.hasTransparentAccess(y, x);
            })) {
          owner = x;
          ++qualified;
        }
      }
      if (qualified == 1) op.setAffinityAttr(owner);
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEResolveBufferOwnerPass() {
  return std::make_unique<AMDAIEResolveBufferOwnerPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
