// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the insertion of `amdaie.core` operations in the
// innermost `scf.forall` operations with thread mapping. Each core has a tile
// location which is computed from the for loop's induction variables. This pass
// will probably be updated in the future to work with loops with block mapping
// as well.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEOpUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-amdaie-insert-cores"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility which returns 'true' is the operation needs to be inserted with an
/// `amdaie.core` op.
/// Some ops are surrounded by scf.for loop nests. Place the entire
/// loop nest inside the amdaie.core op here. Currently look for a
/// subset of ops which we know should be in the core.
/// TODO(newling) improve this design.
static bool isCoreComputeOp(Operation *op) {
  return isa<linalg::LinalgOp, linalg::SoftmaxOp, vector::ContractionOp,
             memref::ExtractStridedMetadataOp, func::CallOp,
             vector::TransferReadOp, vector::TransferWriteOp>(op);
}

/// Utility to map the parallel mapping attributes to the corresponding
/// induction variables.
void getAttributeMapping(SmallVector<scf::ForallOp> forallOps,
                         DenseMap<Attribute, Value> &attrMapping) {
  for (auto forallOp : forallOps) {
    if (!forallOp.getMapping().has_value()) continue;
    SmallVector<Attribute> mapping =
        llvm::to_vector(forallOp.getMapping()->getValue());
    auto ivs = forallOp.getInductionVars();
    for (auto &&[attr, iv] : llvm::zip(mapping, ivs)) attrMapping[attr] = iv;
  }
}

/// Insert core ops inside innermost forall ops around computational ops and
/// add synchronization ops along the way to synchronize with surrounding
/// dma ops.
static LogicalResult insertCoreOps(mlir::ModuleOp moduleOp, int64_t stackSize) {
  // Get the device model from the target attribute.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(moduleOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    return moduleOp->emitOpError()
           << "has no AMDAIEDevice in the target attribute configuration. This "
              "device-specific information is required to insert cores.";
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  IRRewriter rewriter(moduleOp.getContext());
  WalkResult res = moduleOp->walk([&](scf::ForallOp forallOp) {
    // Currently, innermost `scf.forall` operations are expected to have thread
    // mapping and are therefore selected for insertion of cores. Advance if no
    // thread mapping.
    if (!forallOp.getMapping().has_value()) return WalkResult::advance();
    SmallVector<Attribute> mapping =
        llvm::to_vector(forallOp.getMapping()->getValue());
    if (!isa<mlir::gpu::GPUThreadMappingAttr>(*mapping.begin()))
      return WalkResult::advance();

    if (!forallOp.isNormalized()) {
      forallOp.emitOpError()
          << "scf.forall operations must be normalized before core "
             "operation insertion";
      return WalkResult::interrupt();
    }
    auto parentOps = getInclusiveParentsOfType<scf::ForallOp>(forallOp);
    DenseMap<Attribute, Value> attrMapping;
    getAttributeMapping(parentOps, attrMapping);
    mlir::gpu::GPUThreadMappingAttr threadXAttr =
        mlir::gpu::GPUThreadMappingAttr::get(forallOp->getContext(),
                                             mlir::gpu::MappingId::DimX);
    mlir::gpu::GPUThreadMappingAttr threadYAttr =
        mlir::gpu::GPUThreadMappingAttr::get(forallOp->getContext(),
                                             mlir::gpu::MappingId::DimY);
    Value threadY = attrMapping.lookup(threadYAttr);
    if (!threadY) {
      forallOp.emitOpError()
          << "missing threadY attribute mapping: " << threadYAttr;
      return WalkResult::interrupt();
    }
    Value threadX = attrMapping.lookup(threadXAttr);
    if (!threadX) {
      rewriter.setInsertionPoint(forallOp.getBody()->getTerminator());
      threadX =
          rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
    }

    // Find input and output DMAs that need to be added to the core.
    SmallVector<Value> inputDmas;
    SmallVector<Value> outputDmas;
    WalkResult dmaRes = forallOp->walk([&](AMDAIE::DmaCpyNdOp dmaOp) {
      std::optional<uint8_t> sourceMemSpace =
          dmaOp.getSourceMemorySpaceAsUInt();
      std::optional<uint8_t> targetMemSpace =
          dmaOp.getTargetMemorySpaceAsUInt();
      if (!sourceMemSpace || !targetMemSpace) {
        dmaOp.emitOpError() << "expected a source and target memory space";
        return WalkResult::interrupt();
      }
      if (sourceMemSpace.value() == 2 && targetMemSpace.value() == 2) {
        dmaOp->emitOpError()
            << "dma op with both source and target on L1 is not supported";
        return WalkResult::interrupt();
      } else if (sourceMemSpace == 2) {
        outputDmas.push_back(dmaOp);
      } else if (targetMemSpace == 2) {
        inputDmas.push_back(dmaOp);
      }
      return WalkResult::advance();
    });
    if (dmaRes.wasInterrupted()) return WalkResult::interrupt();

    // Create CoreOp at the end of the innermost forall
    rewriter.setInsertionPoint(forallOp.getBody()->getTerminator());
    uint32_t rowOffset = deviceModel.getCoreTileRowStart();
    // `threadY` (the core-row index) is not bounded to the number of
    // physically available core rows: the tile-size selection heuristic that
    // determines the forall's trip counts does not currently guarantee this
    // (e.g. it doesn't tile the M dimension for `linalg.batch_matmul`, unlike
    // `linalg.matmul`, so a wide N can end up needing more row-iterations than
    // there are physical rows). Rather than passing an out-of-range row
    // straight through to `CoreOp::build`'s unconditional `rowOffset + row`
    // (which crashes later in `AMDAIEDeviceModel::getTileType` once the
    // resulting tile location doesn't exist), spill the excess into
    // additional columns: `col = threadX + threadY / numRows`,
    // `row = threadY % numRows`. This keeps every generated tile within the
    // device's actual (numRows x numCols) core grid.
    Value col = threadX;
    Value row = threadY;
    std::optional<int64_t> maybeNumRows = getConfigNumRows(targetAttr);
    if (maybeNumRows && *maybeNumRows > 0) {
      auto numRowsVal = rewriter.create<arith::ConstantIndexOp>(
          rewriter.getUnknownLoc(), *maybeNumRows);
      Value colOffset = rewriter.create<arith::DivUIOp>(
          rewriter.getUnknownLoc(), threadY, numRowsVal);
      row = rewriter.create<arith::RemUIOp>(rewriter.getUnknownLoc(), threadY,
                                            numRowsVal);
      col = rewriter.create<arith::AddIOp>(rewriter.getUnknownLoc(), threadX,
                                           colOffset);
    }
    auto coreOp = rewriter.create<AMDAIE::CoreOp>(
        rewriter.getUnknownLoc(), col, row, rowOffset, inputDmas, outputDmas,
        stackSize);
    Region &region = coreOp.getRegion();
    Block *newBlock = rewriter.createBlock(&region);
    rewriter.setInsertionPointToStart(newBlock);
    auto endOp = rewriter.create<AMDAIE::EndOp>(rewriter.getUnknownLoc());

    // Walk all operations in the workgroup and fill in the CoreOp with
    // computational ops.
    SmallVector<StringRef> ukernelObjectFiles;
    WalkResult forallRes = forallOp->walk([&](Operation *op) {
      // Skip operations already inside core ops
      if (op->getParentOfType<AMDAIE::CoreOp>()) return WalkResult::advance();

      if (op == forallOp) return WalkResult::advance();

      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        // Fetch name of the ukernel function to look up its declaration in the
        // Symbol table.
        StringRef fnName = callOp.getCallee();
        auto fnDecl = dyn_cast_if_present<func::FuncOp>(
            SymbolTable::lookupSymbolIn(moduleOp, fnName));
        assert(fnDecl && "expected function declaration");
        assert(fnDecl->hasAttr("link_with") &&
               "expected 'link_with' construct for the function declaration");
        // From the declaration of the function, we extract the value of
        // attribute "link_with" and attach it to amdaie.core op.
        ukernelObjectFiles.push_back(
            fnDecl->getAttrOfType<StringAttr>("link_with").getValue());
      }

      if (isCoreComputeOp(op)) {
        // Most distant ancestor of 'op' that's a strict descendant of
        // 'forallOp'.
        Operation *ancestor = op;
        while (ancestor->getParentOp() != forallOp) {
          ancestor = ancestor->getParentOp();
        }
        rewriter.setInsertionPoint(endOp);
        rewriter.moveOpBefore(ancestor, endOp);
      }

      return WalkResult::advance();
    });

    if (!ukernelObjectFiles.empty()) {
      // Concatenate all the object file names into a single string, separated
      // by commas.
      coreOp.setLinkWith(
          rewriter.getStringAttr(llvm::join(ukernelObjectFiles, ",")));
    };

    if (forallRes.wasInterrupted()) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEInsertCoresPass
    : public impl::AMDAIEInsertCoresBase<AMDAIEInsertCoresPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AMDAIEDialect>();
  }

  AMDAIEInsertCoresPass() = default;
  AMDAIEInsertCoresPass(const AMDAIEInsertCoresPass &pass){};
  AMDAIEInsertCoresPass(const AMDAIEInsertCoresOptions &options)
      : AMDAIEInsertCoresBase(options) {}

  void runOnOperation() override;
};

void AMDAIEInsertCoresPass::runOnOperation() {
  if (failed(insertCoreOps(getOperation(), stackSize))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertCoresPass(
    AMDAIEInsertCoresOptions options) {
  return std::make_unique<AMDAIEInsertCoresPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
