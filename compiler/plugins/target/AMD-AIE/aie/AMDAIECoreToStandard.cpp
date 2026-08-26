// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "amdaie-standard-lowering"

using namespace mlir;
using namespace xilinx::AIE;

static LogicalResult lockToStd(IRRewriter &rewriter, Operation *parentOp,
                               const std::string &targetArch) {
  OpBuilder::InsertionGuard guard(rewriter);
  MLIRContext *ctx = rewriter.getContext();

  StringAttr privateSym = StringAttr::get(ctx, "private");
  auto buildDecl = [&](const std::string &funcName) {
    rewriter.create<func::FuncOp>(
        rewriter.getUnknownLoc(), funcName,
        FunctionType::get(ctx, {rewriter.getI32Type(), rewriter.getI32Type()},
                          {}),
        privateSym, ArrayAttr{}, ArrayAttr{});
  };

  std::string acquireFunction = "llvm." + targetArch + ".acquire";
  std::string releaseFunction = "llvm." + targetArch + ".release";

  buildDecl(acquireFunction);
  buildDecl(releaseFunction);

  // EXPERIMENTAL: tag each acquire/release func.call with a stable integer
  // identifying which underlying lock it targets (distinct from the
  // acquire/release VALUE, which is usually just 1 for every call and is
  // useless for telling calls to different locks apart). Read back in
  // `coreToStd` to test the "only the first release of each lock needs a
  // delay" hypothesis precisely. Not a real fix -- exploratory only.
  DenseMap<Operation *, int64_t> lockIds;

  WalkResult res = parentOp->walk([&](UseLockOp useLock) {
    if (!isa<DeviceOp>(useLock->getParentOp())) {
      std::string funcName;
      switch (useLock.getAction()) {
        case LockAction::Acquire:
        case LockAction::AcquireGreaterEqual:
          funcName = acquireFunction;
          break;
        case LockAction::Release:
          funcName = releaseFunction;
          break;
        default:
          useLock.emitOpError() << "has an unsupported lock action";
          return WalkResult::interrupt();
      }

      // TODO(max): this can be simplified with
      // SymbolTable::lookupNearestSymbolFrom if DeviceOp ceases to be a
      // SymbolTable
      ModuleOp modOp = useLock->getParentOfType<ModuleOp>();
      func::FuncOp func = modOp.lookupSymbol<func::FuncOp>(funcName);

      int lockValue = useLock.getValue().value_or(1);

      // AIE2 acquire greater equal is encoded as a negative value.
      if (useLock.getAction() == LockAction::AcquireGreaterEqual)
        lockValue = -lockValue;

      rewriter.setInsertionPoint(useLock);
      IntegerAttr lockAttr = rewriter.getI32IntegerAttr(lockValue);
      IntegerType type = IntegerType::get(rewriter.getContext(), 32);
      Location loc = useLock.getLoc();

      SmallVector<Value, 2> args{
          rewriter.create<arith::IndexCastOp>(loc, type, useLock.getLock()),
          rewriter.create<arith::ConstantOp>(loc, type, lockAttr)};

      auto callOp = rewriter.create<func::CallOp>(loc, func, args);
      Operation *lockDefOp = useLock.getLock().getDefiningOp();
      int64_t id = lockIds.try_emplace(lockDefOp, (int64_t)lockIds.size())
                       .first->second;
      callOp->setAttr("amdaie_lock_id", rewriter.getI64IntegerAttr(id));
    }

    rewriter.eraseOp(useLock);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

static void bufferToStd(ModuleOp module, BufferOp buffer,
                        IRRewriter &rewriter) {
  Location loc = buffer.getLoc();
  rewriter.setInsertionPointToStart(module.getBody());
  StringRef symName = name(buffer).getValue();
  MemRefType type = llvm::cast<MemRefType>(buffer.getType());
  // Don't emit initialization for cores that don't "own" the buffer (to
  // prevent duplication in the data section of the elf/object file)
  rewriter.create<memref::GlobalOp>(
      loc, symName, rewriter.getStringAttr("public"), type, nullptr,
      /*constant*/ false,
      /*alignment*/ nullptr);

  for (OpOperand &use : make_early_inc_range(buffer.getResult().getUses())) {
    Operation *user = use.getOwner();
    rewriter.setInsertionPoint(user);

    auto allocated = rewriter.create<memref::GetGlobalOp>(loc, type, symName);
    // Assume that buffers are aligned so they can be vectorized.
    rewriter.create<memref::AssumeAlignmentOp>(loc, allocated, 32);
    use.set(allocated.getResult());
  }

  rewriter.eraseOp(buffer);
}

static void coreToStd(CoreOp coreOp, IRRewriter &rewriter, int tileCol,
                      int tileRow) {
  TileOp t = getTileOp(*coreOp);
  int col = t.getCol();
  int row = t.getRow();

  // Only pull code for the indicated function
  if ((tileRow != row && tileRow != -1) || (tileCol != col && tileCol != -1)) {
    rewriter.eraseOp(coreOp);
    return;
  }

  // The parent should be an AIE.device op.
  rewriter.setInsertionPointAfter(coreOp->getParentOp());

  // LLVM-style of the above (creating a string attribute):
  std::string fName = "core_" + std::to_string(col) + "_" + std::to_string(row);
  auto coreFunc = rewriter.create<func::FuncOp>(
      rewriter.getUnknownLoc(), fName,
      FunctionType::get(rewriter.getContext(), {}, {}));

  IRMapping mapper;
  rewriter.cloneRegionBefore(coreOp.getBody(), coreFunc.getBody(),
                             coreFunc.getBody().begin(), mapper);

  // EXPERIMENTAL: insert a macro-scale delay (real runtime loop, built out
  // of cf.br/cf.cond_br -- SCF-to-CF lowering doesn't run again this late)
  // immediately after the FIRST release of each distinct lock value only
  // (subsequent releases of the same lock ID -- i.e. batch 1, 2, 3, ...'s
  // releases -- get none). Tests the "lock starts pre-charged with an
  // initial credit, so only the very first acquire/release cycle skips a
  // real hardware wait" hypothesis: if this narrower placement still fully
  // fixes correctness, it directly confirms only the first use needed the
  // delay, and is a much cheaper fix shape than delaying after every
  // release. Not a real fix -- exploratory only.
  {
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<func::CallOp> releaseCalls;
    coreFunc.getBody().walk([&](func::CallOp callOp) {
      if (callOp.getCallee().ends_with(".release"))
        releaseCalls.push_back(callOp);
    });
    SmallVector<int64_t> seenLockIds;
    SmallVector<func::CallOp> firstReleaseCalls;
    for (func::CallOp releaseCall : releaseCalls) {
      auto idAttr =
          releaseCall->getAttrOfType<IntegerAttr>("amdaie_lock_id");
      if (!idAttr) continue;
      int64_t lockId = idAttr.getInt();
      if (llvm::is_contained(seenLockIds, lockId)) continue;
      seenLockIds.push_back(lockId);
      firstReleaseCalls.push_back(releaseCall);
    }
    for (func::CallOp releaseCall : firstReleaseCalls) {
      Block *curBlock = releaseCall->getBlock();
      rewriter.setInsertionPointAfter(releaseCall);
      Location loc = releaseCall.getLoc();
      Type i32Type = rewriter.getI32Type();
      Type ptrType = LLVM::LLVMPointerType::get(ctx);
      Value oneI32 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(1));
      Value alloca = rewriter.create<LLVM::AllocaOp>(loc, ptrType, i32Type,
                                                     oneI32, /*alignment=*/0);
      Value dummy = rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(0));
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(0));
      Value bound = rewriter.create<arith::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(10000));
      Value stepOne = rewriter.create<arith::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(1));

      Block *restBlock =
          rewriter.splitBlock(curBlock, rewriter.getInsertionPoint());
      Block *headerBlock = rewriter.createBlock(
          &coreFunc.getBody(), Region::iterator(restBlock),
          TypeRange{i32Type}, {loc});
      Block *bodyBlock = rewriter.createBlock(&coreFunc.getBody(),
                                              Region::iterator(restBlock));

      rewriter.setInsertionPointToEnd(curBlock);
      rewriter.create<cf::BranchOp>(loc, headerBlock, ValueRange{zero});

      rewriter.setInsertionPointToStart(headerBlock);
      Value iv = headerBlock->getArgument(0);
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, iv, bound);
      rewriter.create<cf::CondBranchOp>(loc, cond, bodyBlock, ValueRange{},
                                        restBlock, ValueRange{});

      rewriter.setInsertionPointToStart(bodyBlock);
      rewriter.create<LLVM::StoreOp>(loc, dummy, alloca, /*alignment=*/0,
                                     /*isVolatile=*/true);
      Value next = rewriter.create<arith::AddIOp>(loc, iv, stepOne);
      rewriter.create<cf::BranchOp>(loc, headerBlock, ValueRange{next});
    }
  }

  // Rewrite the AIE.end op
  coreFunc.getBody().walk([&](EndOp endOp) {
    rewriter.setInsertionPointAfter(endOp);
    rewriter.create<func::ReturnOp>(endOp->getLoc(), ValueRange({}));
    rewriter.eraseOp(endOp);
  });

  rewriter.eraseOp(coreOp);
}

// Move all the ops with OpTy inside device, to just before the device.
template <typename OpTy>
void outlineOps(DeviceOp device) {
  SmallVector<OpTy, 16> ops;
  for (const auto &op : device.getOps<OpTy>()) ops.push_back(op);
  for (const auto &op : ops) op->moveBefore(device);
}

namespace mlir::iree_compiler::AMDAIE {
struct AMDAIECoreToStandardPass
    : public impl::AMDAIECoreToStandardBase<AMDAIECoreToStandardPass> {
  AMDAIECoreToStandardPass(const AMDAIECoreToStandardOptions &options)
      : AMDAIECoreToStandardBase(options) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  // Assert that cores are isolated
  static bool coresAreIsolated(ModuleOp m) {
    SmallVector<CoreOp> coreOps;
    m->walk([&](CoreOp coreOp) { coreOps.push_back(coreOp); });
    for (CoreOp coreOp : coreOps) {
      auto walkResult = coreOp->walk([&](Operation *childOp) {
        if (childOp == coreOp) return WalkResult::advance();
        for (Value operand : childOp->getOperands()) {
          if (Operation *operandOp = operand.getDefiningOp()) {
            if (!coreOp->isAncestor(operandOp)) {
              operandOp->emitOpError(
                  "is not in the core in which it is used. Cores must be "
                  "`isolated` before this point.");
              return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted()) return false;
    }
    return true;
  }

  // Ensure that all aie.core ops are isolated from above, i.e. that all
  // operands of ops within an aie.core are produced inside the aie.core (or are
  // block arguments of the core). The expection is ops in the aie dialect --
  // operands produced by for example an aie.buffer may be outside the core.
  static void isolateCores(ModuleOp m) {
    IRRewriter rewriter(m->getContext());
    auto notAieDialect = [](Operation *op) -> bool {
      StringRef dialect = op->getDialect()->getNamespace();
      if (dialect == AIEDialect::getDialectNamespace()) return false;
      return true;
    };
    m->walk([&](CoreOp coreOp) {
      sinkInto(coreOp.getRegion(), rewriter, notAieDialect);
    });
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    if (m.getOps<DeviceOp>().empty()) {
      m.emitOpError("expected AIE.device operation at toplevel");
      return signalPassFailure();
    }
    DeviceOp deviceOp = *m.getOps<DeviceOp>().begin();
    AMDAIEDeviceModel deviceModel =
        getDeviceModel(static_cast<AMDAIEDevice>(deviceOp.getDevice()));

    std::optional<std::string> targetArch = deviceModel.getTargetArchString();
    if (!targetArch.has_value()) {
      deviceOp.emitError() << "doesn't have a target arch string";
      return signalPassFailure();
    }
    // Chess uses `aie2` for both aie2 and aie2p, while peano separates between
    // `aie2` and `aie2p`.
    std::string targetArchStr =
        lowerToChess ? "aie2" : StringRef(targetArch.value()).lower();

    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPointToEnd(m.getBody());

    // Ensure that we don't have an incorrect target triple. This may override
    // some bogus target triple in the original mlir.
    m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               rewriter.getStringAttr(targetArchStr));

    isolateCores(m);

    if (failed(lockToStd(rewriter, m, targetArchStr))) {
      return signalPassFailure();
    }

    m.walk([&](BufferOp buffer) { bufferToStd(m, buffer, rewriter); });

    if (!coresAreIsolated(m)) return signalPassFailure();

    m.walk([&](CoreOp coreOp) { coreToStd(coreOp, rewriter, -1, -1); });

    // Move all the func.func ops and memref.globals from device to module.
    outlineOps<memref::GlobalOp>(deviceOp);
    outlineOps<func::FuncOp>(deviceOp);
    rewriter.eraseOp(deviceOp);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAMDAIECoreToStandardPass(
    AMDAIECoreToStandardOptions options) {
  return std::make_unique<AMDAIECoreToStandardPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
