// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-amdaie-linalg-function-outlining"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to outline the linalg compute op.
static FailureOr<func::FuncOp> outline(IRRewriter &rewriter, ModuleOp moduleOp,
                                       linalg::LinalgOp computeOp,
                                       const std::string &funcName) {
  // Form outlined FunctionType.
  for (const auto &operand : computeOp->getOperands()) {
    // Function signatures where the memrefs have layouts (strides / offsets)
    // do not lower from the func dialect to the llvm dialect. So for now,
    // we just do not try to outline ops with operands with non-identity
    // layouts, because the resulting functions won't lower to LLVM.
    // identity layouts:
    // clang-format off
    // https://github.com/llvm/llvm-project/blob/6b0785390d02193d81d8db7fb12279ffa4651afe/mlir/include/mlir/IR/BuiltinAttributeInterfaces.td#L475
    // clang-format on
    if (auto type = dyn_cast<MemRefType>(operand.getType())) {
      MemRefLayoutAttrInterface layout = type.getLayout();
      assert(layout &&
             "MemRefType layout attribute interface should always be present");
      if (!layout.isIdentity()) return failure();
    }
  }
  // `computeOp`'s region(s) (e.g. a `linalg.generic`'s payload block) may
  // reference values from its original enclosing scope that are not among
  // its ins/outs operands above -- for example, a scalar `arith.constant`
  // an elementwise consumer's payload (like a ReLU's `cmpf`/`select`
  // against 0.0) captures directly. This must be detected on `computeOp`
  // itself, *before* cloning: `computeOp` is still properly nested inside
  // its original function here, so the enclosing scope is a genuine
  // ancestor and `getUsedValuesDefinedAbove` can find the capture. Doing
  // this check after cloning (on `clonedComputeOp`, once it has moved into
  // the new, disconnected function body) does not work -- the captured
  // value's defining op is then in an unrelated part of the IR tree with no
  // ancestor relationship to the clone at all, so the same query silently
  // finds nothing, even though the clone still structurally holds a
  // (now-dangling) reference to it.
  llvm::SetVector<Value> capturedValues;
  mlir::getUsedValuesDefinedAbove(computeOp->getRegions(), capturedValues);
  // Checked before anything is created, so that a failure leaves no
  // half-built function behind in the module.
  for (Value captured : capturedValues) {
    Operation *definingOp = captured.getDefiningOp();
    if (!definingOp || definingOp->getNumOperands() != 0 ||
        definingOp->getNumResults() != 1 || !mlir::isPure(definingOp)) {
      return failure();
    }
  }

  auto funcType = FunctionType::get(
      rewriter.getContext(), computeOp->getOperandTypes(), /*outputTypes=*/{});

  // Form outlined FuncSignature.
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto func =
      rewriter.create<func::FuncOp>(moduleOp.getLoc(), funcName, funcType);
  func.setPrivate();

  // Create an entry func block and map the original operands of the compute
  // op to the block arguments.
  Block *funcBody = func.addEntryBlock();
  rewriter.setInsertionPointToStart(funcBody);
  SmallVector<BlockArgument> funcArgs = llvm::map_to_vector(
      func.getArguments(), [&](BlockArgument bbArg) { return bbArg; });
  unsigned bbArgIndex = 0;
  IRMapping operandMap;
  for (Value origOperand : computeOp->getOperands())
    operandMap.map(origOperand, funcArgs[bbArgIndex++]);

  // Clone the compute op while mapping the operand to the function block
  // arguments.
  Operation *clonedComputeOp = rewriter.clone(*computeOp, operandMap);

  // `computeOp`'s block-argument signature is fixed by its own operand list
  // (unlike a function-like isolated region), so a capture can't be
  // resolved by appending an extra block argument the way
  // `makeRegionIsolatedFromAbove` would for a func body; instead,
  // re-materialize the (pure, zero-operand) defining op directly inside the
  // clone's region(s), replacing uses of the original (now-dangling)
  // captured value there. Without this, the outlined function body is left
  // referencing a value outside its own isolation boundary and fails to
  // verify -- which is why, before this fix, outlining anything with a
  // captured scalar (any fused elementwise consumer, e.g. bias/ReLU) had to
  // be skipped entirely (see `mustOutlineBalanced` below), leaving its code
  // duplicated inline at every call site of the outlined compute it's fused
  // with -- multiplying out to enough core-control-program size to overflow
  // AIE program memory for shapes like VGG16 conv4 (M=12544, N=128) once
  // elementwise fusion is enabled.
  for (Value captured : capturedValues) {
    Operation *definingOp = captured.getDefiningOp();
    for (Region &region : clonedComputeOp->getRegions()) {
      if (region.empty()) continue;
      rewriter.setInsertionPointToStart(&region.front());
      Operation *clonedDef = rewriter.clone(*definingOp);
      mlir::replaceAllUsesInRegionWith(captured, clonedDef->getResult(0), region);
    }
  }

  // Create terminator op returning the cloned compute op's results.
  rewriter.setInsertionPointToEnd(funcBody);
  rewriter.create<func::ReturnOp>(clonedComputeOp->getLoc(), ValueRange({}));

  return func;
}

/// Utility to check whether the linalg should be outlined if the balanced
/// strategy is enabled.
static bool mustOutlineBalanced(linalg::LinalgOp linalgOp) {
  if (isa<linalg::CopyOp, linalg::FillOp>(linalgOp)) return false;
  // Elementwise ops (e.g. a fused bias-add or ReLU consumer) used to be
  // excluded here unconditionally. That was papering over a real bug in
  // `outline()` above: such ops often capture a value from their enclosing
  // scope directly in their payload region (e.g. a ReLU's `cmpf`/`select`
  // against a 0.0 `arith.constant`), which `outline()` used to leave
  // dangling once the op moved into a new, isolated function body, failing
  // verification. Now that `outline()` re-materializes such captures, this
  // exclusion is no longer needed for correctness. Leaving it in place had
  // a real cost: an elementwise consumer fused with a repeatedly-tiled
  // contraction (matmul/conv) op is *not* outlined and *not* deduplicated
  // across call sites the way the contraction itself is, so its code gets
  // inlined at every one of the (possibly hundreds of) tiled call sites --
  // multiplying out to enough core-control-program size to overflow AIE
  // program memory for shapes like VGG16 conv4 (M=12544, N=128) once
  // elementwise fusion into the matmul dispatch is enabled.
  // TODO(newling) not all remaining ops should be outlined, not even all
  // remaining matmuls: below some threshold on size (m*n*k) it's not worth
  // outlining (function call overhead). We should extend the set of ops that
  // are not outlined here.

  return true;
};

class AMDAIELinalgFunctionOutliningPass
    : public impl::AMDAIELinalgFunctionOutliningBase<
          AMDAIELinalgFunctionOutliningPass> {
 public:
  AMDAIELinalgFunctionOutliningPass() = default;
  AMDAIELinalgFunctionOutliningPass(const AMDAIELinalgFunctionOutliningPass &) {
  }
  AMDAIELinalgFunctionOutliningPass(
      const AMDAIELinalgFunctionOutliningOptions &opts)
      : AMDAIELinalgFunctionOutliningBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, LLVM::LLVMDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override;

 private:
  // Used for unique-ifing ID for generating new function names.
  unsigned outlineCounter = 0;

  DenseMap<Operation *, func::FuncOp> computeOpToOutlinedFuncMap;

  static std::string getSpecializedName(linalg::LinalgOp computeOp) {
    // Will result in a function name like `generic_matmul_0_outlined`:
    if (isMatmul(computeOp)) return "_matmul_";
    // Will result in a function name like `generic_0_outlined`:
    return "_";
  }

  std::string generateFuncName(linalg::LinalgOp computeOp) {
    std::string name = computeOp->getName().stripDialect().str() +
                       getSpecializedName(computeOp) +
                       std::to_string(outlineCounter) + "_outlined";
    ++outlineCounter;
    return name;
  }

  FailureOr<func::FuncOp> retrieveOrCreate(IRRewriter &rewriter,
                                           ModuleOp moduleOp,
                                           linalg::LinalgOp computeOp) {
    // Check if the compute op is equivalent to a previously outlined compute
    // op. If it is, retrieve and return the function generated for the previous
    // compute op.
    //
    // Only the compute op's own operands may differ between the two -- they
    // become the outlined function's arguments, so every call site passes its
    // own. Everything else is baked into the outlined body and must match:
    //   * values defined inside the regions (block arguments, payload op
    //     results) must correspond one-to-one, so e.g. `subf %in, %out` and
    //     `subf %out, %in` are not merged;
    //   * values captured from above (see `outline`, which re-materializes
    //     their defining op inside the function) must be defined by
    //     equivalent ops, so e.g. a `mulf` by 2.0 and a `mulf` by 3.0 are not
    //     merged.
    llvm::SetVector<Value> captured;
    mlir::getUsedValuesDefinedAbove(computeOp->getRegions(), captured);
    llvm::SmallPtrSet<Value, 8> ownOperands(computeOp->operand_begin(),
                                            computeOp->operand_end());
    for (auto &[op, func] : computeOpToOutlinedFuncMap) {
      DenseMap<Value, Value> equivalentValues;
      auto checkEquivalent = [&](Value lhs, Value rhs) -> LogicalResult {
        auto it = equivalentValues.find(lhs);
        if (it != equivalentValues.end()) return success(it->second == rhs);
        if (captured.contains(lhs)) {
          Operation *lhsDef = lhs.getDefiningOp();
          Operation *rhsDef = rhs.getDefiningOp();
          return success(lhsDef && rhsDef &&
                         OperationEquivalence::isEquivalentTo(
                             lhsDef, rhsDef,
                             OperationEquivalence::exactValueMatch,
                             /*markEquivalent=*/nullptr,
                             OperationEquivalence::IgnoreLocations));
        }
        return success(ownOperands.contains(lhs));
      };
      auto markEquivalent = [&](Value lhs, Value rhs) {
        equivalentValues[lhs] = rhs;
      };
      if (OperationEquivalence::isEquivalentTo(
              computeOp.getOperation(), op, checkEquivalent, markEquivalent,
              OperationEquivalence::IgnoreLocations)) {
        return func;
      }
    }

    std::string funcName = generateFuncName(computeOp);
    while (moduleOp.lookupSymbol(funcName)) {
      funcName = generateFuncName(computeOp);
    }

    FailureOr<func::FuncOp> maybeFunc =
        outline(rewriter, moduleOp, computeOp, funcName);

    if (succeeded(maybeFunc)) {
      computeOpToOutlinedFuncMap[computeOp] = maybeFunc.value();
    }

    return maybeFunc;
  }
};

void AMDAIELinalgFunctionOutliningPass::runOnOperation() {
  if (outliningStrategy == OutliningStrategy::None) return;

  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<Operation *> toBeErased;
  moduleOp.walk([&](linalg::LinalgOp computeOp) {
    if (outliningStrategy == OutliningStrategy::Balanced &&
        !mustOutlineBalanced(computeOp)) {
      return WalkResult::skip();
    }

    FailureOr<func::FuncOp> maybeFunc =
        retrieveOrCreate(rewriter, moduleOp, computeOp);
    // Outlining is only a code-size optimization: an op that cannot be
    // outlined (see `outline`) stays inline, and the rest are still
    // outlined.
    if (failed(maybeFunc)) return WalkResult::skip();
    func::FuncOp func = maybeFunc.value();

    rewriter.setInsertionPoint(computeOp);

    rewriter.create<func::CallOp>(computeOp.getLoc(), func,
                                  computeOp->getOperands());

    // We cannot immediately erase the compute op because it'd be used for
    // equivalence check.
    toBeErased.push_back(computeOp);
    return WalkResult::advance();
  });
  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELinalgFunctionOutliningPass(
    AMDAIELinalgFunctionOutliningOptions options) {
  return std::make_unique<AMDAIELinalgFunctionOutliningPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
