// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "iree-amdaie-slice-core-constants"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// A loop variable an offset depends on, with the values it takes.
struct LoopVar {
  Value value;
  int64_t lower, step, count;
};

/// The loop that `value` is the induction variable of, if that loop is inside
/// `coreOp` and has constant bounds.
std::optional<LoopVar> asLoopVar(Value value, AMDAIE::CoreOp coreOp) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg) return std::nullopt;
  Operation *owner = arg.getOwner()->getParentOp();
  if (!coreOp->isProperAncestor(owner)) return std::nullopt;
  std::optional<int64_t> lb, ub, step;
  if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
    if (forOp.getInductionVar() != value) return std::nullopt;
    lb = getConstantIntValue(forOp.getLowerBound());
    ub = getConstantIntValue(forOp.getUpperBound());
    step = getConstantIntValue(forOp.getStep());
  } else if (auto forallOp = dyn_cast<scf::ForallOp>(owner)) {
    auto ivs = forallOp.getInductionVars();
    auto it = llvm::find(ivs, value);
    if (it == ivs.end()) return std::nullopt;
    size_t i = std::distance(ivs.begin(), it);
    lb = getConstantIntValue(forallOp.getMixedLowerBound()[i]);
    ub = getConstantIntValue(forallOp.getMixedUpperBound()[i]);
    step = getConstantIntValue(forallOp.getMixedStep()[i]);
  } else {
    return std::nullopt;
  }
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb) return std::nullopt;
  return LoopVar{value, *lb, *step,
                 static_cast<int64_t>(llvm::divideCeil(*ub - *lb, *step))};
}

/// Walk the index computation of `value`, collecting the loop variables it
/// reads. Fails on anything that is not constant arithmetic over them.
bool collectLoopVars(Value value, AMDAIE::CoreOp coreOp,
                     SmallVectorImpl<LoopVar> &vars) {
  if (std::optional<LoopVar> var = asLoopVar(value, coreOp)) {
    if (!llvm::any_of(vars, [&](const LoopVar &v) { return v.value == value; }))
      vars.push_back(*var);
    return true;
  }
  Operation *def = value.getDefiningOp();
  if (!def) return false;
  if (isa<arith::ConstantOp>(def)) return true;
  if (!isa<affine::AffineApplyOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
           arith::IndexCastOp>(def))
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return collectLoopVars(operand, coreOp, vars);
  });
}

/// Evaluate an index computation with the loop variables bound by `env`.
std::optional<int64_t> evaluate(Value value,
                                const DenseMap<Value, int64_t> &env) {
  if (auto it = env.find(value); it != env.end()) return it->second;
  if (std::optional<int64_t> c = getConstantIntValue(value)) return c;
  Operation *def = value.getDefiningOp();
  if (!def) return std::nullopt;
  auto operand = [&](unsigned i) { return evaluate(def->getOperand(i), env); };
  if (auto applyOp = dyn_cast<affine::AffineApplyOp>(def)) {
    SmallVector<Attribute> operands;
    Builder b(def->getContext());
    for (Value v : applyOp.getMapOperands()) {
      std::optional<int64_t> c = evaluate(v, env);
      if (!c) return std::nullopt;
      operands.push_back(b.getIndexAttr(*c));
    }
    SmallVector<Attribute> results;
    if (failed(applyOp.getAffineMap().constantFold(operands, results)) ||
        results.size() != 1)
      return std::nullopt;
    return cast<IntegerAttr>(results[0]).getInt();
  }
  if (isa<arith::IndexCastOp>(def)) return operand(0);
  std::optional<int64_t> lhs = operand(0), rhs = operand(1);
  if (!lhs || !rhs) return std::nullopt;
  if (isa<arith::AddIOp>(def)) return *lhs + *rhs;
  if (isa<arith::SubIOp>(def)) return *lhs - *rhs;
  if (isa<arith::MulIOp>(def)) return *lhs * *rhs;
  return std::nullopt;
}

/// The largest number of loop iterations we enumerate for one view.
constexpr int64_t kMaxIterations = 4096;

/// Give each core a table of only the elements it reads.
///
/// A core-local constant (`__amdaie_core_constant_*`, see
/// `ReadStagedConstantFromCoreMemory`) holds the whole per-channel vector, and
/// every core of the dispatch carries all of it although each reads only the
/// channels of its own column, a run at a time. Once the workgroup is built
/// each core's offsets are a function of its own constant-bound loops, so the
/// runs it reads can be listed: they are copied, in iteration order, into a
/// table of their own, and the view reads run `i` at `i * size`. With the
/// object compiled with `--data-sections`, the linker keeps in each core only
/// the tables that core names.
void sliceCoreConstants(AMDAIE::CoreOp coreOp, ModuleOp moduleOp,
                        SymbolTable &symbolTable,
                        DenseMap<std::pair<Type, Attribute>, StringAttr> &made,
                        unsigned &counter) {
  SmallVector<memref::SubViewOp> views;
  coreOp.walk([&](memref::SubViewOp op) { views.push_back(op); });
  for (memref::SubViewOp view : views) {
    auto getGlobal = view.getSource().getDefiningOp<memref::GetGlobalOp>();
    if (!getGlobal ||
        !getGlobal.getName().starts_with("__amdaie_core_constant_"))
      continue;
    auto global = symbolTable.lookup<memref::GlobalOp>(getGlobal.getName());
    if (!global) continue;
    auto data = dyn_cast_if_present<DenseElementsAttr>(
        global.getInitialValueAttr());
    MemRefType type = global.getType();
    if (!data || data.isSplat() || type.getRank() != 1 ||
        !type.hasStaticShape() || type.getElementTypeBitWidth() % 8 != 0)
      continue;
    if (view.getMixedOffsets().size() != 1) continue;
    Value offset = dyn_cast<Value>(view.getMixedOffsets()[0]);
    if (!offset) continue;
    std::optional<int64_t> size = getConstantIntValue(view.getMixedSizes()[0]);
    std::optional<int64_t> stride =
        getConstantIntValue(view.getMixedStrides()[0]);
    if (!size || !stride || *stride != 1) continue;

    SmallVector<LoopVar> vars;
    if (!collectLoopVars(offset, coreOp, vars)) continue;
    int64_t iterations = 1;
    for (const LoopVar &var : vars) iterations *= var.count;
    int64_t total = type.getNumElements();
    if (iterations > kMaxIterations || iterations * *size >= total) continue;

    // Copy the runs in iteration order, the first loop variable outermost.
    int64_t elementBytes = type.getElementTypeBitWidth() / 8;
    ArrayRef<char> raw = data.getRawData();
    SmallVector<char> slice;
    slice.reserve(iterations * *size * elementBytes);
    DenseMap<Value, int64_t> env;
    bool ok = true;
    std::function<void(unsigned)> enumerate = [&](unsigned i) {
      if (!ok) return;
      if (i == vars.size()) {
        std::optional<int64_t> start = evaluate(offset, env);
        if (!start || *start < 0 || *start + *size > total) {
          ok = false;
          return;
        }
        slice.append(raw.begin() + *start * elementBytes,
                     raw.begin() + (*start + *size) * elementBytes);
        return;
      }
      for (int64_t k = 0; k < vars[i].count; ++k) {
        env[vars[i].value] = vars[i].lower + k * vars[i].step;
        enumerate(i + 1);
      }
    };
    enumerate(0);
    if (!ok) continue;

    auto sliceType =
        MemRefType::get({iterations * *size}, type.getElementType());
    auto sliceData = DenseElementsAttr::getFromRawBuffer(
        RankedTensorType::get(sliceType.getShape(), type.getElementType()),
        slice);
    StringAttr &name = made[{sliceType, sliceData}];
    OpBuilder builder(view);
    if (!name) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(global);
      auto sliceGlobal = builder.create<memref::GlobalOp>(
          global.getLoc(),
          (global.getSymName() + "_s" + std::to_string(counter++)).str(),
          builder.getStringAttr("private"), sliceType, sliceData,
          /*constant=*/true, global.getAlignmentAttr());
      symbolTable.insert(sliceGlobal);
      name = sliceGlobal.getSymNameAttr();
    }

    // The run's index in iteration order, then its start in the slice.
    Location loc = view.getLoc();
    Value index = builder.create<arith::ConstantIndexOp>(loc, 0);
    for (const LoopVar &var : vars) {
      Value k = var.value;
      if (var.lower != 0)
        k = builder.create<arith::SubIOp>(
            loc, k, builder.create<arith::ConstantIndexOp>(loc, var.lower));
      if (var.step != 1)
        k = builder.create<arith::DivUIOp>(
            loc, k, builder.create<arith::ConstantIndexOp>(loc, var.step));
      index = builder.create<arith::AddIOp>(
          loc,
          builder.create<arith::MulIOp>(
              loc, index,
              builder.create<arith::ConstantIndexOp>(loc, var.count)),
          k);
    }
    Value start = builder.create<arith::MulIOp>(
        loc, index, builder.create<arith::ConstantIndexOp>(loc, *size));
    Value sliceRef =
        builder.create<memref::GetGlobalOp>(loc, sliceType, name.getValue());
    auto newView = builder.create<memref::SubViewOp>(
        loc, view.getType(), sliceRef, ArrayRef<OpFoldResult>{start},
        view.getMixedSizes(), view.getMixedStrides());
    view.replaceAllUsesWith(newView.getResult());
    view.erase();
    if (getGlobal->use_empty()) getGlobal.erase();
  }
}

class AMDAIESliceCoreConstantsPass
    : public impl::AMDAIESliceCoreConstantsBase<AMDAIESliceCoreConstantsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    Operation *root = getOperation();
    SmallVector<ModuleOp> modules;
    if (auto m = dyn_cast<ModuleOp>(root)) modules.push_back(m);
    root->walk([&](ModuleOp m) {
      if (m != root) modules.push_back(m);
    });
    for (ModuleOp moduleOp : modules) {
      SymbolTable symbolTable(moduleOp);
      DenseMap<std::pair<Type, Attribute>, StringAttr> made;
      unsigned counter = 0;
      SmallVector<AMDAIE::CoreOp> cores;
      moduleOp.walk([&](AMDAIE::CoreOp c) {
        if (c->getParentOfType<ModuleOp>() == moduleOp) cores.push_back(c);
      });
      for (AMDAIE::CoreOp coreOp : cores)
        sliceCoreConstants(coreOp, moduleOp, symbolTable, made, counter);
      // Whole tables no core names any more: drop their reads and themselves.
      SmallVector<memref::GetGlobalOp> deadReads;
      moduleOp.walk([&](memref::GetGlobalOp op) {
        if (op.getName().starts_with("__amdaie_core_constant_") &&
            op->use_empty())
          deadReads.push_back(op);
      });
      for (memref::GetGlobalOp op : deadReads) op.erase();
      for (auto global :
           llvm::make_early_inc_range(moduleOp.getOps<memref::GlobalOp>())) {
        if (global.getSymName().starts_with("__amdaie_core_constant_") &&
            SymbolTable::symbolKnownUseEmpty(global, moduleOp))
          global.erase();
      }
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIESliceCoreConstantsPass() {
  return std::make_unique<AMDAIESliceCoreConstantsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
