// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Converts the inputs of contraction dispatches to the element type their
// device asks for, after device placement.
//
// Up to here the model keeps its own types (f32), and every dispatch boundary
// stays in them; a device whose executable target carries
// `contraction_input_type` (e.g. bf16, set on amd-aie by
// --iree-amdaie-contraction-input-type) gets the inputs of its contraction
// dispatches converted to that type, the accumulator and the output keeping
// theirs. Dispatches placed on any other device are left alone, so a GPU that
// cannot take bf16 keeps f32 in the same model.
//
// Runs right after `AMDAIEAssignDeviceAffinities` (every `flow.dispatch` has a
// `stream.affinity`) and before the bias fold and the pads, which size their
// work by the element type. Three steps:
//   1. analysis: in each dispatch whose device asks for a type, the named
//      contraction ops (matmul/conv family, `ContractionOpInterface` or
//      `ConvolutionOpInterface` with a multiply) whose inputs are full-tensor
//      loads of bindings read nowhere else; per converted operand, the value it
//      comes from, through `flow.tensor.reshape`s;
//   2. decision, per value, in this order:
//      a. a compile-time constant: converted at compile time into a new global
//         (the f32 global stays while something else reads it);
//      b. the result of a dispatch, read only by dispatches that want it: the
//         conversion is fused into that dispatch before its store, so it
//         writes the narrow type, whatever device it runs on (host, GPU or
//         another accelerator), unless that device's target opts out
//         (`fuse_truncf = false`, set on amd-aie whose cores do not round to
//         nearest even);
//      c. otherwise (a model input, a value something else reads in f32, a
//         producer that opts out): a host dispatch converting it in front of
//         the consumers, the fallback;
//   3. rewrite: the executables of the converting dispatches see the narrow
//      bindings, loads and contraction inputs, and each value is converted as
//      decided. An executable shared with dispatches that do not convert is
//      copied first.
// The conversion rounds to nearest even everywhere (`arith.truncf`), so the
// result does not depend on where it happens.

#include "iree-amd-aie/Transforms/AMDAIEDispatchRewriteUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDevicePlacementUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#define DEBUG_TYPE "iree-amdaie-demote-contraction-dispatch-inputs"

namespace mlir::iree_compiler::AMDAIE {

namespace {

using IREE::TensorExt::DispatchTensorLoadOp;
using IREE::TensorExt::DispatchTensorStoreOp;
using IREE::TensorExt::DispatchTensorType;

/// Whether a dispatch on `target` can write `type` itself, so that a
/// conversion can be fused into it. Any device can, whatever its role (host,
/// GPU, another accelerator), unless its target opts out with
/// `fuse_truncf = false` (see `getConfigFuseTruncF`): the producer of a value
/// a narrow-input device reads converts it, and a host cast dispatch is the
/// fallback only.
static bool canWriteElementType(IREE::HAL::ExecutableTargetAttr target,
                                FloatType type) {
  return target && getConfigFuseTruncF(target);
}

static RankedTensorType withElementType(Type type, Type elemType) {
  auto tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(), elemType,
                               tensorType.getEncoding());
}

/// Whether `op` is a named contraction or convolution (with a multiply, so not
/// a pooling) whose inputs can be converted to `type`: float inputs wider than
/// it with static shapes, and a region builder to rebuild the op with.
static bool isDemotableContraction(linalg::LinalgOp op, FloatType type) {
  if (!isa<linalg::ContractionOpInterface, linalg::ConvolutionOpInterface>(
          op.getOperation()) ||
      isa<linalg::GenericOp>(op) || !op.getRegionBuilder() ||
      !op.hasPureTensorSemantics())
    return false;
  bool hasMul = false;
  op->getRegion(0).walk([&](arith::MulFOp) { hasMul = true; });
  if (!hasMul) return false;
  return llvm::all_of(op.getDpsInputs(), [&](Value input) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto elemType =
        inputType ? dyn_cast<FloatType>(inputType.getElementType()) : nullptr;
    return elemType && inputType.hasStaticShape() &&
           elemType.getWidth() > type.getWidth();
  });
}

/// The binding a contraction input is loaded from in full, or null.
static BlockArgument getLoadedBinding(Value input) {
  DispatchTensorLoadOp load = getFullTensorLoad(input);
  if (!load) return {};
  auto arg = dyn_cast<BlockArgument>(load.getSource());
  if (!arg) return {};
  auto dtt = cast<DispatchTensorType>(arg.getType());
  if (dtt.getAccess() != IREE::TensorExt::TensorAccess::ReadOnly ||
      dtt.asRankedTensorType() != load.getResult().getType())
    return {};
  return arg;
}

/// The contractions of a dispatch func whose inputs are converted, and the
/// bindings (input argument numbers) they read.
struct FuncPlan {
  SmallVector<linalg::LinalgOp> ops;
  llvm::SmallSetVector<unsigned, 4> bindings;
};

/// Finds the contractions of `func` whose every input is a full-tensor load of
/// a read-only binding that nothing but such contractions reads, so the
/// binding can change type without touching anything else.
static FuncPlan analyzeFunc(func::FuncOp func, FloatType type) {
  llvm::SetVector<Operation *> candidates;
  func.walk([&](linalg::LinalgOp op) {
    if (!isDemotableContraction(op, type)) return;
    if (llvm::all_of(op.getDpsInputs(),
                     [](Value input) { return getLoadedBinding(input); }))
      candidates.insert(op);
  });
  // A binding is converted only if all its readers are converted: drop the
  // candidates reading a binding (or load) used elsewhere, until stable.
  auto bindingOnlyFeedsCandidates = [&](BlockArgument arg) {
    for (Operation *user : arg.getUsers()) {
      auto load = dyn_cast<DispatchTensorLoadOp>(user);
      if (!load) return false;
      for (OpOperand &use : load.getResult().getUses()) {
        auto op = dyn_cast<linalg::LinalgOp>(use.getOwner());
        if (!op || !candidates.contains(op) || !op.isDpsInput(&use))
          return false;
      }
    }
    return true;
  };
  // Collect a round's drops first and remove them after it: removing from a
  // SetVector shifts its elements, which would skip or overrun the iteration.
  for (bool changed = true; changed;) {
    SmallVector<Operation *> dropped;
    for (Operation *op : candidates) {
      auto linalgOp = cast<linalg::LinalgOp>(op);
      if (!llvm::all_of(linalgOp.getDpsInputs(), [&](Value input) {
            return bindingOnlyFeedsCandidates(getLoadedBinding(input));
          }))
        dropped.push_back(op);
    }
    for (Operation *op : dropped) candidates.remove(op);
    changed = !dropped.empty();
  }
  FuncPlan plan;
  for (Operation *op : candidates) {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    plan.ops.push_back(linalgOp);
    for (Value input : linalgOp.getDpsInputs())
      plan.bindings.insert(getLoadedBinding(input).getArgNumber());
  }
  return plan;
}

/// Rebuilds the named linalg `op` from its current operands, so its body takes
/// their (converted) element types: the region builder casts them to the
/// accumulator type, as for a mixed-precision op built from scratch.
static void rebuildNamedOp(IRRewriter &rewriter, linalg::LinalgOp op) {
  Location loc = op.getLoc();
  OperationState state(loc, op->getName());
  state.addOperands(op->getOperands());
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  Region *region = state.addRegion();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    SmallVector<Type> argTypes;
    for (Value v : op->getOperands())
      argTypes.push_back(getElementTypeOrSelf(v.getType()));
    SmallVector<Location> argLocs(argTypes.size(), loc);
    Block *body = rewriter.createBlock(region, {}, argTypes, argLocs);
    ImplicitLocOpBuilder b(loc, rewriter);
    op.getRegionBuilder()(b, *body, state.attributes.getAttrs(),
                          /*emitError=*/{});
  }
  rewriter.setInsertionPoint(op);
  Operation *newOp = rewriter.create(state);
  rewriter.replaceOp(op, newOp->getResults());
}

/// Rewrites `func` per `plan`: each converted binding and its loads take
/// `type`, and each contraction is rebuilt on the narrow inputs.
static void demoteFunc(IRRewriter &rewriter, func::FuncOp func,
                       const FuncPlan &plan, FloatType type) {
  for (unsigned argNo : plan.bindings) {
    BlockArgument arg = func.getArgument(argNo);
    auto dtt = cast<DispatchTensorType>(arg.getType());
    RankedTensorType newType =
        withElementType(dtt.asRankedTensorType(), type);
    arg.setType(DispatchTensorType::get(dtt.getAccess(), newType));
    for (Operation *user : llvm::make_early_inc_range(arg.getUsers())) {
      auto load = cast<DispatchTensorLoadOp>(user);
      rewriter.setInsertionPoint(load);
      auto newLoad = rewriter.create<DispatchTensorLoadOp>(
          load.getLoc(), newType, arg, /*sourceDynamicDims=*/ValueRange{});
      rewriter.replaceOp(load, newLoad.getResult());
    }
  }
  for (linalg::LinalgOp op : plan.ops) rebuildNamedOp(rewriter, op);
  SmallVector<Type> argTypes(llvm::map_range(
      func.getArguments(), [](BlockArgument a) { return a.getType(); }));
  func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));
}

/// Returns `v` converted to `elemType` by an elementwise `linalg.generic`.
static Value createTruncF(OpBuilder &b, Location loc, Value v,
                          FloatType elemType) {
  auto type = cast<RankedTensorType>(v.getType());
  Value empty =
      b.create<tensor::EmptyOp>(loc, type.getShape(), elemType).getResult();
  SmallVector<AffineMap> maps(2, b.getMultiDimIdentityMap(type.getRank()));
  SmallVector<utils::IteratorType> iterators(type.getRank(),
                                             utils::IteratorType::parallel);
  return b
      .create<linalg::GenericOp>(
          loc, TypeRange{withElementType(type, elemType)}, ValueRange{v},
          ValueRange{empty}, maps, iterators,
          [&](OpBuilder &nb, Location nl, ValueRange args) {
            Value narrow = nb.create<arith::TruncFOp>(nl, elemType, args[0]);
            nb.create<linalg::YieldOp>(nl, narrow);
          })
      .getResult(0);
}

/// When `v` is the only result of an all-parallel `linalg.generic` read
/// nowhere else, whose init is an unread `tensor.empty`, rebuilds that generic
/// to yield `elemType` (the yielded value converted in its body), as
/// elementwise fusion would have made it, and returns the new result. Null
/// otherwise.
static Value fuseTruncFIntoGeneric(IRRewriter &rewriter, Value v,
                                   FloatType elemType) {
  auto generic = v.getDefiningOp<linalg::GenericOp>();
  if (!generic || generic.getNumResults() != 1 || !v.hasOneUse() ||
      generic.getNumParallelLoops() != generic.getNumLoops())
    return {};
  OpOperand *init = generic.getDpsInitOperand(0);
  auto empty = init->get().getDefiningOp<tensor::EmptyOp>();
  if (!empty || generic.payloadUsesValueFromOperand(init)) return {};
  auto yield = cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  if (!isa<FloatType>(yield.getOperand(0).getType())) return {};

  Location loc = generic.getLoc();
  auto type = cast<RankedTensorType>(v.getType());
  rewriter.setInsertionPoint(generic);
  Value newEmpty =
      rewriter.create<tensor::EmptyOp>(loc, type.getShape(), elemType);
  auto newGeneric = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{withElementType(type, elemType)}, generic.getDpsInputs(),
      ValueRange{newEmpty}, generic.getIndexingMapsArray(),
      generic.getIteratorTypesArray());
  rewriter.inlineRegionBefore(generic.getRegion(), newGeneric.getRegion(),
                              newGeneric.getRegion().end());
  Block *body = newGeneric.getBody();
  body->getArgument(body->getNumArguments() - 1).setType(elemType);
  rewriter.setInsertionPoint(yield);
  Value narrow =
      rewriter.create<arith::TruncFOp>(yield.getLoc(), elemType,
                                       yield.getOperand(0));
  rewriter.modifyOpInPlace(yield, [&] { yield->setOperand(0, narrow); });
  rewriter.replaceAllUsesWith(v, newGeneric.getResult(0));
  rewriter.eraseOp(generic);
  if (empty.use_empty()) rewriter.eraseOp(empty);
  return newGeneric.getResult(0);
}

/// Returns the full-tensor store into `arg`, its only user, or null.
static DispatchTensorStoreOp getFullTensorStore(BlockArgument arg) {
  if (!arg.hasOneUse()) return {};
  auto store = dyn_cast<DispatchTensorStoreOp>(*arg.getUsers().begin());
  if (!store || store.getTarget() != arg) return {};
  for (OpFoldResult o : store.getMixedOffsets())
    if (!isConstantIntValue(o, 0)) return {};
  for (OpFoldResult s : store.getMixedStrides())
    if (!isConstantIntValue(s, 1)) return {};
  auto dtt = cast<DispatchTensorType>(arg.getType());
  if (dtt.getAccess() != IREE::TensorExt::TensorAccess::WriteOnly ||
      store.getValue().getType() != dtt.asRankedTensorType())
    return {};
  return store;
}

/// Returns `v` converted to `elemType` at compile time, as a load of a new
/// immutable global, when `v` is a dense float constant (see
/// `getConstantValue`). `cache` maps a source and shape to the global created
/// for it, so a constant read by several dispatches is converted once. Null
/// otherwise.
static Value createDemotedConstant(
    IRRewriter &rewriter, ModuleOp module, Value v, FloatType elemType,
    int &counter, DenseMap<std::pair<Value, Type>, Operation *> &cache,
    SmallVectorImpl<IREE::Util::GlobalOp> &replacedGlobals) {
  Value root = v;
  while (auto reshapeOp = root.getDefiningOp<IREE::Flow::TensorReshapeOp>())
    root = reshapeOp.getSource();
  auto key = std::make_pair(root, v.getType());
  if (Operation *load = cache.lookup(key)) {
    Operation *copy = rewriter.clone(*load);
    return copy->getResult(0);
  }
  IREE::Util::GlobalOp global;
  auto attr = dyn_cast_if_present<DenseFPElementsAttr>(
      getConstantValue(v, &global));
  if (!attr) return {};
  const llvm::fltSemantics &semantics = elemType.getFloatSemantics();
  DenseElementsAttr demoted =
      attr.mapValues(elemType, [&](const APFloat &value) {
        APFloat narrow = value;
        bool losesInfo;
        narrow.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
        return narrow.bitcastToAPInt();
      });
  Value load = createReplacementConstant(
      rewriter, module, v.getLoc(),
      "__demoted_constant_" + std::to_string(counter++), demoted, global,
      replacedGlobals);
  cache[key] = load.getDefiningOp();
  return load;
}

/// Erases the reshapes, global loads and constants `values` were read through
/// (as `eraseDeadProducers`), as far as nothing else reads them. The values
/// may share a chain (one a reshape of another), so the ops are collected
/// first and erased as they become unused.
static void eraseDeadReshapeChains(IRRewriter &rewriter, ArrayRef<Value> values) {
  llvm::SetVector<Operation *> chain;
  for (Value v : values) {
    for (Operation *op = v.getDefiningOp(); op;) {
      if (!isa<IREE::Flow::TensorReshapeOp, IREE::Util::GlobalLoadOp,
               arith::ConstantOp>(op))
        break;
      chain.insert(op);
      auto reshapeOp = dyn_cast<IREE::Flow::TensorReshapeOp>(op);
      op = reshapeOp ? reshapeOp.getSource().getDefiningOp() : nullptr;
    }
  }
  SmallVector<Operation *> live = chain.takeVector();
  for (bool changed = true; changed;) {
    changed = false;
    for (Operation *&op : live) {
      if (!op || !op->use_empty()) continue;
      rewriter.eraseOp(op);
      op = nullptr;
      changed = true;
    }
  }
}

/// Creates a `flow.executable` + `flow.dispatch` on `hostAffinity` that
/// converts `v` to `elemType`, and returns the converted value. `counter`
/// uniquifies the symbol names.
static Value createCastDispatch(IRRewriter &rewriter, ModuleOp module, Value v,
                                FloatType elemType,
                                IREE::HAL::DeviceAffinityAttr hostAffinity,
                                int &counter) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  RankedTensorType dstType = withElementType(srcType, elemType);
  auto inBinding = DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, srcType);
  auto outBinding = DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::WriteOnly, dstType);
  std::string idx = std::to_string(counter++);

  // Worker func: load the full input, convert it elementwise, store.
  auto funcOp = func::FuncOp::create(
      loc, "cast_dispatch_" + idx,
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value loaded = fb.create<DispatchTensorLoadOp>(
      loc, srcType, entry->getArgument(0), /*sourceDynamicDims=*/ValueRange{});
  Value narrow = createTruncF(fb, loc, loaded, elemType);
  fb.create<DispatchTensorStoreOp>(loc, narrow, entry->getArgument(1),
                                   /*targetDynamicDims=*/ValueRange{});
  fb.create<func::ReturnOp>(loc);

  // Executable wrapping the func, with a default workgroup-count region.
  OpBuilder mb(&module.getBody()->back());
  auto exeOp =
      IREE::Flow::ExecutableOp::create(mb, loc, "cast_executable_" + idx);
  exeOp.setPrivate();
  OpBuilder exeBuilder = OpBuilder::atBlockBegin(&exeOp.getBlock());
  auto innerModule = mlir::ModuleOp::create(exeBuilder, loc);
  innerModule.push_back(funcOp);
  OpBuilder exportBuilder(exeOp.getBody());
  auto exportOp = IREE::Flow::ExecutableExportOp::create(
      exportBuilder, loc, funcOp.getName(), SymbolRefAttr::get(funcOp));
  Block *wcBlock = &exportOp.getWorkgroupCount().emplaceBlock();
  OpBuilder wb = OpBuilder::atBlockBegin(wcBlock);
  Type indexTy = wb.getIndexType();
  auto countOp = wb.create<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
      loc, TypeRange{indexTy, indexTy, indexTy},
      /*ordinal_operands=*/ValueRange{});
  wb.create<IREE::Flow::ReturnOp>(loc, countOp.getResults());

  // Host dispatch, pinned to the host device.
  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{}, /*arguments=*/ValueRange{v},
      /*argument_dims=*/ValueRange{}, /*tied_operands=*/ArrayAttr{});
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}

/// The callers of each dispatch func, kept current as executables are copied.
using CallerMap = DenseMap<Operation *, SmallVector<IREE::Flow::DispatchOp>>;

/// Points `dispatches` (some, not all, callers of one executable) at a copy of
/// that executable, so rewriting the copy leaves the other callers alone.
/// Returns the copy's func.
static func::FuncOp copyExecutableFor(
    ArrayRef<IREE::Flow::DispatchOp> dispatches, ModuleOp module,
    CallerMap &callers) {
  IREE::Flow::DispatchOp first = dispatches.front();
  func::FuncOp oldFunc = getDispatchFunc(first, module);
  SymbolRefAttr ref = *first.getEntryPointRefs().begin();
  auto exe =
      module.lookupSymbol<IREE::Flow::ExecutableOp>(ref.getRootReference());
  auto copy = cast<IREE::Flow::ExecutableOp>(exe->clone());
  // Inserting through a symbol table renames the copy to a unique name.
  SymbolTable(module).insert(copy, std::next(Block::iterator(exe)));
  auto newRef =
      SymbolRefAttr::get(copy.getSymNameAttr(), ref.getNestedReferences());
  auto entryPoints = ArrayAttr::get(module.getContext(), {newRef});
  SmallVector<IREE::Flow::DispatchOp> &oldCallers = callers[oldFunc];
  for (IREE::Flow::DispatchOp dispatch : dispatches) {
    dispatch.setEntryPointsAttr(entryPoints);
    llvm::erase(oldCallers, dispatch);
  }
  func::FuncOp newFunc = getDispatchFunc(first, module);
  callers[newFunc].assign(dispatches.begin(), dispatches.end());
  return newFunc;
}

/// Collects the uses of `v` that read it as is or through
/// `flow.tensor.reshape`s (the reshapes themselves into `reshapes`).
static void collectLeafUses(Value v, SmallVectorImpl<OpOperand *> &leaves,
                            SmallVectorImpl<Operation *> &reshapes) {
  for (OpOperand &use : v.getUses()) {
    auto reshapeOp = dyn_cast<IREE::Flow::TensorReshapeOp>(use.getOwner());
    if (reshapeOp && use.get() == reshapeOp.getSource()) {
      reshapes.push_back(reshapeOp);
      collectLeafUses(reshapeOp.getResult(), leaves, reshapes);
      continue;
    }
    leaves.push_back(&use);
  }
}

/// A value converted to `type`, and the dispatch operands that want it so.
struct Request {
  FloatType type;
  SmallVector<OpOperand *> uses;
};

class AMDAIEDemoteContractionDispatchInputsPass
    : public impl::AMDAIEDemoteContractionDispatchInputsBase<
          AMDAIEDemoteContractionDispatchInputsPass> {
 public:
  void runOnOperation() override;

 private:
  /// Fuses the conversion of `root` (a result of `producer`) into `producer`
  /// when it runs on a device that can write `type` and nothing but `uses`
  /// reads `root`. Returns false, changing nothing, otherwise.
  bool fuseIntoProducer(IRRewriter &rewriter, ModuleOp module,
                        IREE::Flow::DispatchOp producer, OpResult root,
                        const Request &request, CallerMap &callers);
};

bool AMDAIEDemoteContractionDispatchInputsPass::fuseIntoProducer(
    IRRewriter &rewriter, ModuleOp module, IREE::Flow::DispatchOp producer,
    OpResult root, const Request &request, CallerMap &callers) {
  if (!canWriteElementType(getDispatchExecutableTarget(producer, module),
                           request.type))
    return false;
  unsigned resultNo = root.getResultNumber();
  if (producer.getTiedResultOperandIndex(resultNo)) return false;
  auto rootType = dyn_cast<RankedTensorType>(root.getType());
  if (!rootType || !rootType.hasStaticShape()) return false;
  SmallVector<OpOperand *> leaves;
  SmallVector<Operation *> reshapes;
  collectLeafUses(root, leaves, reshapes);
  llvm::SmallPtrSet<OpOperand *, 4> wanted(request.uses.begin(),
                                            request.uses.end());
  if (!llvm::all_of(leaves, [&](OpOperand *use) { return wanted.count(use); }))
    return false;

  // The result's binding: the func takes the dispatch arguments, then one
  // binding per untied result.
  func::FuncOp func = getDispatchFunc(producer, module);
  if (!func) return false;
  unsigned argNo = producer.getArguments().size();
  unsigned numUntied = 0;
  for (unsigned r = 0, e = producer.getNumResults(); r < e; ++r) {
    if (producer.getTiedResultOperandIndex(r)) continue;
    if (r < resultNo) ++argNo;
    ++numUntied;
  }
  if (func.getNumArguments() != producer.getArguments().size() + numUntied)
    return false;
  if (!getFullTensorStore(func.getArgument(argNo))) return false;
  if (callers[func].size() > 1)
    func = copyExecutableFor({producer}, module, callers);

  // Convert before the store and narrow the binding.
  BlockArgument arg = func.getArgument(argNo);
  DispatchTensorStoreOp store = getFullTensorStore(arg);
  Value narrow = fuseTruncFIntoGeneric(rewriter, store.getValue(), request.type);
  if (!narrow) {
    rewriter.setInsertionPoint(store);
    narrow = createTruncF(rewriter, store.getLoc(), store.getValue(),
                          request.type);
  }
  auto dtt = cast<DispatchTensorType>(arg.getType());
  arg.setType(DispatchTensorType::get(
      dtt.getAccess(), withElementType(dtt.asRankedTensorType(), request.type)));
  rewriter.setInsertionPoint(store);
  rewriter.create<DispatchTensorStoreOp>(store.getLoc(), narrow, arg,
                                         /*targetDynamicDims=*/ValueRange{});
  rewriter.eraseOp(store);
  SmallVector<Type> argTypes(llvm::map_range(
      func.getArguments(), [](BlockArgument a) { return a.getType(); }));
  func.setType(rewriter.getFunctionType(argTypes, /*results=*/{}));

  // The dispatch result and the reshapes up to the consumers narrow too.
  root.setType(withElementType(root.getType(), request.type));
  for (Operation *reshape : reshapes)
    reshape->getResult(0).setType(
        withElementType(reshape->getResult(0).getType(), request.type));
  return true;
}

void AMDAIEDemoteContractionDispatchInputsPass::runOnOperation() {
  ModuleOp module = getOperation();
  IRRewriter rewriter(module.getContext());

  IREE::HAL::DeviceAffinityAttr hostAffinity = getHostAffinity(module);
  if (!hostAffinity) return;  // need a host device for the cast dispatches
  int counter = firstFreeHelperIndex(module);

  SmallVector<IREE::Flow::DispatchOp> dispatches;
  CallerMap callers;
  module.walk([&](IREE::Flow::DispatchOp d) {
    dispatches.push_back(d);
    if (func::FuncOp func = getDispatchFunc(d, module))
      callers[func].push_back(d);
  });

  // 1. Analysis: the dispatches whose device asks for a contraction input
  // type, grouped by executable and type.
  llvm::MapVector<std::pair<Operation *, Type>,
                  SmallVector<IREE::Flow::DispatchOp>>
      groups;
  for (IREE::Flow::DispatchOp dispatch : dispatches) {
    FloatType type = getConfigContractionInputType(
        getDispatchExecutableTarget(dispatch, module));
    if (!type) continue;
    func::FuncOp func = getDispatchFunc(dispatch, module);
    if (!func) continue;
    groups[{func.getOperation(), type}].push_back(dispatch);
  }

  // Rewrite each group's executable (a copy when other dispatches call it
  // too) and record, per converted operand, the value it reads.
  llvm::MapVector<std::pair<Value, Type>, Request> requests;
  for (auto &[key, group] : groups) {
    auto func = cast<func::FuncOp>(key.first);
    auto type = cast<FloatType>(key.second);
    FuncPlan plan = analyzeFunc(func, type);
    if (plan.ops.empty()) continue;
    // Every caller passes each converted binding as a static tensor argument.
    bool operandsOk = llvm::all_of(group, [&](IREE::Flow::DispatchOp d) {
      return llvm::all_of(plan.bindings, [&](unsigned argNo) {
        if (argNo >= d.getArguments().size()) return false;
        auto t = dyn_cast<RankedTensorType>(d.getArguments()[argNo].getType());
        return t && t.hasStaticShape();
      });
    });
    if (!operandsOk) continue;
    if (group.size() != callers[func].size()) {
      func = copyExecutableFor(group, module, callers);
      plan = analyzeFunc(func, type);
    }
    demoteFunc(rewriter, func, plan, type);
    for (IREE::Flow::DispatchOp dispatch : group) {
      for (unsigned argNo : plan.bindings) {
        OpOperand &use = dispatch.getArgumentsMutable()[argNo];
        Value root = use.get();
        while (auto reshapeOp =
                   root.getDefiningOp<IREE::Flow::TensorReshapeOp>())
          root = reshapeOp.getSource();
        Request &request = requests[{root, type}];
        request.type = type;
        request.uses.push_back(&use);
      }
    }
  }

  // 2 + 3. Decide per value and convert it.
  DenseMap<std::pair<Value, Type>, Operation *> constantCache;
  DenseMap<std::pair<Value, Type>, Value> castCache;
  SmallVector<IREE::Util::GlobalOp> replacedGlobals;
  for (auto &[key, request] : requests) {
    Value root = key.first;

    // a. Constant: convert at compile time.
    if (isa_and_present<DenseFPElementsAttr>(
            getConstantValue(root, /*global=*/nullptr))) {
      llvm::SmallSetVector<Value, 4> oldValues;
      for (OpOperand *use : request.uses) {
        rewriter.setInsertionPoint(use->getOwner());
        Value demoted =
            createDemotedConstant(rewriter, module, use->get(), request.type,
                                  counter, constantCache, replacedGlobals);
        assert(demoted && "every reshape of a constant is a constant");
        oldValues.insert(use->get());
        use->set(demoted);
      }
      eraseDeadReshapeChains(rewriter, oldValues.getArrayRef());
      continue;
    }

    // b. Result of a dispatch read only here: fuse into it.
    if (auto producer = root.getDefiningOp<IREE::Flow::DispatchOp>()) {
      if (fuseIntoProducer(rewriter, module, producer, cast<OpResult>(root),
                           request, callers))
        continue;
    }

    // c. Otherwise a host cast dispatch per operand value, after its
    // definition, shared by the consumers reading that value.
    for (OpOperand *use : request.uses) {
      Value v = use->get();
      Value &cast = castCache[{v, request.type}];
      if (!cast) {
        if (Operation *def = v.getDefiningOp())
          rewriter.setInsertionPointAfter(def);
        else
          rewriter.setInsertionPointToStart(use->getOwner()->getBlock());
        cast = createCastDispatch(rewriter, module, v, request.type,
                                  hostAffinity, counter);
      }
      use->set(cast);
    }
  }
  eraseDeadGlobals(rewriter, module, replacedGlobals);
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDemoteContractionDispatchInputsPass() {
  return std::make_unique<AMDAIEDemoteContractionDispatchInputsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
