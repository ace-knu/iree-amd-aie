// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/AMDAIEDispatchRewriteUtils.h"

#include "iree-amd-aie/Transforms/Utils/AMDAIEDevicePlacementUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectResourceBlobManager.h"

namespace mlir::iree_compiler::AMDAIE {

IREE::HAL::ExecutableTargetAttr getDispatchTarget(
    IREE::Flow::DispatchOp dispatch, ModuleOp module) {
  auto affinity =
      dispatch->getAttrOfType<IREE::HAL::DeviceAffinityAttr>("stream.affinity");
  if (!affinity) return {};
  Operation *globalOp =
      SymbolTable::lookupNearestSymbolFrom(module, affinity.getDevice());
  auto global = dyn_cast_or_null<IREE::Util::GlobalOpInterface>(globalOp);
  if (!global) return {};
  auto deviceTarget = dyn_cast_or_null<IREE::HAL::DeviceTargetAttr>(
      global.getGlobalInitialValue());
  if (!deviceTarget) return {};
  ArrayRef<IREE::HAL::ExecutableTargetAttr> targets =
      deviceTarget.getExecutableTargets();
  if (targets.empty()) return {};
  IREE::HAL::ExecutableTargetAttr exec = targets.front();
  if (exec.getBackend().getValue() != "amd-aie") return {};
  return exec;
}

func::FuncOp getDispatchFunc(IREE::Flow::DispatchOp dispatch, ModuleOp module) {
  SmallVector<SymbolRefAttr> entryPoints =
      llvm::to_vector(dispatch.getEntryPointRefs());
  if (entryPoints.size() != 1) return nullptr;
  SymbolRefAttr entryPoint = entryPoints.front();
  auto exe = module.lookupSymbol<IREE::Flow::ExecutableOp>(
      entryPoint.getRootReference());
  if (!exe) return nullptr;
  ModuleOp inner = exe.getInnerModule();
  if (!inner) return nullptr;
  return inner.lookupSymbol<func::FuncOp>(entryPoint.getLeafReference());
}

IREE::TensorExt::DispatchTensorLoadOp getFullTensorLoad(Value operand) {
  auto load = operand.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
  if (!load) return nullptr;
  // Only a plain full-tensor load (unit strides, zero offsets) is supported.
  for (OpFoldResult o : load.getMixedOffsets())
    if (!isConstantIntValue(o, 0)) return nullptr;
  for (OpFoldResult s : load.getMixedStrides())
    if (!isConstantIntValue(s, 1)) return nullptr;
  return load;
}

void growBinding(IRRewriter &rewriter, BlockArgument arg,
                 IREE::TensorExt::DispatchTensorLoadOp load,
                 ArrayRef<int64_t> newShape) {
  auto dtt = cast<IREE::TensorExt::DispatchTensorType>(arg.getType());
  auto tensorType = dtt.asRankedTensorType();
  auto newTensorType =
      RankedTensorType::get(newShape, tensorType.getElementType());
  arg.setType(
      IREE::TensorExt::DispatchTensorType::get(dtt.getAccess(), newTensorType));
  rewriter.setInsertionPoint(load);
  auto newLoad = rewriter.create<IREE::TensorExt::DispatchTensorLoadOp>(
      load.getLoc(), newTensorType, arg, /*sourceDynamicDims=*/ValueRange{});
  rewriter.replaceOp(load, newLoad.getResult());
}

unsigned operandDimForLoop(AffineMap map, unsigned loopDim) {
  for (unsigned i = 0, e = map.getNumResults(); i < e; ++i)
    if (cast<AffineDimExpr>(map.getResult(i)).getPosition() == loopDim)
      return i;
  llvm_unreachable("matmul operand map missing expected loop dim");
}

IREE::HAL::DeviceAffinityAttr getHostAffinity(ModuleOp module) {
  // Ask the shared capability table rather than testing for "not amd-aie":
  // with a third backend declared, the first non-amd-aie device need not be a
  // host at all (an accelerator would be picked, and the padding dispatch would
  // land on a device that cannot codegen it).
  //
  // The table currently recognizes only llvm-cpu as a host, so a host declared
  // with any other backend yields null here and the callers return having done
  // nothing. That is silent: the unpadded dispatch later fails divisibility.
  // The right shape is to move the check after
  // the dispatch walk, so "nothing to do" and "something to do but nowhere to
  // put it" can be told apart and only the latter is reported. Not done here
  // because in this build llvm-cpu is the only host backend that can be
  // declared at all (configure.sh builds no other), so the case is currently
  // unreachable rather than merely unlikely.
  return getFirstDeviceWithRole(module, DeviceRole::Host);
}

int firstFreeHelperIndex(ModuleOp module) {
  int next = 0;
  for (Operation &op : module.getBody()->getOperations()) {
    auto sym = dyn_cast<SymbolOpInterface>(&op);
    if (!sym) continue;
    StringRef name = sym.getName();
    for (StringRef prefix :
         {"pad_executable_", "crop_executable_", "concat_executable_",
          "__padded_constant_", "__bias_folded_weight_"}) {
      if (!name.consume_front(prefix)) continue;
      int idx;
      if (!name.getAsInteger(10, idx)) next = std::max(next, idx + 1);
    }
  }
  return next;
}

Value createPaddingDispatch(IRRewriter &rewriter, ModuleOp module, Value v,
                            ArrayRef<int64_t> dstShape,
                            IREE::HAL::DeviceAffinityAttr hostAffinity,
                            int &counter, TypedAttr padValue) {
  Location loc = v.getLoc();
  auto srcType = cast<RankedTensorType>(v.getType());
  if (dstShape == srcType.getShape()) return v;
  auto dstType = RankedTensorType::get(dstShape, srcType.getElementType());
  auto inBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadOnly, srcType);
  auto outBinding = IREE::TensorExt::DispatchTensorType::get(
      IREE::TensorExt::TensorAccess::ReadWrite, dstType);
  std::string idx = std::to_string(counter++);

  // Worker func: load the full input and store it into the low corner of the
  // pad-value-filled output.
  auto funcOp = func::FuncOp::create(
      loc, "pad_dispatch_" + idx,
      rewriter.getFunctionType({inBinding, outBinding}, {}));
  funcOp.setPublic();
  Block *entry = funcOp.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  Value loaded = fb.create<IREE::TensorExt::DispatchTensorLoadOp>(
      loc, srcType, entry->getArgument(0), /*sourceDynamicDims=*/ValueRange{});
  SmallVector<OpFoldResult> offsets(dstShape.size(), fb.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(dstShape.size(), fb.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes;
  for (int64_t d : srcType.getShape()) sizes.push_back(fb.getIndexAttr(d));
  fb.create<IREE::TensorExt::DispatchTensorStoreOp>(
      loc, loaded, entry->getArgument(1), /*targetDynamicDims=*/ValueRange{},
      offsets, sizes, strides);
  fb.create<func::ReturnOp>(loc);

  // Executable wrapping the func, with a default workgroup-count region.
  OpBuilder mb(&module.getBody()->back());
  auto exeOp =
      IREE::Flow::ExecutableOp::create(mb, loc, "pad_executable_" + idx);
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

  // Host splat of the pad value and the dispatch writing into it in place,
  // both pinned to the host device.
  Type elemType = srcType.getElementType();
  TypedAttr fillAttr = padValue ? padValue : rewriter.getZeroAttr(elemType);
  Value padScalar = rewriter.create<arith::ConstantOp>(loc, fillAttr);
  auto splatOp = rewriter.create<IREE::Flow::TensorSplatOp>(
      loc, dstType, padScalar, /*result_dims=*/ValueRange{});
  splatOp->setAttr("stream.affinity", hostAffinity);
  auto dispatchOp = IREE::Flow::DispatchOp::create(
      rewriter, loc, exportOp, /*workload=*/ValueRange{}, TypeRange{dstType},
      /*result_dims=*/ValueRange{},
      /*arguments=*/ValueRange{v, splatOp.getResult()},
      /*argument_dims=*/ValueRange{},
      /*tied_operands=*/rewriter.getIndexArrayAttr({1}));
  dispatchOp->setAttr("stream.affinity", hostAffinity);
  return dispatchOp.getResult(0);
}

/// Returns `attr` as a DenseElementsAttr: as is, or the blob of a
/// `dense_resource` (as an ONNX import stores large weights). Null for any
/// other attr or a resource without its data (elided).
static DenseElementsAttr toDenseElements(Attribute attr) {
  if (auto dense = dyn_cast_if_present<DenseElementsAttr>(attr)) return dense;
  auto resource = dyn_cast_if_present<DenseResourceElementsAttr>(attr);
  if (!resource) return {};
  AsmResourceBlob *blob = resource.getRawHandle().getBlob();
  if (!blob) return {};
  ArrayRef<char> data = blob->getData();
  if (!DenseElementsAttr::isValidRawBuffer(resource.getType(), data)) return {};
  return DenseElementsAttr::getFromRawBuffer(resource.getType(), data);
}

DenseElementsAttr getConstantValue(Value v, IREE::Util::GlobalOp *global) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || !type.hasStaticShape()) return {};
  Value root = v;
  while (auto reshapeOp = root.getDefiningOp<IREE::Flow::TensorReshapeOp>())
    root = reshapeOp.getSource();
  DenseElementsAttr attr;
  if (auto cstOp = root.getDefiningOp<arith::ConstantOp>()) {
    attr = toDenseElements(cstOp.getValue());
  } else if (auto loadOp = root.getDefiningOp<IREE::Util::GlobalLoadOp>()) {
    if (!loadOp.getIsImmutable()) return {};
    auto globalOp = SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOp>(
        loadOp, loadOp.getGlobalAttr());
    if (!globalOp) return {};
    if (global) *global = globalOp;
    attr = toDenseElements(globalOp.getInitialValueAttr());
  }
  return attr ? attr.reshape(type) : DenseElementsAttr();
}

Value createReplacementConstant(
    IRRewriter &rewriter, ModuleOp module, Location loc, StringRef name,
    DenseElementsAttr value, IREE::Util::GlobalOp source,
    SmallVectorImpl<IREE::Util::GlobalOp> &replacedGlobals) {
  OpBuilder moduleBuilder = OpBuilder::atBlockBegin(module.getBody());
  auto global = IREE::Util::GlobalOp::create(
      moduleBuilder, loc, name, /*isMutable=*/false, value.getType(),
      std::optional<TypedAttr>(value));
  global.setPrivate();
  if (source) {
    if (Attribute affinity = source->getAttr("stream.affinity.default"))
      global->setAttr("stream.affinity.default", affinity);
    replacedGlobals.push_back(source);
  }
  auto load = cast<IREE::Util::GlobalLoadOp>(
      global.createLoadOp(loc, rewriter).getOperation());
  load.setIsImmutable(true);
  return load.getResult();
}

void eraseDeadProducers(IRRewriter &rewriter, Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!op->use_empty() ||
        !isa<IREE::Flow::TensorReshapeOp, IREE::Util::GlobalLoadOp,
             arith::ConstantOp>(op))
      return;
    auto reshapeOp = dyn_cast<IREE::Flow::TensorReshapeOp>(op);
    v = reshapeOp ? reshapeOp.getSource() : Value();
    rewriter.eraseOp(op);
    if (!v) return;
  }
}

void eraseDeadGlobals(IRRewriter &rewriter, ModuleOp module,
                      ArrayRef<IREE::Util::GlobalOp> globals) {
  for (IREE::Util::GlobalOp global : globals) {
    SmallVector<IREE::Util::GlobalLoadOp> loads;
    bool otherUses = false;
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(global, module);
    if (!uses) continue;
    for (SymbolTable::SymbolUse use : *uses) {
      auto load = dyn_cast<IREE::Util::GlobalLoadOp>(use.getUser());
      if (!load || !load.use_empty())
        otherUses = true;
      else
        loads.push_back(load);
    }
    if (otherUses) continue;
    for (IREE::Util::GlobalLoadOp load : loads) rewriter.eraseOp(load);
    rewriter.eraseOp(global);
  }
}

}  // namespace mlir::iree_compiler::AMDAIE
