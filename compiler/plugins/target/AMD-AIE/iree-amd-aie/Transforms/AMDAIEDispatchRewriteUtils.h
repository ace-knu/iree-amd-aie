// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers shared by the Flow-phase passes that rewrite NPU dispatches at the
// dispatch boundary (AMDAIEPadContractionDispatches,
// AMDAIEPadPoolingDispatches, AMDAIEFoldBiasIntoContractionDispatches):
// resolving a dispatch's target and executable, matching full-tensor binding
// loads, building host pad dispatches, reading compile-time constant operands,
// and cleaning up what a rewrite leaves dead.

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEDISPATCHREWRITEUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEDISPATCHREWRITEUTILS_H_

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::AMDAIE {

/// Resolves the amd-aie executable target a dispatch is pinned to via its
/// `stream.affinity` (affinity -> device global -> executable target attr).
/// Returns null if the dispatch is not pinned to an amd-aie device.
IREE::HAL::ExecutableTargetAttr getDispatchTarget(
    IREE::Flow::DispatchOp dispatch, ModuleOp module);

/// Returns the func of a dispatch's single entry point, or null.
func::FuncOp getDispatchFunc(IREE::Flow::DispatchOp dispatch, ModuleOp module);

/// Traces a matmul input value to the executable binding (BlockArgument) it is
/// loaded from via a full-tensor `dispatch.tensor.load`.
IREE::TensorExt::DispatchTensorLoadOp getFullTensorLoad(Value operand);

/// Grows executable binding `arg` and its full-tensor `load` to `newShape`
/// (the matmul revalidates from the loaded operand types).
void growBinding(IRRewriter &rewriter, BlockArgument arg,
                 IREE::TensorExt::DispatchTensorLoadOp load,
                 ArrayRef<int64_t> newShape);

/// For a `linalg.matmul` operand `map`, returns the operand-dim position that
/// the given contraction loop dim (M=0, N=1, K=2 -- fixed by linalg.matmul's
/// iterator order; transpose variants only permute the operand maps, not the
/// loop order) occupies. Lets us read each operand's M/N/K extents from its own
/// layout, so a transpose_b matmul (RHS = [N, K], as an ONNX Gemm lowers to) is
/// handled, not just the plain RHS = [K, N] layout.
unsigned operandDimForLoop(AffineMap map, unsigned loopDim);

/// Returns a device affinity pinned to a host (non-amd-aie) device global, or
/// null if none is declared. The padding dispatch runs here (the CPU), where
/// the strided insert is trivially codegen-able.
IREE::HAL::DeviceAffinityAttr getHostAffinity(ModuleOp module);

/// First index `i` such that no `pad_executable_i` / `crop_executable_i` /
/// `concat_executable_i` / `__padded_constant_i` / `__bias_folded_weight_i`
/// symbol exists yet in `module`: the helper builders number their executables
/// and globals from it, so the Flow-phase dispatch passes (which run back to
/// back on the same module) never collide.
int firstFreeHelperIndex(ModuleOp module);

/// Creates a `flow.executable` + `flow.dispatch` that zero-pads `v` up to
/// `dstShape` (high padding), placed on `hostAffinity`. Returns the padded
/// value, or `v` unchanged if `dstShape` already matches. The padded tensor is
/// a `flow.tensor.splat` of the pad value that the dispatch, tied to it, copies
/// `v` into, the form IREE itself gives a `tensor.pad` (a dispatch, unlike
/// `flow.tensor.update`, which only copies contiguous outer-dim sub-ranges,
/// performs the strided placement required to pad an inner dimension). A
/// `tensor.pad` inside the dispatch fails CPU codegen when an inner dim grows
/// by one (`scf.if` result type mismatch). `counter` uniquifies the symbol
/// names.
/// `padValue`, when set, is the constant to pad with instead of 0 (the pooling
/// pass passes its dispatch's init fill value, -inf for max).
Value createPaddingDispatch(IRRewriter &rewriter, ModuleOp module, Value v,
                            ArrayRef<int64_t> dstShape,
                            IREE::HAL::DeviceAffinityAttr hostAffinity,
                            int &counter, TypedAttr padValue = nullptr);

/// Returns the dense value of `v` when it is a compile-time constant: an
/// `arith.constant` or a load of an immutable `util.global` with a dense
/// initial value (inline or `dense_resource`), possibly through
/// `flow.tensor.reshape`s (row-major, so the value is only reshaped). Sets
/// `global` to that global if there is one. A weight read from an external
/// parameter archive is not a compile-time constant here.
DenseElementsAttr getConstantValue(Value v, IREE::Util::GlobalOp *global);

/// Creates a private immutable global `name` holding `value`, which replaces
/// the constant read from `source` (if any): the new global takes its default
/// affinity, and `source` is added to `replacedGlobals` for
/// `eraseDeadGlobals`. Returns an immutable load of the new global.
Value createReplacementConstant(
    IRRewriter &rewriter, ModuleOp module, Location loc, StringRef name,
    DenseElementsAttr value, IREE::Util::GlobalOp source,
    SmallVectorImpl<IREE::Util::GlobalOp> &replacedGlobals);

/// Erases `v`'s producer chain of reshapes, global loads and constants as far
/// as nothing else reads it.
void eraseDeadProducers(IRRewriter &rewriter, Value v);

/// Erases those of `globals` no load reads any more (they would otherwise be
/// kept next to their padded or folded copies until a later cleanup).
void eraseDeadGlobals(IRRewriter &rewriter, ModuleOp module,
                      ArrayRef<IREE::Util::GlobalOp> globals);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_AMDAIEDISPATCHREWRITEUTILS_H_
