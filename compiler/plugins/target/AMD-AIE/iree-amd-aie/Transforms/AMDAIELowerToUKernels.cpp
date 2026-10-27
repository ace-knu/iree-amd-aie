// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Path.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-lower-to-ukernels"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIELowerToUKernelsPass
    : public impl::AMDAIELowerToUKernelsBase<AMDAIELowerToUKernelsPass> {
 public:
  AMDAIELowerToUKernelsPass() = default;
  AMDAIELowerToUKernelsPass(const AMDAIELowerToUKernelsPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() override;
};

namespace {

/// ============================= BEGIN ==================================
/// ================== SAME UTILITIES AS IREE LLVMCPU ====================
/// ======================================================================

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
};

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`. `instanceArgs` are
/// the template arguments the ukernel source needs to define this function;
/// they are recorded with the name in a `ukernel_instance` attribute, which the
/// ukernel compilation turns into the instance list of the source file.
static FnNameAndDefAttrs getFnNameAndDefAttrs(
    RewriterBase &rewriter, std::string ukernelName,
    std::string inputOutputElemType, std::string ukernelObjectFile,
    ArrayRef<std::string> instanceArgs) {
  FnNameAndDefAttrs result;
  std::string ukernelSuffix = "";
  result.name = ukernelName + ukernelSuffix + "_" + inputOutputElemType;
  result.defAttrs.emplace_back(rewriter.getStringAttr("link_with"),
                               rewriter.getStringAttr(ukernelObjectFile));
  SmallVector<Attribute> instance{rewriter.getStringAttr(result.name)};
  for (const std::string &arg : instanceArgs)
    instance.push_back(rewriter.getStringAttr(arg));
  result.defAttrs.emplace_back(rewriter.getStringAttr("ukernel_instance"),
                               rewriter.getArrayAttr(instance));
  return result;
}

/// ============================= END ====================================
/// ================== SAME UTILITIES AS IREE LLVMCPU ====================
/// ======================================================================

/// Utility to fetch the element type as string.
static std::string typeToString(Type type) {
  std::string typeStr;
  llvm::raw_string_ostream rso(typeStr);
  type.print(rso);
  return typeStr;
}

/// We need to fetch the tiling at M, N and K for the input tensors along with
/// the intrinsics that the ukernel supports. The following utility helps us
/// fetch the same.
static std::tuple<int, int, int, int> getTilingInfo(ShapedType shapedType) {
  SmallVector<int64_t> shapeVec(shapedType.getShape());
  int index = 0;
  if (shapeVec.size() == 6) {
    index = 2;
  } else {
    assert(shapeVec.size() == 4 &&
           "lhs/rhs/out shape should have rank either 4 or 6");
  }
  int M = shapeVec[index + 1] * shapeVec[index + 2];
  int N = shapeVec[index] * shapeVec[index + 3];
  return {M, N, shapeVec[index + 2], shapeVec[index + 3]};
}

/// Matches a linalg.generic operation which is basically a tiled matmul and
/// converts it into a iree_codegen.ukernel."iree_amdaie_uk_matmul" operation,
/// that is later lowered into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface> matchMatmulDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return rewriter.notifyMatchFailure(op, "is not a linalg operation");
  }
  bool transposeB = isMatmulTransposeB(linalgOp);
  if (!isMatmul(linalgOp) && !transposeB) {
    return rewriter.notifyMatchFailure(op, "is not a matmul-like operation");
  }

  Value lhs = linalgOp.getDpsInputOperand(0)->get();
  Value rhs = linalgOp.getDpsInputOperand(1)->get();
  Value out = linalgOp.getDpsInitOperand(0)->get();
  auto lhsType = llvm::cast<ShapedType>(lhs.getType());
  auto rhsType = llvm::cast<ShapedType>(rhs.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outElemType = outType.getElementType();
  // The matmul ukernels take packed tiles. An unpacked 2-D tile (the GEMV
  // pipeline's) is left to the non-ukernel path.
  if (!llvm::all_of(TypeRange{lhsType, rhsType, outType}, [](Type type) {
        int64_t rank = cast<ShapedType>(type).getRank();
        return rank == 4 || rank == 6;
      })) {
    return rewriter.notifyMatchFailure(op, "operands are not packed tiles");
  }

  // Tiling for M x K x N as well as the corresponding inner tiling intrinsics
  // r x s x t.
  int M, N, K, r, s, t;
  std::tie(M, K, r, s) = getTilingInfo(lhsType);
  std::tie(std::ignore, N, std::ignore, t) = getTilingInfo(outType);
  // The matmul ukernels process 2x2 vector instructions per step and would
  // read and write past the tile for an odd count, so leave such a tile to the
  // non-ukernel path.
  if ((M / r) % 2 != 0 || (N / t) % 2 != 0) {
    return rewriter.notifyMatchFailure(
        op, "tile needs an even number of vector instructions in M and N");
  }
  // A transposed B is read as [n][k] blocks, which each template supports on
  // its own. Only the npu4 templates do so far.
  // TODO: Other devices tile a transposed B matmul for the ukernel
  // (`KernelDispatch`) although it is not lowered to one here.
  std::optional<AMDAIEDevice> device =
      getConfigAMDAIEDevice(IREE::HAL::ExecutableTargetAttr::lookup(op));
  if (!device || (transposeB && !isAie2P(*device)) ||
      !hasUKernelTemplate(*device, ukernelName,
                          {lhsElemType, rhsElemType, outElemType},
                          transposeB)) {
    return rewriter.notifyMatchFailure(
        op, "no ukernel template for the element types");
  }
  std::string inputOutputElemTypeAndSize =
      typeToString(lhsElemType) + "_" + typeToString(rhsElemType) + "_" +
      typeToString(outElemType) + "_" + std::to_string(M) + "x" +
      std::to_string(N) + "x" + std::to_string(K) + "_" + std::to_string(r) +
      "x" + std::to_string(s) + "x" + std::to_string(t);

  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, transposeB ? ukernelName + "_tb" : ukernelName,
      inputOutputElemTypeAndSize, ukernelObjectName,
      {typeToString(lhsElemType), typeToString(rhsElemType),
       typeToString(outElemType), std::to_string(M), std::to_string(N),
       std::to_string(K), std::to_string(r), std::to_string(s),
       std::to_string(t), transposeB ? "true" : "false"});

  // Create UKernel for AMD-AIE.
  Location loc = linalgOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{lhs, rhs}, out, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

/// Matches the GEMV pipeline's tile, a `(1 x K) x (N x K)^T -> (1 x N)`
/// linalg.matmul on unpacked 2-D operands, and converts it into a call to the
/// matvec ukernel, which adds the products to the output.
static FailureOr<IREE::Codegen::UKernelOpInterface> matchMatvecDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
  if (!matmulOp) return rewriter.notifyMatchFailure(op, "is not a matmul");
  MLIRContext *context = op->getContext();
  AffineExpr m, n, k;
  bindDims(context, m, n, k);
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  if (matmulOp.getIndexingMapsArray() !=
      AffineMap::inferFromExprList(MapList{{m, k}, {n, k}, {m, n}}, context)) {
    return rewriter.notifyMatchFailure(op, "is not a transposed B matmul");
  }
  Value lhs = matmulOp.getDpsInputOperand(0)->get();
  Value rhs = matmulOp.getDpsInputOperand(1)->get();
  Value out = matmulOp.getDpsInitOperand(0)->get();
  auto lhsType = cast<ShapedType>(lhs.getType());
  auto rhsType = cast<ShapedType>(rhs.getType());
  auto outType = cast<ShapedType>(out.getType());
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      lhsType.getDimSize(0) != 1) {
    return rewriter.notifyMatchFailure(op, "is not a static M == 1 tile");
  }
  int64_t N = rhsType.getDimSize(0);
  int64_t K = rhsType.getDimSize(1);
  // The ukernel reads K in 64-lane vectors.
  if (K % 64 != 0) {
    return rewriter.notifyMatchFailure(op, "K is not a multiple of 64");
  }
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outElemType = outType.getElementType();
  std::optional<AMDAIEDevice> device =
      getConfigAMDAIEDevice(IREE::HAL::ExecutableTargetAttr::lookup(op));
  if (!device || !isAie2P(*device) ||
      !hasUKernelTemplate(*device, ukernelName,
                          {lhsElemType, rhsElemType, outElemType},
                          /*transposeB=*/true)) {
    return rewriter.notifyMatchFailure(
        op, "no ukernel template for the element types");
  }

  std::string elemTypeAndSize =
      typeToString(lhsElemType) + "_" + typeToString(rhsElemType) + "_" +
      typeToString(outElemType) + "_1x" + std::to_string(N) + "x" +
      std::to_string(K);
  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, ukernelName, elemTypeAndSize, ukernelObjectName,
      {typeToString(lhsElemType), typeToString(rhsElemType),
       typeToString(outElemType), std::to_string(N), std::to_string(K)});

  // Create UKernel for AMD-AIE.
  Location loc = matmulOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{lhs, rhs}, out, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchFillDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  auto fillOp = dyn_cast<linalg::FillOp>(op);
  if (!fillOp) {
    return rewriter.notifyMatchFailure(op, "is not a fill operation");
  }

  Value input = fillOp.getDpsInputOperand(0)->get();
  if (!(matchPattern(input, m_Zero()) ||
        matchPattern(input, m_AnyZeroFloat()))) {
    return rewriter.notifyMatchFailure(op, "not a zero fill operation");
  }
  Value output = fillOp.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(output.getType());
  Type outElemType = outType.getElementType();
  std::optional<AMDAIEDevice> device =
      getConfigAMDAIEDevice(IREE::HAL::ExecutableTargetAttr::lookup(op));
  if (!device || !hasUKernelTemplate(*device, ukernelName, {outElemType})) {
    return rewriter.notifyMatchFailure(
        op, "no ukernel template for the element type");
  }

  int M, N, r, t;
  if (outType.getRank() == 2) {
    M = outType.getDimSize(0);
    N = outType.getDimSize(1);
  } else {
    // Tiling for M x N as well as the corresponding inner tiling intrinsics r x
    // t.
    std::tie(M, N, r, t) = getTilingInfo(outType);
  }
  // The zero_fill ukernels store whole 16-lane vectors.
  if ((M * N) % 16 != 0) {
    return rewriter.notifyMatchFailure(
        op, "fill is not a whole number of vectors");
  }

  std::string elemTypeAndSize = typeToString(outElemType) + "_" +
                                std::to_string(M) + "x" + std::to_string(N);

  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, ukernelName, elemTypeAndSize, ukernelObjectName,
      {typeToString(outElemType), std::to_string(M), std::to_string(N)});

  // Create UKernel for AMD-AIE.
  Location loc = fillOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{}, output, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

std::optional<Value> checkIsShiftTruncIAndReturnShift(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (!isa<linalg::GenericOp>(linalgOp)) return std::nullopt;
  Block *body = linalgOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  auto truncOp = dyn_cast_if_present<arith::TruncIOp>(yieldVal.getDefiningOp());
  if (!truncOp) return std::nullopt;
  auto shiftOp =
      dyn_cast_if_present<arith::ShRSIOp>(truncOp.getIn().getDefiningOp());
  // If no shift op, the shift value is zero.
  if (!shiftOp) {
    BlockArgument blockArg =
        getBlockArgumentWithOptionalExtOps(truncOp->getOperand(0));
    if (!blockArg || blockArg.getOwner() != body) return std::nullopt;
    rewriter.setInsertionPoint(linalgOp);
    IntegerAttr zeroAttr =
        rewriter.getIntegerAttr(truncOp.getIn().getType(), 0);
    auto zeroVal =
        rewriter.create<arith::ConstantOp>(linalgOp.getLoc(), zeroAttr);
    return zeroVal.getResult();
  }
  Value shiftVal = shiftOp.getRhs();
  BlockArgument blockArg =
      getBlockArgumentWithOptionalExtOps(shiftOp->getOperand(0));
  if (!blockArg || blockArg.getOwner() != body) return std::nullopt;
  return shiftVal;
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchTruncIDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return rewriter.notifyMatchFailure(op, "is not a linalg operation");
  }
  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1) {
    return rewriter.notifyMatchFailure(op, "is not a single input/output op");
  }
  Value input = linalgOp.getDpsInputOperand(0)->get();
  Value output = linalgOp.getDpsInitOperand(0)->get();
  auto inType = llvm::cast<ShapedType>(input.getType());
  auto outType = llvm::cast<ShapedType>(output.getType());
  Type inElemType = inType.getElementType();
  Type outElemType = outType.getElementType();
  // Checked before `checkIsShiftTruncIAndReturnShift`, which may create IR.
  std::optional<AMDAIEDevice> device =
      getConfigAMDAIEDevice(IREE::HAL::ExecutableTargetAttr::lookup(op));
  if (!device ||
      !hasUKernelTemplate(*device, ukernelName, {inElemType, outElemType})) {
    return rewriter.notifyMatchFailure(
        op, "no ukernel template for the element types");
  }

  std::optional<Value> maybeShift =
      checkIsShiftTruncIAndReturnShift(rewriter, linalgOp);
  if (!maybeShift.has_value()) {
    return rewriter.notifyMatchFailure(op,
                                       "is not a shift followed by a trunci");
  }
  Value shiftVal = maybeShift.value();

  // Tiling for M x N as well as the corresponding inner tiling intrinsics r x
  // t.
  int M, N, r, t;
  std::tie(M, N, r, t) = getTilingInfo(outType);
  std::string elemTypeAndSize = typeToString(inElemType) + "_" +
                                typeToString(outElemType) + "_" +
                                std::to_string(M) + "x" + std::to_string(N);

  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, ukernelName, elemTypeAndSize, ukernelObjectName,
      {typeToString(inElemType), typeToString(outElemType), std::to_string(M),
       std::to_string(N)});

  // Create UKernel for AMD-AIE.
  Location loc = linalgOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{input, shiftVal}, output, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchSoftmaxDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  auto softmaxOp = dyn_cast<linalg::SoftmaxOp>(op);
  if (!softmaxOp)
    return rewriter.notifyMatchFailure(op, "is not a softmax operation");

  Value input = softmaxOp.getDpsInputOperand(0)->get();
  Value output = softmaxOp.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(output.getType());
  Type outElemType = outType.getElementType();

  std::string elemTypeAndSize = typeToString(outElemType) + "_" +
                                std::to_string(outType.getDimSize(0)) + "x" +
                                std::to_string(outType.getDimSize(1));

  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, ukernelName, elemTypeAndSize, ukernelObjectName,
      {typeToString(outElemType), std::to_string(outType.getDimSize(0)),
       std::to_string(outType.getDimSize(1))});

  // Create UKernel for AMD-AIE.
  Location loc = softmaxOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{input}, output, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

using TargetPredicate = std::function<bool(IREE::HAL::ExecutableTargetAttr)>;
using MatchAndReplaceFunction =
    std::function<FailureOr<IREE::Codegen::UKernelOpInterface>(
        RewriterBase &, Operation *, const std::string &, const std::string &)>;

template <typename OpType>
struct LowerToUKernelPattern : OpRewritePattern<OpType> {
  LowerToUKernelPattern(MLIRContext *context, TargetPredicate targetPredicate,
                        MatchAndReplaceFunction matchAndReplace,
                        const std::string &ukernelName)
      : OpRewritePattern<OpType>(context),
        targetPredicate(targetPredicate),
        matchAndReplace(matchAndReplace),
        ukernelName(ukernelName) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (targetPredicate &&
        !targetPredicate(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
      return rewriter.notifyMatchFailure(
          op, "the target attribute fails the predicate");
    }
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
    if (!targetAttr ||
        !hasUkernel(targetAttr.getConfiguration(), ukernelName)) {
      return rewriter.notifyMatchFailure(
          op, "no ukernel found with name: " + ukernelName);
    }

    std::string ukernelObjectName = ukernelName + ".o";

    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchAndReplace(rewriter, op, ukernelName, ukernelObjectName);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(op, ukernelOp.value()->getResults());
    return success();
  }

  TargetPredicate targetPredicate;
  MatchAndReplaceFunction matchAndReplace;
  std::string ukernelName;
};

}  // namespace

static constexpr char kMatmulUKernelName[] = "matmul";
static constexpr char kMatvecUKernelName[] = "matvec";
static constexpr char kFillUKernelName[] = "zero_fill";
static constexpr char kTruncIUKernelName[] = "trunci";
static constexpr char kSoftmaxUKernelName[] = "softmax";

void AMDAIELowerToUKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // Enabling a lowering of an op to a microkernel is a trade-off between the
  // potential performance advantage of a microkernel over pure code generation
  // for that op, and the potential benefits of fusions. Indeed, once an op
  // lowered into a microkernel, it will never be fused at any MLIR level.
  // Since microkernels are linked as bitcode, they will still undergo LTO-like
  // optimization in their calling contexts, but we shouldn't expect this to
  // achieve similar results as fusing structured ops.

  // These patterns are unconditionally enabled, because we have strong evidence
  // that it is difficult for codegen to consistently approach microkernels
  // performance, and that consideration overrides the benefit of fusions for
  // these ops.
  auto allTargets = [](auto target) { return true; };
  patterns.insert<LowerToUKernelPattern<linalg::GenericOp>>(
      context, allTargets, matchMatmulDAGForUKernel, kMatmulUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::MatmulOp>>(
      context, allTargets, matchMatmulDAGForUKernel, kMatmulUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::MatmulOp>>(
      context, allTargets, matchMatvecDAGForUKernel, kMatvecUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::FillOp>>(
      context, allTargets, matchFillDAGForUKernel, kFillUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::GenericOp>>(
      context, allTargets, matchTruncIDAGForUKernel, kTruncIUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::SoftmaxOp>>(
      context, allTargets, matchSoftmaxDAGForUKernel, kSoftmaxUKernelName);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELowerToUKernelsPass() {
  return std::make_unique<AMDAIELowerToUKernelsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
