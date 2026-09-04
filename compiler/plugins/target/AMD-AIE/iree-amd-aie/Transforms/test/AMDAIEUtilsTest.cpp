#include "gtest/gtest.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

TEST(FindLargestFactorTest, Test0) {
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 1), 1);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 2), 2);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 3), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 4), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 5), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 6), 6);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 7), 6);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 8), 6);

  EXPECT_EQ(detail::findLargestFactor(/* num = */ 5, /* max = */ 6), 5);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 6), 6);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 7, /* max = */ 6), 1);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 8, /* max = */ 6), 4);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 9, /* max = */ 6), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 10, /* max = */ 6), 5);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 11, /* max = */ 6), 1);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 6), 6);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 13, /* max = */ 6), 1);

  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 5), 4);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 4), 4);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 3), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 2), 2);

  // Simple algo iterates 10'000 -> 7'500 in steps of 1 (2'500 iterations)
  // Current sqrt bounding algo finds and returns 7'500 in 1 iteration.
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 15'000, /* max = */ 10'000),
            7'500);

  // Simple algorithms runs ~10'000 iterations
  // Current sqrt bounding algo runs ~100 iterations
  int firstPrimeAbove1e5 = 10'037;
  EXPECT_EQ(
      detail::findLargestFactor(firstPrimeAbove1e5, firstPrimeAbove1e5 - 1), 1);
}

TEST(OpFoldResultPrinting, Test0) {
  mlir::MLIRContext context;
  llvm::SmallVector<mlir::OpFoldResult> opFoldResults = {};
  EXPECT_EQ(getConstantIntValuesString(opFoldResults), "[]");

  mlir::OpFoldResult three = getAsIndexOpFoldResult(&context, 3);
  opFoldResults.push_back(three);
  EXPECT_EQ(getConstantIntValuesString(opFoldResults), "[3]");

  mlir::OpFoldResult four = getAsIndexOpFoldResult(&context, 4);
  opFoldResults.push_back(four);
  EXPECT_EQ(getConstantIntValuesString(opFoldResults), "[3,4]");

  opFoldResults.push_back(mlir::Value{});
  EXPECT_EQ(getConstantIntValuesString(opFoldResults),
            "[not all constant integers]");
}

// Regression test for isMatmulInDefChain mis-recognizing a packed
// transpose_b matmul. After first-level packing, a source matmul (whether
// plain or transposed) always becomes a `linalg.generic`, so `isMatmul`'s
// Step 0 op-type short-circuit no longer distinguishes it -- only its Step 2
// index-map check does, and that check alone rejects a transpose_b matmul
// (its N dim comes from the RHS's *outer*, not inner, map result). Before the
// fix, `isMatmulInDefChain` asked only `isMatmul`, so it missed exactly this
// case and let a fused elementwise consumer of a transpose_b matmul get
// wrongly promoted to a second shared-memory buffer (see
// AMDAIEBufferizeToAllocation.cpp's `getInputOutputOperands`), which is what
// exhausted core-tile DMA channels for VGG16's FC layers.
TEST(IsMatmulInDefChainTest, RecognizesPackedTransposeB) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();

  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type f32 = builder.getF32Type();
  // A: [M=4, K=8] (unchanged LHS). B: [N=4, K=8] (RHS stored transposed, as
  // an ONNX Gemm(transB=1) lowers to). C: [M=4, N=4].
  auto tensorMK = mlir::RankedTensorType::get({4, 8}, f32);
  auto tensorNK = mlir::RankedTensorType::get({4, 8}, f32);
  auto tensorMN = mlir::RankedTensorType::get({4, 4}, f32);

  mlir::ModuleOp moduleOp = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto funcType =
      builder.getFunctionType({tensorMK, tensorNK, tensorMN}, {tensorMN});
  auto funcOp = mlir::func::FuncOp::create(builder, loc, "test_fn", funcType);
  mlir::Block *entry = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  mlir::Value lhs = entry->getArgument(0);
  mlir::Value rhs = entry->getArgument(1);
  mlir::Value acc = entry->getArgument(2);

  mlir::AffineExpr d0, d1, d2;
  mlir::bindDims(&context, d0, d1, d2);
  // A: (d0, d2) = (M, K), unchanged.
  auto mapA = mlir::AffineMap::get(3, 0, {d0, d2}, &context);
  // B: (d1, d2) = (N, K) -- transposed; a plain (non-transposed) RHS map
  // would be (d2, d1) = (K, N).
  auto mapB = mlir::AffineMap::get(3, 0, {d1, d2}, &context);
  // C: (d0, d1) = (M, N).
  auto mapC = mlir::AffineMap::get(3, 0, {d0, d1}, &context);
  llvm::SmallVector<mlir::utils::IteratorType> iterators = {
      mlir::utils::IteratorType::parallel, mlir::utils::IteratorType::parallel,
      mlir::utils::IteratorType::reduction};

  auto genericOp = mlir::linalg::GenericOp::create(
      builder, loc, mlir::TypeRange{tensorMN}, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{acc}, llvm::ArrayRef<mlir::AffineMap>{mapA, mapB, mapC},
      iterators,
      [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value mul =
            mlir::arith::MulFOp::create(b, loc, args[0], args[1]);
        mlir::Value add = mlir::arith::AddFOp::create(b, loc, args[2], mul);
        mlir::linalg::YieldOp::create(b, loc, add);
      });

  auto linalgOp =
      mlir::cast<mlir::linalg::LinalgOp>(genericOp.getOperation());
  // Sanity: confirm this op is exactly the mismatched case the fix targets --
  // a plain-matmul test rejects it, a transpose_b-matmul test accepts it.
  EXPECT_FALSE(isMatmul(linalgOp));
  EXPECT_TRUE(isMatmulTransposeB(linalgOp));
  EXPECT_TRUE(isMatmulInDefChain(genericOp.getResult(0)));

  moduleOp->erase();
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
