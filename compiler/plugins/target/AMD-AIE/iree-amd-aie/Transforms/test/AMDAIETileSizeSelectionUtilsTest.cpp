#include "gtest/gtest.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIETileSizeSelectionUtils.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

TEST(SelectTileSizeTest, L1TileSizeTest) {
  // The input params are {memoryLimit, numBytesA, numBytesB, numBytesC,
  // numBytesAcc, bufferDepthA, bufferDepthB, bufferDepthC, bufferDepthAcc,
  // inputM, inputN, inputK, vectorM, vectorN, vectorK}. The memory limit is
  // the 64 KB core memory less the default 1024-byte core stack.
  EXPECT_EQ((selectL1TileSizes(
                {64512, 4, 4, 4, 4, 2, 2, 2, 0, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 32, 64}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 2, 2, 4, 4, 2, 2, 2, 0, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 32, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 1, 1, 4, 4, 2, 2, 2, 0, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 32, 128}));

  // With elementwise op
  EXPECT_EQ((selectL1TileSizes(
                {64512, 4, 4, 1, 4, 2, 2, 2, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 32, 64}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 2, 2, 1, 4, 2, 2, 2, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 32, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 1, 1, 1, 4, 2, 2, 2, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 64, 128}));

  // All single buffer
  EXPECT_EQ((selectL1TileSizes(
                {64512, 4, 4, 4, 4, 1, 1, 1, 0, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 32, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 2, 2, 4, 4, 1, 1, 1, 0, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 64, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 1, 1, 4, 4, 1, 1, 1, 0, 512, 512, 512, 4, 4, 8})),
            (TileSize{128, 64, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 2, 2, 2, 2, 1, 1, 1, 0, 308, 2432, 9728, 4, 4, 8})),
            (TileSize{44, 128, 128}));

  // Smaller input shapes
  EXPECT_EQ(
      (selectL1TileSizes({64512, 4, 4, 4, 4, 2, 2, 2, 0, 32, 32, 32, 4, 4, 8})),
      (TileSize{32, 32, 32}));
  EXPECT_EQ(
      (selectL1TileSizes({64512, 2, 2, 1, 4, 2, 2, 2, 1, 32, 32, 32, 4, 4, 8})),
      (TileSize{32, 32, 32}));
  EXPECT_EQ(
      (selectL1TileSizes({64512, 1, 1, 1, 4, 2, 2, 2, 1, 64, 64, 64, 4, 4, 8})),
      (TileSize{64, 64, 64}));

  // Other sanity check
  EXPECT_EQ((selectL1TileSizes(
                {64512, 8, 8, 8, 8, 2, 2, 2, 0, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 16, 64}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 8, 8, 16, 16, 2, 2, 2, 0, 512, 512, 512, 4, 4, 8})),
            (TileSize{16, 16, 64}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 4, 4, 1, 4, 2, 2, 2, 2, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 32, 64}));

  // At the memory limit: 24x80x120 bf16 takes 65280 bytes, over the limit, so
  // N is halved; 16x88x128 bf16 takes exactly 64512 bytes, which fits.
  EXPECT_EQ((selectL1TileSizes(
                {64512, 2, 2, 4, 4, 2, 2, 2, 0, 24, 80, 120, 8, 8, 8})),
            (TileSize{24, 40, 120}));
  EXPECT_EQ((selectL1TileSizes(
                {64512, 2, 2, 4, 4, 2, 2, 2, 0, 16, 88, 128, 8, 8, 8})),
            (TileSize{16, 88, 128}));
}

TEST(SelectTileSizeTest, L2TileSizeTest) {
  // The input params are {memoryLimit, numBytesA, numBytesB, numBytesC,
  // numBytesAcc, bufferDepthA, bufferDepthB, bufferDepthC, bufferDepthAcc,
  // inputM, inputN, inputK, vectorM, vectorN, vectorK}, maxL1TileM, maxL1TileN.

  // (i32, i32) -> i32.
  // Both M/N inputs are much larger than L1 tile sizes.
  EXPECT_EQ(
      (selectL2TileSizes(
          {524288, 4, 4, 4, 4, 2, 2, 2, 0, 512, 512, 64, 4, 4, 8}, 32, 32)),
      (TileSize{256, 128, 64}));
  // Both M/N inputs are small and equal to L1 tile sizes.
  EXPECT_EQ((selectL2TileSizes(
                {524288, 4, 4, 4, 4, 2, 2, 2, 0, 32, 32, 64, 4, 4, 8}, 32, 32)),
            (TileSize{32, 32, 64}));
  // M input is large and N input is small.
  EXPECT_EQ(
      (selectL2TileSizes({524288, 4, 4, 4, 4, 2, 2, 2, 0, 512, 32, 32, 4, 4, 8},
                         32, 32)),
      (TileSize{512, 32, 32}));
  // M input is small and N input is large.
  EXPECT_EQ(
      (selectL2TileSizes(
          {524288, 4, 4, 4, 4, 2, 2, 2, 0, 32, 128, 128, 4, 4, 8}, 32, 32)),
      (TileSize{32, 128, 128}));
  // TileSize {128, 128, 128} fits the memory limit.
  EXPECT_EQ(
      (selectL2TileSizes(
          {524288, 4, 4, 4, 4, 2, 2, 2, 0, 256, 256, 128, 4, 4, 8}, 64, 64)),
      (TileSize{128, 128, 128}));
  // Although TileSize {128, 128, 128} fits the memory limit, M/N tile size
  // doesn't fully divide the input size, so return {64, 64, 128}.
  EXPECT_EQ(
      (selectL2TileSizes(
          {524288, 4, 4, 4, 4, 2, 2, 2, 0, 192, 192, 128, 4, 4, 8}, 64, 64)),
      (TileSize{64, 64, 128}));

  // (bf16, bf16) -> f32.
  EXPECT_EQ(
      (selectL2TileSizes(
          {524288, 2, 2, 4, 4, 2, 2, 2, 0, 512, 512, 64, 4, 4, 8}, 32, 32)),
      (TileSize{256, 128, 64}));
  // (i8, i8) -> i32.
  EXPECT_EQ(
      (selectL2TileSizes(
          {524288, 1, 1, 4, 4, 2, 2, 2, 0, 1024, 1024, 64, 4, 4, 8}, 32, 32)),
      (TileSize{256, 128, 64}));
  // (i8, i8) -> i32, large N input for matmul-elementwise fusion.
  EXPECT_EQ((selectL2TileSizes({524288 * 8, 1, 1, 1, 2, 2, 2, 2, 0, 1024,
                                4096 * 4, 512, 4, 4, 8},
                               64, 64)),
            (TileSize{1024, 1024, 512}));
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
