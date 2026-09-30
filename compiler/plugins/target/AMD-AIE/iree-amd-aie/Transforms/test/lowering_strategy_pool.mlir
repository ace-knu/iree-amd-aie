// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8})' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8 pool-core-split=search})' %s | FileCheck %s --check-prefix=SEARCH

// 2-D max pooling (channels-last, `d*2 + k` maps, maximumf body) takes the
// copy-based pool pipeline: two tile levels over the output dims only (block,
// core), the 2x2 window untiled, and the per-dispatch pipeline override in
// translation_info (see setRootConfigForPoolPipeline).
//
// Default `pool-core-split=spatial-channel`: array columns take OH (its odd
// factor, 7 for every VGG-16 layer), array rows take C (C / 4 channels each),
// so every layer runs on the same 7 x 4 = 28 cores and the core forall is 2-D
// (OH, C). `pool-core-split=search` (SEARCH) spreads cores along one output
// dim only, ranked by core count, then channel tile, then L1 bytes.

// VGG-16 pool1 (HWC): 224x224x64 -> 112x112x64.
// spatial-channel: cols 7 (odd factor of 112), rows 4 (C 64 -> 16 per row).
// The L1 tile is the largest-footprint [tOh, tOw, 16] with tOh | 16, tOw | 112:
// [4, 16, 16] f32 = 2 x (8x32x16 + 4x16x16) x 4 B = 40 KB (beats [1, 56, 16]
// at 35 KB), block [7*4, 16, 64] = [28, 16, 64]; OH 4 x OW 7 = 28 blocks in
// time. Level 0 keeps OH (block dim) and OW (16 != 112); level 1 keeps the two
// core dims OH and C.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [28, 16, 0, 0, 0], [4, 0, 16, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "pool"}>
// CHECK-NOT:   packing_config
// CHECK:       func.func @pool1
// CHECK:       linalg.generic
// CHECK-SAME:    lowering_config = #config
// search: L1 tile [1, 16, 64] (40 KB), 28 cores (7 columns x 4 rows) along OH
// -> block [28, 16, 64]; 4 x 7 blocks; the core forall is 1-D over OH.
// SEARCH:       #config = #iree_codegen.lowering_config<tile_sizes = [
// SEARCH-SAME:      [28, 16, 0, 0, 0], [1, 0, 0, 0, 0]
// SEARCH-SAME:  ]>
// SEARCH:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "pool"}>
// SEARCH:       func.func @pool1
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @pool1() {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x224x64xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<112x112x64xf32>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [224, 224, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x224x64xf32>> -> tensor<224x224x64xf32>
    %3 = tensor.empty() : tensor<112x112x64xf32>
    %4 = tensor.empty() : tensor<2x2xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<112x112x64xf32>) -> tensor<112x112x64xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%2, %4 : tensor<224x224x64xf32>, tensor<2x2xf32>) outs(%5 : tensor<112x112x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.maximumf %out, %in : f32
      linalg.yield %7 : f32
    } -> tensor<112x112x64xf32>
    iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [112, 112, 64], strides = [1, 1, 1] : tensor<112x112x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<112x112x64xf32>>
    return
  }
}

// -----

// VGG-16 pool4 (HWC): 28x28x512 -> 14x14x512.
// spatial-channel: cols 7 (odd factor of 14), rows 4 (128 channels each);
// L1 tile [1, 7, 128] (35 KB), block [7, 7, 512]; OH 2 x OW 2 = 4 blocks.
// 28 cores, where the search only found 4 (OH 14 has no divisor filling
// whole columns).
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [7, 7, 0, 0, 0], [1, 0, 128, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "pool"}>
// CHECK:       func.func @pool4
// search: one column, 4 cores along C, tile [1, 7, 128], block [1, 7, 512];
// 14 x 2 blocks.
// SEARCH:       #config = #iree_codegen.lowering_config<tile_sizes = [
// SEARCH-SAME:      [1, 7, 512, 0, 0], [0, 0, 128, 0, 0]
// SEARCH-SAME:  ]>
// SEARCH:       func.func @pool4
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @pool4() {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x28x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [28, 28, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x28x512xf32>> -> tensor<28x28x512xf32>
    %3 = tensor.empty() : tensor<14x14x512xf32>
    %4 = tensor.empty() : tensor<2x2xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<14x14x512xf32>) -> tensor<14x14x512xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%2, %4 : tensor<28x28x512xf32>, tensor<2x2xf32>) outs(%5 : tensor<14x14x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.maximumf %out, %in : f32
      linalg.yield %7 : f32
    } -> tensor<14x14x512xf32>
    iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [14, 14, 512], strides = [1, 1, 1] : tensor<14x14x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>
    return
  }
}

// -----

// VGG-16 pool5 (HWC): 14x14x512 -> 7x7x512.
// spatial-channel: cols 7 (OH itself), rows 4; tile [1, 7, 128], block
// [7, 7, 512] = the whole output, a single block; 28 cores. OW's tile equals
// its extent at both levels so it is left untiled (0) at both.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [7, 0, 0, 0, 0], [1, 0, 128, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       func.func @pool5
// search: same one-column outcome as pool4: 4 cores along C, tile
// [1, 7, 128], block [1, 7, 512]; 7 blocks.
// SEARCH:       #config = #iree_codegen.lowering_config<tile_sizes = [
// SEARCH-SAME:      [1, 0, 512, 0, 0], [0, 0, 128, 0, 0]
// SEARCH-SAME:  ]>
// SEARCH:       func.func @pool5
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @pool5() {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14x14x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x512xf32>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [14, 14, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14x14x512xf32>> -> tensor<14x14x512xf32>
    %3 = tensor.empty() : tensor<7x7x512xf32>
    %4 = tensor.empty() : tensor<2x2xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<7x7x512xf32>) -> tensor<7x7x512xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%2, %4 : tensor<14x14x512xf32>, tensor<2x2xf32>) outs(%5 : tensor<7x7x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.maximumf %out, %in : f32
      linalg.yield %7 : f32
    } -> tensor<7x7x512xf32>
    iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [7, 7, 512], strides = [1, 1, 1] : tensor<7x7x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x512xf32>>
    return
  }
}
