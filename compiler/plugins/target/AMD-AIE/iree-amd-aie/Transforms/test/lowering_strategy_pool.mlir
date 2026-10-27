// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8})' %s | FileCheck %s --check-prefix=MAX
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8 pool-columns-policy=cost})' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8 pool-columns-policy=odd})' %s | FileCheck %s --check-prefix=ODD
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8 pool-columns-policy=full})' %s | FileCheck %s --check-prefix=FULL
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8 pool-core-split=search})' %s | FileCheck %s --check-prefix=SEARCH

// 2-D max pooling (channels-last, `d*2 + k` maps, maximumf body) takes the
// copy-based pool pipeline: two tile levels over the output dims only (block,
// core), the 2x2 window untiled, and the per-dispatch pipeline override in
// translation_info (see setRootConfigForPoolPipeline).
//
// Default `pool-core-split=spatial-channel`: array columns take OH, array rows
// take C (C / 4 channels each) and the core forall is 2-D (OH, C). How many
// columns OH is spread over is the `pool-columns-policy`: `max` (default, MAX)
// takes OH's largest divisor <= 8 (all 8 when OH divides by 8, else 7 for the
// VGG-16 layers), never padding; `cost` (CHECK) picks the divisor of OH that
// minimises the per-column cost model t = 0.74 + 0.21 (cols - 1) + 1.77 MB /
// cols [ms] (docs_POOL/1001_pool_column_cost_model.md); `odd` (ODD) takes
// OH's odd factor (7 for every VGG-16 layer); `full` (FULL) takes all 8 when
// OH divides by 8 (the preprocessing pads OH so it does; on an unpadded OH it
// falls back to `odd`).
// `pool-core-split=search` (SEARCH) spreads cores along one output dim only,
// ranked by core count, then channel tile and L1 bytes.

// VGG-16 pool1 (HWC): 224x224x64 -> 112x112x64, 16 MB moved.
// cost: 8 columns (5.76 ms) beat 7 (6.05 ms); rows 4 (C 64 -> 16 per row).
// The L1 tile is the largest-footprint [tOh, tOw, 16] with tOh | 14, tOw | 112:
// [7, 14, 16] f32 = 2 x (14x28x16 + 7x14x16) x 4 B = 61 KB, block
// [8*7, 14, 64] = [56, 14, 64]; OH 2 x OW 8 = 16 blocks in time. Level 0 keeps
// OH (block dim) and OW (14 != 112); level 1 keeps the two core dims OH and C.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [56, 14, 0, 0, 0], [7, 0, 16, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "pool"}>
// CHECK-NOT:   packing_config
// CHECK:       func.func @pool1
// max: 112 divides by 8 -> 8 columns, the same layout as cost.
// MAX:          #config = #iree_codegen.lowering_config<tile_sizes = [
// MAX-SAME:         [56, 14, 0, 0, 0], [7, 0, 16, 0, 0]
// MAX-SAME:     ]>
// MAX:          func.func @pool1
// CHECK:       linalg.generic
// CHECK-SAME:    lowering_config = #config
// odd: cols 7 (odd factor of 112); tile [4, 16, 16] (40 KB, beats [1, 56, 16]
// at 35 KB), block [7*4, 16, 64] = [28, 16, 64]; 4 x 7 = 28 blocks.
// ODD:          #config = #iree_codegen.lowering_config<tile_sizes = [
// ODD-SAME:         [28, 16, 0, 0, 0], [4, 0, 16, 0, 0]
// ODD-SAME:     ]>
// ODD:          func.func @pool1
// full: 112 divides by 8, same as cost.
// FULL:         #config = #iree_codegen.lowering_config<tile_sizes = [
// FULL-SAME:        [56, 14, 0, 0, 0], [7, 0, 16, 0, 0]
// FULL-SAME:    ]>
// FULL:         func.func @pool1
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

// VGG-16 pool4 (HWC): 28x28x512 -> 14x14x512, 2 MB moved.
// cost: of OH's divisors 1, 2, 7 the model picks 7 (2.50 ms); odd and full
// (14 does not divide by 8, no padding here) agree. Rows 4 (128 channels
// each); L1 tile [1, 7, 128] (35 KB), block [7, 7, 512]; OH 2 x OW 2 = 4
// blocks. 28 cores, where the search only found 4 (OH 14 has no divisor
// filling whole columns).
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [7, 7, 0, 0, 0], [1, 0, 128, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "pool"}>
// CHECK:       func.func @pool4
// max: 14's largest divisor <= 8 is 7, as odd.
// MAX:          #config = #iree_codegen.lowering_config<tile_sizes = [
// MAX-SAME:         [7, 7, 0, 0, 0], [1, 0, 128, 0, 0]
// MAX-SAME:     ]>
// MAX:          func.func @pool4
// ODD:          #config = #iree_codegen.lowering_config<tile_sizes = [
// ODD-SAME:         [7, 7, 0, 0, 0], [1, 0, 128, 0, 0]
// ODD-SAME:     ]>
// ODD:          func.func @pool4
// FULL:         #config = #iree_codegen.lowering_config<tile_sizes = [
// FULL-SAME:        [7, 7, 0, 0, 0], [1, 0, 128, 0, 0]
// FULL-SAME:    ]>
// FULL:         func.func @pool4
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

// VGG-16 pool5 (HWC): 14x14x512 -> 7x7x512, 0.5 MB moved.
// cost: one column (1.62 ms) beats 7 (2.12 ms: the per-column fixed cost
// outweighs the bandwidth on half a megabyte), rows 4; tile [1, 7, 128],
// block [1, 7, 512]; 7 OH blocks in time on 4 cores. A one-column grid is
// emitted as the 1-D core forall over C (C is the block dim and the only core
// dim), the same config the search reaches. OW's tile equals its extent at
// both levels so it is left untiled (0) at both.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [1, 0, 512, 0, 0], [0, 0, 128, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       func.func @pool5
// max: 7 columns (OH itself), 28 cores, as odd.
// MAX:          #config = #iree_codegen.lowering_config<tile_sizes = [
// MAX-SAME:         [7, 0, 0, 0, 0], [1, 0, 128, 0, 0]
// MAX-SAME:     ]>
// MAX:          func.func @pool5
// odd / full (7 does not divide by 8): cols 7 (OH itself), block [7, 7, 512]
// = the whole output, a single block; 28 cores.
// ODD:          #config = #iree_codegen.lowering_config<tile_sizes = [
// ODD-SAME:         [7, 0, 0, 0, 0], [1, 0, 128, 0, 0]
// ODD-SAME:     ]>
// ODD:          func.func @pool5
// FULL:         #config = #iree_codegen.lowering_config<tile_sizes = [
// FULL-SAME:        [7, 0, 0, 0, 0], [1, 0, 128, 0, 0]
// FULL-SAME:    ]>
// FULL:         func.func @pool5
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


// -----

// VGG-16 pool4 after the preprocessing has padded OH 14 -> 16 for
// `pool-columns-policy=full` (input 28 -> 32 rows): 32x28x512 -> 16x14x512.
// full: 8 columns (2 OH rows each), rows 4; tile [1, 7, 128] (35 KB, [2, 7]
// would be 70 KB), block [8, 7, 512]; OH 2 x OW 2 = 4 blocks; 32 cores.
// cost (2.3 MB moved): 4 columns (2.38 ms) beat 8 (2.72 ms), 2 and 1; with
// 4 OH rows per column the largest L1 tile is [4, 2, 128] (40 KB, over
// [1, 7, 128] at 35 KB), block [16, 2, 512]; 7 OW blocks; 16 cores.
// odd: 16 has no odd factor > 1, so its largest divisor <= 8: 8, as full.
// FULL:         #config = #iree_codegen.lowering_config<tile_sizes = [
// FULL-SAME:        [8, 7, 0, 0, 0], [1, 0, 128, 0, 0]
// FULL-SAME:    ]>
// FULL:         func.func @pool4_padded
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [16, 2, 0, 0, 0], [4, 0, 128, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       func.func @pool4_padded
// max: 16 divides by 8 -> 8 columns, as full.
// MAX:          #config = #iree_codegen.lowering_config<tile_sizes = [
// MAX-SAME:         [8, 7, 0, 0, 0], [1, 0, 128, 0, 0]
// MAX-SAME:     ]>
// MAX:          func.func @pool4_padded
// ODD:          #config = #iree_codegen.lowering_config<tile_sizes = [
// ODD-SAME:         [8, 7, 0, 0, 0], [1, 0, 128, 0, 0]
// ODD-SAME:     ]>
// ODD:          func.func @pool4_padded
// SEARCH:       func.func @pool4_padded
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @pool4_padded() {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x28x512xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x14x512xf32>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [32, 28, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x28x512xf32>> -> tensor<32x28x512xf32>
    %3 = tensor.empty() : tensor<16x14x512xf32>
    %4 = tensor.empty() : tensor<2x2xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<16x14x512xf32>) -> tensor<16x14x512xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%2, %4 : tensor<32x28x512xf32>, tensor<2x2xf32>) outs(%5 : tensor<16x14x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.maximumf %out, %in : f32
      linalg.yield %7 : f32
    } -> tensor<16x14x512xf32>
    iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [16, 14, 512], strides = [1, 1, 1] : tensor<16x14x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x14x512xf32>>
    return
  }
}

// -----

// Halo pre-split pooling (ResNet stem 3x3/2 after the preprocessing split):
// 7 chunks x 17 input rows -> 7 x 8 output rows, C 64. The leading chunk dim
// takes the columns (7), C the rows (16 channels each); with the whole OH of a
// chunk free for the tile the largest L1 footprint is [1, 56, 16]: input
// 3x113x16 + output 1x56x16, x2 buffers = 49 KB. Block [7, 1, 56, 64], 8 OH
// blocks in time. Level 0 keeps N (block dim) and OH (1 != 8); level 1 keeps
// the two core dims N and C.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [7, 1, 0, 0, 0, 0], [1, 0, 0, 16, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       func.func @pool_halo_chunks
// MAX:          #config = #iree_codegen.lowering_config<tile_sizes = [
// MAX-SAME:         [7, 1, 0, 0, 0, 0], [1, 0, 0, 16, 0, 0]
// MAX-SAME:     ]>
// MAX:          func.func @pool_halo_chunks
// The chunk count is whatever the preprocessing made it, under every policy.
// ODD:          #config = #iree_codegen.lowering_config<tile_sizes = [
// ODD-SAME:         [7, 1, 0, 0, 0, 0], [1, 0, 0, 16, 0, 0]
// ODD-SAME:     ]>
// ODD:          func.func @pool_halo_chunks
// FULL:         #config = #iree_codegen.lowering_config<tile_sizes = [
// FULL-SAME:        [7, 1, 0, 0, 0, 0], [1, 0, 0, 16, 0, 0]
// FULL-SAME:    ]>
// FULL:         func.func @pool_halo_chunks
// search: one distributed dim only; N = 7 fills 7 columns? No: 7 is not a
// multiple of the row count, so the search spreads C over one column.
// SEARCH:       func.func @pool_halo_chunks
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @pool_halo_chunks() {
    %cst = arith.constant 0xFF800000 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x17x114x64xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x8x56x64xf32>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [7, 17, 114, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x17x114x64xf32>> -> tensor<7x17x114x64xf32>
    %3 = tensor.empty() : tensor<7x8x56x64xf32>
    %4 = tensor.empty() : tensor<3x3xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<7x8x56x64xf32>) -> tensor<7x8x56x64xf32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%2, %4 : tensor<7x17x114x64xf32>, tensor<3x3xf32>) outs(%5 : tensor<7x8x56x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.maximumf %out, %in : f32
      linalg.yield %7 : f32
    } -> tensor<7x8x56x64xf32>
    iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0, 0, 0, 0], sizes = [7, 8, 56, 64], strides = [1, 1, 1, 1] : tensor<7x8x56x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x8x56x64xf32>>
    return
  }
}
