// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8})' %s | FileCheck %s

// With `native_conv` in the target config, a 2-D conv generic (unit strides
// and dilations, see `getConvDims`; iree-compile generalizes named convs, and a
// named one is rejected, see lowering_strategy_conv_native_errors) takes the
// native conv tile pipeline: translation_info carries the "conv" override and the tile sizes
// are [block, input-channel chunk, core]. Chunks of output rows (`n`, here 4
// halo chunks) go to array columns and output channels to rows; the input
// channels are streamed in multiples of 8 (`chooseConvTiling`). A global
// conv-decompose pipeline keeps the convs it handles (lowering_strategy_conv).

// VGG-16 conv5_1 with its 16 (padded) output rows split into 4 halo chunks and
// an FHWC filter: 4 chunks (columns) x 4 rows x 32 output channels per core,
// 16-channel chunks (41,728 bytes of core memory with double buffering).
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [4, 4, 16, 128, 0, 0, 0], [0, 0, 0, 0, 0, 0, 16], [1, 0, 0, 32, 0, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "conv"}>
// CHECK-NOT:   packing_config
// CHECK:       func.func @conv_chunks_fhwc
// CHECK:       linalg.generic
// CHECK-SAME:    lowering_config = #config
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  func.func @conv_chunks_fhwc() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x6x18x512xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x3x3x512xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x16x512xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 6, 18, 512], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x6x18x512xbf16>> -> tensor<4x6x18x512xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [512, 3, 3, 512], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x3x3x512xbf16>> -> tensor<512x3x3x512xbf16>
    %5 = tensor.empty() : tensor<4x4x16x512xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4x16x512xf32>) -> tensor<4x4x16x512xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<4x6x18x512xbf16>, tensor<512x3x3x512xbf16>) outs(%6 : tensor<4x4x16x512xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %8 = arith.extf %in : bf16 to f32
      %9 = arith.extf %in_0 : bf16 to f32
      %10 = arith.mulf %8, %9 : f32
      %11 = arith.addf %out, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<4x4x16x512xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [4, 4, 16, 512], strides = [1, 1, 1, 1] : tensor<4x4x16x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x16x512xf32>>
    return
  }
}

// -----

// An HWCF conv with the same chunk dim: the memtile split would cut an
// HWCF filter by kernel row, so with chunks on 4 columns its output channels
// are not spread over rows (every core reads the block's 64-channel filter
// tile), input channels in chunks of 8.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [4, 4, 16, 64, 0, 0, 0], [0, 0, 0, 0, 0, 0, 8], [1, 0, 0, 64, 0, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "conv"}>
// CHECK:       func.func @conv_chunks_hwcf
// CHECK:       linalg.generic
// CHECK-SAME:    lowering_config = #config
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  func.func @conv_chunks_hwcf() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x6x18x64xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x64xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x16x64xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 6, 18, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x6x18x64xbf16>> -> tensor<4x6x18x64xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x64xbf16>> -> tensor<3x3x64x64xbf16>
    %5 = tensor.empty() : tensor<4x4x16x64xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4x16x64xf32>) -> tensor<4x4x16x64xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<4x6x18x64xbf16>, tensor<3x3x64x64xbf16>) outs(%6 : tensor<4x4x16x64xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %8 = arith.extf %in : bf16 to f32
      %9 = arith.extf %in_0 : bf16 to f32
      %10 = arith.mulf %8, %9 : f32
      %11 = arith.addf %out, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<4x4x16x64xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [4, 4, 16, 64], strides = [1, 1, 1, 1] : tensor<4x4x16x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x16x64xf32>>
    return
  }
}

// -----

// VGG-16 conv5_1 as IREE forms it: batch 1 folded away (6 loops), input
// pre-padded to 16x16, HWCF filter. Without a chunk dim only output channels
// could be spread, over the rows of one column, but the output-channel tiles of
// the rows would interleave in an HWCF filter, so one core takes them in turn.
// (AMDAIEChunkConvDispatches gives a constant filter the FHWC layout first.)
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [14, 14, 16, 0, 0, 0], [0, 0, 0, 0, 0, 16], [0, 0, 16, 0, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "conv"}>
// CHECK:       func.func @conv_batch_folded
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  func.func @conv_batch_folded() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16x512xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x512x512xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 16, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16x512xbf16>> -> tensor<16x16x512xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 512, 512], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x512x512xbf16>> -> tensor<3x3x512x512xbf16>
    %5 = tensor.empty() : tensor<14x14x512xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<14x14x512xf32>) -> tensor<14x14x512xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<16x16x512xbf16>, tensor<3x3x512x512xbf16>) outs(%6 : tensor<14x14x512xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %8 = arith.extf %in : bf16 to f32
      %9 = arith.extf %in_0 : bf16 to f32
      %10 = arith.mulf %8, %9 : f32
      %11 = arith.addf %out, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<14x14x512xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [14, 14, 512], strides = [1, 1, 1] : tensor<14x14x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>
    return
  }
}

// -----

// 100 output channels and an HWCF bf16 filter: one row (see above), and
// output-channel blocks of 50, whose filter slices (100 bytes) start on 4-byte
// boundaries.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [7, 7, 50, 0, 0, 0], [0, 0, 0, 0, 0, 16], [0, 0, 50, 0, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       func.func @conv_oc100_hwcf
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  func.func @conv_oc100_hwcf() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x64xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x100xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x100xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [9, 9, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x64xbf16>> -> tensor<9x9x64xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 100], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x100xbf16>> -> tensor<3x3x64x100xbf16>
    %5 = tensor.empty() : tensor<7x7x100xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<7x7x100xf32>) -> tensor<7x7x100xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<9x9x64xbf16>, tensor<3x3x64x100xbf16>) outs(%6 : tensor<7x7x100xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<7x7x100xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [7, 7, 100], strides = [1, 1, 1] : tensor<7x7x100xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x100xf32>>
    return
  }
}

// -----

// 8 chunks, 12 output channels and a bf16 (truncf) epilogue: four rows would
// give each core 3 channels, 6-byte epilogue tiles, so three rows of 4 channels
// each are taken.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [8, 2, 16, 12, 0, 0, 0], [0, 0, 0, 0, 0, 0, 64], [1, 0, 0, 4, 0, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       func.func @conv_oc12_trunc
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  func.func @conv_oc12_trunc() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4x18x64xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<12x3x3x64xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2x16x12xbf16>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 4, 18, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4x18x64xbf16>> -> tensor<8x4x18x64xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [12, 3, 3, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<12x3x3x64xbf16>> -> tensor<12x3x3x64xbf16>
    %5 = tensor.empty() : tensor<8x2x16x12xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x2x16x12xf32>) -> tensor<8x2x16x12xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<8x4x18x64xbf16>, tensor<12x3x3x64xbf16>) outs(%6 : tensor<8x2x16x12xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<8x2x16x12xf32>
    %8 = tensor.empty() : tensor<8x2x16x12xbf16>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : tensor<8x2x16x12xf32>) outs(%8 : tensor<8x2x16x12xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %30 = arith.truncf %in : f32 to bf16
      linalg.yield %30 : bf16
    } -> tensor<8x2x16x12xbf16>
    iree_tensor_ext.dispatch.tensor.store %9, %2, offsets = [0, 0, 0, 0], sizes = [8, 2, 16, 12], strides = [1, 1, 1, 1] : tensor<8x2x16x12xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2x16x12xbf16>>
    return
  }
}

// -----

// An in-place ReLU epilogue (the conv result is its input and its init, which
// the payload does not read) is one reader of the result: still the conv
// pipeline, with one epilogue tile in the core memory budget, so the same
// tiles as the ReLU into an empty init below.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [7, 7, 50, 0, 0, 0], [0, 0, 0, 0, 0, 8], [0, 0, 50, 0, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "conv"}>
// CHECK:       func.func @conv_inplace_relu() attributes {translation_info = #translation}
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  func.func @conv_inplace_relu() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x64xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x100xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x100xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [9, 9, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x64xbf16>> -> tensor<9x9x64xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 100], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x100xbf16>> -> tensor<3x3x64x100xbf16>
    %5 = tensor.empty() : tensor<7x7x100xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<7x7x100xf32>) -> tensor<7x7x100xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<9x9x64xbf16>, tensor<3x3x64x100xbf16>) outs(%6 : tensor<7x7x100xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<7x7x100xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<7x7x100xf32>) outs(%7 : tensor<7x7x100xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.maximumf %in, %cst : f32
      linalg.yield %30 : f32
    } -> tensor<7x7x100xf32>
    iree_tensor_ext.dispatch.tensor.store %8, %2, offsets = [0, 0, 0], sizes = [7, 7, 100], strides = [1, 1, 1] : tensor<7x7x100xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x100xf32>>
    return
  }
}

// -----

// The same ReLU into an empty init.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [7, 7, 50, 0, 0, 0], [0, 0, 0, 0, 0, 8], [0, 0, 50, 0, 0, 0]
// CHECK-SAME:  ]>
// CHECK:       func.func @conv_relu_empty_init
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  func.func @conv_relu_empty_init() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x64xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x100xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x100xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [9, 9, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<9x9x64xbf16>> -> tensor<9x9x64xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 64, 100], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x64x100xbf16>> -> tensor<3x3x64x100xbf16>
    %5 = tensor.empty() : tensor<7x7x100xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<7x7x100xf32>) -> tensor<7x7x100xf32>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<9x9x64xbf16>, tensor<3x3x64x100xbf16>) outs(%6 : tensor<7x7x100xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<7x7x100xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<7x7x100xf32>) outs(%5 : tensor<7x7x100xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.maximumf %in, %cst : f32
      linalg.yield %30 : f32
    } -> tensor<7x7x100xf32>
    iree_tensor_ext.dispatch.tensor.store %8, %2, offsets = [0, 0, 0], sizes = [7, 7, 100], strides = [1, 1, 1] : tensor<7x7x100xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x100xf32>>
    return
  }
}
