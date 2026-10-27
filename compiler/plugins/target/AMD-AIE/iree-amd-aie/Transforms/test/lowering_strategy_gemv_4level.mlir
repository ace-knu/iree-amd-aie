// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8 use-tile-pipeline=pack-peel-4-level-tiling})' %s | FileCheck %s

// The GEMV pipeline is selected per dispatch, before the global tile pipeline
// is consulted, so an M == 1 matmul gets the same GEMV config under the
// pack-peel-4-level-tiling global pipeline as under pack-peel. (Without this,
// 4-level tiling would receive the unpadded M=1 that the Flow pad pass leaves
// for GEMV shapes and fail its instruction-size divisibility check.)
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [0, 1024, 0], [0, 0, 256], [0, 32, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "gemv"}>
// CHECK-NOT:   packing_config
// CHECK:       func.func @gemv_fc1_4level
// CHECK:       linalg.matmul
// CHECK-SAME:    {lowering_config = #config}
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @gemv_fc1_4level() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x25088xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x25088xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 25088], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x25088xbf16>> -> tensor<1x25088xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 25088], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x25088xbf16>> -> tensor<4096x25088xbf16>
    %5 = tensor.empty() : tensor<1x4096xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%3, %4 : tensor<1x25088xbf16>, tensor<4096x25088xbf16>) outs(%6 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : tensor<1x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
    return
  }
}

// -----

// A padded M (32) stays on 4-level pack-peel: no override, packing_config present.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [32, 128, 0], [4, 8, 0], [0, 0, 1], [1, 1, 0]
// CHECK-SAME:  ]>
// CHECK:       #packingConfig = #amdaie.packing_config
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom>
// CHECK:       func.func @packpeel4_m32
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @packpeel4_m32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x4096xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xbf16>> -> tensor<32x4096xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>> -> tensor<4096x4096xbf16>
    %5 = tensor.empty() : tensor<32x4096xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x4096xf32>) -> tensor<32x4096xf32>
    %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%3, %4 : tensor<32x4096xbf16>, tensor<4096x4096xbf16>) outs(%6 : tensor<32x4096xf32>) -> tensor<32x4096xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32, 4096], strides = [1, 1] : tensor<32x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x4096xf32>>
    return
  }
}
