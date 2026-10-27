// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8 enable-ukernels=all})' %s | FileCheck %s

// With the matvec ukernel (bf16 x bf16 -> f32) the GEMV K tile is whole
// ukernel vectors of 64: K=320 takes 64 (5 x 64), not 160, the largest factor
// <= 256 the scalar tile would take.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [0, 1024, 0], [0, 0, 64], [0, 32, 0]
// CHECK-SAME:  ]>
// CHECK:       func.func @gemv_matvec_k320
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @gemv_matvec_k320() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x320xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xbf16>> -> tensor<1x320xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x320xbf16>> -> tensor<1024x320xbf16>
    %5 = tensor.empty() : tensor<1x1024xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>]
         ins(%3, %4 : tensor<1x320xbf16>, tensor<1024x320xbf16>) outs(%6 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 1024], strides = [1, 1] : tensor<1x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
    return
  }
}

// -----

// f32 has no matvec template, so K=320 keeps the largest factor <= 256: 160.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [0, 1024, 0], [0, 0, 160]
// CHECK:       func.func @gemv_f32_k320
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @gemv_f32_k320() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x320xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x320xf32>> -> tensor<1x320xf32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 320], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x320xf32>> -> tensor<1024x320xf32>
    %5 = tensor.empty() : tensor<1x1024xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>]
         ins(%3, %4 : tensor<1x320xf32>, tensor<1024x320xf32>) outs(%6 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 1024], strides = [1, 1] : tensor<1x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
    return
  }
}
