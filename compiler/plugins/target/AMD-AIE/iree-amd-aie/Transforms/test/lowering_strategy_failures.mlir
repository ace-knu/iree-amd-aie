// Running with the default pipeline:
// RUN: iree-opt %s --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy)' --verify-diagnostics

#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
builtin.module {

  // expected-error@below {{op failed to have a lowering configuration set for it}}
  func.func @matmul_65x65x65_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<65x65xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<65x65xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<65x65xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [65, 65], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<65x65xbf16>> -> tensor<65x65xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [65, 65], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<65x65xbf16>> -> tensor<65x65xbf16>
    %5 = tensor.empty() : tensor<65x65xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<65x65xf32>) -> tensor<65x65xf32>


    // expected-error@below {{'linalg.matmul' op has element types which must target an AIE instruction size that does not divide M (65), N (65), or K (65). The instruction size is m = 4, n = 4, k = 8.}}
    %7 = linalg.matmul ins(%3, %4 : tensor<65x65xbf16>, tensor<65x65xbf16>) outs(%6 : tensor<65x65xf32>) -> tensor<65x65xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [65, 65], strides = [1, 1] : tensor<65x65xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<65x65xf32>>
    return
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
builtin.module {
  // A GEMV with a prime N has no per-core N tile of at least 2 that divides it
  // (the pad pass grows N when there is a host): fail rather than run wrong.
  // expected-error@below {{op failed to have a lowering configuration set for it}}
  func.func @gemv_prime_n() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1009x256xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1009xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xbf16>> -> tensor<1x256xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1009, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1009x256xbf16>> -> tensor<1009x256xbf16>
    %5 = tensor.empty() : tensor<1x1009xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1009xf32>) -> tensor<1x1009xf32>
    // expected-error@below {{no GEMV tiling of N=1009 with a per-core N tile of at least 2}}
    %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>]
         ins(%3, %4 : tensor<1x256xbf16>, tensor<1009x256xbf16>) outs(%6 : tensor<1x1009xf32>) -> tensor<1x1009xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 1009], strides = [1, 1] : tensor<1x1009xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1009xf32>>
    return
  }
}
