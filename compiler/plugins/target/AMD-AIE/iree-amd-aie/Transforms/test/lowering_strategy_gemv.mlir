// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8})' %s | FileCheck %s

// M == 1 (GEMV) matmuls take the GEMV pipeline instead of pack-peel: M is left
// untiled, N is spread over all 32 cores (L0 block = 32 cores x 32), K is
// streamed in tiles of 256, there is no packing_config, and translation_info
// carries the per-dispatch pipeline override. Every other M stays on pack-peel
// (see isGemvMExtent for why 2..31 is not taken yet).

// VGG-16 fc1 as ONNX Gemm lowers it: M=1, transpose_b weight [N, K].
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [0, 1024, 0], [0, 0, 256], [0, 32, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "gemv"}>
// CHECK-NOT:   packing_config
// CHECK:       func.func @gemv_fc1_transpose_b
// CHECK:       linalg.matmul
// CHECK-SAME:    {lowering_config = #config}
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @gemv_fc1_transpose_b() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x25088xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x25088xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 25088], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x25088xbf16>> -> tensor<1x25088xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 25088], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x25088xbf16>> -> tensor<4096x25088xbf16>
    %5 = tensor.empty() : tensor<1x4096xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>]
         ins(%3, %4 : tensor<1x25088xbf16>, tensor<4096x25088xbf16>) outs(%6 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : tensor<1x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
    return
  }
}

// -----

// N=1000 (fc3, unpadded): no divisor of 1000 spreads it over 32 (or 24) cores
// in whole columns, so the tile reaching the most cores is picked, largest
// first: 50 outputs per core x 20 cores (5 columns) in one L0 block of 1000.
// No padding is required by this pipeline.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [0, 1000, 0], [0, 0, 256], [0, 50, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "gemv"}>
// CHECK-NOT:   packing_config
// CHECK:       func.func @gemv_fc3_n1000
// CHECK:       linalg.matmul
// CHECK-SAME:    {lowering_config = #config}
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @gemv_fc3_n1000() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>> -> tensor<1x4096xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>> -> tensor<1000x4096xbf16>
    %5 = tensor.empty() : tensor<1x1000xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>]
         ins(%3, %4 : tensor<1x4096xbf16>, tensor<1000x4096xbf16>) outs(%6 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 1000], strides = [1, 1] : tensor<1x1000xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
    return
  }
}

// -----

// M=16 (a batch of 16) is not a GEMV shape today: only M == 1 is routed, so
// it keeps the pack-peel config and packing_config, with no pipeline override.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = {{.*}}>
// CHECK:       #packingConfig = #amdaie.packing_config
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom>
// CHECK:       func.func @packpeel_m16
// CHECK:       linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @packpeel_m16() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x4096xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x4096xbf16>> -> tensor<16x4096xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>> -> tensor<4096x4096xbf16>
    %5 = tensor.empty() : tensor<16x4096xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<16x4096xbf16>, tensor<4096x4096xbf16>) outs(%6 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [16, 4096], strides = [1, 1] : tensor<16x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x4096xf32>>
    return
  }
}

// -----

// M=24 likewise stays on pack-peel (the Flow pad pass would have padded it to 32
// before codegen; here M is already a multiple of the 8-row instruction).
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = {{.*}}>
// CHECK:       #packingConfig = #amdaie.packing_config
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom>
// CHECK:       func.func @packpeel_m24
// CHECK:       linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @packpeel_m24() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x4096xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x4096xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [24, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<24x4096xbf16>> -> tensor<24x4096xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>> -> tensor<4096x4096xbf16>
    %5 = tensor.empty() : tensor<24x4096xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<24x4096xf32>) -> tensor<24x4096xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<24x4096xbf16>, tensor<4096x4096xbf16>) outs(%6 : tensor<24x4096xf32>) -> tensor<24x4096xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [24, 4096], strides = [1, 1] : tensor<24x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<24x4096xf32>>
    return
  }
}

// -----

// The GEMV predicate is M == 1 regardless of element type, so f32 M=1 (npu4 has
// no f32 matmul vector instruction; pack-peel used to abort on it) is GEMV too.
// 4-byte operands halve the N tile (the K tile stays 256): n1=32 needs
// 2*(256*4 + 32*256*4 + 32*4) = 66.25 KB > 64 KB, n1=16 still covers 32 cores.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [0, 512, 0], [0, 0, 256], [0, 16, 0]
// CHECK-SAME:  ]>
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "gemv"}>
// CHECK-NOT:   packing_config
// CHECK:       func.func @gemv_f32_m1
// CHECK:       linalg.matmul
// CHECK-SAME:    {lowering_config = #config}
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @gemv_f32_m1() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xf32>> -> tensor<1x4096xf32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<4096x4096xf32>
    %5 = tensor.empty() : tensor<1x4096xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<1x4096xf32>, tensor<4096x4096xf32>) outs(%6 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : tensor<1x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
    return
  }
}

// -----

// M=32 is exactly the threshold (4 rows x 8): pack-peel as before, with a
// packing_config and no pipeline override.
// CHECK:       #config = #iree_codegen.lowering_config<tile_sizes = [
// CHECK-SAME:      [32, 256, 0], [0, 0, 1], [1, 1, 0]
// CHECK-SAME:  ]>
// CHECK:       #packingConfig = #amdaie.packing_config
// CHECK:       #translation = #iree_codegen.translation_info<pipeline = Custom>
// CHECK:       func.func @packpeel_m32_boundary
// CHECK:       linalg.matmul {lowering_config = #config, packing_config = #packingConfig}
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
module {
  func.func @packpeel_m32_boundary() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x4096xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xbf16>> -> tensor<32x4096xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>> -> tensor<4096x4096xbf16>
    %5 = tensor.empty() : tensor<32x4096xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x4096xf32>) -> tensor<32x4096xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<32x4096xbf16>, tensor<4096x4096xbf16>) outs(%6 : tensor<32x4096xf32>) -> tensor<32x4096xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [32, 4096], strides = [1, 1] : tensor<32x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x4096xf32>>
    return
  }
}
