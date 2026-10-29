// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-bufferize-to-allocation{memory-space=2 bufferize-elementwise=true bufferize-operand=linalg-input}))' %s | FileCheck %s

// A packed transpose_b matmul feeding an elementwise op: the matmul result is
// not promoted to another buffer, as for a plain matmul.
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d7, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
func.func @matmul_transpose_b_elementwise(%arg0: tensor<1x16x64x32xbf16>, %arg1: tensor<16x1x64x32xbf16>) -> tensor<1x1x8x16x4x8xbf16> {
  %cst = arith.constant 0.0 : f32
  %a = tensor.empty() : tensor<1x16x4x16x4x8xbf16>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %a : tensor<1x16x64x32xbf16> -> tensor<1x16x4x16x4x8xbf16>
  %b = tensor.empty() : tensor<16x1x8x4x8x8xbf16>
  %pack_0 = linalg.pack %arg1 inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %b : tensor<16x1x64x32xbf16> -> tensor<16x1x8x4x8x8xbf16>
  %0 = tensor.empty() : tensor<1x1x8x16x4x8xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x8x16x4x8xf32>) -> tensor<1x1x8x16x4x8xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<1x16x4x16x4x8xbf16>, tensor<16x1x8x4x8x8xbf16>) outs(%1 : tensor<1x1x8x16x4x8xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %5 = arith.extf %in : bf16 to f32
    %6 = arith.extf %in_0 : bf16 to f32
    %7 = arith.mulf %5, %6 : f32
    %8 = arith.addf %out, %7 : f32
    linalg.yield %8 : f32
  } -> tensor<1x1x8x16x4x8xf32>
  %3 = tensor.empty() : tensor<1x1x8x16x4x8xbf16>
  %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x1x8x16x4x8xf32>) outs(%3 : tensor<1x1x8x16x4x8xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %5 = arith.maximumf %in, %cst : f32
    %6 = arith.truncf %5 : f32 to bf16
    linalg.yield %6 : bf16
  } -> tensor<1x1x8x16x4x8xbf16>
  return %4 : tensor<1x1x8x16x4x8xbf16>
}

// CHECK-LABEL: @matmul_transpose_b_elementwise
// CHECK-NOT:     memref.alloc
// CHECK:         return
