// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-target-device=npu4 --iree-amdaie-enable-native-conv --iree-amdaie-enable-vectorization-passes=false %s | FileCheck %s

// A 3x3 conv whose output rows were split into 8 halo chunks (FHWC filter)
// takes the native conv pipeline with `--iree-amdaie-enable-native-conv`: one
// chunk per array column, output channels over the 4 rows, so every one of
// the 32 cores of npu4 gets a core program.

// CHECK-LABEL: hal.executable.export public @conv_chunks_dispatch_0_conv_8x2x16x64x3x3x64_bf16xbf16xf32
// CHECK:       aie.device(npu4)
// CHECK-COUNT-32: aie.core(
// CHECK-NOT:   aie.core(
func.func @conv_chunks(%arg0: tensor<8x4x18x64xbf16>, %arg1: tensor<64x3x3x64xbf16>) -> tensor<8x2x16x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<8x2x16x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x2x16x64xf32>) -> tensor<8x2x16x64xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
       ins(%arg0, %arg1 : tensor<8x4x18x64xbf16>, tensor<64x3x3x64xbf16>) outs(%1 : tensor<8x2x16x64xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %a = arith.extf %in : bf16 to f32
    %b = arith.extf %in_0 : bf16 to f32
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %out, %m : f32
    linalg.yield %s : f32
  } -> tensor<8x2x16x64xf32>
  return %2 : tensor<8x2x16x64xf32>
}
