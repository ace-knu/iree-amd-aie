// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-lower-to-ukernels,cse,canonicalize))" %s | FileCheck %s

// This first case demonstrates no lowering to ukernel when the corresponding
// config is set to "none".
func.func @disabled_ukernel(%arg0 : tensor<?x?x?x?xi32>, %arg1 : tensor<?x?x?x?xi32>,
    %arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
} {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>)
                        outs(%arg2 : tensor<?x?x?x?xi32>)
      {
        ^bb0(%in: i32, %in_9: i32, %out: i32):
          %22 = arith.muli %in, %in_9 : i32
          %23 = arith.addi %out, %22 : i32
          linalg.yield %23 : i32
      } -> tensor<?x?x?x?xi32>

  return %0 : tensor<?x?x?x?xi32>
}
//      CHECK: func @disabled_ukernel
//      CHECK:  linalg.generic
//  CHECK-NOT:  iree_codegen.ukernel.generic

// -----

func.func @generic_matmul_i32i32i32_pad_pack(%arg0 : tensor<8x16x4x8xi32>, %arg1 : tensor<16x8x8x4xi32>,
    %arg2 : tensor<16x16x4x4xi32>) -> tensor<16x16x4x4xi32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<8x16x4x8xi32>, tensor<16x8x8x4xi32>)
                        outs(%arg2 : tensor<16x16x4x4xi32>)
      {
        ^bb0(%in: i32, %in_9: i32, %out: i32):
          %22 = arith.muli %in, %in_9 : i32
          %23 = arith.addi %out, %22 : i32
          linalg.yield %23 : i32
      } -> tensor<16x16x4x4xi32>

  return %0 : tensor<16x16x4x4xi32>
}
//      CHECK: func @generic_matmul_i32i32i32_pad_pack(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x16x4x8xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<16x8x8x4xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<16x16x4x4xi32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_i32_i32_i32_64x64x64_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o", ukernel_instance = ["matmul_i32_i32_i32_64x64x64_4x8x4", "i32", "i32", "i32", "64", "64", "64", "4", "8", "4", "false"]}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @generic_matmul_bf16bf16f32_pad_pack(%arg0: tensor<8x16x4x8xbf16>, %arg1: tensor<16x8x8x4xbf16>,
      %arg2: tensor<16x16x4x4xf32>) -> tensor<16x16x4x4xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "reduction",
                         "parallel", "parallel", "reduction"]
                        } ins(%arg0, %arg1 : tensor<8x16x4x8xbf16>, tensor<16x8x8x4xbf16>)
                          outs(%arg2 : tensor<16x16x4x4xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_0 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x16x4x4xf32>
    return %0 : tensor<16x16x4x4xf32>
  }
}
//      CHECK: func @generic_matmul_bf16bf16f32_pad_pack(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x16x4x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<16x8x8x4xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<16x16x4x4xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_f32_64x64x64_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o", ukernel_instance = [
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @generic_matmul_bf16bf16f32_pack_peel(%arg0: tensor<1x1x8x16x4x8xbf16>, %arg1: tensor<1x1x16x8x8x4xbf16>,
      %arg2: tensor<1x1x16x16x4x4xf32>) -> tensor<1x1x16x16x4x4xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
                                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
                                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>],
                      iterator_types = ["parallel", "parallel", "reduction",
                                        "parallel", "parallel", "reduction",
                                        "parallel", "parallel", "reduction"]
                        } ins(%arg0, %arg1 : tensor<1x1x8x16x4x8xbf16>, tensor<1x1x16x8x8x4xbf16>)
                          outs(%arg2 : tensor<1x1x16x16x4x4xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_0 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<1x1x16x16x4x4xf32>
    return %0 : tensor<1x1x16x16x4x4xf32>
  }
}
//      CHECK: func @generic_matmul_bf16bf16f32_pack_peel(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x8x16x4x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x1x16x8x8x4xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x16x16x4x4xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_f32_64x64x64_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o", ukernel_instance = [
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @generic_matmul_bf16bf16f32_pack_peel_objectfifo(%arg0: tensor<1x1x4x8x4x8xbf16>, %arg1: tensor<1x1x8x4x8x4xbf16>,
      %arg2: tensor<1x1x8x8x4x4xf32>) -> tensor<1x1x8x8x4x4xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
                                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
                                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>],
                      iterator_types = ["parallel", "parallel", "reduction",
                                        "parallel", "parallel", "reduction",
                                        "parallel", "parallel", "reduction"]
                        } ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xbf16>, tensor<1x1x8x4x8x4xbf16>)
                          outs(%arg2 : tensor<1x1x8x8x4x4xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_0 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<1x1x8x8x4x4xf32>
    return %0 : tensor<1x1x8x8x4x4xf32>
  }
}
//      CHECK: func @generic_matmul_bf16bf16f32_pack_peel_objectfifo(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x4x8x4x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x1x8x4x8x4xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x8x8x4x4xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_f32_32x32x32_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o", ukernel_instance = [
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

// A matmul with a transposed B, which the npu4 bf16 ukernel reads as [n][k]
// blocks.
#map_a = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map_b = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d5, d4, d7, d8)>
#map_c = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d0, d4, d3, d6, d7)>
func.func @generic_matmul_transpose_b_bf16bf16f32(%arg0: tensor<1x1x16x2x8x8xbf16>, %arg1: tensor<1x1x16x4x8x8xbf16>,
    %arg2: tensor<1x1x4x2x8x8xf32>) -> tensor<1x1x4x2x8x8xf32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
} {
  %0 = linalg.generic {indexing_maps = [#map_a, #map_b, #map_c],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<1x1x16x2x8x8xbf16>, tensor<1x1x16x4x8x8xbf16>)
                        outs(%arg2 : tensor<1x1x4x2x8x8xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %1 = arith.extf %in : bf16 to f32
    %2 = arith.extf %in_0 : bf16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1x1x4x2x8x8xf32>
  return %0 : tensor<1x1x4x2x8x8xf32>
}
//      CHECK: func @generic_matmul_transpose_b_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x16x2x8x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x1x16x4x8x8xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x4x2x8x8xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_tb_bf16_bf16_f32_16x32x128_8x8x8"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o", ukernel_instance = ["matmul_tb_bf16_bf16_f32_16x32x128_8x8x8", "bf16", "bf16", "f32", "16", "32", "128", "8", "8", "8", "true"]}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

// Only npu4 has a ukernel for a transposed B.
#map_a = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map_b = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d5, d4, d7, d8)>
#map_c = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d0, d4, d3, d6, d7)>
func.func @generic_matmul_transpose_b_npu1(%arg0: tensor<1x1x16x2x8x8xbf16>, %arg1: tensor<1x1x16x4x8x8xbf16>,
    %arg2: tensor<1x1x4x2x8x8xf32>) -> tensor<1x1x4x2x8x8xf32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %0 = linalg.generic {indexing_maps = [#map_a, #map_b, #map_c],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<1x1x16x2x8x8xbf16>, tensor<1x1x16x4x8x8xbf16>)
                        outs(%arg2 : tensor<1x1x4x2x8x8xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %1 = arith.extf %in : bf16 to f32
    %2 = arith.extf %in_0 : bf16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1x1x4x2x8x8xf32>
  return %0 : tensor<1x1x4x2x8x8xf32>
}
//      CHECK: func @generic_matmul_transpose_b_npu1(
//  CHECK-NOT:   iree_codegen.ukernel.generic
//      CHECK:   linalg.generic

// -----

func.func @zero_fill(%arg0 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %cst = arith.constant 0.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg0 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  return %fill : tensor<16x16x4x4xbf16>
}
//      CHECK: func @zero_fill(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<16x16x4x4xbf16>)
//  CHECK-NOT:   linalg.fill
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "zero_fill_bf16_64x64"
// CHECK-SAME:       outs(%[[ARG0]] :
// CHECK-SAME:       fn_def_attrs {link_with = "zero_fill.o", ukernel_instance = ["zero_fill_bf16_64x64", "bf16", "64", "64"]}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @non_zero_fill(%arg0 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %cst = arith.constant 7.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg0 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  return %fill : tensor<16x16x4x4xbf16>
}
//      CHECK: func @non_zero_fill
//      CHECK:   linalg.fill
//  CHECK-NOT:   iree_codegen.ukernel.generic

// -----

func.func @zero_fill_with_matmul(%arg0 : tensor<8x16x4x8xbf16>, %arg1 : tensor<16x8x8x4xbf16>,
    %arg2 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %cst = arith.constant 0.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg2 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  %matmul = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<8x16x4x8xbf16>, tensor<16x8x8x4xbf16>)
                        outs(%fill : tensor<16x16x4x4xbf16>)
      {
        ^bb0(%in: bf16, %in_9: bf16, %out: bf16):
          %22 = arith.mulf %in, %in_9 : bf16
          %23 = arith.addf %out, %22 : bf16
          linalg.yield %23 : bf16
      } -> tensor<16x16x4x4xbf16>
  return %matmul : tensor<16x16x4x4xbf16>
}
//      CHECK: func @zero_fill_with_matmul(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x16x4x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<16x8x8x4xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<16x16x4x4xbf16>)
//  CHECK-NOT:   linalg.fill
//      CHECK:   %[[ZERO_FILL_MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "zero_fill_bf16_64x64"
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "zero_fill.o", ukernel_instance = [
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MATMUL_MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_bf16_64x64x64_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ZERO_FILL_MICRO_KERNEL]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o", ukernel_instance = [
//      CHECK:   return %[[MATMUL_MICRO_KERNEL]]

// -----

// In this test, we have a linalg.fill followed by a linalg.matmul, followed by
// a linalg.generic representing x -> x + 1. The linalg.matmul should be lowered
// to a ukernel, as should the linalg.fill, but the final elementwise addition
// should not be.
func.func @zero_fill_matmul_elmwise(%arg0 : tensor<8x16x4x8xbf16>, %arg1 : tensor<16x8x8x4xbf16>,
    %arg2 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb",
  {target_device = "npu1_4col", ukernels = "all"}>
} {
  %cst = arith.constant 0.0 : bf16
  %cst_1 = arith.constant 1.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg2 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  %matmul = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<8x16x4x8xbf16>, tensor<16x8x8x4xbf16>)
                        outs(%fill : tensor<16x16x4x4xbf16>)
      {
        ^bb0(%in: bf16, %in_9: bf16, %out: bf16):
          %22 = arith.mulf %in, %in_9 : bf16
          %23 = arith.addf %out, %22 : bf16
          linalg.yield %23 : bf16
      } -> tensor<16x16x4x4xbf16>
  // Perform an elementwise addition of 1 to the result of the matmul.
  %matmul_add = linalg.generic {indexing_maps = [
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
                           ins(%matmul : tensor<16x16x4x4xbf16>)
                           outs(%arg2 : tensor<16x16x4x4xbf16>)
      {
      ^bb0(%in: bf16, %out: bf16):
          %1 = arith.addf %in, %cst_1 : bf16
          linalg.yield %1 : bf16
      } -> tensor<16x16x4x4xbf16>

  return %matmul_add : tensor<16x16x4x4xbf16>
}
// CHECK: func @zero_fill_matmul_elmwise
// CHECK-NOT: linalg.fill
// CHECK: iree_codegen.ukernel.generic "zero_fill_bf16_64x64"
// CHECK-NOT: linalg.fill
// CHECK: iree_codegen.ukernel.generic "matmul_bf16_bf16_bf16_64x64x64_4x8x4"
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK: return

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @generic_matmul_bf16bf16f32_pack_peel_objectfifo(
      %arg0: tensor<1x1x4x4x8x8xbf16>, %arg1: tensor<1x1x4x4x8x8xbf16>,
      %arg2: tensor<1x1x4x4x8x8xf32>) -> tensor<1x1x4x4x8x8xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
        ],
        iterator_types = [
          "parallel", "parallel", "reduction",
          "parallel", "parallel", "reduction",
          "parallel", "parallel", "reduction"
        ]
      } ins(%arg0, %arg1 : tensor<1x1x4x4x8x8xbf16>, tensor<1x1x4x4x8x8xbf16>)
        outs(%arg2 : tensor<1x1x4x4x8x8xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %1 = arith.extf %in : bf16 to f32
          %2 = arith.extf %in_0 : bf16 to f32
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %out, %3 : f32
          linalg.yield %4 : f32
      } -> tensor<1x1x4x4x8x8xf32>
    return %0 : tensor<1x1x4x4x8x8xf32>
  }
}
//      CHECK: func @generic_matmul_bf16bf16f32_pack_peel_objectfifo(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x4x4x8x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x1x4x4x8x8xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x4x4x8x8xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_f32_32x32x32_8x8x8"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o", ukernel_instance = [
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

// The npu4 ukernels have no f32 matmul or i16 zero fill template, so these are
// not lowered.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
module {
  func.func @generic_matmul_f32f32f32_no_template(
      %arg0: tensor<1x1x4x4x4x8xf32>, %arg1: tensor<1x1x4x4x8x4xf32>,
      %arg2: tensor<1x1x4x4x4x4xf32>) -> tensor<1x1x4x4x4x4xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
        ],
        iterator_types = [
          "parallel", "parallel", "reduction",
          "parallel", "parallel", "reduction",
          "parallel", "parallel", "reduction"
        ]
      } ins(%arg0, %arg1 : tensor<1x1x4x4x4x8xf32>, tensor<1x1x4x4x8x4xf32>)
        outs(%arg2 : tensor<1x1x4x4x4x4xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %1 = arith.mulf %in, %in_0 : f32
          %2 = arith.addf %out, %1 : f32
          linalg.yield %2 : f32
      } -> tensor<1x1x4x4x4x4xf32>
    return %0 : tensor<1x1x4x4x4x4xf32>
  }
  func.func @zero_fill_i16_no_template(%arg0: tensor<1x1x4x4x4x4xi16>) -> tensor<1x1x4x4x4x4xi16> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %c0 = arith.constant 0 : i16
    %0 = linalg.fill ins(%c0 : i16) outs(%arg0 : tensor<1x1x4x4x4x4xi16>) -> tensor<1x1x4x4x4x4xi16>
    return %0 : tensor<1x1x4x4x4x4xi16>
  }
}
// CHECK-LABEL: func @generic_matmul_f32f32f32_no_template(
//       CHECK:   linalg.generic
//   CHECK-NOT:   iree_codegen.ukernel.generic
// CHECK-LABEL: func @zero_fill_i16_no_template(
//       CHECK:   linalg.fill
//   CHECK-NOT:   iree_codegen.ukernel.generic

// -----

func.func @shift_trunci(%arg0 : tensor<16x16x4x4xi32>) -> tensor<16x16x4x4xi8> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
} {
  %cst_shift = arith.constant 5 : i32
  %0 = tensor.empty() : tensor<16x16x4x4xi8>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]
                      } ins(%arg0 : tensor<16x16x4x4xi32>) outs(%0 : tensor<16x16x4x4xi8>) {
      ^bb0(%in: i32, %out: i8):
        %2 = arith.shrsi %in, %cst_shift : i32
        %3 = arith.trunci %2 : i32 to i8
        linalg.yield %3 : i8
    } -> tensor<16x16x4x4xi8>
  return %1 : tensor<16x16x4x4xi8>
}
// CHECK-LABEL:  func @shift_trunci
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<16x16x4x4xi32>)
// CHECK:        %[[C5:.+]] = arith.constant 5 : i32
// CHECK-NOT:    linalg.generic
// CHECK:        %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "trunci_i32_i8_64x64"
// CHECK-SAME:       ins(%[[ARG0]], %[[C5]] : tensor<16x16x4x4xi32>, i32
// CHECK-SAME:       fn_def_attrs {link_with = "trunci.o", ukernel_instance = ["trunci_i32_i8_64x64", "i32", "i8", "64", "64"]}
// CHECK-SAME:       -> tensor<16x16x4x4xi8>
// CHECK:        return %[[MICRO_KERNEL]]

// -----

func.func @trunci(%arg0 : tensor<16x16x4x4xi16>) -> tensor<16x16x4x4xi8> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
} {
  %0 = tensor.empty() : tensor<16x16x4x4xi8>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]
                      } ins(%arg0 : tensor<16x16x4x4xi16>) outs(%0 : tensor<16x16x4x4xi8>) {
      ^bb0(%in: i16, %out: i8):
        %2 = arith.trunci %in : i16 to i8
        linalg.yield %2 : i8
    } -> tensor<16x16x4x4xi8>
  return %1 : tensor<16x16x4x4xi8>
}
// The npu4 trunci ukernel has no i16 -> i8 template, so this is not lowered.
// Expect it to be lowered again once such a template is added.
// CHECK-LABEL:  func @trunci
// CHECK:        linalg.generic
// CHECK-NOT:    iree_codegen.ukernel.generic

// -----

// A tile with an odd number of vector instructions in M (56 = 7 x 8) is not
// lowered: the ukernels process 2x2 instructions per step.
func.func @generic_matmul_bf16bf16f32_odd_m(%arg0 : tensor<4x7x8x8xbf16>, %arg1 : tensor<4x4x8x8xbf16>,
    %arg2 : tensor<4x7x8x8xf32>) -> tensor<4x7x8x8xf32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
} {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<4x7x8x8xbf16>, tensor<4x4x8x8xbf16>)
                        outs(%arg2 : tensor<4x7x8x8xf32>)
      {
        ^bb0(%in: bf16, %in_9: bf16, %out: f32):
          %1 = arith.extf %in : bf16 to f32
          %2 = arith.extf %in_9 : bf16 to f32
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %out, %3 : f32
          linalg.yield %4 : f32
      } -> tensor<4x7x8x8xf32>
  return %0 : tensor<4x7x8x8xf32>
}
//      CHECK: func @generic_matmul_bf16bf16f32_odd_m
//      CHECK:  linalg.generic
//  CHECK-NOT:  iree_codegen.ukernel.generic

// -----

// The GEMV pipeline's tile is an unpacked 2-D matmul, which the matvec ukernel
// takes for bf16 (not f32, which has no template). Its zero fill is lowered at
// any size, a whole number of 16-lane vectors (1x32) or not (1x26).
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
module {
  func.func @gemv_tile_bf16(%arg0: tensor<1x256xbf16>, %arg1: tensor<32x256xbf16>,
      %arg2: tensor<1x32xf32>) -> tensor<1x32xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %cst = arith.constant 0.0 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<1x32xf32>) -> tensor<1x32xf32>
    %1 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
        ins(%arg0, %arg1 : tensor<1x256xbf16>, tensor<32x256xbf16>) outs(%0 : tensor<1x32xf32>) -> tensor<1x32xf32>
    return %1 : tensor<1x32xf32>
  }
  func.func @gemv_tile_f32(%arg0: tensor<1x256xf32>, %arg1: tensor<32x256xf32>,
      %arg2: tensor<1x32xf32>) -> tensor<1x32xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
        ins(%arg0, %arg1 : tensor<1x256xf32>, tensor<32x256xf32>) outs(%arg2 : tensor<1x32xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }
  func.func @gemv_tile_k_not_vector(%arg0: tensor<1x96xbf16>, %arg1: tensor<32x96xbf16>,
      %arg2: tensor<1x32xf32>) -> tensor<1x32xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
        ins(%arg0, %arg1 : tensor<1x96xbf16>, tensor<32x96xbf16>) outs(%arg2 : tensor<1x32xf32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }
  func.func @gemv_tile_fill_partial_vector(%arg0: tensor<1x26xf32>) -> tensor<1x26xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %cst = arith.constant 0.0 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<1x26xf32>) -> tensor<1x26xf32>
    return %0 : tensor<1x26xf32>
  }
}
// CHECK-LABEL: func @gemv_tile_bf16(
//       CHECK:   %[[FILL:.+]] = iree_codegen.ukernel.generic "zero_fill_f32_1x32"
//       CHECK:   iree_codegen.ukernel.generic "matvec_bf16_bf16_f32_1x32x256"
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<1x256xbf16>, tensor<32x256xbf16>)
//  CHECK-SAME:     outs(%[[FILL]] : tensor<1x32xf32>)
//  CHECK-SAME:     link_with = "matvec.o"
//  CHECK-SAME:     ukernel_instance = ["matvec_bf16_bf16_f32_1x32x256", "bf16", "bf16", "f32", "32", "256"]
//   CHECK-NOT:   linalg.matmul
// CHECK-LABEL: func @gemv_tile_f32(
//       CHECK:   linalg.matmul
//   CHECK-NOT:   iree_codegen.ukernel.generic
// CHECK-LABEL: func @gemv_tile_k_not_vector(
//       CHECK:   linalg.matmul
//   CHECK-NOT:   iree_codegen.ukernel.generic
// CHECK-LABEL: func @gemv_tile_fill_partial_vector(
//   CHECK-NOT:   linalg.fill
//       CHECK:   iree_codegen.ukernel.generic "zero_fill_f32_1x26"

// -----

// A zero fill that is neither an unpacked 2-D tile nor a packed 4-D/6-D one
// (here 3-D) is left as a linalg.fill.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
module {
  func.func @fill_3d(%arg0: tensor<2x4x16xf32>) -> tensor<2x4x16xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %cst = arith.constant 0.0 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
    return %0 : tensor<2x4x16xf32>
  }
}
// CHECK-LABEL: func @fill_3d(
//       CHECK:   linalg.fill
//   CHECK-NOT:   iree_codegen.ukernel.generic
