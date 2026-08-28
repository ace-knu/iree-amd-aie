// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-fold-bias-into-contraction)' %s | FileCheck %s

// A conv2d (NCHW/FCHW: no channels-last conversion) whose accumulator is a
// `linalg.broadcast` of a 1-D bias is folded into a single conv with one
// extra input channel: the channel holds the constant 1.0 across the whole
// (already-padded) input, and the filter gets a matching extra input-channel
// plane that is zero except a `bias` tap at (kh=0, kw=0). No `linalg.broadcast`
// survives.

// CHECK-LABEL: @conv_bias_nchw
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x3x18x18xf32>
// CHECK-SAME:    %[[FILT:[a-zA-Z0-9]+]]: tensor<8x3x3x3xf32>
// CHECK-SAME:    %[[BIAS:[a-zA-Z0-9]+]]: tensor<8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: %[[IN_EMPTY:.+]] = tensor.empty() : tensor<1x4x18x18xf32>
// CHECK: %[[IN_INS:.+]] = tensor.insert_slice %[[IN]] into %[[IN_EMPTY]][0, 0, 0, 0] [1, 3, 18, 18] [1, 1, 1, 1]
// CHECK: %[[ONES:.+]] = linalg.fill
// CHECK: %[[IN_AUG:.+]] = tensor.insert_slice %[[ONES]] into %[[IN_INS]][0, 3, 0, 0] [1, 1, 18, 18] [1, 1, 1, 1]
// CHECK: %[[FILT_EMPTY:.+]] = tensor.empty() : tensor<8x4x3x3xf32>
// CHECK: %[[FILT_INS:.+]] = tensor.insert_slice %[[FILT]] into %[[FILT_EMPTY]][0, 0, 0, 0] [8, 3, 3, 3] [1, 1, 1, 1]
// CHECK: %[[TAP_ZERO:.+]] = linalg.fill
// CHECK: %[[BIAS_4D:.+]] = tensor.expand_shape %[[BIAS]] {{.*}} output_shape [8, 1, 1, 1]
// CHECK: %[[TAP:.+]] = tensor.insert_slice %[[BIAS_4D]] into %[[TAP_ZERO]][0, 0, 0, 0] [8, 1, 1, 1] [1, 1, 1, 1]
// CHECK: %[[FILT_AUG:.+]] = tensor.insert_slice %[[TAP]] into %[[FILT_INS]][0, 3, 0, 0] [8, 1, 3, 3] [1, 1, 1, 1]
// CHECK: %[[ZERO_FILL:.+]] = linalg.fill
// CHECK: %[[RESULT:.+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[IN_AUG]], %[[FILT_AUG]] : tensor<1x4x18x18xf32>, tensor<8x4x3x3xf32>) outs(%[[ZERO_FILL]] : tensor<1x8x16x16xf32>)
// CHECK: return %[[RESULT]]
func.func @conv_bias_nchw(%input: tensor<1x3x18x18xf32>, %filter: tensor<8x3x3x3xf32>,
                          %bias: tensor<8xf32>) -> tensor<1x8x16x16xf32> {
  %empty = tensor.empty() : tensor<1x8x16x16xf32>
  %broadcasted = linalg.broadcast ins(%bias : tensor<8xf32>) outs(%empty : tensor<1x8x16x16xf32>) dimensions = [0, 2, 3]
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x3x18x18xf32>, tensor<8x3x3x3xf32>)
      outs(%broadcasted : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
  return %result : tensor<1x8x16x16xf32>
}

// -----

// The channels-last form: `--iree-preprocessing-convert-conv-to-channels-last`
// (part of VGG16's actual recipe) runs *before* this pass (Preprocessing's
// command-line pipeline runs before the plugin's own additions -- see file
// header), so by the time this pass sees VGG16's real convs they are already
// `linalg.conv_2d_nhwc_hwcf`, and the bias broadcast is wrapped in a
// `linalg.transpose` (NCHW-shaped broadcast -> NHWC-shaped accumulator).
// Folded the same way, resolved back through the transpose.

// CHECK-LABEL: @conv_bias_nhwc
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]: tensor<1x18x18x3xf32>
// CHECK-SAME:    %[[FILT:[a-zA-Z0-9]+]]: tensor<3x3x3x8xf32>
// CHECK-SAME:    %[[BIAS:[a-zA-Z0-9]+]]: tensor<8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK-NOT: linalg.transpose
// CHECK: %[[IN_EMPTY:.+]] = tensor.empty() : tensor<1x18x18x4xf32>
// CHECK: %[[IN_INS:.+]] = tensor.insert_slice %[[IN]] into %[[IN_EMPTY]][0, 0, 0, 0] [1, 18, 18, 3] [1, 1, 1, 1]
// CHECK: %[[ONES:.+]] = linalg.fill
// CHECK: %[[IN_AUG:.+]] = tensor.insert_slice %[[ONES]] into %[[IN_INS]][0, 0, 0, 3] [1, 18, 18, 1] [1, 1, 1, 1]
// CHECK: %[[FILT_EMPTY:.+]] = tensor.empty() : tensor<3x3x4x8xf32>
// CHECK: %[[FILT_INS:.+]] = tensor.insert_slice %[[FILT]] into %[[FILT_EMPTY]][0, 0, 0, 0] [3, 3, 3, 8] [1, 1, 1, 1]
// CHECK: %[[TAP_ZERO:.+]] = linalg.fill
// CHECK: %[[BIAS_4D:.+]] = tensor.expand_shape %[[BIAS]] {{.*}} output_shape [1, 1, 1, 8]
// CHECK: %[[TAP:.+]] = tensor.insert_slice %[[BIAS_4D]] into %[[TAP_ZERO]][0, 0, 0, 0] [1, 1, 1, 8] [1, 1, 1, 1]
// CHECK: %[[FILT_AUG:.+]] = tensor.insert_slice %[[TAP]] into %[[FILT_INS]][0, 0, 3, 0] [3, 3, 1, 8] [1, 1, 1, 1]
// CHECK: %[[ZERO_FILL:.+]] = linalg.fill
// CHECK: %[[RESULT:.+]] = linalg.conv_2d_nhwc_hwcf {{.*}} ins(%[[IN_AUG]], %[[FILT_AUG]] : tensor<1x18x18x4xf32>, tensor<3x3x4x8xf32>) outs(%[[ZERO_FILL]] : tensor<1x16x16x8xf32>)
// CHECK: return %[[RESULT]]
func.func @conv_bias_nhwc(%input: tensor<1x18x18x3xf32>, %filter: tensor<3x3x3x8xf32>,
                          %bias: tensor<8xf32>) -> tensor<1x16x16x8xf32> {
  %empty_nchw = tensor.empty() : tensor<1x8x16x16xf32>
  %broadcasted = linalg.broadcast ins(%bias : tensor<8xf32>) outs(%empty_nchw : tensor<1x8x16x16xf32>) dimensions = [0, 2, 3]
  %empty_nhwc = tensor.empty() : tensor<1x16x16x8xf32>
  %transposed = linalg.transpose ins(%broadcasted : tensor<1x8x16x16xf32>) outs(%empty_nhwc : tensor<1x16x16x8xf32>) permutation = [0, 2, 3, 1]
  %result = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x18x18x3xf32>, tensor<3x3x3x8xf32>)
      outs(%transposed : tensor<1x16x16x8xf32>) -> tensor<1x16x16x8xf32>
  return %result : tensor<1x16x16x8xf32>
}

// -----

// A conv2d whose accumulator is a plain fill(0) (no bias) must be left
// completely untouched: the pass only matches a broadcast accumulator.

// CHECK-LABEL: @conv_only
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[RESULT:.+]] = linalg.conv_2d_nchw_fchw
// CHECK-SAME: outs(%[[FILL]]
// CHECK-NOT: tensor.insert_slice
// CHECK: return %[[RESULT]]
func.func @conv_only(%input: tensor<1x3x18x18xf32>, %filter: tensor<8x3x3x3xf32>) -> tensor<1x8x16x16xf32> {
  %empty = tensor.empty() : tensor<1x8x16x16xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
  %result = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%input, %filter : tensor<1x3x18x18xf32>, tensor<8x3x3x3xf32>)
      outs(%fill : tensor<1x8x16x16xf32>) -> tensor<1x8x16x16xf32>
  return %result : tensor<1x8x16x16xf32>
}

// A matmul whose accumulator is a broadcast of a 1-D bias is folded into a
// single K+1 matmul: bias becomes an extra reduction row of rhs, and lhs gets
// a matching extra column of the constant 1.0. No bias operand, and no
// separate bias-add, survives.

// CHECK-LABEL: @matmul_bias
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<64x128xf32>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[BIAS:[a-zA-Z0-9]+]]: tensor<128xf32>
// CHECK-NOT: linalg.generic
// CHECK: %[[RHS_EMPTY:.+]] = tensor.empty() : tensor<129x128xf32>
// CHECK: %[[RHS_INS:.+]] = tensor.insert_slice %[[RHS]] into %[[RHS_EMPTY]][0, 0] [128, 128] [1, 1]
// CHECK: %[[BIAS_2D:.+]] = tensor.expand_shape %[[BIAS]] {{.*}} output_shape [1, 128]
// CHECK: %[[RHS_AUG:.+]] = tensor.insert_slice %[[BIAS_2D]] into %[[RHS_INS]][128, 0] [1, 128] [1, 1]
// CHECK: %[[LHS_EMPTY:.+]] = tensor.empty() : tensor<64x129xf32>
// CHECK: %[[LHS_INS:.+]] = tensor.insert_slice %[[LHS]] into %[[LHS_EMPTY]][0, 0] [64, 128] [1, 1]
// CHECK: %[[ONES:.+]] = linalg.fill
// CHECK: %[[LHS_AUG:.+]] = tensor.insert_slice %[[ONES]] into %[[LHS_INS]][0, 128] [64, 1] [1, 1]
// CHECK: %[[ZERO_FILL:.+]] = linalg.fill
// CHECK: %[[RESULT:.+]] = linalg.matmul ins(%[[LHS_AUG]], %[[RHS_AUG]] : tensor<64x129xf32>, tensor<129x128xf32>) outs(%[[ZERO_FILL]] : tensor<64x128xf32>)
// CHECK: return %[[RESULT]]
func.func @matmul_bias(%lhs: tensor<64x128xf32>, %rhs: tensor<128x128xf32>,
                        %bias: tensor<128xf32>) -> tensor<64x128xf32> {
  %empty = tensor.empty() : tensor<64x128xf32>
  %bias_bcast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias : tensor<128xf32>) outs(%empty : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<64x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x128xf32>)
                          outs(%bias_bcast : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}

// -----

// The other source shape: a plain fill(0)-accumulated matmul immediately
// followed by a separate bias-add generic (what a direct "y = x @ W + b"
// lowering, e.g. onnx.Gemm via torch-to-linalg, produces without any detach
// pass involved). Folded the same way as the accumulator form.

// CHECK-LABEL: @matmul_post_add_bias
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<64x128xf32>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[BIAS:[a-zA-Z0-9]+]]: tensor<128xf32>
// CHECK-NOT: linalg.generic
// CHECK: tensor.insert_slice %[[RHS]]
// CHECK: tensor.expand_shape %[[BIAS]]
// CHECK: tensor.insert_slice %[[LHS]]
// CHECK: %[[RESULT:.+]] = linalg.matmul
// CHECK: return %[[RESULT]]
func.func @matmul_post_add_bias(%lhs: tensor<64x128xf32>, %rhs: tensor<128x128xf32>,
                                 %bias: tensor<128xf32>) -> tensor<64x128xf32> {
  %empty = tensor.empty() : tensor<64x128xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
  %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x128xf32>)
                      outs(%fill : tensor<64x128xf32>) -> tensor<64x128xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%mm, %bias : tensor<64x128xf32>, tensor<128xf32>) outs(%empty : tensor<64x128xf32>) {
    ^bb0(%m: f32, %b: f32, %out: f32):
      %sum = arith.addf %m, %b : f32
      linalg.yield %sum : f32
  } -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}

// -----

// A plain fill(0)-initialized matmul (no bias) must be left completely
// untouched: the pass only matches a broadcast-generic accumulator.

// CHECK-LABEL: @matmul_only
// CHECK: %[[FILL:.+]] = linalg.fill
// CHECK: %[[RESULT:.+]] = linalg.matmul
// CHECK-SAME: outs(%[[FILL]]
// CHECK-NOT: tensor.insert_slice
// CHECK: return %[[RESULT]]
func.func @matmul_only(%lhs: tensor<64x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<64x128xf32> {
  %empty = tensor.empty() : tensor<64x128xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x128xf32>)
                          outs(%fill : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}

// -----

// A bias broadcast over the M (row) dim rather than N (column) is not the
// pattern this pass targets, and must be left untouched.

// CHECK-LABEL: @matmul_row_bias
// CHECK: linalg.generic
// CHECK: %[[RESULT:.+]] = linalg.matmul
// CHECK-NOT: tensor.insert_slice
// CHECK: return %[[RESULT]]
func.func @matmul_row_bias(%lhs: tensor<64x128xf32>, %rhs: tensor<128x128xf32>,
                            %bias: tensor<64xf32>) -> tensor<64x128xf32> {
  %empty = tensor.empty() : tensor<64x128xf32>
  %bias_bcast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias : tensor<64xf32>) outs(%empty : tensor<64x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<64x128xf32>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x128xf32>)
                          outs(%bias_bcast : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}
