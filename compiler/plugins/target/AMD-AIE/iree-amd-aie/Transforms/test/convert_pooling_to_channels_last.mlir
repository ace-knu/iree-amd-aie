// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-convert-pooling-to-channels-last{channel-multiple=4}))' %s | FileCheck %s

// VGG-16 pool1: NCHW 1x64x224x224 -> 1x64x112x112, 2x2 window, stride 2.
// The input is transposed to NHWC, the -inf fill init is rebuilt in NHWC, the
// pooling becomes pooling_nhwc_max with the same strides/dilations, and the
// result is transposed back to NCHW.
// CHECK-LABEL: func.func @pool_nchw_max
// CHECK-SAME:    %[[IN:[a-zA-Z0-9_]+]]: tensor<1x64x224x224xf32>
// CHECK-DAG:     %[[NINF:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:     %[[WIN:.+]] = tensor.empty() : tensor<2x2xf32>
// CHECK:         %[[IN_E:.+]] = tensor.empty() : tensor<1x224x224x64xf32>
// CHECK:         %[[IN_T:.+]] = linalg.transpose ins(%[[IN]] : tensor<1x64x224x224xf32>) outs(%[[IN_E]] : tensor<1x224x224x64xf32>) permutation = [0, 2, 3, 1]
// CHECK:         %[[INIT_E:.+]] = tensor.empty() : tensor<1x112x112x64xf32>
// CHECK:         %[[INIT:.+]] = linalg.fill ins(%[[NINF]] : f32) outs(%[[INIT_E]] : tensor<1x112x112x64xf32>)
// CHECK:         %[[POOL:.+]] = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[IN_T]], %[[WIN]] : tensor<1x224x224x64xf32>, tensor<2x2xf32>) outs(%[[INIT]] : tensor<1x112x112x64xf32>)
// CHECK:         %[[OUT_E:.+]] = tensor.empty() : tensor<1x64x112x112xf32>
// CHECK:         %[[OUT:.+]] = linalg.transpose ins(%[[POOL]] : tensor<1x112x112x64xf32>) outs(%[[OUT_E]] : tensor<1x64x112x112xf32>) permutation = [0, 3, 1, 2]
// CHECK-NOT:     linalg.pooling_nchw_max
// CHECK:         return %[[OUT]]
func.func @pool_nchw_max(%in: tensor<1x64x224x224xf32>) -> tensor<1x64x112x112xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %win = tensor.empty() : tensor<2x2xf32>
  %e = tensor.empty() : tensor<1x64x112x112xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
  %r = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%in, %win : tensor<1x64x224x224xf32>, tensor<2x2xf32>) outs(%f : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
  return %r : tensor<1x64x112x112xf32>
}

// -----

// An init that is not a plain fill is transposed as-is (the pooling then
// accumulates into the transposed init).
// CHECK-LABEL: func.func @pool_nchw_max_generic_init
// CHECK:         linalg.transpose ins(%{{.+}} : tensor<1x8x16x16xf32>) {{.*}} permutation = [0, 2, 3, 1]
// CHECK:         linalg.transpose ins(%{{.+}} : tensor<1x8x8x8xf32>) {{.*}} permutation = [0, 2, 3, 1]
// CHECK:         linalg.pooling_nhwc_max
// CHECK:         linalg.transpose {{.*}} permutation = [0, 3, 1, 2]
// CHECK-NOT:     linalg.pooling_nchw_max
func.func @pool_nchw_max_generic_init(%in: tensor<1x8x16x16xf32>, %init: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %win = tensor.empty() : tensor<2x2xf32>
  %r = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%in, %win : tensor<1x8x16x16xf32>, tensor<2x2xf32>) outs(%init : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}

// -----

// Other pooling flavours and convolutions are left untouched.
// CHECK-LABEL: func.func @untouched
// CHECK:         linalg.pooling_nchw_sum
// CHECK-NOT:     linalg.transpose
func.func @untouched(%in: tensor<1x8x16x16xf32>, %init: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %win = tensor.empty() : tensor<2x2xf32>
  %r = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%in, %win : tensor<1x8x16x16xf32>, tensor<2x2xf32>) outs(%init : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}

// -----

// C not a multiple of channel-multiple (4): the NHWC input is padded 6 -> 8
// channels with the init's fill value, the fill init is rebuilt at 8 channels,
// the pooling runs on 8 and the result is sliced back to 6 before the
// transpose to NCHW. (VGG-16's 64/128/256/512 channels are never padded.)
// CHECK-LABEL: func.func @pool_nchw_max_pad_channels
// CHECK-SAME:    %[[IN:[a-zA-Z0-9_]+]]: tensor<1x6x16x16xf32>
// CHECK-DAG:     %[[NINF:.+]] = arith.constant 0xFF800000 : f32
// CHECK:         %[[IN_T:.+]] = linalg.transpose ins(%[[IN]] : tensor<1x6x16x16xf32>) {{.*}} permutation = [0, 2, 3, 1]
// CHECK:         %[[PAD:.+]] = tensor.pad %[[IN_T]] low[0, 0, 0, 0] high[0, 0, 0, 2]
// CHECK:           tensor.yield %[[NINF]] : f32
// CHECK:         } : tensor<1x16x16x6xf32> to tensor<1x16x16x8xf32>
// CHECK:         %[[INIT_E:.+]] = tensor.empty() : tensor<1x8x8x8xf32>
// CHECK:         %[[INIT:.+]] = linalg.fill ins(%[[NINF]] : f32) outs(%[[INIT_E]] : tensor<1x8x8x8xf32>)
// CHECK:         %[[POOL:.+]] = linalg.pooling_nhwc_max
// CHECK-SAME:      ins(%[[PAD]], %{{.+}} : tensor<1x16x16x8xf32>, tensor<2x2xf32>) outs(%[[INIT]] : tensor<1x8x8x8xf32>)
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[POOL]][0, 0, 0, 0] [1, 8, 8, 6] [1, 1, 1, 1] : tensor<1x8x8x8xf32> to tensor<1x8x8x6xf32>
// CHECK:         linalg.transpose ins(%[[SLICE]] : tensor<1x8x8x6xf32>) {{.*}} permutation = [0, 3, 1, 2]
// CHECK-NOT:     linalg.pooling_nchw_max
func.func @pool_nchw_max_pad_channels(%in: tensor<1x6x16x16xf32>) -> tensor<1x6x8x8xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %win = tensor.empty() : tensor<2x2xf32>
  %e = tensor.empty() : tensor<1x6x8x8xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x6x8x8xf32>) -> tensor<1x6x8x8xf32>
  %r = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%in, %win : tensor<1x6x16x16xf32>, tensor<2x2xf32>) outs(%f : tensor<1x6x8x8xf32>) -> tensor<1x6x8x8xf32>
  return %r : tensor<1x6x8x8xf32>
}
