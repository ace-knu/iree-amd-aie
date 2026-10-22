// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-convert-pooling-to-channels-last{channel-multiple=4 halo-split-columns=8}))' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-convert-pooling-to-channels-last{channel-multiple=4 halo-split-columns=8 columns-policy=cost}))' %s | FileCheck %s --check-prefix=COST
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-amdaie-convert-pooling-to-channels-last{channel-multiple=4 halo-split-columns=8 columns-policy=full}))' %s | FileCheck %s --check-prefix=FULL

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
// With row padding on, OH 112 already divides by the 8 columns: no pad.
// FULL-LABEL: func.func @pool_nchw_max
// FULL-NOT:     tensor.pad
// FULL:         linalg.pooling_nhwc_max {{.*}} outs(%{{.+}} : tensor<1x112x112x64xf32>)
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


// -----

// Overlapping window (3x3/2, the ResNet stem pool on its padded 114x114
// input -> 56x56): under the default `columns-policy=max` OH 56 divides by
// the 8 columns, so the NHWC pooling is pre-split into 8 chunks of 7 output
// rows; chunk i gathers input rows [14i, 14i + 15) (its 7 windows' bounding
// box, 1 halo row shared with the next chunk), the chunks are stacked on a
// leading dim, the pooling runs batched over them and the result is reshaped
// back to [1, 56, 56, 64] before the NCHW transpose.
// CHECK-LABEL: func.func @pool_nchw_max_3x3s2_halo
// CHECK:         %[[IN_T:.+]] = linalg.transpose ins(%{{.+}} : tensor<1x64x114x114xf32>) {{.*}} permutation = [0, 2, 3, 1]
// CHECK-DAG:     %[[C0:.+]] = tensor.extract_slice %[[IN_T]][0, 0, 0, 0] [1, 15, 114, 64] [1, 1, 1, 1] : tensor<1x114x114x64xf32> to tensor<1x15x114x64xf32>
// CHECK-DAG:     %[[C1:.+]] = tensor.extract_slice %[[IN_T]][0, 14, 0, 0] [1, 15, 114, 64] [1, 1, 1, 1]
// CHECK-DAG:     %[[C7:.+]] = tensor.extract_slice %[[IN_T]][0, 98, 0, 0] [1, 15, 114, 64] [1, 1, 1, 1]
// CHECK:         %[[STACK:.+]] = tensor.concat dim(0) %[[C0]], %[[C1]], %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[C7]] : {{.*}} -> tensor<8x15x114x64xf32>
// CHECK:         %[[INIT:.+]] = linalg.fill {{.*}} -> tensor<8x7x56x64xf32>
// CHECK:         %[[POOL:.+]] = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[STACK]], %{{.+}} : tensor<8x15x114x64xf32>, tensor<3x3xf32>) outs(%[[INIT]] : tensor<8x7x56x64xf32>)
// CHECK:         %[[FLAT:.+]] = tensor.collapse_shape %[[POOL]] {{\[}}[0, 1], [2], [3]] : tensor<8x7x56x64xf32> into tensor<56x56x64xf32>
// CHECK:         %[[RES:.+]] = tensor.expand_shape %[[FLAT]] {{\[}}[0, 1], [2], [3]] output_shape [1, 56, 56, 64] : tensor<56x56x64xf32> into tensor<1x56x56x64xf32>
// CHECK:         linalg.transpose ins(%[[RES]] : tensor<1x56x56x64xf32>) {{.*}} permutation = [0, 3, 1, 2]
// CHECK-NOT:     linalg.pooling_nchw_max
// cost: 7 chunks of 8 output rows (the cost model picks 7 columns for this
// shape), each 7*2+3 = 17 input rows, chunk i from row 16*i.
// COST-LABEL: func.func @pool_nchw_max_3x3s2_halo
// COST-DAG:     tensor.extract_slice %{{.+}}[0, 0, 0, 0] [1, 17, 114, 64] [1, 1, 1, 1]
// COST-DAG:     tensor.extract_slice %{{.+}}[0, 16, 0, 0] [1, 17, 114, 64] [1, 1, 1, 1]
// COST-DAG:     tensor.extract_slice %{{.+}}[0, 96, 0, 0] [1, 17, 114, 64] [1, 1, 1, 1]
// COST:         tensor.concat dim(0) {{.*}} -> tensor<7x17x114x64xf32>
// COST:         linalg.pooling_nhwc_max {{.*}} outs(%{{.+}} : tensor<7x8x56x64xf32>)
// COST:         tensor.collapse_shape {{.*}} : tensor<7x8x56x64xf32> into tensor<56x56x64xf32>
// full: 8 chunks of 7 output rows, as the default (56 divides by 8, no row
// padding).
// FULL-LABEL: func.func @pool_nchw_max_3x3s2_halo
// FULL-DAG:     tensor.extract_slice %{{.+}}[0, 0, 0, 0] [1, 15, 114, 64] [1, 1, 1, 1]
// FULL-DAG:     tensor.extract_slice %{{.+}}[0, 14, 0, 0] [1, 15, 114, 64] [1, 1, 1, 1]
// FULL-DAG:     tensor.extract_slice %{{.+}}[0, 98, 0, 0] [1, 15, 114, 64] [1, 1, 1, 1]
// FULL:         tensor.concat dim(0) {{.*}} -> tensor<8x15x114x64xf32>
// FULL:         linalg.pooling_nhwc_max {{.*}} outs(%{{.+}} : tensor<8x7x56x64xf32>)
// FULL:         tensor.collapse_shape {{.*}} : tensor<8x7x56x64xf32> into tensor<56x56x64xf32>
func.func @pool_nchw_max_3x3s2_halo(%in: tensor<1x64x114x114xf32>) -> tensor<1x64x56x56xf32> {
  %ninf = arith.constant 0xFF800000 : f32
  %win = tensor.empty() : tensor<3x3xf32>
  %e = tensor.empty() : tensor<1x64x56x56xf32>
  %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %r = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
         ins(%in, %win : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%f : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %r : tensor<1x64x56x56xf32>
}

// -----

// OH not a multiple of the column count (VGG-16 pool3: 56x56 -> 28x28, C 8):
// the preprocessing never pads OH, under any policy. Row padding for the
// column policy is done at the Flow dispatch boundary by
// AMDAIEPadPoolingDispatches (see pad_pooling_dispatches.mlir), where the
// dispatch's init fill is already inside the dispatch and can grow in place.
// CHECK-LABEL: func.func @pool_nchw_max_pad_rows
// CHECK-NOT:     tensor.pad
// CHECK:         linalg.pooling_nhwc_max {{.*}} outs(%{{.+}} : tensor<1x28x28x8xf32>)
// CHECK-NOT:     tensor.pad
// COST-LABEL: func.func @pool_nchw_max_pad_rows
// COST-NOT:     tensor.pad
// COST:         linalg.pooling_nhwc_max {{.*}} outs(%{{.+}} : tensor<1x28x28x8xf32>)
// COST-NOT:     tensor.pad
// FULL-LABEL: func.func @pool_nchw_max_pad_rows
// FULL-NOT:     tensor.pad
// FULL:         linalg.pooling_nhwc_max {{.*}} outs(%{{.+}} : tensor<1x28x28x8xf32>)
// FULL-NOT:     tensor.pad
func.func @pool_nchw_max_pad_rows(%in: tensor<1x8x56x56xf32>) -> tensor<1x8x28x28xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %win = tensor.empty() : tensor<2x2xf32>
  %e = tensor.empty() : tensor<1x8x28x28xf32>
  %init = linalg.fill ins(%cst : f32) outs(%e : tensor<1x8x28x28xf32>) -> tensor<1x8x28x28xf32>
  %0 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%in, %win : tensor<1x8x56x56xf32>, tensor<2x2xf32>) outs(%init : tensor<1x8x28x28xf32>) -> tensor<1x8x28x28xf32>
  return %0 : tensor<1x8x28x28xf32>
}
