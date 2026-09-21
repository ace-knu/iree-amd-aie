// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-cleanup))' --split-input-file %s | FileCheck %s

// A broadcast whose only new dimension has extent 1 moves no data, so it is a
// reshape. Keeping it a compute op costs the consumer its streaming schedule.

// CHECK-LABEL: func.func @unit_broadcast_is_a_reshape
//       CHECK:   %[[E:.+]] = tensor.expand_shape %arg0 {{\[}}[0, 1], [2]] output_shape [1, 32, 832]
//   CHECK-NOT:   linalg.generic
//       CHECK:   return %[[E]]
func.func @unit_broadcast_is_a_reshape(%arg0: tensor<32x832xi8>) -> tensor<1x32x832xi8> {
  %0 = tensor.empty() : tensor<1x32x832xi8>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%arg0 : tensor<32x832xi8>) outs(%0 : tensor<1x32x832xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<1x32x832xi8>
  return %1 : tensor<1x32x832xi8>
}

// -----

// A broadcast that really replicates data has to stay a copy.

// CHECK-LABEL: func.func @real_broadcast_is_kept
//       CHECK:   linalg.generic
//   CHECK-NOT:   tensor.expand_shape
func.func @real_broadcast_is_kept(%arg0: tensor<32x832xi8>) -> tensor<12x32x832xi8> {
  %0 = tensor.empty() : tensor<12x32x832xi8>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%arg0 : tensor<32x832xi8>) outs(%0 : tensor<12x32x832xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<12x32x832xi8>
  return %1 : tensor<12x32x832xi8>
}

// -----

// A unit-extent broadcast that also transposes is not a reshape.

// CHECK-LABEL: func.func @transposing_broadcast_is_kept
//       CHECK:   linalg.generic
//   CHECK-NOT:   tensor.expand_shape
func.func @transposing_broadcast_is_kept(%arg0: tensor<832x32xi8>) -> tensor<1x32x832xi8> {
  %0 = tensor.empty() : tensor<1x32x832xi8>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%arg0 : tensor<832x32xi8>) outs(%0 : tensor<1x32x832xi8>) {
  ^bb0(%in: i8, %out: i8):
    linalg.yield %in : i8
  } -> tensor<1x32x832xi8>
  return %1 : tensor<1x32x832xi8>
}

// -----

// A body that computes something is not a copy, even with a unit broadcast.

// CHECK-LABEL: func.func @computing_body_is_kept
//       CHECK:   linalg.generic
//   CHECK-NOT:   tensor.expand_shape
func.func @computing_body_is_kept(%arg0: tensor<32x832xi8>) -> tensor<1x32x832xi8> {
  %c1 = arith.constant 1 : i8
  %0 = tensor.empty() : tensor<1x32x832xi8>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%arg0 : tensor<32x832xi8>) outs(%0 : tensor<1x32x832xi8>) {
  ^bb0(%in: i8, %out: i8):
    %2 = arith.addi %in, %c1 : i8
    linalg.yield %2 : i8
  } -> tensor<1x32x832xi8>
  return %1 : tensor<1x32x832xi8>
}
