// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-insert-copy-ops))" %s | FileCheck %s

// `linalg.softmax` overwrites its destination, so the destination only gets a
// buffer -- the contents are not copied in, and the `linalg.fill` feeding it is
// left behind unused.
// CHECK: func.func @softmax_insert_copy_ops
// CHECK:   %[[FILL:.*]] = linalg.fill {{.*}}) -> tensor<1x32xbf16>
// CHECK:   %[[ALLOC0:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:   %[[COPYIN:.*]] = linalg.copy ins(%arg0 : tensor<1x32xbf16>) outs(%[[ALLOC0]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[ALLOC1:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK-NOT:   linalg.copy ins(%[[FILL]]
// CHECK:   %[[SOFTMAX:.*]] = linalg.softmax dimension(1) ins(%[[COPYIN]] : tensor<1x32xbf16>) outs(%[[ALLOC1]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[ALLOC2:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:   %[[COPYOUT:.*]] = linalg.copy ins(%[[SOFTMAX]] : tensor<1x32xbf16>) outs(%[[ALLOC2]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   return %[[COPYOUT]] : tensor<1x32xbf16>
func.func @softmax_insert_copy_ops(%in0: tensor<1x32xbf16>) -> tensor<1x32xbf16> {
  %cst = arith.constant 0.0 : bf16
  %0 = tensor.empty() : tensor<1x32xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 :tensor<1x32xbf16>) -> tensor<1x32xbf16>
  %2 = linalg.softmax dimension(1) ins(%in0 : tensor<1x32xbf16>) outs(%1 : tensor<1x32xbf16>) -> tensor<1x32xbf16>
  return %2 : tensor<1x32xbf16>
}

// -----

// CHECK: func.func @softmax_forall
// CHECK:   %[[FORALL:.*]] = scf.forall (%[[ARG1:.*]]) in (1) shared_outs(%[[ARG2:.*]] = %{{.*}}) -> (tensor<1x32xbf16>) {
// CHECK:     %[[FILL:.*]] = linalg.fill {{.*}} -> tensor<1x32xbf16>
// CHECK:     %[[ALLOC0:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:     %[[COPYIN:.*]] = linalg.copy ins(%arg0 : tensor<1x32xbf16>) outs(%[[ALLOC0]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:     %[[ALLOC1:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK-NOT:     linalg.copy ins(%[[FILL]]
// CHECK:     %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG2]][%[[ARG1]], 0] [1, 32] [1, 1] : tensor<1x32xbf16> to tensor<1x32xbf16>
// CHECK:     %[[SOFTMAX:.*]] = linalg.softmax dimension(1) ins(%[[COPYIN]] : tensor<1x32xbf16>) outs(%[[ALLOC1]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:     %[[COPYOUT:.*]] = linalg.copy ins(%[[SOFTMAX]] : tensor<1x32xbf16>) outs(%[[EXTRACT]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   } {mapping = [#gpu.block<y>]}
// CHECK:   return %[[FORALL]] : tensor<1x32xbf16>
func.func @softmax_forall(%arg0: tensor<1x32xbf16>) -> tensor<1x32xbf16> {
  %cst = arith.constant 0.0 : bf16
  %0 = tensor.empty() : tensor<1x32xbf16>
  %1 = scf.forall (%arg1) in (1) shared_outs(%arg2 = %0) -> (tensor<1x32xbf16>) {
    %2 = linalg.fill ins(%cst : bf16) outs(%arg2 : tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %3 = linalg.softmax dimension(1) ins(%arg0 : tensor<1x32xbf16>) outs(%2 : tensor<1x32xbf16>) -> tensor<1x32xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg2[0, 0] [1, 32] [1, 1] : tensor<1x32xbf16> into tensor<1x32xbf16>
    }
  } {mapping = [#gpu.block<y>]}
  return %1 : tensor<1x32xbf16>
}

// -----

// CHECK: func.func @generic_insert_copy_ops
// CHECK:   %[[FILL:.*]] = linalg.fill {{.*}} -> tensor<128xbf16>
// CHECK:   %[[ALLOC0:.*]] = bufferization.alloc_tensor() : tensor<128x128xbf16>
// CHECK:   %[[COPYIN:.*]] = linalg.copy ins(%arg0 : tensor<128x128xbf16>) outs(%[[ALLOC0]] : tensor<128x128xbf16>) -> tensor<128x128xbf16>
// CHECK:   %[[ALLOC1:.*]] = bufferization.alloc_tensor() : tensor<128xbf16>
// CHECK:   %[[COPYINIT:.*]] = linalg.copy ins(%[[FILL]] : tensor<128xbf16>) outs(%[[ALLOC1]] : tensor<128xbf16>) -> tensor<128xbf16>
// CHECK:   %[[GENERIC:.*]] = linalg.generic {{.*}} ins(%[[COPYIN]] : tensor<128x128xbf16>) outs(%[[COPYINIT]] : tensor<128xbf16>)
// CHECK:   %[[ALLOC2:.*]] = bufferization.alloc_tensor() : tensor<128xbf16>
// CHECK:   %[[COPYOUT:.*]] = linalg.copy ins(%[[GENERIC]] : tensor<128xbf16>) outs(%[[ALLOC2]] : tensor<128xbf16>) -> tensor<128xbf16>
// CHECK:   return %[[COPYOUT]] : tensor<128xbf16>
func.func @generic_insert_copy_ops(%arg0: tensor<128x128xbf16>) -> tensor<128xbf16> {
  %cst = arith.constant 0.0 : bf16
  %3 = tensor.empty() : tensor<128xbf16>
  %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<128xbf16>) -> tensor<128xbf16>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<128x128xbf16>) outs(%4 : tensor<128xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %6 = arith.addf %in, %out : bf16
    linalg.yield %6 : bf16
  } -> tensor<128xbf16>
  return %5 : tensor<128xbf16>
}

// -----

// An elementwise generic ignores the value already in its destination, so that
// destination gets a buffer but no copy-in -- unlike the reduction above, whose
// body reads `%out`.
// CHECK: func.func @elementwise_destination_not_copied_in
// CHECK:   %[[ALLOC0:.*]] = bufferization.alloc_tensor() : tensor<128xi8>
// CHECK:   %[[COPYIN:.*]] = linalg.copy ins(%arg0 : tensor<128xi8>) outs(%[[ALLOC0]] : tensor<128xi8>) -> tensor<128xi8>
// CHECK:   %[[ALLOC1:.*]] = bufferization.alloc_tensor() : tensor<128xi32>
// CHECK-NOT:   linalg.copy ins(%arg1
// CHECK:   %[[GENERIC:.*]] = linalg.generic {{.*}} ins(%[[COPYIN]] : tensor<128xi8>) outs(%[[ALLOC1]] : tensor<128xi32>)
#map2 = affine_map<(d0) -> (d0)>
func.func @elementwise_destination_not_copied_in(%arg0: tensor<128xi8>, %arg1: tensor<128xi32>) -> tensor<128xi32> {
  %0 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%arg0 : tensor<128xi8>) outs(%arg1 : tensor<128xi32>) {
  ^bb0(%in: i8, %out: i32):
    %1 = arith.extsi %in : i8 to i32
    linalg.yield %1 : i32
  } -> tensor<128xi32>
  return %0 : tensor<128xi32>
}
