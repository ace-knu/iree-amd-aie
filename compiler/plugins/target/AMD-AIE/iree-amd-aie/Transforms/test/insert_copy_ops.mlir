// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-insert-copy-ops))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-insert-copy-ops{use-init-as-result-dest=true}))" %s | FileCheck %s --check-prefix=INIT-DEST

// CHECK: func.func @softmax_insert_copy_ops
// CHECK:   %[[FILL:.*]] = linalg.fill {{.*}}) -> tensor<1x32xbf16>
// CHECK:   %[[ALLOC0:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:   %[[COPYIN:.*]] = linalg.copy ins(%arg0 : tensor<1x32xbf16>) outs(%[[ALLOC0]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[ALLOC1:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:   %[[COPYINIT:.*]] = linalg.copy ins(%[[FILL]] : tensor<1x32xbf16>) outs(%[[ALLOC1]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[SOFTMAX:.*]] = linalg.softmax dimension(1) ins(%[[COPYIN]] : tensor<1x32xbf16>) outs(%[[COPYINIT]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
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
// CHECK:     %[[COPYINIT:.*]] = linalg.copy ins(%[[FILL]] : tensor<1x32xbf16>) outs(%[[ALLOC1]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:     %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG2]][%[[ARG1]], 0] [1, 32] [1, 1] : tensor<1x32xbf16> to tensor<1x32xbf16>
// CHECK:     %[[SOFTMAX:.*]] = linalg.softmax dimension(1) ins(%[[COPYIN]] : tensor<1x32xbf16>) outs(%[[COPYINIT]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
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

// A named contraction op is a target too. By default the result is copied to a
// fresh allocation; with `use-init-as-result-dest` it is copied back into the
// tensor the init came from (here the copy standing in for a lower copy level),
// as an unpack would after a pack.
// CHECK: func.func @matmul_second_copy_level
// CHECK:   %[[INIT:.*]] = linalg.copy ins(%arg2 : tensor<1x32xf32>)
// CHECK:   %[[COPYA:.*]] = linalg.copy ins(%arg0 : tensor<1x64xbf16>)
// CHECK:   %[[COPYB:.*]] = linalg.copy ins(%arg1 : tensor<32x64xbf16>)
// CHECK:   %[[COPYINIT:.*]] = linalg.copy ins(%[[INIT]] : tensor<1x32xf32>)
// CHECK:   %[[MATMUL:.*]] = linalg.matmul {{.*}} ins(%[[COPYA]], %[[COPYB]] : {{.*}}) outs(%[[COPYINIT]] : tensor<1x32xf32>)
// CHECK:   %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<1x32xf32>
// CHECK:   %[[COPYOUT:.*]] = linalg.copy ins(%[[MATMUL]] : tensor<1x32xf32>) outs(%[[ALLOC]] : tensor<1x32xf32>)
// CHECK:   return %[[COPYOUT]]
// INIT-DEST: func.func @matmul_second_copy_level
// INIT-DEST:   %[[INIT:.*]] = linalg.copy ins(%arg2 : tensor<1x32xf32>)
// INIT-DEST:   %[[MATMUL:.*]] = linalg.matmul
// INIT-DEST-NOT: bufferization.alloc_tensor
// INIT-DEST:   %[[COPYOUT:.*]] = linalg.copy ins(%[[MATMUL]] : tensor<1x32xf32>) outs(%[[INIT]] : tensor<1x32xf32>)
// INIT-DEST:   return %[[COPYOUT]]
func.func @matmul_second_copy_level(%arg0: tensor<1x64xbf16>, %arg1: tensor<32x64xbf16>, %arg2: tensor<1x32xf32>) -> tensor<1x32xf32> {
  %0 = bufferization.alloc_tensor() : tensor<1x32xf32>
  %1 = linalg.copy ins(%arg2 : tensor<1x32xf32>) outs(%0 : tensor<1x32xf32>) -> tensor<1x32xf32>
  %2 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                      affine_map<(d0, d1, d2) -> (d1, d2)>,
                                      affine_map<(d0, d1, d2) -> (d0, d1)>]
       ins(%arg0, %arg1 : tensor<1x64xbf16>, tensor<32x64xbf16>) outs(%1 : tensor<1x32xf32>) -> tensor<1x32xf32>
  return %2 : tensor<1x32xf32>
}
