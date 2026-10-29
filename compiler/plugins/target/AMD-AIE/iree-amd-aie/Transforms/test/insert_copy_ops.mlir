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
// CHECK:     %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG2]][0, 0] [1, 32] [1, 1] : tensor<1x32xbf16> to tensor<1x32xbf16>
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

// The result copy lands on the slice the forall's terminator inserts the
// result at. Here the forall tiles N (dim 1), so the destination slice is
// [0, %iv], not [%iv, 0]: with the wrong dim every block but the first would
// be written outside the output.
// CHECK: func.func @matmul_forall_over_n
// CHECK:   scf.forall (%[[IV:.*]]) = (0) to (1024) step (256) shared_outs(%[[OUT:.*]] = %{{.*}})
// CHECK:     %[[COPYINIT:.*]] = linalg.copy ins(%{{.*}} : tensor<1x256xf32>) outs(%{{.*}} : tensor<1x256xf32>)
// CHECK:     %[[EXTRACT:.*]] = tensor.extract_slice %[[OUT]][0, %[[IV]]] [1, 256] [1, 1] : tensor<1x1024xf32> to tensor<1x256xf32>
// CHECK:     %[[MATMUL:.*]] = linalg.matmul {{.*}} outs(%[[COPYINIT]] : tensor<1x256xf32>)
// CHECK:     linalg.copy ins(%[[MATMUL]] : tensor<1x256xf32>) outs(%[[EXTRACT]] : tensor<1x256xf32>)
func.func @matmul_forall_over_n(%arg0: tensor<1x64xbf16>, %arg1: tensor<1024x64xbf16>) -> tensor<1x1024xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x1024xf32>
  %1 = scf.forall (%arg2) = (0) to (1024) step (256) shared_outs(%arg3 = %0) -> (tensor<1x1024xf32>) {
    %2 = tensor.extract_slice %arg1[%arg2, 0] [256, 64] [1, 1] : tensor<1024x64xbf16> to tensor<256x64xbf16>
    %3 = tensor.extract_slice %arg3[0, %arg2] [1, 256] [1, 1] : tensor<1x1024xf32> to tensor<1x256xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %5 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>]
         ins(%arg0, %2 : tensor<1x64xbf16>, tensor<256x64xbf16>) outs(%4 : tensor<1x256xf32>) -> tensor<1x256xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg3[0, %arg2] [1, 256] [1, 1] : tensor<1x256xf32> into tensor<1x1024xf32>
    }
  } {mapping = [#gpu.block<y>]}
  return %1 : tensor<1x1024xf32>
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

// -----

// An input with no defined contents (a pooling window `tensor.empty`, present
// only to size the reduction loops) is not copied; the data input and the
// result are.
// CHECK-LABEL: func.func @pool_window_not_copied
// CHECK:       %[[WIN:.+]] = tensor.empty() : tensor<2x2xf32>
// CHECK:       %[[IN_ALLOC:.+]] = bufferization.alloc_tensor() : tensor<4x4x8xf32>
// CHECK:       %[[IN:.+]] = linalg.copy ins(%{{.+}} : tensor<4x4x8xf32>) outs(%[[IN_ALLOC]]
// CHECK-NOT:   linalg.copy ins(%[[WIN]]
// CHECK:       %[[OUT_ALLOC:.+]] = bufferization.alloc_tensor() : tensor<2x2x8xf32>
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[IN]], %[[WIN]] :
// CHECK:       linalg.copy
func.func @pool_window_not_copied(%in: tensor<4x4x8xf32>, %init: tensor<2x2x8xf32>) -> tensor<2x2x8xf32> {
  %win = tensor.empty() : tensor<2x2xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%in, %win : tensor<4x4x8xf32>, tensor<2x2xf32>) outs(%init : tensor<2x2x8xf32>) {
  ^bb0(%a: f32, %w: f32, %o: f32):
    %m = arith.maximumf %o, %a : f32
    linalg.yield %m : f32
  } -> tensor<2x2x8xf32>
  return %r : tensor<2x2x8xf32>
}

// -----

// An elementwise op fed by a contraction is its epilogue (GEMV + ReLU): it
// reads the contraction result where it is computed, so neither that result
// nor the epilogue input is copied, its unread init becomes a fresh buffer, and
// only the epilogue result is copied out.
// CHECK-LABEL: func.func @matmul_epilogue_forall_over_n
// CHECK:       scf.forall (%[[IV:.*]]) = (0) to (1024) step (256) shared_outs(%[[OUT:.*]] = %{{.*}})
// CHECK:         %[[MATMUL:.*]] = linalg.matmul
// CHECK-NOT:     linalg.copy ins(%[[MATMUL]]
// CHECK:         %[[EMPTY:.*]] = bufferization.alloc_tensor() : tensor<1x256xbf16>
// CHECK-NOT:     linalg.copy
// CHECK:         %[[EXTRACT:.*]] = tensor.extract_slice %[[OUT]][0, %[[IV]]] [1, 256] [1, 1]
// CHECK:         %[[RELU:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[MATMUL]] : tensor<1x256xf32>) outs(%[[EMPTY]] : tensor<1x256xbf16>)
// CHECK:         %[[COPYOUT:.*]] = linalg.copy ins(%[[RELU]] : tensor<1x256xbf16>) outs(%[[EXTRACT]] : tensor<1x256xbf16>)
// CHECK:         tensor.parallel_insert_slice %[[COPYOUT]] into %[[OUT]][0, %[[IV]]]
func.func @matmul_epilogue_forall_over_n(%arg0: tensor<1x64xbf16>, %arg1: tensor<1024x64xbf16>) -> tensor<1x1024xbf16> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x1024xbf16>
  %1 = scf.forall (%arg2) = (0) to (1024) step (256) shared_outs(%arg3 = %0) -> (tensor<1x1024xbf16>) {
    %2 = tensor.extract_slice %arg1[%arg2, 0] [256, 64] [1, 1] : tensor<1024x64xbf16> to tensor<256x64xbf16>
    %3 = tensor.empty() : tensor<1x256xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %5 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>]
         ins(%arg0, %2 : tensor<1x64xbf16>, tensor<256x64xbf16>) outs(%4 : tensor<1x256xf32>) -> tensor<1x256xf32>
    %6 = tensor.extract_slice %arg3[0, %arg2] [1, 256] [1, 1] : tensor<1x1024xbf16> to tensor<1x256xbf16>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
         ins(%5 : tensor<1x256xf32>) outs(%6 : tensor<1x256xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %8 = arith.maximumf %in, %cst : f32
      %9 = arith.truncf %8 : f32 to bf16
      linalg.yield %9 : bf16
    } -> tensor<1x256xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %7 into %arg3[0, %arg2] [1, 256] [1, 1] : tensor<1x256xbf16> into tensor<1x1024xbf16>
    }
  } {mapping = [#gpu.block<y>]}
  return %1 : tensor<1x1024xbf16>
}
