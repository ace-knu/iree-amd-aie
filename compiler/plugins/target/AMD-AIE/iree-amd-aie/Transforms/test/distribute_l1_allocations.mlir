// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-distribute-l1-allocations)" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: distribute_l1_memory_test_0

// The L2 allocation becomes private to each thread:
// CHECK: %[[L2ALLOC:.+]] = memref.alloc() : memref<1x1x32x32xi32, 2>

// The linalg.fill acts directly on the private allocation, not a view of the
// shared allocation:
// CHECK: linalg.fill
// CHECK-SAME: outs(%[[L2ALLOC]] : memref<1x1x32x32xi32, 2>)
// CHECK: linalg.fill
// CHECK-SAME: outs(%[[L2ALLOC]] : memref<1x1x32x32xi32, 2>)
// CHECK: memref.dealloc %[[L2ALLOC]] : memref<1x1x32x32xi32, 2>

func.func @distribute_l1_memory_test_0() {
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() : memref<2x2x32x32xi32, 2>
  scf.forall (%arg2, %arg3) in (2, 2) {
    %subview = memref.subview %alloc[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xi32, 2> to memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>)
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  scf.forall (%arg2, %arg3) in (2, 2) {
    %subview = memref.subview %alloc[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xi32, 2> to memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>)
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  memref.dealloc %alloc : memref<2x2x32x32xi32, 2>
  return
}

// -----

// CHECK-LABEL: distribute_l1_memory_nested_loops
// CHECK: %[[L2ALLOC:.+]] = memref.alloc() : memref<1x1x32x32xi32, 2>
// CHECK: linalg.fill
// CHECK-SAME: outs(%[[L2ALLOC]] : memref<1x1x32x32xi32, 2>)
// CHECK: memref.dealloc %[[L2ALLOC]] : memref<1x1x32x32xi32, 2>

#map = affine_map<()[s0, s1] -> (s0 + s1)>
func.func @distribute_l1_memory_nested_loops() {
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() : memref<4x4x32x32xi32, 2>
  scf.forall (%arg0, %arg1) = (0, 0) to (4, 4) step (2, 2) {
    scf.forall (%arg2, %arg3) in (2, 2) {
      %0 = affine.apply #map()[%arg0, %arg2]
      %1 = affine.apply #map()[%arg1, %arg3]
      %subview = memref.subview %alloc[%0, %1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x4x32x32xi32, 2> to memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 2>
      linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x32x32xi32, strided<[4096, 1024, 32, 1], offset: ?>, 2>)
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  memref.dealloc %alloc : memref<4x4x32x32xi32, 2>
  return
}

// -----

// CHECK-LABEL: @transfer_read_test()
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<1x8xbf16, 2>
// CHECK: vector.transfer_read %[[ALLOC]]
// CHECK-SAME: memref<1x8xbf16, 2>, vector<1x8xbf16>

func.func @transfer_read_test(){
  %alloc = memref.alloc() : memref<4x8xbf16, 2>
  scf.forall (%arg0) in (4) {
    %c0 = arith.constant 0 : index
    %c0_bf16 = arith.constant 0.000000e+00 : bf16
    %subview = memref.subview %alloc[%arg0, 0] [1, 8] [1, 1] :
    memref<4x8xbf16, 2> to memref<1x8xbf16, strided<[8, 1], offset: ?>, 2>
    %vector = vector.transfer_read %subview[%c0, %c0], %c0_bf16 {in_bounds = [true, true]} :
    memref<1x8xbf16, strided<[8, 1], offset: ?>, 2>, vector<1x8xbf16>
  } {mapping = [#gpu.thread<x>]}
  return
}

// -----

// CHECK: @transfer_write_test(%[[VECTOR:.+]]: vector<1x8xbf16>)
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<1x8xbf16, 2>
// CHECK: scf.forall
// CHECK: vector.transfer_write %[[VECTOR]], %[[ALLOC]]
// CHECK-SAME: vector<1x8xbf16>, memref<1x8xbf16, 2>

func.func @transfer_write_test(%vector : vector<1x8xbf16>){
  %alloc = memref.alloc() : memref<4x8xbf16, 2>
  scf.forall (%arg0) in (4) {
    %c0 = arith.constant 0 : index
    %c0_bf16 = arith.constant 0.000000e+00 : bf16
    %subview = memref.subview %alloc[%arg0, 0] [1, 8] [1, 1] :
    memref<4x8xbf16, 2> to memref<1x8xbf16, strided<[8, 1], offset: ?>, 2>
    vector.transfer_write %vector, %subview[%c0, %c0] {} : vector<1x8xbf16>, memref<1x8xbf16, strided<[8, 1], offset: ?>, 2>
  } {mapping = [#gpu.thread<x>]}
  return
}

// -----

// CHECK-LABEL: @distribute_l1_memory_for_1x1_case
//      CHECK:   %[[L2ALLOC:.+]] = memref.alloc() : memref<1x1x32x32xi32, 2>
//      CHECK:      linalg.fill
// CHECK-SAME:        outs(%[[L2ALLOC]] : memref<1x1x32x32xi32, 2>)
//      CHECK:      linalg.fill
// CHECK-SAME:        outs(%[[L2ALLOC]] : memref<1x1x32x32xi32, 2>)
func.func @distribute_l1_memory_for_1x1_case() {
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloc() : memref<2x2x32x32xi32, 2>
  scf.forall (%arg0, %arg1) in (2,2) {
    %subview = memref.subview %alloc[%arg0, %arg1, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xi32, 2> to memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>
    scf.forall (%arg2, %arg3) in (1, 1) {
      linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>)
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    scf.forall (%arg2, %arg3) in (1, 1) {
      linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>)
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  }
  memref.dealloc %alloc : memref<2x2x32x32xi32, 2>
  return
}

// -----

// A pack-peel accumulator whose zero-fill sits *outside* the per-thread forall:
// the pack writes the whole undistributed L1 buffer once and each thread reads
// a subview of it. This is what an int8 `batch_matmul_transpose_b` produces,
// because there the fill lands one pack level short and so cannot be fused into
// the forall.
//
// Every element of the packed buffer holds the same value, so the layout
// changes on the way are irrelevant: the fill is re-emitted directly on the
// per-thread allocation, and the pack that fed the shared one is dropped.
// CHECK-LABEL: @distribute_l1_constant_fill_through_pack
// CHECK: %[[PRIV:.+]] = memref.alloc() : memref<1x1x1x4x1x8x8xi32, 2 : i32>
// CHECK-NOT: linalg.pack
// CHECK: scf.forall
// CHECK: linalg.fill
// CHECK-SAME: outs(%[[PRIV]] : memref<1x1x1x4x1x8x8xi32, 2 : i32>)
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[PRIV]] : memref<1x1x1x4x1x8x8xi32, 2 : i32>)
func.func @distribute_l1_constant_fill_through_pack(%out : memref<1x1x1x4x1x8x8xi32, 2 : i32>) {
  %c0_i32 = arith.constant 0 : i32
  %acc_l2 = memref.alloc() : memref<1x1x4x8x32xi32, 1 : i32>
  %acc_copy = memref.alloc() : memref<1x1x4x8x32xi32, 1 : i32>
  %acc_l1 = memref.alloc() : memref<1x1x4x4x1x8x8xi32, 2 : i32>
  linalg.fill ins(%c0_i32 : i32) outs(%acc_l2 : memref<1x1x4x8x32xi32, 1 : i32>)
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%acc_l2 : memref<1x1x4x8x32xi32, 1 : i32>)
    outs(%acc_copy : memref<1x1x4x8x32xi32, 1 : i32>) {
  ^bb0(%in: i32, %out_0: i32):
    linalg.yield %in : i32
  }
  linalg.pack %acc_copy outer_dims_perm = [0, 1, 2, 4, 3] inner_dims_pos = [3, 4]
    inner_tiles = [8, 8] into %acc_l1
    : memref<1x1x4x8x32xi32, 1 : i32> -> memref<1x1x4x4x1x8x8xi32, 2 : i32>
  scf.forall (%arg0, %arg1) in (4, 1) {
    %subview = memref.subview %acc_l1[0, 0, %arg0, 0, 0, 0, 0] [1, 1, 1, 4, 1, 8, 8] [1, 1, 1, 1, 1, 1, 1]
      : memref<1x1x4x4x1x8x8xi32, 2 : i32> to memref<1x1x1x4x1x8x8xi32, strided<[1024, 1024, 256, 64, 64, 8, 1], offset: ?>, 2 : i32>
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>,
                       affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%subview : memref<1x1x1x4x1x8x8xi32, strided<[1024, 1024, 256, 64, 64, 8, 1], offset: ?>, 2 : i32>)
      outs(%out : memref<1x1x1x4x1x8x8xi32, 2 : i32>) {
    ^bb0(%in: i32, %o: i32):
      linalg.yield %in : i32
    }
  } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  memref.dealloc %acc_l1 : memref<1x1x4x4x1x8x8xi32, 2 : i32>
  memref.dealloc %acc_copy : memref<1x1x4x8x32xi32, 1 : i32>
  memref.dealloc %acc_l2 : memref<1x1x4x8x32xi32, 1 : i32>
  return
}
