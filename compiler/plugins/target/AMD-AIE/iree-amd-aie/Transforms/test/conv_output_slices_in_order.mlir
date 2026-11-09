// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-conv-output-slices-in-order))" %s | FileCheck %s

// Chunked conv output [2 chunks, 2 rows, 4 cols, 8 channels] in L2, each core
// writing one chunk's 4 channels (2 cores per chunk). The cores' tiles become
// contiguous 32-element slices (slice = channel offset / 4, i.e. the channel
// offset steps 8 = rows * cols elements), and the write-out reads the slices in
// order and scatters them into the NHWC L3 tile (the slice dim steps 4
// channels).

// CHECK-LABEL: func.func @chunked_output
// CHECK:         scf.forall (%[[N:.+]], %[[OC:.+]]) = (0, 0) to (2, 8) step (1, 4)
// CHECK:           amdaie.dma_cpy_nd(%{{.+}}[%[[N]], %[[OC]], 0, 0, 0] [1, 1, 2, 4, 4] [64, 8, 16, 4, 1], %{{.+}}[0, 0, 0, 0] [1, 2, 4, 4] [32, 16, 4, 1])
// CHECK:         amdaie.dma_cpy_nd(%{{.+}}[0, 0, 0, 0, 0] [2, 2, 2, 4, 4] [64, 4, 32, 8, 1], %{{.+}}[0, 0, 0, 0, 0] [2, 2, 2, 4, 4] [64, 32, 16, 4, 1])
#translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "conv"}>
func.func @chunked_output(%arg0: memref<2x2x4x8xbf16>) attributes {translation_info = #translation} {
  %l1 = memref.alloc() : memref<1x2x4x4xbf16, 2 : i32>
  %l2 = memref.alloc() : memref<2x2x4x8xbf16, 1 : i32>
  %lof_l1 = amdaie.logicalobjectfifo.from_memref %l1, {} : memref<1x2x4x4xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x4x4xbf16, 2 : i32>>
  %lof_l2w = amdaie.logicalobjectfifo.from_memref %l2, {} : memref<2x2x4x8xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x4x8xbf16, 1 : i32>>
  %lof_l2r = amdaie.logicalobjectfifo.from_memref %l2, {} : memref<2x2x4x8xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x4x8xbf16, 1 : i32>>
  %lof_l3 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<2x2x4x8xbf16> -> !amdaie.logicalobjectfifo<memref<2x2x4x8xbf16>>
  scf.forall (%n, %oc) = (0, 0) to (2, 8) step (1, 4) {
    %0 = amdaie.dma_cpy_nd(%lof_l2w[%n, 0, 0, %oc] [1, 2, 4, 4] [64, 32, 8, 1], %lof_l1[0, 0, 0, 0] [1, 2, 4, 4] [32, 16, 4, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x4x8xbf16, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x4x4xbf16, 2 : i32>>)
  } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
  %1 = amdaie.dma_cpy_nd(%lof_l3[0, 0, 0, 0] [2, 2, 4, 8] [64, 32, 8, 1], %lof_l2r[0, 0, 0, 0] [2, 2, 4, 8] [64, 32, 8, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x4x8xbf16>>, !amdaie.logicalobjectfifo<memref<2x2x4x8xbf16, 1 : i32>>)
  return
}

// -----

// A batch-folded conv output [2 rows, 4 cols, 8 channels] written by the cores
// of one column into L3 at an offset (the block of a larger output): no chunk
// dim, and the L3 offsets stay on their dims.

// CHECK-LABEL: func.func @unchunked_output
// CHECK:         scf.forall (%[[OC:.+]]) = (0) to (8) step (4)
// CHECK:           amdaie.dma_cpy_nd(%{{.+}}[%[[OC]], 0, 0, 0] [1, 2, 4, 4] [8, 16, 4, 1], %{{.+}}[0, 0, 0] [2, 4, 4] [16, 4, 1])
// CHECK:         amdaie.dma_cpy_nd(%{{.+}}[0, %[[OH:.+]], 0, 8] [2, 2, 4, 4] [4, 128, 16, 1], %{{.+}}[0, 0, 0, 0] [2, 2, 4, 4] [32, 16, 4, 1])
#translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "conv"}>
func.func @unchunked_output(%arg0: memref<8x8x16xbf16>, %oh: index) attributes {translation_info = #translation} {
  %l1 = memref.alloc() : memref<2x4x4xbf16, 2 : i32>
  %l2 = memref.alloc() : memref<2x4x8xbf16, 1 : i32>
  %lof_l1 = amdaie.logicalobjectfifo.from_memref %l1, {} : memref<2x4x4xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<2x4x4xbf16, 2 : i32>>
  %lof_l2w = amdaie.logicalobjectfifo.from_memref %l2, {} : memref<2x4x8xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x4x8xbf16, 1 : i32>>
  %lof_l2r = amdaie.logicalobjectfifo.from_memref %l2, {} : memref<2x4x8xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x4x8xbf16, 1 : i32>>
  %lof_l3 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<8x8x16xbf16> -> !amdaie.logicalobjectfifo<memref<8x8x16xbf16>>
  scf.forall (%oc) = (0) to (8) step (4) {
    %0 = amdaie.dma_cpy_nd(%lof_l2w[0, 0, %oc] [2, 4, 4] [32, 8, 1], %lof_l1[0, 0, 0] [2, 4, 4] [16, 4, 1]) : (!amdaie.logicalobjectfifo<memref<2x4x8xbf16, 1 : i32>>, !amdaie.logicalobjectfifo<memref<2x4x4xbf16, 2 : i32>>)
  } {mapping = [#gpu.thread<y>]}
  %1 = amdaie.dma_cpy_nd(%lof_l3[%oh, 0, 8] [2, 4, 8] [128, 16, 1], %lof_l2r[0, 0, 0] [2, 4, 8] [32, 8, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x16xbf16>>, !amdaie.logicalobjectfifo<memref<2x4x8xbf16, 1 : i32>>)
  return
}

// -----

// Not on the conv tile pipeline: left as is.

// CHECK-LABEL: func.func @other_pipeline
// CHECK:           amdaie.dma_cpy_nd(%{{.+}}[%{{.+}}, 0, 0, %{{.+}}] [1, 2, 4, 4] [64, 32, 8, 1], %{{.+}}[0, 0, 0, 0] [1, 2, 4, 4] [32, 16, 4, 1])
// CHECK:         amdaie.dma_cpy_nd(%{{.+}}[0, 0, 0, 0] [2, 2, 4, 8] [64, 32, 8, 1], %{{.+}}[0, 0, 0, 0] [2, 2, 4, 8] [64, 32, 8, 1])
func.func @other_pipeline(%arg0: memref<2x2x4x8xbf16>) {
  %l1 = memref.alloc() : memref<1x2x4x4xbf16, 2 : i32>
  %l2 = memref.alloc() : memref<2x2x4x8xbf16, 1 : i32>
  %lof_l1 = amdaie.logicalobjectfifo.from_memref %l1, {} : memref<1x2x4x4xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x4x4xbf16, 2 : i32>>
  %lof_l2w = amdaie.logicalobjectfifo.from_memref %l2, {} : memref<2x2x4x8xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x4x8xbf16, 1 : i32>>
  %lof_l2r = amdaie.logicalobjectfifo.from_memref %l2, {} : memref<2x2x4x8xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x4x8xbf16, 1 : i32>>
  %lof_l3 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<2x2x4x8xbf16> -> !amdaie.logicalobjectfifo<memref<2x2x4x8xbf16>>
  scf.forall (%n, %oc) = (0, 0) to (2, 8) step (1, 4) {
    %0 = amdaie.dma_cpy_nd(%lof_l2w[%n, 0, 0, %oc] [1, 2, 4, 4] [64, 32, 8, 1], %lof_l1[0, 0, 0, 0] [1, 2, 4, 4] [32, 16, 4, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x4x8xbf16, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x4x4xbf16, 2 : i32>>)
  } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
  %1 = amdaie.dma_cpy_nd(%lof_l3[0, 0, 0, 0] [2, 2, 4, 8] [64, 32, 8, 1], %lof_l2r[0, 0, 0, 0] [2, 2, 4, 8] [64, 32, 8, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x4x8xbf16>>, !amdaie.logicalobjectfifo<memref<2x2x4x8xbf16, 1 : i32>>)
  return
}
