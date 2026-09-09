// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-amdaie-lower-executable-target)))))" %s | FileCheck %s

// The GEMV pipeline is selected by the `amdaie.tile_pipeline = "gemv"` entry
// that the lowering strategy put in translation_info (the global pipeline
// option is left at its pack-peel default). Expected shape, mirroring
// pack-peel with copies in place of packs:
//   * the whole N block of C lives in one local-memory alloc that the K loop
//     carries; each core slices it, so no C is read back per K step
//   * A and B are copied L3 -> L2 (N block x K tile) and L2 -> L1 (per core)
//     in every K iteration
//   * fill is fused into the first peeled iteration, the C write-out into the
//     last one
//   * no linalg.pack anywhere.

// CHECK-NOT:   linalg.pack
// CHECK-DAG:   memref.alloc() : memref<1x1024xf32, 2 : i32>
// CHECK-DAG:   memref.alloc() : memref<32x256xbf16, 2 : i32>
// CHECK-DAG:   memref.alloc() : memref<1x256xbf16, 2 : i32>
// CHECK-DAG:   memref.alloc() : memref<1024x256xbf16, 1 : i32>
// CHECK-DAG:   memref.alloc() : memref<1x256xbf16, 1 : i32>
// CHECK-DAG:   memref.alloc() : memref<1x1024xf32, 1 : i32>
// CHECK:       scf.forall (%{{.+}}) = (0) to (4096) step (1024)
// First (peeled) K iteration: L3 -> L2 copies, then per-core L2 -> L1 copies
// and the fill on the core's slice of the L1 C block.
// CHECK:         linalg.copy ins(%{{.+}} : memref<1x256xbf16, strided<[4096, 1]>{{.*}}) outs(%{{.+}} : memref<1x256xbf16, 1 : i32>)
// CHECK:         linalg.copy ins(%{{.+}} : memref<1024x256xbf16, strided<[4096, 1], offset: ?>{{.*}}) outs(%{{.+}} : memref<1024x256xbf16, 1 : i32>)
// CHECK:         scf.forall (%{{.+}}) = (0) to (1024) step (32)
// CHECK:           linalg.copy ins(%{{.+}} : memref<1x256xbf16, 1 : i32>) outs(%{{.+}} : memref<1x256xbf16, 2 : i32>)
// CHECK:           linalg.copy ins(%{{.+}} : memref<32x256xbf16, strided<[256, 1], offset: ?>, 1 : i32>) outs(%{{.+}} : memref<32x256xbf16, 2 : i32>)
// CHECK:           linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : memref<1x32xf32, strided<[1024, 1], offset: ?>, 2 : i32>)
// CHECK:           linalg.matmul
// CHECK-NOT:       linalg.copy
// CHECK:         } {mapping = [#gpu.thread<y>]}
// Middle K iterations: A/B copies only, no fill, no C traffic.
// CHECK:         scf.for %{{.+}} = %c256 to %c3840 step %c256
// CHECK:           linalg.copy {{.*}} outs(%{{.+}} : memref<1x256xbf16, 1 : i32>)
// CHECK:           linalg.copy {{.*}} outs(%{{.+}} : memref<1024x256xbf16, 1 : i32>)
// CHECK:           scf.forall (%{{.+}}) = (0) to (1024) step (32)
// CHECK:             linalg.copy {{.*}} outs(%{{.+}} : memref<1x256xbf16, 2 : i32>)
// CHECK:             linalg.copy {{.*}} outs(%{{.+}} : memref<32x256xbf16, 2 : i32>)
// CHECK-NOT:         linalg.fill
// CHECK:             linalg.matmul
// CHECK-NOT:         linalg.copy
// CHECK:           } {mapping = [#gpu.thread<y>]}
// CHECK:         }
// Last (peeled) K iteration: the core writes its C slice L1 -> L2, then the
// block goes L2 -> L3.
// CHECK:         scf.forall (%{{.+}}) = (0) to (1024) step (32)
// CHECK:           linalg.matmul
// CHECK:           linalg.copy ins(%{{.+}} : memref<1x32xf32, strided<[1024, 1], offset: ?>, 2 : i32>) outs(%{{.+}} : memref<1x32xf32, strided<[1024, 1], offset: ?>, 1 : i32>)
// CHECK:         } {mapping = [#gpu.thread<y>]}
// CHECK:         linalg.copy ins(%{{.+}} : memref<1x1024xf32, 1 : i32>) outs(%{{.+}} : memref<1x1024xf32, strided<[4096, 1], offset: ?>{{.*}})
// CHECK:       } {mapping = [#gpu.block<y>]}
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1024, 0], [0, 0, 256], [0, 32, 0]]>
#translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "gemv"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @gemv {
  hal.executable.variant public @amdaie_pdi_fb target(<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>) {
    builtin.module {
      func.func @gemv_fc2_dispatch_0_matmul_1x4096x4096_bf16xbf16xf32() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>> -> tensor<1x4096xbf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>> -> tensor<4096x4096xbf16>
        %5 = tensor.empty() : tensor<1x4096xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
        %7 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                            affine_map<(d0, d1, d2) -> (d1, d2)>,
                                            affine_map<(d0, d1, d2) -> (d0, d1)>]
             {lowering_config = #config}
             ins(%3, %4 : tensor<1x4096xbf16>, tensor<4096x4096xbf16>) outs(%6 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : tensor<1x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
        return
      }
    }
  }
}
