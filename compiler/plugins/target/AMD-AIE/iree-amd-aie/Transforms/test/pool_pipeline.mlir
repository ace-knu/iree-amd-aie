// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-amdaie-lower-executable-target)))))" %s | FileCheck %s

// The pool pipeline is selected by the `amdaie.tile_pipeline = "pool"` entry
// the lowering strategy put in translation_info. Expected shape, a copy-based
// two-level lowering of a 2x2/2 max pooling (16x16x16 -> 8x8x16, one block of
// 8 cores along OH, core tile [1, 8, 16]; OW and C are single blocks / single
// tiles and stay untiled, so both foralls are 1-D):
//   * the block input/output are copied L3 -> L2 / L2 -> L3 as whole slabs
//   * each core copies its input rows (2 x 16 x 16, the bounding box of its
//     windows) L2 -> L1 and its output row L1 -> L2
//   * the 2x2 window operand is never copied: it only gets a local alloc
//   * the window (reduction) loops stay inside the core op; no linalg.pack.

// CHECK-NOT:   linalg.pack
// CHECK-DAG:   %[[OUT_L1:.+]] = memref.alloc() : memref<1x8x16xf32, 2 : i32>
// CHECK-DAG:   %[[IN_L1:.+]] = memref.alloc() : memref<2x16x16xf32, 2 : i32>
// CHECK-DAG:   %[[OUT_L2:.+]] = memref.alloc() : memref<8x8x16xf32, 1 : i32>
// CHECK-DAG:   %[[IN_L2:.+]] = memref.alloc() : memref<16x16x16xf32, 1 : i32>
// CHECK-DAG:   %[[WIN:.+]] = memref.alloc() : memref<2x2xf32, 2 : i32>
// CHECK:       scf.forall (%{{.+}}) = (0) to (8) step (8)
// CHECK:         linalg.copy ins(%{{.+}} : memref<16x16x16xf32{{.*}}) outs(%[[IN_L2]] : memref<16x16x16xf32, 1 : i32>)
// CHECK:         scf.forall (%{{.+}}) in (8)
// CHECK:           linalg.copy ins(%{{.+}} : memref<2x16x16xf32, strided<[256, 16, 1], offset: ?>, 1 : i32>) outs(%[[IN_L1]] : memref<2x16x16xf32, 2 : i32>)
// CHECK:           linalg.fill ins(%{{.+}} : f32) outs(%[[OUT_L1]] : memref<1x8x16xf32, 2 : i32>)
// CHECK-NOT:       linalg.copy
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[IN_L1]], %[[WIN]] : memref<2x16x16xf32, 2 : i32>, memref<2x2xf32, 2 : i32>)
// The max is expanded to cmpf + select (peano cannot legalize a scalar fmaximum).
// CHECK:             arith.cmpf ugt
// CHECK:             arith.select
// CHECK-NOT:         arith.maximumf
// CHECK:           linalg.copy ins(%[[OUT_L1]] : memref<1x8x16xf32, 2 : i32>) outs(%{{.+}} : memref<1x8x16xf32, strided<[128, 16, 1], offset: ?>, 1 : i32>)
// CHECK:         } {mapping = [#gpu.thread<y>]}
// CHECK:         linalg.copy ins(%[[OUT_L2]] : memref<8x8x16xf32, 1 : i32>) outs(%{{.+}} : memref<8x8x16xf32{{.*}})
// CHECK:       } {mapping = [#gpu.block<y>]}
#config = #iree_codegen.lowering_config<tile_sizes = [[8, 0, 0, 0, 0], [1, 0, 0, 0, 0]]>
#translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "pool"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @pool {
  hal.executable.variant public @amdaie_pdi_fb target(<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>) {
    builtin.module {
      func.func @pool_dispatch_0_conv_8x8x16x2x2_f32() attributes {translation_info = #translation} {
        %cst = arith.constant 0xFF800000 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16x16xf32>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8x16xf32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [16, 16, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16x16xf32>> -> tensor<16x16x16xf32>
        %3 = tensor.empty() : tensor<8x8x16xf32>
        %4 = tensor.empty() : tensor<2x2xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8x8x16xf32>) -> tensor<8x8x16xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"], lowering_config = #config} ins(%2, %4 : tensor<16x16x16xf32>, tensor<2x2xf32>) outs(%5 : tensor<8x8x16xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %7 = arith.maximumf %out, %in : f32
          linalg.yield %7 : f32
        } -> tensor<8x8x16xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [8, 8, 16], strides = [1, 1, 1] : tensor<8x8x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8x16xf32>>
        return
      }
    }
  }
}
