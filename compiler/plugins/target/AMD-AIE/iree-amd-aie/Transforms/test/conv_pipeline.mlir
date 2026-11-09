// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-amdaie-lower-executable-target)))))" %s | FileCheck %s

// The native conv pipeline is selected by the `amdaie.tile_pipeline = "conv"`
// entry that the lowering strategy put in translation_info. Expected shape,
// the GEMV pipeline's with the filter window inside the core op:
//   * the block's accumulator lives in one local-memory alloc that the
//     input-channel loop carries; each core slices it, so no partial sum moves
//     between channel chunks
//   * per channel chunk, the input (all 4 halo chunks, with their halo rows and
//     columns) and the filter (all output channels) are copied L3 -> L2, and
//     each core copies its chunk and its output-channel tile L2 -> L1
//   * chunks go to array columns and output channels to rows (thread x, y)
//   * fill in the first peeled chunk, write-out in the last
//   * no linalg.pack anywhere.

// CHECK-LABEL: func.func @conv_chunks_fhwc
// CHECK-NOT:   linalg.pack
// CHECK-DAG:   memref.alloc() : memref<4x4x16x64xf32, 2 : i32>
// CHECK-DAG:   memref.alloc() : memref<16x3x3x16xbf16, 2 : i32>
// CHECK-DAG:   memref.alloc() : memref<1x6x18x16xbf16, 2 : i32>
// CHECK-DAG:   memref.alloc() : memref<64x3x3x16xbf16, 1 : i32>
// CHECK-DAG:   memref.alloc() : memref<4x6x18x16xbf16, 1 : i32>
// CHECK-DAG:   memref.alloc() : memref<4x4x16x64xf32, 1 : i32>
// CHECK:       scf.forall (%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) = (0, 0, 0, 0) to (4, 4, 16, 64) step (4, 4, 16, 64)
// First (peeled) channel chunk: L3 -> L2 copies, then per core the L2 -> L1
// copies, the fill on the core's slice of the L1 accumulator and the conv.
// CHECK:         linalg.copy ins(%{{.+}} : memref<4x6x18x16xbf16, strided<[6912, 1152, 64, 1]>{{.*}}) outs(%{{.+}} : memref<4x6x18x16xbf16, 1 : i32>)
// CHECK:         linalg.copy ins(%{{.+}} : memref<64x3x3x16xbf16, strided<[576, 192, 64, 1]>{{.*}}) outs(%{{.+}} : memref<64x3x3x16xbf16, 1 : i32>)
// CHECK:         scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (4, 64) step (1, 16)
// CHECK:           linalg.copy ins(%{{.+}} : memref<1x6x18x16xbf16, strided<[1728, 288, 16, 1], offset: ?>, 1 : i32>) outs(%{{.+}} : memref<1x6x18x16xbf16, 2 : i32>)
// CHECK:           linalg.copy ins(%{{.+}} : memref<16x3x3x16xbf16, strided<[144, 48, 16, 1], offset: ?>, 1 : i32>) outs(%{{.+}} : memref<16x3x3x16xbf16, 2 : i32>)
// CHECK:           linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : memref<1x4x16x16xf32, strided<[4096, 1024, 64, 1], offset: ?>, 2 : i32>)
// CHECK:           linalg.generic
// CHECK-NOT:       linalg.copy
// CHECK:         } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
// Middle chunks: input/filter copies only, no fill, no output traffic.
// CHECK:         scf.for %{{.+}} = %c16 to %c48 step %c16
// CHECK:           linalg.copy {{.*}} outs(%{{.+}} : memref<4x6x18x16xbf16, 1 : i32>)
// CHECK:           linalg.copy {{.*}} outs(%{{.+}} : memref<64x3x3x16xbf16, 1 : i32>)
// CHECK:           scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (4, 64) step (1, 16)
// CHECK:             linalg.copy {{.*}} outs(%{{.+}} : memref<1x6x18x16xbf16, 2 : i32>)
// CHECK:             linalg.copy {{.*}} outs(%{{.+}} : memref<16x3x3x16xbf16, 2 : i32>)
// CHECK-NOT:         linalg.fill
// CHECK:             linalg.generic
// CHECK-NOT:         linalg.copy
// CHECK:           } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
// CHECK:         }
// Last (peeled) chunk: each core writes its accumulator slice L1 -> L2, then
// the block goes L2 -> L3.
// CHECK:         scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (4, 64) step (1, 16)
// CHECK:           linalg.generic
// CHECK:           linalg.copy ins(%{{.+}} : memref<1x4x16x16xf32, strided<[4096, 1024, 64, 1], offset: ?>, 2 : i32>) outs(%{{.+}} : memref<1x4x16x16xf32, strided<[4096, 1024, 64, 1], offset: ?>, 1 : i32>)
// CHECK:         } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
// CHECK:         linalg.copy ins(%{{.+}} : memref<4x4x16x64xf32, 1 : i32>) outs(%{{.+}} : memref<4x4x16x64xf32, #hal.descriptor_type<storage_buffer>>)
// CHECK:       } {mapping = [#gpu.block<y>, #gpu.block<x>, #gpu.block<z>, #gpu.block<linear_dim_0>]}
#config = #iree_codegen.lowering_config<tile_sizes = [[4, 4, 16, 64, 0, 0, 0], [0, 0, 0, 0, 0, 0, 16], [1, 0, 0, 16, 0, 0, 0]]>
#translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "conv"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @conv {
  hal.executable.variant public @amdaie_pdi_fb target(<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>) {
    builtin.module {
      func.func @conv_chunks_fhwc() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x6x18x64xbf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x3x3x64xbf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x16x64xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 6, 18, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x6x18x64xbf16>> -> tensor<4x6x18x64xbf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [64, 3, 3, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x3x3x64xbf16>> -> tensor<64x3x3x64xbf16>
        %5 = tensor.empty() : tensor<4x4x16x64xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4x16x64xf32>) -> tensor<4x4x16x64xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
                                              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>,
                                              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
                             iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
             ins(%3, %4 : tensor<4x6x18x64xbf16>, tensor<64x3x3x64xbf16>) outs(%6 : tensor<4x4x16x64xf32>) attrs = {lowering_config = #config} {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %8 = arith.extf %in : bf16 to f32
          %9 = arith.extf %in_0 : bf16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %out, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<4x4x16x64xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [4, 4, 16, 64], strides = [1, 1, 1, 1] : tensor<4x4x16x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x16x64xf32>>
        return
      }
    }
  }
}

// -----

// With a fused elementwise epilogue (ReLU) the epilogue reads the accumulator
// where it is computed, in the last channel chunk, and only its result moves
// L1 -> L2 -> L3 (AMDAIEInsertCopyOps stages it as it does a GEMV epilogue).
// CHECK-LABEL: func.func @conv_relu
// First chunk: fill and conv, no epilogue, no output traffic.
// CHECK:       scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (8, 64) step (1, 16)
// CHECK:         linalg.fill
// CHECK:         linalg.generic
// CHECK-NOT:     arith.cmpf
// CHECK-NOT:     linalg.copy ins(%{{.+}} : memref<1x2x16x16xf32
// CHECK:       } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
// Last chunk: conv, ReLU into the epilogue tile, and that tile goes L1 -> L2.
// CHECK:       scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (8, 64) step (1, 16)
// CHECK-NOT:     linalg.fill
// CHECK:         arith.cmpf ugt
// CHECK:         linalg.copy ins(%{{.+}} : memref<1x2x16x16xf32, strided<[2048, 1024, 64, 1], offset: ?>, 2 : i32>) outs(%{{.+}} : memref<1x2x16x16xf32, strided<[2048, 1024, 64, 1], offset: ?>, 1 : i32>)
// CHECK:       } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
// CHECK:       linalg.copy ins(%{{.+}} : memref<8x2x16x64xf32, 1 : i32>) outs(%{{.+}} : memref<8x2x16x64xf32, #hal.descriptor_type<storage_buffer>>)
#config = #iree_codegen.lowering_config<tile_sizes = [[8, 2, 16, 64, 0, 0, 0], [0, 0, 0, 0, 0, 0, 32], [1, 0, 0, 16, 0, 0, 0]]>
#translation = #iree_codegen.translation_info<pipeline = Custom, {amdaie.tile_pipeline = "conv"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @conv_relu {
  hal.executable.variant public @amdaie_pdi_fb target(<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>) {
    builtin.module {
      func.func @conv_relu() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4x18x64xbf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x3x3x64xbf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2x16x64xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [8, 4, 18, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x4x18x64xbf16>> -> tensor<8x4x18x64xbf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [64, 3, 3, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x3x3x64xbf16>> -> tensor<64x3x3x64xbf16>
        %5 = tensor.empty() : tensor<8x2x16x64xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8x2x16x64xf32>) -> tensor<8x2x16x64xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
                                              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>,
                                              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>],
                             iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
             ins(%3, %4 : tensor<8x4x18x64xbf16>, tensor<64x3x3x64xbf16>) outs(%6 : tensor<8x2x16x64xf32>) attrs = {lowering_config = #config} {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %8 = arith.extf %in : bf16 to f32
          %9 = arith.extf %in_0 : bf16 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %out, %10 : f32
          linalg.yield %11 : f32
        } -> tensor<8x2x16x64xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                              iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
             ins(%7 : tensor<8x2x16x64xf32>) outs(%5 : tensor<8x2x16x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %13 = arith.cmpf ugt, %in, %cst : f32
          %14 = arith.select %13, %in, %cst : f32
          linalg.yield %14 : f32
        } -> tensor<8x2x16x64xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %2, offsets = [0, 0, 0, 0], sizes = [8, 2, 16, 64], strides = [1, 1, 1, 1] : tensor<8x2x16x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2x16x64xf32>>
        return
      }
    }
  }
}
