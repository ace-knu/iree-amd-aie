// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{target-device=npu4 num-rows=4 num-cols=8})' %s

// With `native_conv`, a conv the native conv tile pipeline cannot take is
// rejected when its lowering strategy is chosen, with the reason, instead of
// failing in a later pass. The pipeline starts the accumulator from a fill and
// fuses at most one unary elementwise epilogue with identity maps, which reads
// only the conv result and is only stored.

// A bias-initialized output (the bias broadcast as the conv's init).
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  // expected-error @+1 {{failed to have a lowering configuration set for it}}
  func.func @conv_bias_init() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [6, 6, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>> -> tensor<6x6x8xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>> -> tensor<3x3x8x8xbf16>
    %bias = arith.constant dense<0.5> : tensor<8xf32>
    %5 = tensor.empty() : tensor<4x4x8xf32>
    %6 = linalg.broadcast ins(%bias : tensor<8xf32>) outs(%5 : tensor<4x4x8xf32>) dimensions = [0, 1]
    // expected-error @+1 {{native conv pipeline: the output must start from a linalg.fill}}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<6x6x8xbf16>, tensor<3x3x8x8xbf16>) outs(%6 : tensor<4x4x8xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<4x4x8xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    return
  }
}

// -----

// A residual add after the conv (a second tensor read by the epilogue).
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  // expected-error @+1 {{failed to have a lowering configuration set for it}}
  func.func @conv_residual() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x8xf32>>
    %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [6, 6, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>> -> tensor<6x6x8xbf16>
    %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>> -> tensor<3x3x8x8xbf16>
    %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x8xf32>> -> tensor<4x4x8xf32>
    %7 = tensor.empty() : tensor<4x4x8xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4x4x8xf32>) -> tensor<4x4x8xf32>
    // expected-error @+1 {{native conv pipeline: its result is read by 'linalg.generic', which is not a unary elementwise epilogue with identity maps}}
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%4, %5 : tensor<6x6x8xbf16>, tensor<3x3x8x8xbf16>) outs(%8 : tensor<4x4x8xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<4x4x8xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %6 : tensor<4x4x8xf32>, tensor<4x4x8xf32>) outs(%7 : tensor<4x4x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %30 = arith.addf %in, %in_0 : f32
      linalg.yield %30 : f32
    } -> tensor<4x4x8xf32>
    iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    return
  }
}

// -----

// A named conv: the pipeline's copy insertion and producer fusion handle the
// generic form, which is what iree-compile hands the backend.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  // expected-error @+1 {{failed to have a lowering configuration set for it}}
  func.func @conv_named() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x6x18x16xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x16x16xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x4x16x16xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 6, 18, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x6x18x16xbf16>> -> tensor<2x6x18x16xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 16, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x16x16xbf16>> -> tensor<3x3x16x16xbf16>
    %5 = tensor.empty() : tensor<2x4x16x16xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x4x16x16xf32>) -> tensor<2x4x16x16xf32>
    // expected-error @+1 {{native conv pipeline: needs the conv as a linalg.generic}}
    %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %4 : tensor<2x6x18x16xbf16>, tensor<3x3x16x16xbf16>) outs(%6 : tensor<2x4x16x16xf32>) -> tensor<2x4x16x16xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 4, 16, 16], strides = [1, 1, 1, 1] : tensor<2x4x16x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x4x16x16xf32>>
    return
  }
}

// -----

// A chain of epilogues (ReLU, then truncf in a second generic): the pipeline
// gives one epilogue a core tile of its own, and IREE's elementwise fusion
// normally leaves one generic.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  // expected-error @+1 {{failed to have a lowering configuration set for it}}
  func.func @conv_epilogue_chain() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xbf16>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [6, 6, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>> -> tensor<6x6x8xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>> -> tensor<3x3x8x8xbf16>
    %5 = tensor.empty() : tensor<4x4x8xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4x8xf32>) -> tensor<4x4x8xf32>
    // expected-error @+1 {{native conv pipeline: its epilogue's result is read by 'linalg.generic' (a chain of epilogues is not supported; fuse them into one linalg.generic)}}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<6x6x8xbf16>, tensor<3x3x8x8xbf16>) outs(%6 : tensor<4x4x8xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<4x4x8xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<4x4x8xf32>) outs(%5 : tensor<4x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.maximumf %in, %cst : f32
      linalg.yield %30 : f32
    } -> tensor<4x4x8xf32>
    %9 = tensor.empty() : tensor<4x4x8xbf16>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<4x4x8xf32>) outs(%9 : tensor<4x4x8xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %31 = arith.truncf %in : f32 to bf16
      linalg.yield %31 : bf16
    } -> tensor<4x4x8xbf16>
    iree_tensor_ext.dispatch.tensor.store %10, %2, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xbf16>>
    return
  }
}

// -----

// An epilogue that reads its own init (out = conv + out on a readwrite
// output): the pipeline computes the epilogue tile from the accumulator alone.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  // expected-error @+1 {{failed to have a lowering configuration set for it}}
  func.func @conv_epilogue_reads_init() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x8xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [6, 6, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>> -> tensor<6x6x8xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>> -> tensor<3x3x8x8xbf16>
    %prev = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x8xf32>> -> tensor<4x4x8xf32>
    %5 = tensor.empty() : tensor<4x4x8xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4x8xf32>) -> tensor<4x4x8xf32>
    // expected-error @+1 {{native conv pipeline: its epilogue reads its own init (only an epilogue of the conv result alone is supported)}}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<6x6x8xbf16>, tensor<3x3x8x8xbf16>) outs(%6 : tensor<4x4x8xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<4x4x8xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<4x4x8xf32>) outs(%prev : tensor<4x4x8xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %30 = arith.addf %in, %acc : f32
      linalg.yield %30 : f32
    } -> tensor<4x4x8xf32>
    iree_tensor_ext.dispatch.tensor.store %8, %2, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x8xf32>>
    return
  }
}

// -----

// Two epilogues of one conv result: the pipeline has one output path, the
// result or one epilogue of it.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  // expected-error @+1 {{failed to have a lowering configuration set for it}}
  func.func @conv_two_epilogues() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>>
    %b2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    %b3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [6, 6, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>> -> tensor<6x6x8xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>> -> tensor<3x3x8x8xbf16>
    %5 = tensor.empty() : tensor<4x4x8xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4x8xf32>) -> tensor<4x4x8xf32>
    // expected-error @+1 {{native conv pipeline: its result has 2 readers (one store or one unary elementwise epilogue is supported)}}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<6x6x8xbf16>, tensor<3x3x8x8xbf16>) outs(%6 : tensor<4x4x8xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<4x4x8xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<4x4x8xf32>) outs(%5 : tensor<4x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.maximumf %in, %cst : f32
      linalg.yield %30 : f32
    } -> tensor<4x4x8xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<4x4x8xf32>) outs(%5 : tensor<4x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %31 = arith.mulf %in, %in : f32
      linalg.yield %31 : f32
    } -> tensor<4x4x8xf32>
    iree_tensor_ext.dispatch.tensor.store %8, %b2, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    iree_tensor_ext.dispatch.tensor.store %9, %b3, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    return
  }
}

// -----

// The raw result stored next to its epilogue: once an epilogue reads it, the
// accumulator is not copied out (AMDAIEInsertCopyOps).
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  // expected-error @+1 {{failed to have a lowering configuration set for it}}
  func.func @conv_raw_and_epilogue() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>>
    %b2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    %b3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [6, 6, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>> -> tensor<6x6x8xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>> -> tensor<3x3x8x8xbf16>
    %5 = tensor.empty() : tensor<4x4x8xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4x8xf32>) -> tensor<4x4x8xf32>
    // expected-error @+1 {{native conv pipeline: its result has 2 readers (one store or one unary elementwise epilogue is supported)}}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<6x6x8xbf16>, tensor<3x3x8x8xbf16>) outs(%6 : tensor<4x4x8xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<4x4x8xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<4x4x8xf32>) outs(%5 : tensor<4x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.maximumf %in, %cst : f32
      linalg.yield %30 : f32
    } -> tensor<4x4x8xf32>
    iree_tensor_ext.dispatch.tensor.store %7, %b2, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    iree_tensor_ext.dispatch.tensor.store %8, %b3, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    return
  }
}

// -----

// The epilogue's result stored twice.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#target = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4"}>
module attributes {hal.executable.target = #target} {
  // expected-error @+1 {{failed to have a lowering configuration set for it}}
  func.func @conv_epilogue_stored_twice() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>>
    %b2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    %b3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [6, 6, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x6x8xbf16>> -> tensor<6x6x8xbf16>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x8xbf16>> -> tensor<3x3x8x8xbf16>
    %5 = tensor.empty() : tensor<4x4x8xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4x8xf32>) -> tensor<4x4x8xf32>
    // expected-error @+1 {{native conv pipeline: its epilogue's result has 2 readers (one store is supported)}}
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>,
                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
         ins(%3, %4 : tensor<6x6x8xbf16>, tensor<3x3x8x8xbf16>) outs(%6 : tensor<4x4x8xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %20 = arith.extf %in : bf16 to f32
      %21 = arith.extf %in_0 : bf16 to f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %out, %22 : f32
      linalg.yield %23 : f32
    } -> tensor<4x4x8xf32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<4x4x8xf32>) outs(%5 : tensor<4x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.maximumf %in, %cst : f32
      linalg.yield %30 : f32
    } -> tensor<4x4x8xf32>
    iree_tensor_ext.dispatch.tensor.store %8, %b2, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    iree_tensor_ext.dispatch.tensor.store %8, %b3, offsets = [0, 0, 0], sizes = [4, 4, 8], strides = [1, 1, 1] : tensor<4x4x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4x8xf32>>
    return
  }
}
