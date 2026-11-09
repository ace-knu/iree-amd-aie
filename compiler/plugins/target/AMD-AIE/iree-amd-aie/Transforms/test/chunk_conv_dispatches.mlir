// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-chunk-conv-dispatches)' %s | FileCheck %s

// A batch-folded 1x1 conv (OH = 4) with an HWCF constant filter writing its
// whole result: OH is split into 2 chunks of 2 rows. The host gathers the
// input chunks (no halo for a 1x1 window) into a new tensor, the filter
// becomes an FHWC constant, and the executable runs a 7-loop conv on
// [2, 2, 2, 8] x [2, 1, 1, 8] into a new [2, 2, 2, 2] binding, which the host
// reads as [4, 2, 2].

// CHECK-DAG:   #[[$IN:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK-DAG:   #[[$FHWC:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
// CHECK-DAG:   #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK-DAG:   #[[$ROWS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d1, d2, d3)>
// CHECK-DAG:   #[[$ID:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.global private @__conv_fhwc_weight_1
// CHECK-SAME:    dense<{{\[\[\[}}[1.000000e+00, 3.000000e+00, 5.000000e+00, 7.000000e+00, 9.000000e+00, 1.100000e+01, 1.300000e+01, 1.500000e+01]]], {{\[\[}}[2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01]]]]> : tensor<2x1x1x8xbf16>
// CHECK-NOT:   util.global private @weight
// CHECK-LABEL: func.func @conv_1x1
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2x2x8xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x1x1x8xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2x2x2xf32>>
// CHECK:         %[[FILL:.+]] = linalg.fill {{.*}} -> tensor<2x2x2x2xf32>
// CHECK:         %[[CONV:.+]] = linalg.generic {indexing_maps = [#[[$IN]], #[[$FHWC]], #[[$OUT]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<2x2x2x8xbf16>, tensor<2x1x1x8xbf16>) outs(%[[FILL]] : tensor<2x2x2x2xf32>)
// CHECK:           arith.mulf
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[CONV]], %arg2, offsets = [0, 0, 0, 0], sizes = [2, 2, 2, 2]
// CHECK-LABEL: func.func @halo_dispatch_0
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2x2x8xbf16>>
// CHECK:         linalg.generic {indexing_maps = [#[[$ROWS]], #[[$ID]]]
// CHECK-SAME:      ins(%{{.+}} : tensor<4x2x8xbf16>) outs(%{{.+}} : tensor<2x2x2x8xbf16>)
// CHECK-LABEL: util.func public @main
// CHECK:         %[[IN:.+]] = flow.dispatch @halo_executable_0::@halo_dispatch_0(%arg0) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<4x2x8xbf16>) -> tensor<2x2x2x8xbf16>
// CHECK:         %[[W:.+]] = util.global.load immutable @__conv_fhwc_weight_1 : tensor<2x1x1x8xbf16>
// CHECK:         %[[OUT:.+]] = flow.dispatch @dispatch_0::@conv_1x1(%[[IN]], %[[W]]) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x2x2x8xbf16>, tensor<2x1x1x8xbf16>) -> tensor<2x2x2x2xf32>
// CHECK:         %[[ROWS:.+]] = flow.tensor.reshape %[[OUT]] : tensor<2x2x2x2xf32> -> tensor<4x2x2xf32>
// CHECK:         util.return %[[ROWS]]
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]]]> : tensor<1x1x8x2xbf16>
  flow.executable private @dispatch_0 {
    flow.executable.export public @conv_1x1 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_1x1(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [4, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>> -> tensor<4x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<4x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<4x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %5 = arith.extf %in : bf16 to f32
          %6 = arith.extf %in_0 : bf16 to f32
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<4x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [4, 2, 2], strides = [1, 1, 1] : tensor<4x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<4x2x8xbf16>) -> tensor<4x2x2xf32> {
    %w = util.global.load immutable @weight : tensor<1x1x8x2xbf16>
    %0 = flow.dispatch @dispatch_0::@conv_1x1(%arg0, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xf32>
    util.return %0 : tensor<4x2x2xf32>
  }
}

// -----

// A 3x3 conv as the bias fold leaves it: its input is a host pad dispatch that
// adds ones channels (7 -> 8), it ends in a ReLU, and it writes into the
// interior of the next layer's padded input (a result tied at [1, 1, 0]).
// The gather reads the pad's source into a ones splat of 2 chunks of
// 2 + 2 halo rows (the pad dispatch goes away), the executable writes its own
// [2, 2, 2, 2] result with the ReLU on it, and a host dispatch tied to the
// padded tensor copies the rows in at [1, 1, 0].

// CHECK-DAG:   #[[$IN:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK-DAG:   #[[$FHWC:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
// CHECK-DAG:   #[[$OUT:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK-DAG:   #[[$ROWS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d1, d2, d3)>
// CHECK-DAG:   #[[$ID:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: util.global private @__conv_fhwc_weight_3
// CHECK-SAME:    dense<1.000000e+00> : tensor<2x3x3x8xbf16>
// CHECK-LABEL: func.func @conv_3x3_relu
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4x4x8xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3x3x8xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2x2x2xf32>>
// CHECK:         %[[CONV:.+]] = linalg.generic {indexing_maps = [#[[$IN]], #[[$FHWC]], #[[$OUT]]]
// CHECK-SAME:      ins(%{{.+}}, %{{.+}} : tensor<2x4x4x8xbf16>, tensor<2x3x3x8xbf16>)
// CHECK:         %[[RELU:.+]] = linalg.generic {indexing_maps = [#[[$ID]], #[[$ID]]]
// CHECK-SAME:      ins(%[[CONV]] : tensor<2x2x2x2xf32>)
// CHECK:           arith.cmpf ugt
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[RELU]], %arg2, offsets = [0, 0, 0, 0], sizes = [2, 2, 2, 2]
// CHECK-NOT:   flow.executable private @pad_executable_0
// CHECK-LABEL: func.func @halo_dispatch_2
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x4x7xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2x4x4x8xbf16>>
// CHECK:         %[[G:.+]] = linalg.generic {indexing_maps = [#[[$ROWS]], #[[$ID]]]
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[G]], %arg1, offsets = [0, 0, 0, 0], sizes = [2, 4, 4, 7], strides = [1, 1, 1, 1]
// CHECK-LABEL: func.func @insert_dispatch_4
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x2xf32>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<6x4x2xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %arg1, offsets = [1, 1, 0], sizes = [4, 2, 2], strides = [1, 1, 1]
// CHECK-LABEL: util.func public @main
// CHECK-NOT:     pad_dispatch_0
// CHECK:         %[[ONE:.+]] = arith.constant 1.000000e+00 : bf16
// CHECK:         %[[ONES:.+]] = flow.tensor.splat %[[ONE]] : tensor<2x4x4x8xbf16>
// CHECK:         %[[IN:.+]] = flow.dispatch @halo_executable_2::@halo_dispatch_2(%arg0, %[[ONES]]) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<6x4x7xbf16>, tensor<2x4x4x8xbf16>) -> %[[ONES]]
// CHECK:         %[[W:.+]] = util.global.load immutable @__conv_fhwc_weight_3
// CHECK:         %[[OUT:.+]] = flow.dispatch @dispatch_0::@conv_3x3_relu(%[[IN]], %[[W]]) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x4x4x8xbf16>, tensor<2x3x3x8xbf16>) -> tensor<2x2x2x2xf32>
// CHECK:         %[[ROWS:.+]] = flow.tensor.reshape %[[OUT]] : tensor<2x2x2x2xf32> -> tensor<4x2x2xf32>
// CHECK:         %[[PADDED:.+]] = flow.dispatch @insert_executable_4::@insert_dispatch_4(%[[ROWS]], %arg1) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<4x2x2xf32>, tensor<6x4x2xf32>) -> %arg1
// CHECK:         util.return %[[PADDED]]
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @__bias_folded_weight_1 {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<1.0> : tensor<3x3x8x2xbf16>
  flow.executable private @dispatch_0 {
    flow.executable.export public @conv_3x3_relu workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_3x3_relu(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x4x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<6x4x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [6, 4, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x4x8xbf16>> -> tensor<6x4x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x2xbf16>> -> tensor<3x3x8x2xbf16>
        %2 = tensor.empty() : tensor<4x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<6x4x8xbf16>, tensor<3x3x8x2xbf16>) outs(%3 : tensor<4x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %6 = arith.extf %in : bf16 to f32
          %7 = arith.extf %in_0 : bf16 to f32
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<4x2x2xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<4x2x2xf32>) outs(%2 : tensor<4x2x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.cmpf ugt, %in, %cst : f32
          %7 = arith.select %6, %in, %cst : f32
          linalg.yield %7 : f32
        } -> tensor<4x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg2, offsets = [1, 1, 0], sizes = [4, 2, 2], strides = [1, 1, 1] : tensor<4x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<6x4x2xf32>>
        return
      }
    }
  }
  flow.executable private @pad_executable_0 {
    flow.executable.export public @pad_dispatch_0 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pad_dispatch_0(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x4x7xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<6x4x8xbf16>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [6, 4, 7], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x4x7xbf16>> -> tensor<6x4x7xbf16>
        iree_tensor_ext.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 0], sizes = [6, 4, 7], strides = [1, 1, 1] : tensor<6x4x7xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<6x4x8xbf16>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<6x4x7xbf16>, %arg1: tensor<6x4x2xf32>) -> tensor<6x4x2xf32> {
    %one = arith.constant 1.000000e+00 : bf16
    %ones = flow.tensor.splat %one : tensor<6x4x8xbf16> attributes {stream.affinity = #hal.device.affinity<@cpu>}
    %in = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0, %ones) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<6x4x7xbf16>, tensor<6x4x8xbf16>) -> %ones
    %w = util.global.load immutable @__bias_folded_weight_1 : tensor<3x3x8x2xbf16>
    %0 = flow.dispatch @dispatch_0::@conv_3x3_relu(%in, %w, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<6x4x8xbf16>, tensor<3x3x8x2xbf16>, tensor<6x4x2xf32>) -> %arg1
    util.return %0 : tensor<6x4x2xf32>
  }
}

// -----

// Not chunked: OH = 2 leaves no chunk count with two rows per chunk (the conv
// keeps its rows but gets an FHWC copy of its filter, so the rows of its column
// can take output-channel tiles), a
// target without `native_conv` keeps its convs as they are, an elementwise
// consumer that reads the conv value as its init is not an epilogue that can
// be moved, an input loaded as a smaller slice of its binding is not a
// full-tensor load, and a chain of two epilogues is more than the conv
// pipeline takes.

// CHECK-LABEL: util.func public @main
// CHECK:         flow.dispatch @dispatch_0::@conv_oh2(%arg0, %{{.+}}) {{.*}} : (tensor<2x2x8xbf16>, tensor<2x1x1x8xbf16>) -> tensor<2x2x2xf32>
// CHECK:         flow.dispatch @dispatch_1::@conv_no_native(%arg1, %{{.+}}) {{.*}} : (tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xf32>
// CHECK:         flow.dispatch @dispatch_2::@conv_reads_init(%arg1, %{{.+}}) {{.*}} : (tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xf32>
// CHECK:         flow.dispatch @dispatch_3::@conv_partial_load(%arg2, %{{.+}}) {{.*}} : (tensor<4x3x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xf32>
// CHECK:         flow.dispatch @dispatch_4::@conv_chain(%arg1, %{{.+}}) {{.*}} : (tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xbf16>
// CHECK-NOT:     halo_dispatch
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @npu_default = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<1.0> : tensor<1x1x8x2xbf16>
  flow.executable private @dispatch_0 {
    flow.executable.export public @conv_oh2 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_oh2(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [2, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2x8xbf16>> -> tensor<2x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<2x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<2x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<2x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %5 = arith.extf %in : bf16 to f32
          %6 = arith.extf %in_0 : bf16 to f32
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<2x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [2, 2, 2], strides = [1, 1, 1] : tensor<2x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2x2xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_1 {
    flow.executable.export public @conv_no_native workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_no_native(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [4, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>> -> tensor<4x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<4x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<4x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %5 = arith.extf %in : bf16 to f32
          %6 = arith.extf %in_0 : bf16 to f32
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<4x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [4, 2, 2], strides = [1, 1, 1] : tensor<4x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_2 {
    flow.executable.export public @conv_reads_init workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_reads_init(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %ones = arith.constant dense<1.000000e+00> : tensor<4x2x2xf32>
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [4, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>> -> tensor<4x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<4x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<4x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %6 = arith.extf %in : bf16 to f32
          %7 = arith.extf %in_0 : bf16 to f32
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<4x2x2xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%ones : tensor<4x2x2xf32>) outs(%4 : tensor<4x2x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.addf %in, %out : f32
          linalg.yield %6 : f32
        } -> tensor<4x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg2, offsets = [0, 0, 0], sizes = [4, 2, 2], strides = [1, 1, 1] : tensor<4x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_3 {
    flow.executable.export public @conv_partial_load workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_partial_load(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x3x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [4, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x3x8xbf16>> -> tensor<4x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<4x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<4x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %6 = arith.extf %in : bf16 to f32
          %7 = arith.extf %in_0 : bf16 to f32
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<4x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [4, 2, 2], strides = [1, 1, 1] : tensor<4x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_4 {
    flow.executable.export public @conv_chain workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_chain(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xbf16>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [4, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>> -> tensor<4x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<4x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<4x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %9 = arith.extf %in : bf16 to f32
          %10 = arith.extf %in_0 : bf16 to f32
          %11 = arith.mulf %9, %10 : f32
          %12 = arith.addf %out, %11 : f32
          linalg.yield %12 : f32
        } -> tensor<4x2x2xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<4x2x2xf32>) outs(%2 : tensor<4x2x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.maximumf %in, %cst : f32
          linalg.yield %9 : f32
        } -> tensor<4x2x2xf32>
        %6 = tensor.empty() : tensor<4x2x2xbf16>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<4x2x2xf32>) outs(%6 : tensor<4x2x2xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %9 = arith.truncf %in : f32 to bf16
          linalg.yield %9 : bf16
        } -> tensor<4x2x2xbf16>
        iree_tensor_ext.dispatch.tensor.store %7, %arg2, offsets = [0, 0, 0], sizes = [4, 2, 2], strides = [1, 1, 1] : tensor<4x2x2xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xbf16>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<2x2x8xbf16>, %arg1: tensor<4x2x8xbf16>, %arg2: tensor<4x3x8xbf16>) -> (tensor<2x2x2xf32>, tensor<4x2x2xf32>, tensor<4x2x2xf32>, tensor<4x2x2xf32>, tensor<4x2x2xbf16>) {
    %w = util.global.load immutable @weight : tensor<1x1x8x2xbf16>
    %0 = flow.dispatch @dispatch_0::@conv_oh2(%arg0, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<2x2x2xf32>
    %1 = flow.dispatch @dispatch_1::@conv_no_native(%arg1, %w) {stream.affinity = #hal.device.affinity<@npu_default>} : (tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xf32>
    %2 = flow.dispatch @dispatch_2::@conv_reads_init(%arg1, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xf32>
    %3 = flow.dispatch @dispatch_3::@conv_partial_load(%arg2, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<4x3x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xf32>
    %4 = flow.dispatch @dispatch_4::@conv_chain(%arg1, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xbf16>
    util.return %0, %1, %2, %3, %4 : tensor<2x2x2xf32>, tensor<4x2x2xf32>, tensor<4x2x2xf32>, tensor<4x2x2xf32>, tensor<4x2x2xbf16>
  }
}

// -----

// Two convs reading the same filter global (IREE deduplicates equal
// constants): both are chunked, each gets its own FHWC copy, and the shared
// global, listed once per replacement, is erased once.

// CHECK-LABEL: util.global private @__conv_fhwc_weight_3
// CHECK-SAME:    tensor<2x1x1x8xbf16>
// CHECK-LABEL: util.global private @__conv_fhwc_weight_1
// CHECK-SAME:    tensor<2x1x1x8xbf16>
// CHECK-NOT:   util.global private @weight
// CHECK-LABEL: util.func public @main
// CHECK:         %[[A:.+]] = flow.dispatch @dispatch_0::@conv_a({{.*}}) {{.*}} -> tensor<2x2x2x2xf32>
// CHECK:         flow.tensor.reshape %[[A]] : tensor<2x2x2x2xf32> -> tensor<4x2x2xf32>
// CHECK:         %[[B:.+]] = flow.dispatch @dispatch_1::@conv_b({{.*}}) {{.*}} -> tensor<3x2x2x2xf32>
// CHECK:         flow.tensor.reshape %[[B]] : tensor<3x2x2x2xf32> -> tensor<6x2x2xf32>
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]]]> : tensor<1x1x8x2xbf16>
  flow.executable private @dispatch_0 {
    flow.executable.export public @conv_a workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_a(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [4, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2x8xbf16>> -> tensor<4x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<4x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<4x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %6 = arith.extf %in : bf16 to f32
          %7 = arith.extf %in_0 : bf16 to f32
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<4x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [4, 2, 2], strides = [1, 1, 1] : tensor<4x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_1 {
    flow.executable.export public @conv_b workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_b(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x2x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<6x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [6, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x2x8xbf16>> -> tensor<6x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<6x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<6x2x2xf32>) -> tensor<6x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<6x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<6x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %6 = arith.extf %in : bf16 to f32
          %7 = arith.extf %in_0 : bf16 to f32
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<6x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [6, 2, 2], strides = [1, 1, 1] : tensor<6x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<6x2x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<4x2x8xbf16>, %arg1: tensor<6x2x8xbf16>) -> (tensor<4x2x2xf32>, tensor<6x2x2xf32>) {
    %w = util.global.load immutable @weight : tensor<1x1x8x2xbf16>
    %0 = flow.dispatch @dispatch_0::@conv_a(%arg0, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<4x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<4x2x2xf32>
    %1 = flow.dispatch @dispatch_1::@conv_b(%arg1, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<6x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<6x2x2xf32>
    util.return %0, %1 : tensor<4x2x2xf32>, tensor<6x2x2xf32>
  }
}

// -----

// A 3x3 HWCF filter with distinct values: the FHWC copy holds w[kh][kw][ic][oc]
// at [oc][kh][kw][ic], kernel rows and columns in their original order.

// CHECK-LABEL: util.global private @__conv_fhwc_weight_1
// CHECK-SAME:    dense<{{\[\[\[\[}}1.000000e+00], [3.000000e+00], [5.000000e+00]], {{\[\[}}7.000000e+00], [9.000000e+00], [1.100000e+01]], {{\[\[}}1.300000e+01], [1.500000e+01], [1.700000e+01]]], {{\[\[\[}}2.000000e+00], [4.000000e+00], [6.000000e+00]], {{\[\[}}8.000000e+00], [1.000000e+01], [1.200000e+01]], {{\[\[}}1.400000e+01], [1.600000e+01], [1.800000e+01]]]]> : tensor<2x3x3x1xbf16>
// CHECK-LABEL: util.func public @main
// CHECK:         flow.dispatch @dispatch_0::@conv_3x3_values({{.*}}) {{.*}} -> tensor<2x2x2x2xf32>
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]], [[[7.0, 8.0]], [[9.0, 10.0]], [[11.0, 12.0]]], [[[13.0, 14.0]], [[15.0, 16.0]], [[17.0, 18.0]]]]> : tensor<3x3x1x2xbf16>
  flow.executable private @dispatch_0 {
    flow.executable.export public @conv_3x3_values workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_3x3_values(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x4x1xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [6, 4, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<6x4x1xbf16>> -> tensor<6x4x1xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1x2xbf16>> -> tensor<3x3x1x2xbf16>
        %2 = tensor.empty() : tensor<4x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<6x4x1xbf16>, tensor<3x3x1x2xbf16>) outs(%3 : tensor<4x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %5 = arith.extf %in : bf16 to f32
          %6 = arith.extf %in_0 : bf16 to f32
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<4x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [4, 2, 2], strides = [1, 1, 1] : tensor<4x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<6x4x1xbf16>) -> tensor<4x2x2xf32> {
    %w = util.global.load immutable @weight : tensor<3x3x1x2xbf16>
    %0 = flow.dispatch @dispatch_0::@conv_3x3_values(%arg0, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<6x4x1xbf16>, tensor<3x3x1x2xbf16>) -> tensor<4x2x2xf32>
    util.return %0 : tensor<4x2x2xf32>
  }
}

// -----

// OH = 5 has no divisor that gives two rows per chunk, so it is padded to 3
// chunks of 2 rows (one more column than 2 chunks of 3, and two fewer rows
// each than one chunk of 5). The gather reads one zero row past each input
// (a `tensor.pad` in the halo dispatch, still merged with the ones-channel
// pad), and the host takes the first 5 of the 6 rows the chunks write with a
// `flow.tensor.slice`: copied into the padded tensor for the tied 3x3 conv,
// returned for the 1x1 conv.

// CHECK-DAG:   #[[$ROWS:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d1, d2, d3)>
// CHECK-LABEL: func.func @conv_3x3_oh5
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x4x4x8xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x2x2x2xf32>>
// CHECK-LABEL: func.func @conv_1x1_oh5
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x2x2x8xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x2x2x2xf32>>
// CHECK-LABEL: func.func @halo_dispatch_2
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x4x7xbf16>>
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK:         %[[PADDED:.+]] = tensor.pad %{{.+}} low[0, 0, 0] high[1, 0, 0]
// CHECK:           tensor.yield %[[ZERO]] : bf16
// CHECK:         } : tensor<7x4x7xbf16> to tensor<8x4x7xbf16>
// CHECK:         linalg.generic {indexing_maps = [#[[$ROWS]], #{{.+}}]
// CHECK-SAME:      ins(%[[PADDED]] : tensor<8x4x7xbf16>) outs(%{{.+}} : tensor<3x4x4x7xbf16>)
// CHECK-LABEL: func.func @insert_dispatch_4
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x2x2xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %arg1, offsets = [1, 1, 0], sizes = [5, 2, 2]
// CHECK-LABEL: func.func @halo_dispatch_5
// CHECK:         tensor.pad %{{.+}} low[0, 0, 0] high[1, 0, 0]
// CHECK:         } : tensor<5x2x8xbf16> to tensor<6x2x8xbf16>
// CHECK-LABEL: util.func public @main
// CHECK:         %[[CHUNKS:.+]] = flow.dispatch @dispatch_0::@conv_3x3_oh5({{.+}}) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<3x4x4x8xbf16>, tensor<2x3x3x8xbf16>) -> tensor<3x2x2x2xf32>
// CHECK:         %[[ROWS:.+]] = flow.tensor.reshape %[[CHUNKS]] : tensor<3x2x2x2xf32> -> tensor<6x2x2xf32>
// CHECK:         %[[OUT:.+]] = flow.tensor.slice %[[ROWS]][%{{.+}}, %{{.+}}, %{{.+}} for %{{.+}}, %{{.+}}, %{{.+}}] : tensor<6x2x2xf32> -> tensor<5x2x2xf32>
// CHECK:         %[[PADDED:.+]] = flow.dispatch @insert_executable_4::@insert_dispatch_4(%[[OUT]], %arg1)
// CHECK:         %[[CHUNKS1:.+]] = flow.dispatch @dispatch_1::@conv_1x1_oh5({{.+}}) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<3x2x2x8xbf16>, tensor<2x1x1x8xbf16>) -> tensor<3x2x2x2xf32>
// CHECK:         %[[ROWS1:.+]] = flow.tensor.reshape %[[CHUNKS1]] : tensor<3x2x2x2xf32> -> tensor<6x2x2xf32>
// CHECK:         %[[OUT1:.+]] = flow.tensor.slice %[[ROWS1]]{{.+}} : tensor<6x2x2xf32> -> tensor<5x2x2xf32>
// CHECK:         util.return %[[PADDED]], %[[OUT1]]
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {native_conv = true, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @__bias_folded_weight_1 {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<1.0> : tensor<3x3x8x2xbf16>
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]]]> : tensor<1x1x8x2xbf16>
  flow.executable private @dispatch_0 {
    flow.executable.export public @conv_3x3_oh5 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_3x3_oh5(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x4x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<7x4x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [7, 4, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x4x8xbf16>> -> tensor<7x4x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [3, 3, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x8x2xbf16>> -> tensor<3x3x8x2xbf16>
        %2 = tensor.empty() : tensor<5x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<5x2x2xf32>) -> tensor<5x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<7x4x8xbf16>, tensor<3x3x8x2xbf16>) outs(%3 : tensor<5x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %6 = arith.extf %in : bf16 to f32
          %7 = arith.extf %in_0 : bf16 to f32
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %out, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<5x2x2xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<5x2x2xf32>) outs(%2 : tensor<5x2x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.cmpf ugt, %in, %cst : f32
          %7 = arith.select %6, %in, %cst : f32
          linalg.yield %7 : f32
        } -> tensor<5x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg2, offsets = [1, 1, 0], sizes = [5, 2, 2], strides = [1, 1, 1] : tensor<5x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<7x4x2xf32>>
        return
      }
    }
  }
  flow.executable private @pad_executable_0 {
    flow.executable.export public @pad_dispatch_0 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pad_dispatch_0(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x4x7xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<7x4x8xbf16>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [7, 4, 7], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<7x4x7xbf16>> -> tensor<7x4x7xbf16>
        iree_tensor_ext.dispatch.tensor.store %0, %arg1, offsets = [0, 0, 0], sizes = [7, 4, 7], strides = [1, 1, 1] : tensor<7x4x7xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<7x4x8xbf16>>
        return
      }
    }
  }
  flow.executable private @dispatch_1 {
    flow.executable.export public @conv_1x1_oh5 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_1x1_oh5(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x2x8xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [5, 2, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x2x8xbf16>> -> tensor<5x2x8xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 2], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x8x2xbf16>> -> tensor<1x1x8x2xbf16>
        %2 = tensor.empty() : tensor<5x2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<5x2x2xf32>) -> tensor<5x2x2xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d2)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %1 : tensor<5x2x8xbf16>, tensor<1x1x8x2xbf16>) outs(%3 : tensor<5x2x2xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %5 = arith.extf %in : bf16 to f32
          %6 = arith.extf %in_0 : bf16 to f32
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %out, %7 : f32
          linalg.yield %8 : f32
        } -> tensor<5x2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0], sizes = [5, 2, 2], strides = [1, 1, 1] : tensor<5x2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x2x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<7x4x7xbf16>, %arg1: tensor<7x4x2xf32>, %arg2: tensor<5x2x8xbf16>) -> (tensor<7x4x2xf32>, tensor<5x2x2xf32>) {
    %one = arith.constant 1.000000e+00 : bf16
    %ones = flow.tensor.splat %one : tensor<7x4x8xbf16> attributes {stream.affinity = #hal.device.affinity<@cpu>}
    %in = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0, %ones) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<7x4x7xbf16>, tensor<7x4x8xbf16>) -> %ones
    %w = util.global.load immutable @__bias_folded_weight_1 : tensor<3x3x8x2xbf16>
    %0 = flow.dispatch @dispatch_0::@conv_3x3_oh5(%in, %w, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<7x4x8xbf16>, tensor<3x3x8x2xbf16>, tensor<7x4x2xf32>) -> %arg1
    %w1 = util.global.load immutable @weight : tensor<1x1x8x2xbf16>
    %1 = flow.dispatch @dispatch_1::@conv_1x1_oh5(%arg2, %w1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<5x2x8xbf16>, tensor<1x1x8x2xbf16>) -> tensor<5x2x2xf32>
    util.return %0, %1 : tensor<7x4x2xf32>, tensor<5x2x2xf32>
  }
}
