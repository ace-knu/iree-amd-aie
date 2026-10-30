// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(iree-amdaie-fold-bias-into-contraction-dispatches)' %s | FileCheck %s

// FC layer (ONNX Gemm, transpose_b weight [N, K]) with a fused bias + ReLU +
// truncf: the bias moves into K. The executable loses the bias binding and
// `addf` and reads [1, K+1] x [N, K+1]; the host pads the LHS with a ones
// column (a splat of 1 the pad dispatch copies the LHS into) and loads a new weight global whose last K column is
// the bias, rounded to bf16. The original weight global is dropped.

// CHECK-LABEL: util.global private @__bias_folded_weight_1
// CHECK-SAME:    {stream.affinity.default = #hal.device.affinity<@cpu>}
// CHECK-SAME:    dense<{{\[}}[1.000000e+00, 2.000000e+00, 5.000000e-01], [3.000000e+00, 4.000000e+00, -2.500000e-01]]> : tensor<2x3xbf16>
// CHECK-NOT:   @weight
// CHECK-LABEL: func.func @fc
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xbf16>>
// CHECK:         %[[MM:.+]] = linalg.matmul {{.*}} ins(%{{.+}}, %{{.+}} : tensor<1x3xbf16>, tensor<2x3xbf16>)
// CHECK:         linalg.generic {{.*}} ins(%[[MM]] : tensor<1x2xf32>)
// CHECK-NOT:       arith.addf
// CHECK:           arith.cmpf
// CHECK:           arith.truncf
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x3xbf16>>
// CHECK-LABEL: util.func public @main
// CHECK:         %[[ONE:.+]] = arith.constant 1.000000e+00 : bf16
// CHECK:         %[[ONES:.+]] = flow.tensor.splat %[[ONE]] : tensor<1x3xbf16> attributes {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK:         %[[LHS:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0, %[[ONES]]) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<1x2xbf16>, tensor<1x3xbf16>) -> %[[ONES]]
// CHECK:         %[[W:.+]] = util.global.load immutable @__bias_folded_weight_1 : tensor<2x3xbf16>
// CHECK:         flow.dispatch @dispatch_0::@fc(%[[LHS]], %[[W]]) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x3xbf16>, tensor<2x3xbf16>) -> tensor<1x2xbf16>
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xbf16>
  util.global private @bias {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[0.5, -0.25]]> : tensor<1x2xf32>
  flow.executable private @dispatch_0 {
    flow.executable.export public @fc workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xbf16>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xbf16>> -> tensor<1x2xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>> -> tensor<2x2xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>> -> tensor<1x2xf32>
        %3 = tensor.empty() : tensor<1x2xbf16>
        %4 = tensor.empty() : tensor<1x2xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %6 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x2xbf16>, tensor<2x2xbf16>) outs(%5 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %2 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%3 : tensor<1x2xbf16>) {
        ^bb0(%in: f32, %in_0: f32, %out: bf16):
          %8 = arith.addf %in, %in_0 : f32
          %9 = arith.cmpf ugt, %8, %cst : f32
          %10 = arith.select %9, %8, %cst : f32
          %11 = arith.truncf %10 : f32 to bf16
          linalg.yield %11 : bf16
        } -> tensor<1x2xbf16>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xbf16>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<1x2xbf16>) -> tensor<1x2xbf16> {
    %w = util.global.load immutable @weight : tensor<2x2xbf16>
    %b = util.global.load immutable @bias : tensor<1x2xf32>
    %0 = flow.dispatch @dispatch_0::@fc(%arg0, %w, %b) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x2xbf16>, tensor<2x2xbf16>, tensor<1x2xf32>) -> tensor<1x2xbf16>
    util.return %0 : tensor<1x2xbf16>
  }
}

// -----

// A bias add alone (the classifier FC) leaves nothing for the consumer to do:
// it is removed and the matmul result is stored directly. Here the weight is
// [K, N] and the bias a 1-D [N] read through (d0, d1) -> (d1), so the bias is
// the weight's new last row.

// CHECK-LABEL: util.global private @__bias_folded_weight_1
// CHECK-SAME:    dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e-01, -2.500000e-01]]> : tensor<3x2xf32>
// CHECK-LABEL: func.func @fc_add_only
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x3xf32>, tensor<3x2xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[MM]]
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @fc_add_only workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc_add_only(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %3 = tensor.empty() : tensor<2x2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x2xf32>) -> tensor<2x2xf32>
        %5 = linalg.matmul ins(%0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%4 : tensor<2x2xf32>) -> tensor<2x2xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<2x2xf32>, tensor<2xf32>) outs(%3 : tensor<2x2xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %7 = arith.addf %in, %in_0 : f32
          linalg.yield %7 : f32
        } -> tensor<2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %w = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %b = arith.constant dense<[0.5, -0.25]> : tensor<2xf32>
    %0 = flow.dispatch @dispatch_0::@fc_add_only(%arg0, %w, %b) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    util.return %0 : tensor<2x2xf32>
  }
}

// -----

// A bias that is not a compile-time constant cannot be folded; the dispatch
// would reach the backend with three inputs, so the pass fails here.
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @fc_runtime_bias workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc_runtime_bias(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %3 = tensor.empty() : tensor<2x2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x2xf32>) -> tensor<2x2xf32>
        %5 = linalg.matmul ins(%0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%4 : tensor<2x2xf32>) -> tensor<2x2xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<2x2xf32>, tensor<2xf32>) outs(%3 : tensor<2x2xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %7 = arith.addf %in, %in_0 : f32
          linalg.yield %7 : f32
        } -> tensor<2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2xf32>) -> tensor<2x2xf32> {
    %w = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    // expected-error @+1 {{cannot fold the bias of this NPU contraction dispatch into K: the bias is not a float constant}}
    %0 = flow.dispatch @dispatch_0::@fc_runtime_bias(%arg0, %w, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    util.return %0 : tensor<2x2xf32>
  }
}

// -----

// A dynamic M (e.g. batch) cannot be folded: the ones column and the K+1 LHS
// are built with static shapes.
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @fc_dynamic_m workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc_dynamic_m(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>, %arg3: index, %arg4: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %m = iree_tensor_ext.dispatch.workload.ordinal %arg3, 0 : index
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [%m, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2xf32>>{%m} -> tensor<?x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %3 = tensor.empty(%m) : tensor<?x2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<?x2xf32>) -> tensor<?x2xf32>
        %5 = linalg.matmul ins(%0, %1 : tensor<?x2xf32>, tensor<2x2xf32>) outs(%4 : tensor<?x2xf32>) -> tensor<?x2xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<?x2xf32>, tensor<2xf32>) outs(%3 : tensor<?x2xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %7 = arith.addf %in, %in_0 : f32
          linalg.yield %7 : f32
        } -> tensor<?x2xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg4, offsets = [0, 0], sizes = [%m, 2], strides = [1, 1] : tensor<?x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2xf32>>{%m}
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<?x2xf32>, %m: index) -> tensor<?x2xf32> {
    %w = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %b = arith.constant dense<[5.0, 6.0]> : tensor<2xf32>
    // expected-error @+1 {{cannot fold the bias of this NPU contraction dispatch into K: the dispatch has dynamically shaped operands or results}}
    %0 = flow.dispatch @dispatch_0::@fc_dynamic_m[%m](%arg0, %w, %b, %m) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<?x2xf32>{%m}, tensor<2x2xf32>, tensor<2xf32>, index) -> tensor<?x2xf32>{%m}
    util.return %0 : tensor<?x2xf32>
  }
}

// -----

// im2col convolution: the matmul accumulates onto a host dispatch that
// broadcasts a constant [N] bias over M. The matmul starts from fill(0)
// instead, the bias becomes the weight's last K row (the weight read through a
// reshape, which goes with it), and the broadcast dispatch is dropped.

// CHECK-LABEL: util.global private @__bias_folded_weight_1
// CHECK-SAME:    dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e-01, -2.500000e-01]]> : tensor<3x2xbf16>
// CHECK-NOT:   @weight
// CHECK-LABEL: func.func @conv
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x3xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x2xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>
// CHECK:         %[[FILL:.+]] = linalg.fill
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<4x3xbf16>, tensor<3x2xbf16>) outs(%[[FILL]] : tensor<4x2xf32>)
// CHECK:         linalg.generic {{.*}} ins(%[[MM]] : tensor<4x2xf32>)
// CHECK-LABEL: util.func public @main
// CHECK-NOT:     @broadcast
// CHECK-NOT:     flow.tensor.reshape
// CHECK:         %[[LHS:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0, %{{[a-z0-9_]+}})
// CHECK:         %[[W:.+]] = util.global.load immutable @__bias_folded_weight_1 : tensor<3x2xbf16>
// CHECK:         flow.dispatch @dispatch_0::@conv(%[[LHS]], %[[W]]) {{.*}} : (tensor<4x3xbf16>, tensor<3x2xbf16>) -> tensor<4x2xf32>
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[[1.0, 2.0]], [[3.0, 4.0]]]> : tensor<2x1x2xbf16>
  util.global private @bias {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[0.5, -0.25]> : tensor<2xf32>
  flow.executable private @broadcast {
    flow.executable.export public @broadcast workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @broadcast(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %1 = tensor.empty() : tensor<4x2xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<2xf32>) outs(%1 : tensor<4x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x2xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : tensor<4x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_0 {
    flow.executable.export public @conv workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xbf16>> -> tensor<4x2xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>> -> tensor<2x2xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xf32>> -> tensor<4x2xf32>
        %3 = tensor.empty() : tensor<4x2xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<4x2xbf16>, tensor<2x2xbf16>) outs(%2 : tensor<4x2xf32>) -> tensor<4x2xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4x2xf32>) outs(%3 : tensor<4x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.maximumf %in, %cst : f32
          linalg.yield %6 : f32
        } -> tensor<4x2xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg3, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : tensor<4x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<4x2xbf16>) -> tensor<4x2xf32> {
    %b = util.global.load immutable @bias : tensor<2xf32>
    %acc = flow.dispatch @broadcast::@broadcast(%b) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<2xf32>) -> tensor<4x2xf32>
    %w3 = util.global.load immutable @weight : tensor<2x1x2xbf16>
    %w = flow.tensor.reshape %w3 : tensor<2x1x2xbf16> -> tensor<2x2xbf16>
    %0 = flow.dispatch @dispatch_0::@conv(%arg0, %w, %acc) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<4x2xbf16>, tensor<2x2xbf16>, tensor<4x2xf32>) -> tensor<4x2xf32>
    util.return %0 : tensor<4x2xf32>
  }
}

// -----

// The broadcast may instead be hoisted by const-eval into a global that a
// `util.initializer` sets, here from an inline constant; the global load is
// the accumulator.

// CHECK-LABEL: util.global private @__bias_folded_weight_1
// CHECK-SAME:    dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e-01, -2.500000e-01]]> : tensor<3x2xbf16>
// CHECK-LABEL: util.func public @main
// CHECK-NOT:     @acc_global
// CHECK:         flow.dispatch @dispatch_0::@conv(%{{.+}}, %{{.+}}) {{.*}} : (tensor<4x3xbf16>, tensor<3x2xbf16>) -> tensor<4x2xf32>
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xbf16>
  util.global private @acc_global {stream.affinity.default = #hal.device.affinity<@cpu>} : tensor<4x2xf32>
  flow.executable private @broadcast {
    flow.executable.export public @broadcast workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @broadcast(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>) {
        %cst = arith.constant dense<[0.5, -0.25]> : tensor<2xf32>
        %1 = tensor.empty() : tensor<4x2xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<2xf32>) outs(%1 : tensor<4x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x2xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg0, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : tensor<4x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>
        return
      }
    }
  }
  util.initializer attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
    %0 = flow.dispatch @broadcast::@broadcast() {stream.affinity = #hal.device.affinity<@cpu>} : () -> tensor<4x2xf32>
    util.global.store %0, @acc_global : tensor<4x2xf32>
    util.return
  }
  flow.executable private @dispatch_0 {
    flow.executable.export public @conv workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xbf16>> -> tensor<4x2xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>> -> tensor<2x2xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xf32>> -> tensor<4x2xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<4x2xbf16>, tensor<2x2xbf16>) outs(%2 : tensor<4x2xf32>) -> tensor<4x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg3, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : tensor<4x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<4x2xbf16>) -> tensor<4x2xf32> {
    %w = util.global.load immutable @weight : tensor<2x2xbf16>
    %acc = util.global.load immutable @acc_global : tensor<4x2xf32>
    %0 = flow.dispatch @dispatch_0::@conv(%arg0, %w, %acc) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<4x2xbf16>, tensor<2x2xbf16>, tensor<4x2xf32>) -> tensor<4x2xf32>
    util.return %0 : tensor<4x2xf32>
  }
}

// -----

// A runtime accumulator is no broadcast bias: the dispatch would reach the
// backend with three inputs, so the pass fails here.
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @acc workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @acc(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xf32>> -> tensor<4x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x2xf32>> -> tensor<4x2xf32>
        %3 = linalg.matmul ins(%0, %1 : tensor<4x2xf32>, tensor<2x2xf32>) outs(%2 : tensor<4x2xf32>) -> tensor<4x2xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0, 0], sizes = [4, 2], strides = [1, 1] : tensor<4x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x2xf32> {
    %w = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    // expected-error @+1 {{cannot fold the bias of this NPU contraction dispatch into K: its accumulator input is not a broadcast of a float constant bias (a residual or partial sum cannot be folded)}}
    %0 = flow.dispatch @dispatch_0::@acc(%arg0, %w, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
    util.return %0 : tensor<4x2xf32>
  }
}

// -----

// A weight and bias stored as `dense_resource` blobs (as an ONNX import stores
// large initializers) are folded like inline constants.

// CHECK-LABEL: util.global private @__bias_folded_weight_{{[0-9]+}}
// CHECK-SAME:    dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e-01, -2.500000e-01]]> : tensor<3x2xf32>
// CHECK-LABEL: func.func @fc_resource_weight
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x3xf32>, tensor<3x2xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[MM]]
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @fc_resource_weight workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc_resource_weight(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %3 = tensor.empty() : tensor<2x2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x2xf32>) -> tensor<2x2xf32>
        %5 = linalg.matmul ins(%0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%4 : tensor<2x2xf32>) -> tensor<2x2xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %2 : tensor<2x2xf32>, tensor<2xf32>) outs(%3 : tensor<2x2xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %7 = arith.addf %in, %in_0 : f32
          linalg.yield %7 : f32
        } -> tensor<2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg3, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %w = arith.constant dense_resource<w> : tensor<2x2xf32>
    %b = arith.constant dense_resource<b> : tensor<2xf32>
    %0 = flow.dispatch @dispatch_0::@fc_resource_weight(%arg0, %w, %b) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
    util.return %0 : tensor<2x2xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040",
      b: "0x040000000000003F000080BE"
    }
  }
#-}
