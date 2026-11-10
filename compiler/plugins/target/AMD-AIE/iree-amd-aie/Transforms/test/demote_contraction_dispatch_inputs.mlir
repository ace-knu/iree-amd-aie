// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-demote-contraction-dispatch-inputs)' %s | FileCheck %s

// The NPU asks for bf16 contraction inputs. The matmul's LHS is the result of
// a host dispatch read only by the matmul (through a reshape): the conversion
// is fused into that dispatch's generic, which then writes bf16. The weight is
// a constant: it is converted at compile time (round to nearest even:
// 1.00390625 -> 1.0, 1.01171875 -> 1.015625) into a new global, and the f32
// global, read nowhere else, is dropped. The matmul accumulates and writes
// f32 as before.

// CHECK-LABEL: util.global private @__demoted_constant_0
// CHECK-SAME:    dense<{{\[\[}}1.000000e+00, 1.015630e+00], [2.000000e+00, -3.000000e+00]]> : tensor<2x2xbf16>
// CHECK-NOT:   util.global private @weight

// CHECK-LABEL: func.func @im2col
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xbf16>>
// CHECK:         %[[G:.+]] = linalg.generic
// CHECK-SAME:      outs(%{{.+}} : tensor<4x4xbf16>)
// CHECK:           %[[T:.+]] = arith.truncf %{{.+}} : f32 to bf16
// CHECK:           linalg.yield %[[T]] : bf16
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[G]], %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xbf16>

// CHECK-LABEL: func.func @mm
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<8x2xbf16>, tensor<2x2xbf16>) outs(%{{.+}} : tensor<8x2xf32>)

// CHECK-LABEL: util.func public @main
// CHECK:         %[[P:.+]] = flow.dispatch @im2col::@im2col(%arg0) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<4x4xf32>) -> tensor<4x4xbf16>
// CHECK:         %[[R:.+]] = flow.tensor.reshape %[[P]] : tensor<4x4xbf16> -> tensor<8x2xbf16>
// CHECK:         %[[W:.+]] = util.global.load immutable @__demoted_constant_0 : tensor<2x2xbf16>
// CHECK:         flow.dispatch @mm::@mm(%[[R]], %[[W]]) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xbf16>, tensor<2x2xbf16>) -> tensor<8x2xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {contraction_input_type = bf16, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight = dense<[[1.00390625, 1.01171875], [2.0, -3.0]]> : tensor<2x2xf32>
  flow.executable private @im2col {
    flow.executable.export public @im2col workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @im2col(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = tensor.empty() : tensor<4x4xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  flow.executable private @mm {
    flow.executable.export public @mm workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>> -> tensor<8x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = tensor.empty() : tensor<8x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8x2xf32>) -> tensor<8x2xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<8x2xf32>, tensor<2x2xf32>) outs(%3 : tensor<8x2xf32>) -> tensor<8x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : tensor<8x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<4x4xf32>) -> tensor<8x2xf32> {
    %0 = flow.dispatch @im2col::@im2col(%arg0) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = flow.tensor.reshape %0 : tensor<4x4xf32> -> tensor<8x2xf32>
    %w = util.global.load immutable @weight : tensor<2x2xf32>
    %2 = flow.dispatch @mm::@mm(%1, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
    util.return %2 : tensor<8x2xf32>
  }
}

// -----

// Two NPU matmuls share one executable, converted once. The first reads a
// model input, the second a host result that is also returned in f32: neither
// can be converted inside a producer, so each gets a host cast dispatch, and
// the f32 value stays for its other reader.

// CHECK-LABEL: func.func @mm
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<8x2xbf16>, tensor<2x2xbf16>) outs(%{{.+}} : tensor<8x2xf32>)

// CHECK-LABEL: func.func @relu
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>

// CHECK:       flow.executable private @cast_executable_0
// CHECK:         func.func @cast_dispatch_0(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xbf16>>)
// CHECK:           arith.truncf %{{.+}} : f32 to bf16
// CHECK:       flow.executable private @cast_executable_1
// CHECK:       flow.executable private @cast_executable_2

// CHECK-LABEL: util.func public @main
// CHECK-DAG:     %[[A:.+]] = flow.dispatch @cast_executable_0::@cast_dispatch_0(%arg0) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<8x2xf32>) -> tensor<8x2xbf16>
// CHECK-DAG:     %[[B:.+]] = flow.dispatch @cast_executable_1::@cast_dispatch_1(%arg1) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<2x2xf32>) -> tensor<2x2xbf16>
// CHECK:         %[[M1:.+]] = flow.dispatch @mm::@mm(%[[A]], %[[B]]) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xbf16>, tensor<2x2xbf16>) -> tensor<8x2xf32>
// CHECK:         %[[RELU:.+]] = flow.dispatch @relu::@relu(%[[M1]]) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<8x2xf32>) -> tensor<8x2xf32>
// CHECK:         %[[C:.+]] = flow.dispatch @cast_executable_2::@cast_dispatch_2(%[[RELU]]) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<8x2xf32>) -> tensor<8x2xbf16>
// CHECK:         %[[M2:.+]] = flow.dispatch @mm::@mm(%[[C]], %[[B]]) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xbf16>, tensor<2x2xbf16>) -> tensor<8x2xf32>
// CHECK:         util.return %[[RELU]], %[[M2]] : tensor<8x2xf32>, tensor<8x2xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {contraction_input_type = bf16, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @mm {
    flow.executable.export public @mm workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>> -> tensor<8x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = tensor.empty() : tensor<8x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8x2xf32>) -> tensor<8x2xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<8x2xf32>, tensor<2x2xf32>) outs(%3 : tensor<8x2xf32>) -> tensor<8x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : tensor<8x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>
        return
      }
    }
  }
  flow.executable private @relu {
    flow.executable.export public @relu workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @relu(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>> -> tensor<8x2xf32>
        %1 = tensor.empty() : tensor<8x2xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<8x2xf32>) outs(%1 : tensor<8x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.maximumf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<8x2xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : tensor<8x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<8x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<8x2xf32>, tensor<8x2xf32>) {
    %0 = flow.dispatch @mm::@mm(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
    %1 = flow.dispatch @relu::@relu(%0) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<8x2xf32>) -> tensor<8x2xf32>
    %2 = flow.dispatch @mm::@mm(%1, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
    util.return %1, %2 : tensor<8x2xf32>, tensor<8x2xf32>
  }
}

// -----

// One executable called on the NPU (asks for bf16) and on a device that does
// not ask (here the CPU, as it would be a GPU): the NPU caller gets its own
// copy of the executable to convert, the other caller keeps the f32 one. The
// weight both read is converted for the NPU only; its f32 global stays for the
// other caller.

// CHECK-LABEL: util.global private @__demoted_constant_{{[0-9]+}}
// CHECK-SAME:    tensor<2x2xbf16>
// CHECK:       util.global private @weight
// CHECK-SAME:    tensor<2x2xf32>

// CHECK:       flow.executable private @mm {
// CHECK:         func.func @mm
// CHECK-SAME:      %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<8x2xf32>, tensor<2x2xf32>) outs(%{{.+}} : tensor<8x2xf32>)
// CHECK:       flow.executable private @[[COPY:mm_[0-9]+]] {
// CHECK:         func.func @mm
// CHECK-SAME:      %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xbf16>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<8x2xbf16>, tensor<2x2xbf16>) outs(%{{.+}} : tensor<8x2xf32>)

// CHECK-LABEL: util.func public @main
// CHECK-DAG:     %[[W:.+]] = util.global.load immutable @weight : tensor<2x2xf32>
// CHECK-DAG:     %[[WB:.+]] = util.global.load immutable @__demoted_constant_{{[0-9]+}} : tensor<2x2xbf16>
// CHECK-DAG:     %[[A:.+]] = flow.dispatch @cast_executable_{{[0-9]+}}::@cast_dispatch_{{[0-9]+}}(%arg0)
// CHECK:         flow.dispatch @[[COPY]]::@mm(%[[A]], %[[WB]]) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xbf16>, tensor<2x2xbf16>) -> tensor<8x2xf32>
// CHECK:         flow.dispatch @mm::@mm(%arg1, %[[W]]) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<8x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {contraction_input_type = bf16, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  flow.executable private @mm {
    flow.executable.export public @mm workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>> -> tensor<8x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = tensor.empty() : tensor<8x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8x2xf32>) -> tensor<8x2xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<8x2xf32>, tensor<2x2xf32>) outs(%3 : tensor<8x2xf32>) -> tensor<8x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : tensor<8x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> (tensor<8x2xf32>, tensor<8x2xf32>) {
    %w = util.global.load immutable @weight : tensor<2x2xf32>
    %0 = flow.dispatch @mm::@mm(%arg0, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
    %1 = flow.dispatch @mm::@mm(%arg1, %w) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<8x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
    util.return %0, %1 : tensor<8x2xf32>, tensor<8x2xf32>
  }
}

// -----

// No device asks for a type (no `contraction_input_type`): nothing changes.

// CHECK-LABEL: util.func public @main
// CHECK-NOT:     cast_executable
// CHECK:         flow.dispatch @mm::@mm(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @mm {
    flow.executable.export public @mm workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x2xf32>> -> tensor<8x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = tensor.empty() : tensor<8x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<8x2xf32>) -> tensor<8x2xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<8x2xf32>, tensor<2x2xf32>) outs(%3 : tensor<8x2xf32>) -> tensor<8x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [8, 2], strides = [1, 1] : tensor<8x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<8x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<8x2xf32> {
    %0 = flow.dispatch @mm::@mm(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<8x2xf32>, tensor<2x2xf32>) -> tensor<8x2xf32>
    util.return %0 : tensor<8x2xf32>
  }
}

// -----

// A binding is converted only when contractions are its only readers. In one
// NPU dispatch, matmul A reads %arg0 and %arg1 alone: converted. Matmul B's
// %arg3 is also read by an add: B stays f32. Matmul C shares %arg2 with B, so
// once B is out C is dropped too (the last candidate, in a second round), and
// %arg2 and %arg4 stay f32. Only A's inputs get cast dispatches.

// CHECK-LABEL: func.func @mm3
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>
// CHECK-SAME:    %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>
// CHECK-SAME:    %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x2xbf16>, tensor<2x2xbf16>)
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x2xf32>, tensor<2x2xf32>)
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x2xf32>, tensor<2x2xf32>)

// CHECK-LABEL: util.func public @main
// CHECK-DAG:     %[[A:.+]] = flow.dispatch @cast_executable_{{[0-9]+}}::@cast_dispatch_{{[0-9]+}}(%arg0)
// CHECK-DAG:     %[[B:.+]] = flow.dispatch @cast_executable_{{[0-9]+}}::@cast_dispatch_{{[0-9]+}}(%arg1)
// CHECK-NOT:     cast_executable
// CHECK:         flow.dispatch @mm3::@mm3(%[[A]], %[[B]], %arg2, %arg3, %arg4) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
module attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {contraction_input_type = bf16, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @mm3 {
    flow.executable.export public @mm3 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm3(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg4: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg5: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>, %arg6: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>, %arg7: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>, %arg8: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %3 = iree_tensor_ext.dispatch.tensor.load %arg3, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %arg4, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %e = tensor.empty() : tensor<2x2xf32>
        %f = linalg.fill ins(%cst : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
        %a = linalg.matmul ins(%0, %1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
        %b = linalg.matmul ins(%2, %3 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
        %c = linalg.matmul ins(%4, %2 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%f : tensor<2x2xf32>) -> tensor<2x2xf32>
        %d = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<2x2xf32>) outs(%e : tensor<2x2xf32>) {
        ^bb0(%in: f32, %out: f32):
          %s = arith.addf %in, %in : f32
          linalg.yield %s : f32
        } -> tensor<2x2xf32>
        iree_tensor_ext.dispatch.tensor.store %a, %arg5, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        iree_tensor_ext.dispatch.tensor.store %b, %arg6, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        iree_tensor_ext.dispatch.tensor.store %c, %arg7, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        iree_tensor_ext.dispatch.tensor.store %d, %arg8, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : tensor<2x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x2xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>, %arg3: tensor<2x2xf32>, %arg4: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
    %0:4 = flow.dispatch @mm3::@mm3(%arg0, %arg1, %arg2, %arg3, %arg4) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
    util.return %0#0, %0#1, %0#2, %0#3 : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>
  }
}
