// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-pad-contraction-dispatches)' %s | FileCheck %s

// A matmul dispatch pinned to the amd-aie (npu4) device whose reduction dim
// K=27 is not a multiple of the L2 reduction tile (32) has its operands padded
// to K=32: the executable bindings/loads/matmul grow, and each host operand is
// zero-padded by a separate padding dispatch placed on the host (CPU). The
// output (K is a reduction dim) is unchanged.

// CHECK-LABEL: func.func @matmul
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<256x32xbf16>, tensor<32x64xbf16>) outs(%{{.+}} : tensor<256x64xf32>)

// The padding dispatches zero-pad each operand to K=32 (strided insert via
// tensor.pad inside a dispatch).
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad

// CHECK-LABEL: util.func public @main
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<256x27xbf16>) -> tensor<256x32xbf16>
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<27x64xbf16>) -> tensor<32x64xbf16>
// CHECK:         flow.dispatch @dispatch_2::@matmul(%[[LPAD]], %[[RPAD]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@npu>}
// CHECK-SAME:      : (tensor<256x32xbf16>, tensor<32x64xbf16>) -> tensor<256x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_2 {
    flow.executable.export public @matmul workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x27xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [256, 27], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x27xbf16>> -> tensor<256x27xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [27, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>> -> tensor<27x64xbf16>
        %2 = tensor.empty() : tensor<256x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<256x27xbf16>, tensor<27x64xbf16>) outs(%3 : tensor<256x64xf32>) -> tensor<256x64xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : tensor<256x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<256x27xbf16>, %arg1: tensor<27x64xbf16>) -> tensor<256x64xf32> {
    %0 = flow.dispatch @dispatch_2::@matmul(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<256x27xbf16>, tensor<27x64xbf16>) -> tensor<256x64xf32>
    util.return %0 : tensor<256x64xf32>
  }
}

// -----

// A plain matmul whose output row dim M=196 is not a multiple of the M tile
// (num_rows*instrM = 4*8 = 32) is padded to M=224. Unlike K, M is an output
// dim: the LHS grows on M, the output init (empty/fill), matmul result and
// output binding grow to [224,128], and the padded dispatch result is cropped
// back to [196,128] by an extract_slice dispatch on the host. N=128 and K=64 are
// already divisible, so only the LHS gets a padding dispatch.

// CHECK-LABEL: func.func @matmul_m
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x64xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<224x128xf32>>
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<224x64xbf16>, tensor<64x128xbf16>) outs(%{{.+}} : tensor<224x128xf32>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[MM]], %arg2, offsets = [0, 0], sizes = [224, 128]

// The LHS is zero-padded (M 196 -> 224); the RHS is untouched; the padded
// result is cropped back with an extract_slice dispatch on the host.
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[28, 0]
// CHECK-NOT:   flow.executable private @pad_executable_1
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [196, 128] [1, 1]

// CHECK-LABEL: util.func public @main_m
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<196x64xbf16>) -> tensor<224x64xbf16>
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_m::@matmul_m(%[[LPAD]], %arg1)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@npu>}
// CHECK-SAME:      : (tensor<224x64xbf16>, tensor<64x128xbf16>) -> tensor<224x128xf32>
// CHECK:         flow.dispatch @crop_executable_1::@crop_dispatch_1(%[[MMR]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<224x128xf32>) -> tensor<196x128xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_m {
    flow.executable.export public @matmul_m workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_m(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>> -> tensor<196x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xbf16>> -> tensor<64x128xbf16>
        %2 = tensor.empty() : tensor<196x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<196x128xf32>) -> tensor<196x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<196x64xbf16>, tensor<64x128xbf16>) outs(%3 : tensor<196x128xf32>) -> tensor<196x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [196, 128], strides = [1, 1] : tensor<196x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x128xf32>>
        return
      }
    }
  }
  util.func public @main_m(%arg0: tensor<196x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<196x128xf32> {
    %0 = flow.dispatch @dispatch_m::@matmul_m(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x64xbf16>, tensor<64x128xbf16>) -> tensor<196x128xf32>
    util.return %0 : tensor<196x128xf32>
  }
}

// -----

// A plain matmul whose inner output dim N=200 is not a multiple of the N tile
// (num_cols*instrN = 8*8 = 64) is padded to N=256. N is the inner output dim:
// the RHS grows on N, the output init/matmul/store grow to [256,256], and the
// padded result is cropped back to [256,200] with a strided extract_slice
// dispatch on the host. M=256 and K=64 are already divisible, so only the RHS
// gets a padding dispatch.

// CHECK-LABEL: func.func @matmul_n
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x256xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x256xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<256x64xbf16>, tensor<64x256xbf16>) outs(%{{.+}} : tensor<256x256xf32>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %arg2, offsets = [0, 0], sizes = [256, 256]

// Only the RHS is zero-padded (N 200 -> 256); the result is cropped back on N.
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[0, 56]
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [256, 200] [1, 1]

// CHECK-LABEL: util.func public @main_n
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg1)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<64x200xbf16>) -> tensor<64x256xbf16>
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_n::@matmul_n(%arg0, %[[RPAD]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@npu>}
// CHECK-SAME:      : (tensor<256x64xbf16>, tensor<64x256xbf16>) -> tensor<256x256xf32>
// CHECK:         flow.dispatch @crop_executable_1::@crop_dispatch_1(%[[MMR]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<256x256xf32>) -> tensor<256x200xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_n {
    flow.executable.export public @matmul_n workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_n(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x200xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x200xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>> -> tensor<256x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 200], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x200xbf16>> -> tensor<64x200xbf16>
        %2 = tensor.empty() : tensor<256x200xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256x200xf32>) -> tensor<256x200xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<256x64xbf16>, tensor<64x200xbf16>) outs(%3 : tensor<256x200xf32>) -> tensor<256x200xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [256, 200], strides = [1, 1] : tensor<256x200xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x200xf32>>
        return
      }
    }
  }
  util.func public @main_n(%arg0: tensor<256x64xbf16>, %arg1: tensor<64x200xbf16>) -> tensor<256x200xf32> {
    %0 = flow.dispatch @dispatch_n::@matmul_n(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<256x64xbf16>, tensor<64x200xbf16>) -> tensor<256x200xf32>
    util.return %0 : tensor<256x200xf32>
  }
}

// -----

// A plain matmul with BOTH output dims non-divisible: M=196 -> 224 (outer) and
// N=200 -> 256 (inner). LHS and RHS each get a padding dispatch, the output
// grows to [224,256], and a single extract_slice dispatch crops both dims back
// to [196,200] on the host.

// CHECK-LABEL: func.func @matmul_mn
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x64xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x256xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<224x256xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<224x64xbf16>, tensor<64x256xbf16>) outs(%{{.+}} : tensor<224x256xf32>)

// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[28, 0]
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[0, 56]
// CHECK:       flow.executable private @crop_executable_2
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [196, 200] [1, 1]

// CHECK-LABEL: util.func public @main_mn
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      : (tensor<196x64xbf16>) -> tensor<224x64xbf16>
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1)
// CHECK-SAME:      : (tensor<64x200xbf16>) -> tensor<64x256xbf16>
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_mn::@matmul_mn(%[[LPAD]], %[[RPAD]])
// CHECK-SAME:      : (tensor<224x64xbf16>, tensor<64x256xbf16>) -> tensor<224x256xf32>
// CHECK:         flow.dispatch @crop_executable_2::@crop_dispatch_2(%[[MMR]])
// CHECK-SAME:      : (tensor<224x256xf32>) -> tensor<196x200xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_mn {
    flow.executable.export public @matmul_mn workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_mn(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x200xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x200xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>> -> tensor<196x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 200], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x200xbf16>> -> tensor<64x200xbf16>
        %2 = tensor.empty() : tensor<196x200xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<196x200xf32>) -> tensor<196x200xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<196x64xbf16>, tensor<64x200xbf16>) outs(%3 : tensor<196x200xf32>) -> tensor<196x200xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [196, 200], strides = [1, 1] : tensor<196x200xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x200xf32>>
        return
      }
    }
  }
  util.func public @main_mn(%arg0: tensor<196x64xbf16>, %arg1: tensor<64x200xbf16>) -> tensor<196x200xf32> {
    %0 = flow.dispatch @dispatch_mn::@matmul_mn(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x64xbf16>, tensor<64x200xbf16>) -> tensor<196x200xf32>
    util.return %0 : tensor<196x200xf32>
  }
}

// -----

// A matmul whose K=64 is already a multiple of 32 is left untouched (no padding
// dispatch, no shape change) -- the pass is a no-op for divisible dispatches.

// CHECK-NOT:   @pad_executable
// CHECK-NOT:   @crop_executable
// CHECK-LABEL: func.func @matmul_ok
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>>
// CHECK-LABEL: util.func public @main_ok
// CHECK:         flow.dispatch @dispatch_ok::@matmul_ok(%arg0, %arg1)
// CHECK-SAME:      : (tensor<256x64xbf16>, tensor<64x64xbf16>) -> tensor<256x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_ok {
    flow.executable.export public @matmul_ok workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_ok(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>> -> tensor<256x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>> -> tensor<64x64xbf16>
        %2 = tensor.empty() : tensor<256x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<256x64xbf16>, tensor<64x64xbf16>) outs(%3 : tensor<256x64xf32>) -> tensor<256x64xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : tensor<256x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
        return
      }
    }
  }
  util.func public @main_ok(%arg0: tensor<256x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<256x64xf32> {
    %0 = flow.dispatch @dispatch_ok::@matmul_ok(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<256x64xbf16>, tensor<64x64xbf16>) -> tensor<256x64xf32>
    util.return %0 : tensor<256x64xf32>
  }
}

// -----

// A transpose_b matmul (RHS = [N, K], the layout an ONNX Gemm lowers to) whose
// inner output dim N=200 is not a multiple of the N tile (8*8 = 64) is padded to
// N=256. The N and K dims are read from the matmul's indexing maps, so N is taken
// from RHS dim 0 (not mistaken for K at RHS dim 1): the RHS grows on its N dim
// (dim 0), the output grows to [32,256] and is cropped back to [32,200]. M=32 and
// K=64 are already divisible, so only the RHS gets a padding dispatch.

// CHECK-LABEL: func.func @matmul_tb
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x256xf32>>
// CHECK:         linalg.matmul {{.*}}ins(%{{.+}}, %{{.+}} : tensor<32x64xbf16>, tensor<256x64xbf16>) outs(%{{.+}} : tensor<32x256xf32>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %arg2, offsets = [0, 0], sizes = [32, 256]

// Only the RHS is zero-padded (N 200 -> 256 on its dim 0); the result is cropped
// back on N.
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[56, 0]
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [32, 200] [1, 1]

// CHECK-LABEL: util.func public @main_tb
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg1)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<200x64xbf16>) -> tensor<256x64xbf16>
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_tb::@matmul_tb(%arg0, %[[RPAD]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@npu>}
// CHECK-SAME:      : (tensor<32x64xbf16>, tensor<256x64xbf16>) -> tensor<32x256xf32>
// CHECK:         flow.dispatch @crop_executable_1::@crop_dispatch_1(%[[MMR]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<32x256xf32>) -> tensor<32x200xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_tb {
    flow.executable.export public @matmul_tb workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_tb(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<200x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x200xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>> -> tensor<32x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [200, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<200x64xbf16>> -> tensor<200x64xbf16>
        %2 = tensor.empty() : tensor<32x200xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x200xf32>) -> tensor<32x200xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<32x64xbf16>, tensor<200x64xbf16>) outs(%3 : tensor<32x200xf32>) -> tensor<32x200xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [32, 200], strides = [1, 1] : tensor<32x200xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x200xf32>>
        return
      }
    }
  }
  util.func public @main_tb(%arg0: tensor<32x64xbf16>, %arg1: tensor<200x64xbf16>) -> tensor<32x200xf32> {
    %0 = flow.dispatch @dispatch_tb::@matmul_tb(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x64xbf16>, tensor<200x64xbf16>) -> tensor<32x200xf32>
    util.return %0 : tensor<32x200xf32>
  }
}

// -----

// A GEMV-like matmul (M=1, below the GEMV threshold numRows*instr_m = 32) is
// lowered by the GEMV pipeline, which needs neither M nor N padding: M stays 1
// and N=1000 (not a multiple of 64) stays 1000. No pad or crop dispatch is
// created; K=4096 is already a multiple of 32.

// CHECK-NOT:   @pad_executable
// CHECK-NOT:   @crop_executable
// CHECK-LABEL: func.func @matmul_gemv
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
// CHECK-LABEL: util.func public @main_gemv
// CHECK:         flow.dispatch @dispatch_gemv::@matmul_gemv(%arg0, %arg1)
// CHECK-SAME:      : (tensor<1x4096xbf16>, tensor<1000x4096xbf16>) -> tensor<1x1000xf32>
// CHECK-NOT:   @pad_executable
// CHECK-NOT:   @crop_executable
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv {
    flow.executable.export public @matmul_gemv workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>> -> tensor<1x4096xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>> -> tensor<1000x4096xbf16>
        %2 = tensor.empty() : tensor<1x1000xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x4096xbf16>, tensor<1000x4096xbf16>) outs(%3 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 1000], strides = [1, 1] : tensor<1x1000xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
        return
      }
    }
  }
  util.func public @main_gemv(%arg0: tensor<1x4096xbf16>, %arg1: tensor<1000x4096xbf16>) -> tensor<1x1000xf32> {
    %0 = flow.dispatch @dispatch_gemv::@matmul_gemv(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x4096xbf16>, tensor<1000x4096xbf16>) -> tensor<1x1000xf32>
    util.return %0 : tensor<1x1000xf32>
  }
}

// -----

// A GEMV-like matmul still gets its K padded (K=100 -> 128, multiple of 32): the
// GEMV pipeline takes any K, but a 32-aligned K keeps the DMA runs long. K is a
// reduction dim, so both operands grow and nothing is cropped.

// CHECK-LABEL: func.func @matmul_gemv_k
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x128xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[0, 28]
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[0, 28]
// CHECK-NOT:   @crop_executable
// CHECK-LABEL: util.func public @main_gemv_k
// CHECK:         flow.dispatch @dispatch_gemv_k::@matmul_gemv_k
// CHECK-SAME:      : (tensor<1x128xbf16>, tensor<1000x128xbf16>) -> tensor<1x1000xf32>
// CHECK-NOT:   @crop_executable
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv_k {
    flow.executable.export public @matmul_gemv_k workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv_k(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x100xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x100xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 100], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x100xbf16>> -> tensor<1x100xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1000, 100], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x100xbf16>> -> tensor<1000x100xbf16>
        %2 = tensor.empty() : tensor<1x1000xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x100xbf16>, tensor<1000x100xbf16>) outs(%3 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 1000], strides = [1, 1] : tensor<1x1000xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
        return
      }
    }
  }
  util.func public @main_gemv_k(%arg0: tensor<1x100xbf16>, %arg1: tensor<1000x100xbf16>) -> tensor<1x1000xf32> {
    %0 = flow.dispatch @dispatch_gemv_k::@matmul_gemv_k(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x100xbf16>, tensor<1000x100xbf16>) -> tensor<1x1000xf32>
    util.return %0 : tensor<1x1000xf32>
  }
}

// -----

// With the GEMV pipeline switched off on the target (`gemv_pipeline = false`,
// from --iree-amdaie-enable-gemv-pipeline=false) the same M=1 matmul is padded
// as before: M 1 -> 32 and N 1000 -> 1024, with a crop back to [1, 1000].

// CHECK-LABEL: func.func @matmul_gemv_off
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[31, 0]
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[24, 0]
// CHECK:       flow.executable private @crop_executable_2
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [1, 1000] [1, 1]
// CHECK-LABEL: util.func public @main_gemv_off
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {gemv_pipeline = false, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv_off {
    flow.executable.export public @matmul_gemv_off workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv_off(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>> -> tensor<1x4096xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>> -> tensor<1000x4096xbf16>
        %2 = tensor.empty() : tensor<1x1000xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x4096xbf16>, tensor<1000x4096xbf16>) outs(%3 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 1000], strides = [1, 1] : tensor<1x1000xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
        return
      }
    }
  }
  util.func public @main_gemv_off(%arg0: tensor<1x4096xbf16>, %arg1: tensor<1000x4096xbf16>) -> tensor<1x1000xf32> {
    %0 = flow.dispatch @dispatch_gemv_off::@matmul_gemv_off(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x4096xbf16>, tensor<1000x4096xbf16>) -> tensor<1x1000xf32>
    util.return %0 : tensor<1x1000xf32>
  }
}
