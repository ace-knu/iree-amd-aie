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

// An f32 matmul has no AIE vector instruction, so it is padded to the scalar
// pack size instead (4x4x8): M=196 is not a multiple of num_rows*4 = 16 and is
// padded to M=208. N=128 and K=64 are already divisible.

// CHECK-LABEL: func.func @matmul_f32
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<208x64xf32>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<208x128xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<208x64xf32>, tensor<64x128xf32>) outs(%{{.+}} : tensor<208x128xf32>)
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[12, 0]
// CHECK-NOT:   flow.executable private @pad_executable_1
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [196, 128] [1, 1]
// CHECK-LABEL: util.func public @main_f32
// CHECK:         flow.dispatch @dispatch_f32::@matmul_f32
// CHECK-SAME:      : (tensor<208x64xf32>, tensor<64x128xf32>) -> tensor<208x128xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_f32 {
    flow.executable.export public @matmul_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xf32>> -> tensor<196x64xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>> -> tensor<64x128xf32>
        %2 = tensor.empty() : tensor<196x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<196x128xf32>) -> tensor<196x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<196x64xf32>, tensor<64x128xf32>) outs(%3 : tensor<196x128xf32>) -> tensor<196x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [196, 128], strides = [1, 1] : tensor<196x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x128xf32>>
        return
      }
    }
  }
  util.func public @main_f32(%arg0: tensor<196x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<196x128xf32> {
    %0 = flow.dispatch @dispatch_f32::@matmul_f32(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x64xf32>, tensor<64x128xf32>) -> tensor<196x128xf32>
    util.return %0 : tensor<196x128xf32>
  }
}

// -----

// With ukernels enabled the f32 matmul is still padded as without them
// (M=196 -> 208, not doubled to a multiple of 32): there is no f32 matmul
// ukernel, so it is not tiled for one.

// CHECK-LABEL: func.func @matmul_f32_uk
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<208x64xf32>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<208x128xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<208x64xf32>, tensor<64x128xf32>) outs(%{{.+}} : tensor<208x128xf32>)
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[12, 0]
// CHECK-NOT:   flow.executable private @pad_executable_1
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [196, 128] [1, 1]
// CHECK-LABEL: util.func public @main_f32_uk
// CHECK:         flow.dispatch @dispatch_f32_uk::@matmul_f32_uk
// CHECK-SAME:      : (tensor<208x64xf32>, tensor<64x128xf32>) -> tensor<208x128xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "all"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_f32_uk {
    flow.executable.export public @matmul_f32_uk workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_f32_uk(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xf32>> -> tensor<196x64xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xf32>> -> tensor<64x128xf32>
        %2 = tensor.empty() : tensor<196x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<196x128xf32>) -> tensor<196x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<196x64xf32>, tensor<64x128xf32>) outs(%3 : tensor<196x128xf32>) -> tensor<196x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [196, 128], strides = [1, 1] : tensor<196x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x128xf32>>
        return
      }
    }
  }
  util.func public @main_f32_uk(%arg0: tensor<196x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<196x128xf32> {
    %0 = flow.dispatch @dispatch_f32_uk::@matmul_f32_uk(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x64xf32>, tensor<64x128xf32>) -> tensor<196x128xf32>
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

// With ukernels enabled the M/N multiples double (num_rows*2*instrM = 64,
// num_cols*2*instrN = 128), because the matmul ukernels need an even number of
// vector instructions per core. M=196 is padded to 256 (not 224) and N=64,
// already divisible without ukernels, is padded to 128.

// CHECK-LABEL: func.func @matmul_uk
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x128xf32>>
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<256x64xbf16>, tensor<64x128xbf16>) outs(%{{.+}} : tensor<256x128xf32>)
// CHECK-LABEL: util.func public @main_uk
// CHECK:         flow.dispatch @dispatch_uk::@matmul_uk
// CHECK-SAME:      : (tensor<256x64xbf16>, tensor<64x128xbf16>) -> tensor<256x128xf32>
// CHECK:         flow.dispatch @crop_executable_2::@crop_dispatch_2
// CHECK-SAME:      : (tensor<256x128xf32>) -> tensor<196x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "matmul"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_uk {
    flow.executable.export public @matmul_uk workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_uk(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>> -> tensor<196x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>> -> tensor<64x64xbf16>
        %2 = tensor.empty() : tensor<196x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<196x64xbf16>, tensor<64x64xbf16>) outs(%3 : tensor<196x64xf32>) -> tensor<196x64xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xf32>>
        return
      }
    }
  }
  util.func public @main_uk(%arg0: tensor<196x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<196x64xf32> {
    %0 = flow.dispatch @dispatch_uk::@matmul_uk(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x64xbf16>, tensor<64x64xbf16>) -> tensor<196x64xf32>
    util.return %0 : tensor<196x64xf32>
  }
}

// -----

// With ukernels enabled a transpose_b matmul gets the same doubled multiples as
// a plain matmul: M 2 -> 64, N 64 -> 128. (M == 1 takes the GEMV pipeline and
// is not padded, see @matmul_gemv.)

// CHECK-LABEL: func.func @matmul_tb_uk
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128xf32>>
// CHECK-LABEL: util.func public @main_tb_uk
// CHECK:         flow.dispatch @dispatch_tb_uk::@matmul_tb_uk
// CHECK-SAME:      : (tensor<64x64xbf16>, tensor<128x64xbf16>) -> tensor<64x128xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "matmul"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_tb_uk {
    flow.executable.export public @matmul_tb_uk workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_tb_uk(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x64xbf16>> -> tensor<2x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>> -> tensor<64x64xbf16>
        %2 = tensor.empty() : tensor<2x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x64xf32>) -> tensor<2x64xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<2x64xbf16>, tensor<64x64xbf16>) outs(%3 : tensor<2x64xf32>) -> tensor<2x64xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [2, 64], strides = [1, 1] : tensor<2x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x64xf32>>
        return
      }
    }
  }
  util.func public @main_tb_uk(%arg0: tensor<2x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<2x64xf32> {
    %0 = flow.dispatch @dispatch_tb_uk::@matmul_tb_uk(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x64xbf16>, tensor<64x64xbf16>) -> tensor<2x64xf32>
    util.return %0 : tensor<2x64xf32>
  }
}

// -----

// A GEMV-like matmul (M=1) is lowered by the GEMV pipeline, which never pads
// M (it stays 1). N is padded to a multiple of every core times 8 outputs
// (32 x 8 = 256), so the weight grows 1000 -> 1024 along its N dim (dim 0 of
// the [N, K] weight) and the [1, 1024] result is cropped back to [1, 1000] on
// the host. The activation is untouched (K=4096 is a multiple of 32).

// CHECK-LABEL: func.func @matmul_gemv
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[24, 0]
// CHECK:           tensor.yield %[[ZERO]] : bf16
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [1, 1000] [1, 1]
// CHECK-LABEL: util.func public @main_gemv
// CHECK:         %[[W:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg1)
// CHECK:         %[[R:.+]] = flow.dispatch @dispatch_gemv::@matmul_gemv(%arg0, %[[W]])
// CHECK-SAME:      : (tensor<1x4096xbf16>, tensor<1024x4096xbf16>) -> tensor<1x1024xf32>
// CHECK:         flow.dispatch @crop_executable_1::@crop_dispatch_1(%[[R]])
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
// reduction dim, so both operands grow along it; N=1000 is padded to 1024 for
// the core count as above, so the weight grows on both dims and the result is
// cropped. Both pads are zero.

// CHECK-LABEL: func.func @matmul_gemv_k
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x128xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[0, 28]
// CHECK:           tensor.yield %[[ZERO]] : bf16
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         %[[ZERO1:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[24, 28]
// CHECK:           tensor.yield %[[ZERO1]] : bf16
// CHECK:       flow.executable private @crop_executable_2
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [1, 1000] [1, 1]
// CHECK-LABEL: util.func public @main_gemv_k
// CHECK:         flow.dispatch @dispatch_gemv_k::@matmul_gemv_k
// CHECK-SAME:      : (tensor<1x128xbf16>, tensor<1024x128xbf16>) -> tensor<1x1024xf32>
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

// With the matvec ukernel (ukernels = "all", bf16 x bf16 -> f32) the GEMV K is
// padded to whole ukernel vectors of 64: K=96 -> 128, where the default multiple
// of 32 would leave it. N=1024 is already a multiple of 256.

// CHECK-LABEL: func.func @matmul_gemv_k_matvec
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x128xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[0, 32]
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[0, 32]
// CHECK-NOT:   crop_executable
// CHECK-LABEL: util.func public @main_gemv_k_matvec
// CHECK:         flow.dispatch @dispatch_gemv_k_matvec::@matmul_gemv_k_matvec
// CHECK-SAME:      : (tensor<1x128xbf16>, tensor<1024x128xbf16>) -> tensor<1x1024xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "all"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv_k_matvec {
    flow.executable.export public @matmul_gemv_k_matvec workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv_k_matvec(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x96xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x96xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 96], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x96xbf16>> -> tensor<1x96xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1024, 96], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x96xbf16>> -> tensor<1024x96xbf16>
        %2 = tensor.empty() : tensor<1x1024xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x96xbf16>, tensor<1024x96xbf16>) outs(%3 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 1024], strides = [1, 1] : tensor<1x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
        return
      }
    }
  }
  util.func public @main_gemv_k_matvec(%arg0: tensor<1x96xbf16>, %arg1: tensor<1024x96xbf16>) -> tensor<1x1024xf32> {
    %0 = flow.dispatch @dispatch_gemv_k_matvec::@matmul_gemv_k_matvec(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x96xbf16>, tensor<1024x96xbf16>) -> tensor<1x1024xf32>
    util.return %0 : tensor<1x1024xf32>
  }
}

// -----

// M=2 is not routed to the GEMV pipeline (only M == 1 is, see isGemvShape),
// so it is padded to the pack-peel multiples like any other matmul: M 2 -> 32
// with a crop back to [2, 1000], and N 1000 -> 1024.

// CHECK-LABEL: func.func @matmul_m2
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[30, 0]
// CHECK:       flow.executable private @crop_executable_2
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [2, 1000] [1, 1]
// CHECK-LABEL: util.func public @main_m2
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_m2 {
    flow.executable.export public @matmul_m2 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_m2(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x1000xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x4096xbf16>> -> tensor<2x4096xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1000, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>> -> tensor<1000x4096xbf16>
        %2 = tensor.empty() : tensor<2x1000xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<2x1000xf32>) -> tensor<2x1000xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<2x4096xbf16>, tensor<1000x4096xbf16>) outs(%3 : tensor<2x1000xf32>) -> tensor<2x1000xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [2, 1000], strides = [1, 1] : tensor<2x1000xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x1000xf32>>
        return
      }
    }
  }
  util.func public @main_m2(%arg0: tensor<2x4096xbf16>, %arg1: tensor<1000x4096xbf16>) -> tensor<2x1000xf32> {
    %0 = flow.dispatch @dispatch_m2::@matmul_m2(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<2x4096xbf16>, tensor<1000x4096xbf16>) -> tensor<2x1000xf32>
    util.return %0 : tensor<2x1000xf32>
  }
}

// -----

// M=1 with a [K, N] weight (not transpose_b) is not routed to the GEMV pipeline
// either (see isGemvShape), so M is padded 1 -> 32 like M=2 rather than left
// at 1 for GEMV. N 1000 -> 1024.

// CHECK-LABEL: func.func @matmul_m1_kn
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x256xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1024xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[31, 0]
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [1, 1000] [1, 1]
// CHECK-LABEL: util.func public @main_m1_kn
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_m1_kn {
    flow.executable.export public @matmul_m1_kn workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_m1_kn(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1000xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xbf16>> -> tensor<1x256xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [256, 1000], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x1000xbf16>> -> tensor<256x1000xbf16>
        %2 = tensor.empty() : tensor<1x1000xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<1x256xbf16>, tensor<256x1000xbf16>) outs(%3 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 1000], strides = [1, 1] : tensor<1x1000xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
        return
      }
    }
  }
  util.func public @main_m1_kn(%arg0: tensor<1x256xbf16>, %arg1: tensor<256x1000xbf16>) -> tensor<1x1000xf32> {
    %0 = flow.dispatch @dispatch_m1_kn::@matmul_m1_kn(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x256xbf16>, tensor<256x1000xbf16>) -> tensor<1x1000xf32>
    util.return %0 : tensor<1x1000xf32>
  }
}

// -----

// With `gemv_pad_n = false` on the target (--iree-amdaie-gemv-pad-n=false) a
// GEMV's N is left alone whatever core count it reaches: no pad, no crop.

// CHECK-LABEL: func.func @matmul_gemv_nopad
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>
// CHECK-LABEL: util.func public @main_gemv_nopad
// CHECK:         flow.dispatch @dispatch_gemv_nopad::@matmul_gemv_nopad(%arg0, %arg1)
// CHECK-SAME:      : (tensor<1x4096xbf16>, tensor<1000x4096xbf16>) -> tensor<1x1000xf32>
// CHECK-NOT:   @pad_executable
// CHECK-NOT:   @crop_executable
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {gemv_pad_n = false, num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv_nopad {
    flow.executable.export public @matmul_gemv_nopad workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv_nopad(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1000x4096xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1000xf32>>) {
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
  util.func public @main_gemv_nopad(%arg0: tensor<1x4096xbf16>, %arg1: tensor<1000x4096xbf16>) -> tensor<1x1000xf32> {
    %0 = flow.dispatch @dispatch_gemv_nopad::@matmul_gemv_nopad(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x4096xbf16>, tensor<1000x4096xbf16>) -> tensor<1x1000xf32>
    util.return %0 : tensor<1x1000xf32>
  }
}

// -----

// N=4096 already reaches all 32 cores (4096 = 32 x 128 x ... with n1 = 32),
// so a full-array GEMV is never padded.

// CHECK-LABEL: func.func @matmul_gemv_full
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>>
// CHECK-LABEL: util.func public @main_gemv_full
// CHECK:         flow.dispatch @dispatch_gemv_full::@matmul_gemv_full(%arg0, %arg1)
// CHECK-NOT:   @pad_executable
// CHECK-NOT:   @crop_executable
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv_full {
    flow.executable.export public @matmul_gemv_full workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv_full(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>> -> tensor<1x4096xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>> -> tensor<4096x4096xbf16>
        %2 = tensor.empty() : tensor<1x4096xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x4096xbf16>, tensor<4096x4096xbf16>) outs(%3 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : tensor<1x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
        return
      }
    }
  }
  util.func public @main_gemv_full(%arg0: tensor<1x4096xbf16>, %arg1: tensor<4096x4096xbf16>) -> tensor<1x4096xf32> {
    %0 = flow.dispatch @dispatch_gemv_full::@matmul_gemv_full(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<1x4096xf32>
    util.return %0 : tensor<1x4096xf32>
  }
}

// -----

// N=2144 = 32 x 67 reaches 32 cores unpadded only as n1 = 1 (67 blocks).
// Padded to a multiple of 256, N' = 2304 = 32 x 72 (n1 = 24, 3 blocks).

// CHECK-LABEL: func.func @matmul_gemv_2144
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2304x4096xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2304xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[160, 0]
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [1, 2144] [1, 1]
// CHECK-LABEL: util.func public @main_gemv_2144
// CHECK:         flow.dispatch @dispatch_gemv_2144::@matmul_gemv_2144
// CHECK-SAME:      : (tensor<1x4096xbf16>, tensor<2304x4096xbf16>) -> tensor<1x2304xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv_2144 {
    flow.executable.export public @matmul_gemv_2144 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv_2144(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2144x4096xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2144xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4096xbf16>> -> tensor<1x4096xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2144, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2144x4096xbf16>> -> tensor<2144x4096xbf16>
        %2 = tensor.empty() : tensor<1x2144xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x2144xf32>) -> tensor<1x2144xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x4096xbf16>, tensor<2144x4096xbf16>) outs(%3 : tensor<1x2144xf32>) -> tensor<1x2144xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 2144], strides = [1, 1] : tensor<1x2144xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2144xf32>>
        return
      }
    }
  }
  util.func public @main_gemv_2144(%arg0: tensor<1x4096xbf16>, %arg1: tensor<2144x4096xbf16>) -> tensor<1x2144xf32> {
    %0 = flow.dispatch @dispatch_gemv_2144::@matmul_gemv_2144(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x4096xbf16>, tensor<2144x4096xbf16>) -> tensor<1x2144xf32>
    util.return %0 : tensor<1x2144xf32>
  }
}

// -----

// A matmul whose result feeds a fused ReLU-like elementwise consumer
// (cmpf+select, as normal IREE fusion produces e.g. for a "FC -> ReLU"
// sequence) before the store -- so the store does not consume the matmul
// result directly. When only K=27 is misaligned (M=256 and N=64 are already
// divisible, so no output padding is needed), the LHS/RHS are still padded to
// K=32: K is a reduction dim, so the extra products are zero and the
// matmul's result (and everything fused after it) is unchanged. No
// crop_executable is created since the output is never touched.

// CHECK-LABEL: func.func @matmul_relu_k
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<256x32xbf16>, tensor<32x64xbf16>) outs(%{{.+}} : tensor<256x64xf32>)
// CHECK:         %[[RELU:.+]] = linalg.generic {{.*}} ins(%[[MM]] : tensor<256x64xf32>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[RELU]], %arg2, offsets = [0, 0], sizes = [256, 64]

// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad
// CHECK-NOT:   @crop_executable

// CHECK-LABEL: util.func public @main_relu_k
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<256x27xbf16>) -> tensor<256x32xbf16>
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<27x64xbf16>) -> tensor<32x64xbf16>
// CHECK:         flow.dispatch @dispatch_relu_k::@matmul_relu_k(%[[LPAD]], %[[RPAD]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@npu>}
// CHECK-SAME:      : (tensor<256x32xbf16>, tensor<32x64xbf16>) -> tensor<256x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_relu_k {
    flow.executable.export public @matmul_relu_k workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_relu_k(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x27xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [256, 27], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x27xbf16>> -> tensor<256x27xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [27, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>> -> tensor<27x64xbf16>
        %2 = tensor.empty() : tensor<256x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<256x27xbf16>, tensor<27x64xbf16>) outs(%3 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %5 = tensor.empty() : tensor<256x64xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<256x64xf32>) outs(%5 : tensor<256x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.cmpf ogt, %in, %cst : f32
          %8 = arith.select %7, %in, %cst : f32
          linalg.yield %8 : f32
        } -> tensor<256x64xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg2, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : tensor<256x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
        return
      }
    }
  }
  util.func public @main_relu_k(%arg0: tensor<256x27xbf16>, %arg1: tensor<27x64xbf16>) -> tensor<256x64xf32> {
    %0 = flow.dispatch @dispatch_relu_k::@matmul_relu_k(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<256x27xbf16>, tensor<27x64xbf16>) -> tensor<256x64xf32>
    util.return %0 : tensor<256x64xf32>
  }
}

// -----

// Same as above (matmul result feeds a fused ReLU-like consumer before the
// store), but M=196 is ALSO misaligned (in addition to K=27), so padOut is
// true -- and the consumer's `outs` operand is literally the SAME
// `tensor.empty` (%2) that the matmul's own accumulator is built from
// (`fill(%2)`), exactly the shared-init shape that appears in VGG16's real
// FC1/FC2 dispatches (a matmul followed by a fused ReLU, both fed from one
// `tensor.empty`). This is now recognized and handled: the consumer is a
// pure shape-preserving unary map of the matmul result (one input, one
// output, 2-D identity maps, all-parallel iterators), so growing the output
// through it is safe (the padded region's ReLU output is discarded by the
// crop). LHS/RHS are padded (M 196->224, K 27->32), the matmul AND the ReLU
// generic both grow to [224,64], the ReLU gets a *fresh* `tensor.empty` (not
// the old shared one, which is erased along with the matmul's own old
// fill/empty -- this only succeeds if the consumer's `outs` was rewired away
// from the shared empty first), and the result is cropped back to [196,64].

// CHECK-LABEL: func.func @matmul_relu_mk
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x32xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<224x64xf32>>
// CHECK:         %[[NEWEMPTY:.+]] = tensor.empty() : tensor<224x64xf32>
// CHECK:         %[[NEWFILL:.+]] = linalg.fill {{.*}} outs(%[[NEWEMPTY]] : tensor<224x64xf32>)
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<224x32xbf16>, tensor<32x64xbf16>) outs(%[[NEWFILL]] : tensor<224x64xf32>)
// CHECK:         %[[RELUINIT:.+]] = tensor.empty() : tensor<224x64xf32>
// CHECK:         %[[RELU:.+]] = linalg.generic {{.*}} ins(%[[MM]] : tensor<224x64xf32>) outs(%[[RELUINIT]] : tensor<224x64xf32>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[RELU]], %arg2, offsets = [0, 0], sizes = [224, 64]

// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad
// CHECK:       flow.executable private @crop_executable_2
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [196, 64] [1, 1]

// CHECK-LABEL: util.func public @main_relu_mk
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      : (tensor<196x27xbf16>) -> tensor<224x32xbf16>
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1)
// CHECK-SAME:      : (tensor<27x64xbf16>) -> tensor<32x64xbf16>
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_relu_mk::@matmul_relu_mk(%[[LPAD]], %[[RPAD]])
// CHECK-SAME:      : (tensor<224x32xbf16>, tensor<32x64xbf16>) -> tensor<224x64xf32>
// CHECK:         flow.dispatch @crop_executable_2::@crop_dispatch_2(%[[MMR]])
// CHECK-SAME:      : (tensor<224x64xf32>) -> tensor<196x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_relu_mk {
    flow.executable.export public @matmul_relu_mk workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_relu_mk(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x27xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 27], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x27xbf16>> -> tensor<196x27xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [27, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>> -> tensor<27x64xbf16>
        %2 = tensor.empty() : tensor<196x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<196x27xbf16>, tensor<27x64xbf16>) outs(%3 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<196x64xf32>) outs(%2 : tensor<196x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.cmpf ogt, %in, %cst : f32
          %8 = arith.select %7, %in, %cst : f32
          linalg.yield %8 : f32
        } -> tensor<196x64xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg2, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xf32>>
        return
      }
    }
  }
  util.func public @main_relu_mk(%arg0: tensor<196x27xbf16>, %arg1: tensor<27x64xbf16>) -> tensor<196x64xf32> {
    %0 = flow.dispatch @dispatch_relu_mk::@matmul_relu_mk(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x27xbf16>, tensor<27x64xbf16>) -> tensor<196x64xf32>
    util.return %0 : tensor<196x64xf32>
  }
}

// -----

// Same as @main_relu_mk (M 196->224, K 27->32), but the fused consumer
// narrows the type (f32 -> bf16, as a real ReLU-then-demote epilogue would
// for the next bf16 matmul's input). Regression test for the dispatch's own
// result type (used for the un-cropped [mPad,nPad] result and by the crop
// dispatch): it must come from the output BINDING (bf16), not from the
// matmul's own accumulator type (f32) -- getting this wrong is a loud
// verifier failure (a store of a bf16 value into an f32-typed result), not
// silently wrong numerics.
// CHECK-LABEL: func.func @matmul_trunc_mk
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<224x64xbf16>>
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<224x32xbf16>, tensor<32x64xbf16>) outs(%{{.+}} : tensor<224x64xf32>)
// CHECK:         %[[TRUNCINIT:.+]] = tensor.empty() : tensor<224x64xbf16>
// CHECK:         %[[TRUNC:.+]] = linalg.generic {{.*}} ins(%[[MM]] : tensor<224x64xf32>) outs(%[[TRUNCINIT]] : tensor<224x64xbf16>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[TRUNC]], %arg2, offsets = [0, 0], sizes = [224, 64]

// CHECK-LABEL: util.func public @main_trunc_mk
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      : (tensor<196x27xbf16>) -> tensor<224x32xbf16>
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1)
// CHECK-SAME:      : (tensor<27x64xbf16>) -> tensor<32x64xbf16>
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_trunc_mk::@matmul_trunc_mk(%[[LPAD]], %[[RPAD]])
// CHECK-SAME:      : (tensor<224x32xbf16>, tensor<32x64xbf16>) -> tensor<224x64xbf16>
// CHECK:         flow.dispatch @crop_executable_{{[0-9]+}}::{{@.+}}(%[[MMR]])
// CHECK-SAME:      : (tensor<224x64xbf16>) -> tensor<196x64xbf16>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_trunc_mk {
    flow.executable.export public @matmul_trunc_mk workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_trunc_mk(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x27xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xbf16>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 27], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x27xbf16>> -> tensor<196x27xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [27, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>> -> tensor<27x64xbf16>
        %2 = tensor.empty() : tensor<196x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<196x27xbf16>, tensor<27x64xbf16>) outs(%3 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %5 = tensor.empty() : tensor<196x64xbf16>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<196x64xf32>) outs(%5 : tensor<196x64xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %7 = arith.truncf %in : f32 to bf16
          linalg.yield %7 : bf16
        } -> tensor<196x64xbf16>
        iree_tensor_ext.dispatch.tensor.store %6, %arg2, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xbf16>>
        return
      }
    }
  }
  util.func public @main_trunc_mk(%arg0: tensor<196x27xbf16>, %arg1: tensor<27x64xbf16>) -> tensor<196x64xbf16> {
    %0 = flow.dispatch @dispatch_trunc_mk::@matmul_trunc_mk(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x27xbf16>, tensor<27x64xbf16>) -> tensor<196x64xbf16>
    util.return %0 : tensor<196x64xbf16>
  }
}

// -----

// Boundary case that must still be safely skipped: a CHAIN of two
// shape-preserving elementwise consumers (ReLU then negate) sits between the
// matmul and the store, with M=196 and K=27 both misaligned. Only a single
// intervening op is recognized (`getMatmulInfo`'s `isShapePreservingUnaryMapOf`
// check requires the consumer's one real input to be exactly the matmul's
// result) -- the second generic's input is the *first* generic's result, not
// the matmul's, so the match fails and `MatmulInfo::store`/`outArg` stay
// null. Since padOut is true (M is misaligned) and store/outArg are
// unresolved, the whole dispatch is left conservatively untouched, same as
// the fully-generic "growing an unknown consumer" fallback -- no pad, no
// crop, no crash.

// CHECK-NOT:   @pad_executable
// CHECK-NOT:   @crop_executable
// CHECK-LABEL: func.func @matmul_chain_mk
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x27xbf16>>
// CHECK-LABEL: util.func public @main_chain_mk
// CHECK:         flow.dispatch @dispatch_chain_mk::@matmul_chain_mk(%arg0, %arg1)
// CHECK-SAME:      : (tensor<196x27xbf16>, tensor<27x64xbf16>) -> tensor<196x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_chain_mk {
    flow.executable.export public @matmul_chain_mk workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_chain_mk(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x27xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 27], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x27xbf16>> -> tensor<196x27xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [27, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>> -> tensor<27x64xbf16>
        %2 = tensor.empty() : tensor<196x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<196x27xbf16>, tensor<27x64xbf16>) outs(%3 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %5 = tensor.empty() : tensor<196x64xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<196x64xf32>) outs(%5 : tensor<196x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %7 = arith.cmpf ogt, %in, %cst : f32
          %8 = arith.select %7, %in, %cst : f32
          linalg.yield %8 : f32
        } -> tensor<196x64xf32>
        %9 = tensor.empty() : tensor<196x64xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<196x64xf32>) outs(%9 : tensor<196x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.negf %in : f32
          linalg.yield %11 : f32
        } -> tensor<196x64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %arg2, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xf32>>
        return
      }
    }
  }
  util.func public @main_chain_mk(%arg0: tensor<196x27xbf16>, %arg1: tensor<27x64xbf16>) -> tensor<196x64xf32> {
    %0 = flow.dispatch @dispatch_chain_mk::@matmul_chain_mk(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x27xbf16>, tensor<27x64xbf16>) -> tensor<196x64xf32>
    util.return %0 : tensor<196x64xf32>
  }
}

// -----

// Accumulator-as-bias form (MatmulInfo's accLoad/accArg, no fill/empty at
// all): the matmul's `outs` is a full-tensor load of a [M, N] bias operand
// straight from a binding (e.g. a conv2d+bias dispatch whose bias was already
// broadcast to the output shape by an earlier op) rather than a `fill(empty)`
// zero-init. K=27 is misaligned (padOut is false), so this is K-only padding:
// LHS/RHS are padded and the accumulator is left completely untouched -- same
// as the fill/empty form, K-only padding never touches the output side.

// CHECK-LABEL: func.func @matmul_accbias_k
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xf32>>
// CHECK-SAME:    %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
// CHECK:         %[[ACC:.+]] = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [256, 64]
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<256x32xbf16>, tensor<32x64xbf16>) outs(%[[ACC]] : tensor<256x64xf32>)
// CHECK-NOT:     @crop_executable

// CHECK-LABEL: util.func public @main_accbias_k
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      : (tensor<256x27xbf16>) -> tensor<256x32xbf16>
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1)
// CHECK-SAME:      : (tensor<27x64xbf16>) -> tensor<32x64xbf16>
// CHECK:         flow.dispatch @dispatch_accbias_k::@matmul_accbias_k(%[[LPAD]], %[[RPAD]], %arg2)
// CHECK-SAME:      : (tensor<256x32xbf16>, tensor<32x64xbf16>, tensor<256x64xf32>) -> tensor<256x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_accbias_k {
    flow.executable.export public @matmul_accbias_k workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_accbias_k(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x27xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [256, 27], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x27xbf16>> -> tensor<256x27xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [27, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<27x64xbf16>> -> tensor<27x64xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x64xf32>> -> tensor<256x64xf32>
        %3 = linalg.matmul ins(%0, %1 : tensor<256x27xbf16>, tensor<27x64xbf16>) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0, 0], sizes = [256, 64], strides = [1, 1] : tensor<256x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x64xf32>>
        return
      }
    }
  }
  util.func public @main_accbias_k(%arg0: tensor<256x27xbf16>, %arg1: tensor<27x64xbf16>, %arg2: tensor<256x64xf32>) -> tensor<256x64xf32> {
    %0 = flow.dispatch @dispatch_accbias_k::@matmul_accbias_k(%arg0, %arg1, %arg2) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<256x27xbf16>, tensor<27x64xbf16>, tensor<256x64xf32>) -> tensor<256x64xf32>
    util.return %0 : tensor<256x64xf32>
  }
}

// -----

// Accumulator-as-bias form combined with a fused ReLU consumer and M
// misaligned (padOut is true this time, unlike the K-only case above) -- the
// VGG16 conv11~13 shape: bias is a plain load (no fill/empty), M=196 needs
// padding to 224, K/N are already aligned. The accumulator binding/load is
// grown exactly like LHS/RHS are (via growBinding, driven by growMatmulInit's
// accLoad/accArg branch), and this caller's host bias operand is zero-padded
// by its own padding dispatch alongside LHS -- the padded region is discarded
// by the crop, so its value doesn't matter. The ReLU consumer gets a fresh
// init (growElementwiseConsumerInit), same as the fill/empty form.

// CHECK-LABEL: func.func @matmul_accbias_relu_m
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x64xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x64xf32>>
// CHECK-SAME:    %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<224x64xf32>>
// CHECK:         %[[ACC:.+]] = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [224, 64]
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<224x64xbf16>, tensor<64x64xbf16>) outs(%[[ACC]] : tensor<224x64xf32>)
// CHECK:         %[[RELUINIT:.+]] = tensor.empty() : tensor<224x64xf32>
// CHECK:         %[[RELU:.+]] = linalg.generic {{.*}} ins(%[[MM]] : tensor<224x64xf32>) outs(%[[RELUINIT]] : tensor<224x64xf32>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[RELU]], %arg3, offsets = [0, 0], sizes = [224, 64]

// CHECK:       flow.executable private @pad_executable_0
// CHECK:         tensor.pad
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         tensor.pad
// CHECK:       flow.executable private @crop_executable_2
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [196, 64] [1, 1]

// CHECK-LABEL: util.func public @main_accbias_relu_m
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      : (tensor<196x64xbf16>) -> tensor<224x64xbf16>
// CHECK:         %[[BPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg2)
// CHECK-SAME:      : (tensor<196x64xf32>) -> tensor<224x64xf32>
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_accbias_relu_m::@matmul_accbias_relu_m(%[[LPAD]], %arg1, %[[BPAD]])
// CHECK-SAME:      : (tensor<224x64xbf16>, tensor<64x64xbf16>, tensor<224x64xf32>) -> tensor<224x64xf32>
// CHECK:         flow.dispatch @crop_executable_2::@crop_dispatch_2(%[[MMR]])
// CHECK-SAME:      : (tensor<224x64xf32>) -> tensor<196x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_accbias_relu_m {
    flow.executable.export public @matmul_accbias_relu_m workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_accbias_relu_m(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>> -> tensor<196x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>> -> tensor<64x64xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xf32>> -> tensor<196x64xf32>
        %3 = linalg.matmul ins(%0, %1 : tensor<196x64xbf16>, tensor<64x64xbf16>) outs(%2 : tensor<196x64xf32>) -> tensor<196x64xf32>
        %4 = tensor.empty() : tensor<196x64xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<196x64xf32>) outs(%4 : tensor<196x64xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.cmpf ogt, %in, %cst : f32
          %7 = arith.select %6, %in, %cst : f32
          linalg.yield %7 : f32
        } -> tensor<196x64xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg3, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x64xf32>>
        return
      }
    }
  }
  util.func public @main_accbias_relu_m(%arg0: tensor<196x64xbf16>, %arg1: tensor<64x64xbf16>, %arg2: tensor<196x64xf32>) -> tensor<196x64xf32> {
    %0 = flow.dispatch @dispatch_accbias_relu_m::@matmul_accbias_relu_m(%arg0, %arg1, %arg2) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x64xbf16>, tensor<64x64xbf16>, tensor<196x64xf32>) -> tensor<196x64xf32>
    util.return %0 : tensor<196x64xf32>
  }
}

// -----

// Accumulator-bias K-column fold: the accumulator is a host broadcast of a
// constant bias and the weight is a constant global, so instead of padding the
// bias operand the bias is folded into the contraction's K dim -- the weight
// grows one K row holding the bias (a new compile-time constant), the LHS is
// padded with 1.0 so K index 3 contracts against it, and the bias binding is
// dropped, leaving a two-operand dispatch with a plain zero-init. That is what
// makes a convolution's bias reach the NPU at all: a third L3->L2 operand has
// no shim DMA producer channel to bind to.
//
// K=3 -> 4 -> 32, and N=1 -> 64 so the output side is grown and cropped too,
// exercising the fold together with `padOut` (VGG16's conv11~13 shape).

// The folded weight is a fresh constant, declared at the top of the module.
// CHECK-LABEL: util.global private @__bias_folded_const_
// CHECK-SAME:    : tensor<32x64xbf16>

// CHECK-LABEL: func.func @matmul_biasfold
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x32xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
// The bias binding is gone; the accumulator is a zero fill again.
// CHECK-NOT:     dispatch.tensor.load %arg2
// CHECK:         %[[FILL:.+]] = linalg.fill
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x32xbf16>, tensor<32x64xbf16>) outs(%[[FILL]] : tensor<32x64xf32>)

// The LHS (M > 1) is padded by a plain zero-pad dispatch, then a dispatch
// tied to its result writes columns [3, 32): 1.0 at K index 3 and 0.0 after
// it (a zero operand is faster on the AIE core's scalar matmul).
// CHECK-LABEL: func.func @pad_dispatch_
// CHECK:         tensor.pad %{{.+}} low[0, 0] high[0, 29]
// CHECK:         } : tensor<32x3xbf16> to tensor<32x32xbf16>
// CHECK-LABEL: func.func @bias_ones_dispatch_
// CHECK-SAME:    (%[[BUF:.+]]: !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x32xbf16>>)
// CHECK:         %[[COLS:.+]] = linalg.generic
// CHECK:           %[[K:.+]] = linalg.index 1
// CHECK:           arith.cmpi eq, %[[K]], %c0
// CHECK:           arith.select
// CHECK:         dispatch.tensor.store %[[COLS]], %[[BUF]], offsets = [0, 3], sizes = [32, 29]

// CHECK-LABEL: util.func public @main_biasfold
// CHECK:         %[[ZPAD:.+]] = flow.dispatch @pad_executable_{{[0-9]+}}::@pad_dispatch_{{[0-9]+}}(%arg0)
// CHECK-SAME:      : (tensor<32x3xbf16>) -> tensor<32x32xbf16>
// CHECK:         %[[LPAD:.+]] = flow.dispatch @bias_ones_executable_{{[0-9]+}}::@bias_ones_dispatch_{{[0-9]+}}(%[[ZPAD]])
// CHECK-SAME:      : (tensor<32x32xbf16>) -> %[[ZPAD]]
// CHECK:         %[[W:.+]] = util.global.load @__bias_folded_const_
// CHECK:         %[[MM:.+]] = flow.dispatch @dispatch_biasfold::@matmul_biasfold(%[[LPAD]], %[[W]])
// CHECK-SAME:      : (tensor<32x32xbf16>, tensor<32x64xbf16>) -> tensor<32x64xf32>
// CHECK:         flow.dispatch @crop_executable_{{[0-9]+}}::@crop_dispatch_{{[0-9]+}}(%[[MM]])
// CHECK-SAME:      : (tensor<32x64xf32>) -> tensor<32x1xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight = dense<[[1.000000e+00], [2.000000e+00], [3.000000e+00]]> : tensor<3x1xbf16>
  util.global private @bias = dense<7.000000e+00> : tensor<1xf32>
  flow.executable private @dispatch_bcast {
    flow.executable.export public @bcast workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @bcast(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xf32>> -> tensor<1xf32>
        %1 = tensor.empty() : tensor<32x1xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<1xf32>) outs(%1 : tensor<32x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x1xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : tensor<32x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_biasfold {
    flow.executable.export public @matmul_biasfold workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_biasfold(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x1xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3xbf16>> -> tensor<32x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [3, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x1xbf16>> -> tensor<3x1xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1xf32>> -> tensor<32x1xf32>
        %3 = linalg.matmul ins(%0, %1 : tensor<32x3xbf16>, tensor<3x1xbf16>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : tensor<32x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>
        return
      }
    }
  }
  util.func public @main_biasfold(%arg0: tensor<32x3xbf16>) -> tensor<32x1xf32> {
    %w = util.global.load immutable @weight : tensor<3x1xbf16>
    %b = util.global.load immutable @bias : tensor<1xf32>
    %acc = flow.dispatch @dispatch_bcast::@bcast(%b) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<1xf32>) -> tensor<32x1xf32>
    %0 = flow.dispatch @dispatch_biasfold::@matmul_biasfold(%arg0, %w, %acc) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x3xbf16>, tensor<3x1xbf16>, tensor<32x1xf32>) -> tensor<32x1xf32>
    util.return %0 : tensor<32x1xf32>
  }
}

// -----

// The same foldable shape, but one executable shared by two dispatches. The
// fold rewrites the executable's signature, which per-caller operand padding
// cannot keep consistent across callers, so a shared executable is not folded:
// it keeps its bias binding and every caller is padded the ordinary way.

// CHECK-NOT:   @__bias_folded_const_
// CHECK-LABEL: func.func @matmul_shared
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>>
// CHECK:         %[[ACC:.+]] = iree_tensor_ext.dispatch.tensor.load %arg2
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x32xbf16>, tensor<32x64xbf16>) outs(%[[ACC]] : tensor<32x64xf32>)
// CHECK-LABEL: util.func public @main_shared
// CHECK-NOT:     @pad_ones_executable_
// CHECK:         flow.dispatch @dispatch_shared::@matmul_shared(
// CHECK-SAME:      : (tensor<32x32xbf16>, tensor<32x64xbf16>, tensor<32x64xf32>) -> tensor<32x64xf32>
// CHECK:         flow.dispatch @dispatch_shared::@matmul_shared(
// CHECK-SAME:      : (tensor<32x32xbf16>, tensor<32x64xbf16>, tensor<32x64xf32>) -> tensor<32x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight3 = dense<[[1.000000e+00], [2.000000e+00], [3.000000e+00]]> : tensor<3x1xbf16>
  util.global private @bias3 = dense<7.000000e+00> : tensor<1xf32>
  flow.executable private @dispatch_bcast3 {
    flow.executable.export public @bcast3 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @bcast3(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xf32>> -> tensor<1xf32>
        %1 = tensor.empty() : tensor<32x1xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<1xf32>) outs(%1 : tensor<32x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x1xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : tensor<32x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_shared {
    flow.executable.export public @matmul_shared workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_shared(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x1xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3xbf16>> -> tensor<32x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [3, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x1xbf16>> -> tensor<3x1xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1xf32>> -> tensor<32x1xf32>
        %3 = linalg.matmul ins(%0, %1 : tensor<32x3xbf16>, tensor<3x1xbf16>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : tensor<32x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>
        return
      }
    }
  }
  util.func public @main_shared(%arg0: tensor<32x3xbf16>, %arg1: tensor<32x3xbf16>) -> (tensor<32x1xf32>, tensor<32x1xf32>) {
    %w = util.global.load immutable @weight3 : tensor<3x1xbf16>
    %b = util.global.load immutable @bias3 : tensor<1xf32>
    %acc = flow.dispatch @dispatch_bcast3::@bcast3(%b) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<1xf32>) -> tensor<32x1xf32>
    %0 = flow.dispatch @dispatch_shared::@matmul_shared(%arg0, %w, %acc) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x3xbf16>, tensor<3x1xbf16>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %1 = flow.dispatch @dispatch_shared::@matmul_shared(%arg1, %w, %acc) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x3xbf16>, tensor<3x1xbf16>, tensor<32x1xf32>) -> tensor<32x1xf32>
    util.return %0, %1 : tensor<32x1xf32>, tensor<32x1xf32>
  }
}

// -----

// Same shapes and the same constant weight as above, but the accumulator is a
// runtime value rather than a broadcast of a constant bias. The fold needs
// *both* halves to be constant -- it has to produce the bias row at compile
// time -- so it does not apply, and the dispatch keeps its three operands:
// K is padded to 32 (not 4 -> 32), the weight is padded by the ordinary
// constant path, and the bias operand is zero-padded like any other value.

// CHECK-LABEL: util.global private @__padded_const_
// CHECK-NOT:   @__bias_folded_const_

// CHECK-LABEL: func.func @matmul_runtimeacc
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>>
// CHECK:         %[[ACC:.+]] = iree_tensor_ext.dispatch.tensor.load %arg2
// CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x32xbf16>, tensor<32x64xbf16>) outs(%[[ACC]] : tensor<32x64xf32>)

// CHECK-LABEL: util.func public @main_runtimeacc
// CHECK-NOT:     @pad_ones_executable_
// CHECK:         flow.dispatch @dispatch_runtimeacc::@matmul_runtimeacc(
// CHECK-SAME:      : (tensor<32x32xbf16>, tensor<32x64xbf16>, tensor<32x64xf32>) -> tensor<32x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight2 = dense<[[1.000000e+00], [2.000000e+00], [3.000000e+00]]> : tensor<3x1xbf16>
  flow.executable private @dispatch_runtimeacc {
    flow.executable.export public @matmul_runtimeacc workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_runtimeacc(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x1xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3xbf16>> -> tensor<32x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [3, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x1xbf16>> -> tensor<3x1xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1xf32>> -> tensor<32x1xf32>
        %3 = linalg.matmul ins(%0, %1 : tensor<32x3xbf16>, tensor<3x1xbf16>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : tensor<32x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>
        return
      }
    }
  }
  util.func public @main_runtimeacc(%arg0: tensor<32x3xbf16>, %arg1: tensor<32x1xf32>) -> tensor<32x1xf32> {
    %w = util.global.load immutable @weight2 : tensor<3x1xbf16>
    %0 = flow.dispatch @dispatch_runtimeacc::@matmul_runtimeacc(%arg0, %w, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x3xbf16>, tensor<3x1xbf16>, tensor<32x1xf32>) -> tensor<32x1xf32>
    util.return %0 : tensor<32x1xf32>
  }
}

// -----

// The LHS is produced by a host dispatch (im2col + bf16 truncation, reshaped to
// the 2-D matmul operand) that nothing else reads. With the bias folded, the
// padding is *not* folded into that dispatch: padding inside it folds the
// reshape and pads into a `map_store` that recomputes every element's row and
// column with a div/mod on the CPU backend. The im2col dispatch keeps its
// shape, and the LHS takes a zero-pad dispatch plus a ones-column dispatch
// tied to its result.

// CHECK-LABEL: func.func @im2col_grown
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x3x1xbf16>>
// CHECK-NOT:     tensor.pad
// CHECK:         return

// CHECK-LABEL: util.func public @main_grown
// CHECK:         %[[IM:.+]] = flow.dispatch @dispatch_im2col::@im2col_grown(%arg0)
// CHECK-SAME:      : (tensor<32x3x1xf32>) -> tensor<32x3x1xbf16>
// CHECK:         %[[R:.+]] = flow.tensor.reshape %[[IM]] : tensor<32x3x1xbf16> -> tensor<32x3xbf16>
// CHECK:         %[[ZPAD:.+]] = flow.dispatch @pad_executable_{{[0-9]+}}::@pad_dispatch_{{[0-9]+}}(%[[R]])
// CHECK-SAME:      : (tensor<32x3xbf16>) -> tensor<32x32xbf16>
// CHECK:         %[[LHS:.+]] = flow.dispatch @bias_ones_executable_{{[0-9]+}}::@bias_ones_dispatch_{{[0-9]+}}(%[[ZPAD]])
// CHECK:         flow.dispatch @dispatch_grown::@matmul_grown(%[[LHS]],
// CHECK-SAME:      : (tensor<32x32xbf16>, tensor<32x64xbf16>) -> tensor<32x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight3 = dense<[[1.000000e+00], [2.000000e+00], [3.000000e+00]]> : tensor<3x1xbf16>
  util.global private @bias3 = dense<7.000000e+00> : tensor<1xf32>
  flow.executable private @dispatch_im2col {
    flow.executable.export public @im2col_grown workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @im2col_grown(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3x1xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x3x1xbf16>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 3, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3x1xf32>> -> tensor<32x3x1xf32>
        %1 = tensor.empty() : tensor<32x3x1xbf16>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<32x3x1xf32>) outs(%1 : tensor<32x3x1xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %3 = arith.truncf %in : f32 to bf16
          linalg.yield %3 : bf16
        } -> tensor<32x3x1xbf16>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0, 0], sizes = [32, 3, 1], strides = [1, 1, 1] : tensor<32x3x1xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x3x1xbf16>>
        return
      }
    }
  }
  flow.executable private @dispatch_bcast3 {
    flow.executable.export public @bcast3 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @bcast3(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [1], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1xf32>> -> tensor<1xf32>
        %1 = tensor.empty() : tensor<32x1xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<1xf32>) outs(%1 : tensor<32x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x1xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : tensor<32x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_grown {
    flow.executable.export public @matmul_grown workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_grown(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x1xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x3xbf16>> -> tensor<32x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [3, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x1xbf16>> -> tensor<3x1xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x1xf32>> -> tensor<32x1xf32>
        %3 = linalg.matmul ins(%0, %1 : tensor<32x3xbf16>, tensor<3x1xbf16>) outs(%2 : tensor<32x1xf32>) -> tensor<32x1xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg3, offsets = [0, 0], sizes = [32, 1], strides = [1, 1] : tensor<32x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1xf32>>
        return
      }
    }
  }
  util.func public @main_grown(%arg0: tensor<32x3x1xf32>) -> tensor<32x1xf32> {
    %w = util.global.load immutable @weight3 : tensor<3x1xbf16>
    %b = util.global.load immutable @bias3 : tensor<1xf32>
    %lhs3 = flow.dispatch @dispatch_im2col::@im2col_grown(%arg0) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<32x3x1xf32>) -> tensor<32x3x1xbf16>
    %lhs = flow.tensor.reshape %lhs3 : tensor<32x3x1xbf16> -> tensor<32x3xbf16>
    %acc = flow.dispatch @dispatch_bcast3::@bcast3(%b) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<1xf32>) -> tensor<32x1xf32>
    %0 = flow.dispatch @dispatch_grown::@matmul_grown(%lhs, %w, %acc) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x3xbf16>, tensor<3x1xbf16>, tensor<32x1xf32>) -> tensor<32x1xf32>
    util.return %0 : tensor<32x1xf32>
  }
}

// -----

// Post-add bias fold: a matmul + bias as an ONNX Gemm (VGG16's FC layers)
// arrives once the add is fused into the dispatch -- a zero-init transpose_b
// matmul whose result feeds a two-input elementwise op that adds a constant
// [1, N] bias, then applies a ReLU and truncates to bf16. The bias becomes row
// K=3 of the constant weight (7 and -8 here), the bias binding goes away, and
// the consumer keeps only the ReLU and the truncation. M == 1 with transpose_b
// is a GEMV: M stays 1, N takes the GEMV multiple (2 -> 256) and the folded K
// is aligned to the whole GEMV K tile: K=3 -> 4 -> 256.

// CHECK-LABEL: util.global private @__bias_folded_const_
// CHECK-SAME:    dense<"0x803F00404040E04000000000
// CHECK-SAME:    : tensor<256x256xbf16>

// CHECK-LABEL: func.func @fc_relu
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x256xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x256xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x256xbf16>>)
// CHECK:         %[[FILL:.+]] = linalg.fill
// CHECK:         %[[MM:.+]] = linalg.matmul {{.*}} ins(%{{.+}}, %{{.+}} : tensor<1x256xbf16>, tensor<256x256xbf16>) outs(%[[FILL]] : tensor<1x256xf32>)
// CHECK:         %[[RELU:.+]] = linalg.generic {{.*}} ins(%[[MM]] : tensor<1x256xf32>) outs(%{{.+}} : tensor<1x256xbf16>)
// CHECK-NOT:       arith.addf
// CHECK:           arith.cmpf ugt
// CHECK:           arith.select
// CHECK:           arith.truncf
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[RELU]], %arg2

// The LHS gets its 1.0 at K index 3 without a padding dispatch: M stays 1,
// so the real data is one contiguous run at the start of the padded row, and
// a constant [1, 256] (1.0 at index 3, bf16 0x3F80) takes it through a
// flow.tensor.update.
// CHECK-NOT:   func.func @pad_ones_dispatch_
// CHECK-LABEL: util.func public @main_fc
// CHECK:         %[[ONES:.+]] = arith.constant dense<"0x000000000000803F0000
// CHECK:         %[[LPAD:.+]] = flow.tensor.update %arg0, %[[ONES]][%{{.+}}, %{{.+}}] : tensor<1x3xbf16> -> %[[ONES]] as tensor<1x256xbf16>
// CHECK:         %[[W:.+]] = util.global.load @__bias_folded_const_
// CHECK:         %[[R:.+]] = flow.dispatch @dispatch_fc::@fc_relu(%[[LPAD]], %[[W]])
// CHECK-SAME:      : (tensor<1x256xbf16>, tensor<256x256xbf16>) -> tensor<1x256xbf16>
// CHECK:         flow.dispatch @crop_executable_{{[0-9]+}}::@crop_dispatch_{{[0-9]+}}(%[[R]])
// CHECK-SAME:      : (tensor<1x256xbf16>) -> tensor<1x2xbf16>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @fcw = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xbf16>
  util.global private @fcb = dense<[[7.000000e+00, -8.000000e+00]]> : tensor<1x2xf32>
  flow.executable private @dispatch_fc {
    flow.executable.export public @fc_relu workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc_relu(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xbf16>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>> -> tensor<1x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>> -> tensor<2x3xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>> -> tensor<1x2xf32>
        %3 = tensor.empty() : tensor<1x2xbf16>
        %4 = tensor.empty() : tensor<1x2xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %6 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x3xbf16>, tensor<2x3xbf16>) outs(%5 : tensor<1x2xf32>) -> tensor<1x2xf32>
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
  util.func public @main_fc(%arg0: tensor<1x3xbf16>) -> tensor<1x2xbf16> {
    %w = util.global.load immutable @fcw : tensor<2x3xbf16>
    %b = util.global.load immutable @fcb : tensor<1x2xf32>
    %0 = flow.dispatch @dispatch_fc::@fc_relu(%arg0, %w, %b) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x3xbf16>, tensor<2x3xbf16>, tensor<1x2xf32>) -> tensor<1x2xbf16>
    util.return %0 : tensor<1x2xbf16>
  }
}

// -----

// The same fold with a 1-D bias read through (d0, d1) -> (d1), operands of the
// add swapped, and nothing after the add (VGG16's FC3): once the add is gone
// the consumer would only copy the matmul result, so it is dropped and the
// matmul result is stored directly.

// CHECK-LABEL: util.global private @__bias_folded_const_
// CHECK-SAME:    dense<"0x803F00404040E04000000000
// CHECK-LABEL: func.func @fc_add
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x256xf32>>)
// CHECK:         %[[MM:.+]] = linalg.matmul
// CHECK-NOT:     linalg.generic
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[MM]], %arg2
// CHECK-LABEL: util.func public @main_fc_add
// CHECK:         flow.dispatch @dispatch_fc_add::@fc_add(
// CHECK-SAME:      : (tensor<1x256xbf16>, tensor<256x256xbf16>) -> tensor<1x256xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @fcw2 = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xbf16>
  util.global private @fcb2 = dense<[7.000000e+00, -8.000000e+00]> : tensor<2xf32>
  flow.executable private @dispatch_fc_add {
    flow.executable.export public @fc_add workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc_add(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>> -> tensor<1x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>> -> tensor<2x3xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
        %4 = tensor.empty() : tensor<1x2xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %6 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x3xbf16>, tensor<2x3xbf16>) outs(%5 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %2 : tensor<1x2xf32>, tensor<2xf32>) outs(%4 : tensor<1x2xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in_0, %in : f32
          linalg.yield %8 : f32
        } -> tensor<1x2xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xf32>>
        return
      }
    }
  }
  util.func public @main_fc_add(%arg0: tensor<1x3xbf16>) -> tensor<1x2xf32> {
    %w = util.global.load immutable @fcw2 : tensor<2x3xbf16>
    %b = util.global.load immutable @fcb2 : tensor<2xf32>
    %0 = flow.dispatch @dispatch_fc_add::@fc_add(%arg0, %w, %b) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x3xbf16>, tensor<2x3xbf16>, tensor<2xf32>) -> tensor<1x2xf32>
    util.return %0 : tensor<1x2xf32>
  }
}

// -----

// The same FC, but the bias is a runtime value. The fold has to build the
// bias row at compile time, so it does not apply: the bias binding stays, and
// since M/N need padding and the output is still behind a two-input consumer,
// the dispatch is left untouched, as before this fold existed.

// CHECK-NOT:   @__bias_folded_const_
// CHECK-LABEL: func.func @fc_rt
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>>
// CHECK:         arith.addf
// CHECK-LABEL: util.func public @main_fc_rt
// CHECK-NOT:     @pad_ones_executable_
// CHECK:         flow.dispatch @dispatch_fc_rt::@fc_rt(
// CHECK-SAME:      : (tensor<1x3xbf16>, tensor<2x3xbf16>, tensor<1x2xf32>) -> tensor<1x2xbf16>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @fcw3 = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xbf16>
  util.global private @fcb3 = dense<[[7.000000e+00, -8.000000e+00]]> : tensor<1x2xf32>
  flow.executable private @dispatch_fc_rt {
    flow.executable.export public @fc_rt workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc_rt(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xbf16>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>> -> tensor<1x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>> -> tensor<2x3xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>> -> tensor<1x2xf32>
        %3 = tensor.empty() : tensor<1x2xbf16>
        %4 = tensor.empty() : tensor<1x2xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %6 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x3xbf16>, tensor<2x3xbf16>) outs(%5 : tensor<1x2xf32>) -> tensor<1x2xf32>
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
  util.func public @main_fc_rt(%arg0: tensor<1x3xbf16>, %b: tensor<1x2xf32>) -> tensor<1x2xbf16> {
    %w = util.global.load immutable @fcw3 : tensor<2x3xbf16>
    %0 = flow.dispatch @dispatch_fc_rt::@fc_rt(%arg0, %w, %b) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x3xbf16>, tensor<2x3xbf16>, tensor<1x2xf32>) -> tensor<1x2xbf16>
    util.return %0 : tensor<1x2xbf16>
  }
}

// -----

// The payload uses the matmul value again after the bias add (here it clamps
// the sum against the bias-free product). Folding the bias into the matmul
// would hand that second use the biased value, so the fold does not apply and
// the dispatch is left as it was.

// CHECK-NOT:   @__bias_folded_const_
// CHECK-LABEL: func.func @fc_mm2
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>>
// CHECK:         arith.addf
// CHECK-LABEL: util.func public @main_fc_mm2
// CHECK:         flow.dispatch @dispatch_fc_mm2::@fc_mm2(
// CHECK-SAME:      : (tensor<1x3xbf16>, tensor<2x3xbf16>, tensor<1x2xf32>) -> tensor<1x2xbf16>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @fcw4 = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xbf16>
  util.global private @fcb4 = dense<[[7.000000e+00, -8.000000e+00]]> : tensor<1x2xf32>
  flow.executable private @dispatch_fc_mm2 {
    flow.executable.export public @fc_mm2 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fc_mm2(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>>, %arg3: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xbf16>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>> -> tensor<1x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>> -> tensor<2x3xbf16>
        %2 = iree_tensor_ext.dispatch.tensor.load %arg2, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2xf32>> -> tensor<1x2xf32>
        %3 = tensor.empty() : tensor<1x2xbf16>
        %4 = tensor.empty() : tensor<1x2xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %6 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x3xbf16>, tensor<2x3xbf16>) outs(%5 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %2 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%3 : tensor<1x2xbf16>) {
        ^bb0(%in: f32, %in_0: f32, %out: bf16):
          %8 = arith.addf %in, %in_0 : f32
          %9 = arith.cmpf ugt, %8, %in : f32
          %10 = arith.select %9, %8, %in : f32
          %11 = arith.truncf %10 : f32 to bf16
          linalg.yield %11 : bf16
        } -> tensor<1x2xbf16>
        iree_tensor_ext.dispatch.tensor.store %7, %arg3, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xbf16>>
        return
      }
    }
  }
  util.func public @main_fc_mm2(%arg0: tensor<1x3xbf16>) -> tensor<1x2xbf16> {
    %w = util.global.load immutable @fcw4 : tensor<2x3xbf16>
    %b = util.global.load immutable @fcb4 : tensor<1x2xf32>
    %0 = flow.dispatch @dispatch_fc_mm2::@fc_mm2(%arg0, %w, %b) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x3xbf16>, tensor<2x3xbf16>, tensor<1x2xf32>) -> tensor<1x2xbf16>
    util.return %0 : tensor<1x2xbf16>
  }
}
