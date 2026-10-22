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

// With ukernels enabled a transpose_b matmul (GEMV-like, M=1) gets the same
// doubled multiples as a plain matmul: M 1 -> 64, N 64 -> 128.

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
      func.func @matmul_tb_uk(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x64xbf16>> -> tensor<1x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xbf16>> -> tensor<64x64xbf16>
        %2 = tensor.empty() : tensor<1x64xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x64xf32>) -> tensor<1x64xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x64xbf16>, tensor<64x64xbf16>) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : tensor<1x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64xf32>>
        return
      }
    }
  }
  util.func public @main_tb_uk(%arg0: tensor<1x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<1x64xf32> {
    %0 = flow.dispatch @dispatch_tb_uk::@matmul_tb_uk(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x64xbf16>, tensor<64x64xbf16>) -> tensor<1x64xf32>
    util.return %0 : tensor<1x64xf32>
  }
}
