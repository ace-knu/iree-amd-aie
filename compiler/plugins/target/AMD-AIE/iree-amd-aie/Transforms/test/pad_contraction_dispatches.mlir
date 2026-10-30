// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-pad-contraction-dispatches)' --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s

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

// The padding dispatches copy each operand into the low corner of a zero
// splat of the K=32 shape (a strided store inside a dispatch).
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [256, 27], strides = [1, 1] : tensor<256x27xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<256x32xbf16>>
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [27, 64], strides = [1, 1] : tensor<27x64xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x64xbf16>>

// CHECK-LABEL: util.func public @main
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0, %{{[a-z0-9_]+}})
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<256x27xbf16>, tensor<256x32xbf16>) -> %{{.+}}
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1, %{{[a-z0-9_]+}})
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<27x64xbf16>, tensor<32x64xbf16>) -> %{{.+}}
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<224x64xbf16>>
// CHECK-NOT:   flow.executable private @pad_executable_1
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [196, 128] [1, 1]

// CHECK-LABEL: util.func public @main_m
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0, %{{[a-z0-9_]+}})
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<196x64xbf16>, tensor<224x64xbf16>) -> %{{.+}}
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<208x64xf32>>
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<208x64xf32>>
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [64, 200], strides = [1, 1] : tensor<64x200xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<64x256xbf16>>
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [256, 200] [1, 1]

// CHECK-LABEL: util.func public @main_n
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg1, %{{[a-z0-9_]+}})
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<64x200xbf16>, tensor<64x256xbf16>) -> %{{.+}}
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : tensor<196x64xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<224x64xbf16>>
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [64, 200], strides = [1, 1] : tensor<64x200xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<64x256xbf16>>
// CHECK:       flow.executable private @crop_executable_2
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [196, 200] [1, 1]

// CHECK-LABEL: util.func public @main_mn
// CHECK:         %[[LPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0, %{{[a-z0-9_]+}})
// CHECK-SAME:      : (tensor<196x64xbf16>, tensor<224x64xbf16>) -> %{{.+}}
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_1::@pad_dispatch_1(%arg1, %{{[a-z0-9_]+}})
// CHECK-SAME:      : (tensor<64x200xbf16>, tensor<64x256xbf16>) -> %{{.+}}
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [200, 64], strides = [1, 1] : tensor<200x64xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<256x64xbf16>>
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [32, 200] [1, 1]

// CHECK-LABEL: util.func public @main_tb
// CHECK:         %[[RPAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg1, %{{[a-z0-9_]+}})
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<200x64xbf16>, tensor<256x64xbf16>) -> %{{.+}}
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1000, 4096], strides = [1, 1] : tensor<1000x4096xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1024x4096xbf16>>
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [1, 1000] [1, 1]
// CHECK-LABEL: util.func public @main_gemv
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK:         %[[SPLAT:.+]] = flow.tensor.splat %[[ZERO]] : tensor<1024x4096xbf16>
// CHECK:         %[[W:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg1, %[[SPLAT]])
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1, 100], strides = [1, 1] : tensor<1x100xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x128xbf16>>
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1000, 100], strides = [1, 1] : tensor<1000x100xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1024x128xbf16>>
// CHECK:       flow.executable private @crop_executable_2
// CHECK:         tensor.extract_slice %{{.+}}[0, 0] [1, 1000] [1, 1]
// CHECK-LABEL: util.func public @main_gemv_k
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK:         flow.tensor.splat %[[ZERO]] : tensor<1x128xbf16>
// CHECK:         %[[ZERO1:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK:         flow.tensor.splat %[[ZERO1]] : tensor<1024x128xbf16>
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1, 96], strides = [1, 1] : tensor<1x96xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x128xbf16>>
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1024, 96], strides = [1, 1] : tensor<1024x96xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1024x128xbf16>>
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

// From K = 256 on, the matvec GEMV K is padded to the full GEMV K tile of 256
// instead: K=4097 (e.g. after a bias fold) -> 4352, not 4160, since
// KernelDispatch takes the 256 tile only for a K it divides.

// CHECK-LABEL: func.func @matmul_gemv_k256
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4352xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4352xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1, 4097], strides = [1, 1] : tensor<1x4097xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x4352xbf16>>
// CHECK:       flow.executable private @pad_executable_1
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1024, 4097], strides = [1, 1] : tensor<1024x4097xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1024x4352xbf16>>
// CHECK-NOT:   crop_executable
// CHECK-LABEL: util.func public @main_gemv_k256
// CHECK:         flow.dispatch @dispatch_gemv_k256::@matmul_gemv_k256
// CHECK-SAME:      : (tensor<1x4352xbf16>, tensor<1024x4352xbf16>) -> tensor<1x1024xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "all"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv_k256 {
    flow.executable.export public @matmul_gemv_k256 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv_k256(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4097xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4097xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 4097], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4097xbf16>> -> tensor<1x4097xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1024, 4097], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4097xbf16>> -> tensor<1024x4097xbf16>
        %2 = tensor.empty() : tensor<1x1024xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x4097xbf16>, tensor<1024x4097xbf16>) outs(%3 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 1024], strides = [1, 1] : tensor<1x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1024xf32>>
        return
      }
    }
  }
  util.func public @main_gemv_k256(%arg0: tensor<1x4097xbf16>, %arg1: tensor<1024x4097xbf16>) -> tensor<1x1024xf32> {
    %0 = flow.dispatch @dispatch_gemv_k256::@matmul_gemv_k256(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x4097xbf16>, tensor<1024x4097xbf16>) -> tensor<1x1024xf32>
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [2, 4096], strides = [1, 1] : tensor<2x4096xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x4096xbf16>>
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : tensor<1x256xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<32x256xbf16>>
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
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, %{{.+}}, offsets = [0, 0], sizes = [2144, 4096], strides = [1, 1] : tensor<2144x4096xbf16> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<2304x4096xbf16>>
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

// A folded FC layer (M == 1, transpose_b, K = 8 + 1 after the bias fold) with a
// ReLU + truncf epilogue: K is padded (9 -> 32) and, the GEMV shape spreading N
// over every core, N too (64 -> 256). The epilogue grows with the matmul
// result and the bf16 result is cropped back to [1, 64].

// CHECK-LABEL: func.func @gemv_epilogue
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x32xbf16>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32xbf16>>
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x256xbf16>>
// CHECK:         %[[MM:.+]] = linalg.matmul {{.*}} ins(%{{.+}}, %{{.+}} : tensor<1x32xbf16>, tensor<256x32xbf16>) outs(%{{.+}} : tensor<1x256xf32>)
// CHECK:         %[[EPI:.+]] = linalg.generic {{.*}} ins(%[[MM]] : tensor<1x256xf32>) outs(%{{.+}} : tensor<1x256xbf16>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[EPI]], %arg2
// CHECK-LABEL: util.func public @main_gemv_epilogue
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_gemv::@gemv_epilogue({{.*}} : (tensor<1x32xbf16>, tensor<256x32xbf16>) -> tensor<1x256xbf16>
// CHECK:         flow.dispatch @crop_executable_2::@crop_dispatch_2(%[[MMR]]) {{.*}} : (tensor<1x256xbf16>) -> tensor<1x64xbf16>
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv {
    flow.executable.export public @gemv_epilogue workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @gemv_epilogue(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x9xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x9xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64xbf16>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 9], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x9xbf16>> -> tensor<1x9xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 9], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x9xbf16>> -> tensor<64x9xbf16>
        %2 = tensor.empty() : tensor<1x64xbf16>
        %3 = tensor.empty() : tensor<1x64xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x64xf32>) -> tensor<1x64xf32>
        %5 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x9xbf16>, tensor<64x9xbf16>) outs(%4 : tensor<1x64xf32>) -> tensor<1x64xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x64xf32>) outs(%2 : tensor<1x64xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %7 = arith.maximumf %in, %cst : f32
          %8 = arith.truncf %7 : f32 to bf16
          linalg.yield %8 : bf16
        } -> tensor<1x64xbf16>
        iree_tensor_ext.dispatch.tensor.store %6, %arg2, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : tensor<1x64xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64xbf16>>
        return
      }
    }
  }
  util.func public @main_gemv_epilogue(%arg0: tensor<1x9xbf16>, %arg1: tensor<64x9xbf16>) -> tensor<1x64xbf16> {
    %0 = flow.dispatch @dispatch_gemv::@gemv_epilogue(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x9xbf16>, tensor<64x9xbf16>) -> tensor<1x64xbf16>
    util.return %0 : tensor<1x64xbf16>
  }
}

// -----

// An output dim padded under an epilogue (M 196 -> 224, ReLU + truncf): the
// epilogue grows with the matmul result, the store writes the grown epilogue result, and the crop dispatch takes the
// epilogue's element type.

// CHECK-LABEL: func.func @matmul_m_epilogue
// CHECK-SAME:    %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<224x128xbf16>>
// CHECK:         %[[MM:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<224x64xbf16>, tensor<64x128xbf16>) outs(%{{.+}} : tensor<224x128xf32>)
// CHECK:         %[[EPI:.+]] = linalg.generic {{.*}} ins(%[[MM]] : tensor<224x128xf32>) outs(%{{.+}} : tensor<224x128xbf16>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[EPI]], %arg2, offsets = [0, 0], sizes = [224, 128]
// CHECK-LABEL: util.func public @main_m_epilogue
// CHECK:         %[[MMR:.+]] = flow.dispatch @dispatch_me::@matmul_m_epilogue({{.*}} -> tensor<224x128xbf16>
// CHECK:         flow.dispatch @crop_executable_1::@crop_dispatch_1(%[[MMR]]) {{.*}} : (tensor<224x128xbf16>) -> tensor<196x128xbf16>
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_me {
    flow.executable.export public @matmul_m_epilogue workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_m_epilogue(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x128xbf16>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [196, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<196x64xbf16>> -> tensor<196x64xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [64, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x128xbf16>> -> tensor<64x128xbf16>
        %2 = tensor.empty() : tensor<196x128xbf16>
        %3 = tensor.empty() : tensor<196x128xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<196x128xf32>) -> tensor<196x128xf32>
        %5 = linalg.matmul ins(%0, %1 : tensor<196x64xbf16>, tensor<64x128xbf16>) outs(%4 : tensor<196x128xf32>) -> tensor<196x128xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<196x128xf32>) outs(%2 : tensor<196x128xbf16>) {
        ^bb0(%in: f32, %out: bf16):
          %7 = arith.maximumf %in, %cst : f32
          %8 = arith.truncf %7 : f32 to bf16
          linalg.yield %8 : bf16
        } -> tensor<196x128xbf16>
        iree_tensor_ext.dispatch.tensor.store %6, %arg2, offsets = [0, 0], sizes = [196, 128], strides = [1, 1] : tensor<196x128xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<196x128xbf16>>
        return
      }
    }
  }
  util.func public @main_m_epilogue(%arg0: tensor<196x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<196x128xbf16> {
    %0 = flow.dispatch @dispatch_me::@matmul_m_epilogue(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<196x64xbf16>, tensor<64x128xbf16>) -> tensor<196x128xbf16>
    util.return %0 : tensor<196x128xbf16>
  }
}

// -----

// A constant weight (an immutable global with a dense value, read only here)
// is padded at compile time instead: a new global holds it zero-padded,
// K 3 -> 32 and N 2 -> 256, each [N, K] row in place, and the original global
// is dropped. The runtime LHS still gets a host pad dispatch.

// CHECK-LABEL: util.global private @__padded_constant_1
// CHECK-SAME:    dense<{{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 0.000000e+00, {{.*}}], [4.000000e+00, 5.000000e+00, 6.000000e+00, 0.000000e+00, {{.*}}], [0.000000e+00, {{.*}} : tensor<256x32xbf16>
// CHECK-NOT:   @weight
// CHECK-LABEL: util.func public @main_const
// CHECK:         %[[LHS:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0, %{{[a-z0-9_]+}})
// CHECK:         %[[W:.+]] = util.global.load immutable @__padded_constant_1 : tensor<256x32xbf16>
// CHECK:         flow.dispatch @dispatch_const::@matmul_const(%[[LHS]], %[[W]])
// CHECK-NEXT:    flow.dispatch @crop_executable_2::@crop_dispatch_2
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xbf16>
  flow.executable private @dispatch_const {
    flow.executable.export public @matmul_const workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_const(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>> -> tensor<1x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>> -> tensor<2x3xbf16>
        %2 = tensor.empty() : tensor<1x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x3xbf16>, tensor<2x3xbf16>) outs(%3 : tensor<1x2xf32>) -> tensor<1x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xf32>>
        return
      }
    }
  }
  util.func public @main_const(%arg0: tensor<1x3xbf16>) -> tensor<1x2xf32> {
    %w = util.global.load immutable @weight : tensor<2x3xbf16>
    %0 = flow.dispatch @dispatch_const::@matmul_const(%arg0, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x3xbf16>, tensor<2x3xbf16>) -> tensor<1x2xf32>
    util.return %0 : tensor<1x2xf32>
  }
}

// -----

// A constant global something else reads too is padded at run time as before,
// so its padded copy does not sit next to the original.

// CHECK-LABEL: util.func public @main_shared_const
// CHECK-NOT:     __padded_constant
// CHECK:         %[[W:.+]] = util.global.load immutable @weight : tensor<2x3xbf16>
// CHECK:         flow.dispatch @pad_executable_{{[0-9]+}}::@pad_dispatch_{{[0-9]+}}(%[[W]], %{{[a-z0-9_]+}})
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @weight {stream.affinity.default = #hal.device.affinity<@cpu>} = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xbf16>
  flow.executable private @dispatch_shared_const {
    flow.executable.export public @matmul_shared_const workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_shared_const(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x3xbf16>> -> tensor<1x3xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x3xbf16>> -> tensor<2x3xbf16>
        %2 = tensor.empty() : tensor<1x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x2xf32>) -> tensor<1x2xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x3xbf16>, tensor<2x3xbf16>) outs(%3 : tensor<1x2xf32>) -> tensor<1x2xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x2xf32>>
        return
      }
    }
  }
  util.func public @main_shared_const(%arg0: tensor<1x3xbf16>) -> (tensor<1x2xf32>, tensor<2x3xbf16>) {
    %w = util.global.load immutable @weight : tensor<2x3xbf16>
    %0 = flow.dispatch @dispatch_shared_const::@matmul_shared_const(%arg0, %w) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x3xbf16>, tensor<2x3xbf16>) -> tensor<1x2xf32>
    %w2 = util.global.load immutable @weight : tensor<2x3xbf16>
    util.return %0, %w2 : tensor<1x2xf32>, tensor<2x3xbf16>
  }
}
