// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-split-large-contraction-dispatches)' %s | FileCheck %s

// A transpose_b matmul (RHS = [N,K], the layout an ONNX Gemm lowers to) with a
// large reduction K=8192 (> 4096) and output N=1024 (> 512) is split along N into
// N/512 = 2 sub-dispatches. Each sub-dispatch runs a cloned matmul executable
// [32,8192]x[512,8192]->[32,512] on a `flow.tensor.slice` view of the weight
// (its static base offset is recoverable because each clone has a single caller),
// and the two [32,512] results are concatenated back to [32,1024] by a host
// concat dispatch. This keeps each weight's L3->L2 shim DMA small enough to fold
// within the shim addressing limits.

// The split materializes a host concat executable and a cloned, shrunk matmul
// executable ([32,8192]x[512,8192]->[32,512]) per chunk.
// CHECK-DAG:   flow.executable private @concat_executable_{{[0-9]+}}
// CHECK-DAG:   flow.executable private @dispatch_tb_ns{{[0-9]+}}
// CHECK-DAG:   linalg.matmul {{.*}}ins(%{{.+}}, %{{.+}} : tensor<32x8192xbf16>, tensor<512x8192xbf16>) outs(%{{.+}} : tensor<32x512xf32>)

// The rewritten caller: 2 weight-slice views, 2 NPU matmul sub-dispatches, 1
// host concat.
// CHECK-LABEL: util.func public @main_tb
// CHECK:         %[[S0:.+]] = flow.tensor.slice %arg1[%c0, {{.*}}] : tensor<1024x8192xbf16> -> tensor<512x8192xbf16>
// CHECK:         %[[O0:.+]] = flow.dispatch @dispatch_tb_ns{{[0-9]+}}::{{@.+}}(%arg0, %[[S0]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@npu>}
// CHECK-SAME:      : (tensor<32x8192xbf16>, tensor<512x8192xbf16>) -> tensor<32x512xf32>
// CHECK:         %[[S1:.+]] = flow.tensor.slice %arg1[%c512{{.*}}, {{.*}}] : tensor<1024x8192xbf16> -> tensor<512x8192xbf16>
// CHECK:         %[[O1:.+]] = flow.dispatch @dispatch_tb_ns{{[0-9]+}}::{{@.+}}(%arg0, %[[S1]])
// CHECK-SAME:      : (tensor<32x8192xbf16>, tensor<512x8192xbf16>) -> tensor<32x512xf32>
// CHECK:         flow.dispatch @concat_executable_{{[0-9]+}}::{{@.+}}(%[[O0]], %[[O1]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<32x512xf32>, tensor<32x512xf32>) -> tensor<32x1024xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_tb {
    flow.executable.export public @matmul_tb workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_tb(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x8192xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x8192xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x8192xbf16>> -> tensor<32x8192xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1024, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x8192xbf16>> -> tensor<1024x8192xbf16>
        %2 = tensor.empty() : tensor<32x1024xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<32x8192xbf16>, tensor<1024x8192xbf16>) outs(%3 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [32, 1024], strides = [1, 1] : tensor<32x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>
        return
      }
    }
  }
  util.func public @main_tb(%arg0: tensor<32x8192xbf16>, %arg1: tensor<1024x8192xbf16>) -> tensor<32x1024xf32> {
    %0 = flow.dispatch @dispatch_tb::@matmul_tb(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x8192xbf16>, tensor<1024x8192xbf16>) -> tensor<32x1024xf32>
    util.return %0 : tensor<32x1024xf32>
  }
}

// -----

// Reduction K=4096 is not above the threshold (4096) -> no split (the single
// dispatch's weight DMA already folds), so the pass is a no-op.

// CHECK-NOT:   @slice_executable
// CHECK-NOT:   @concat_executable
// CHECK-LABEL: util.func public @main_smallk
// CHECK:         flow.dispatch @dispatch_smallk::@matmul_smallk(%arg0, %arg1)
// CHECK-SAME:      : (tensor<32x4096xbf16>, tensor<1024x4096xbf16>) -> tensor<32x1024xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_smallk {
    flow.executable.export public @matmul_smallk workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_smallk(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x4096xbf16>> -> tensor<32x4096xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [1024, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x4096xbf16>> -> tensor<1024x4096xbf16>
        %2 = tensor.empty() : tensor<32x1024xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<32x4096xbf16>, tensor<1024x4096xbf16>) outs(%3 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [32, 1024], strides = [1, 1] : tensor<32x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>
        return
      }
    }
  }
  util.func public @main_smallk(%arg0: tensor<32x4096xbf16>, %arg1: tensor<1024x4096xbf16>) -> tensor<32x1024xf32> {
    %0 = flow.dispatch @dispatch_smallk::@matmul_smallk(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x4096xbf16>, tensor<1024x4096xbf16>) -> tensor<32x1024xf32>
    util.return %0 : tensor<32x1024xf32>
  }
}

// -----

// A plain (non-transpose_b) matmul RHS = [K,N] has N as its inner dim, which is
// not a contiguous outer slice, so the pass leaves it untouched even at large K.

// CHECK-NOT:   @slice_executable
// CHECK-NOT:   @concat_executable
// CHECK-LABEL: util.func public @main_plain
// CHECK:         flow.dispatch @dispatch_plain::@matmul_plain(%arg0, %arg1)
// CHECK-SAME:      : (tensor<32x8192xbf16>, tensor<8192x1024xbf16>) -> tensor<32x1024xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_plain {
    flow.executable.export public @matmul_plain workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_plain(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x8192xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x1024xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [32, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x8192xbf16>> -> tensor<32x8192xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [8192, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x1024xbf16>> -> tensor<8192x1024xbf16>
        %2 = tensor.empty() : tensor<32x1024xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<32x8192xbf16>, tensor<8192x1024xbf16>) outs(%3 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [32, 1024], strides = [1, 1] : tensor<32x1024xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x1024xf32>>
        return
      }
    }
  }
  util.func public @main_plain(%arg0: tensor<32x8192xbf16>, %arg1: tensor<8192x1024xbf16>) -> tensor<32x1024xf32> {
    %0 = flow.dispatch @dispatch_plain::@matmul_plain(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<32x8192xbf16>, tensor<8192x1024xbf16>) -> tensor<32x1024xf32>
    util.return %0 : tensor<32x1024xf32>
  }
}

// -----

// A GEMV-like matmul (M=1) with dense0's K=25088 and N=4096 is not split even
// though both exceed the thresholds: it is lowered by the GEMV pipeline, whose
// weight DMA subsumes the whole K loop into one descriptor, so the shim
// programming count the split caps is already small.

// CHECK-NOT:   @slice_executable
// CHECK-NOT:   @concat_executable
// CHECK-NOT:   _ns{{[0-9]+}}
// CHECK-LABEL: util.func public @main_gemv
// CHECK:         flow.dispatch @dispatch_gemv::@matmul_gemv(%arg0, %arg1)
// CHECK-SAME:      : (tensor<1x25088xbf16>, tensor<4096x25088xbf16>) -> tensor<1x4096xf32>
// CHECK-NOT:   @concat_executable
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_gemv {
    flow.executable.export public @matmul_gemv workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_gemv(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x25088xbf16>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x25088xbf16>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [1, 25088], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x25088xbf16>> -> tensor<1x25088xbf16>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4096, 25088], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x25088xbf16>> -> tensor<4096x25088xbf16>
        %2 = tensor.empty() : tensor<1x4096xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
        %4 = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins(%0, %1 : tensor<1x25088xbf16>, tensor<4096x25088xbf16>) outs(%3 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [1, 4096], strides = [1, 1] : tensor<1x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4096xf32>>
        return
      }
    }
  }
  util.func public @main_gemv(%arg0: tensor<1x25088xbf16>, %arg1: tensor<4096x25088xbf16>) -> tensor<1x4096xf32> {
    %0 = flow.dispatch @dispatch_gemv::@matmul_gemv(%arg0, %arg1) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<1x25088xbf16>, tensor<4096x25088xbf16>) -> tensor<1x4096xf32>
    util.return %0 : tensor<1x4096xf32>
  }
}
