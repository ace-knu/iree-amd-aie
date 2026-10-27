// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{accelerate-pooling=true})' %s | FileCheck %s --check-prefix=POOL

// Heterogeneous placement: a contraction/conv dispatch is pinned to the amd-aie
// (NPU) device, everything else (here an elementwise dispatch) to the llvm-cpu
// device, and the device topology is injected so cross-device buffers resolve.
// The links are asymmetric: the CPU can host-map an amdxdna BO, but the NPU
// cannot bind a CPU allocation (it is not a BO), so only cpu -> npu is
// transparent.

// CHECK: module attributes {
// CHECK-SAME: stream.topology = #hal.device.topology<links = [(@npu -> @cpu = {}), (@cpu -> @npu = {transparent_access = true})]>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @elementwise workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @elementwise(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.maximumf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_1 {
    flow.executable.export public @matmul workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %2 = tensor.empty() : tensor<128x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
  util.func public @hetero(%arg0: tensor<16384xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    // CHECK: flow.dispatch @dispatch_0::@elementwise
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}
    %0 = flow.dispatch @dispatch_0::@elementwise(%arg0) : (tensor<16384xf32>) -> tensor<16384xf32>
    %1 = flow.tensor.reshape %0 : tensor<16384xf32> -> tensor<128x128xf32>
    // CHECK: flow.dispatch @dispatch_1::@matmul
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@npu>}
    %2 = flow.dispatch @dispatch_1::@matmul(%1, %arg1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    util.return %2 : tensor<128x128xf32>
  }
}

// -----

// Multiple accelerators: the single matmul takes the first round-robin slot
// (@npu0), and the topology meshes just the used devices @npu0 <-> @cpu --
// the unused accelerator @npu1 is not referenced (DCE'd).

// CHECK: module attributes {
// CHECK-SAME: stream.topology = #hal.device.topology<links = [(@npu0 -> @cpu = {}), (@cpu -> @npu0 = {transparent_access = true})]>
module attributes {stream.affinity.default = #hal.device.affinity<@npu0>} {
  util.global private @npu0 = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @npu1 = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @elementwise workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @elementwise(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0], sizes = [16384], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16384xf32>> -> tensor<16384xf32>
        %1 = tensor.empty() : tensor<16384xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<16384xf32>) outs(%1 : tensor<16384xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.maximumf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<16384xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0], sizes = [16384], strides = [1] : tensor<16384xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16384xf32>>
        return
      }
    }
  }
  flow.executable private @dispatch_1 {
    flow.executable.export public @matmul workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %2 = tensor.empty() : tensor<128x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
  util.func public @multi_accel(%arg0: tensor<16384xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    // CHECK: flow.dispatch @dispatch_0::@elementwise
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}
    %0 = flow.dispatch @dispatch_0::@elementwise(%arg0) : (tensor<16384xf32>) -> tensor<16384xf32>
    %1 = flow.tensor.reshape %0 : tensor<16384xf32> -> tensor<128x128xf32>
    // CHECK: flow.dispatch @dispatch_1::@matmul
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@npu0>}
    %2 = flow.dispatch @dispatch_1::@matmul(%1, %arg1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    util.return %2 : tensor<128x128xf32>
  }
}

// -----

// Host-only (no accelerator declared): the contraction also runs on the host,
// and no topology is injected (a single device needs no cross-device links).

// CHECK-NOT: stream.topology
// CHECK-LABEL: util.func public @host_only
module attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_1 {
    flow.executable.export public @matmul workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x128xf32>> -> tensor<128x128xf32>
        %2 = tensor.empty() : tensor<128x128xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [128, 128], strides = [1, 1] : tensor<128x128xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<128x128xf32>>
        return
      }
    }
  }
  util.func public @host_only(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    // CHECK: flow.dispatch @dispatch_1::@matmul
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}
    %2 = flow.dispatch @dispatch_1::@matmul(%arg0, %arg1) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    util.return %2 : tensor<128x128xf32>
  }
}

// -----

// Convolution vs pooling: both match `isaConvolutionOpInterface` structurally
// (windowed reduction), so the reduction body decides placement. A max-pool
// generic (body: `max`) is NOT a multiply-accumulate and runs on the host; the
// SAME structure with a mul-add body is a real convolution and runs on the NPU.
// A named `linalg.conv_2d_nhwc_hwcf` (im2col-free direct conv) also runs on the
// NPU, confirming the direct-conv routing path is preserved.

module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  // Max-pool: structurally a convolution, body is `max` (not mul-add).
  flow.executable private @dispatch_pool {
    flow.executable.export public @maxpool workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @maxpool(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8x8xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4x4xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [8, 8, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8x8xf32>> -> tensor<8x8x8xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = tensor.empty() : tensor<8x4x4xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2 + d3, d2 * 2 + d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<8x8x8xf32>, tensor<2x2xf32>) outs(%2 : tensor<8x4x4xf32>) {
        ^bb0(%in: f32, %win: f32, %out: f32):
          %4 = arith.maximumf %out, %in : f32
          linalg.yield %4 : f32
        } -> tensor<8x4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [0, 0, 0], sizes = [8, 4, 4], strides = [1, 1, 1] : tensor<8x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4x4xf32>>
        return
      }
    }
  }
  // Convolution generic: SAME structure as the pool, body is mul-add.
  flow.executable private @dispatch_convgen {
    flow.executable.export public @convgen workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @convgen(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8x8xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4x4xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [8, 8, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8x8xf32>> -> tensor<8x8x8xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = tensor.empty() : tensor<8x4x4xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 2 + d3, d2 * 2 + d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<8x8x8xf32>, tensor<2x2xf32>) outs(%2 : tensor<8x4x4xf32>) {
        ^bb0(%in: f32, %win: f32, %out: f32):
          %4 = arith.mulf %in, %win : f32
          %5 = arith.addf %out, %4 : f32
          linalg.yield %5 : f32
        } -> tensor<8x4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %3, %arg2, offsets = [0, 0, 0], sizes = [8, 4, 4], strides = [1, 1, 1] : tensor<8x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4x4xf32>>
        return
      }
    }
  }
  // Named direct convolution (im2col-free path).
  flow.executable private @dispatch_conv {
    flow.executable.export public @conv workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x8x8x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4x8xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x6x6x8xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0, 0], sizes = [1, 8, 8, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x8x8x4xf32>> -> tensor<1x8x8x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0, 0, 0], sizes = [3, 3, 4, 8], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4x8xf32>> -> tensor<3x3x4x8xf32>
        %2 = tensor.empty() : tensor<1x6x6x8xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
        %4 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) outs(%3 : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0, 0, 0], sizes = [1, 6, 6, 8], strides = [1, 1, 1, 1] : tensor<1x6x6x8xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x6x6x8xf32>>
        return
      }
    }
  }
  util.func public @conv_vs_pool(%img: tensor<8x8x8xf32>, %win: tensor<2x2xf32>, %cimg: tensor<1x8x8x4xf32>, %filt: tensor<3x3x4x8xf32>) -> (tensor<8x4x4xf32>, tensor<8x4x4xf32>, tensor<1x6x6x8xf32>) {
    // CHECK: flow.dispatch @dispatch_pool::@maxpool
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}
    %0 = flow.dispatch @dispatch_pool::@maxpool(%img, %win) : (tensor<8x8x8xf32>, tensor<2x2xf32>) -> tensor<8x4x4xf32>
    // CHECK: flow.dispatch @dispatch_convgen::@convgen
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@npu>}
    %1 = flow.dispatch @dispatch_convgen::@convgen(%img, %win) : (tensor<8x8x8xf32>, tensor<2x2xf32>) -> tensor<8x4x4xf32>
    // CHECK: flow.dispatch @dispatch_conv::@conv
    // CHECK-SAME: {stream.affinity = #hal.device.affinity<@npu>}
    %2 = flow.dispatch @dispatch_conv::@conv(%cimg, %filt) : (tensor<1x8x8x4xf32>, tensor<3x3x4x8xf32>) -> tensor<1x6x6x8xf32>
    util.return %0, %1, %2 : tensor<8x4x4xf32>, tensor<8x4x4xf32>, tensor<1x6x6x8xf32>
  }
}

// -----

// A 2-D max pooling generic (channels-last, `d*2 + k` maps, maximumf body)
// structurally looks like a convolution but has no multiply-accumulate body.
// It stays on the host by default and is accelerator work (here the NPU) with
// `accelerate-pooling`.

// CHECK-LABEL: util.func public @pool
// CHECK: flow.dispatch @pool_dispatch::@pool
// CHECK-SAME: {stream.affinity = #hal.device.affinity<@cpu>}
// POOL-LABEL: util.func public @pool
// POOL: flow.dispatch @pool_dispatch::@pool
// POOL-SAME: {stream.affinity = #hal.device.affinity<@npu>}
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  flow.executable private @pool_dispatch {
    flow.executable.export public @pool workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pool(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16x16xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8x16xf32>>) {
        %cst = arith.constant 0xFF800000 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [16, 16, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16x16xf32>> -> tensor<16x16x16xf32>
        %1 = tensor.empty() : tensor<8x8x16xf32>
        %2 = tensor.empty() : tensor<2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<8x8x16xf32>) -> tensor<8x8x16xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %2 : tensor<16x16x16xf32>, tensor<2x2xf32>) outs(%3 : tensor<8x8x16xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %5 = arith.maximumf %out, %in : f32
          linalg.yield %5 : f32
        } -> tensor<8x8x16xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg1, offsets = [0, 0, 0], sizes = [8, 8, 16], strides = [1, 1, 1] : tensor<8x8x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x8x16xf32>>
        return
      }
    }
  }
  util.func public @pool(%arg0: tensor<16x16x16xf32>) -> tensor<8x8x16xf32> {
    %0 = flow.dispatch @pool_dispatch::@pool(%arg0) : (tensor<16x16x16xf32>) -> tensor<8x8x16xf32>
    util.return %0 : tensor<8x8x16xf32>
  }
}

// -----

// A max pooling fused with another compute op (here the NHWC -> NCHW transpose
// that dispatch formation attaches when no channels-last conv follows) stays
// on the host even with `accelerate-pooling`: the pool pipeline handles a
// dispatch of fill + pooling only.
// CHECK-LABEL: util.func public @pool_fused_transpose
// CHECK:         flow.dispatch @dispatch_pool_t::@pool_t
// CHECK-SAME:      stream.affinity = #hal.device.affinity<@cpu>
// POOL-LABEL: util.func public @pool_fused_transpose
// POOL:         flow.dispatch @dispatch_pool_t::@pool_t
// POOL-SAME:      stream.affinity = #hal.device.affinity<@cpu>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  flow.executable private @dispatch_pool_t {
    flow.executable.export public @pool_t workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pool_t(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8x8xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4x4xf32>>) {
        %cst = arith.constant 0xFF800000 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [8, 8, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8x8x8xf32>> -> tensor<8x8x8xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [2, 2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x2xf32>> -> tensor<2x2xf32>
        %2 = tensor.empty() : tensor<4x4x8xf32>
        %f = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4x8xf32>) -> tensor<4x4x8xf32>
        %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %1 : tensor<8x8x8xf32>, tensor<2x2xf32>) outs(%f : tensor<4x4x8xf32>) {
        ^bb0(%in: f32, %win: f32, %out: f32):
          %4 = arith.maximumf %out, %in : f32
          linalg.yield %4 : f32
        } -> tensor<4x4x8xf32>
        %5 = tensor.empty() : tensor<8x4x4xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<4x4x8xf32>) outs(%5 : tensor<8x4x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<8x4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %arg2, offsets = [0, 0, 0], sizes = [8, 4, 4], strides = [1, 1, 1] : tensor<8x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8x4x4xf32>>
        return
      }
    }
  }
  util.func public @pool_fused_transpose(%a: tensor<8x8x8xf32>, %w: tensor<2x2xf32>) -> tensor<8x4x4xf32> {
    %r = flow.dispatch @dispatch_pool_t::@pool_t(%a, %w) : (tensor<8x8x8xf32>, tensor<2x2xf32>) -> tensor<8x4x4xf32>
    util.return %r : tensor<8x4x4xf32>
  }
}
