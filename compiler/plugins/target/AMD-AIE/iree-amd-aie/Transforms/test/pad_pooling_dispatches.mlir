// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-pad-pooling-dispatches)' %s | FileCheck %s

// VGG-16 pool4 (28x28x512 -> 14x14x512, 2x2/2) pinned to an npu4 target whose
// pool column policy is `full`: OH 14 is padded up to 16 (a multiple of the 8
// columns). The input grows by 2 x 2 = 4 rows on the host (pad dispatch, fill
// value -inf so the padded rows never win a max), the executable's input
// binding/load, init (empty + fill), pooling result and output binding/store
// grow in place, and the padded result is cropped back to 14 rows on the host.

// CHECK-LABEL: func.func @pool4
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x28x512xf32>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x14x512xf32>>
// CHECK:         %[[CST:.+]] = arith.constant 0xFF800000 : f32
// CHECK:         %[[IN:.+]] = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [32, 28, 512]
// CHECK:         %[[E:.+]] = tensor.empty() : tensor<16x14x512xf32>
// CHECK:         %[[F:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[E]] : tensor<16x14x512xf32>)
// CHECK:         %[[P:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[IN]], %{{.+}} : tensor<32x28x512xf32>, tensor<2x2xf32>) outs(%[[F]] : tensor<16x14x512xf32>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[P]], %arg1, offsets = [0, 0, 0], sizes = [16, 14, 512]

// The host pad dispatch pads with the fill value, the crop dispatch slices the
// 14 valid rows back.
// CHECK:       flow.executable private @pad_executable_0
// CHECK:         %[[NINF:.+]] = arith.constant 0xFF800000 : f32
// CHECK:         tensor.pad %{{.+}} low[0, 0, 0] high[4, 0, 0]
// CHECK:           tensor.yield %[[NINF]] : f32
// CHECK:       flow.executable private @crop_executable_1
// CHECK:         tensor.extract_slice %{{.+}}[0, 0, 0] [14, 14, 512] [1, 1, 1]

// CHECK-LABEL: util.func public @main
// CHECK:         %[[PAD:.+]] = flow.dispatch @pad_executable_0::@pad_dispatch_0(%arg0)
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<28x28x512xf32>) -> tensor<32x28x512xf32>
// CHECK:         %[[OUT:.+]] = flow.dispatch @dispatch_0::@pool4(%[[PAD]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@npu>}
// CHECK-SAME:      : (tensor<32x28x512xf32>) -> tensor<16x14x512xf32>
// CHECK:         %[[CROP:.+]] = flow.dispatch @crop_executable_1::@crop_dispatch_1(%[[OUT]])
// CHECK-SAME:      {stream.affinity = #hal.device.affinity<@cpu>}
// CHECK-SAME:      : (tensor<16x14x512xf32>) -> tensor<14x14x512xf32>
// CHECK:         util.return %[[CROP]]
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, pool_columns_policy = "full", pooling = true, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_0 {
    flow.executable.export public @pool4 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pool4(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x28x512xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>) {
        %cst = arith.constant 0xFF800000 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [28, 28, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x28x512xf32>> -> tensor<28x28x512xf32>
        %1 = tensor.empty() : tensor<14x14x512xf32>
        %2 = tensor.empty() : tensor<2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<14x14x512xf32>) -> tensor<14x14x512xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %2 : tensor<28x28x512xf32>, tensor<2x2xf32>) outs(%3 : tensor<14x14x512xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %5 = arith.maximumf %out, %in : f32
          linalg.yield %5 : f32
        } -> tensor<14x14x512xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg1, offsets = [0, 0, 0], sizes = [14, 14, 512], strides = [1, 1, 1] : tensor<14x14x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: tensor<28x28x512xf32>) -> tensor<14x14x512xf32> {
    %0 = flow.dispatch @dispatch_0::@pool4(%arg0) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<28x28x512xf32>) -> tensor<14x14x512xf32>
    util.return %0 : tensor<14x14x512xf32>
  }
}

// -----

// The same pool4 under the default `max` policy (never pads): untouched, no
// pad / crop dispatches.

// CHECK-LABEL: func.func @pool4_max
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x28x512xf32>>
// CHECK-SAME:    %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>
// CHECK-NOT:   pad_executable
// CHECK-LABEL: util.func public @main_max
// CHECK:         flow.dispatch @dispatch_1::@pool4_max(%arg0)
// CHECK-SAME:      : (tensor<28x28x512xf32>) -> tensor<14x14x512xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, pool_columns_policy = "max", pooling = true, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_1 {
    flow.executable.export public @pool4_max workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pool4_max(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x28x512xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>) {
        %cst = arith.constant 0xFF800000 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [28, 28, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<28x28x512xf32>> -> tensor<28x28x512xf32>
        %1 = tensor.empty() : tensor<14x14x512xf32>
        %2 = tensor.empty() : tensor<2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<14x14x512xf32>) -> tensor<14x14x512xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %2 : tensor<28x28x512xf32>, tensor<2x2xf32>) outs(%3 : tensor<14x14x512xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %5 = arith.maximumf %out, %in : f32
          linalg.yield %5 : f32
        } -> tensor<14x14x512xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg1, offsets = [0, 0, 0], sizes = [14, 14, 512], strides = [1, 1, 1] : tensor<14x14x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<14x14x512xf32>>
        return
      }
    }
  }
  util.func public @main_max(%arg0: tensor<28x28x512xf32>) -> tensor<14x14x512xf32> {
    %0 = flow.dispatch @dispatch_1::@pool4_max(%arg0) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<28x28x512xf32>) -> tensor<14x14x512xf32>
    util.return %0 : tensor<14x14x512xf32>
  }
}

// -----

// VGG-16 pool1 (224x224x64 -> 112x112x64) under `full`: OH 112 already divides
// by 8, nothing to do.

// CHECK-LABEL: func.func @pool1
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x224x64xf32>>
// CHECK-NOT:   pad_executable
// CHECK-LABEL: util.func public @main_pool1
// CHECK:         flow.dispatch @dispatch_2::@pool1(%arg0)
// CHECK-SAME:      : (tensor<224x224x64xf32>) -> tensor<112x112x64xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, pool_columns_policy = "full", pooling = true, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_2 {
    flow.executable.export public @pool1 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pool1(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x224x64xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<112x112x64xf32>>) {
        %cst = arith.constant 0xFF800000 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [224, 224, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<224x224x64xf32>> -> tensor<224x224x64xf32>
        %1 = tensor.empty() : tensor<112x112x64xf32>
        %2 = tensor.empty() : tensor<2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<112x112x64xf32>) -> tensor<112x112x64xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %2 : tensor<224x224x64xf32>, tensor<2x2xf32>) outs(%3 : tensor<112x112x64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %5 = arith.maximumf %out, %in : f32
          linalg.yield %5 : f32
        } -> tensor<112x112x64xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg1, offsets = [0, 0, 0], sizes = [112, 112, 64], strides = [1, 1, 1] : tensor<112x112x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<112x112x64xf32>>
        return
      }
    }
  }
  util.func public @main_pool1(%arg0: tensor<224x224x64xf32>) -> tensor<112x112x64xf32> {
    %0 = flow.dispatch @dispatch_2::@pool1(%arg0) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<224x224x64xf32>) -> tensor<112x112x64xf32>
    util.return %0 : tensor<112x112x64xf32>
  }
}

// -----

// VGG-16 pool5 (14x14x512 -> 7x7x512, 0.5 MB) under `cost`: padding 7 -> 8 for
// 8 columns would cost more (per-column fixed cost + host pad/crop) than the
// 1-column layout the model picks, so the dispatch is left alone.

// CHECK-LABEL: func.func @pool5_cost
// CHECK-SAME:    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<14x14x512xf32>>
// CHECK-NOT:   pad_executable
// CHECK-LABEL: util.func public @main_pool5
// CHECK:         flow.dispatch @dispatch_3::@pool5_cost(%arg0)
// CHECK-SAME:      : (tensor<14x14x512xf32>) -> tensor<7x7x512xf32>
module attributes {stream.affinity.default = #hal.device.affinity<@npu>} {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, pool_columns_policy = "cost", pooling = true, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @dispatch_3 {
    flow.executable.export public @pool5_cost workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pool5_cost(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<14x14x512xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x512xf32>>) {
        %cst = arith.constant 0xFF800000 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0, 0], sizes = [14, 14, 512], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<14x14x512xf32>> -> tensor<14x14x512xf32>
        %1 = tensor.empty() : tensor<7x7x512xf32>
        %2 = tensor.empty() : tensor<2x2xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<7x7x512xf32>) -> tensor<7x7x512xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2 + d3, d1 * 2 + d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%0, %2 : tensor<14x14x512xf32>, tensor<2x2xf32>) outs(%3 : tensor<7x7x512xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %5 = arith.maximumf %out, %in : f32
          linalg.yield %5 : f32
        } -> tensor<7x7x512xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg1, offsets = [0, 0, 0], sizes = [7, 7, 512], strides = [1, 1, 1] : tensor<7x7x512xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<7x7x512xf32>>
        return
      }
    }
  }
  util.func public @main_pool5(%arg0: tensor<14x14x512xf32>) -> tensor<7x7x512xf32> {
    %0 = flow.dispatch @dispatch_3::@pool5_cost(%arg0) {stream.affinity = #hal.device.affinity<@npu>} : (tensor<14x14x512xf32>) -> tensor<7x7x512xf32>
    util.return %0 : tensor<7x7x512xf32>
  }
}
