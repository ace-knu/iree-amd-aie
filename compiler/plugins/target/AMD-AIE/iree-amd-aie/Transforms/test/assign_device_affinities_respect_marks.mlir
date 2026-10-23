// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities)' %s | FileCheck %s --check-prefix=OFF
// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{respect-marks=true})' %s | FileCheck %s --check-prefix=ON

// respect-marks (debug, default off). Off: every dispatch is placed as before
// (contraction/conv -> accelerators round-robin in declaration order, else ->
// host), ignoring any mark. On: (1) a dispatch that already has stream.affinity
// keeps it; (2) otherwise a stream.affinity on a linalg op in its executable
// pins it (contraction/conv root first); (3) unmarked dispatches are placed as
// before, and marked ones do not advance the round-robin.
// Case 1: marks in the opposite order of the round-robin.
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @mm_gpu {
    flow.executable.export public @mm_gpu workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm_gpu(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %4 = linalg.matmul {stream.affinity = #hal.device.affinity<@gpu>} ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  flow.executable private @mm_npu {
    flow.executable.export public @mm_npu workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm_npu(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %4 = linalg.matmul {stream.affinity = #hal.device.affinity<@npu>} ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  util.func public @reversed(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // OFF-LABEL: util.func public @reversed
    // ON-LABEL: util.func public @reversed
    // OFF: flow.dispatch @mm_gpu::@mm_gpu
    // OFF-SAME: stream.affinity = #hal.device.affinity<@npu>
    // ON: flow.dispatch @mm_gpu::@mm_gpu
    // ON-SAME: stream.affinity = #hal.device.affinity<@gpu>
    %0 = flow.dispatch @mm_gpu::@mm_gpu(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    // OFF: flow.dispatch @mm_npu::@mm_npu
    // OFF-SAME: stream.affinity = #hal.device.affinity<@gpu>
    // ON: flow.dispatch @mm_npu::@mm_npu
    // ON-SAME: stream.affinity = #hal.device.affinity<@npu>
    %1 = flow.dispatch @mm_npu::@mm_npu(%0, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    util.return %1 : tensor<4x4xf32>
  }
}

// -----

// Case 2: an affinity already on the dispatch is kept (off: overwritten).
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @mm {
    flow.executable.export public @mm workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  util.func public @existing(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // OFF-LABEL: util.func public @existing
    // ON-LABEL: util.func public @existing
    // OFF: flow.dispatch @mm::@mm
    // OFF-SAME: stream.affinity = #hal.device.affinity<@npu>
    // ON: flow.dispatch @mm::@mm
    // ON-SAME: stream.affinity = #hal.device.affinity<@cpu>
    %0 = flow.dispatch @mm::@mm(%a, %b) {stream.affinity = #hal.device.affinity<@cpu>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    util.return %0 : tensor<4x4xf32>
  }
}

// -----

// Case 3: a marked non-contraction dispatch moves to the marked accelerator,
// and the topology covers the devices actually used (here @gpu and the host).
// With the option off everything stays on the host, so no topology is injected.
// ON: module attributes {
// ON-SAME: stream.topology = #hal.device.topology<links = [(@gpu -> @cpu = {transparent_access = true}), (@cpu -> @gpu = {transparent_access = true})]>
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @ew_gpu {
    flow.executable.export public @ew_gpu workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @ew_gpu(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = tensor.empty() : tensor<4x4xf32>
        %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], stream.affinity = #hal.device.affinity<@gpu>} ins(%0 : tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %3 = arith.maximumf %in, %cst : f32
          linalg.yield %3 : f32
        } -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %2, %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  util.func public @elementwise(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // OFF-LABEL: util.func public @elementwise
    // ON-LABEL: util.func public @elementwise
    // OFF: flow.dispatch @ew_gpu::@ew_gpu
    // OFF-SAME: stream.affinity = #hal.device.affinity<@cpu>
    // ON: flow.dispatch @ew_gpu::@ew_gpu
    // ON-SAME: stream.affinity = #hal.device.affinity<@gpu>
    %0 = flow.dispatch @ew_gpu::@ew_gpu(%a) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    util.return %0 : tensor<4x4xf32>
  }
}

// -----

// Case 4: a marked dispatch between two unmarked ones does not consume a
// round-robin slot (on: npu, [cpu], gpu; off: npu, gpu, npu).
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @mm_a {
    flow.executable.export public @mm_a workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm_a(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  flow.executable private @mm_cpu {
    flow.executable.export public @mm_cpu workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm_cpu(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %4 = linalg.matmul {stream.affinity = #hal.device.affinity<@cpu>} ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  flow.executable private @mm_b {
    flow.executable.export public @mm_b workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm_b(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %4 = linalg.matmul ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  util.func public @mixed(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // OFF-LABEL: util.func public @mixed
    // ON-LABEL: util.func public @mixed
    // OFF: flow.dispatch @mm_a::@mm_a
    // OFF-SAME: stream.affinity = #hal.device.affinity<@npu>
    // ON: flow.dispatch @mm_a::@mm_a
    // ON-SAME: stream.affinity = #hal.device.affinity<@npu>
    %0 = flow.dispatch @mm_a::@mm_a(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    // OFF: flow.dispatch @mm_cpu::@mm_cpu
    // OFF-SAME: stream.affinity = #hal.device.affinity<@gpu>
    // ON: flow.dispatch @mm_cpu::@mm_cpu
    // ON-SAME: stream.affinity = #hal.device.affinity<@cpu>
    %1 = flow.dispatch @mm_cpu::@mm_cpu(%0, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    // OFF: flow.dispatch @mm_b::@mm_b
    // OFF-SAME: stream.affinity = #hal.device.affinity<@npu>
    // ON: flow.dispatch @mm_b::@mm_b
    // ON-SAME: stream.affinity = #hal.device.affinity<@gpu>
    %2 = flow.dispatch @mm_b::@mm_b(%1, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    util.return %2 : tensor<4x4xf32>
  }
}

// -----

// Case 5: conflicting marks in one executable (root matmul @npu, fused
// elementwise @cpu): warning, and the contraction root's mark wins.
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  // expected-warning @+1 {{dispatch has conflicting device marks; using @npu (contraction/conv root)}}
  flow.executable private @mm_conflict {
    flow.executable.export public @mm_conflict workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm_conflict(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %4 = linalg.matmul {stream.affinity = #hal.device.affinity<@npu>} ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"], stream.affinity = #hal.device.affinity<@cpu>} ins(%4 : tensor<4x4xf32>) outs(%2 : tensor<4x4xf32>) {
        ^bb0(%in: f32, %out: f32):
          %6 = arith.maximumf %in, %cst : f32
          linalg.yield %6 : f32
        } -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %5, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  util.func public @conflict(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // OFF-LABEL: util.func public @conflict
    // ON-LABEL: util.func public @conflict
    // OFF: flow.dispatch @mm_conflict::@mm_conflict
    // OFF-SAME: stream.affinity = #hal.device.affinity<@npu>
    // ON: flow.dispatch @mm_conflict::@mm_conflict
    // ON-SAME: stream.affinity = #hal.device.affinity<@npu>
    %0 = flow.dispatch @mm_conflict::@mm_conflict(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    util.return %0 : tensor<4x4xf32>
  }
}

// -----

// Case 6: a mark naming something that is not a placement device is an error
// when respected (off: ignored, round-robin as usual).
module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @mm_bad {
    flow.executable.export public @mm_bad workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mm_bad(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4xf32>) -> tensor<4x4xf32>
        %4 = linalg.matmul {stream.affinity = #hal.device.affinity<@nope>} ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  util.func public @bad_mark(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // OFF-LABEL: util.func public @bad_mark
    // OFF: flow.dispatch @mm_bad::@mm_bad
    // OFF-SAME: stream.affinity = #hal.device.affinity<@npu>
    // expected-error @+1 {{device mark @nope does not name a placement device}}
    %0 = flow.dispatch @mm_bad::@mm_bad(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    util.return %0 : tensor<4x4xf32>
  }
}
