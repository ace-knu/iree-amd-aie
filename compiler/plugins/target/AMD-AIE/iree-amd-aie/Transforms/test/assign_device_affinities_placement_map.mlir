// RUN: iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities)' %s | FileCheck %s --check-prefix=OFF
// RUN: iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities)' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFLOG --allow-empty
// RUN: iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d0=gpu;d1=npu;*=cpu"})' %s | FileCheck %s --check-prefix=MAP
// RUN: iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d0=gpu;d1=npu;*=cpu"})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MAPLOG --allow-empty
// RUN: iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d0-d1=gpu;~elementwise=npu"})' %s | FileCheck %s --check-prefix=RNG
// RUN: iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d0-d1=gpu;~elementwise=npu"})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=RNGLOG --allow-empty
// RUN: iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d3,d0=@gpu;*=@cpu"})' %s | FileCheck %s --check-prefix=LST
// RUN: ! iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d9=gpu;*=cpu"})' %s -o /dev/null 2> %t.errnum
// RUN: FileCheck %s --check-prefix=ERRNUM --input-file=%t.errnum
// RUN: ! iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="*=cpu;d0=gpu"})' %s -o /dev/null 2> %t.errshadow
// RUN: FileCheck %s --check-prefix=ERRSHADOW --input-file=%t.errshadow
// RUN: ! iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d0=tpu"})' %s -o /dev/null 2> %t.errdev
// RUN: FileCheck %s --check-prefix=ERRDEV --input-file=%t.errdev
// RUN: ! iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d0gpu"})' %s -o /dev/null 2> %t.errsyn
// RUN: FileCheck %s --check-prefix=ERRSYN --input-file=%t.errsyn
// RUN: ! iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d3-d1=gpu"})' %s -o /dev/null 2> %t.errrev
// RUN: FileCheck %s --check-prefix=ERRREV --input-file=%t.errrev
// RUN: ! iree-opt --pass-pipeline='builtin.module(iree-amdaie-assign-device-affinities{placement-map="d0=@tpu"})' %s -o /dev/null 2> %t.errsym
// RUN: FileCheck %s --check-prefix=ERRSYM --input-file=%t.errsym

// placement-map (experiment, default empty = off). Rules '<selector>=<device>'
// pin dispatches by the dispatch number N of the executable name
// '<func>_dispatch_N' (dN, dN-dM, dA,dB), by a name substring (~substr) or all
// (*); the first matching rule wins. Unmatched dispatches are placed as before
// (contraction/conv -> accelerators round-robin, else -> host) and mapped ones
// do not advance the round-robin. With the flag the placement is printed to
// stderr; without it nothing is printed.
// Dispatches: d0 matmul, d1 matmul, d2 elementwise, d3 matmul.

// Without the flag: round-robin npu, gpu, (host), npu.
// OFF-LABEL: util.func public @model
// OFF: flow.dispatch @model_dispatch_0::@model_dispatch_0_matmul_4x4x4_f32
// OFF-SAME: stream.affinity = #hal.device.affinity<@npu>
// OFF: flow.dispatch @model_dispatch_1::@model_dispatch_1_matmul_4x4x4_f32
// OFF-SAME: stream.affinity = #hal.device.affinity<@gpu>
// OFF: flow.dispatch @model_dispatch_2::@model_dispatch_2_elementwise_4x4_f32
// OFF-SAME: stream.affinity = #hal.device.affinity<@cpu>
// OFF: flow.dispatch @model_dispatch_3::@model_dispatch_3_matmul_4x4x4_f32
// OFF-SAME: stream.affinity = #hal.device.affinity<@npu>
// OFFLOG-NOT: placement-map

// Numbers, opposite of the round-robin, with '*' as the default.
// MAP-LABEL: util.func public @model
// MAP: flow.dispatch @model_dispatch_0::
// MAP-SAME: stream.affinity = #hal.device.affinity<@gpu>
// MAP: flow.dispatch @model_dispatch_1::
// MAP-SAME: stream.affinity = #hal.device.affinity<@npu>
// MAP: flow.dispatch @model_dispatch_2::
// MAP-SAME: stream.affinity = #hal.device.affinity<@cpu>
// MAP: flow.dispatch @model_dispatch_3::
// MAP-SAME: stream.affinity = #hal.device.affinity<@cpu>
// MAPLOG: [amdaie placement-map] rules=3 dispatches=4 npu=1 gpu=1 cpu=2 unmatched=0
// MAPLOG-NEXT: [amdaie placement-map] rule#1 d0=gpu: 1
// MAPLOG-NEXT: [amdaie placement-map] rule#2 d1=npu: 1
// MAPLOG-NEXT: [amdaie placement-map] rule#3 *=cpu: 2
// MAPLOG-NEXT: [amdaie placement-map] d0 model_dispatch_0_matmul_4x4x4_f32 rule#1 d0=gpu @gpu (gpu)
// MAPLOG-NEXT: [amdaie placement-map] d1 model_dispatch_1_matmul_4x4x4_f32 rule#2 d1=npu @npu (npu)
// MAPLOG-NEXT: [amdaie placement-map] d2 model_dispatch_2_elementwise_4x4_f32 rule#3 *=cpu @cpu (cpu)
// MAPLOG-NEXT: [amdaie placement-map] d3 model_dispatch_3_matmul_4x4x4_f32 rule#3 *=cpu @cpu (cpu)

// Range and substring; d3 matches no rule, so it takes the round-robin, which
// starts at the first accelerator because mapped dispatches do not advance it.
// Sending the elementwise dispatch to amd-aie warns (no contraction/conv).
// RNG-LABEL: util.func public @model
// RNG: flow.dispatch @model_dispatch_0::
// RNG-SAME: stream.affinity = #hal.device.affinity<@gpu>
// RNG: flow.dispatch @model_dispatch_1::
// RNG-SAME: stream.affinity = #hal.device.affinity<@gpu>
// RNG: flow.dispatch @model_dispatch_2::
// RNG-SAME: stream.affinity = #hal.device.affinity<@npu>
// RNG: flow.dispatch @model_dispatch_3::
// RNG-SAME: stream.affinity = #hal.device.affinity<@npu>
// RNGLOG: warning: placement-map places a dispatch without a contraction/convolution on amd-aie device @npu (rule '~elementwise=npu'); amd-aie may fail to lower it
// RNGLOG: [amdaie placement-map] rules=2 dispatches=4 npu=2 gpu=2 cpu=0 unmatched=1
// RNGLOG-NEXT: [amdaie placement-map] rule#1 d0-d1=gpu: 2
// RNGLOG-NEXT: [amdaie placement-map] rule#2 ~elementwise=npu: 1
// RNGLOG: [amdaie placement-map] d3 model_dispatch_3_matmul_4x4x4_f32 (no rule: default placement) @npu (npu)

// List and direct device symbols; only @gpu and @cpu are used, so the
// injected topology covers just those two.
// LST: module attributes {
// LST-SAME: stream.topology = #hal.device.topology<links = [(@gpu -> @cpu = {}), (@cpu -> @gpu = {transparent_access = true})]>
// LST-LABEL: util.func public @model
// LST: flow.dispatch @model_dispatch_0::
// LST-SAME: stream.affinity = #hal.device.affinity<@gpu>
// LST: flow.dispatch @model_dispatch_1::
// LST-SAME: stream.affinity = #hal.device.affinity<@cpu>
// LST: flow.dispatch @model_dispatch_2::
// LST-SAME: stream.affinity = #hal.device.affinity<@cpu>
// LST: flow.dispatch @model_dispatch_3::
// LST-SAME: stream.affinity = #hal.device.affinity<@gpu>

// Errors.
// ERRNUM: error: placement-map rule #1 'd9=gpu' matches no dispatch
// ERRSHADOW: error: placement-map rule #2 'd0=gpu' matches no dispatch
// ERRDEV: error: placement-map rule 1 'd0=tpu': unknown device 'tpu'
// ERRSYN: error: placement-map rule 1 'd0gpu': expected '<selector>=<device>'
// ERRREV: error: placement-map rule 1 'd3-d1=gpu': reversed dispatch range 'd3-d1'
// ERRSYM: error: placement-map rule 1 'd0=@tpu': device @tpu is not a placement device of this module

module {
  util.global private @npu = #hal.device.target<"amdxdna", [#hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>]> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan", [#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {}>]> : !hal.device
  util.global private @cpu = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>]> : !hal.device
  flow.executable private @model_dispatch_0 {
    flow.executable.export public @model_dispatch_0_matmul_4x4x4_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @model_dispatch_0_matmul_4x4x4_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
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
  flow.executable private @model_dispatch_1 {
    flow.executable.export public @model_dispatch_1_matmul_4x4x4_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @model_dispatch_1_matmul_4x4x4_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
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
  flow.executable private @model_dispatch_2 {
    flow.executable.export public @model_dispatch_2_elementwise_4x4_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @model_dispatch_2_elementwise_4x4_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
        %0 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %1 = iree_tensor_ext.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>> -> tensor<4x4xf32>
        %2 = tensor.empty() : tensor<4x4xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%2 : tensor<4x4xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %5 = arith.addf %in, %in_0 : f32
          linalg.yield %5 : f32
        } -> tensor<4x4xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %arg2, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : tensor<4x4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>
        return
      }
    }
  }
  flow.executable private @model_dispatch_3 {
    flow.executable.export public @model_dispatch_3_matmul_4x4x4_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @model_dispatch_3_matmul_4x4x4_f32(%arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg1: !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4xf32>>, %arg2: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4xf32>>) {
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
  util.func public @model(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = flow.dispatch @model_dispatch_0::@model_dispatch_0_matmul_4x4x4_f32(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = flow.dispatch @model_dispatch_1::@model_dispatch_1_matmul_4x4x4_f32(%0, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = flow.dispatch @model_dispatch_2::@model_dispatch_2_elementwise_4x4_f32(%1, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = flow.dispatch @model_dispatch_3::@model_dispatch_3_matmul_4x4x4_f32(%2, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    util.return %3 : tensor<4x4xf32>
  }
}
