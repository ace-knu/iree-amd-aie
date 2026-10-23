// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-annotate-input-devices)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(iree-amdaie-annotate-input-devices)' %s -o /dev/null

// Per-argument input devices are recorded from the stream.tensor.import
// affinities with the stream.topology owner rule (every importer must be able
// to read the chosen device). The 3-device table: GPU and CPU can read the
// NPU, CPU and GPU can read each other, the NPU reads nobody else.

// Single import per argument; existing reflection entries are kept.
// CHECK-LABEL: util.func public @single
// CHECK-SAME: iree.reflection = {amdaie.input_device_ordinals = "0,0,1", amdaie.input_devices = "amdxdna,amdxdna,vulkan", iree.abi.declaration = "keep"}
module attributes {stream.topology = #hal.device.topology<links = [(@npu -> @gpu = {}), (@npu -> @cpu = {}), (@gpu -> @npu = {transparent_access = true}), (@gpu -> @cpu = {transparent_access = true}), (@cpu -> @npu = {transparent_access = true}), (@cpu -> @gpu = {transparent_access = true})]>} {
  util.global private @npu = #hal.device.target<"amdxdna"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.global private @cpu = #hal.device.target<"local"> : !hal.device
  util.func public @single(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "keep"}} {
    %c16 = arith.constant 16 : index
    %0 = stream.tensor.import on(#hal.device.affinity<@npu>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %1 = stream.tensor.import on(#hal.device.affinity<@npu>) %arg1 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %2 = stream.tensor.import on(#hal.device.affinity<@gpu>) %arg2 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    util.return
  }
}

// -----

// Multiple importers (one import per consumer device): {NPU, GPU} -> NPU (the
// GPU can read the NPU, not the reverse), regardless of import order.
// CHECK-LABEL: util.func public @multi
// CHECK-SAME: iree.reflection = {amdaie.input_device_ordinals = "0,1,0", amdaie.input_devices = "amdxdna,vulkan,amdxdna"}
module attributes {stream.topology = #hal.device.topology<links = [(@npu -> @gpu = {}), (@npu -> @cpu = {}), (@gpu -> @npu = {transparent_access = true}), (@gpu -> @cpu = {transparent_access = true}), (@cpu -> @npu = {transparent_access = true}), (@cpu -> @gpu = {transparent_access = true})]>} {
  util.global private @npu = #hal.device.target<"amdxdna"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.global private @cpu = #hal.device.target<"local"> : !hal.device
  util.func public @multi(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) attributes {iree.abi.stub} {
    %c16 = arith.constant 16 : index
    %0 = stream.tensor.import on(#hal.device.affinity<@npu>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %1 = stream.tensor.import on(#hal.device.affinity<@gpu>) %arg1 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %2 = stream.tensor.import on(#hal.device.affinity<@gpu>) %arg2 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %3 = stream.tensor.import on(#hal.device.affinity<@gpu>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %4 = stream.tensor.import on(#hal.device.affinity<@npu>) %arg2 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    util.return
  }
}

// -----

// Tie: GPU and CPU can read each other, both qualify -> both recorded in
// compile (device declaration) order; tooling picks the lead if listed, else
// the first. Scalar (never imported) arguments are empty entries.
// CHECK-LABEL: util.func public @tie_and_scalar
// CHECK-SAME: iree.reflection = {amdaie.input_device_ordinals = "1|2,,0", amdaie.input_devices = "vulkan|local,,amdxdna"}
module attributes {stream.topology = #hal.device.topology<links = [(@npu -> @gpu = {}), (@npu -> @cpu = {}), (@gpu -> @npu = {transparent_access = true}), (@gpu -> @cpu = {transparent_access = true}), (@cpu -> @npu = {transparent_access = true}), (@cpu -> @gpu = {transparent_access = true})]>} {
  util.global private @npu = #hal.device.target<"amdxdna"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.global private @cpu = #hal.device.target<"local"> : !hal.device
  util.func public @tie_and_scalar(%arg0: !hal.buffer_view, %arg1: i32, %arg2: !hal.buffer_view) attributes {iree.abi.stub} {
    %c16 = arith.constant 16 : index
    %0 = stream.tensor.import on(#hal.device.affinity<@cpu>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %1 = stream.tensor.import on(#hal.device.affinity<@gpu>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %2 = stream.tensor.import on(#hal.device.affinity<@npu>) %arg2 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    util.return
  }
}

// -----

// The sync wrapper only calls the $async function (plus fences): it receives
// the callee's entries remapped to its own argument numbers. The fence argument
// of the $async function is never imported, so its entry is empty.
// CHECK-LABEL: util.func public @f$async
// CHECK-SAME: iree.reflection = {amdaie.input_device_ordinals = "0,2,", amdaie.input_devices = "amdxdna,local,"}
// CHECK-LABEL: util.func public @f(
// CHECK-SAME: iree.reflection = {amdaie.input_device_ordinals = "2,0", amdaie.input_devices = "local,amdxdna"}
module attributes {stream.topology = #hal.device.topology<links = [(@npu -> @gpu = {}), (@npu -> @cpu = {}), (@gpu -> @npu = {transparent_access = true}), (@gpu -> @cpu = {transparent_access = true}), (@cpu -> @npu = {transparent_access = true}), (@cpu -> @gpu = {transparent_access = true})]>} {
  util.global private @npu = #hal.device.target<"amdxdna"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.global private @cpu = #hal.device.target<"local"> : !hal.device
  util.func public @f$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence) attributes {iree.abi.model = "coarse-fences", iree.abi.stub} {
    %c16 = arith.constant 16 : index
    %0 = stream.tensor.import on(#hal.device.affinity<@npu>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %1 = stream.tensor.import on(#hal.device.affinity<@cpu>) %arg1 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    util.return
  }
  util.func public @f(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) attributes {iree.abi.stub} {
    %fence = util.null : !hal.fence
    util.call @f$async(%arg1, %arg0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.fence) -> ()
    util.return
  }
}

// -----

// No importer can read the other: recorded as '?' (distinct from the empty
// never-imported entry) with a warning; tooling must refuse to guess.
// CHECK-LABEL: util.func public @unresolvable
// CHECK-SAME: iree.reflection = {amdaie.input_device_ordinals = "?,0", amdaie.input_devices = "?,device_a"}
module attributes {stream.topology = #hal.device.topology<links = [(@dev_a -> @dev_b = {}), (@dev_b -> @dev_a = {})]>} {
  util.global private @dev_a = #hal.device.target<"device_a"> : !hal.device
  util.global private @dev_b = #hal.device.target<"device_b"> : !hal.device
  // expected-warning @+1 {{input 0 is imported on devices that cannot all read any one of them; recorded as '?' in amdaie.input_devices (tooling requires --input_device for it)}}
  util.func public @unresolvable(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) attributes {iree.abi.stub} {
    %c16 = arith.constant 16 : index
    %0 = stream.tensor.import on(#hal.device.affinity<@dev_a>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %1 = stream.tensor.import on(#hal.device.affinity<@dev_b>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %2 = stream.tensor.import on(#hal.device.affinity<@dev_a>) %arg1 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    util.return
  }
}

// -----

// Without a stream.topology the pass is a no-op.
// CHECK-LABEL: util.func public @no_topology
// CHECK-SAME: attributes {iree.abi.stub}
// CHECK-NOT: amdaie.input_devices
module {
  util.global private @npu = #hal.device.target<"amdxdna"> : !hal.device
  util.func public @no_topology(%arg0: !hal.buffer_view) attributes {iree.abi.stub} {
    %c16 = arith.constant 16 : index
    %0 = stream.tensor.import on(#hal.device.affinity<@npu>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    util.return
  }
}
