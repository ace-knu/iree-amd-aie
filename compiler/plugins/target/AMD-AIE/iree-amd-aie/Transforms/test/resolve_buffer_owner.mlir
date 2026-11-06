// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-amdaie-resolve-buffer-owner)' %s | FileCheck %s

// Owner of #hal.device.optimal allocations from the stream.topology table:
// candidate X qualifies iff every candidate Y has (Y -> X) transparent_access.
// One qualifier resolves the allocation. On a tie, the qualifier that is the
// only device using the resource outside initializers owns it; otherwise the
// allocation stays optimal<> (runtime lead).

// Tie (CPU and GPU can read each other). A constant written once by the CPU in
// an initializer and read per invocation only by the GPU (through a global)
// is owned by the GPU.
// CHECK-LABEL: util.initializer
// CHECK: stream.resource.alloca uninitialized on(#hal.device.affinity<@gpu>) : !stream.resource<constant>
module attributes {stream.topology = #hal.device.topology<links = [(@cpu -> @gpu = {transparent_access = true}), (@gpu -> @cpu = {transparent_access = true})]>} {
  util.global private @cpu = #hal.device.target<"local"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.global private @hoisted : !stream.resource<constant>
  util.initializer {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c255_i32 = arith.constant 255 : i32
    %res, %tp = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>) : !stream.resource<constant>{%c16} => !stream.timepoint
    %t1 = stream.cmd.execute on(#hal.device.affinity<@cpu>) await(%tp) => with(%res as %a: !stream.resource<constant>{%c16}) {
      stream.cmd.fill %c255_i32, %a[%c0 for %c16] : i32 -> !stream.resource<constant>{%c16}
    } => !stream.timepoint
    %ready = stream.timepoint.await sync %t1 => %res : !stream.resource<constant>{%c16}
    util.global.store %ready, @hoisted : !stream.resource<constant>
    util.return
  }
  util.func public @read_on_gpu(%arg0: !stream.resource<external>) -> !stream.timepoint {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %hoisted = util.global.load immutable @hoisted : !stream.resource<constant>
    %t = stream.cmd.execute on(#hal.device.affinity<@gpu>) with(%hoisted as %a: !stream.resource<constant>{%c16}, %arg0 as %b: !stream.resource<external>{%c16}) {
      stream.cmd.copy %a[%c0], %b[%c0], %c16 : !stream.resource<constant>{%c16} -> !stream.resource<external>{%c16}
    } => !stream.timepoint
    util.return %t : !stream.timepoint
  }
}

// -----

// Tie, and both qualifiers use the transient per invocation: it stays
// optimal<> (alloca and dealloca), as before.
// CHECK-LABEL: util.func public @used_by_both
// CHECK: stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>)
// CHECK: stream.resource.dealloca on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>)
module attributes {stream.topology = #hal.device.topology<links = [(@cpu -> @gpu = {transparent_access = true}), (@gpu -> @cpu = {transparent_access = true})]>} {
  util.global private @cpu = #hal.device.target<"local"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.func public @used_by_both(%arg0: !stream.resource<external>) -> !stream.timepoint {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c255_i32 = arith.constant 255 : i32
    %res, %tp = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>) : !stream.resource<transient>{%c16} => !stream.timepoint
    %t1 = stream.cmd.execute on(#hal.device.affinity<@cpu>) await(%tp) => with(%res as %a: !stream.resource<transient>{%c16}) {
      stream.cmd.fill %c255_i32, %a[%c0 for %c16] : i32 -> !stream.resource<transient>{%c16}
    } => !stream.timepoint
    %t2 = stream.cmd.execute on(#hal.device.affinity<@gpu>) await(%t1) => with(%res as %a: !stream.resource<transient>{%c16}, %arg0 as %b: !stream.resource<external>{%c16}) {
      stream.cmd.copy %a[%c0], %b[%c0], %c16 : !stream.resource<transient>{%c16} -> !stream.resource<external>{%c16}
    } => !stream.timepoint
    %t3 = stream.resource.dealloca on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>) await(%t2) => %res : !stream.resource<transient>{%c16} => !stream.timepoint
    util.return %t3 : !stream.timepoint
  }
}

// -----

// One qualifier (the NPU cannot read the GPU): resolved by the table alone,
// whoever uses the resource (unchanged rule).
// CHECK-LABEL: util.func public @one_qualifier
// CHECK: stream.resource.alloca uninitialized on(#hal.device.affinity<@npu>)
// CHECK: stream.resource.dealloca on(#hal.device.affinity<@npu>)
module attributes {stream.topology = #hal.device.topology<links = [(@npu -> @gpu = {}), (@gpu -> @npu = {transparent_access = true})]>} {
  util.global private @npu = #hal.device.target<"amdxdna"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.func public @one_qualifier(%arg0: !stream.resource<external>) -> !stream.timepoint {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c255_i32 = arith.constant 255 : i32
    %res, %tp = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@npu>, #hal.device.affinity<@gpu>]>) : !stream.resource<transient>{%c16} => !stream.timepoint
    %t1 = stream.cmd.execute on(#hal.device.affinity<@gpu>) await(%tp) => with(%res as %a: !stream.resource<transient>{%c16}) {
      stream.cmd.fill %c255_i32, %a[%c0 for %c16] : i32 -> !stream.resource<transient>{%c16}
    } => !stream.timepoint
    %t2 = stream.resource.dealloca on(#hal.device.optimal<[#hal.device.affinity<@npu>, #hal.device.affinity<@gpu>]>) await(%t1) => %res : !stream.resource<transient>{%c16} => !stream.timepoint
    util.return %t2 : !stream.timepoint
  }
}

// -----

// Tie, and per invocation only the GPU uses the transient (also through a
// stream.timepoint.await result): the GPU owns it and its dealloca follows.
// CHECK-LABEL: util.func public @used_by_one
// CHECK: stream.resource.alloca uninitialized on(#hal.device.affinity<@gpu>)
// CHECK: stream.resource.dealloca on(#hal.device.affinity<@gpu>)
module attributes {stream.topology = #hal.device.topology<links = [(@cpu -> @gpu = {transparent_access = true}), (@gpu -> @cpu = {transparent_access = true})]>} {
  util.global private @cpu = #hal.device.target<"local"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.func public @used_by_one(%arg0: !stream.resource<external>) -> !stream.timepoint {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c255_i32 = arith.constant 255 : i32
    %res, %tp = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>) : !stream.resource<transient>{%c16} => !stream.timepoint
    %t1 = stream.cmd.execute on(#hal.device.affinity<@gpu>) await(%tp) => with(%res as %a: !stream.resource<transient>{%c16}) {
      stream.cmd.fill %c255_i32, %a[%c0 for %c16] : i32 -> !stream.resource<transient>{%c16}
    } => !stream.timepoint
    %ready = stream.timepoint.await %t1 => %res : !stream.resource<transient>{%c16}
    %t2 = stream.cmd.execute on(#hal.device.affinity<@gpu>) with(%ready as %a: !stream.resource<transient>{%c16}, %arg0 as %b: !stream.resource<external>{%c16}) {
      stream.cmd.copy %a[%c0], %b[%c0], %c16 : !stream.resource<transient>{%c16} -> !stream.resource<external>{%c16}
    } => !stream.timepoint
    %t3 = stream.resource.dealloca on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>) await(%t2) => %ready : !stream.resource<transient>{%c16} => !stream.timepoint
    util.return %t3 : !stream.timepoint
  }
}

// -----

// Tie, but the resource escapes the function (returned): its other users are
// unknown, so it stays optimal<>.
// CHECK-LABEL: util.func public @escapes
// CHECK: stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>)
module attributes {stream.topology = #hal.device.topology<links = [(@cpu -> @gpu = {transparent_access = true}), (@gpu -> @cpu = {transparent_access = true})]>} {
  util.global private @cpu = #hal.device.target<"local"> : !hal.device
  util.global private @gpu = #hal.device.target<"vulkan"> : !hal.device
  util.func public @escapes() -> (!stream.resource<transient>, !stream.timepoint) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c255_i32 = arith.constant 255 : i32
    %res, %tp = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@cpu>, #hal.device.affinity<@gpu>]>) : !stream.resource<transient>{%c16} => !stream.timepoint
    %t1 = stream.cmd.execute on(#hal.device.affinity<@gpu>) await(%tp) => with(%res as %a: !stream.resource<transient>{%c16}) {
      stream.cmd.fill %c255_i32, %a[%c0 for %c16] : i32 -> !stream.resource<transient>{%c16}
    } => !stream.timepoint
    util.return %res, %t1 : !stream.resource<transient>, !stream.timepoint
  }
}
