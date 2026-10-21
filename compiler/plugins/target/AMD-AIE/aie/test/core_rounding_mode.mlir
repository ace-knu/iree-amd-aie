// RUN: iree-opt --pass-pipeline="builtin.module(amdaie-standard-lowering)" %s --split-input-file | FileCheck %s --check-prefix=AUTO
// RUN: iree-opt --pass-pipeline="builtin.module(amdaie-standard-lowering{rounding-mode=none})" %s --split-input-file | FileCheck %s --check-prefix=NONE
// RUN: iree-opt --pass-pipeline="builtin.module(amdaie-standard-lowering{rounding-mode=floor})" %s --split-input-file | FileCheck %s --check-prefix=FLOOR
// RUN: iree-opt --pass-pipeline="builtin.module(amdaie-standard-lowering{lower-to-chess=1})" %s --split-input-file | FileCheck %s --check-prefix=CHESS
// RUN: iree-opt --pass-pipeline="builtin.module(amdaie-standard-lowering{lower-to-chess=1 rounding-mode=conv_even})" %s --split-input-file 2>&1 | FileCheck %s --check-prefix=CHESS-EXPLICIT
// RUN: iree-opt --pass-pipeline="builtin.module(amdaie-standard-lowering{rounding-mode=bogus})" %s --split-input-file --verify-diagnostics -o /dev/null

// The rounding mode is written to crRnd (control register 1 on AIE2P) as the
// first operation of the core, before any lock or compute op.

// AUTO-LABEL:  module @rnd_npu4
// AUTO:          func.func @core_1_2() {
// AUTO-NEXT:       %[[REG1:.*]] = arith.constant 1 : i32
// AUTO-NEXT:       %[[VAL1:.*]] = arith.constant 12 : i32
// AUTO-NEXT:       call @llvm.aie2p.set.ctrl.reg(%[[REG1]], %[[VAL1]]) : (i32, i32) -> ()
// AUTO-NEXT:       return
// AUTO:          func.func @core_0_2() {
// AUTO-NEXT:       %[[REG:.*]] = arith.constant 1 : i32
// AUTO-NEXT:       %[[VAL:.*]] = arith.constant 12 : i32
// AUTO-NEXT:       call @llvm.aie2p.set.ctrl.reg(%[[REG]], %[[VAL]]) : (i32, i32) -> ()
// AUTO:            call @llvm.aie2p.acquire
// AUTO:          func.func private @llvm.aie2p.set.ctrl.reg(i32, i32)

// NONE-LABEL:  module @rnd_npu4
// NONE-NOT:      set.ctrl.reg
// NONE:          call @llvm.aie2p.acquire
// NONE-NOT:      set.ctrl.reg

// FLOOR-LABEL: module @rnd_npu4
// FLOOR:         func.func @core_0_2() {
// FLOOR-NEXT:      %[[REG:.*]] = arith.constant 1 : i32
// FLOOR-NEXT:      %[[VAL:.*]] = arith.constant 0 : i32
// FLOOR-NEXT:      call @llvm.aie2p.set.ctrl.reg(%[[REG]], %[[VAL]]) : (i32, i32) -> ()

// CHESS-LABEL: module @rnd_npu4
// CHESS-NOT:     set.ctrl.reg
// CHESS:         call @llvm.aie2.acquire
// CHESS-NOT:     set.ctrl.reg

// CHESS-EXPLICIT: warning: rounding mode 'conv_even' is ignored when lowering to chess
// CHESS-EXPLICIT-NOT: set.ctrl.reg

module @rnd_npu4 {
 // expected-error @+1 {{unknown rounding mode 'bogus'}}
 aie.device(npu4) {
  %t02 = aie.tile(0, 2)
  %t12 = aie.tile(1, 2)
  %c02 = aie.core(%t02) {
    %c56 = arith.constant 56 : index
    aie.use_lock(%c56, Acquire, 0)
    aie.use_lock(%c56, Release, 1)
    aie.end
  }
  %c12 = aie.core(%t12) {
    aie.end
  }
 }
}

// -----

// 'auto' writes nothing on AIE2; an explicit mode is written to crRnd
// (control register 6 on AIE2).

// AUTO-LABEL:  module @rnd_npu1
// AUTO-NOT:      set.ctrl.reg

// NONE-LABEL:  module @rnd_npu1
// NONE-NOT:      set.ctrl.reg

// FLOOR-LABEL: module @rnd_npu1
// FLOOR:         func.func @core_0_2() {
// FLOOR-NEXT:      %[[REG:.*]] = arith.constant 6 : i32
// FLOOR-NEXT:      %[[VAL:.*]] = arith.constant 0 : i32
// FLOOR-NEXT:      call @llvm.aie2.set.ctrl.reg(%[[REG]], %[[VAL]]) : (i32, i32) -> ()
// FLOOR:         func.func private @llvm.aie2.set.ctrl.reg(i32, i32)

// CHESS-LABEL: module @rnd_npu1
// CHESS-NOT:     set.ctrl.reg

module @rnd_npu1 {
 // expected-error @+1 {{unknown rounding mode 'bogus'}}
 aie.device(npu1_4col) {
  %t02 = aie.tile(0, 2)
  %c02 = aie.core(%t02) {
    %c56 = arith.constant 56 : index
    aie.use_lock(%c56, Acquire, 0)
    aie.use_lock(%c56, Release, 1)
    aie.end
  }
 }
}
