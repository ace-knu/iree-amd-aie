// RUN: iree-opt %s -split-input-file --convert-aievec-to-llvm -canonicalize | FileCheck %s

// CHECK-LABEL: @shuffle_single_operand_nocast
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>

#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {
func.func @shuffle_single_operand_nocast(%lhs : vector<16xi32>)
                -> vector<16xi32> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(34 : i32) : i32
  // CHECK: %[[RHS:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
  // CHECK: %[[R:.*]] = "xllvm.intr.aie2.vshuffle"(%[[LHS]], %[[RHS]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  %0 = aievec.shuffle %lhs [t32_4x4] : vector<16xi32>
  // CHECK: return %[[R]] : vector<16xi32>
  return %0 : vector<16xi32>
}
}

// -----

// CHECK-LABEL: @shuffle_two_operands_nocast
// CHECK-SAME: %[[LHS:.*]]: vector<16xi32>,
// CHECK-SAME: %[[RHS:.*]]: vector<16xi32>

#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {
func.func @shuffle_two_operands_nocast(%lhs : vector<16xi32>,
                                       %rhs : vector<16xi32>)
                -> vector<16xi32> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK: %[[R:.*]] = "xllvm.intr.aie2.vshuffle"(%[[LHS]], %[[RHS]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  %0 = aievec.shuffle %lhs, %rhs [t32_4x8_lo] : vector<16xi32>
  // CHECK: return %[[R]] : vector<16xi32>
  return %0 : vector<16xi32>
}
}

// -----

// CHECK-LABEL: @shuffle_single_operand_cast
// CHECK-SAME: %[[V:.*]]: vector<32xbf16>

#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {
func.func @shuffle_single_operand_cast(%lhs : vector<32xbf16>)
                -> vector<32xbf16> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(42 : i32) : i32
  // CHECK: %[[RHS:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
  // CHECK: %[[LHS:.]] = llvm.bitcast %[[V]] : vector<32xbf16> to vector<16xi32>
  // CHECK: %[[S:.*]] = "xllvm.intr.aie2.vshuffle"(%[[LHS]], %[[RHS]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  // CHECK: %[[R:.]] = llvm.bitcast %[[S]] : vector<16xi32> to vector<32xbf16>
  %0 = aievec.shuffle %lhs [t16_8x2] : vector<32xbf16>
  // CHECK: return %[[R]] : vector<32xbf16>
  return %0 : vector<32xbf16>
}
}

// -----

// CHECK-LABEL: @shuffle_two_operands_cast
// CHECK-SAME: %[[LV:.*]]: vector<32xbf16>,
// CHECK-SAME: %[[RV:.*]]: vector<32xbf16>

#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {
func.func @shuffle_two_operands_cast(%lhs : vector<32xbf16>,
                                     %rhs : vector<32xbf16>)
                -> vector<32xbf16> {
  // CHECK: %[[M:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK: %[[L:.*]] = llvm.bitcast %[[LV]] : vector<32xbf16> to vector<16xi32>
  // CHECK: %[[R:.*]] = llvm.bitcast %[[RV]] : vector<32xbf16> to vector<16xi32>
  // CHECK: %[[S:.*]] = "xllvm.intr.aie2.vshuffle"(%[[L]], %[[R]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  // CHECK: %[[R:.]] = llvm.bitcast %[[S]] : vector<16xi32> to vector<32xbf16>
  %0 = aievec.shuffle %lhs, %rhs [t16_16x4_lo] : vector<32xbf16>
  // CHECK: return %[[R]] : vector<32xbf16>
  return %0 : vector<32xbf16>
}
}

// -----

// AIE2P has its own `vshuffle` intrinsic, with the same (lhs, rhs, mode) shape
// as AIE2's. It has no undef intrinsic though, so a single-operand mode gets a
// plain LLVM undef for the operand the hardware ignores.
//
// This is what an int8 `batch_matmul_transpose_b` needs: `t8_8x8` transposes the
// 8x8 int8 tile the contraction's B operand is read as.
// CHECK-LABEL: @shuffle_t8_8x8_aie2p
// CHECK-SAME: %[[LHS:.*]]: vector<64xi8>

#npu4 = #hal.executable.target<"npu4", "npu4", {target_device = "npu4"}>
module attributes {hal.executable.target = #npu4} {
func.func @shuffle_t8_8x8_aie2p(%lhs : vector<64xi8>) -> vector<64xi8> {
  // CHECK-DAG: %[[M:.*]] = llvm.mlir.constant(35 : i32) : i32
  // CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : vector<16xi32>
  // CHECK: %[[C:.*]] = llvm.bitcast %[[LHS]] : vector<64xi8> to vector<16xi32>
  // CHECK: %[[R:.*]] = "xllvm.intr.aie2p.vshuffle"(%[[C]], %[[UNDEF]], %[[M]]) :
  // CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
  // CHECK: %[[B:.*]] = llvm.bitcast %[[R]] : vector<16xi32> to vector<64xi8>
  %0 = aievec.shuffle %lhs [t8_8x8] : vector<64xi8>
  // CHECK: return %[[B]] : vector<64xi8>
  return %0 : vector<64xi8>
}
}

// -----

// Two-operand modes pass both through on AIE2P as well.
// CHECK-LABEL: @shuffle_two_operands_aie2p
#npu4 = #hal.executable.target<"npu4", "npu4", {target_device = "npu4"}>
module attributes {hal.executable.target = #npu4} {
func.func @shuffle_two_operands_aie2p(%lhs : vector<16xi32>,
                                      %rhs : vector<16xi32>)
                -> vector<16xi32> {
  // CHECK-NOT: llvm.mlir.undef
  // CHECK: "xllvm.intr.aie2p.vshuffle"
  %0 = aievec.shuffle %lhs, %rhs [t32_4x8_lo] : vector<16xi32>
  return %0 : vector<16xi32>
}
}
