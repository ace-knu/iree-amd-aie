// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-expand-roundeven))' --split-input-file %s | FileCheck %s

// peano's aie2p backend legalizes none of the float rounding intrinsics, and
// upstream's expansion goes through copysign, which it does not legalize
// either. What is left is fptosi/sitofp, fabs, fcmp and select.
//
// The expansion does not reproduce NaN behaviour, so it only applies where the
// value is on its way to a narrowing integer cast, which is undefined for NaN
// in the float form too.
// CHECK-LABEL: func @roundeven_in_a_quant_tail
//   CHECK-NOT:   math.roundeven
//   CHECK-NOT:   math.copysign
//   CHECK-NOT:   math.floor
//   CHECK-DAG:   arith.fptosi {{.*}} : f32 to i32
//   CHECK-DAG:   arith.sitofp {{.*}} : i32 to f32
//   CHECK-DAG:   arith.subf
//   CHECK-DAG:   math.absf
//   CHECK-DAG:   arith.cmpf ogt
//   CHECK-DAG:   arith.cmpf oeq
//   CHECK-DAG:   arith.cmpi ne
//   CHECK-DAG:   arith.select
func.func @roundeven_in_a_quant_tail(%x: f32) -> i8 {
  %lo = arith.constant -1.280000e+02 : f32
  %hi = arith.constant 1.270000e+02 : f32
  %0 = math.roundeven %x : f32
  %1 = arith.maximumf %0, %lo : f32
  %2 = arith.minimumf %1, %hi : f32
  %3 = arith.fptosi %2 : f32 to i8
  return %3 : i8
}

// -----

// A roundeven whose result stays a float keeps its exact NaN behaviour.
// CHECK-LABEL: func @roundeven_result_escapes_as_float
//       CHECK:   math.roundeven
func.func @roundeven_result_escapes_as_float(%x: f32) -> f32 {
  %0 = math.roundeven %x : f32
  return %0 : f32
}

// -----

// Nor is it enough to merely be used by float arithmetic on the way out.
// CHECK-LABEL: func @roundeven_feeding_float_arithmetic
//       CHECK:   math.roundeven
func.func @roundeven_feeding_float_arithmetic(%x: f32) -> f32 {
  %0 = math.roundeven %x : f32
  %1 = arith.mulf %0, %x : f32
  return %1 : f32
}

// -----

// f64 is not covered, and must be left for someone else to handle.
// CHECK-LABEL: func @roundeven_f64
//       CHECK:   math.roundeven
func.func @roundeven_f64(%x: f64) -> i8 {
  %lo = arith.constant -1.280000e+02 : f64
  %hi = arith.constant 1.270000e+02 : f64
  %0 = math.roundeven %x : f64
  %1 = arith.maximumf %0, %lo : f64
  %2 = arith.minimumf %1, %hi : f64
  %3 = arith.fptosi %2 : f64 to i8
  return %3 : i8
}
