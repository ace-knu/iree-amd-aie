// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-integer-requantization))' --split-input-file %s | FileCheck %s

// The int8 requantization tail of a fused matmul: scale by a pair of constants,
// round half to even, clamp, narrow. aie2p has no scalar float arithmetic, so
// all of this has to become integer math.
// CHECK-LABEL: func @requant_i32_to_i8
//   CHECK-NOT:   arith.sitofp
//   CHECK-NOT:   arith.mulf
//   CHECK-NOT:   arith.divf
//   CHECK-NOT:   math.roundeven
//   CHECK-NOT:   arith.maximumf
//   CHECK-NOT:   arith.minimumf
//   CHECK-NOT:   arith.fptosi
//       CHECK:   arith.extsi %{{.*}} : i32 to i64
//       CHECK:   arith.muli
//       CHECK:   arith.shrsi
//       CHECK:   arith.andi
//   CHECK-DAG:   arith.cmpi sgt
//   CHECK-DAG:   arith.cmpi eq
//       CHECK:   arith.select
//   CHECK-DAG:   arith.maxsi
//   CHECK-DAG:   arith.minsi
//       CHECK:   arith.trunci {{.*}} : i64 to i8
func.func @requant_i32_to_i8(%acc: i32) -> i8 {
  %s1 = arith.constant 2.280560e-03 : f32
  %s2 = arith.constant 0.592755735 : f32
  %zp = arith.constant 0.000000e+00 : f32
  %lo = arith.constant -1.280000e+02 : f32
  %hi = arith.constant 1.270000e+02 : f32
  %0 = arith.sitofp %acc : i32 to f32
  %1 = arith.mulf %0, %s1 : f32
  %2 = arith.divf %1, %s2 : f32
  %3 = math.roundeven %2 : f32
  %4 = arith.addf %3, %zp : f32
  %5 = arith.maximumf %4, %lo : f32
  %6 = arith.minimumf %5, %hi : f32
  %7 = arith.fptosi %6 : f32 to i8
  return %7 : i8
}

// -----

// A single divide, with no multiply, is the same shape.
// CHECK-LABEL: func @requant_divide_only
//   CHECK-NOT:   math.roundeven
//       CHECK:   arith.muli
func.func @requant_divide_only(%acc: i32) -> i8 {
  %s2 = arith.constant 0.0237346236 : f32
  %lo = arith.constant -1.280000e+02 : f32
  %hi = arith.constant 1.270000e+02 : f32
  %0 = arith.sitofp %acc : i32 to f32
  %1 = arith.divf %0, %s2 : f32
  %2 = math.roundeven %1 : f32
  %3 = arith.maximumf %2, %lo : f32
  %4 = arith.minimumf %3, %hi : f32
  %5 = arith.fptosi %4 : f32 to i8
  return %5 : i8
}

// -----

// A tail whose scale is not a compile-time constant has to be left alone.
// CHECK-LABEL: func @requant_dynamic_scale
//       CHECK:   math.roundeven
//       CHECK:   arith.fptosi
func.func @requant_dynamic_scale(%acc: i32, %s2: f32) -> i8 {
  %lo = arith.constant -1.280000e+02 : f32
  %hi = arith.constant 1.270000e+02 : f32
  %0 = arith.sitofp %acc : i32 to f32
  %1 = arith.divf %0, %s2 : f32
  %2 = math.roundeven %1 : f32
  %3 = arith.maximumf %2, %lo : f32
  %4 = arith.minimumf %3, %hi : f32
  %5 = arith.fptosi %4 : f32 to i8
  return %5 : i8
}

// -----

// A float tail that does not start at an integer accumulator (the input is a
// real f32 tensor) cannot be integerized either.
// CHECK-LABEL: func @requant_float_rooted
//       CHECK:   math.roundeven
//       CHECK:   arith.fptosi
func.func @requant_float_rooted(%x: f32) -> i8 {
  %s2 = arith.constant 0.0237346236 : f32
  %lo = arith.constant -1.280000e+02 : f32
  %hi = arith.constant 1.270000e+02 : f32
  %1 = arith.divf %x, %s2 : f32
  %2 = math.roundeven %1 : f32
  %3 = arith.maximumf %2, %lo : f32
  %4 = arith.minimumf %3, %hi : f32
  %5 = arith.fptosi %4 : f32 to i8
  return %5 : i8
}

// -----

// A clamp range far wider than a quantization tail's is declined: the pass only
// rewrites what it can check, and checking this many steps is not worth it.
// CHECK-LABEL: func @requant_implausible_clamp_range
//       CHECK:   math.roundeven
//       CHECK:   arith.fptosi
func.func @requant_implausible_clamp_range(%acc: i32) -> i32 {
  %s2 = arith.constant 5.000000e-01 : f32
  %lo = arith.constant -1.000000e+06 : f32
  %hi = arith.constant 1.000000e+06 : f32
  %0 = arith.sitofp %acc : i32 to f32
  %1 = arith.divf %0, %s2 : f32
  %2 = math.roundeven %1 : f32
  %3 = arith.maximumf %2, %lo : f32
  %4 = arith.minimumf %3, %hi : f32
  %5 = arith.fptosi %4 : f32 to i32
  return %5 : i32
}

// -----

// A negative scale would break the monotonicity the exactness check relies on,
// so it is declined too.
// CHECK-LABEL: func @requant_negative_scale
//       CHECK:   math.roundeven
//       CHECK:   arith.fptosi
func.func @requant_negative_scale(%acc: i32) -> i8 {
  %s2 = arith.constant -0.0237346236 : f32
  %lo = arith.constant -1.280000e+02 : f32
  %hi = arith.constant 1.270000e+02 : f32
  %0 = arith.sitofp %acc : i32 to f32
  %1 = arith.divf %0, %s2 : f32
  %2 = math.roundeven %1 : f32
  %3 = arith.maximumf %2, %lo : f32
  %4 = arith.minimumf %3, %hi : f32
  %5 = arith.fptosi %4 : f32 to i8
  return %5 : i8
}
