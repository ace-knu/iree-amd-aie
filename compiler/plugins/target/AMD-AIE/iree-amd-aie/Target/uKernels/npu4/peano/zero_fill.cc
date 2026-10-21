// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

template <int M, int N, int r, int u>
void zero_fill_vectorized(v16int32 *__restrict pC, unsigned offsetC) {
  static_assert(M * N / r / u > 0);
  v16int32 zeros = broadcast_zero_to_v16int32();
#pragma clang loop unroll_count(M *N / r / u)
  for (unsigned i = offsetC / r; i < offsetC / r + M * N / r; i++) {
    pC[i] = zeros;
  }
}

template <int M, int N, int r, int u>
void zero_fill_vectorized(v16float *__restrict pC, unsigned offsetC) {
  static_assert(M * N / r / u > 0);
  v16float zeros = broadcast_zero_to_v16float();
#pragma clang loop unroll_count(M *N / r / u)
  for (unsigned i = offsetC / r; i < offsetC / r + M * N / r; i++) {
    pC[i] = zeros;
  }
}

// clang-format off
extern "C" {

// A vectorized zeroization function on a buffer of size M * N with vectorization size
// 'r' and unrolling factor 'u'. The unrolling factor is typically found experimentally.
#define zero_fill_named_c_func(name, ctype_out, M, N, r, u)  \
  void name(ctype_out *c_out, unsigned offsetC) {             \
    zero_fill_vectorized<M, N, r, u>(c_out, offsetC);         \
  }

// The compiler passes the functions a device calls as `UK_INSTANCES`, a list
// of `UK_INSTANCE(name, type, M, N)` with MLIR element type names.
#define zero_fill_instance_i32(name, M, N) zero_fill_named_c_func(name, v16int32, M, N, 16, 8)
#define zero_fill_instance_f32(name, M, N) zero_fill_named_c_func(name, v16float, M, N, 16, 8)
#define UK_INSTANCE(name, type, M, N) zero_fill_instance_##type(name, M, N)

#ifndef UK_INSTANCES
#error "UK_INSTANCES must list the functions to define"
#endif
UK_INSTANCES

}  // extern "C"
// clang-format on
