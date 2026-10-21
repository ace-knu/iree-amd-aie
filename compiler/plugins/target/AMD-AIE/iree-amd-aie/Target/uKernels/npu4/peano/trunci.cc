// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

template <int M, int N, int r, int u>
void trunci_vectorized(v32int32 *__restrict in, int64_t offsetIn, int64_t shift,
                       v32int8 *__restrict out, int64_t offsetOut) {
  static_assert(M * N / r / u > 0);
#pragma clang loop unroll_count(M *N / r / u)
  for (unsigned i = 0; i < M * N / r; i++) {
    out[offsetOut + i] = ssrs((v32acc32)in[offsetIn + i], shift, 0);
  }
}

// clang-format off
extern "C" {

// A vectorized truncation function on a buffer of size M * N with vectorization size
// 'r' and unrolling factor 'u'. The unrolling factor is typically found experimentally.
#define trunci_named_c_func(name, ctype_in, ctype_out, M, N, r, u)                                \
  void name(                                                                                      \
      ctype_in *in, int64_t offsetIn, int64_t shift, ctype_out *out, int64_t offsetOut) {         \
    trunci_vectorized<M, N, r, u>(in, offsetIn, shift, out, offsetOut);                              \
  }

// The compiler passes the functions a device calls as `UK_INSTANCES`, a list
// of `UK_INSTANCE(name, in, out, M, N)` with MLIR element type names.
#define trunci_instance_i32_i8(name, M, N) trunci_named_c_func(name, v32int32, v32int8, M, N, 32, 8)
#define UK_INSTANCE(name, in, out, M, N) trunci_instance_##in##_##out(name, M, N)

#ifndef UK_INSTANCES
#error "UK_INSTANCES must list the functions to define"
#endif
UK_INSTANCES

}  // extern "C"
// clang-format on
