// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Zeroes an M x N buffer with mlir-aie's aie_kernels/common/zero.h (943cbc4),
// without its event markers, behind the (pointer, offset) operands of a
// ukernel call. A whole number of native vectors keeps every tile of the
// buffer vector aligned and takes the compile-time kernel; any other size
// takes the runtime-length one, which aligns its own stores.

#include <stdint.h>

#include "aie_kernels/common/zero.h"

#include <aie_api/aie.hpp>

template <typename T, int M, int N>
static inline void zero_fill(T *__restrict c) {
  if constexpr (M * N % aie::native_vector_length_v<T> == 0)
    zero_vectorized<T, M, N, false>(c);
  else
    zero_vectorized(c, M * N);
}

// clang-format off
extern "C" {

#define zero_fill_named_c_func(name, ctype_out, M, N)          \
  void name(ctype_out *c_out, unsigned offsetC) {             \
    zero_fill<ctype_out, M, N>(c_out + offsetC);              \
  }

// The compiler passes the functions a device calls as `UK_INSTANCES`, a list
// of `UK_INSTANCE(name, type, M, N)` with MLIR element type names.
#define zero_fill_instance_i32(name, M, N) zero_fill_named_c_func(name, int32_t, M, N)
#define zero_fill_instance_f32(name, M, N) zero_fill_named_c_func(name, float, M, N)
#define UK_INSTANCE(name, type, M, N) zero_fill_instance_##type(name, M, N)

#ifndef UK_INSTANCES
#error "UK_INSTANCES must list the functions to define"
#endif
UK_INSTANCES

}  // extern "C"
