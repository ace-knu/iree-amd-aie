// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The bf16 GEMV ukernel: c[0 .. n) += A * b over a row-major A (n x k) and a
// vector b (k). Taken from mlir-aie's aie_kernels/linalg/mv_bf16.cc (943cbc4),
// its row-major vectorized kernel, with the changes the GEMV pipeline needs:
// c is f32 and the products are added to it (the pipeline streams K through
// L1 and fills c once), k comes from the instance instead of DIM_K, and the
// entry points take the (pointer, offset) operands of a ukernel call.

#include <stdint.h>

#include "aie_kernels/aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif

// v and w hold rows of L partial sums each. Adds lane i + L / 2 onto lane i of
// every row, the pairing reduce_add uses, and returns v's rows then w's.
template <unsigned L, unsigned N>
static inline aie::vector<float, N> fold2(aie::vector<float, N> v,
                                          aie::vector<float, N> w) {
  auto [lo, hi] = aie::interleave_unzip(v, w, L / 2);
  return aie::add(lo, hi);
}

// Sums each row of L lanes in v into one lane. A 16-lane vector folds against
// itself, since narrower unzips lower to per-lane extract and insert.
template <unsigned L, unsigned N>
static inline aie::vector<float, 16> fold_rows(aie::vector<float, N> v) {
  if constexpr (L == 1)
    return v.template grow_extract<16>(0);
  else if constexpr (N == 16)
    return fold_rows<L / 2, 16>(fold2<L>(v, v).template extract<16>(0));
  else
    return fold_rows<L / 2, N / 2>(
        fold2<L>(v.template extract<N / 2>(0), v.template extract<N / 2>(1)));
}

// Packs four rows' accumulators into one, quarter q holding row q's 16 partial
// sums, halving each row the way reduce_add does. Down to 16 lanes a halving
// is a permutation of accumulator quarters, so it needs no shuffle.
template <uint32_t r>
static inline aie::accum<accfloat, 64>
pack_rows4(aie::accum<accfloat, r> a0, aie::accum<accfloat, r> a1,
           aie::accum<accfloat, r> a2, aie::accum<accfloat, r> a3) {
  if constexpr (r > 64) {
    auto half = [](aie::accum<accfloat, r> x) {
      return aie::add(x.template extract<r / 2>(0),
                      x.template extract<r / 2>(1));
    };
    return pack_rows4<r / 2>(half(a0), half(a1), half(a2), half(a3));
  } else if constexpr (r == 16) {
    return aie::concat(a0, a1, a2, a3);
  } else {
    static_assert(r == 32 || r == 64);
    // Two rows of 32 lanes per accumulator.
    aie::accum<accfloat, 64> t01, t23;
    if constexpr (r == 64) {
      auto pair = [](aie::accum<accfloat, 64> x, aie::accum<accfloat, 64> y) {
        return aie::add(
            aie::concat(x.template extract<32>(0), y.template extract<32>(0)),
            aie::concat(x.template extract<32>(1), y.template extract<32>(1)));
      };
      t01 = pair(a0, a1);
      t23 = pair(a2, a3);
    } else {
      t01 = aie::concat(a0, a1);
      t23 = aie::concat(a2, a3);
    }
    return aie::add(
        aie::concat(t01.template extract<16>(0), t01.template extract<16>(2),
                    t23.template extract<16>(0), t23.template extract<16>(2)),
        aie::concat(t01.template extract<16>(1), t01.template extract<16>(3),
                    t23.template extract<16>(1), t23.template extract<16>(3)));
  }
}

// Four rows of A times b, packed by pack_rows4. Each row walks its own cursor:
// one pointer stepped through all four rows chains every load on the previous
// load's post-increment. The first chunk multiplies instead of accumulating
// onto zeros, which would be reloaded from the stack for every group. Short
// rows unroll fully, so a whole group is one block the scheduler can overlap
// with its neighbor. Not inlined, it returns its accumulator through the stack.
template <uint32_t r, uint32_t k>
__attribute__((always_inline)) static inline aie::accum<accfloat, 64>
mac_rows4(const bfloat16 *__restrict a, const bfloat16 *__restrict b) {
  constexpr uint32_t chunks = k / r;
  const bfloat16 *__restrict a0 = a;
  const bfloat16 *__restrict a1 = a + k;
  const bfloat16 *__restrict a2 = a + 2 * k;
  const bfloat16 *__restrict a3 = a + 3 * k;
  const bfloat16 *__restrict pb = b;
  aie::vector<bfloat16, r> b_0 = aie::load_v<r>(pb);
  pb += r;
  aie::accum<accfloat, r> acc0 = aie::mul(aie::load_v<r>(a0), b_0);
  a0 += r;
  aie::accum<accfloat, r> acc1 = aie::mul(aie::load_v<r>(a1), b_0);
  a1 += r;
  aie::accum<accfloat, r> acc2 = aie::mul(aie::load_v<r>(a2), b_0);
  a2 += r;
  aie::accum<accfloat, r> acc3 = aie::mul(aie::load_v<r>(a3), b_0);
  a3 += r;
  auto step = [&]() {
    aie::vector<bfloat16, r> b_vec = aie::load_v<r>(pb);
    pb += r;
    acc0 = aie::mac(acc0, aie::load_v<r>(a0), b_vec);
    a0 += r;
    acc1 = aie::mac(acc1, aie::load_v<r>(a1), b_vec);
    a1 += r;
    acc2 = aie::mac(acc2, aie::load_v<r>(a2), b_vec);
    a2 += r;
    acc3 = aie::mac(acc3, aie::load_v<r>(a3), b_vec);
    a3 += r;
  };
  if constexpr (chunks <= 4) {
    AIE_LOOP_UNROLL_FULL
    for (uint32_t i = 1; i < chunks; i++)
      step();
  } else {
    for (uint32_t i = 1; i < chunks; i++)
      step();
  }
  return pack_rows4<r>(acc0, acc1, acc2, acc3);
}

/*
Matrix-vector multiplication kernel

 - m: Number of output rows == number of rows in the input matrix
 - k: Number of columns in the input matrix == length of the input vector
 - a: Pointer to the input matrix, stored in row-major order
 - b: Pointer to the input vector
 - c: Pointer to the output vector, which the products are added to
 - r: Vector size; data from the matrix and vector will be loaded in and
processed in chunks of this size
*/
template <uint32_t r, uint32_t k>
void matvec_vectorized(uint32_t m, const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b, float *__restrict c) {
  static_assert(k % r == 0);
  static_assert(k >= r);
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr uint32_t chunks = k / r;

  // Four rows at a time. One b chunk then feeds four macs instead of one, and
  // one transposed tree sums all four rows where reduce_add_v runs four trees
  // side by side. The tree is a latency chain, so each group's finishes in the
  // next iteration, under that group's macs. Behind a mac loop only the store
  // waits: the packed accumulator would spill across the loop. On AIE2 four
  // 64-lane accumulators fill the accumulator file, so the packed one would
  // spill there beside the next group's too.
#if AIE_TUNED_AIE2P
  constexpr bool fold_late = chunks <= 4;
#else
  constexpr bool fold_late = chunks <= 4 && r < 64;
#endif
  auto defer = [](aie::accum<accfloat, 64> u) {
    if constexpr (fold_late)
      return u;
    else
      return fold_rows<16>(u.template to_vector<float>());
  };
  auto store4 = [](float *__restrict out, auto deferred) {
    aie::vector<float, 16> sums;
    if constexpr (fold_late)
      sums = fold_rows<16>(deferred.template to_vector<float>());
    else
      sums = deferred;
    out[0] += sums[0];
    out[1] += sums[1];
    out[2] += sums[2];
    out[3] += sums[3];
  };
  uint32_t groups = m / 4;
  if (groups > 0) {
    auto prev = defer(mac_rows4<r, k>(a, b));
    for (uint32_t g = 1; g < groups; g++, c += 4) {
      a += 4 * k;
      auto next = defer(mac_rows4<r, k>(a, b));
      store4(c, prev);
      prev = next;
    }
    store4(c, prev);
    a += 4 * k;
    c += 4;
  }

  // m need not be a multiple of four.
  for (uint32_t row = groups * 4; row < m; row++, c++) {
#if AIE_TUNED_AIE2
    // mac_elem_16_2 sums lanes i and i + 16, so each accumulator folds 32
    // products per chunk.
    if constexpr (r % 64 == 0 && chunks > 1) {
      constexpr uint32_t P = r / 32;
      v16accfloat acc[P];
      AIE_LOOP_UNROLL_FULL
      for (uint32_t q = 0; q < P; q++)
        acc[q] = mul_elem_16_2(aie::load_v<32>(a + 32 * q).to_native(),
                               aie::load_v<32>(b + 32 * q).to_native());
      a += r;
      const bfloat16 *__restrict pb = b + r;
      AIE_LOOP_MIN_ITERATION_COUNT(chunks - 1)
      for (uint32_t i = 1; i < chunks; i++, a += r, pb += r) {
        AIE_LOOP_UNROLL_FULL
        for (uint32_t q = 0; q < P; q++)
          acc[q] =
              mac_elem_16_2(aie::load_v<32>(a + 32 * q).to_native(),
                            aie::load_v<32>(pb + 32 * q).to_native(), acc[q]);
      }
      aie::accum<accfloat, 16> sum(acc[0]);
      AIE_LOOP_UNROLL_FULL
      for (uint32_t q = 1; q < P; q++)
        sum = aie::add(sum, aie::accum<accfloat, 16>(acc[q]));
      *c += aie::reduce_add(sum.to_vector<float>());
      continue;
    }
#endif
    aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
    AIE_LOOP_MIN_ITERATION_COUNT(chunks)
    for (uint32_t i = 0; i < chunks; i++, a += r)
      acc = aie::mac(acc, aie::load_v<r>(a), aie::load_v<r>(b + i * r));
    *c += aie::reduce_add(acc.template to_vector<float>());
  }
  ::aie::set_rounding(saved_rounding);
}

// clang-format off
extern "C" {

// The GEMV pipeline's tile `(1 x K) x (N x K)^T -> (1 x N)`: the matmul's LHS
// is the vector b, its transposed RHS the matrix A.
#define matvec_instance_bf16_bf16_f32(name, N, K)                               \
  void name(const bfloat16 *b_in, unsigned offsetB, const bfloat16 *a_in,       \
            unsigned offsetA, float *c_out, unsigned offsetC) {                 \
    matvec_vectorized<VEC_SIZE, K>(N, a_in + offsetA, b_in + offsetB,           \
                                   c_out + offsetC);                            \
  }

// The compiler passes the functions a device calls as `UK_INSTANCES`, a list
// of `UK_INSTANCE(name, lhs, rhs, acc, N, K)` with MLIR element type names.
#define UK_INSTANCE(name, lhs, rhs, acc, N, K)                                  \
  matvec_instance_##lhs##_##rhs##_##acc(name, N, K)

#ifndef UK_INSTANCES
#error "UK_INSTANCES must list the functions to define"
#endif
UK_INSTANCES

}  // extern "C"
// clang-format on
