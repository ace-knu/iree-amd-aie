// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The matmul ukernel: C += A * B over 8x8x8 micro-tiles, bf16 x bf16 -> f32
// (through bfp16) and i8 x i8 -> i32. Taken from mlir-aie's
// aie_kernels/linalg/mm_aie2p.h (943cbc4), its matmul_vectorized_2x2_mmul,
// with the changes the iree L1 layout needs. Each micro-tile is row-major as
// upstream, but the pack places the tiles in a different order:
// - A tiles are ordered [k][m], not [m][k].
// - C tiles are ordered [n][m] (upstream's C_COL_MAJ order), each tile still
//   row-major, so no tile is transposed.
// - B tiles are ordered [n][k] holding k x n tiles without a transposed B
//   (upstream's B_COL_MAJ order, untransposed tiles), and [k][n] holding n x k
//   tiles with one (upstream's row-major order, transposed tiles).
// The entry points take the (pointer, offset) operands of a ukernel call.

#include <stdint.h>
#include <type_traits>

#define AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 1

#include "aie_kernels/aie_arch.h"
#include "aie_kernels/aie_kernel_utils.h"

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t, bool TransposeB,
          bool unroll_k = false>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  // The bfp16 emulation converts every A and B tile from bf16 to bfp16 before
  // the MAC, and that conversion follows the core rounding mode. The default
  // mode is rounding_mode::floor, which rounds toward negative infinity, so
  // each converted element is biased low and the bias accumulates over the K
  // reduction rather than cancelling. Select round-to-nearest-even for the
  // duration of the kernel and restore the caller's mode before returning. Only
  // the bf16 shapes convert, so the integer shapes keep whatever mode the
  // caller set.
  constexpr bool emulated_bf16 = std::is_same_v<T_in, bfloat16>;
  aie::rounding_mode saved_rounding = aie::rounding_mode::floor;
  if constexpr (emulated_bf16)
    saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {

      T_out *__restrict pCj = pC + z * MMUL::size_C;
      // Peano does not strength-reduce the j-indexed tile addresses, so these
      // cursors step with j rather than rebuilding each address from j.
      const T_in *__restrict pAz = pA + z * MMUL::size_A;
      const T_in *__restrict pBj = pB;

      for (unsigned j = 0; j < colB; j += 2) {

        T_out *__restrict pC1 = pCj;
        T_out *__restrict pC2 = pCj + rowA * MMUL::size_C;
        pCj += 2 * rowA * MMUL::size_C;
        const T_in *__restrict pA1 = pAz;
        const T_in *__restrict pA2 = pAz + MMUL::size_A;
        const T_in *__restrict pB1 = pBj;
        const T_in *__restrict pB2;
        if constexpr (TransposeB) {
          pB2 = pBj + MMUL::size_B;
          pBj += 2 * MMUL::size_B;
        } else {
          pB2 = pBj + colA * MMUL::size_B;
          pBj += 2 * colA * MMUL::size_B;
        }

        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_A> A1;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        // Load partial results from C buffer for accumulation in-place. The
        // zero_fill function handles the zeroing of data when a new
        // accumulation is needed (after the 'K' reduction dimension)
        aie::vector<T_out, MMUL::size_C> acc_C00 =
            aie::load_v<MMUL::size_C>(pC1);
        aie::vector<T_out, MMUL::size_C> acc_C01 =
            aie::load_v<MMUL::size_C>(pC2);
        aie::vector<T_out, MMUL::size_C> acc_C10 =
            aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
        aie::vector<T_out, MMUL::size_C> acc_C11 =
            aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

        // Starting a short unrolled reduction with mul and adding the loaded
        // C tiles after it keeps them off the MAC chain. Longer reductions
        // spill more this way. Integer sums are exact in any order.
        constexpr bool add_c_last = unroll_k && colA <= 4 && AIE_TUNED_AIE2P &&
                                    std::is_integral_v<T_out>;

        MMUL C00, C01, C10, C11;
        if constexpr (!add_c_last) {
          C00 = MMUL(acc_C00);
          C01 = MMUL(acc_C01);
          C10 = MMUL(acc_C10);
          C11 = MMUL(acc_C11);
        }

        auto k_step = [&](bool first) {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += rowA * MMUL::size_A;
          A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += rowA * MMUL::size_A;
          if constexpr (TransposeB) {
            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
            pB1 += MMUL::size_B * colB;
            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
            pB2 += MMUL::size_B * colB;
          } else {
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B;
            B1 = aie::load_v<MMUL::size_B>(pB2);
            pB2 += MMUL::size_B;
          }

          if (add_c_last && first) {
            C00.mul(A0, B0);
            C01.mul(A0, B1);
            C10.mul(A1, B0);
            C11.mul(A1, B1);
          } else {
            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);
          }
        };

        // Peano only software-pipelines an innermost single-block loop, so
        // unrolling K lets 'j', with its C loads and stores, pipeline. The
        // caller enables it only where one mmul step is one instruction.
        // The unrolled reduction does not fit the register file: int8 ->
        // int32 spills 16 bytes of stack per unit of K. Rolling long
        // reductions back up is not a uniform win on hardware either.
        if constexpr (unroll_k) {
          AIE_LOOP_UNROLL_FULL
          for (unsigned i = 0; i < colA; ++i)
            k_step(i == 0);
        } else {
          for (unsigned i = 0; i < colA; ++i)
            k_step(false);
        }
        if constexpr (add_c_last) {
          C00 = MMUL(aie::add(C00.to_accum(), acc_C00));
          C01 = MMUL(aie::add(C01.to_accum(), acc_C01));
          C10 = MMUL(aie::add(C10.to_accum(), acc_C10));
          C11 = MMUL(aie::add(C11.to_accum(), acc_C11));
        }

        aie::store_v(pC1, C00.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C01.template to_vector<T_out>());
        pC2 += MMUL::size_C;
        aie::store_v(pC1, C10.template to_vector<T_out>());
        pC1 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<T_out>());
        pC2 += MMUL::size_C;
      }
    }

  if constexpr (emulated_bf16)
    aie::set_rounding(saved_rounding);
}

template <unsigned m, unsigned k, unsigned n, bool TransposeB>
static inline void matmul_vectorized_8x8x8_i8_i8_i32(const int8 *__restrict pA,
                                                     const int8 *__restrict pB,
                                                     int32 *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  // Upstream unrolls K below K = 416, but the unrolled reduction spills 16
  // bytes of stack per unit of K (1472 bytes at K = 128) past the 1024-byte
  // core stack iree assigns by default. Rolled, it needs 64 bytes and is no
  // slower on the npu4 benchmarks.
  constexpr bool unroll_k = false;

  return matmul_vectorized_2x2_mmul<int8, int32, (m / r), (k / s), (n / t), r,
                                    s, t, TransposeB, unroll_k>(pA, pB, pC);
}

// TODO: The goal is every element type combination the hardware supports, each
// with and without a transposed B. Only this template supports a transposed B
// for now. When a template is added or gains it, add it to the table in
// `hasUKernelTemplate` (Transforms/Utils/AMDAIEUtils.cpp).
template <unsigned m, unsigned k, unsigned n, bool TransposeB>
static inline void
matmul_vectorized_8x8x8_bf16_bf16_f32(const bfloat16 *__restrict pA,
                                      const bfloat16 *__restrict pB,
                                      float *__restrict pC) {
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<bfloat16, float, (m / r), (k / s),
                                    (n / t), r, s, t, TransposeB>(pA, pB, pC);
}

// clang-format off
extern "C" {

#define matmul_named_c_func(name, lhs_ctype_in, lhs_mlir_type_in,                                                \
                            rhs_ctype_in, rhs_mlir_type_in,                                                      \
                            acc_ctype_out, acc_mlir_type_out, M, N, K, r, s, t,                                  \
                            transpose_b)                                                                         \
  void name(                                                                                                     \
      lhs_ctype_in *a_in, unsigned offsetA, rhs_ctype_in *b_in, unsigned offsetB,                                \
      acc_ctype_out *c_out, unsigned offsetC) {                                                                  \
    matmul_vectorized_##r##x##s##x##t##_##lhs_mlir_type_in##_##rhs_mlir_type_in##_##acc_mlir_type_out<           \
        M, K, N, transpose_b>(a_in + offsetA, b_in + offsetB, c_out + offsetC);                                  \
  }

// The compiler passes the functions a device calls as `UK_INSTANCES`, a list
// of `UK_INSTANCE(name, lhs, rhs, acc, M, N, K, r, s, t, transpose_b)` with
// MLIR element type names.
#define UK_CTYPE_i8 int8
#define UK_CTYPE_i32 int32
#define UK_CTYPE_bf16 bfloat16
#define UK_CTYPE_f32 float
#define UK_INSTANCE(name, lhs, rhs, acc, M, N, K, r, s, t, transpose_b)        \
  matmul_named_c_func(name, UK_CTYPE_##lhs, lhs, UK_CTYPE_##rhs, rhs,           \
                      UK_CTYPE_##acc, acc, M, N, K, r, s, t, transpose_b)

#ifndef UK_INSTANCES
#error "UK_INSTANCES must list the functions to define"
#endif
UK_INSTANCES

}  // extern "C"
// clang-format on
