// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace {

constexpr int kVecLanes = 32;

// Largest row this kernel is instantiated for; bounds the on-core scratch.
constexpr int kMaxRowLanes = 2048;

// Horizontal max-reduce a v32bfloat16 to a single bf16.
INTRINSIC(bfloat16) reduce_max_v32bf16(v32bfloat16 v) {
  const v32bfloat16 z = broadcast_zero_to_v32bfloat16();
  v = max(v, shift(v, z, 16));
  v = max(v, shift(v, z, 8));
  v = max(v, shift(v, z, 4));
  v = max(v, shift(v, z, 2));
  v = max(v, shift(v, z, 1));
  return ext_elem(v, 0);
}

// Horizontal add-reduce a v32accfloat to a scalar float.
INTRINSIC(float) reduce_add_v32accf(v32accfloat acc) {
  const v16accfloat z = broadcast_zero_to_v16accfloat();
  // Fold 32 -> 16 lanes (lossless), then a log2(16) = 4-step shift/add
  // butterfly down to lane 0, all in accfloat.
  v16accfloat r = add(extract_v16accfloat(acc, 0), extract_v16accfloat(acc, 1));
  r = add(r, shift(r, z, 8));
  r = add(r, shift(r, z, 4));
  r = add(r, shift(r, z, 2));
  r = add(r, shift(r, z, 1));
  // Lane 0 holds the full f32 sum; narrow just that scalar to read it out.
  return (float)ext_elem(set_v32bfloat16(0, to_v16bfloat16(r)), 0);
}

// Three-pass softmax over a contiguous bf16 row of length `n`, where
// `n % 32 == 0`. Math:
//   pass 1: m = max_i(x_i) * log2e
//   pass 2: e_i = 2^(x_i * log2e - m) ; sum = sum_i e_i ; out[i] = e_i
//   pass 3: out[i] = e_i * (1 / sum)
inline void softmax_bf16_impl(bfloat16 *restrict in, bfloat16 *restrict out,
                              int32_t n) {
  event0();

  const int32_t iters = n / kVecLanes;

  // bf16-representable value of log2(e). 1.4453125 = 0x3FB9 in bf16, same
  // constant mlir-aie's aie_kernels/aie2p/softmax.cc uses.
  const v32bfloat16 log2e_v = broadcast_to_v32bfloat16((bfloat16)1.4453125f);

  // ---- Pass 1: running max of the raw row -----------------------------
  // 2^(x*log2e) is monotonic in x and log2e > 0, so
  //   max_i(x_i * log2e) = log2e * max_i(x_i).
  // Take the max in the raw x domain (no per-element multiply / demote)
  // and fold the log2e factor into the single scalar multiply below.
  v32bfloat16 max_v = broadcast_to_v32bfloat16((bfloat16)-32768.0f);
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)in;
    for (int32_t i = 0; i < iters; ++i) {
      max_v = max(max_v, *pIn++);
    }
  }
  // m = max_i(x_i) * log2e (one scalar multiply, kept in f32). Pre-negate
  // and broadcast so pass 2 can fuse the scale-and-shift into one MAC.
  const float neg_m = -((float)reduce_max_v32bf16(max_v) * 1.4453125f);
  const v16accfloat neg_m16 = broadcast_to_v16accfloat(neg_m);
  const v32accfloat neg_m_acc = concat(neg_m16, neg_m16);

  // ---- Pass 2: e_i = exp2(x_i * log2e - m); accumulate sum; store e_i --
  // shifted = x_i * log2e - m in a single fused multiply-accumulate:
  // mac_elem_32 computes neg_m_acc + x * log2e.
  v32accfloat sum_acc = ups(broadcast_zero_to_v32bfloat16());
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)in;
    v32bfloat16 *restrict pOut = (v32bfloat16 *)out;
    for (int32_t i = 0; i < iters; ++i) {
      v32bfloat16 x = *pIn++;
      v32accfloat shifted = mac_elem_32(x, /*sgn_x=*/1, log2e_v,
                                        /*sgn_y=*/1, neg_m_acc);
      v32bfloat16 e = exp2(shifted);
      *pOut++ = e;
      sum_acc = add(sum_acc, ups(e));
    }
  }

  // ---- Pass 3: out_i = e_i * (1 / sum) --------------------------------
  bfloat16 inv_sum = (bfloat16)inv(reduce_add_v32accf(sum_acc));
  v32bfloat16 inv_sum_v = broadcast_to_v32bfloat16(inv_sum);
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)out;
    v32bfloat16 *restrict pOut = (v32bfloat16 *)out;
    for (int32_t i = 0; i < iters; ++i) {
      v32bfloat16 e = *pIn++;
      v32accfloat scaled = mul_elem_32(e, /*sgn_x=*/1, inv_sum_v,
                                       /*sgn_y=*/1);
      *pOut++ = to_v32bfloat16(scaled);
    }
  }

  event1();
}

}  // namespace

// Quantized softmax over a contiguous int8 row of length `n` (`n % 32 == 0`),
// producing int8 again:
//
//   out = clamp(round(softmax(in * s_in) / s_out), -128, 127)
//
// The caller passes `log2eScaled = s_in * log2(e)` and `outScale = 1 / s_out`,
// which is all the dequantize/requantize arithmetic the surrounding graph would
// otherwise do element by element around this kernel. Folding it in costs
// nothing: the kernel already multiplies by log2(e) on the way into `exp2`, and
// already multiplies by `1 / sum` on the way out.
//
// That matters more than it looks: aie2p has no scalar float arithmetic, so a
// dequantize/requantize left outside the kernel becomes soft-float libcalls
// (`__mulsf3`, `__divsf3`, ...) that cost several KB of a core's 16KB program
// memory. Here the only per-element conversions are `fix2float` and a vector
// `bfloat16_to_int`, both single instructions.
//
// The int8 inputs are whole numbers of magnitude <= 128, which bf16 represents
// exactly, so subtracting the row maximum is exact and `s_in` only ever enters
// through the constant folded into log2(e).
inline void softmax_i8_impl(int8 *restrict in, int8 *restrict out, int32_t n,
                            float log2eScaled, float outScale) {
  event0();

  const int32_t iters = n / kVecLanes;

  // Widen the row to bf16 once. There is no vector int-to-float conversion on
  // aie2p, but the scalar one is a single `fx2flt`.
  bfloat16 x[kMaxRowLanes];
  for (int32_t i = 0; i < n; ++i) x[i] = (bfloat16)fix2float((int32_t)in[i], 0);

  const v32bfloat16 log2e_v = broadcast_to_v32bfloat16((bfloat16)log2eScaled);
  const v32bfloat16 outScale_v = broadcast_to_v32bfloat16((bfloat16)outScale);

  // ---- Pass 1: row maximum, in the exact integer domain -----------------
  v32bfloat16 max_v = broadcast_to_v32bfloat16((bfloat16)-32768.0f);
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)x;
    for (int32_t i = 0; i < iters; ++i) max_v = max(max_v, *pIn++);
  }
  const v32bfloat16 m_v =
      broadcast_to_v32bfloat16(reduce_max_v32bf16(max_v));
  // -(m * log2e * s_in), built with vector ops so that no scalar float
  // arithmetic (which aie2p lacks) is needed per row.
  const v32accfloat zero_acc = ups(broadcast_zero_to_v32bfloat16());
  const v32accfloat neg_m_acc =
      sub(zero_acc, mul_elem_32(m_v, /*sgn_x=*/1, log2e_v, /*sgn_y=*/1));

  // ---- Pass 2: e = exp2((x - m) * log2e * s_in); sum -------------------
  bfloat16 e_row[kMaxRowLanes];
  v32accfloat sum_acc = ups(broadcast_zero_to_v32bfloat16());
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)x;
    v32bfloat16 *restrict pOut = (v32bfloat16 *)e_row;
    for (int32_t i = 0; i < iters; ++i) {
      v32bfloat16 e = exp2(mac_elem_32(*pIn++, /*sgn_x=*/1, log2e_v,
                                       /*sgn_y=*/1, neg_m_acc));
      *pOut++ = e;
      sum_acc = add(sum_acc, ups(e));
    }
  }

  // ---- Pass 3: out = clamp(floor(e * (1/sum) * outScale + 0.5)) --------
  bfloat16 inv_sum = (bfloat16)inv(reduce_add_v32accf(sum_acc));
  v32bfloat16 inv_sum_v = broadcast_to_v32bfloat16(inv_sum);
  const v16accfloat half16 = broadcast_to_v16accfloat(0.5f);
  const v32accfloat half_acc = concat(half16, half16);
  // The int8 range, as bf16 (both exact). See the clamp below.
  const v32bfloat16 lo_v = broadcast_to_v32bfloat16((bfloat16)-128.0f);
  const v32bfloat16 hi_v = broadcast_to_v32bfloat16((bfloat16)127.0f);
  {
    const v32bfloat16 *restrict pIn = (const v32bfloat16 *)e_row;
    v32int8 *restrict pOut = (v32int8 *)out;
    for (int32_t i = 0; i < iters; ++i) {
      v32bfloat16 p = to_v32bfloat16(
          mul_elem_32(*pIn++, /*sgn_x=*/1, inv_sum_v, /*sgn_y=*/1));
      // Scale to the output quantization grid and bias by half, so the floor
      // below rounds to nearest.
      v32bfloat16 scaled = to_v32bfloat16(
          mac_elem_32(p, /*sgn_x=*/1, outScale_v, /*sgn_y=*/1, half_acc));
      // Clamp before the narrowing: `ssrs` only saturates when the core's
      // saturation mode is on, and it is off by default, so an out-of-range
      // value wraps. Here that happens whenever the output scale was calibrated
      // on probabilities below 1 (1 / s_out > 127.5, true in 6 of BERT's 12
      // layers): a probability near 1 lands above 127 and came out as a large
      // negative value instead (131 -> -125). The same clamp as layernorm.cc.
      scaled = min(max(scaled, lo_v), hi_v);
      v16int32 lo = bfloat16_to_int(extract_v16bfloat16(scaled, 0), 0);
      v16int32 hi = bfloat16_to_int(extract_v16bfloat16(scaled, 1), 0);
      *pOut++ = ssrs((v32acc32)concat(lo, hi), 0, 0);
    }
  }

  event1();
}

// Entry-point factory.
// TODO(avarma): Revisit this to have just one ABI.
//
// Two ABIs are exposed here, so the same softmax.o serves both the `.aiec`
// per-row flow and the linalg.softmax -> ukernel matcher flow.
//
//   1. Bareptr-friendly per-N row specialisations.
//
//   2. Per-(M,N) tile specialisations.
extern "C" {
// UKernelGenericOp lowers every memref offset as MLIR `index`, which is i64 in
// the generated core LLVM IR. Keep these wrapper parameters i64 too: a 32-bit
// C `unsigned` shifts later pointer/scalar arguments in the AIE ABI and can
// silently make a quantized softmax write zeros.

#define SOFTMAX_BF16_PER_N(N)                                                \
  void softmax_bf16_##N(bfloat16 *restrict input, int64_t input_offset,      \
                        bfloat16 *restrict output, int64_t output_offset) {  \
    softmax_bf16_impl(input + input_offset, output + output_offset, (N));    \
  }

SOFTMAX_BF16_PER_N(32)
SOFTMAX_BF16_PER_N(64)
SOFTMAX_BF16_PER_N(128)
SOFTMAX_BF16_PER_N(256)
SOFTMAX_BF16_PER_N(512)
SOFTMAX_BF16_PER_N(1024)
SOFTMAX_BF16_PER_N(2048)

#undef SOFTMAX_BF16_PER_N

#define SOFTMAX_BF16_PER_MxN(M, N)                                             \
  void softmax_bf16_##M##x##N(bfloat16 *restrict input, int64_t input_offset,  \
                              bfloat16 *restrict output,                       \
                              int64_t output_offset) {                         \
    bfloat16 *restrict in = input + input_offset;                              \
    bfloat16 *restrict out = output + output_offset;                           \
    for (int32_t r = 0; r < (M); ++r) {                                        \
      softmax_bf16_impl(in + r * (N), out + r * (N), (N));                     \
    }                                                                          \
  }

SOFTMAX_BF16_PER_MxN(4, 128)
SOFTMAX_BF16_PER_MxN(96, 32)

#undef SOFTMAX_BF16_PER_MxN

//   3. Quantized per-(M,N) tiles: int8 in, int8 out, with the surrounding
//      dequantize/requantize scales folded in.

#define SOFTMAX_I8_PER_MxN(M, N)                                            \
  void softmax_i8_##M##x##N(int8 *restrict input, int64_t input_offset,     \
                            int8 *restrict output, int64_t output_offset,   \
                            float log2eScaled, float outScale) {            \
    int8 *restrict in = input + input_offset;                               \
    int8 *restrict out = output + output_offset;                            \
    for (int32_t r = 0; r < (M); ++r) {                                     \
      softmax_i8_impl(in + r * (N), out + r * (N), (N), log2eScaled,        \
                      outScale);                                            \
    }                                                                       \
  }

SOFTMAX_I8_PER_MxN(96, 32)
SOFTMAX_I8_PER_MxN(32, 32)

#undef SOFTMAX_I8_PER_MxN

}  // extern "C"
