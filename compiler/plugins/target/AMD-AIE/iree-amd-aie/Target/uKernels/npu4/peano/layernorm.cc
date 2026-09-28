// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace {

constexpr int kVecLanes = 32;

// The chunk scratch is filled one bfloat16 at a time (the int8->float
// conversion is scalar on aie2p) and read back as one 32-lane vector, which is
// a genuine reinterpretation.
typedef v32bfloat16 v32bfloat16_alias __attribute__((may_alias));

// Horizontal add of a v32accfloat, with the total left in lane 0 and never
// narrowed on the way out.
//
// softmax.cc's `reduce_add_v32accf` reads its result out through bf16, which is
// fine for a reciprocal but not here: the row sum of int8 data reaches ~10^5,
// where bf16's 8-bit mantissa is worth ~256, and that lands directly on the
// mean. Keeping the reduction in accfloat holds the sum exactly -- a row of 768
// int8 values sums to at most 97,536 and squares to at most 12,386,304, both
// well inside f32's exact-integer range.
INTRINSIC(v16accfloat) reduce_add_v32accf_exact(v32accfloat acc) {
  const v16accfloat z = broadcast_zero_to_v16accfloat();
  // Fold 32 -> 16 lanes (lossless), then a log2(16) = 4-step shift/add
  // butterfly down to lane 0.
  v16accfloat r = add(extract_v16accfloat(acc, 0), extract_v16accfloat(acc, 1));
  r = add(r, shift(r, z, 8));
  r = add(r, shift(r, z, 4));
  r = add(r, shift(r, z, 2));
  r = add(r, shift(r, z, 1));
  return r;
}

// Read lane 0 of an accumulator as a full-precision float, through memory.
//
// Both of aie2p's accumulator-to-float conversions (`to_v16float`,
// `to_v8float`) fail instruction selection -- peano declares the intrinsics but
// cannot lower them -- so the only lossless way out of an accumulator is a
// store. That is exactly what it compiles to: a single `vst bmll0, [p]` with no
// conversion, after which an ordinary scalar load reads the lane back. Going
// out through bf16 instead, the way softmax does, would cost the mean its
// mantissa.
INTRINSIC(float) lane0(v16accfloat r, float *tmp) {
  *(v16accfloat *)tmp = r;
  // Read the lane back with memcpy rather than a plain load: the store and the
  // load go through different types, and telling the compiler they alias (which
  // a bare `tmp[0]` does not) is what keeps it from reordering or dropping the
  // store.
  float v;
  __builtin_memcpy(&v, tmp, sizeof(v));
  return v;
}

/// Sum and sum of squares of one int8 row, both exact.
///
/// Taken in the integer domain with the core's int8 MAC array, 64 lanes at a
/// time, so the statistics never read the bf16 scratch. A row of 768 int8
/// values sums to at most 97,536 and squares to at most 12,386,304, and each of
/// the 64 partial lanes carries at most 12 chunks' worth, so every accumulator
/// stays exact in int32.
inline void rowStats(int8 *restrict in, int32_t n, int32_t *lanes, float &sum,
                     float &sumSq) {
  v64acc32 sum_acc = broadcast_zero_to_v64acc32();
  v64acc32 sq_acc = broadcast_zero_to_v64acc32();
  {
    const v64int8 *restrict pIn = (const v64int8 *)in;
    const v64uint8 ones = (v64uint8)broadcast_to_v64int8(1);
    for (int32_t i = 0; i < n / 64; ++i) {
      v64int8 x = *pIn++;
      sum_acc = mac_elem_64(x, /*sgn_x=*/1, ones, /*sgn_y=*/0, sum_acc);
      sq_acc = mac_elem_64(x, /*sgn_x=*/1, (v64uint8)x, /*sgn_y=*/1, sq_acc);
    }
  }
  // Fold the lanes on the scalar side: integer adds are native here, and it
  // avoids assuming anything about an accumulator butterfly's lane order.
  int32_t sumI = 0, sumSqI = 0;
  *(v64acc32 *)lanes = sum_acc;
  for (int32_t i = 0; i < 64; ++i) sumI += lanes[i];
  *(v64acc32 *)lanes = sq_acc;
  for (int32_t i = 0; i < 64; ++i) sumSqI += lanes[i];
  // Exact: both are whole numbers well inside f32's exact-integer range.
  sum = fix2float(sumI, 0);
  sumSq = fix2float(sumSqI, 0);
}

/// Sum and sum of squares of one int16 row.
///
/// The wider input is what lets a producer -- a residual add, say -- hand over
/// its result without first squeezing it back into int8. aie2p has an int16 MAC
/// that accumulates into 64-bit lanes, so the sums stay exact: a row of 768
/// int16 values squares to at most 8.2e11, which needs the 64-bit accumulator
/// and would overflow the int8 path's 32-bit one. (There is no such MAC for
/// int32 operands, which is why this variant is int16 and not wider.)
inline void rowStats(int16 *restrict in, int32_t n, int32_t *lanes, float &sum,
                     float &sumSq) {
  v32acc64 sum_acc = broadcast_zero_to_v32acc64();
  v32acc64 sq_acc = broadcast_zero_to_v32acc64();
  {
    const v32int16 *restrict pIn = (const v32int16 *)in;
    const v32uint16 ones = broadcast_to_v32uint16((unsigned short)1);
    for (int32_t i = 0; i < n / 32; ++i) {
      v32int16 x = *pIn++;
      sum_acc = mac_elem_32(x, /*sgn_x=*/1, ones, /*sgn_y=*/0, sum_acc);
      sq_acc = mac_elem_32(x, /*sgn_x=*/1, (v32uint16)x, /*sgn_y=*/1, sq_acc);
    }
  }
  int64_t *restrict wide = (int64_t *)lanes;
  int64_t sumI = 0, sumSqI = 0;
  *(v32acc64 *)wide = sum_acc;
  for (int32_t i = 0; i < 32; ++i) sumI += wide[i];
  *(v32acc64 *)wide = sq_acc;
  for (int32_t i = 0; i < 32; ++i) sumSqI += wide[i];
  // The sum still fits int32. The sum of squares does not, so drop the low 16
  // bits before converting: what is left is under 2^24 and therefore exact in
  // f32, and the bits dropped are worth 2^-24 of the value.
  sum = fix2float((int32_t)sumI, 0);
  sumSq = fix2float((int32_t)(sumSqI >> 16), -16);
}

// Row-wise LayerNorm over a contiguous int8 row of length `n` (`n % 32 == 0`),
// producing int8 again.
//
//   out = quantize(gamma * (x*s - mean) / sqrt(var + eps) + beta)
//
// The caller folds both quantization scales away before the kernel ever sees
// them, which is what keeps this free of scalar float arithmetic:
//
//   * The input scale `s` cancels. With m and v the mean and variance of the
//     *integer* samples, (x*s - m*s) / sqrt(v*s^2 + eps) = (x - m) /
//     sqrt(v + eps/s^2), so `s` survives only inside the epsilon the caller
//     passes as `epsilonScaled = eps / s^2`.
//   * The output scale divides gamma and beta, which are compile-time
//     constants, so `gammaBeta` arrives as gamma/s_out followed by beta/s_out.
//
// aie2p has no scalar float arithmetic -- a single multiply becomes a
// `__mulsf3` libcall -- so every arithmetic step below happens in a vector,
// including the per-row mean and variance, which run in f32 lanes
// (`mul_elem_16_accuracy_safe`) rather than through bf16. Scalars are only ever
// splatted into vectors or extracted from them, and the one transcendental is
// `invsqrt`, a single aie2p NLF instruction.
//
// The int8 inputs are whole numbers of magnitude <= 128, which bf16 represents
// exactly, so widening the row costs nothing in accuracy. What bf16 does cost
// is documented at each rounding below.
//
// Kept deliberately small on the stack. Both mlir-aie (`aie.core`'s
// `stack_size`) and IREE (`--iree-amdaie-stack-size`) default to 0x400 = 1 KB,
// and mlir-aie does not check the frame against it: an oversized frame does not
// fault, it runs off the end and the first rows of a tile come out with
// corrupted statistics, deterministically, while later rows stay bit-exact.
// (IREE does check, and fails the compile with an explicit message.)
template <typename InT>
inline void layernorm_impl(InT *restrict inBase, int8 *restrict outBase,
                              int32_t m, int32_t n,
                              const bfloat16 *restrict gammaBeta,
                              float epsilonScaled, float invN,
                              bfloat16 *restrict chunk, float *tmp,
                              int32_t *lanes) {
  event0();

  const int32_t iters = n / kVecLanes;
  for (int32_t r = 0; r < m; ++r) {
  InT *restrict in = inBase + r * n;
  int8 *restrict out = outBase + r * n;

  // ---- Pass 1: sum and sum of squares, straight off the int8 row --------
  // Both are taken in the integer domain with the core's int8 MAC array, 64
  // lanes at a time, so the statistics never read the bf16 scratch. A row of
  // 768 int8 values sums to at most 97,536 and squares to at most 12,386,304,
  // and each of the 64 partial lanes carries at most 12 chunks' worth, so every
  // accumulator stays exact in int32.
  float rowSum, rowSumSq;
  rowStats(in, n, lanes, rowSum, rowSumSq);

  // ---- Mean, variance and 1/sqrt, all in f32 lanes ----------------------
  // var = E[x^2] - mean^2 would cancel catastrophically on float data, but
  // these are exact integer sums, so the subtraction is exact up to f32's
  // rounding of the two products.
  const v16float invN_v = broadcast_to_v16float(invN);
  const v16accfloat mean_a =
      mul_elem_16_accuracy_safe(broadcast_to_v16float(rowSum), invN_v);
  const float mean = lane0(mean_a, tmp);
  const v16float mean_v = broadcast_to_v16float(mean);
  v16accfloat var_a =
      sub(mul_elem_16_accuracy_safe(broadcast_to_v16float(rowSumSq), invN_v),
          mul_elem_16_accuracy_safe(mean_v, mean_v));
  var_a = add(var_a, broadcast_to_v16accfloat(epsilonScaled));
  const float rstd = invsqrt(lane0(var_a, tmp));

  // ---- Pass 2: normalize, scale, shift and requantize -------------------
  const v16accfloat mean16 = broadcast_to_v16accfloat(mean);
  const v32accfloat mean_acc = concat(mean16, mean16);
  const v32bfloat16 rstd_v = broadcast_to_v32bfloat16((bfloat16)rstd);
  const v16accfloat half16 = broadcast_to_v16accfloat(0.5f);
  const v32accfloat half_acc = concat(half16, half16);
  // The int8 range, as bf16 (both exact). See the clamp below.
  const v32bfloat16 lo_v = broadcast_to_v32bfloat16((bfloat16)-128.0f);
  const v32bfloat16 hi_v = broadcast_to_v32bfloat16((bfloat16)127.0f);
  {
    const v32bfloat16 *restrict pGamma = (const v32bfloat16 *)gammaBeta;
    const v32bfloat16 *restrict pBeta = (const v32bfloat16 *)(gammaBeta + n);
    v32int8 *restrict pOut = (v32int8 *)out;
    for (int32_t i = 0; i < iters; ++i) {
      // Widen this chunk only. There is no vector int-to-float conversion on
      // aie2p, but the scalar one is a single `fx2flt`, and since the
      // statistics above read the int8 row directly there is no reason to keep
      // a whole widened row around -- 32 lanes of scratch instead of 768 keeps
      // the frame under the 1 KB core stack every backend defaults to.
      for (int32_t j = 0; j < kVecLanes; ++j)
        chunk[j] = (bfloat16)fix2float((int32_t)in[i * kVecLanes + j], 0);
      v32bfloat16 x = *(const v32bfloat16_alias *)chunk;
      // Centring stays in accfloat, so the mean keeps its f32 mantissa; only
      // the (small) centred value is narrowed for the multiply below.
      v32bfloat16 centred = to_v32bfloat16(sub(ups(x), mean_acc));
      // gamma/s_out scaled by 1/sqrt(var): one rounding on a per-feature
      // constant, in the same class as gamma arriving as bf16 to begin with.
      v32bfloat16 scale = to_v32bfloat16(mul_elem_32(rstd_v, /*sgn_x=*/1,
                                                     *pGamma++, /*sgn_y=*/1));
      v32accfloat y = mac_elem_32(centred, /*sgn_x=*/1, scale, /*sgn_y=*/1,
                                  ups(*pBeta++));
      // Bias by half so the floor below rounds to nearest, for both signs.
      v32bfloat16 biased = to_v32bfloat16(add(y, half_acc));
      // Clamp here, before the narrowing: `ssrs` only saturates when the core's
      // saturation mode is on, and it is off by default, so an out-of-range
      // value wraps instead (BERT layer 5's [CLS] outlier feature, about -132,
      // came out as +124). Turning the mode on for the kernel is not an option
      // -- it changes the bf16 conversions above as well. Clamping the biased
      // value to [-128, 127] leaves the floor giving -128..127 exactly, which
      // is the clamp the float form spells out.
      biased = min(max(biased, lo_v), hi_v);
      v16int32 lo = bfloat16_to_int(extract_v16bfloat16(biased, 0), 0);
      v16int32 hi = bfloat16_to_int(extract_v16bfloat16(biased, 1), 0);
      *pOut++ = ssrs((v32acc32)concat(lo, hi), 0, 0);
    }
  }
  }

  event1();
}

}  // namespace

// Per-(M,N) tile entry points, matching softmax.cc's shape-specialised naming
// so the same `iree_amdaie_uk_layernorm_i8_<M>x<N>` lookup works.
//
// UKernelGenericOp lowers every memref offset as MLIR `index`, which is i64 in
// the generated core LLVM IR. Keep these wrapper parameters i64 too: a 32-bit
// C `unsigned` shifts later pointer/scalar arguments in the AIE ABI and can
// silently make the kernel write zeros -- softmax hit exactly that.
extern "C" {

/* Operand order follows what `iree_codegen.ukernel.generic` emits: every input
   as a (pointer, offset) pair, then the outputs the same way, then the plain
   scalars. */                                                                \
#define LAYERNORM_I8_PER_MxN(M, N)                                            \
  void layernorm_i8_##M##x##N(                                                \
      int8 *restrict input, int64_t input_offset,                             \
      bfloat16 *restrict gamma_beta, int64_t gamma_beta_offset,               \
      int8 *restrict output, int64_t output_offset,                           \
      float epsilon_scaled) {                                                 \
    int8 *restrict in = input + input_offset;                                 \
    int8 *restrict out = output + output_offset;                              \
    const bfloat16 *restrict gb = gamma_beta + gamma_beta_offset;             \
    /* Read and written as whole vectors, so they have to carry the vector \
       alignment; a plain stack array does not. */                            \
    __attribute__((aligned(64))) bfloat16 chunk[32];                          \
    __attribute__((aligned(64))) float tmp[16];                               \
    __attribute__((aligned(64))) int32_t lanes[64];                           \
    layernorm_impl(in, out, (M), (N), gb, epsilon_scaled,                     \
                   1.0f / (float)(N), chunk, tmp, lanes);                     \
  }

LAYERNORM_I8_PER_MxN(16, 768)
LAYERNORM_I8_PER_MxN(8, 768)

#undef LAYERNORM_I8_PER_MxN

/* int16 input, int8 output. Same normalization, same packed gamma/beta and
   scaled epsilon; only the samples are wider, so that a producer such as a
   residual add can hand over its result without quantizing it back to int8
   first. The caller's epsilon is scaled by that producer's step instead. */
#define LAYERNORM_I16_PER_MxN(M, N)                                           \
  void layernorm_i16_##M##x##N(                                               \
      int16 *restrict input, int64_t input_offset,                            \
      bfloat16 *restrict gamma_beta, int64_t gamma_beta_offset,               \
      int8 *restrict output, int64_t output_offset,                           \
      float epsilon_scaled) {                                                 \
    int16 *restrict in = input + input_offset;                                \
    int8 *restrict out = output + output_offset;                              \
    const bfloat16 *restrict gb = gamma_beta + gamma_beta_offset;             \
    /* Read and written as whole vectors, so they have to carry the vector    \
       alignment; a plain stack array does not. */                            \
    __attribute__((aligned(64))) bfloat16 chunk[32];                          \
    __attribute__((aligned(64))) float tmp[16];                               \
    __attribute__((aligned(64))) int32_t lanes[64];                           \
    layernorm_impl(in, out, (M), (N), gb, epsilon_scaled,                     \
                   1.0f / (float)(N), chunk, tmp, lanes);                     \
  }

LAYERNORM_I16_PER_MxN(16, 768)
LAYERNORM_I16_PER_MxN(8, 768)

#undef LAYERNORM_I16_PER_MxN

// Bare-pointer entry points, for driving the same kernel from a hand-written
// mlir-aie design.
//
// softmax.cc exposes two ABIs for the same reason. Here there is also a hard
// limit forcing it: aiecc asserts (`eraseOp ... expected that op has no uses`)
// while lowering an `aie.core` that calls an external function with more than
// four arguments once any of them is an integer, so the offset pairs of the
// UKernelGenericOp ABI above cannot be expressed in that flow at all. Measured
// boundary: (memref, memref, memref, f32) lowers; adding one i64 does not.
#define LAYERNORM_I8_BARE_MxN(M, N)                                          \
  void layernorm_i8_bare_##M##x##N(int8 *restrict input,                     \
                                   int8 *restrict output,                    \
                                   bfloat16 *restrict gamma_beta,            \
                                   float epsilon_scaled) {                   \
    layernorm_i8_##M##x##N(input, 0, gamma_beta, 0, output, 0,               \
                           epsilon_scaled);                                  \
  }

LAYERNORM_I8_BARE_MxN(16, 768)
LAYERNORM_I8_BARE_MxN(8, 768)

#undef LAYERNORM_I8_BARE_MxN

#define LAYERNORM_I16_BARE_MxN(M, N)                                         \
  void layernorm_i16_bare_##M##x##N(int16 *restrict input,                   \
                                    int8 *restrict output,                   \
                                    bfloat16 *restrict gamma_beta,           \
                                    float epsilon_scaled) {                  \
    layernorm_i16_##M##x##N(input, 0, gamma_beta, 0, output, 0,              \
                            epsilon_scaled);                                 \
  }

LAYERNORM_I16_BARE_MxN(16, 768)
LAYERNORM_I16_BARE_MxN(8, 768)

#undef LAYERNORM_I16_BARE_MxN

}  // extern "C"
