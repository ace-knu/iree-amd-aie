// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace {

constexpr int kVecLanes = 32;

// Scratch filled through one type and read back through another -- widened
// samples written as scalars and read as a vector, accumulators stored and read
// back as float or int lanes -- is a genuine reinterpretation.
typedef v16float v16float_alias __attribute__((may_alias));
typedef v16int32 v16int32_alias __attribute__((may_alias));

// `x + kRoundingMagic` for |x| < 2^22 lands in [2^23, 2^24), where an f32 has
// a spacing of exactly one: the addition itself rounds x to an integer (to
// nearest, ties to even, on aie2p's float accumulator), and
// the sum's bit pattern is `kMagicBits + that integer`. That is the only way
// to get an f32 lane to an integer here -- the one vector float-to-int
// conversion aie2p has is `bfloat16_to_int`, and bf16 cannot hold an int8
// value's neighbourhood: its spacing between 64 and 128 is 0.5.
constexpr float kRoundingMagic = 12582912.0f;  // 1.5 * 2^23
constexpr int32_t kMagicBits = 0x4B400000;

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

// All sixteen lanes of an accumulator as floats, or as the bits of those
// floats, through 64 bytes of scratch -- the same way out as `lane0`.
INTRINSIC(v16float) toFloat(v16accfloat r, float *scratch) {
  *(v16accfloat *)scratch = r;
  return *(const v16float_alias *)scratch;
}
INTRINSIC(v16int32) toBits(v16accfloat r, float *scratch) {
  *(v16accfloat *)scratch = r;
  return *(const v16int32_alias *)scratch;
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
// in f32 lanes (`mul_elem_16_accuracy_safe`), from the statistics to the
// requantization. Scalars are only ever splatted into vectors or extracted
// from them, and the one transcendental is `invsqrt`, a single aie2p NLF
// instruction, refined by a Newton step. Nothing goes through bf16 except
// gamma and beta, which arrive in it.
//
// Against an exact reference the result is exact but for values within f32's
// resolution of a rounding boundary (1 in 24,576 on a BERT-base LayerNorm).
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
  const float var = lane0(var_a, tmp);
  // `invsqrt` is an aie2p NLF instruction of its own accuracy. One Newton step,
  // r * (1.5 - 0.5 * var * r^2), in accurate f32 lanes, brings it to f32's
  // whatever that is: the result multiplies the whole row.
  float rstd = invsqrt(var);
  {
    const v16float r_v = broadcast_to_v16float(rstd);
    const float r2 = lane0(mul_elem_16_accuracy_safe(r_v, r_v), tmp);
    const float halfVarR2 = lane0(
        mul_elem_16_accuracy_safe(broadcast_to_v16float(0.5f * var),
                                  broadcast_to_v16float(r2)),
        tmp);
    const float step = lane0(sub(broadcast_to_v16accfloat(1.5f),
                                 broadcast_to_v16accfloat(halfVarR2)),
                             tmp);
    rstd = lane0(mul_elem_16_accuracy_safe(r_v, broadcast_to_v16float(step)),
                 tmp);
  }

  // ---- Pass 2: normalize, scale, shift and requantize, in f32 lanes -----
  // Nothing on the way from the samples to the int8 result goes through bf16.
  // It used to, and every one of those narrowings rounds toward -infinity on
  // aie2p; the row came out with a gain of 0.9947 and a -0.048 LSB offset
  // against an exact reference, which SmoothQuant made visible: it flattens a
  // LayerNorm's output, so one normalized unit spans three times as many
  // output steps.
  //
  // 16 lanes at a time, since that is what the accurate f32 multiply takes.
  // `lanes` is free once the statistics are in, and gives the four 64-byte
  // scratch areas the accumulators leave through.
  const v16accfloat mean16 = broadcast_to_v16accfloat(mean);
  const v16float rstd16 = broadcast_to_v16float(rstd);
  const v16accfloat magic16 = broadcast_to_v16accfloat(kRoundingMagic);
  const v16int32 lo16 = broadcast_to_v16int32(kMagicBits - 128);
  const v16int32 hi16 = broadcast_to_v16int32(kMagicBits + 127);
  float *restrict wide = (float *)chunk;
  float *restrict scratch = (float *)lanes;
  {
    const v16bfloat16 *restrict pGamma = (const v16bfloat16 *)gammaBeta;
    const v16bfloat16 *restrict pBeta = (const v16bfloat16 *)(gammaBeta + n);
    v32int8 *restrict pOut = (v32int8 *)out;
    for (int32_t i = 0; i < iters; ++i) {
      v16int32 halves[2];
      for (int32_t h = 0; h < 2; ++h) {
        // There is no vector int-to-float conversion on aie2p, but the scalar
        // one is a single `fx2flt` and exact.
        const int32_t base = i * kVecLanes + h * 16;
        for (int32_t j = 0; j < 16; ++j)
          wide[j] = fix2float((int32_t)in[base + j], 0);
        const v16float x = *(const v16float_alias *)wide;
        const v16float centred =
            toFloat(sub((v16accfloat)x, mean16), scratch);
        const v16float gamma = toFloat(ups(*pGamma++), scratch + 16);
        const v16float scale =
            toFloat(mul_elem_16_accuracy_safe(rstd16, gamma), scratch + 32);
        // The accumulator's own addition rounds to nearest, ties to even --
        // unlike the conversions to bf16, which round toward -infinity -- so
        // adding the magic number rounds y the way ORT's QuantizeLinear does,
        // with no bias by a half. It goes in last: added any earlier it would
        // round beta to an integer.
        const v16accfloat y = mac_elem_16_accuracy_safe(
            centred, scale, ups(*pBeta++), /*zero_acc=*/0, /*sub_mul=*/0,
            /*sub_acc1=*/0);
        // Clamp as integers: -128..127 around the magic number.
        halves[h] =
            min(max(toBits(add(y, magic16), scratch + 48), lo16), hi16);
      }
      // `ssrs` wraps rather than saturates (the core's saturation mode is off),
      // so it keeps each lane's low byte, which after the clamp is exactly the
      // int8 value: the magic number's low byte is zero.
      *pOut++ = ssrs((v32acc32)concat(halves[0], halves[1]), 0, 0);
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
LAYERNORM_I8_PER_MxN(4, 768)

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
LAYERNORM_I16_PER_MxN(4, 768)

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
LAYERNORM_I8_BARE_MxN(4, 768)

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
LAYERNORM_I16_BARE_MxN(4, 768)

#undef LAYERNORM_I16_BARE_MxN

}  // extern "C"
