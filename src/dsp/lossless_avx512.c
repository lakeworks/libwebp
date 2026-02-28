// Copyright 2024 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// AVX-512 variant of methods for lossless decoder
//
// Author: lakeworks (gh@lkwh.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_AVX512)

#include <stddef.h>
#include <immintrin.h>

#include "src/dsp/cpu.h"
#include "src/dsp/lossless.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// Predictor Transform

static WEBP_INLINE void Average2_m512i(const __m512i* const a0,
                                       const __m512i* const a1,
                                       __m512i* const avg) {
  // (a + b) >> 1 = ((a + b + 1) >> 1) - ((a ^ b) & 1)
  const __m512i ones = _mm512_set1_epi8(1);
  const __m512i avg1 = _mm512_avg_epu8(*a0, *a1);
  const __m512i one = _mm512_and_si512(_mm512_xor_si512(*a0, *a1), ones);
  *avg = _mm512_sub_epi8(avg1, one);
}

// Batch versions of predictor functions.

// Predictor0: ARGB_BLACK.
static void PredictorAdd0_AVX512(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m512i black = _mm512_set1_epi32((int)ARGB_BLACK);
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    const __m512i res = _mm512_add_epi8(src, black);
    _mm512_storeu_si512((__m512i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsAdd_SSE[0](in + i, NULL, num_pixels - i, out + i);
  }
  (void)upper;
}

// Predictor1 (left): serial dependency on out[i-1] — stays at AVX2.

// Macro for fully parallel predictors (top row based).
#define GENERATE_PREDICTOR_1(X, IN)                                         \
  static void PredictorAdd##X##_AVX512(const uint32_t* in,                  \
                                       const uint32_t* upper, int num_pixels, \
                                       uint32_t* WEBP_RESTRICT out) {       \
    int i;                                                                  \
    for (i = 0; i + 16 <= num_pixels; i += 16) {                            \
      const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);       \
      const __m512i other = _mm512_loadu_si512((const __m512i*)&(IN));      \
      const __m512i res = _mm512_add_epi8(src, other);                      \
      _mm512_storeu_si512((__m512i*)&out[i], res);                          \
    }                                                                       \
    if (i != num_pixels) {                                                  \
      VP8LPredictorsAdd_SSE[(X)](in + i, upper + i, num_pixels - i, out + i); \
    }                                                                       \
  }

// Predictor2: Top.
GENERATE_PREDICTOR_1(2, upper[i])
// Predictor3: Top-right.
GENERATE_PREDICTOR_1(3, upper[i + 1])
// Predictor4: Top-left.
GENERATE_PREDICTOR_1(4, upper[i - 1])
#undef GENERATE_PREDICTOR_1

// Due to averages with integers, values cannot be accumulated in parallel for
// predictors 5 to 7 (they depend on the current output L = out[i-1]).
// Predictors 5, 6, 7, 13 remain at SSE level (not implemented in AVX2 either).

#define GENERATE_PREDICTOR_2(X, IN)                                         \
  static void PredictorAdd##X##_AVX512(const uint32_t* in,                  \
                                       const uint32_t* upper, int num_pixels, \
                                       uint32_t* WEBP_RESTRICT out) {       \
    int i;                                                                  \
    for (i = 0; i + 16 <= num_pixels; i += 16) {                            \
      const __m512i Tother = _mm512_loadu_si512((const __m512i*)&(IN));     \
      const __m512i T = _mm512_loadu_si512((const __m512i*)&upper[i]);      \
      const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);       \
      __m512i avg, res;                                                     \
      Average2_m512i(&T, &Tother, &avg);                                    \
      res = _mm512_add_epi8(avg, src);                                      \
      _mm512_storeu_si512((__m512i*)&out[i], res);                          \
    }                                                                       \
    if (i != num_pixels) {                                                  \
      VP8LPredictorsAdd_SSE[(X)](in + i, upper + i, num_pixels - i, out + i); \
    }                                                                       \
  }
// Predictor8: average TL T.
GENERATE_PREDICTOR_2(8, upper[i - 1])
// Predictor9: average T TR.
GENERATE_PREDICTOR_2(9, upper[i + 1])
#undef GENERATE_PREDICTOR_2

// Predictors 10, 11, 12: serial dependency on L (out[i-1]) — these stay
// at AVX2 (dispatched from lossless_avx2.c). No AVX-512 benefit.
// Predictors 5, 6, 7, 13: also serial, stay at SSE level.

//------------------------------------------------------------------------------
// Subtract-Green Transform (inverse)

static void AddGreenToBlueAndRed_AVX512(const uint32_t* const src,
                                        int num_pixels, uint32_t* dst) {
  int i;
  const __m512i kCstShuffle = _mm512_set_epi8(
      -1, 61, -1, 61, -1, 57, -1, 57, -1, 53, -1, 53, -1, 49, -1, 49,
      -1, 45, -1, 45, -1, 41, -1, 41, -1, 37, -1, 37, -1, 33, -1, 33,
      -1, 29, -1, 29, -1, 25, -1, 25, -1, 21, -1, 21, -1, 17, -1, 17,
      -1, 13, -1, 13, -1,  9, -1,  9, -1,  5, -1,  5, -1,  1, -1,  1);
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i in = _mm512_loadu_si512((const __m512i*)&src[i]);
    const __m512i in_0g0g = _mm512_shuffle_epi8(in, kCstShuffle);
    const __m512i out = _mm512_add_epi8(in, in_0g0g);
    _mm512_storeu_si512((__m512i*)&dst[i], out);
  }
  if (i != num_pixels) {
    VP8LAddGreenToBlueAndRed_SSE(src + i, num_pixels - i, dst + i);
  }
}

//------------------------------------------------------------------------------
// Color Transform (inverse)

static void TransformColorInverse_AVX512(const VP8LMultipliers* const m,
                                         const uint32_t* const src,
                                         int num_pixels, uint32_t* dst) {
#define CST(X)  (((int16_t)(m->X << 8)) >> 5)   // sign-extend
  const __m512i mults_rb =
      _mm512_set1_epi32((int)((uint32_t)CST(green_to_red) << 16 |
                              (CST(green_to_blue) & 0xffff)));
  const __m512i mults_b2 = _mm512_set1_epi32(CST(red_to_blue));
#undef CST
  // perm1: extract green as [0, G, 0, G] per pixel for mulhi_epi16.
  // In set_epi8(e63..e0), e0 is byte 0. We need byte0=0, byte1=G(idx 1).
  const __m512i perm1 = _mm512_set_epi8(
      61, -1, 61, -1, 57, -1, 57, -1, 53, -1, 53, -1, 49, -1, 49, -1,
      45, -1, 45, -1, 41, -1, 41, -1, 37, -1, 37, -1, 33, -1, 33, -1,
      29, -1, 29, -1, 25, -1, 25, -1, 21, -1, 21, -1, 17, -1, 17, -1,
      13, -1, 13, -1,  9, -1,  9, -1,  5, -1,  5, -1,  1, -1,  1, -1);
  // perm2: extract new R as [0, R, 0, 0] per pixel for mulhi_epi16.
  // R is byte 2 within each 4-byte pixel. We need it at byte 1 of each dword
  // so that the lower 16-bit word = R*256, matching the AVX2 pattern.
  // In set_epi8(e63..e0), e0 is byte 0: byte0=-1, byte1=R(idx 2), byte2=-1.
  const __m512i perm2 = _mm512_set_epi8(
      -1, -1, 62, -1, -1, -1, 58, -1, -1, -1, 54, -1, -1, -1, 50, -1,
      -1, -1, 46, -1, -1, -1, 42, -1, -1, -1, 38, -1, -1, -1, 34, -1,
      -1, -1, 30, -1, -1, -1, 26, -1, -1, -1, 22, -1, -1, -1, 18, -1,
      -1, -1, 14, -1, -1, -1, 10, -1, -1, -1,  6, -1, -1, -1,  2, -1);
  int i;
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i A = _mm512_loadu_si512((const __m512i*)(src + i));
    const __m512i B = _mm512_shuffle_epi8(A, perm1);   // g0g0
    const __m512i C = _mm512_mulhi_epi16(B, mults_rb);
    const __m512i D = _mm512_add_epi8(A, C);
    const __m512i E = _mm512_shuffle_epi8(D, perm2);   // new red
    const __m512i F = _mm512_mulhi_epi16(E, mults_b2);
    const __m512i G = _mm512_add_epi8(D, F);
    // Blend: keep alpha+green from A (bytes 1,3), red+blue from G (bytes 0,2).
    // mask bit=1 selects from A (second arg), bit=0 selects from G (first arg).
    // Per pixel [B,G,R,A]: bits = 0b1010 = 0xA, repeated 16 times.
    const __m512i out = _mm512_mask_blend_epi8((__mmask64)0xAAAAAAAAAAAAAAAAULL,
                                               G, A);
    _mm512_storeu_si512((__m512i*)&dst[i], out);
  }
  if (i != num_pixels) {
    VP8LTransformColorInverse_SSE(m, src + i, num_pixels - i, dst + i);
  }
}

//------------------------------------------------------------------------------
// Color-space conversion functions

static void ConvertBGRAToRGBA_AVX512(const uint32_t* WEBP_RESTRICT src,
                                     int num_pixels,
                                     uint8_t* WEBP_RESTRICT dst) {
  const __m512i* in = (const __m512i*)src;
  __m512i* out = (__m512i*)dst;
  const __m512i kShufMask = _mm512_set_epi8(
      63, 60, 61, 62, 59, 56, 57, 58, 55, 52, 53, 54, 51, 48, 49, 50,
      47, 44, 45, 46, 43, 40, 41, 42, 39, 36, 37, 38, 35, 32, 33, 34,
      31, 28, 29, 30, 27, 24, 25, 26, 23, 20, 21, 22, 19, 16, 17, 18,
      15, 12, 13, 14, 11,  8,  9, 10,  7,  4,  5,  6,  3,  0,  1,  2);
  while (num_pixels >= 16) {
    const __m512i A = _mm512_loadu_si512(in++);
    const __m512i B = _mm512_shuffle_epi8(A, kShufMask);
    _mm512_storeu_si512(out++, B);
    num_pixels -= 16;
  }
  if (num_pixels > 0) {
    VP8LConvertBGRAToRGBA_SSE((const uint32_t*)in, num_pixels, (uint8_t*)out);
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LDspInitAVX512(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LDspInitAVX512(void) {
  // True 512-bit parallel predictors (16 px/iter, no serial dependency).
  VP8LPredictorsAdd[0] = PredictorAdd0_AVX512;
  VP8LPredictorsAdd[2] = PredictorAdd2_AVX512;
  VP8LPredictorsAdd[3] = PredictorAdd3_AVX512;
  VP8LPredictorsAdd[4] = PredictorAdd4_AVX512;
  VP8LPredictorsAdd[8] = PredictorAdd8_AVX512;
  VP8LPredictorsAdd[9] = PredictorAdd9_AVX512;

  // Full-row transforms: process entire image width, maximum AVX-512 benefit.
  VP8LAddGreenToBlueAndRed = AddGreenToBlueAndRed_AVX512;
  VP8LTransformColorInverse = TransformColorInverse_AVX512;
  VP8LConvertBGRAToRGBA = ConvertBGRAToRGBA_AVX512;

  // Left at AVX2 -- predictors 1, 10, 11, 12: serial dependency on
  //   previous output pixel (left, or avg involving left). These use
  //   256-bit AVX2 internally anyway -- overriding wastes dispatch and
  //   keeps CPU in heavy AVX-512 power state for zero benefit.
}

#else  // !WEBP_USE_AVX512

WEBP_DSP_INIT_STUB(VP8LDspInitAVX512)

#endif  // WEBP_USE_AVX512
