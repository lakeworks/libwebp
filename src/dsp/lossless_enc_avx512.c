// Copyright 2024 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// AVX-512 variant of methods for lossless encoder
//
// Author: lakeworks (gh@lkwh.com)

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_AVX512)
#include <immintrin.h>

#include <assert.h>
#include <stddef.h>

#include "src/dsp/cpu.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// Subtract-Green Transform

static void SubtractGreenFromBlueAndRed_AVX512(uint32_t* argb_data,
                                               int num_pixels) {
  int i;
  const __m512i kCstShuffle = _mm512_set_epi8(
      -1, 61, -1, 61, -1, 57, -1, 57, -1, 53, -1, 53, -1, 49, -1, 49,
      -1, 45, -1, 45, -1, 41, -1, 41, -1, 37, -1, 37, -1, 33, -1, 33,
      -1, 29, -1, 29, -1, 25, -1, 25, -1, 21, -1, 21, -1, 17, -1, 17,
      -1, 13, -1, 13, -1,  9, -1,  9, -1,  5, -1,  5, -1,  1, -1,  1);
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i in = _mm512_loadu_si512((const __m512i*)&argb_data[i]);
    const __m512i in_0g0g = _mm512_shuffle_epi8(in, kCstShuffle);
    const __m512i out = _mm512_sub_epi8(in, in_0g0g);
    _mm512_storeu_si512((__m512i*)&argb_data[i], out);
  }
  if (i != num_pixels) {
    VP8LSubtractGreenFromBlueAndRed_SSE(argb_data + i, num_pixels - i);
  }
}

//------------------------------------------------------------------------------
// Color Transform

// For sign-extended multiplying constants, pre-shifted by 5:
#define CST_5b(X) (((int16_t)((uint16_t)(X) << 8)) >> 5)

#define MK_CST_16(HI, LO) \
  _mm512_set1_epi32((int)(((uint32_t)(HI) << 16) | ((LO) & 0xffff)))

static void TransformColor_AVX512(const VP8LMultipliers* WEBP_RESTRICT const m,
                                  uint32_t* WEBP_RESTRICT argb_data,
                                  int num_pixels) {
  const __m512i mults_rb =
      MK_CST_16(CST_5b(m->green_to_red), CST_5b(m->green_to_blue));
  const __m512i mults_b2 = MK_CST_16(CST_5b(m->red_to_blue), 0);
  const __m512i mask_rb = _mm512_set1_epi32(0x00ff00ff);
  const __m512i kCstShuffle = _mm512_set_epi8(
      61, -1, 61, -1, 57, -1, 57, -1, 53, -1, 53, -1, 49, -1, 49, -1,
      45, -1, 45, -1, 41, -1, 41, -1, 37, -1, 37, -1, 33, -1, 33, -1,
      29, -1, 29, -1, 25, -1, 25, -1, 21, -1, 21, -1, 17, -1, 17, -1,
      13, -1, 13, -1,  9, -1,  9, -1,  5, -1,  5, -1,  1, -1,  1, -1);
  int i;
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i in = _mm512_loadu_si512((const __m512i*)&argb_data[i]);
    const __m512i A = _mm512_shuffle_epi8(in, kCstShuffle);      // g0g0
    const __m512i B = _mm512_mulhi_epi16(A, mults_rb);  // x dr  x db1
    const __m512i C = _mm512_slli_epi16(in, 8);         // r 0   b   0
    const __m512i D = _mm512_mulhi_epi16(C, mults_b2);  // x db2 0   0
    const __m512i E = _mm512_srli_epi32(D, 16);         // 0 0   x db2
    const __m512i F = _mm512_add_epi8(E, B);            // x dr  x  db
    const __m512i G = _mm512_and_si512(F, mask_rb);     // 0 dr  0  db
    const __m512i out = _mm512_sub_epi8(in, G);
    _mm512_storeu_si512((__m512i*)&argb_data[i], out);
  }
  if (i != num_pixels) {
    VP8LTransformColor_SSE(m, argb_data + i, num_pixels - i);
  }
}

//------------------------------------------------------------------------------
#define SPAN 16
static void CollectColorBlueTransforms_AVX512(
    const uint32_t* WEBP_RESTRICT argb, int stride, int tile_width,
    int tile_height, int green_to_blue, int red_to_blue, uint32_t histo[]) {
  const __m512i mult =
      MK_CST_16(CST_5b(red_to_blue) + 256, CST_5b(green_to_blue));
  // perm: extract [0, G, 0, R] per pixel for mulhi_epi16.
  // In set_epi8(e63..e0): e0=byte0, e1=byte1, etc.
  // Per pixel: byte0=0(-1), byte1=G(1), byte2=0(-1), byte3=R(2).
  const __m512i perm = _mm512_set_epi8(
      14, -1, 13, -1, 10, -1,  9, -1,  6, -1,  5, -1,  2, -1,  1, -1,
      14, -1, 13, -1, 10, -1,  9, -1,  6, -1,  5, -1,  2, -1,  1, -1,
      14, -1, 13, -1, 10, -1,  9, -1,  6, -1,  5, -1,  2, -1,  1, -1,
      14, -1, 13, -1, 10, -1,  9, -1,  6, -1,  5, -1,  2, -1,  1, -1);
  if (tile_width >= 16) {
    int y, i;
    for (y = 0; y < tile_height; ++y) {
      uint8_t values[64];
      const uint32_t* const src = argb + y * stride;
      const __m512i A1 = _mm512_loadu_si512((const __m512i*)src);
      const __m512i B1 = _mm512_shuffle_epi8(A1, perm);
      const __m512i C1 = _mm512_mulhi_epi16(B1, mult);
      const __m512i D1 = _mm512_sub_epi16(A1, C1);
      __m512i E = _mm512_add_epi16(_mm512_srli_epi32(D1, 16), D1);
      int x;
      for (x = 16; x + 16 <= tile_width; x += 16) {
        const __m512i A2 = _mm512_loadu_si512((const __m512i*)(src + x));
        __m512i B2, C2, D2;
        _mm512_storeu_si512((__m512i*)values, E);
        for (i = 0; i < 64; i += 4) ++histo[values[i]];
        B2 = _mm512_shuffle_epi8(A2, perm);
        C2 = _mm512_mulhi_epi16(B2, mult);
        D2 = _mm512_sub_epi16(A2, C2);
        E = _mm512_add_epi16(_mm512_srli_epi32(D2, 16), D2);
      }
      _mm512_storeu_si512((__m512i*)values, E);
      for (i = 0; i < 64; i += 4) ++histo[values[i]];
    }
  }
  {
    const int left_over = tile_width & 15;
    if (left_over > 0) {
      VP8LCollectColorBlueTransforms_SSE(argb + tile_width - left_over, stride,
                                         left_over, tile_height, green_to_blue,
                                         red_to_blue, histo);
    }
  }
}

static void CollectColorRedTransforms_AVX512(
    const uint32_t* WEBP_RESTRICT argb, int stride, int tile_width,
    int tile_height, int green_to_red, uint32_t histo[]) {
  const __m512i mult = MK_CST_16(0, CST_5b(green_to_red));
  const __m512i mask_g = _mm512_set1_epi32(0x0000ff00);
  if (tile_width >= 16) {
    int y, i;
    for (y = 0; y < tile_height; ++y) {
      uint8_t values[64];
      const uint32_t* const src = argb + y * stride;
      const __m512i A1 = _mm512_loadu_si512((const __m512i*)src);
      const __m512i B1 = _mm512_and_si512(A1, mask_g);
      const __m512i C1 = _mm512_madd_epi16(B1, mult);
      __m512i D = _mm512_sub_epi16(A1, C1);
      int x;
      for (x = 16; x + 16 <= tile_width; x += 16) {
        const __m512i A2 = _mm512_loadu_si512((const __m512i*)(src + x));
        __m512i B2, C2;
        _mm512_storeu_si512((__m512i*)values, D);
        for (i = 2; i < 64; i += 4) ++histo[values[i]];
        B2 = _mm512_and_si512(A2, mask_g);
        C2 = _mm512_madd_epi16(B2, mult);
        D = _mm512_sub_epi16(A2, C2);
      }
      _mm512_storeu_si512((__m512i*)values, D);
      for (i = 2; i < 64; i += 4) ++histo[values[i]];
    }
  }
  {
    const int left_over = tile_width & 15;
    if (left_over > 0) {
      VP8LCollectColorRedTransforms_SSE(argb + tile_width - left_over, stride,
                                        left_over, tile_height, green_to_red,
                                        histo);
    }
  }
}
#undef SPAN
#undef MK_CST_16

//------------------------------------------------------------------------------

// Note we are adding uint32_t's as *signed* int32's (using _mm512_add_epi32).
// But that's ok since the histogram values are less than 1<<28.
static void AddVector_AVX512(const uint32_t* WEBP_RESTRICT a,
                             const uint32_t* WEBP_RESTRICT b,
                             uint32_t* WEBP_RESTRICT out, int size) {
  int i = 0;
  int aligned_size = size & ~63;
  assert(size >= 32);
  assert(size % 2 == 0);

  // Process 64 uint32_t per iteration (4 x 512-bit loads).
  while (i < aligned_size) {
    const __m512i a0 = _mm512_loadu_si512((const __m512i*)&a[i + 0]);
    const __m512i a1 = _mm512_loadu_si512((const __m512i*)&a[i + 16]);
    const __m512i a2 = _mm512_loadu_si512((const __m512i*)&a[i + 32]);
    const __m512i a3 = _mm512_loadu_si512((const __m512i*)&a[i + 48]);
    const __m512i b0 = _mm512_loadu_si512((const __m512i*)&b[i + 0]);
    const __m512i b1 = _mm512_loadu_si512((const __m512i*)&b[i + 16]);
    const __m512i b2 = _mm512_loadu_si512((const __m512i*)&b[i + 32]);
    const __m512i b3 = _mm512_loadu_si512((const __m512i*)&b[i + 48]);
    _mm512_storeu_si512((__m512i*)&out[i + 0], _mm512_add_epi32(a0, b0));
    _mm512_storeu_si512((__m512i*)&out[i + 16], _mm512_add_epi32(a1, b1));
    _mm512_storeu_si512((__m512i*)&out[i + 32], _mm512_add_epi32(a2, b2));
    _mm512_storeu_si512((__m512i*)&out[i + 48], _mm512_add_epi32(a3, b3));
    i += 64;
  }

  if ((size & 32) != 0) {
    const __m512i a0 = _mm512_loadu_si512((const __m512i*)&a[i + 0]);
    const __m512i a1 = _mm512_loadu_si512((const __m512i*)&a[i + 16]);
    const __m512i b0 = _mm512_loadu_si512((const __m512i*)&b[i + 0]);
    const __m512i b1 = _mm512_loadu_si512((const __m512i*)&b[i + 16]);
    _mm512_storeu_si512((__m512i*)&out[i + 0], _mm512_add_epi32(a0, b0));
    _mm512_storeu_si512((__m512i*)&out[i + 16], _mm512_add_epi32(a1, b1));
    i += 32;
  }

  if ((size & 16) != 0) {
    const __m512i a0 = _mm512_loadu_si512((const __m512i*)&a[i]);
    const __m512i b0 = _mm512_loadu_si512((const __m512i*)&b[i]);
    _mm512_storeu_si512((__m512i*)&out[i], _mm512_add_epi32(a0, b0));
    i += 16;
  }

  size &= 15;
  for (; size > 0; --size, ++i) {
    out[i] = a[i] + b[i];
  }
}

static void AddVectorEq_AVX512(const uint32_t* WEBP_RESTRICT a,
                               uint32_t* WEBP_RESTRICT out, int size) {
  int i = 0;
  int aligned_size = size & ~63;
  assert(size >= 32);
  assert(size % 2 == 0);

  while (i < aligned_size) {
    const __m512i a0 = _mm512_loadu_si512((const __m512i*)&a[i + 0]);
    const __m512i a1 = _mm512_loadu_si512((const __m512i*)&a[i + 16]);
    const __m512i a2 = _mm512_loadu_si512((const __m512i*)&a[i + 32]);
    const __m512i a3 = _mm512_loadu_si512((const __m512i*)&a[i + 48]);
    const __m512i b0 = _mm512_loadu_si512((const __m512i*)&out[i + 0]);
    const __m512i b1 = _mm512_loadu_si512((const __m512i*)&out[i + 16]);
    const __m512i b2 = _mm512_loadu_si512((const __m512i*)&out[i + 32]);
    const __m512i b3 = _mm512_loadu_si512((const __m512i*)&out[i + 48]);
    _mm512_storeu_si512((__m512i*)&out[i + 0], _mm512_add_epi32(a0, b0));
    _mm512_storeu_si512((__m512i*)&out[i + 16], _mm512_add_epi32(a1, b1));
    _mm512_storeu_si512((__m512i*)&out[i + 32], _mm512_add_epi32(a2, b2));
    _mm512_storeu_si512((__m512i*)&out[i + 48], _mm512_add_epi32(a3, b3));
    i += 64;
  }

  if ((size & 32) != 0) {
    const __m512i a0 = _mm512_loadu_si512((const __m512i*)&a[i + 0]);
    const __m512i a1 = _mm512_loadu_si512((const __m512i*)&a[i + 16]);
    const __m512i b0 = _mm512_loadu_si512((const __m512i*)&out[i + 0]);
    const __m512i b1 = _mm512_loadu_si512((const __m512i*)&out[i + 16]);
    _mm512_storeu_si512((__m512i*)&out[i + 0], _mm512_add_epi32(a0, b0));
    _mm512_storeu_si512((__m512i*)&out[i + 16], _mm512_add_epi32(a1, b1));
    i += 32;
  }

  if ((size & 16) != 0) {
    const __m512i a0 = _mm512_loadu_si512((const __m512i*)&a[i]);
    const __m512i b0 = _mm512_loadu_si512((const __m512i*)&out[i]);
    _mm512_storeu_si512((__m512i*)&out[i], _mm512_add_epi32(a0, b0));
    i += 16;
  }

  size &= 15;
  for (; size > 0; --size, ++i) {
    out[i] += a[i];
  }
}

//------------------------------------------------------------------------------
// Entropy

#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ)

static uint64_t CombinedShannonEntropy_AVX512(const uint32_t X[256],
                                              const uint32_t Y[256]) {
  int i;
  uint64_t retval = 0;
  uint32_t sumX = 0, sumXY = 0;
  const __m512i zero = _mm512_setzero_si512();

  // Process 16 uint32_t per mask group using AVX-512 native mask registers.
  // This avoids the pack-narrow-movemask chain that AVX2 requires.
  for (i = 0; i < 256; i += 16) {
    const __m512i xv = _mm512_loadu_si512((const __m512i*)(X + i));
    const __m512i yv = _mm512_loadu_si512((const __m512i*)(Y + i));
    const __mmask16 mx = _mm512_cmpneq_epi32_mask(xv, zero);
    uint32_t my = (uint32_t)(_mm512_cmpneq_epi32_mask(yv, zero) | mx);

    while (my) {
      const int32_t j = BitsCtz(my);
      uint32_t xy;
      if (((uint32_t)mx >> j) & 1) {
        const int x = X[i + j];
        sumXY += x;
        retval += VP8LFastSLog2(x);
      }
      xy = X[i + j] + Y[i + j];
      sumX += xy;
      retval += VP8LFastSLog2(xy);
      my &= my - 1;
    }
  }
  retval = VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY) - retval;
  return retval;
}

#else

#define DONT_USE_COMBINED_SHANNON_ENTROPY_AVX512_FUNC

#endif

//------------------------------------------------------------------------------

static int VectorMismatch_AVX512(const uint32_t* const array1,
                                 const uint32_t* const array2, int length) {
  int match_len;

  if (length >= 48) {
    __m512i A0 = _mm512_loadu_si512((const __m512i*)&array1[0]);
    __m512i A1 = _mm512_loadu_si512((const __m512i*)&array2[0]);
    match_len = 0;
    do {
      const __mmask16 cmpA = _mm512_cmpeq_epi32_mask(A0, A1);
      const __m512i B0 =
          _mm512_loadu_si512((const __m512i*)&array1[match_len + 16]);
      const __m512i B1 =
          _mm512_loadu_si512((const __m512i*)&array2[match_len + 16]);
      if (cmpA != 0xFFFF) {
        match_len += BitsCtz(~(uint32_t)cmpA);
        goto end;
      }
      match_len += 16;

      {
        const __mmask16 cmpB = _mm512_cmpeq_epi32_mask(B0, B1);
        A0 = _mm512_loadu_si512((const __m512i*)&array1[match_len + 16]);
        A1 = _mm512_loadu_si512((const __m512i*)&array2[match_len + 16]);
        if (cmpB != 0xFFFF) {
          match_len += BitsCtz(~(uint32_t)cmpB);
          goto end;
        }
        match_len += 16;
      }
    } while (match_len + 48 < length);
  } else {
    match_len = 0;
    if (length >= 16) {
      const __mmask16 cmp = _mm512_cmpeq_epi32_mask(
          _mm512_loadu_si512((const __m512i*)&array1[0]),
          _mm512_loadu_si512((const __m512i*)&array2[0]));
      if (cmp == 0xFFFF) {
        match_len = 16;
        if (length >= 32) {
          const __mmask16 cmp2 = _mm512_cmpeq_epi32_mask(
              _mm512_loadu_si512((const __m512i*)&array1[16]),
              _mm512_loadu_si512((const __m512i*)&array2[16]));
          if (cmp2 == 0xFFFF) {
            match_len = 32;
          } else {
            match_len += BitsCtz(~(uint32_t)cmp2);
            goto end;
          }
        }
      } else {
        match_len = BitsCtz(~(uint32_t)cmp);
        goto end;
      }
    }
  }

end:
  while (match_len < length && array1[match_len] == array2[match_len]) {
    ++match_len;
  }
  return match_len;
}

// Bundles multiple (1, 2, 4 or 8) pixels into a single pixel.
static void BundleColorMap_AVX512(const uint8_t* WEBP_RESTRICT const row,
                                  int width, int xbits,
                                  uint32_t* WEBP_RESTRICT dst) {
  int x = 0;
  assert(xbits >= 0);
  assert(xbits <= 3);
  switch (xbits) {
    case 0: {
      // Store 0xff000000 | (row[x] << 8). Use vpmovzxbd to zero-extend
      // 16 bytes to 16 dwords, then shift and OR.
      const __m512i mask_or = _mm512_set1_epi32((int)0xff000000);
      for (x = 0; x + 16 <= width; x += 16, dst += 16) {
        const __m128i in = _mm_loadu_si128((const __m128i*)&row[x]);
        const __m512i wide = _mm512_cvtepu8_epi32(in);
        const __m512i shifted = _mm512_slli_epi32(wide, 8);
        _mm512_storeu_si512((__m512i*)dst,
                            _mm512_or_si512(shifted, mask_or));
      }
      break;
    }
    case 1: {
      // Pack pairs: 0xff000000 | ((row[2k] | row[2k+1]<<4) << 8).
      // Use vpmovzxwd to zero-extend 16-bit pairs to 32-bit, then
      // combine nibbles via multiply-and-mask.
      const __m256i mul = _mm256_set1_epi16(0x110);
      const __m256i ff = _mm256_set1_epi16((short)0xff00);
      for (x = 0; x + 32 <= width; x += 32, dst += 16) {
        const __m256i in = _mm256_loadu_si256((const __m256i*)&row[x]);
        const __m256i tmp = _mm256_mullo_epi16(in, mul);
        const __m256i pack = _mm256_and_si256(tmp, ff);
        // Zero-extend 16-bit to 32-bit using AVX-512 vpmovzxwd.
        const __m512i wide = _mm512_cvtepu16_epi32(pack);
        const __m512i result = _mm512_or_si512(wide,
                                  _mm512_set1_epi32((int)0xff000000));
        _mm512_storeu_si512((__m512i*)dst, result);
      }
      break;
    }
    case 2: {
      const __m512i mask_or = _mm512_set1_epi32((int)0xff000000);
      const __m512i mul_cst = _mm512_set1_epi16(0x0104);
      const __m512i mask_mul = _mm512_set1_epi16(0x0f00);
      for (x = 0; x + 64 <= width; x += 64, dst += 16) {
        const __m512i in = _mm512_loadu_si512((const __m512i*)&row[x]);
        const __m512i mul = _mm512_mullo_epi16(in, mul_cst);
        const __m512i tmp = _mm512_and_si512(mul, mask_mul);
        const __m512i shift = _mm512_srli_epi32(tmp, 12);
        const __m512i pack = _mm512_or_si512(shift, tmp);
        const __m512i res = _mm512_or_si512(pack, mask_or);
        _mm512_storeu_si512((__m512i*)dst, res);
      }
      break;
    }
    default: {
      assert(xbits == 3);
      for (x = 0; x + 64 <= width; x += 64, dst += 8) {
        const __m512i in = _mm512_loadu_si512((const __m512i*)&row[x]);
        const __m512i shift = _mm512_slli_epi64(in, 7);
        const uint64_t move = _mm512_movepi8_mask(shift);
        dst[0] = 0xff000000 | (((uint32_t)(move >>  0) & 0xff) << 8);
        dst[1] = 0xff000000 | (((uint32_t)(move >>  8) & 0xff) << 8);
        dst[2] = 0xff000000 | (((uint32_t)(move >> 16) & 0xff) << 8);
        dst[3] = 0xff000000 | (((uint32_t)(move >> 24) & 0xff) << 8);
        dst[4] = 0xff000000 | (((uint32_t)(move >> 32) & 0xff) << 8);
        dst[5] = 0xff000000 | (((uint32_t)(move >> 40) & 0xff) << 8);
        dst[6] = 0xff000000 | (((uint32_t)(move >> 48) & 0xff) << 8);
        dst[7] = 0xff000000 | (((uint32_t)(move >> 56) & 0xff) << 8);
      }
      break;
    }
  }
  if (x != width) {
    VP8LBundleColorMap_SSE(row + x, width - x, xbits, dst);
  }
}

//------------------------------------------------------------------------------
// Batch version of Predictor Transform subtraction

static WEBP_INLINE void Average2_m512i(const __m512i* const a0,
                                       const __m512i* const a1,
                                       __m512i* const avg) {
  // (a + b) >> 1 = ((a + b + 1) >> 1) - ((a ^ b) & 1)
  const __m512i ones = _mm512_set1_epi8(1);
  const __m512i avg1 = _mm512_avg_epu8(*a0, *a1);
  const __m512i one = _mm512_and_si512(_mm512_xor_si512(*a0, *a1), ones);
  *avg = _mm512_sub_epi8(avg1, one);
}

// Predictor0: ARGB_BLACK.
static void PredictorSub0_AVX512(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m512i black = _mm512_set1_epi32((int)ARGB_BLACK);
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    const __m512i res = _mm512_sub_epi8(src, black);
    _mm512_storeu_si512((__m512i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[0](in + i, NULL, num_pixels - i, out + i);
  }
  (void)upper;
}

#define GENERATE_PREDICTOR_1(X, IN)                                          \
  static void PredictorSub##X##_AVX512(                                      \
      const uint32_t* const in, const uint32_t* const upper, int num_pixels, \
      uint32_t* WEBP_RESTRICT const out) {                                   \
    int i;                                                                   \
    for (i = 0; i + 16 <= num_pixels; i += 16) {                             \
      const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);        \
      const __m512i pred = _mm512_loadu_si512((const __m512i*)&(IN));        \
      const __m512i res = _mm512_sub_epi8(src, pred);                        \
      _mm512_storeu_si512((__m512i*)&out[i], res);                           \
    }                                                                        \
    if (i != num_pixels) {                                                   \
      VP8LPredictorsSub_SSE[(X)](in + i, WEBP_OFFSET_PTR(upper, i),          \
                                 num_pixels - i, out + i);                   \
    }                                                                        \
  }

GENERATE_PREDICTOR_1(1, in[i - 1])       // Predictor1: L
GENERATE_PREDICTOR_1(2, upper[i])        // Predictor2: T
GENERATE_PREDICTOR_1(3, upper[i + 1])    // Predictor3: TR
GENERATE_PREDICTOR_1(4, upper[i - 1])    // Predictor4: TL
#undef GENERATE_PREDICTOR_1

// Predictor5: avg2(avg2(L, TR), T)
static void PredictorSub5_AVX512(const uint32_t* in, const uint32_t* upper,
                                 int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i L = _mm512_loadu_si512((const __m512i*)&in[i - 1]);
    const __m512i T = _mm512_loadu_si512((const __m512i*)&upper[i]);
    const __m512i TR = _mm512_loadu_si512((const __m512i*)&upper[i + 1]);
    const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    __m512i avg, pred, res;
    Average2_m512i(&L, &TR, &avg);
    Average2_m512i(&avg, &T, &pred);
    res = _mm512_sub_epi8(src, pred);
    _mm512_storeu_si512((__m512i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[5](in + i, upper + i, num_pixels - i, out + i);
  }
}

#define GENERATE_PREDICTOR_2(X, A, B)                                         \
  static void PredictorSub##X##_AVX512(const uint32_t* in,                    \
                                       const uint32_t* upper, int num_pixels, \
                                       uint32_t* WEBP_RESTRICT out) {         \
    int i;                                                                    \
    for (i = 0; i + 16 <= num_pixels; i += 16) {                              \
      const __m512i tA = _mm512_loadu_si512((const __m512i*)&(A));            \
      const __m512i tB = _mm512_loadu_si512((const __m512i*)&(B));            \
      const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);         \
      __m512i pred, res;                                                      \
      Average2_m512i(&tA, &tB, &pred);                                        \
      res = _mm512_sub_epi8(src, pred);                                       \
      _mm512_storeu_si512((__m512i*)&out[i], res);                            \
    }                                                                         \
    if (i != num_pixels) {                                                    \
      VP8LPredictorsSub_SSE[(X)](in + i, upper + i, num_pixels - i, out + i); \
    }                                                                         \
  }

GENERATE_PREDICTOR_2(6, in[i - 1], upper[i - 1])   // Predictor6: avg(L, TL)
GENERATE_PREDICTOR_2(7, in[i - 1], upper[i])        // Predictor7: avg(L, T)
GENERATE_PREDICTOR_2(8, upper[i - 1], upper[i])     // Predictor8: avg(TL, T)
GENERATE_PREDICTOR_2(9, upper[i], upper[i + 1])     // Predictor9: avg(T, TR)
#undef GENERATE_PREDICTOR_2

// Predictor10: avg(avg(L,TL), avg(T, TR)).
static void PredictorSub10_AVX512(const uint32_t* in, const uint32_t* upper,
                                  int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i L = _mm512_loadu_si512((const __m512i*)&in[i - 1]);
    const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    const __m512i TL = _mm512_loadu_si512((const __m512i*)&upper[i - 1]);
    const __m512i T = _mm512_loadu_si512((const __m512i*)&upper[i]);
    const __m512i TR = _mm512_loadu_si512((const __m512i*)&upper[i + 1]);
    __m512i avgTTR, avgLTL, avg, res;
    Average2_m512i(&T, &TR, &avgTTR);
    Average2_m512i(&L, &TL, &avgLTL);
    Average2_m512i(&avgTTR, &avgLTL, &avg);
    res = _mm512_sub_epi8(src, avg);
    _mm512_storeu_si512((__m512i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[10](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor11: select.
static void GetSumAbsDiff32_AVX512(const __m512i* const A,
                                   const __m512i* const B,
                                   __m512i* const out) {
  // Unpack pairs of 32-bit values to 64-bit for SAD computation.
  const __m512i A_lo = _mm512_unpacklo_epi32(*A, *A);
  const __m512i B_lo = _mm512_unpacklo_epi32(*B, *A);
  const __m512i A_hi = _mm512_unpackhi_epi32(*A, *A);
  const __m512i B_hi = _mm512_unpackhi_epi32(*B, *A);
  const __m512i s_lo = _mm512_sad_epu8(A_lo, B_lo);
  const __m512i s_hi = _mm512_sad_epu8(A_hi, B_hi);
  *out = _mm512_packs_epi32(s_lo, s_hi);
}

static void PredictorSub11_AVX512(const uint32_t* in, const uint32_t* upper,
                                  int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i L = _mm512_loadu_si512((const __m512i*)&in[i - 1]);
    const __m512i T = _mm512_loadu_si512((const __m512i*)&upper[i]);
    const __m512i TL = _mm512_loadu_si512((const __m512i*)&upper[i - 1]);
    const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    __m512i pa, pb;
    GetSumAbsDiff32_AVX512(&T, &TL, &pa);   // pa = sum |T-TL|
    GetSumAbsDiff32_AVX512(&L, &TL, &pb);   // pb = sum |L-TL|
    {
      // Use AVX-512 mask comparison: pb > pa -> select L, else T.
      const __mmask16 mask = _mm512_cmpgt_epi32_mask(pb, pa);
      const __m512i pred = _mm512_mask_blend_epi32(mask, T, L);
      const __m512i res = _mm512_sub_epi8(src, pred);
      _mm512_storeu_si512((__m512i*)&out[i], res);
    }
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[11](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor12: ClampedSubSubtractFull.
static void PredictorSub12_AVX512(const uint32_t* in, const uint32_t* upper,
                                  int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m512i zero = _mm512_setzero_si512();
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    const __m512i L = _mm512_loadu_si512((const __m512i*)&in[i - 1]);
    const __m512i L_lo = _mm512_unpacklo_epi8(L, zero);
    const __m512i L_hi = _mm512_unpackhi_epi8(L, zero);
    const __m512i T = _mm512_loadu_si512((const __m512i*)&upper[i]);
    const __m512i T_lo = _mm512_unpacklo_epi8(T, zero);
    const __m512i T_hi = _mm512_unpackhi_epi8(T, zero);
    const __m512i TL = _mm512_loadu_si512((const __m512i*)&upper[i - 1]);
    const __m512i TL_lo = _mm512_unpacklo_epi8(TL, zero);
    const __m512i TL_hi = _mm512_unpackhi_epi8(TL, zero);
    const __m512i diff_lo = _mm512_sub_epi16(T_lo, TL_lo);
    const __m512i diff_hi = _mm512_sub_epi16(T_hi, TL_hi);
    const __m512i pred_lo = _mm512_add_epi16(L_lo, diff_lo);
    const __m512i pred_hi = _mm512_add_epi16(L_hi, diff_hi);
    const __m512i pred = _mm512_packus_epi16(pred_lo, pred_hi);
    const __m512i res = _mm512_sub_epi8(src, pred);
    _mm512_storeu_si512((__m512i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[12](in + i, upper + i, num_pixels - i, out + i);
  }
}

// Predictor13: ClampedAddSubtractHalf.
static void PredictorSub13_AVX512(const uint32_t* in, const uint32_t* upper,
                                  int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  const __m512i zero = _mm512_setzero_si512();
  for (i = 0; i + 16 <= num_pixels; i += 16) {
    const __m512i L = _mm512_loadu_si512((const __m512i*)&in[i - 1]);
    const __m512i src = _mm512_loadu_si512((const __m512i*)&in[i]);
    const __m512i T = _mm512_loadu_si512((const __m512i*)&upper[i]);
    const __m512i TL = _mm512_loadu_si512((const __m512i*)&upper[i - 1]);
    // lo half.
    const __m512i L_lo = _mm512_unpacklo_epi8(L, zero);
    const __m512i T_lo = _mm512_unpacklo_epi8(T, zero);
    const __m512i TL_lo = _mm512_unpacklo_epi8(TL, zero);
    const __m512i sum_lo = _mm512_add_epi16(T_lo, L_lo);
    const __m512i avg_lo = _mm512_srli_epi16(sum_lo, 1);
    const __m512i A1_lo = _mm512_sub_epi16(avg_lo, TL_lo);
    const __mmask32 bit_fix_mask_lo = _mm512_cmpgt_epi16_mask(TL_lo, avg_lo);
    const __m512i A2_lo = _mm512_mask_sub_epi16(A1_lo, bit_fix_mask_lo, A1_lo,
                                                _mm512_set1_epi16(-1));
    const __m512i A3_lo = _mm512_srai_epi16(A2_lo, 1);
    const __m512i A4_lo = _mm512_add_epi16(avg_lo, A3_lo);
    // hi half.
    const __m512i L_hi = _mm512_unpackhi_epi8(L, zero);
    const __m512i T_hi = _mm512_unpackhi_epi8(T, zero);
    const __m512i TL_hi = _mm512_unpackhi_epi8(TL, zero);
    const __m512i sum_hi = _mm512_add_epi16(T_hi, L_hi);
    const __m512i avg_hi = _mm512_srli_epi16(sum_hi, 1);
    const __m512i A1_hi = _mm512_sub_epi16(avg_hi, TL_hi);
    const __mmask32 bit_fix_mask_hi = _mm512_cmpgt_epi16_mask(TL_hi, avg_hi);
    const __m512i A2_hi = _mm512_mask_sub_epi16(A1_hi, bit_fix_mask_hi, A1_hi,
                                                _mm512_set1_epi16(-1));
    const __m512i A3_hi = _mm512_srai_epi16(A2_hi, 1);
    const __m512i A4_hi = _mm512_add_epi16(avg_hi, A3_hi);

    const __m512i pred = _mm512_packus_epi16(A4_lo, A4_hi);
    const __m512i res = _mm512_sub_epi8(src, pred);
    _mm512_storeu_si512((__m512i*)&out[i], res);
  }
  if (i != num_pixels) {
    VP8LPredictorsSub_SSE[13](in + i, upper + i, num_pixels - i, out + i);
  }
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LEncDspInitAVX512(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInitAVX512(void) {
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_AVX512;
  VP8LTransformColor = TransformColor_AVX512;
  VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_AVX512;
  VP8LCollectColorRedTransforms = CollectColorRedTransforms_AVX512;
  VP8LAddVector = AddVector_AVX512;
  VP8LAddVectorEq = AddVectorEq_AVX512;
#if !defined(DONT_USE_COMBINED_SHANNON_ENTROPY_AVX512_FUNC)
  VP8LCombinedShannonEntropy = CombinedShannonEntropy_AVX512;
#endif
  VP8LVectorMismatch = VectorMismatch_AVX512;
  VP8LBundleColorMap = BundleColorMap_AVX512;

  VP8LPredictorsSub[0] = PredictorSub0_AVX512;
  VP8LPredictorsSub[1] = PredictorSub1_AVX512;
  VP8LPredictorsSub[2] = PredictorSub2_AVX512;
  VP8LPredictorsSub[3] = PredictorSub3_AVX512;
  VP8LPredictorsSub[4] = PredictorSub4_AVX512;
  VP8LPredictorsSub[5] = PredictorSub5_AVX512;
  VP8LPredictorsSub[6] = PredictorSub6_AVX512;
  VP8LPredictorsSub[7] = PredictorSub7_AVX512;
  VP8LPredictorsSub[8] = PredictorSub8_AVX512;
  VP8LPredictorsSub[9] = PredictorSub9_AVX512;
  VP8LPredictorsSub[10] = PredictorSub10_AVX512;
  VP8LPredictorsSub[11] = PredictorSub11_AVX512;
  VP8LPredictorsSub[12] = PredictorSub12_AVX512;
  VP8LPredictorsSub[13] = PredictorSub13_AVX512;
  VP8LPredictorsSub[14] = PredictorSub0_AVX512;  // <- security sentinels
  VP8LPredictorsSub[15] = PredictorSub0_AVX512;
}

#else  // !WEBP_USE_AVX512

WEBP_DSP_INIT_STUB(VP8LEncDspInitAVX512)

#endif  // WEBP_USE_AVX512
