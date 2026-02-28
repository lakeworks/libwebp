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
// Helper: extract a specific byte from each 32-bit element of a 512-bit
// register and increment histogram entries.
//
// Strategy: use vpshufb (AVX-512BW, already required) within each 128-bit
// lane to gather the target byte from each of 4 dwords into lane positions
// 0-3. Then extract 4 x __m128i lanes and use _mm_extract_epi8 for indices.
// This avoids the store-forwarding stall (~10 cycles per load) that occurs
// when a 64-byte SIMD store is followed by 16 scalar byte loads.
//
// On Zen 5: vpshufb zmm = 1 cycle, 4x vextracti32x4 = 1 cycle each.
// Total: ~5 cycles + 16 extract ops vs old 64B store + 16 stalled loads.
static WEBP_INLINE void UpdateHisto16_AVX512(const __m512i V,
                                             const __m512i lane_shuf,
                                             uint32_t histo[]) {
  // Gather target byte of each dword to positions 0,1,2,3 within each
  // 128-bit lane.  The shuffle pattern is caller-provided so Blue (byte 0)
  // and Red (byte 2) share this helper.
  const __m512i packed = _mm512_shuffle_epi8(V, lane_shuf);
  const __m128i q0 = _mm512_castsi512_si128(packed);
  const __m128i q1 = _mm512_extracti32x4_epi32(packed, 1);
  const __m128i q2 = _mm512_extracti32x4_epi32(packed, 2);
  const __m128i q3 = _mm512_extracti32x4_epi32(packed, 3);
  ++histo[(uint8_t)_mm_extract_epi8(q0, 0)];
  ++histo[(uint8_t)_mm_extract_epi8(q0, 1)];
  ++histo[(uint8_t)_mm_extract_epi8(q0, 2)];
  ++histo[(uint8_t)_mm_extract_epi8(q0, 3)];
  ++histo[(uint8_t)_mm_extract_epi8(q1, 0)];
  ++histo[(uint8_t)_mm_extract_epi8(q1, 1)];
  ++histo[(uint8_t)_mm_extract_epi8(q1, 2)];
  ++histo[(uint8_t)_mm_extract_epi8(q1, 3)];
  ++histo[(uint8_t)_mm_extract_epi8(q2, 0)];
  ++histo[(uint8_t)_mm_extract_epi8(q2, 1)];
  ++histo[(uint8_t)_mm_extract_epi8(q2, 2)];
  ++histo[(uint8_t)_mm_extract_epi8(q2, 3)];
  ++histo[(uint8_t)_mm_extract_epi8(q3, 0)];
  ++histo[(uint8_t)_mm_extract_epi8(q3, 1)];
  ++histo[(uint8_t)_mm_extract_epi8(q3, 2)];
  ++histo[(uint8_t)_mm_extract_epi8(q3, 3)];
}

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
  // Byte 0 of each 32-bit element -> lane positions 0,1,2,3 (within 128-bit).
  // Per 128-bit lane with dwords [D0, D1, D2, D3] at bytes [0..3, 4..7, 8..11,
  // 12..15]: want byte0=pos0, byte4=pos1, byte8=pos2, byte12=pos3.
  const __m512i blue_shuf = _mm512_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12,  8,  4,  0,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12,  8,  4,  0,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12,  8,  4,  0,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12,  8,  4,  0);
  if (tile_width >= 16) {
    int y;
    for (y = 0; y < tile_height; ++y) {
      const uint32_t* const src = argb + y * stride;
      int x;
      for (x = 0; x + 16 <= tile_width; x += 16) {
        const __m512i A = _mm512_loadu_si512((const __m512i*)(src + x));
        const __m512i B = _mm512_shuffle_epi8(A, perm);
        const __m512i C = _mm512_mulhi_epi16(B, mult);
        const __m512i D = _mm512_sub_epi16(A, C);
        const __m512i E = _mm512_add_epi16(_mm512_srli_epi32(D, 16), D);
        UpdateHisto16_AVX512(E, blue_shuf, histo);
      }
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
  // Byte 2 of each dword -> lane positions 0,1,2,3 within each 128-bit lane.
  const __m512i red_shuf = _mm512_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 10,  6,  2,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 10,  6,  2,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 10,  6,  2,
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 10,  6,  2);
  if (tile_width >= 16) {
    int y;
    for (y = 0; y < tile_height; ++y) {
      const uint32_t* const src = argb + y * stride;
      int x;
      for (x = 0; x + 16 <= tile_width; x += 16) {
        const __m512i A = _mm512_loadu_si512((const __m512i*)(src + x));
        const __m512i B = _mm512_and_si512(A, mask_g);
        const __m512i C = _mm512_madd_epi16(B, mult);
        const __m512i D = _mm512_sub_epi16(A, C);
        UpdateHisto16_AVX512(D, red_shuf, histo);
      }
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

//------------------------------------------------------------------------------
// Fully vectorized v*log2(v) using AVX-512CD (vplzcntd) + kLog2Table gather.
//
// Computes FastSLog2 for 8 uint32 values expanded to 8 uint64 results.
// Values 0..255: gathered from kSLog2Table[256] (precomputed uint64_t).
// Values 256..65535: vplzcntd + kLog2Table[256] gather + fixed-point multiply.
//   log_cnt = floor(log2(v)) - 7
//   v_norm = v >> log_cnt     (range [128..255])
//   result = v * (kLog2Table[v_norm] + (log_cnt << 23))
//          + LOG_2_RECIPROCAL_FIXED * (v & ((1 << log_cnt) - 1))
// All inputs must be < 65536. 'nz_mask' marks truly nonzero lanes;
// zero lanes are forced to 1 by the caller and masked out in the result.
//
// On Zen 5: ~20 uops per call (8 elements), vs ~120 uops for 8 scalar calls.
static WEBP_INLINE __m512i FastSLog2_8x_AVX512(const __m256i v32,
                                                const __mmask8 nz_mask) {
  // Zero-extend 8 x uint32 to 8 x uint64 for final arithmetic
  const __m512i v = _mm512_cvtepu32_epi64(v32);

  // --- Small path: v in [1..255], direct kSLog2Table lookup ---
  const __m512i k255_64 = _mm512_set1_epi64(255);
  const __mmask8 is_small = _mm512_cmple_epu64_mask(v, k255_64) & nz_mask;
  // Gather 64-bit values from kSLog2Table. Scale=8 (sizeof(uint64_t)).
  const __m512i slog_tbl = _mm512_mask_i64gather_epi64(
      _mm512_setzero_si512(), is_small, v, (const void*)kSLog2Table, 8);

  // --- Large path: v in [256..65535], vectorized log2 ---
  const __mmask8 is_large = (~is_small) & nz_mask;
  if (is_large == 0) return slog_tbl;

  // lzcnt on 32-bit values, then floor_log2 = 31 - lzcnt
  const __m256i lzcnt = _mm256_lzcnt_epi32(v32);
  const __m256i floor_log2 = _mm256_sub_epi32(_mm256_set1_epi32(31), lzcnt);
  const __m256i log_cnt = _mm256_sub_epi32(floor_log2, _mm256_set1_epi32(7));

  // v_norm = v >> log_cnt  (brings v into [128..255] for table lookup)
  const __m256i v_norm = _mm256_srlv_epi32(v32, log_cnt);

  // Gather log2 fractional bits: kLog2Table[v_norm], scale=4 (uint32_t)
  // v_norm values are in [128..255] for 'is_large' lanes, and may be arbitrary
  // for other lanes. Since kLog2Table[256] is the full table, all v_norm values
  // in [0..255] are safe indices. For lanes where is_large=0, v_norm could be
  // out of range (e.g., v=1 -> lzcnt=31, log_cnt=24, v_norm=0), but
  // kLog2Table[0]=0 which is harmless. The result is only used for is_large lanes.
  const __m256i log2_frac = _mm256_i32gather_epi32(
      (const int*)kLog2Table, v_norm, 4);

  // log2_fixed = log2_frac + (log_cnt << LOG_2_PRECISION_BITS)
  const __m256i log2_fixed = _mm256_add_epi32(
      log2_frac, _mm256_slli_epi32(log_cnt, LOG_2_PRECISION_BITS));

  // correction = LOG_2_RECIPROCAL_FIXED * (v & ((1 << log_cnt) - 1))
  // v_frac < 256, LOG_2_RECIPROCAL_FIXED ~= 12.1M, product < 3.1G (fits 32b)
  const __m256i y = _mm256_sllv_epi32(_mm256_set1_epi32(1), log_cnt);
  const __m256i v_frac = _mm256_and_si256(v32,
                                           _mm256_sub_epi32(y,
                                               _mm256_set1_epi32(1)));
  const __m256i corr32 = _mm256_mullo_epi32(
      _mm256_set1_epi32((int)LOG_2_RECIPROCAL_FIXED), v_frac);

  // result = v * log2_fixed + correction, all in 64-bit
  // v < 65536 (16 bits), log2_fixed < 2^28 => product < 44 bits, fits uint64
  const __m512i log2_fixed_64 = _mm512_cvtepu32_epi64(log2_fixed);
  const __m512i corr_64 = _mm512_cvtepu32_epi64(corr32);
  // _mm512_mul_epu32 multiplies low 32 bits of each 64-bit lane -> 64-bit result
  // Both operands have zeros in the high 32 bits (from cvtepu32), so this is exact.
  const __m512i prod = _mm512_mul_epu32(v, log2_fixed_64);
  const __m512i result_large = _mm512_add_epi64(prod, corr_64);

  // Blend: small from table lookup, large from computed
  return _mm512_mask_blend_epi64(is_large, slog_tbl, result_large);
}

// Fully vectorized CombinedShannonEntropy: processes 16 histogram bins per
// iteration, computing v*log2(v) entirely in SIMD (no scalar VP8LFastSLog2).
// Values >= 65536 fall back to scalar (extremely rare -- histogram bins rarely
// exceed 65536 even for 4k images). Requires AVX-512CD for vplzcntd.
//
// Expected speedup vs AVX2: ~3-5x on Zen 5 (AVX2 does scalar VP8LFastSLog2
// for every nonzero element; we do a fixed-cost SIMD computation per 8 values).
static uint64_t CombinedShannonEntropy_AVX512(const uint32_t X[256],
                                              const uint32_t Y[256]) {
  int i;
  // Four 8 x uint64 accumulators (rotating to hide latency)
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  __m512i acc2 = _mm512_setzero_si512();
  __m512i acc3 = _mm512_setzero_si512();
  __m512i sum_x_vec = _mm512_setzero_si512();   // 16 x uint32 sum of x
  __m512i sum_xy_vec = _mm512_setzero_si512();  // 16 x uint32 sum of xy
  const __m512i zero = _mm512_setzero_si512();
  const __m512i k65536 = _mm512_set1_epi32(65536);
  const __m256i one256 = _mm256_set1_epi32(1);
  uint64_t retval_overflow = 0;

  for (i = 0; i < 256; i += 16) {
    const __m512i xv = _mm512_loadu_si512((const __m512i*)(X + i));
    const __m512i yv = _mm512_loadu_si512((const __m512i*)(Y + i));
    const __m512i xyv = _mm512_add_epi32(xv, yv);

    // Nonzero masks
    const __mmask16 mx = _mm512_cmpneq_epi32_mask(xv, zero);
    const __mmask16 mxy = _mm512_cmpneq_epi32_mask(xyv, zero);

    // Accumulate 32-bit sums (only for nonzero entries, matching C ref)
    sum_x_vec = _mm512_mask_add_epi32(sum_x_vec, mx, sum_x_vec, xv);
    sum_xy_vec = _mm512_mask_add_epi32(sum_xy_vec, mxy, sum_xy_vec, xyv);

    // Check for overflow (values >= 65536) -- scalar fallback
    const __mmask16 x_of = _mm512_cmp_epu32_mask(xv, k65536, _MM_CMPINT_NLT);
    const __mmask16 xy_of = _mm512_cmp_epu32_mask(xyv, k65536, _MM_CMPINT_NLT);
    if (x_of) {
      uint32_t m = (uint32_t)x_of;
      while (m) {
        retval_overflow += VP8LFastSLog2(X[i + BitsCtz(m)]);
        m &= m - 1;
      }
    }
    if (xy_of) {
      uint32_t m = (uint32_t)xy_of;
      while (m) {
        const int j = BitsCtz(m);
        retval_overflow += VP8LFastSLog2(X[i + j] + Y[i + j]);
        m &= m - 1;
      }
    }

    // Vectorized path for values < 65536
    {
      const __mmask16 mx_safe = mx & ~x_of;
      const __mmask16 mxy_safe = mxy & ~xy_of;
      // Low 8 elements
      const __mmask8 mx_lo = (__mmask8)(mx_safe & 0xFF);
      const __mmask8 mxy_lo = (__mmask8)(mxy_safe & 0xFF);
      if (mx_lo) {
        const __m256i xv_lo = _mm256_mask_blend_epi32(
            mx_lo, one256, _mm512_castsi512_si256(xv));
        acc0 = _mm512_add_epi64(acc0, FastSLog2_8x_AVX512(xv_lo, mx_lo));
      }
      if (mxy_lo) {
        const __m256i xyv_lo = _mm256_mask_blend_epi32(
            mxy_lo, one256, _mm512_castsi512_si256(xyv));
        acc1 = _mm512_add_epi64(acc1, FastSLog2_8x_AVX512(xyv_lo, mxy_lo));
      }
      // High 8 elements
      {
        const __mmask8 mx_hi = (__mmask8)((mx_safe >> 8) & 0xFF);
        const __mmask8 mxy_hi = (__mmask8)((mxy_safe >> 8) & 0xFF);
        if (mx_hi) {
          const __m256i xv_hi = _mm256_mask_blend_epi32(
              mx_hi, one256, _mm512_extracti32x8_epi32(xv, 1));
          acc2 = _mm512_add_epi64(acc2, FastSLog2_8x_AVX512(xv_hi, mx_hi));
        }
        if (mxy_hi) {
          const __m256i xyv_hi = _mm256_mask_blend_epi32(
              mxy_hi, one256, _mm512_extracti32x8_epi32(xyv, 1));
          acc3 = _mm512_add_epi64(acc3, FastSLog2_8x_AVX512(xyv_hi, mxy_hi));
        }
      }
    }
  }

  // Horizontal reduction of 8 x uint64 accumulators -> scalar
  {
    const __m512i acc_total = _mm512_add_epi64(
        _mm512_add_epi64(acc0, acc1), _mm512_add_epi64(acc2, acc3));
    // Reduce 8 uint64 -> 4 -> 2 -> 1
    const __m256i acc_lo = _mm512_castsi512_si256(acc_total);
    const __m256i acc_hi = _mm512_extracti64x4_epi64(acc_total, 1);
    const __m256i acc_4 = _mm256_add_epi64(acc_lo, acc_hi);
    const __m128i acc_2lo = _mm256_castsi256_si128(acc_4);
    const __m128i acc_2hi = _mm256_extracti128_si256(acc_4, 1);
    const __m128i acc_2 = _mm_add_epi64(acc_2lo, acc_2hi);
    const __m128i acc_1 = _mm_add_epi64(acc_2, _mm_srli_si128(acc_2, 8));
    uint64_t retval = (uint64_t)_mm_cvtsi128_si64(acc_1) + retval_overflow;

    // Reduce 16 x uint32 sums -> scalar
    // Fold 512 -> 256 -> 128, then hadd within 128
    const __m256i sx_lo = _mm512_castsi512_si256(sum_x_vec);
    const __m256i sx_hi = _mm512_extracti32x8_epi32(sum_x_vec, 1);
    const __m256i sx_8 = _mm256_add_epi32(sx_lo, sx_hi);
    const __m128i sx_4lo = _mm256_castsi256_si128(sx_8);
    const __m128i sx_4hi = _mm256_extracti128_si256(sx_8, 1);
    const __m128i sx_4 = _mm_add_epi32(sx_4lo, sx_4hi);
    const __m128i sx_2 = _mm_add_epi32(sx_4, _mm_srli_si128(sx_4, 8));
    const __m128i sx_1 = _mm_add_epi32(sx_2, _mm_srli_si128(sx_2, 4));
    const uint32_t sumX = (uint32_t)_mm_cvtsi128_si32(sx_1);

    const __m256i sxy_lo = _mm512_castsi512_si256(sum_xy_vec);
    const __m256i sxy_hi = _mm512_extracti32x8_epi32(sum_xy_vec, 1);
    const __m256i sxy_8 = _mm256_add_epi32(sxy_lo, sxy_hi);
    const __m128i sxy_4lo = _mm256_castsi256_si128(sxy_8);
    const __m128i sxy_4hi = _mm256_extracti128_si256(sxy_8, 1);
    const __m128i sxy_4 = _mm_add_epi32(sxy_4lo, sxy_4hi);
    const __m128i sxy_2 = _mm_add_epi32(sxy_4, _mm_srli_si128(sxy_4, 8));
    const __m128i sxy_1 = _mm_add_epi32(sxy_2, _mm_srli_si128(sxy_2, 4));
    const uint32_t sumXY = (uint32_t)_mm_cvtsi128_si32(sxy_1);

    retval = VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY) - retval;
    return retval;
  }
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
      // Vectorize the bit-packing: extract MSBs into a 64-bit mask via
      // vpmovb2m, then widen the 8 mask bytes to 8 uint32 values using
      // vpmovzxbd + shift + OR. Avoids 8 scalar shift+mask+store ops.
      const __m256i mask_or_256 = _mm256_set1_epi32((int)0xff000000);
      for (x = 0; x + 64 <= width; x += 64, dst += 8) {
        const __m512i in = _mm512_loadu_si512((const __m512i*)&row[x]);
        const __m512i shift = _mm512_slli_epi64(in, 7);
        const uint64_t move = _mm512_movepi8_mask(shift);
        const __m128i move_bytes = _mm_set_epi64x(0, (long long)move);
        const __m256i wide = _mm256_cvtepu8_epi32(move_bytes);
        const __m256i shifted = _mm256_slli_epi32(wide, 8);
        const __m256i result = _mm256_or_si256(shifted, mask_or_256);
        _mm256_storeu_si256((__m256i*)dst, result);
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
  // Simple byte-parallel transforms: clear 512-bit wins (16 px/iter vs 8).
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed_AVX512;
  VP8LTransformColor = TransformColor_AVX512;
  VP8LBundleColorMap = BundleColorMap_AVX512;

  // CollectColor transforms: uses vpshufb (AVX-512BW) to extract histogram
  // indices without store-forwarding stalls. Zen 5 vpshufb is ~1 cycle vs
  // ~160 cycles for 16 store-forwarding stalls from 64B store -> byte loads.
  VP8LCollectColorBlueTransforms = CollectColorBlueTransforms_AVX512;
  VP8LCollectColorRedTransforms = CollectColorRedTransforms_AVX512;

  // Trivially parallel predictors: no cross-element dependencies.
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
  VP8LPredictorsSub[14] = PredictorSub0_AVX512;  // security sentinels
  VP8LPredictorsSub[15] = PredictorSub0_AVX512;

  // Encoder PredictorSub[11,12,13] are genuinely parallel (read from input,
  // not output), unlike their decoder counterparts. 16 px/iter vs 8.
  VP8LPredictorsSub[11] = PredictorSub11_AVX512;
  VP8LPredictorsSub[12] = PredictorSub12_AVX512;
  VP8LPredictorsSub[13] = PredictorSub13_AVX512;

  // Fully vectorized CombinedShannonEntropy: uses vplzcntd (AVX-512CD) +
  // kLog2Table gather to compute v*log2(v) entirely in SIMD, eliminating
  // all scalar VP8LFastSLog2 calls. ~3-5x faster than AVX2 version.
#if !defined(DONT_USE_COMBINED_SHANNON_ENTROPY_AVX512_FUNC)
  VP8LCombinedShannonEntropy = CombinedShannonEntropy_AVX512;
#endif

  // Left at AVX2 -- VectorMismatch: marginal benefit, early-exit dominated.
  // Left at AVX2 -- AddVector/AddVectorEq: marginal for typical sizes
  //   (40-536 elements); memory-bound, identical throughput at 256/512-bit.
}

#else  // !WEBP_USE_AVX512

WEBP_DSP_INIT_STUB(VP8LEncDspInitAVX512)

#endif  // WEBP_USE_AVX512
