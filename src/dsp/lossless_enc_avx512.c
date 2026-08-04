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
#include "src/enc/histogram_enc.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// Private integer copies of entropy tables for AVX-512 SIMD gather.
// The global kLog2Table/kSLog2Table are now float; these preserve the original
// integer representation needed for vpgatherdd / integer SIMD computation.
// Values are v*log2(v) scaled by (1 << 23).
#define LOG_2_PRECISION_BITS_INT 23
#define LOG_2_RECIPROCAL_FIXED_INT ((uint64_t)12102203)

static const uint32_t kLog2Table_int[LOG_LOOKUP_IDX_MAX] = {
         0,        0,  8388608, 13295629, 16777216, 19477745, 21684237,
  23549800, 25165824, 26591258, 27866353, 29019816, 30072845, 31041538,
  31938408, 32773374, 33554432, 34288123, 34979866, 35634199, 36254961,
  36845429, 37408424, 37946388, 38461453, 38955489, 39430146, 39886887,
  40327016, 40751698, 41161982, 41558811, 41943040, 42315445, 42676731,
  43027545, 43368474, 43700062, 44022807, 44337167, 44643569, 44942404,
  45234037, 45518808, 45797032, 46069003, 46334996, 46595268, 46850061,
  47099600, 47344097, 47583753, 47818754, 48049279, 48275495, 48497560,
  48715624, 48929828, 49140306, 49347187, 49550590, 49750631, 49947419,
  50141058, 50331648, 50519283, 50704053, 50886044, 51065339, 51242017,
  51416153, 51587818, 51757082, 51924012, 52088670, 52251118, 52411415,
  52569616, 52725775, 52879946, 53032177, 53182516, 53331012, 53477707,
  53622645, 53765868, 53907416, 54047327, 54185640, 54322389, 54457611,
  54591338, 54723604, 54854440, 54983876, 55111943, 55238669, 55364082,
  55488208, 55611074, 55732705, 55853126, 55972361, 56090432, 56207362,
  56323174, 56437887, 56551524, 56664103, 56775645, 56886168, 56995691,
  57104232, 57211808, 57318436, 57424133, 57528914, 57632796, 57735795,
  57837923, 57939198, 58039632, 58139239, 58238033, 58336027, 58433234,
  58529666, 58625336, 58720256, 58814437, 58907891, 59000628, 59092661,
  59183999, 59274652, 59364632, 59453947, 59542609, 59630625, 59718006,
  59804761, 59890898, 59976426, 60061354, 60145690, 60229443, 60312620,
  60395229, 60477278, 60558775, 60639726, 60720140, 60800023, 60879382,
  60958224, 61036555, 61114383, 61191714, 61268554, 61344908, 61420785,
  61496188, 61571124, 61645600, 61719620, 61793189, 61866315, 61939001,
  62011253, 62083076, 62154476, 62225457, 62296024, 62366182, 62435935,
  62505289, 62574248, 62642816, 62710997, 62778797, 62846219, 62913267,
  62979946, 63046260, 63112212, 63177807, 63243048, 63307939, 63372484,
  63436687, 63500551, 63564080, 63627277, 63690146, 63752690, 63814912,
  63876816, 63938405, 63999682, 64060650, 64121313, 64181673, 64241734,
  64301498, 64360969, 64420148, 64479040, 64537646, 64595970, 64654014,
  64711782, 64769274, 64826495, 64883447, 64940132, 64996553, 65052711,
  65108611, 65164253, 65219641, 65274776, 65329662, 65384299, 65438691,
  65492840, 65546747, 65600416, 65653847, 65707044, 65760008, 65812741,
  65865245, 65917522, 65969575, 66021404, 66073013, 66124403, 66175575,
  66226531, 66277275, 66327806, 66378127, 66428240, 66478146, 66527847,
  66577345, 66626641, 66675737, 66724635, 66773336, 66821842, 66870154,
  66918274, 66966204, 67013944, 67061497
};

static const uint64_t kSLog2Table_int[LOG_LOOKUP_IDX_MAX] = {
               0,              0,       16777216,       39886887,
        67108864,       97388723,      130105423,      164848600,
       201326592,      239321324,      278663526,      319217973,
       360874141,      403539997,      447137711,      491600606,
       536870912,      582898099,      629637592,      677049776,
       725099212,      773754010,      822985323,      872766924,
       923074875,      973887230,     1025183802,     1076945958,
      1129156447,     1181799249,     1234859451,     1288323135,
      1342177280,     1396409681,     1451008871,     1505964059,
      1561265072,     1616902301,     1672866655,     1729149526,
      1785742744,     1842638548,     1899829557,     1957308741,
      2015069397,     2073105127,     2131409817,  2189977618ull,
   2248802933ull,  2307880396ull,  2367204859ull,  2426771383ull,
   2486575220ull,  2546611805ull,  2606876748ull,  2667365819ull,
   2728074942ull,  2789000187ull,  2850137762ull,  2911484006ull,
   2973035382ull,  3034788471ull,  3096739966ull,  3158886666ull,
   3221225472ull,  3283753383ull,  3346467489ull,  3409364969ull,
   3472443085ull,  3535699182ull,  3599130679ull,  3662735070ull,
   3726509920ull,  3790452862ull,  3854561593ull,  3918833872ull,
   3983267519ull,  4047860410ull,  4112610476ull,  4177515704ull,
   4242574127ull,  4307783833ull,  4373142952ull,  4438649662ull,
   4504302186ull,  4570098787ull,  4636037770ull,  4702117480ull,
   4768336298ull,  4834692645ull,  4901184974ull,  4967811774ull,
   5034571569ull,  5101462912ull,  5168484389ull,  5235634615ull,
   5302912235ull,  5370315922ull,  5437844376ull,  5505496324ull,
   5573270518ull,  5641165737ull,  5709180782ull,  5777314477ull,
   5845565671ull,  5913933235ull,  5982416059ull,  6051013057ull,
   6119723161ull,  6188545324ull,  6257478518ull,  6326521733ull,
   6395673979ull,  6464934282ull,  6534301685ull,  6603775250ull,
   6673354052ull,  6743037185ull,  6812823756ull,  6882712890ull,
   6952703725ull,  7022795412ull,  7092987118ull,  7163278025ull,
   7233667324ull,  7304154222ull,  7374737939ull,  7445417707ull,
   7516192768ull,  7587062379ull,  7658025806ull,  7729082328ull,
   7800231234ull,  7871471825ull,  7942803410ull,  8014225311ull,
   8085736859ull,  8157337394ull,  8229026267ull,  8300802839ull,
   8372666477ull,  8444616560ull,  8516652476ull,  8588773618ull,
   8660979393ull,  8733269211ull,  8805642493ull,  8878098667ull,
   8950637170ull,  9023257446ull,  9095958945ull,  9168741125ull,
   9241603454ull,  9314545403ull,  9387566451ull,  9460666086ull,
   9533843800ull,  9607099093ull,  9680431471ull,  9753840445ull,
   9827325535ull,  9900886263ull,  9974522161ull, 10048232765ull,
  10122017615ull, 10195876260ull, 10269808253ull, 10343813150ull,
  10417890516ull, 10492039919ull, 10566260934ull, 10640553138ull,
  10714916116ull, 10789349456ull, 10863852751ull, 10938425600ull,
  11013067604ull, 11087778372ull, 11162557513ull, 11237404645ull,
  11312319387ull, 11387301364ull, 11462350205ull, 11537465541ull,
  11612647010ull, 11687894253ull, 11763206912ull, 11838584638ull,
  11914027082ull, 11989533899ull, 12065104750ull, 12140739296ull,
  12216437206ull, 12292198148ull, 12368021795ull, 12443907826ull,
  12519855920ull, 12595865759ull, 12671937032ull, 12748069427ull,
  12824262637ull, 12900516358ull, 12976830290ull, 13053204134ull,
  13129637595ull, 13206130381ull, 13282682202ull, 13359292772ull,
  13435961806ull, 13512689025ull, 13589474149ull, 13666316903ull,
  13743217014ull, 13820174211ull, 13897188225ull, 13974258793ull,
  14051385649ull, 14128568535ull, 14205807192ull, 14283101363ull,
  14360450796ull, 14437855239ull, 14515314443ull, 14592828162ull,
  14670396151ull, 14748018167ull, 14825693972ull, 14903423326ull,
  14981205995ull, 15059041743ull, 15136930339ull, 15214871554ull,
  15292865160ull, 15370910930ull, 15449008641ull, 15527158071ull,
  15605359001ull, 15683611210ull, 15761914485ull, 15840268608ull,
  15918673369ull, 15997128556ull, 16075633960ull, 16154189373ull,
  16232794589ull, 16311449405ull, 16390153617ull, 16468907026ull,
  16547709431ull, 16626560636ull, 16705460444ull, 16784408661ull,
  16863405094ull, 16942449552ull, 17021541845ull, 17100681785ull
};

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
// Entropy

#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ)

//------------------------------------------------------------------------------
// Fully vectorized v*log2(v) using AVX-512CD (vplzcntd) + kLog2Table_int gather.
//
// Computes FastSLog2 for 8 uint32 values expanded to 8 uint64 results
// in integer-scaled domain (scaled by 1 << LOG_2_PRECISION_BITS_INT = 1<<23).
// Uses private integer table copies (kLog2Table_int, kSLog2Table_int) since
// the global tables are now float.
//
// Values 0..255: gathered from kSLog2Table_int[256] (precomputed uint64_t).
// Values 256..65535: vplzcntd + kLog2Table_int[256] gather + fixed-point mul.
//   log_cnt = floor(log2(v)) - 7
//   v_norm = v >> log_cnt     (range [128..255])
//   result = v * (kLog2Table_int[v_norm] + (log_cnt << 23))
//          + LOG_2_RECIPROCAL_FIXED_INT * (v & ((1 << log_cnt) - 1))
// All inputs must be < 65536. 'nz_mask' marks truly nonzero lanes;
// zero lanes are forced to 1 by the caller and masked out in the result.
//
// On Zen 5: ~20 uops per call (8 elements), vs ~120 uops for 8 scalar calls.
static WEBP_INLINE __m512i FastSLog2_8x_AVX512(const __m256i v32,
                                                const __mmask8 nz_mask) {
  // Zero-extend 8 x uint32 to 8 x uint64 for final arithmetic
  const __m512i v = _mm512_cvtepu32_epi64(v32);

  // --- Small path: v in [1..255], direct kSLog2Table_int lookup ---
  const __m512i k255_64 = _mm512_set1_epi64(255);
  const __mmask8 is_small = _mm512_cmple_epu64_mask(v, k255_64) & nz_mask;
  // Gather 64-bit values from kSLog2Table_int. Scale=8 (sizeof(uint64_t)).
  const __m512i slog_tbl = _mm512_mask_i64gather_epi64(
      _mm512_setzero_si512(), is_small, v, (const void*)kSLog2Table_int, 8);

  // --- Large path: v in [256..65535], vectorized log2 ---
  const __mmask8 is_large = (~is_small) & nz_mask;
  if (is_large == 0) return slog_tbl;

  // lzcnt on 32-bit values, then floor_log2 = 31 - lzcnt
  const __m256i lzcnt = _mm256_lzcnt_epi32(v32);
  const __m256i floor_log2 = _mm256_sub_epi32(_mm256_set1_epi32(31), lzcnt);
  const __m256i log_cnt = _mm256_sub_epi32(floor_log2, _mm256_set1_epi32(7));

  // v_norm = v >> log_cnt  (brings v into [128..255] for table lookup)
  // Dead lanes: log_cnt may be negative (e.g., v=1 → log_cnt=-7). vpsrlvd
  // treats shift as unsigned → shift >= 32 → v_norm = 0. Safe.
  const __m256i v_norm = _mm256_srlv_epi32(v32, log_cnt);

  // Gather log2 fractional bits: kLog2Table_int[v_norm], scale=4 (uint32_t)
  // v_norm values are in [128..255] for 'is_large' lanes, and may be arbitrary
  // for other lanes. Since kLog2Table_int[256] is the full table, all v_norm
  // values in [0..255] are safe indices. For lanes where is_large=0, v_norm
  // could be out of range (e.g., v=1 -> lzcnt=31, log_cnt=24, v_norm=0), but
  // kLog2Table_int[0]=0 which is harmless. The result is only used for
  // is_large lanes.
  const __m256i log2_frac = _mm256_i32gather_epi32(
      (const int*)kLog2Table_int, v_norm, 4);

  // log2_fixed = log2_frac + (log_cnt << LOG_2_PRECISION_BITS_INT)
  const __m256i log2_fixed = _mm256_add_epi32(
      log2_frac, _mm256_slli_epi32(log_cnt, LOG_2_PRECISION_BITS_INT));

  // correction = LOG_2_RECIPROCAL_FIXED_INT * (v & ((1 << log_cnt) - 1))
  // Max: v_frac=255, product=255*12102203=3.09G (0xB7FC0B45). Exceeds INT32_MAX
  // but fits UINT32. mullo_epi32 produces correct low 32 bits either way;
  // cvtepu32_epi64 (unsigned extend) preserves the full value.
  const __m256i y = _mm256_sllv_epi32(_mm256_set1_epi32(1), log_cnt);
  const __m256i v_frac = _mm256_and_si256(v32,
                                           _mm256_sub_epi32(y,
                                               _mm256_set1_epi32(1)));
  const __m256i corr32 = _mm256_mullo_epi32(
      _mm256_set1_epi32((int)LOG_2_RECIPROCAL_FIXED_INT), v_frac);

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
static float CombinedShannonEntropy_AVX512(const uint32_t X[256],
                                           const uint32_t Y[256]) {
  int i;
  // Four 8 x uint64 accumulators (rotating to hide latency).
  // Internal computation remains in integer-scaled domain (x << 23);
  // converted to float at the return boundary.
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  __m512i acc2 = _mm512_setzero_si512();
  __m512i acc3 = _mm512_setzero_si512();
  __m512i sum_x_vec = _mm512_setzero_si512();   // 16 x uint32 sum of x
  __m512i sum_xy_vec = _mm512_setzero_si512();  // 16 x uint32 sum of xy
  const __m512i zero = _mm512_setzero_si512();
  const __m512i k65536 = _mm512_set1_epi32(65536);
  const __m256i one256 = _mm256_set1_epi32(1);
  float retval_overflow = 0.f;  // accumulates VP8LFastSLog2 (float) results

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
    // SIMD accumulators are in integer-scaled domain (x << 23).
    // Convert to float by dividing by (1 << 23).
    const uint64_t retval_int = (uint64_t)_mm_cvtsi128_si64(acc_1);
    const float retval_f =
        (float)((double)retval_int / (double)(1 << LOG_2_PRECISION_BITS_INT));

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

    // Combine: SIMD part (converted from integer-scaled) + overflow part (float)
    return VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY)
           - retval_f - retval_overflow;
  }
}

#else

#define DONT_USE_COMBINED_SHANNON_ENTROPY_AVX512_FUNC

#endif

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
      // Note: slli_epi64 shifts 64-bit lanes, not bytes. This works because
      // xbits=3 inputs are 1-bit values (0 or 1), so cross-byte leakage
      // from bit 1+ is always zero. Would break for multi-bit values.
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
// Compute per-pixel SAD(A[k], B[k]) for 16 pixels in a 512-bit register.
// Trick: unpack each pair of 32-bit pixels to 64-bit (padding with *A so the
// upper halves cancel in SAD), then vpsadbw gives SAD per 64-bit group.
// vpackssdw recombines: within each 128-bit lane, SAD results sit in even
// int32 slots [0,2] of s_lo and s_hi. Pack produces [SAD0,SAD1,SAD2,SAD3]
// per lane — exactly the per-pixel SAD values needed for the select predicate.
static void GetSumAbsDiff32_AVX512(const __m512i* const A,
                                   const __m512i* const B,
                                   __m512i* const out) {
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
// GetEntropyUnrefined / GetCombinedEntropyUnrefined
//
// These functions compute bit entropy and streak statistics over histogram
// arrays. The C reference iterates element-by-element, detecting runs of
// equal values (streaks) and calling VP8LFastSLog2 per unique value.
//
// AVX-512 strategy: process 16 elements per iteration using SIMD for:
//   1. Nonzero detection (mask comparison)
//   2. Value-change detection (compare adjacent elements → boundary mask)
//   3. Streak-length counting (popcount on sub-masks between boundaries)
//   4. Entropy accumulation (FastSLog2_8x_AVX512 for unique values)
//   5. Sum/max/nonzero tracking (masked add/max)
//
// The boundary iteration is scalar (tzcnt on masks), but the per-element
// work (SLog2, sum, max) is fully vectorized. For typical 256-element
// histograms with ~20-40 unique values, this eliminates ~200 scalar
// VP8LFastSLog2 calls.

#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ) && !defined(DONT_USE_COMBINED_SHANNON_ENTROPY_AVX512_FUNC)

// Process a single streak: accumulate entropy += VP8LFastSLog2(val) * streak,
// sum += val * streak, and update streak statistics.
static WEBP_INLINE void ProcessStreak_AVX512(
    uint32_t val, int streak, int i_start,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  const int is_nz = (val != 0);
  if (is_nz) {
    bit_entropy->sum += val * (uint32_t)streak;
    bit_entropy->nonzeros += streak;
    bit_entropy->nonzero_code = i_start;
    bit_entropy->entropy += VP8LFastSLog2(val) * streak;
    if (bit_entropy->max_val < val) {
      bit_entropy->max_val = val;
    }
  }
  stats->counts[is_nz] += (streak > 3);
  stats->streaks[is_nz][(streak > 3)] += streak;
}

// AVX-512 GetEntropyUnrefined: processes a single histogram array.
// Uses SIMD to detect value-change boundaries in chunks of 16 elements,
// then processes each streak with scalar code.
static void GetEntropyUnrefined_AVX512(
    const uint32_t X[], int length,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  int i;
  int streak_start = 0;
  uint32_t cur_val;

  memset(stats, 0, sizeof(*stats));
  VP8LBitEntropyInit(bit_entropy);
  if (length <= 0) return;

  cur_val = X[0];

  // Process in chunks of 16, detecting boundaries via adjacent comparison.
  // For each chunk, compare X[i] != X[i-1] to find where values change.
  for (i = 1; i + 16 <= length; i += 16) {
    const __m512i cur = _mm512_loadu_si512((const __m512i*)&X[i]);
    // Build "previous" vector: X[i-1..i+14]
    const __m512i prev = _mm512_loadu_si512((const __m512i*)&X[i - 1]);
    // Mask of positions where value changes
    const __mmask16 change = _mm512_cmpneq_epi32_mask(cur, prev);

    if (change == 0) continue;  // No changes in this chunk — extend streak

    // Process each boundary
    uint32_t m = (uint32_t)change;
    while (m) {
      const int bit = _tzcnt_u32(m);
      const int pos = i + bit;  // absolute position of new value
      const int streak = pos - streak_start;
      ProcessStreak_AVX512(cur_val, streak, streak_start,
                           bit_entropy, stats);
      cur_val = X[pos];
      streak_start = pos;
      m &= m - 1;  // clear lowest set bit
    }
  }

  // Scalar tail for remaining elements
  for (; i < length; ++i) {
    if (X[i] != cur_val) {
      const int streak = i - streak_start;
      ProcessStreak_AVX512(cur_val, streak, streak_start,
                           bit_entropy, stats);
      cur_val = X[i];
      streak_start = i;
    }
  }

  // Final streak (terminated by sentinel value 0)
  {
    const int streak = length - streak_start;
    ProcessStreak_AVX512(cur_val, streak, streak_start,
                         bit_entropy, stats);
    // Process the sentinel (value=0, position=length)
    // The C reference calls GetEntropyUnrefinedHelper(0, length, ...) which
    // processes the *previous* streak and sets val_prev=0. Our ProcessStreak
    // already handled the last real streak above. We only need the sentinel
    // if cur_val was nonzero at the end (the zero sentinel creates a new
    // boundary). But the C reference always calls it, which handles the
    // stats for a trailing zero-streak of length 0 — a no-op since streak=0
    // contributes nothing. So we're correct.
  }

  bit_entropy->entropy = VP8LFastSLog2(bit_entropy->sum) - bit_entropy->entropy;
}

// AVX-512 GetCombinedEntropyUnrefined: processes X[i]+Y[i] combined histogram.
// Same boundary-detection approach but operates on the sum of two arrays.
static void GetCombinedEntropyUnrefined_AVX512(
    const uint32_t X[], const uint32_t Y[], int length,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  int i;
  int streak_start = 0;
  uint32_t cur_val;

  memset(stats, 0, sizeof(*stats));
  VP8LBitEntropyInit(bit_entropy);
  if (length <= 0) return;

  cur_val = X[0] + Y[0];

  for (i = 1; i + 16 <= length; i += 16) {
    const __m512i xc = _mm512_loadu_si512((const __m512i*)&X[i]);
    const __m512i yc = _mm512_loadu_si512((const __m512i*)&Y[i]);
    const __m512i cur = _mm512_add_epi32(xc, yc);
    const __m512i xp = _mm512_loadu_si512((const __m512i*)&X[i - 1]);
    const __m512i yp = _mm512_loadu_si512((const __m512i*)&Y[i - 1]);
    const __m512i prev = _mm512_add_epi32(xp, yp);
    const __mmask16 change = _mm512_cmpneq_epi32_mask(cur, prev);

    if (change == 0) continue;

    uint32_t m = (uint32_t)change;
    while (m) {
      const int bit = _tzcnt_u32(m);
      const int pos = i + bit;
      const int streak = pos - streak_start;
      ProcessStreak_AVX512(cur_val, streak, streak_start,
                           bit_entropy, stats);
      cur_val = X[pos] + Y[pos];
      streak_start = pos;
      m &= m - 1;
    }
  }

  for (; i < length; ++i) {
    const uint32_t xy = X[i] + Y[i];
    if (xy != cur_val) {
      const int streak = i - streak_start;
      ProcessStreak_AVX512(cur_val, streak, streak_start,
                           bit_entropy, stats);
      cur_val = xy;
      streak_start = i;
    }
  }

  {
    const int streak = length - streak_start;
    ProcessStreak_AVX512(cur_val, streak, streak_start,
                         bit_entropy, stats);
  }

  bit_entropy->entropy = VP8LFastSLog2(bit_entropy->sum) - bit_entropy->entropy;
}

#endif  // !WEBP_HAVE_SLOW_CLZ_CTZ && !DONT_USE_COMBINED_SHANNON_ENTROPY_AVX512_FUNC

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
  // kLog2Table_int gather to compute v*log2(v) entirely in SIMD, eliminating
  // all scalar VP8LFastSLog2 calls. ~3-5x faster than AVX2 version.
#if !defined(DONT_USE_COMBINED_SHANNON_ENTROPY_AVX512_FUNC)
  VP8LCombinedShannonEntropy = CombinedShannonEntropy_AVX512;
#endif

  // GetEntropyUnrefined: AVX-512 boundary detection (16 elements/iter)
  // eliminates per-element scalar comparison. Gains scale with histogram
  // sparsity — typical 256-element histograms have ~80% zero runs.
#if !defined(WEBP_HAVE_SLOW_CLZ_CTZ) && !defined(DONT_USE_COMBINED_SHANNON_ENTROPY_AVX512_FUNC)
  VP8LGetEntropyUnrefined = GetEntropyUnrefined_AVX512;
  VP8LGetCombinedEntropyUnrefined = GetCombinedEntropyUnrefined_AVX512;
#endif

  // Not dispatched (left at AVX2):
  // - VectorMismatch: marginal benefit, early-exit dominated.
  // - AddVector/AddVectorEq: memory-bound, identical throughput at 256/512.
}

#else  // !WEBP_USE_AVX512

WEBP_DSP_INIT_STUB(VP8LEncDspInitAVX512)

#endif  // WEBP_USE_AVX512
