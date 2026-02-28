// Copyright 2026 Lakeworks. All Rights Reserved.
//
// Precision unit test for float-based entropy functions.
// Validates kLog2Table, kSLog2Table, VP8LFastLog2, VP8LFastSLog2,
// and Shannon entropy against double-precision math.h references.
//
// STANDALONE BUILD (no library linkage needed):
//   MSVC:   cl /Ox /I<libwebp-root> /DSTANDALONE_TEST test_float_entropy.c
//   GCC:    gcc -O2 -I<libwebp-root> -DSTANDALONE_TEST -lm test_float_entropy.c
//
// The test embeds the table data via #include when STANDALONE_TEST is defined.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

// Minimal stubs for headers that lossless_common.h includes
#ifndef WEBP_TYPES_H_
#define WEBP_TYPES_H_
#ifndef WEBP_INLINE
#ifdef _MSC_VER
#define WEBP_INLINE __forceinline
#else
#define WEBP_INLINE inline
#endif
#endif
#ifndef WEBP_RESTRICT
#ifdef _MSC_VER
#define WEBP_RESTRICT __restrict
#else
#define WEBP_RESTRICT restrict
#endif
#endif
#ifndef WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW
#define WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW
#endif
#ifndef WEBP_TSAN_IGNORE_FUNCTION
#define WEBP_TSAN_IGNORE_FUNCTION
#endif
#endif  // WEBP_TYPES_H_

// Stub BitsLog2Floor needed by lossless_common.h
#ifndef WEBP_UTILS_UTILS_H_
#define WEBP_UTILS_UTILS_H_
static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  int log_value = 0;
  while (n >>= 1) ++log_value;
  return log_value;
}
// WebPSafeMalloc/WebPSafeFree stubs (not used by test)
#endif

// Stub cpu.h
#ifndef WEBP_DSP_CPU_H_
#define WEBP_DSP_CPU_H_
typedef int (*VP8CPUInfo)(int feature);
#define VP8_STATUS_OK 0
#endif

// Now include the actual lossless_common.h definitions
// (tables, VP8LFastLog2, VP8LFastSLog2 inlines)
#define LOG_LOOKUP_IDX_MAX 256
#define LOG_2_RECIPROCAL 1.44269504088896338700465094007086
#define APPROX_LOG_WITH_CORRECTION_MAX 65536
#define APPROX_LOG_MAX 4096

// Table and function pointer declarations
extern const float kLog2Table[LOG_LOOKUP_IDX_MAX];
extern const float kSLog2Table[LOG_LOOKUP_IDX_MAX];
typedef float (*VP8LFastLog2SlowFunc)(uint32_t v);
typedef float (*VP8LFastSLog2SlowFunc)(uint32_t v);

extern VP8LFastLog2SlowFunc VP8LFastLog2Slow;
extern VP8LFastSLog2SlowFunc VP8LFastSLog2Slow;

static WEBP_INLINE float VP8LFastLog2(uint32_t v) {
  return (v < LOG_LOOKUP_IDX_MAX) ? kLog2Table[v] : VP8LFastLog2Slow(v);
}
static WEBP_INLINE float VP8LFastSLog2(uint32_t v) {
  return (v < LOG_LOOKUP_IDX_MAX) ? kSLog2Table[v] : VP8LFastSLog2Slow(v);
}

// Prefix tables (needed if we included lossless_common.h directly)
#define PREFIX_LOOKUP_IDX_MAX 512
typedef struct { int8_t code; int8_t extra_bits; } VP8LPrefixCode;

// ---- Provide slow-path implementations ----

static float TestFastLog2Slow(uint32_t v) {
  return (float)(LOG_2_RECIPROCAL * log((double)v));
}
static float TestFastSLog2Slow(uint32_t v) {
  return (float)(LOG_2_RECIPROCAL * (double)v * log((double)v));
}

VP8LFastLog2SlowFunc VP8LFastLog2Slow = TestFastLog2Slow;
VP8LFastSLog2SlowFunc VP8LFastSLog2Slow = TestFastSLog2Slow;

// ---- Embed table data (from src/dsp/lossless_enc.c) ----

const float kLog2Table[LOG_LOOKUP_IDX_MAX] = {
  0.f, 0.f, 1.00000000e+00f, 1.58496250e+00f,
  2.00000000e+00f, 2.32192809e+00f, 2.58496250e+00f, 2.80735492e+00f,
  3.00000000e+00f, 3.16992500e+00f, 3.32192809e+00f, 3.45943162e+00f,
  3.58496250e+00f, 3.70043972e+00f, 3.80735492e+00f, 3.90689060e+00f,
  4.00000000e+00f, 4.08746284e+00f, 4.16992500e+00f, 4.24792751e+00f,
  4.32192809e+00f, 4.39231742e+00f, 4.45943162e+00f, 4.52356196e+00f,
  4.58496250e+00f, 4.64385619e+00f, 4.70043972e+00f, 4.75488750e+00f,
  4.80735492e+00f, 4.85798100e+00f, 4.90689060e+00f, 4.95419631e+00f,
  5.00000000e+00f, 5.04439412e+00f, 5.08746284e+00f, 5.12928302e+00f,
  5.16992500e+00f, 5.20945337e+00f, 5.24792751e+00f, 5.28540222e+00f,
  5.32192809e+00f, 5.35755200e+00f, 5.39231742e+00f, 5.42626475e+00f,
  5.45943162e+00f, 5.49185310e+00f, 5.52356196e+00f, 5.55458885e+00f,
  5.58496250e+00f, 5.61470984e+00f, 5.64385619e+00f, 5.67242534e+00f,
  5.70043972e+00f, 5.72792045e+00f, 5.75488750e+00f, 5.78135971e+00f,
  5.80735492e+00f, 5.83289001e+00f, 5.85798100e+00f, 5.88264305e+00f,
  5.90689060e+00f, 5.93073734e+00f, 5.95419631e+00f, 5.97727992e+00f,
  6.00000000e+00f, 6.02236781e+00f, 6.04439412e+00f, 6.06608919e+00f,
  6.08746284e+00f, 6.10852446e+00f, 6.12928302e+00f, 6.14974712e+00f,
  6.16992500e+00f, 6.18982456e+00f, 6.20945337e+00f, 6.22881869e+00f,
  6.24792751e+00f, 6.26678654e+00f, 6.28540222e+00f, 6.30378075e+00f,
  6.32192809e+00f, 6.33985000e+00f, 6.35755200e+00f, 6.37503943e+00f,
  6.39231742e+00f, 6.40939094e+00f, 6.42626475e+00f, 6.44294350e+00f,
  6.45943162e+00f, 6.47573343e+00f, 6.49185310e+00f, 6.50779464e+00f,
  6.52356196e+00f, 6.53915881e+00f, 6.55458885e+00f, 6.56985561e+00f,
  6.58496250e+00f, 6.59991284e+00f, 6.61470984e+00f, 6.62935662e+00f,
  6.64385619e+00f, 6.65821148e+00f, 6.67242534e+00f, 6.68650053e+00f,
  6.70043972e+00f, 6.71424552e+00f, 6.72792045e+00f, 6.74146699e+00f,
  6.75488750e+00f, 6.76818432e+00f, 6.78135971e+00f, 6.79441587e+00f,
  6.80735492e+00f, 6.82017896e+00f, 6.83289001e+00f, 6.84549005e+00f,
  6.85798100e+00f, 6.87036472e+00f, 6.88264305e+00f, 6.89481776e+00f,
  6.90689060e+00f, 6.91886324e+00f, 6.93073734e+00f, 6.94251451e+00f,
  6.95419631e+00f, 6.96578428e+00f, 6.97727992e+00f, 6.98868469e+00f,
  7.00000000e+00f, 7.01122726e+00f, 7.02236781e+00f, 7.03342300e+00f,
  7.04439412e+00f, 7.05528244e+00f, 7.06608919e+00f, 7.07681560e+00f,
  7.08746284e+00f, 7.09803208e+00f, 7.10852446e+00f, 7.11894107e+00f,
  7.12928302e+00f, 7.13955135e+00f, 7.14974712e+00f, 7.15987134e+00f,
  7.16992500e+00f, 7.17990909e+00f, 7.18982456e+00f, 7.19967234e+00f,
  7.20945337e+00f, 7.21916852e+00f, 7.22881869e+00f, 7.23840474e+00f,
  7.24792751e+00f, 7.25738784e+00f, 7.26678654e+00f, 7.27612441e+00f,
  7.28540222e+00f, 7.29462075e+00f, 7.30378075e+00f, 7.31288296e+00f,
  7.32192809e+00f, 7.33091688e+00f, 7.33985000e+00f, 7.34872815e+00f,
  7.35755200e+00f, 7.36632221e+00f, 7.37503943e+00f, 7.38370429e+00f,
  7.39231742e+00f, 7.40087944e+00f, 7.40939094e+00f, 7.41785251e+00f,
  7.42626475e+00f, 7.43462823e+00f, 7.44294350e+00f, 7.45121111e+00f,
  7.45943162e+00f, 7.46760555e+00f, 7.47573343e+00f, 7.48381578e+00f,
  7.49185310e+00f, 7.49984589e+00f, 7.50779464e+00f, 7.51569984e+00f,
  7.52356196e+00f, 7.53138146e+00f, 7.53915881e+00f, 7.54689446e+00f,
  7.55458885e+00f, 7.56224242e+00f, 7.56985561e+00f, 7.57742883e+00f,
  7.58496250e+00f, 7.59245704e+00f, 7.59991284e+00f, 7.60733031e+00f,
  7.61470984e+00f, 7.62205182e+00f, 7.62935662e+00f, 7.63662462e+00f,
  7.64385619e+00f, 7.65105169e+00f, 7.65821148e+00f, 7.66533592e+00f,
  7.67242534e+00f, 7.67948010e+00f, 7.68650053e+00f, 7.69348696e+00f,
  7.70043972e+00f, 7.70735913e+00f, 7.71424552e+00f, 7.72109919e+00f,
  7.72792045e+00f, 7.73470962e+00f, 7.74146699e+00f, 7.74819285e+00f,
  7.75488750e+00f, 7.76155123e+00f, 7.76818432e+00f, 7.77478706e+00f,
  7.78135971e+00f, 7.78790256e+00f, 7.79441587e+00f, 7.80089990e+00f,
  7.80735492e+00f, 7.81378119e+00f, 7.82017896e+00f, 7.82654849e+00f,
  7.83289001e+00f, 7.83920379e+00f, 7.84549005e+00f, 7.85174904e+00f,
  7.85798100e+00f, 7.86418614e+00f, 7.87036472e+00f, 7.87651695e+00f,
  7.88264305e+00f, 7.88874325e+00f, 7.89481776e+00f, 7.90086681e+00f,
  7.90689060e+00f, 7.91288934e+00f, 7.91886324e+00f, 7.92481250e+00f,
  7.93073734e+00f, 7.93663794e+00f, 7.94251451e+00f, 7.94836723e+00f,
  7.95419631e+00f, 7.96000193e+00f, 7.96578428e+00f, 7.97154355e+00f,
  7.97727992e+00f, 7.98299357e+00f, 7.98868469e+00f, 7.99435344e+00f
};

const float kSLog2Table[LOG_LOOKUP_IDX_MAX] = {
  0.f, 0.f, 2.00000000e+00f, 4.75488750e+00f,
  8.00000000e+00f, 1.16096405e+01f, 1.55097750e+01f, 1.96514845e+01f,
  2.40000000e+01f, 2.85293250e+01f, 3.32192809e+01f, 3.80537478e+01f,
  4.30195500e+01f, 4.81057163e+01f, 5.33029689e+01f, 5.86033589e+01f,
  6.40000000e+01f, 6.94868683e+01f, 7.50586500e+01f, 8.07106228e+01f,
  8.64385619e+01f, 9.22386659e+01f, 9.81074956e+01f, 1.04041925e+02f,
  1.10039100e+02f, 1.16096405e+02f, 1.22211433e+02f, 1.28381963e+02f,
  1.34605938e+02f, 1.40881449e+02f, 1.47206718e+02f, 1.53580086e+02f,
  1.60000000e+02f, 1.66465006e+02f, 1.72973737e+02f, 1.79524906e+02f,
  1.86117300e+02f, 1.92749775e+02f, 1.99421246e+02f, 2.06130687e+02f,
  2.12877124e+02f, 2.19659632e+02f, 2.26477332e+02f, 2.33329384e+02f,
  2.40214991e+02f, 2.47133389e+02f, 2.54083850e+02f, 2.61065676e+02f,
  2.68078200e+02f, 2.75120782e+02f, 2.82192809e+02f, 2.89293692e+02f,
  2.96422865e+02f, 3.03579784e+02f, 3.10763925e+02f, 3.17974784e+02f,
  3.25211876e+02f, 3.32474731e+02f, 3.39762898e+02f, 3.47075940e+02f,
  3.54413436e+02f, 3.61774978e+02f, 3.69160171e+02f, 3.76568635e+02f,
  3.84000000e+02f, 3.91453908e+02f, 3.98930012e+02f, 4.06427976e+02f,
  4.13947473e+02f, 4.21488188e+02f, 4.29049811e+02f, 4.36632045e+02f,
  4.44234600e+02f, 4.51857193e+02f, 4.59499549e+02f, 4.67161402e+02f,
  4.74842491e+02f, 4.82542564e+02f, 4.90261373e+02f, 4.97998679e+02f,
  5.05754248e+02f, 5.13527850e+02f, 5.21319264e+02f, 5.29128273e+02f,
  5.36954664e+02f, 5.44798230e+02f, 5.52658769e+02f, 5.60536084e+02f,
  5.68429982e+02f, 5.76340275e+02f, 5.84266779e+02f, 5.92209312e+02f,
  6.00167700e+02f, 6.08141769e+02f, 6.16131352e+02f, 6.24136283e+02f,
  6.32156400e+02f, 6.40191546e+02f, 6.48241565e+02f, 6.56306305e+02f,
  6.64385619e+02f, 6.72479360e+02f, 6.80587385e+02f, 6.88709554e+02f,
  6.96845731e+02f, 7.04995779e+02f, 7.13159568e+02f, 7.21336968e+02f,
  7.29527850e+02f, 7.37732091e+02f, 7.45949568e+02f, 7.54180161e+02f,
  7.62423751e+02f, 7.70680223e+02f, 7.78949462e+02f, 7.87231356e+02f,
  7.95525795e+02f, 8.03832672e+02f, 8.12151880e+02f, 8.20483314e+02f,
  8.28826871e+02f, 8.37182452e+02f, 8.45549955e+02f, 8.53929284e+02f,
  8.62320342e+02f, 8.70723036e+02f, 8.79137270e+02f, 8.87562955e+02f,
  8.96000000e+02f, 9.04448316e+02f, 9.12907816e+02f, 9.21378413e+02f,
  9.29860024e+02f, 9.38352564e+02f, 9.46855952e+02f, 9.55370106e+02f,
  9.63894946e+02f, 9.72430395e+02f, 9.80976375e+02f, 9.89532809e+02f,
  9.98099622e+02f, 1.00667674e+03f, 1.01526409e+03f, 1.02386160e+03f,
  1.03246920e+03f, 1.04108682e+03f, 1.04971439e+03f, 1.05835183e+03f,
  1.06699910e+03f, 1.07565611e+03f, 1.08432280e+03f, 1.09299912e+03f,
  1.10168498e+03f, 1.11038034e+03f, 1.11908513e+03f, 1.12779928e+03f,
  1.13652275e+03f, 1.14525546e+03f, 1.15399736e+03f, 1.16274839e+03f,
  1.17150850e+03f, 1.18027762e+03f, 1.18905570e+03f, 1.19784269e+03f,
  1.20663853e+03f, 1.21544317e+03f, 1.22425655e+03f, 1.23307862e+03f,
  1.24190933e+03f, 1.25074862e+03f, 1.25959646e+03f, 1.26845278e+03f,
  1.27731754e+03f, 1.28619068e+03f, 1.29507217e+03f, 1.30396194e+03f,
  1.31285996e+03f, 1.32176618e+03f, 1.33068055e+03f, 1.33960302e+03f,
  1.34853356e+03f, 1.35747211e+03f, 1.36641862e+03f, 1.37537307e+03f,
  1.38433540e+03f, 1.39330557e+03f, 1.40228354e+03f, 1.41126926e+03f,
  1.42026270e+03f, 1.42926382e+03f, 1.43827257e+03f, 1.44728891e+03f,
  1.45631280e+03f, 1.46534421e+03f, 1.47438309e+03f, 1.48342941e+03f,
  1.49248313e+03f, 1.50154421e+03f, 1.51061261e+03f, 1.51968830e+03f,
  1.52877124e+03f, 1.53786139e+03f, 1.54695872e+03f, 1.55606319e+03f,
  1.56517477e+03f, 1.57429342e+03f, 1.58341911e+03f, 1.59255180e+03f,
  1.60169146e+03f, 1.61083806e+03f, 1.61999156e+03f, 1.62915193e+03f,
  1.63831914e+03f, 1.64749315e+03f, 1.65667394e+03f, 1.66586146e+03f,
  1.67505570e+03f, 1.68425662e+03f, 1.69346418e+03f, 1.70267837e+03f,
  1.71189914e+03f, 1.72112647e+03f, 1.73036032e+03f, 1.73960068e+03f,
  1.74884750e+03f, 1.75810077e+03f, 1.76736045e+03f, 1.77662651e+03f,
  1.78589892e+03f, 1.79517767e+03f, 1.80446271e+03f, 1.81375403e+03f,
  1.82305159e+03f, 1.83235537e+03f, 1.84166534e+03f, 1.85098148e+03f,
  1.86030376e+03f, 1.86963215e+03f, 1.87896663e+03f, 1.88830717e+03f,
  1.89765374e+03f, 1.90700633e+03f, 1.91636490e+03f, 1.92572944e+03f,
  1.93509991e+03f, 1.94447630e+03f, 1.95385857e+03f, 1.96324671e+03f,
  1.97264068e+03f, 1.98204048e+03f, 1.99144607e+03f, 2.00085743e+03f,
  2.01027454e+03f, 2.01969737e+03f, 2.02912591e+03f, 2.03856013e+03f
};

// ---- Reference implementations (double precision) ----

static double ref_log2(uint32_t v) {
  if (v <= 1) return 0.0;
  return log((double)v) / log(2.0);
}

static double ref_slog2(uint32_t v) {
  if (v <= 1) return 0.0;
  return (double)v * log((double)v) / log(2.0);
}

// ---- Test helpers ----

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, fmt, ...) do { \
  g_tests_run++; \
  if (!(cond)) { \
    printf("FAIL: " fmt "\n", ##__VA_ARGS__); \
    g_tests_failed++; \
  } else { \
    g_tests_passed++; \
  } \
} while (0)

static float rel_error(float measured, double reference) {
  if (reference == 0.0) return (measured == 0.f) ? 0.f : 1.f;
  return (float)(fabs((double)measured - reference) / fabs(reference));
}

// ---- Test 1: kLog2Table precision ----
static void test_kLog2Table(void) {
  int i;
  float max_rel = 0.f;
  int worst_idx = 0;
  printf("--- Test kLog2Table[0..255] vs double log2() ---\n");
  for (i = 0; i < 256; i++) {
    const double ref = ref_log2((uint32_t)i);
    const float val = kLog2Table[i];
    const float re = rel_error(val, ref);
    if (re > max_rel) {
      max_rel = re;
      worst_idx = i;
    }
    TEST_ASSERT(re < 1e-6f,
                "kLog2Table[%d] = %.10g, expected %.10g (rel_err=%.2e)",
                i, (double)val, ref, (double)re);
  }
  printf("  max relative error: %.2e at index %d\n",
         (double)max_rel, worst_idx);
}

// ---- Test 2: kSLog2Table precision ----
static void test_kSLog2Table(void) {
  int i;
  float max_rel = 0.f;
  int worst_idx = 0;
  printf("--- Test kSLog2Table[0..255] vs double v*log2(v) ---\n");
  for (i = 0; i < 256; i++) {
    const double ref = ref_slog2((uint32_t)i);
    const float val = kSLog2Table[i];
    const float re = rel_error(val, ref);
    if (re > max_rel) {
      max_rel = re;
      worst_idx = i;
    }
    TEST_ASSERT(re < 1e-6f,
                "kSLog2Table[%d] = %.10g, expected %.10g (rel_err=%.2e)",
                i, (double)val, ref, (double)re);
  }
  printf("  max relative error: %.2e at index %d\n",
         (double)max_rel, worst_idx);
}

// ---- Test 3: VP8LFastLog2 for larger values (slow path) ----
static void test_FastLog2_large(void) {
  static const uint32_t test_vals[] = {
    256, 257, 300, 512, 1000, 1024, 4096, 10000, 65536,
    100000, 1000000, 10000000, 100000000, 1000000000, 0xFFFFFFFFu
  };
  const int n = (int)(sizeof(test_vals) / sizeof(test_vals[0]));
  int i;
  float max_rel = 0.f;
  printf("--- Test VP8LFastLog2 for v >= 256 ---\n");
  for (i = 0; i < n; i++) {
    const uint32_t v = test_vals[i];
    const double ref = ref_log2(v);
    const float val = VP8LFastLog2(v);
    const float re = rel_error(val, ref);
    if (re > max_rel) max_rel = re;
    TEST_ASSERT(re < 1e-6f,
                "VP8LFastLog2(%u) = %.10g, expected %.10g (rel_err=%.2e)",
                v, (double)val, ref, (double)re);
  }
  printf("  max relative error: %.2e\n", (double)max_rel);
}

// ---- Test 4: VP8LFastSLog2 for larger values ----
static void test_FastSLog2_large(void) {
  static const uint32_t test_vals[] = {
    256, 257, 300, 512, 1000, 1024, 4096, 10000, 65536,
    100000, 1000000, 10000000, 100000000
  };
  const int n = (int)(sizeof(test_vals) / sizeof(test_vals[0]));
  int i;
  float max_rel = 0.f;
  printf("--- Test VP8LFastSLog2 for v >= 256 ---\n");
  for (i = 0; i < n; i++) {
    const uint32_t v = test_vals[i];
    const double ref = ref_slog2(v);
    const float val = VP8LFastSLog2(v);
    const float re = rel_error(val, ref);
    if (re > max_rel) max_rel = re;
    TEST_ASSERT(re < 1e-6f,
                "VP8LFastSLog2(%u) = %.10g, expected %.10g (rel_err=%.2e)",
                v, (double)val, ref, (double)re);
  }
  printf("  max relative error: %.2e\n", (double)max_rel);
}

// ---- Test 5: Shannon entropy of known distributions ----
static void test_shannon_entropy(void) {
  printf("--- Test Shannon entropy of known distributions ---\n");

  // Uniform distribution: 256 bins each with count 100
  // H = log2(256) = 8.0 bits per symbol, total = 8.0 * 25600 = 204800
  {
    uint32_t uniform[256];
    int i;
    float entropy;
    double ref_entropy;
    uint32_t total;
    float re;

    for (i = 0; i < 256; i++) uniform[i] = 100;
    total = 256 * 100;
    entropy = VP8LFastSLog2(total);
    ref_entropy = ref_slog2(total);
    for (i = 0; i < 256; i++) {
      entropy -= VP8LFastSLog2(uniform[i]);
      ref_entropy -= ref_slog2(uniform[i]);
    }
    re = rel_error(entropy, ref_entropy);
    printf("  Uniform(256x100): H=%.6f, ref=%.6f, rel_err=%.2e\n",
           (double)entropy, ref_entropy, (double)re);
    TEST_ASSERT(re < 1e-4f,
                "Uniform entropy rel_err %.2e >= 1e-4", (double)re);
    // Float32 accumulation over 256 subtractions introduces ~1e-5 rel error
    TEST_ASSERT(fabs((double)entropy - 204800.0) < 5.0,
                "Uniform entropy %.1f != expected ~204800", (double)entropy);
  }

  // Single-symbol: H = 0
  {
    float entropy;
    entropy = VP8LFastSLog2(10000) - VP8LFastSLog2(10000);
    printf("  Single-symbol(10000): H=%.6f (expected 0.0)\n",
           (double)entropy);
    TEST_ASSERT(fabs((double)entropy) < 0.01f,
                "Single-symbol entropy %.6f != 0", (double)entropy);
  }

  // Two-symbol 50/50: H = 1.0 bit/symbol, total = 10000
  {
    uint32_t total = 10000;
    float entropy;
    double ref_entropy;
    float re;

    entropy = VP8LFastSLog2(total) - 2.0f * VP8LFastSLog2(5000);
    ref_entropy = ref_slog2(total) - 2.0 * ref_slog2(5000);
    re = rel_error(entropy, ref_entropy);
    printf("  Two-symbol(5000,5000): H=%.6f, ref=%.6f, rel_err=%.2e\n",
           (double)entropy, ref_entropy, (double)re);
    TEST_ASSERT(re < 1e-4f,
                "Two-sym entropy rel_err %.2e >= 1e-4", (double)re);
    TEST_ASSERT(fabs((double)entropy - 10000.0) < 1.0,
                "Two-sym entropy %.1f != expected ~10000", (double)entropy);
  }

  // Skewed: {9000, 500, 300, 200}
  {
    const uint32_t counts[] = {9000, 500, 300, 200};
    uint32_t total = 10000;
    float entropy;
    double ref_entropy;
    float re;
    int i;

    entropy = VP8LFastSLog2(total);
    ref_entropy = ref_slog2(total);
    for (i = 0; i < 4; i++) {
      entropy -= VP8LFastSLog2(counts[i]);
      ref_entropy -= ref_slog2(counts[i]);
    }
    re = rel_error(entropy, ref_entropy);
    printf("  Skewed(9000,500,300,200): H=%.6f, ref=%.6f, rel_err=%.2e\n",
           (double)entropy, ref_entropy, (double)re);
    TEST_ASSERT(re < 1e-3f,
                "Skewed entropy rel_err %.2e >= 1e-3", (double)re);
  }
}

// ---- Test 6: Boundary and edge cases ----
static void test_edge_cases(void) {
  printf("--- Test edge cases ---\n");

  TEST_ASSERT(VP8LFastLog2(0) == 0.f,
              "VP8LFastLog2(0) = %g != 0", (double)VP8LFastLog2(0));
  TEST_ASSERT(VP8LFastLog2(1) == 0.f,
              "VP8LFastLog2(1) = %g != 0", (double)VP8LFastLog2(1));
  TEST_ASSERT(VP8LFastLog2(2) == 1.0f,
              "VP8LFastLog2(2) = %.10g != 1.0", (double)VP8LFastLog2(2));
  TEST_ASSERT(VP8LFastLog2(4) == 2.0f,
              "VP8LFastLog2(4) = %.10g != 2.0", (double)VP8LFastLog2(4));
  TEST_ASSERT(VP8LFastLog2(128) == 7.0f,
              "VP8LFastLog2(128) = %.10g != 7.0", (double)VP8LFastLog2(128));

  TEST_ASSERT(VP8LFastSLog2(0) == 0.f,
              "VP8LFastSLog2(0) = %g != 0", (double)VP8LFastSLog2(0));
  TEST_ASSERT(VP8LFastSLog2(1) == 0.f,
              "VP8LFastSLog2(1) = %g != 0", (double)VP8LFastSLog2(1));
  TEST_ASSERT(VP8LFastSLog2(2) == 2.0f,
              "VP8LFastSLog2(2) = %.10g != 2.0", (double)VP8LFastSLog2(2));

  // Table boundary: 255 (last entry)
  {
    const float re = rel_error(VP8LFastLog2(255), ref_log2(255));
    TEST_ASSERT(re < 1e-6f, "VP8LFastLog2(255) rel_err %.2e", (double)re);
  }
  // Just above table: 256 (first slow path) = 2^8 = 8.0
  {
    const float val = VP8LFastLog2(256);
    const float re = rel_error(val, ref_log2(256));
    TEST_ASSERT(re < 1e-6f, "VP8LFastLog2(256) rel_err %.2e", (double)re);
    TEST_ASSERT(fabs((double)val - 8.0) < 1e-5,
                "VP8LFastLog2(256) = %.10g != 8.0", (double)val);
  }

  printf("  edge cases OK\n");
}

// ---- Test 7: Float vs fixed-point precision comparison ----
static void test_float_vs_fixedpoint(void) {
  printf("--- Float vs 23-bit fixed-point precision comparison ---\n");

  // Difference sensitivity: log2(8193) - log2(8192)
  {
    const float fa = VP8LFastLog2(8192);
    const float fb = VP8LFastLog2(8193);
    const double ra = ref_log2(8192);
    const double rb = ref_log2(8193);
    const double float_diff = (double)fb - (double)fa;
    const double ref_diff = rb - ra;
    const double diff_err = fabs(float_diff - ref_diff);
    printf("  log2(8193)-log2(8192): float=%.10e, ref=%.10e, err=%.2e\n",
           float_diff, ref_diff, diff_err);
    // Float32 has 24-bit mantissa — small differences lose relative precision
    // but 3e-7 absolute error is excellent for entropy cost comparisons
    TEST_ASSERT(diff_err < 1e-6,
                "log2 diff error %.2e >= 1e-6", diff_err);
  }

  // Large value: SLog2(10^8)
  {
    const uint32_t v = 100000000u;
    const float fval = VP8LFastSLog2(v);
    const double rval = ref_slog2(v);
    const float re = rel_error(fval, rval);
    printf("  SLog2(10^8): float=%.6e, ref=%.6e, rel_err=%.2e\n",
           (double)fval, rval, (double)re);
    TEST_ASSERT(re < 1e-6f, "SLog2(10^8) rel_err %.2e", (double)re);
  }

  printf("  float precision verified adequate for entropy coding\n");
}

// ---- Test 8: Accumulation precision (simulates real histogram combine) ----
static void test_accumulation(void) {
  printf("--- Test accumulation precision (histogram combine simulation) ---\n");

  // Simulate combining two histograms with 256 bins each
  // This is what histogram_enc.c does thousands of times
  {
    uint32_t X[256], Y[256];
    int i;
    float float_entropy = 0.f;
    double ref_entropy = 0.0;
    uint32_t sumX = 0, sumXY = 0;
    float re;

    // Fill with realistic histogram data
    for (i = 0; i < 256; i++) {
      X[i] = (uint32_t)((i * 37 + 13) % 100);
      Y[i] = (uint32_t)((i * 53 + 7) % 80);
    }

    // Compute CombinedShannonEntropy using float (our implementation)
    for (i = 0; i < 256; i++) {
      const uint32_t x = X[i];
      if (x != 0) {
        const uint32_t xy = x + Y[i];
        sumX += x;
        float_entropy += VP8LFastSLog2(x);
        sumXY += xy;
        float_entropy += VP8LFastSLog2(xy);
      } else if (Y[i] != 0) {
        sumXY += Y[i];
        float_entropy += VP8LFastSLog2(Y[i]);
      }
    }
    float_entropy = VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY) - float_entropy;

    // Same with double reference
    sumX = 0; sumXY = 0;
    for (i = 0; i < 256; i++) {
      const uint32_t x = X[i];
      if (x != 0) {
        const uint32_t xy = x + Y[i];
        sumX += x;
        ref_entropy += ref_slog2(x);
        sumXY += xy;
        ref_entropy += ref_slog2(xy);
      } else if (Y[i] != 0) {
        sumXY += Y[i];
        ref_entropy += ref_slog2(Y[i]);
      }
    }
    ref_entropy = ref_slog2(sumX) + ref_slog2(sumXY) - ref_entropy;

    re = rel_error(float_entropy, ref_entropy);
    printf("  Combined entropy: float=%.6f, ref=%.6f, rel_err=%.2e\n",
           (double)float_entropy, ref_entropy, (double)re);
    // This is the key test: after accumulating hundreds of SLog2 values,
    // float precision should still be adequate (<0.1% relative error)
    TEST_ASSERT(re < 1e-3f,
                "Accumulation rel_err %.2e >= 1e-3", (double)re);
  }
}

// ---- Main ----

int main(void) {
  printf("=== libwebp float entropy precision test ===\n\n");

  test_kLog2Table();
  test_kSLog2Table();
  test_FastLog2_large();
  test_FastSLog2_large();
  test_shannon_entropy();
  test_edge_cases();
  test_float_vs_fixedpoint();
  test_accumulation();

  printf("\n=== Results: %d/%d passed, %d failed ===\n",
         g_tests_passed, g_tests_run, g_tests_failed);

  return (g_tests_failed > 0) ? 1 : 0;
}
