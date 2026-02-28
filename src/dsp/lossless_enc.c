// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transform methods for lossless encoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "src/dsp/cpu.h"
#include "src/dsp/dsp.h"
#include "src/dsp/lossless.h"
#include "src/dsp/lossless_common.h"
#include "src/enc/histogram_enc.h"
#include "src/utils/utils.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

// lookup table for small values of log2(int) as float.
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

// lookup table for small values of v * log2(v) as float.
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

const VP8LPrefixCode kPrefixEncodeCode[PREFIX_LOOKUP_IDX_MAX] = {
  { 0, 0}, { 0, 0}, { 1, 0}, { 2, 0}, { 3, 0}, { 4, 1}, { 4, 1}, { 5, 1},
  { 5, 1}, { 6, 2}, { 6, 2}, { 6, 2}, { 6, 2}, { 7, 2}, { 7, 2}, { 7, 2},
  { 7, 2}, { 8, 3}, { 8, 3}, { 8, 3}, { 8, 3}, { 8, 3}, { 8, 3}, { 8, 3},
  { 8, 3}, { 9, 3}, { 9, 3}, { 9, 3}, { 9, 3}, { 9, 3}, { 9, 3}, { 9, 3},
  { 9, 3}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4},
  {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4},
  {10, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4},
  {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4},
  {11, 4}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
  {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
  {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
  {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
  {12, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
  {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
  {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
  {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
  {13, 5}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
};

const uint8_t kPrefixEncodeExtraBitsValue[PREFIX_LOOKUP_IDX_MAX] = {
   0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  2,  3,  0,  1,  2,  3,
   0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
  96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
  127,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
  96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126
};

static float FastSLog2Slow_C(uint32_t v) {
  assert(v >= LOG_LOOKUP_IDX_MAX);
  return (float)(LOG_2_RECIPROCAL * (double)v * log((double)v));
}

static float FastLog2Slow_C(uint32_t v) {
  assert(v >= LOG_LOOKUP_IDX_MAX);
  return (float)(LOG_2_RECIPROCAL * log((double)v));
}

//------------------------------------------------------------------------------
// Methods to calculate Entropy (Shannon).

// Compute the combined Shanon's entropy for distribution {X} and {X+Y}
static float CombinedShannonEntropy_C(const uint32_t X[256],
                                      const uint32_t Y[256]) {
  int i;
  float retval = 0.f;
  uint32_t sumX = 0, sumXY = 0;
  for (i = 0; i < 256; ++i) {
    const uint32_t x = X[i];
    if (x != 0) {
      const uint32_t xy = x + Y[i];
      sumX += x;
      retval += VP8LFastSLog2(x);
      sumXY += xy;
      retval += VP8LFastSLog2(xy);
    } else if (Y[i] != 0) {
      sumXY += Y[i];
      retval += VP8LFastSLog2(Y[i]);
    }
  }
  retval = VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY) - retval;
  return retval;
}

static float ShannonEntropy_C(const uint32_t* X, int n) {
  int i;
  float retval = 0.f;
  uint32_t sumX = 0;
  for (i = 0; i < n; ++i) {
    const int x = X[i];
    if (x != 0) {
      sumX += x;
      retval += VP8LFastSLog2(x);
    }
  }
  retval = VP8LFastSLog2(sumX) - retval;
  return retval;
}

void VP8LBitEntropyInit(VP8LBitEntropy* const entropy) {
  entropy->entropy = 0.f;
  entropy->sum = 0;
  entropy->nonzeros = 0;
  entropy->max_val = 0;
  entropy->nonzero_code = VP8L_NON_TRIVIAL_SYM;
}

void VP8LBitsEntropyUnrefined(const uint32_t* WEBP_RESTRICT const array, int n,
                              VP8LBitEntropy* WEBP_RESTRICT const entropy) {
  int i;

  VP8LBitEntropyInit(entropy);

  for (i = 0; i < n; ++i) {
    if (array[i] != 0) {
      entropy->sum += array[i];
      entropy->nonzero_code = i;
      ++entropy->nonzeros;
      entropy->entropy += VP8LFastSLog2(array[i]);
      if (entropy->max_val < array[i]) {
        entropy->max_val = array[i];
      }
    }
  }
  entropy->entropy = VP8LFastSLog2(entropy->sum) - entropy->entropy;
}

static WEBP_INLINE void GetEntropyUnrefinedHelper(
    uint32_t val, int i, uint32_t* WEBP_RESTRICT const val_prev,
    int* WEBP_RESTRICT const i_prev,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  const int streak = i - *i_prev;

  // Gather info for the bit entropy.
  if (*val_prev != 0) {
    bit_entropy->sum += (*val_prev) * streak;
    bit_entropy->nonzeros += streak;
    bit_entropy->nonzero_code = *i_prev;
    bit_entropy->entropy += VP8LFastSLog2(*val_prev) * streak;
    if (bit_entropy->max_val < *val_prev) {
      bit_entropy->max_val = *val_prev;
    }
  }

  // Gather info for the Huffman cost.
  stats->counts[*val_prev != 0] += (streak > 3);
  stats->streaks[*val_prev != 0][(streak > 3)] += streak;

  *val_prev = val;
  *i_prev = i;
}

static void GetEntropyUnrefined_C(
    const uint32_t X[], int length,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  int i;
  int i_prev = 0;
  uint32_t x_prev = X[0];

  memset(stats, 0, sizeof(*stats));
  VP8LBitEntropyInit(bit_entropy);

  for (i = 1; i < length; ++i) {
    const uint32_t x = X[i];
    if (x != x_prev) {
      GetEntropyUnrefinedHelper(x, i, &x_prev, &i_prev, bit_entropy, stats);
    }
  }
  GetEntropyUnrefinedHelper(0, i, &x_prev, &i_prev, bit_entropy, stats);

  bit_entropy->entropy = VP8LFastSLog2(bit_entropy->sum) - bit_entropy->entropy;
}

static void GetCombinedEntropyUnrefined_C(
    const uint32_t X[], const uint32_t Y[], int length,
    VP8LBitEntropy* WEBP_RESTRICT const bit_entropy,
    VP8LStreaks* WEBP_RESTRICT const stats) {
  int i = 1;
  int i_prev = 0;
  uint32_t xy_prev = X[0] + Y[0];

  memset(stats, 0, sizeof(*stats));
  VP8LBitEntropyInit(bit_entropy);

  for (i = 1; i < length; ++i) {
    const uint32_t xy = X[i] + Y[i];
    if (xy != xy_prev) {
      GetEntropyUnrefinedHelper(xy, i, &xy_prev, &i_prev, bit_entropy, stats);
    }
  }
  GetEntropyUnrefinedHelper(0, i, &xy_prev, &i_prev, bit_entropy, stats);

  bit_entropy->entropy = VP8LFastSLog2(bit_entropy->sum) - bit_entropy->entropy;
}

//------------------------------------------------------------------------------

void VP8LSubtractGreenFromBlueAndRed_C(uint32_t* argb_data, int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const int argb = (int)argb_data[i];
    const int green = (argb >> 8) & 0xff;
    const uint32_t new_r = (((argb >> 16) & 0xff) - green) & 0xff;
    const uint32_t new_b = (((argb >>  0) & 0xff) - green) & 0xff;
    argb_data[i] = ((uint32_t)argb & 0xff00ff00u) | (new_r << 16) | new_b;
  }
}

static WEBP_INLINE int ColorTransformDelta(int8_t color_pred, int8_t color) {
  return ((int)color_pred * color) >> 5;
}

static WEBP_INLINE int8_t U32ToS8(uint32_t v) {
  return (int8_t)(v & 0xff);
}

void VP8LTransformColor_C(const VP8LMultipliers* WEBP_RESTRICT const m,
                          uint32_t* WEBP_RESTRICT data, int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint32_t argb = data[i];
    const int8_t green = U32ToS8(argb >>  8);
    const int8_t red   = U32ToS8(argb >> 16);
    int new_red = red & 0xff;
    int new_blue = argb & 0xff;
    new_red -= ColorTransformDelta((int8_t)m->green_to_red, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta((int8_t)m->green_to_blue, green);
    new_blue -= ColorTransformDelta((int8_t)m->red_to_blue, red);
    new_blue &= 0xff;
    data[i] = (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);
  }
}

static WEBP_INLINE uint8_t TransformColorRed(uint8_t green_to_red,
                                             uint32_t argb) {
  const int8_t green = U32ToS8(argb >> 8);
  int new_red = argb >> 16;
  new_red -= ColorTransformDelta((int8_t)green_to_red, green);
  return (new_red & 0xff);
}

static WEBP_INLINE uint8_t TransformColorBlue(uint8_t green_to_blue,
                                              uint8_t red_to_blue,
                                              uint32_t argb) {
  const int8_t green = U32ToS8(argb >>  8);
  const int8_t red   = U32ToS8(argb >> 16);
  int new_blue = argb & 0xff;
  new_blue -= ColorTransformDelta((int8_t)green_to_blue, green);
  new_blue -= ColorTransformDelta((int8_t)red_to_blue, red);
  return (new_blue & 0xff);
}

void VP8LCollectColorRedTransforms_C(const uint32_t* WEBP_RESTRICT argb,
                                     int stride,
                                     int tile_width, int tile_height,
                                     int green_to_red, uint32_t histo[]) {
  while (tile_height-- > 0) {
    int x;
    for (x = 0; x < tile_width; ++x) {
      ++histo[TransformColorRed((uint8_t)green_to_red, argb[x])];
    }
    argb += stride;
  }
}

void VP8LCollectColorBlueTransforms_C(const uint32_t* WEBP_RESTRICT argb,
                                      int stride,
                                      int tile_width, int tile_height,
                                      int green_to_blue, int red_to_blue,
                                      uint32_t histo[]) {
  while (tile_height-- > 0) {
    int x;
    for (x = 0; x < tile_width; ++x) {
      ++histo[TransformColorBlue((uint8_t)green_to_blue, (uint8_t)red_to_blue,
                                 argb[x])];
    }
    argb += stride;
  }
}

//------------------------------------------------------------------------------

static int VectorMismatch_C(const uint32_t* const array1,
                            const uint32_t* const array2, int length) {
  int match_len = 0;

  while (match_len < length && array1[match_len] == array2[match_len]) {
    ++match_len;
  }
  return match_len;
}

// Bundles multiple (1, 2, 4 or 8) pixels into a single pixel.
void VP8LBundleColorMap_C(const uint8_t* WEBP_RESTRICT const row,
                          int width, int xbits, uint32_t* WEBP_RESTRICT dst) {
  int x;
  if (xbits > 0) {
    const int bit_depth = 1 << (3 - xbits);
    const int mask = (1 << xbits) - 1;
    uint32_t code = 0xff000000;
    for (x = 0; x < width; ++x) {
      const int xsub = x & mask;
      if (xsub == 0) {
        code = 0xff000000;
      }
      code |= row[x] << (8 + bit_depth * xsub);
      dst[x >> xbits] = code;
    }
  } else {
    for (x = 0; x < width; ++x) dst[x] = 0xff000000 | (row[x] << 8);
  }
}

//------------------------------------------------------------------------------

static uint32_t ExtraCost_C(const uint32_t* population, int length) {
  int i;
  uint32_t cost = population[4] + population[5];
  assert(length % 2 == 0);
  for (i = 2; i < length / 2 - 1; ++i) {
    cost += i * (population[2 * i + 2] + population[2 * i + 3]);
  }
  return cost;
}

//------------------------------------------------------------------------------

static void AddVector_C(const uint32_t* WEBP_RESTRICT a,
                        const uint32_t* WEBP_RESTRICT b,
                        uint32_t* WEBP_RESTRICT out, int size) {
  int i;
  for (i = 0; i < size; ++i) out[i] = a[i] + b[i];
}

static void AddVectorEq_C(const uint32_t* WEBP_RESTRICT a,
                          uint32_t* WEBP_RESTRICT out, int size) {
  int i;
  for (i = 0; i < size; ++i) out[i] += a[i];
}

//------------------------------------------------------------------------------
// Image transforms.

static void PredictorSub0_C(const uint32_t* in, const uint32_t* upper,
                            int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i < num_pixels; ++i) out[i] = VP8LSubPixels(in[i], ARGB_BLACK);
  (void)upper;
}

static void PredictorSub1_C(const uint32_t* in, const uint32_t* upper,
                            int num_pixels, uint32_t* WEBP_RESTRICT out) {
  int i;
  for (i = 0; i < num_pixels; ++i) out[i] = VP8LSubPixels(in[i], in[i - 1]);
  (void)upper;
}

// It subtracts the prediction from the input pixel and stores the residual
// in the output pixel.
#define GENERATE_PREDICTOR_SUB(PREDICTOR_I)                                \
static void PredictorSub##PREDICTOR_I##_C(const uint32_t* in,              \
                                          const uint32_t* upper,           \
                                          int num_pixels,                  \
                                          uint32_t* WEBP_RESTRICT out) {   \
  int x;                                                                   \
  assert(upper != NULL);                                                   \
  for (x = 0; x < num_pixels; ++x) {                                       \
    const uint32_t pred =                                                  \
        VP8LPredictor##PREDICTOR_I##_C(&in[x - 1], upper + x);             \
    out[x] = VP8LSubPixels(in[x], pred);                                   \
  }                                                                        \
}

GENERATE_PREDICTOR_SUB(2)
GENERATE_PREDICTOR_SUB(3)
GENERATE_PREDICTOR_SUB(4)
GENERATE_PREDICTOR_SUB(5)
GENERATE_PREDICTOR_SUB(6)
GENERATE_PREDICTOR_SUB(7)
GENERATE_PREDICTOR_SUB(8)
GENERATE_PREDICTOR_SUB(9)
GENERATE_PREDICTOR_SUB(10)
GENERATE_PREDICTOR_SUB(11)
GENERATE_PREDICTOR_SUB(12)
GENERATE_PREDICTOR_SUB(13)

//------------------------------------------------------------------------------

VP8LProcessEncBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed;
VP8LProcessEncBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed_SSE;

VP8LTransformColorFunc VP8LTransformColor;
VP8LTransformColorFunc VP8LTransformColor_SSE;

VP8LCollectColorBlueTransformsFunc VP8LCollectColorBlueTransforms;
VP8LCollectColorBlueTransformsFunc VP8LCollectColorBlueTransforms_SSE;
VP8LCollectColorRedTransformsFunc VP8LCollectColorRedTransforms;
VP8LCollectColorRedTransformsFunc VP8LCollectColorRedTransforms_SSE;

VP8LFastLog2SlowFunc VP8LFastLog2Slow;
VP8LFastSLog2SlowFunc VP8LFastSLog2Slow;

VP8LCostFunc VP8LExtraCost;
VP8LCombinedShannonEntropyFunc VP8LCombinedShannonEntropy;
VP8LShannonEntropyFunc VP8LShannonEntropy;

VP8LGetEntropyUnrefinedFunc VP8LGetEntropyUnrefined;
VP8LGetCombinedEntropyUnrefinedFunc VP8LGetCombinedEntropyUnrefined;

VP8LAddVectorFunc VP8LAddVector;
VP8LAddVectorEqFunc VP8LAddVectorEq;

VP8LVectorMismatchFunc VP8LVectorMismatch;
VP8LBundleColorMapFunc VP8LBundleColorMap;
VP8LBundleColorMapFunc VP8LBundleColorMap_SSE;

VP8LPredictorAddSubFunc VP8LPredictorsSub[16];
VP8LPredictorAddSubFunc VP8LPredictorsSub_C[16];
VP8LPredictorAddSubFunc VP8LPredictorsSub_SSE[16];

extern VP8CPUInfo VP8GetCPUInfo;
extern void VP8LEncDspInitSSE2(void);
extern void VP8LEncDspInitSSE41(void);
extern void VP8LEncDspInitAVX2(void);
extern void VP8LEncDspInitAVX512(void);
extern void VP8LEncDspInitNEON(void);
extern void VP8LEncDspInitMIPS32(void);
extern void VP8LEncDspInitMIPSdspR2(void);
extern void VP8LEncDspInitMSA(void);

WEBP_DSP_INIT_FUNC(VP8LEncDspInit) {
  VP8LDspInit();

#if !WEBP_NEON_OMIT_C_CODE
  VP8LSubtractGreenFromBlueAndRed = VP8LSubtractGreenFromBlueAndRed_C;

  VP8LTransformColor = VP8LTransformColor_C;
#endif

  VP8LCollectColorBlueTransforms = VP8LCollectColorBlueTransforms_C;
  VP8LCollectColorRedTransforms = VP8LCollectColorRedTransforms_C;

  VP8LFastLog2Slow = FastLog2Slow_C;
  VP8LFastSLog2Slow = FastSLog2Slow_C;

  VP8LExtraCost = ExtraCost_C;
  VP8LCombinedShannonEntropy = CombinedShannonEntropy_C;
  VP8LShannonEntropy = ShannonEntropy_C;

  VP8LGetEntropyUnrefined = GetEntropyUnrefined_C;
  VP8LGetCombinedEntropyUnrefined = GetCombinedEntropyUnrefined_C;

  VP8LAddVector = AddVector_C;
  VP8LAddVectorEq = AddVectorEq_C;

  VP8LVectorMismatch = VectorMismatch_C;
  VP8LBundleColorMap = VP8LBundleColorMap_C;

  VP8LPredictorsSub[0] = PredictorSub0_C;
  VP8LPredictorsSub[1] = PredictorSub1_C;
  VP8LPredictorsSub[2] = PredictorSub2_C;
  VP8LPredictorsSub[3] = PredictorSub3_C;
  VP8LPredictorsSub[4] = PredictorSub4_C;
  VP8LPredictorsSub[5] = PredictorSub5_C;
  VP8LPredictorsSub[6] = PredictorSub6_C;
  VP8LPredictorsSub[7] = PredictorSub7_C;
  VP8LPredictorsSub[8] = PredictorSub8_C;
  VP8LPredictorsSub[9] = PredictorSub9_C;
  VP8LPredictorsSub[10] = PredictorSub10_C;
  VP8LPredictorsSub[11] = PredictorSub11_C;
  VP8LPredictorsSub[12] = PredictorSub12_C;
  VP8LPredictorsSub[13] = PredictorSub13_C;
  VP8LPredictorsSub[14] = PredictorSub0_C;  // <- padding security sentinels
  VP8LPredictorsSub[15] = PredictorSub0_C;

  VP8LPredictorsSub_C[0] = PredictorSub0_C;
  VP8LPredictorsSub_C[1] = PredictorSub1_C;
  VP8LPredictorsSub_C[2] = PredictorSub2_C;
  VP8LPredictorsSub_C[3] = PredictorSub3_C;
  VP8LPredictorsSub_C[4] = PredictorSub4_C;
  VP8LPredictorsSub_C[5] = PredictorSub5_C;
  VP8LPredictorsSub_C[6] = PredictorSub6_C;
  VP8LPredictorsSub_C[7] = PredictorSub7_C;
  VP8LPredictorsSub_C[8] = PredictorSub8_C;
  VP8LPredictorsSub_C[9] = PredictorSub9_C;
  VP8LPredictorsSub_C[10] = PredictorSub10_C;
  VP8LPredictorsSub_C[11] = PredictorSub11_C;
  VP8LPredictorsSub_C[12] = PredictorSub12_C;
  VP8LPredictorsSub_C[13] = PredictorSub13_C;
  VP8LPredictorsSub_C[14] = PredictorSub0_C;  // <- padding security sentinels
  VP8LPredictorsSub_C[15] = PredictorSub0_C;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_HAVE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8LEncDspInitSSE2();
#if defined(WEBP_HAVE_SSE41)
      if (VP8GetCPUInfo(kSSE4_1)) {
        VP8LEncDspInitSSE41();
#if defined(WEBP_HAVE_AVX2)
        if (VP8GetCPUInfo(kAVX2)) {
          VP8LEncDspInitAVX2();
#if defined(WEBP_HAVE_AVX512)
          if (VP8GetCPUInfo(kAVX512)) {
            VP8LEncDspInitAVX512();
          }
#endif
        }
#endif
      }
#endif
    }
#endif
#if defined(WEBP_USE_MIPS32)
    if (VP8GetCPUInfo(kMIPS32)) {
      VP8LEncDspInitMIPS32();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      VP8LEncDspInitMIPSdspR2();
    }
#endif
#if defined(WEBP_USE_MSA)
    if (VP8GetCPUInfo(kMSA)) {
      VP8LEncDspInitMSA();
    }
#endif
  }

#if defined(WEBP_HAVE_NEON)
  if (WEBP_NEON_OMIT_C_CODE ||
      (VP8GetCPUInfo != NULL && VP8GetCPUInfo(kNEON))) {
    VP8LEncDspInitNEON();
  }
#endif

  assert(VP8LSubtractGreenFromBlueAndRed != NULL);
  assert(VP8LTransformColor != NULL);
  assert(VP8LCollectColorBlueTransforms != NULL);
  assert(VP8LCollectColorRedTransforms != NULL);
  assert(VP8LFastLog2Slow != NULL);
  assert(VP8LFastSLog2Slow != NULL);
  assert(VP8LExtraCost != NULL);
  assert(VP8LCombinedShannonEntropy != NULL);
  assert(VP8LShannonEntropy != NULL);
  assert(VP8LGetEntropyUnrefined != NULL);
  assert(VP8LGetCombinedEntropyUnrefined != NULL);
  assert(VP8LAddVector != NULL);
  assert(VP8LAddVectorEq != NULL);
  assert(VP8LVectorMismatch != NULL);
  assert(VP8LBundleColorMap != NULL);
  assert(VP8LPredictorsSub[0] != NULL);
  assert(VP8LPredictorsSub[1] != NULL);
  assert(VP8LPredictorsSub[2] != NULL);
  assert(VP8LPredictorsSub[3] != NULL);
  assert(VP8LPredictorsSub[4] != NULL);
  assert(VP8LPredictorsSub[5] != NULL);
  assert(VP8LPredictorsSub[6] != NULL);
  assert(VP8LPredictorsSub[7] != NULL);
  assert(VP8LPredictorsSub[8] != NULL);
  assert(VP8LPredictorsSub[9] != NULL);
  assert(VP8LPredictorsSub[10] != NULL);
  assert(VP8LPredictorsSub[11] != NULL);
  assert(VP8LPredictorsSub[12] != NULL);
  assert(VP8LPredictorsSub[13] != NULL);
  assert(VP8LPredictorsSub[14] != NULL);
  assert(VP8LPredictorsSub[15] != NULL);
  assert(VP8LPredictorsSub_C[0] != NULL);
  assert(VP8LPredictorsSub_C[1] != NULL);
  assert(VP8LPredictorsSub_C[2] != NULL);
  assert(VP8LPredictorsSub_C[3] != NULL);
  assert(VP8LPredictorsSub_C[4] != NULL);
  assert(VP8LPredictorsSub_C[5] != NULL);
  assert(VP8LPredictorsSub_C[6] != NULL);
  assert(VP8LPredictorsSub_C[7] != NULL);
  assert(VP8LPredictorsSub_C[8] != NULL);
  assert(VP8LPredictorsSub_C[9] != NULL);
  assert(VP8LPredictorsSub_C[10] != NULL);
  assert(VP8LPredictorsSub_C[11] != NULL);
  assert(VP8LPredictorsSub_C[12] != NULL);
  assert(VP8LPredictorsSub_C[13] != NULL);
  assert(VP8LPredictorsSub_C[14] != NULL);
  assert(VP8LPredictorsSub_C[15] != NULL);
}

//------------------------------------------------------------------------------
