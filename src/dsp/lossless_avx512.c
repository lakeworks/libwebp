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

// TODO(lakeworks): Phase 4 implementation — widen AVX2 functions to 512-bit.
// For now, stub: the dispatch chain calls this but it sets no function pointers,
// so AVX2 implementations remain active.

WEBP_TSAN_IGNORE_FUNCTION void VP8LDspInitAVX512(void) {
  // Placeholder — AVX-512 lossless decoder functions will be added here.
}

#else  // !WEBP_USE_AVX512

WEBP_DSP_INIT_STUB(VP8LDspInitAVX512)

#endif  // WEBP_USE_AVX512
