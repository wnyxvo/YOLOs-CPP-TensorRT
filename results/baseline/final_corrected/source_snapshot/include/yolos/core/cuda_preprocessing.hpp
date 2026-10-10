#pragma once

// ============================================================================
// CUDA Preprocessing — Host API
// ============================================================================
// Declares the GPU-accelerated letterbox + BGR→RGB + normalize pipeline.
// Call letterboxPreprocess() to transform a raw uint8 BGR image on the
// device into NCHW float directly in the TRT input buffer.
//
// Author: YOLOs-TRT Team
// ============================================================================

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>

namespace yolos {
namespace cuda {

struct LetterboxGeometry { double scale; int newW, newH, padLeft, padTop; };
// Match Python round (ties to even) and use double for the host geometry.
inline LetterboxGeometry letterboxGeometry(int srcW, int srcH, int dstW, int dstH) {
    double r = std::min(double(dstH) / srcH, double(dstW) / srcW);
    int w = int(std::nearbyint(srcW * r)), h = int(std::nearbyint(srcH * r));
    if (w < 1 || h < 1) throw std::invalid_argument("Letterbox resized dimension is zero");
    return {r, w, h, int(std::nearbyint((dstW-w)/2.0 - 0.1)),
                     int(std::nearbyint((dstH-h)/2.0 - 0.1))};
}


/// @brief GPU letterbox + BGR→RGB + normalize in a single kernel.
///
/// The source image must already reside in GPU memory (HWC, uint8, BGR).
/// The output is written in NCHW float (RGB, normalised [0,1]) directly
/// into the caller-provided device buffer (e.g. TRT input buffer).
///
/// Padding colour is 114/255 (Ultralytics convention).
/// Asymmetric padding with -0.1/+0.1 rounding matches Ultralytics exactly.
///
/// @param d_src   Device pointer to source BGR uint8 image (H*W*3 bytes)
/// @param srcW    Source image width
/// @param srcH    Source image height
/// @param d_dst   Device pointer to destination NCHW float buffer
/// @param dstW    Destination (letterbox) width
/// @param dstH    Destination (letterbox) height
/// @param stream  CUDA stream for async execution
void letterboxPreprocess(
    const uint8_t* d_src, int srcW, int srcH,
    float* d_dst, int dstW, int dstH,
    cudaStream_t stream = nullptr
);

} // namespace cuda
} // namespace yolos
