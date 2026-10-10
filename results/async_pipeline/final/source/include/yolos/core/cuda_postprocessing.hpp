#pragma once
#include <cuda_runtime.h>
#include <memory>
#include "yolos/core/detection_postprocessing.hpp"

namespace yolos::cuda {
// Persistent buffers and scratch storage; no per-frame cudaMalloc and no internal synchronization.
class DetectionPostprocessor {
public:
    explicit DetectionPostprocessor(int predictions, bool gpuNms);
    ~DetectionPostprocessor();
    DetectionPostprocessor(const DetectionPostprocessor&)=delete;
    DetectionPostprocessor& operator=(const DetectionPostprocessor&)=delete;
    void filter(const float* output,int classes,float conf,cudaStream_t stream);
    void nms(float iou,int maxDet,cudaStream_t stream);
    void copyToHost(int maxDet,cudaStream_t stream);
    const post::PacketHeader& header() const;
    const post::Candidate* records() const;
    size_t transferBytes(int maxDet) const;
    size_t deviceBytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yolos::cuda
