#pragma once

// ============================================================================
// YOLO Object Detection (TensorRT Backend)
// ============================================================================
// Object detection using YOLO models with support for multiple versions
// (v7, v8, v10, v11, v26, NAS) through runtime auto-detection or explicit
// selection. Uses TensorRT for high-performance GPU inference.
//
// Author: YOLOs-TRT Team
// ============================================================================

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <cfloat>

#include "yolos/core/types.hpp"
#include "yolos/core/version.hpp"
#include "yolos/core/utils.hpp"
#include "yolos/core/preprocessing.hpp"
#include "yolos/core/nms.hpp"
#include "yolos/core/drawing.hpp"
#include "yolos/core/trt_session_base.hpp"
#include "yolos/core/cuda_postprocessing.hpp"

namespace yolos {
namespace det {

// ============================================================================
// Detection Result Structure
// ============================================================================

/// @brief Detection result containing bounding box, confidence, and class ID
struct Detection {
    BoundingBox box;    ///< Axis-aligned bounding box
    std::array<float, 4> xyxy{0, 0, 0, 0}; ///< Float coordinates; integer box is for drawing.
    float conf{0.0f};   ///< Confidence score
    int classId{-1};    ///< Class ID

    Detection() = default;
    Detection(const BoundingBox& box_, float conf_, int classId_)
        : box(box_), xyxy{float(box_.x), float(box_.y), float(box_.x + box_.width), float(box_.y + box_.height)}, conf(conf_), classId(classId_) {}
};

// ============================================================================
// YOLODetector Base Class
// ============================================================================

/// @brief Base YOLO detector with runtime version auto-detection
class YOLODetector : public TrtSessionBase {
public:
    /// @brief Constructor
    /// @param enginePath Path to the TensorRT engine file (.trt / .engine)
    /// @param labelsPath Path to the class names file
    /// @param version YOLO version (Auto for runtime detection)
    /// @param dlaCore DLA core index (-1 = GPU, 0/1 = DLA on Jetson)
    YOLODetector(const std::string& enginePath,
                 const std::string& labelsPath,
                 YOLOVersion version = YOLOVersion::Auto,
                 int dlaCore = -1)
        : TrtSessionBase(enginePath, dlaCore),
          version_(version) {
        classNames_ = utils::getClassNames(labelsPath);
        classColors_ = drawing::generateColors(classNames_);

        // Resolve version once at construction if Auto (avoids per-frame branch)
        if (version_ == YOLOVersion::Auto) {
            version_ = detectVersion();
        }
    }

    virtual ~YOLODetector() = default;

    /// @brief Run detection on an image — full GPU pipeline
    /// @param image Input image (BGR format)
    /// @param confThreshold Confidence threshold
    /// @param iouThreshold IoU threshold for NMS
    /// @return Vector of detections
    virtual std::vector<Detection> detect(const cv::Mat& image,
                                          float confThreshold = 0.4f,
                                          float iouThreshold = 0.45f) {
        configurePostprocess(confThreshold,iouThreshold);
        // Full GPU pipeline: pinned H2D → CUDA letterbox → TRT infer → D2H
        inferGpu(image);

        // Postprocess using cached scale/pad from GPU preprocess
        return postprocess(image.size(), inputShape_, version_, confThreshold, iouThreshold);
    }

    /// @brief Draw detections on an image
    void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) const {
        for (const auto& det : detections) {
            if (det.classId >= 0 && static_cast<size_t>(det.classId) < classNames_.size()) {
                std::string label = classNames_[det.classId] + ": " +
                                   std::to_string(static_cast<int>(det.conf * 100)) + "%";
                const cv::Scalar& color = classColors_[det.classId % classColors_.size()];
                drawing::drawBoundingBox(image, det.box, label, color);
            }
        }
    }

    /// @brief Draw detections with semi-transparent mask fill
    void drawDetectionsWithMask(cv::Mat& image, const std::vector<Detection>& detections, float alpha = 0.4f) const {
        for (const auto& det : detections) {
            if (det.classId >= 0 && static_cast<size_t>(det.classId) < classNames_.size()) {
                std::string label = classNames_[det.classId] + ": " +
                                   std::to_string(static_cast<int>(det.conf * 100)) + "%";
                const cv::Scalar& color = classColors_[det.classId % classColors_.size()];
                drawing::drawBoundingBoxWithMask(image, det.box, label, color, alpha);
            }
        }
    }

    enum class PostprocessMode { CPU, GPUFilter, GPUNms };
    void setPostprocessMode(const std::string& mode) {
        PostprocessMode next;
        if (mode=="cpu") next=PostprocessMode::CPU;
        else if (mode=="gpu-filter") next=PostprocessMode::GPUFilter;
        else if (mode=="gpu-nms") next=PostprocessMode::GPUNms;
        else throw std::invalid_argument("postprocess must be cpu, gpu-filter, or gpu-nms");
        if (next!=PostprocessMode::CPU) {
            const auto& shape=getOutputShape();
            if ((version_!=YOLOVersion::V8 && version_!=YOLOVersion::V11) || numOutputs()!=1 ||
                shape.size()!=3 || shape[0]!=1 || shape[1]!=int64_t(classNames_.size()+4) ||
                shape[2]<1 || shape[2]>30000)
                throw std::invalid_argument("GPU postprocessing requires standard YOLOv8/v11 [1,4+classes,N], N<=30000");
            gpuPost_=std::make_unique<cuda::DetectionPostprocessor>(int(shape[2]),next==PostprocessMode::GPUNms);
        } else gpuPost_.reset();
        postprocessMode_=next;
    }
    const char* postprocessModeName() const {
        return postprocessMode_==PostprocessMode::CPU ? "cpu" : postprocessMode_==PostprocessMode::GPUFilter ? "gpu-filter" : "gpu-nms";
    }
    size_t lastCandidateCount() const {return lastCandidateCount_;}
    size_t lastResultCount() const {return lastResultCount_;}
    size_t lastD2hBytes() const {return gpuPost_ ? gpuPost_->transferBytes(maxDet_) : outputTensors_[0].bytes;}
    size_t postprocessDeviceBytes() const {return gpuPost_ ? gpuPost_->deviceBytes() : 0;}

    void setMaxDet(int n) {
        if (n < 1 || n > 30000) throw std::invalid_argument("max_det must be in [1,30000]");
        maxDet_ = n;
    }

    /// @brief Get class names
    [[nodiscard]] const std::vector<std::string>& getClassNames() const { return classNames_; }

    /// @brief Get class colors
    [[nodiscard]] const std::vector<cv::Scalar>& getClassColors() const { return classColors_; }

protected:
    int maxDet_{300};
    PostprocessMode postprocessMode_{PostprocessMode::CPU};
    std::unique_ptr<cuda::DetectionPostprocessor> gpuPost_;
    float gpuConf_{.4f},gpuIou_{.45f};
    size_t lastCandidateCount_{0},lastResultCount_{0};
    void configurePostprocess(float conf,float iou) {
        if (!std::isfinite(conf)||!std::isfinite(iou)||conf<0||conf>1||iou<0||iou>1)
            throw std::invalid_argument("Detection thresholds must be finite in [0,1]");
        gpuConf_=conf;gpuIou_=iou;
    }
    void filterGpuOutputs() override {
        if (gpuPost_) gpuPost_->filter(static_cast<const float*>(outputTensors_[0].deviceBuf.data()),
                                       int(classNames_.size()),gpuConf_,stream_.get());
    }
    void nmsGpuOutputs() override {
        if (postprocessMode_==PostprocessMode::GPUNms) gpuPost_->nms(gpuIou_,maxDet_,stream_.get());
    }
    void copyOutputsToHost() override {
        if (gpuPost_) gpuPost_->copyToHost(maxDet_,stream_.get());
        else TrtSessionBase::copyOutputsToHost();
    }
    std::vector<Detection> finishCandidates(const std::vector<post::Candidate>& candidates,const cv::Size& size) {
        auto result=finishCandidatesWithGeometry(candidates,size,cachedScaleX_,cachedScaleY_,cachedPadX_,cachedPadY_);
        lastResultCount_=result.size();
        return result;
    }
    // Explicit geometry lets asynchronous slots retain their own inverse transform.
    static std::vector<Detection> finishCandidatesWithGeometry(const std::vector<post::Candidate>& candidates,
                                                              const cv::Size& size,float gainX,float gainY,
                                                              float padX,float padY) {
        std::vector<Detection> result;result.reserve(candidates.size());
        for (const auto& c:candidates) {
            Detection d;d.conf=c.score;d.classId=c.classId;
            d.xyxy={std::clamp((c.x1-padX)/gainX,0.f,float(size.width)),
                    std::clamp((c.y1-padY)/gainY,0.f,float(size.height)),
                    std::clamp((c.x2-padX)/gainX,0.f,float(size.width)),
                    std::clamp((c.y2-padY)/gainY,0.f,float(size.height))};
            int x=int(std::round(d.xyxy[0])),y=int(std::round(d.xyxy[1]));
            d.box=BoundingBox(x,y,int(std::round(d.xyxy[2]))-x,int(std::round(d.xyxy[3]))-y);
            result.push_back(d);
        }
        return result;
    }
    std::vector<Detection> postprocessCompressed(const cv::Size& size) {
        const auto& h=gpuPost_->header();
        const auto capacity=size_t(getOutputShape()[2]);
        if (h.overflow || h.candidateCount>capacity) throw std::runtime_error("GPU candidate capacity exceeded; no silent truncation");
        if (h.invalid) throw std::runtime_error("Non-finite GPU candidate/score");
        lastCandidateCount_=h.candidateCount;
        size_t count=postprocessMode_==PostprocessMode::GPUNms ? h.resultCount : h.candidateCount;
        if (count>capacity || (postprocessMode_==PostprocessMode::GPUNms && count>size_t(maxDet_)))
            throw std::runtime_error("Invalid GPU result count");
        std::vector<post::Candidate> candidates(gpuPost_->records(),gpuPost_->records()+count);
        if (postprocessMode_==PostprocessMode::GPUFilter) candidates=post::nms(std::move(candidates),gpuIou_,maxDet_);
        return finishCandidates(candidates,size);
    }

    YOLOVersion version_{YOLOVersion::Auto};
    std::vector<std::string> classNames_;
    std::vector<cv::Scalar> classColors_;

    /// @brief Detect YOLO version from output tensor shape
    YOLOVersion detectVersion() {
        const auto& outputShape = getOutputShape(0);
        return version::detectFromOutputShape(outputShape, numOutputs());
    }

    /// @brief Postprocess dispatch — uses cached scale/pad from GPU preprocess
    virtual std::vector<Detection> postprocess(const cv::Size& originalSize,
                                               const cv::Size& /*resizedShape*/,
                                               YOLOVersion version,
                                               float confThreshold,
                                               float iouThreshold) {
        if (gpuPost_) return postprocessCompressed(originalSize);
        // Use cached scale/pad computed during GPU preprocessing
        const float invScale = 1.0f / getCachedScale();
        const float padX = getCachedPadX();
        const float padY = getCachedPadY();

        switch (version) {
            case YOLOVersion::V7:
                return postprocessV7(originalSize, invScale, padX, padY, confThreshold, iouThreshold);
            case YOLOVersion::V10:
            case YOLOVersion::V26:
                return postprocessV10(originalSize, invScale, padX, padY, confThreshold);
            case YOLOVersion::NAS:
                return postprocessNAS(originalSize, invScale, padX, padY, confThreshold, iouThreshold);
            default:
                return postprocessStandard(originalSize, invScale, padX, padY, confThreshold, iouThreshold);
        }
    }

    /// @brief Standard postprocess for YOLOv8/v11 format [batch, features, boxes]
    std::vector<Detection> postprocessStandard(const cv::Size& originalSize,
                                               float invScale, float padX, float padY,
                                               float confThreshold, float iouThreshold) {
        const float* __restrict__ rawOutput = getOutputData(0);
        const auto& outputShape = getOutputShape(0);

        if (outputShape.size() != 3) throw std::runtime_error("Expected 3D standard detection output");
        const int numFeatures = static_cast<int>(outputShape[1]);
        const int numDets     = static_cast<int>(outputShape[2]);
        const int numClasses  = numFeatures - 4;

        if (numDets == 0 || numClasses <= 0) return {};

        if (outputShape.size() != 3 || outputShape[0] != 1 ||
            static_cast<size_t>(numClasses) != classNames_.size())
            throw std::runtime_error("Standard detection requires [1,4+classes,boxes] with matching labels");
        auto candidates=post::collect(rawOutput,numDets,numClasses,confThreshold);
        lastCandidateCount_=candidates.size();
        auto kept=post::nms(std::move(candidates),iouThreshold,maxDet_);
        (void)invScale;(void)padX;(void)padY;
        return finishCandidates(kept,originalSize);
    }

    /// @brief Postprocess for YOLOv7 format [batch, boxes, features]
    std::vector<Detection> postprocessV7(const cv::Size& originalSize,
                                         float invScale, float padX, float padY,
                                         float confThreshold, float iouThreshold) {
        const float* rawOutput = getOutputData(0);
        const auto& outputShape = getOutputShape(0);

        const int numDets    = static_cast<int>(outputShape[1]);
        const int numFeats   = static_cast<int>(outputShape[2]);
        const int numClasses = numFeats - 5;

        if (numDets == 0 || numClasses <= 0) return {};

        const int imgW = originalSize.width, imgH = originalSize.height;

        std::vector<BoundingBox> boxes;  boxes.reserve(512);
        std::vector<float> confs;        confs.reserve(512);
        std::vector<int> classIds;       classIds.reserve(512);

        for (int d = 0; d < numDets; ++d) {
            const float* row = rawOutput + d * numFeats;
            if (row[4] <= confThreshold) continue;

            int bestC = 0; float bestS = row[5];
            for (int c = 1; c < numClasses; ++c) {
                if (row[5 + c] > bestS) { bestS = row[5 + c]; bestC = c; }
            }

            const float left = (row[0] - row[2] * 0.5f - padX) * invScale;
            const float top  = (row[1] - row[3] * 0.5f - padY) * invScale;

            BoundingBox box;
            box.x      = utils::clamp(static_cast<int>(left), 0, imgW - 1);
            box.y      = utils::clamp(static_cast<int>(top),  0, imgH - 1);
            box.width  = utils::clamp(static_cast<int>(row[2] * invScale), 1, imgW - box.x);
            box.height = utils::clamp(static_cast<int>(row[3] * invScale), 1, imgH - box.y);

            boxes.push_back(box); confs.push_back(row[4]); classIds.push_back(bestC);
        }

        std::vector<int> indices;
        nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold, indices);

        std::vector<Detection> dets; dets.reserve(indices.size());
        for (int idx : indices) dets.emplace_back(boxes[idx], confs[idx], classIds[idx]);
        return dets;
    }

    /// @brief Postprocess for YOLOv10/v26 format [batch, boxes, 6] (end-to-end, no NMS)
    std::vector<Detection> postprocessV10(const cv::Size& originalSize,
                                          float invScale, float padX, float padY,
                                          float confThreshold) {
        const float* rawOutput = getOutputData(0);
        const int numDets = static_cast<int>(getOutputShape(0)[1]);
        const int imgW = originalSize.width, imgH = originalSize.height;

        std::vector<Detection> dets; dets.reserve(numDets);

        for (int i = 0; i < numDets; ++i) {
            const float* r = rawOutput + i * 6;
            if (r[4] <= confThreshold) continue;

            const float x1 = (r[0] - padX) * invScale;
            const float y1 = (r[1] - padY) * invScale;
            const float x2 = (r[2] - padX) * invScale;
            const float y2 = (r[3] - padY) * invScale;

            BoundingBox box;
            box.x      = utils::clamp(static_cast<int>(x1), 0, imgW - 1);
            box.y      = utils::clamp(static_cast<int>(y1), 0, imgH - 1);
            box.width  = utils::clamp(static_cast<int>(x2 - x1), 1, imgW - box.x);
            box.height = utils::clamp(static_cast<int>(y2 - y1), 1, imgH - box.y);

            dets.emplace_back(box, r[4], static_cast<int>(r[5]));
        }
        return dets;
    }

    /// @brief Postprocess for YOLO-NAS format (two outputs: boxes and scores)
    std::vector<Detection> postprocessNAS(const cv::Size& originalSize,
                                          float invScale, float padX, float padY,
                                          float confThreshold, float iouThreshold) {
        if (numOutputs() < 2)
            return postprocessStandard(originalSize, invScale, padX, padY, confThreshold, iouThreshold);

        const float* boxOut   = getOutputData(0);
        const float* scoreOut = getOutputData(1);
        const int numDets     = static_cast<int>(getOutputShape(0)[1]);
        const int numClasses  = static_cast<int>(getOutputShape(1)[2]);
        const int imgW = originalSize.width, imgH = originalSize.height;

        std::vector<BoundingBox> boxes; boxes.reserve(512);
        std::vector<float> confs;       confs.reserve(512);
        std::vector<int> classIds;      classIds.reserve(512);

        for (int i = 0; i < numDets; ++i) {
            int bestC = 0; float bestS = scoreOut[i * numClasses];
            for (int c = 1; c < numClasses; ++c) {
                const float s = scoreOut[i * numClasses + c];
                if (s > bestS) { bestS = s; bestC = c; }
            }
            if (bestS <= confThreshold) continue;

            const float x1 = (boxOut[i * 4 + 0] - padX) * invScale;
            const float y1 = (boxOut[i * 4 + 1] - padY) * invScale;
            const float x2 = (boxOut[i * 4 + 2] - padX) * invScale;
            const float y2 = (boxOut[i * 4 + 3] - padY) * invScale;

            BoundingBox box;
            box.x      = utils::clamp(static_cast<int>(x1), 0, imgW - 1);
            box.y      = utils::clamp(static_cast<int>(y1), 0, imgH - 1);
            box.width  = utils::clamp(static_cast<int>(x2 - x1), 1, imgW - box.x);
            box.height = utils::clamp(static_cast<int>(y2 - y1), 1, imgH - box.y);

            boxes.push_back(box); confs.push_back(bestS); classIds.push_back(bestC);
        }

        std::vector<int> indices;
        nms::NMSBoxesBatched(boxes, confs, classIds, confThreshold, iouThreshold, indices);

        std::vector<Detection> dets; dets.reserve(indices.size());
        for (int idx : indices) dets.emplace_back(boxes[idx], confs[idx], classIds[idx]);
        return dets;
    }
};

// ============================================================================
// Version-Specific Detector Subclasses
// ============================================================================

/// @brief YOLOv7 detector (forces V7 postprocessing)
class YOLOv7Detector : public YOLODetector {
public:
    YOLOv7Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V7, dlaCore) {}
};

/// @brief YOLOv8 detector (forces standard postprocessing)
class YOLOv8Detector : public YOLODetector {
public:
    YOLOv8Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V8, dlaCore) {}
};

/// @brief YOLOv10 detector (forces V10 end-to-end postprocessing)
class YOLOv10Detector : public YOLODetector {
public:
    YOLOv10Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V10, dlaCore) {}
};

/// @brief YOLOv11 detector (forces standard postprocessing)
class YOLOv11Detector : public YOLODetector {
public:
    YOLOv11Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V11, dlaCore) {}
};

/// @brief YOLO-NAS detector (forces NAS postprocessing)
class YOLONASDetector : public YOLODetector {
public:
    YOLONASDetector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::NAS, dlaCore) {}
};

/// @brief YOLOv26 detector (forces V26 end-to-end postprocessing)
class YOLO26Detector : public YOLODetector {
public:
    YOLO26Detector(const std::string& enginePath, const std::string& labelsPath, int dlaCore = -1)
        : YOLODetector(enginePath, labelsPath, YOLOVersion::V26, dlaCore) {}
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a detector with explicit version selection
inline std::unique_ptr<YOLODetector> createDetector(const std::string& enginePath,
                                                    const std::string& labelsPath,
                                                    YOLOVersion version = YOLOVersion::Auto,
                                                    int dlaCore = -1) {
    switch (version) {
        case YOLOVersion::V7:
            return std::make_unique<YOLOv7Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V8:
            return std::make_unique<YOLOv8Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V10:
            return std::make_unique<YOLOv10Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V11:
            return std::make_unique<YOLOv11Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::V26:
            return std::make_unique<YOLO26Detector>(enginePath, labelsPath, dlaCore);
        case YOLOVersion::NAS:
            return std::make_unique<YOLONASDetector>(enginePath, labelsPath, dlaCore);
        default:
            return std::make_unique<YOLODetector>(enginePath, labelsPath, YOLOVersion::Auto, dlaCore);
    }
}

} // namespace det
} // namespace yolos
