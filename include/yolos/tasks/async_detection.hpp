#pragma once
#include "yolos/tasks/detection.hpp"
#include <array>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#ifdef YOLOS_PIPELINE_NVTX
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>
#endif

namespace yolos::det {

// Single producer / single collector. Configuration is exclusive to both.
// One existing ExecutionContext and the original inference-only Graph are shared.
class AsyncYOLODetector : public YOLODetector {
public:
    using YOLODetector::YOLODetector;
    using Clock = std::chrono::steady_clock;
    struct Result {
        uint64_t frameId{0};
        cv::Size originalSize;
        std::vector<Detection> detections;
        size_t candidates{0},d2hBytes{0},slot{0};
        double latencyMs{0},admittedLatencyMs{0},submitWaitMs{0},submitHostMs{0};
        double stagingMs{0},collectionQueueMs{0},collectEventWaitMs{0},cpuPostMs{0};
    };
    struct Counters { size_t accepted{0},collected{0},highWater{0},capacityWaits{0}; };

    ~AsyncYOLODetector() override {
        if (!async_) return;
        size_t pending=0;
        {
            std::lock_guard<std::mutex> lock(async_->mutex);
            async_->closed=true;pending=async_->pending;
        }
        async_->changed.notify_all();
        // Caller must join producer/collector before destroying the detector.
        // Finish all device users before slot buffers or the shared Graph/context die.
        try { async_->upload.synchronize();stream_.synchronize();async_->download.synchronize(); }
        catch (const std::exception& e) {std::cerr << "Async shutdown: " << e.what() << '\n';}
        if (pending) std::cerr << "[WARN] " << pending << " accepted frames were not collected; call drain()/stopAsync() before destruction\n";
    }

    void setPostprocessMode(const std::string& mode) { requireIdleConfiguration();YOLODetector::setPostprocessMode(mode); }
    void setMaxDet(int limit) { requireIdleConfiguration();YOLODetector::setMaxDet(limit); }
    void setGraphEnabled(bool enabled) { requireIdleConfiguration();TrtSessionBase::setGraphEnabled(enabled); }
    void setProfiling(bool enabled) { requireIdleConfiguration();TrtSessionBase::setProfiling(enabled); }
    std::vector<Detection> detect(const cv::Mat& image,float conf=.4f,float iou=.45f) override {
        requireIdleConfiguration();return YOLODetector::detect(image,conf,iou);
    }

    void enableAsync(size_t slots,size_t maxImageBytes,float conf=.25f,float iou=.45f,bool timeline=false) {
        requireIdleConfiguration();
        if (slots<1 || slots>3 || maxImageBytes<3) throw std::invalid_argument("Async requires 1..3 slots and a positive image-byte capacity");
        if (postprocessMode_!=PostprocessMode::GPUFilter)
            throw std::invalid_argument("Initial async implementation supports gpu-filter + CPU NMS only; all synchronous modes remain available");
        if (profiling_) throw std::invalid_argument("Disable synchronous stage profiling before async; use a separate Nsight timeline");
#ifndef YOLOS_PIPELINE_NVTX
        if (timeline) throw std::invalid_argument("This build has no NVTX annotations");
#endif
        configurePostprocess(conf,iou);
        stream_.synchronize(); // setup only, after any earlier synchronous use
        int device=0;CUDA_CHECK(cudaGetDevice(&device));
        auto state=std::make_unique<State>(slots,maxImageBytes,int(getOutputShape()[2]),device,conf,iou,maxDet_,timeline);
        async_=std::move(state);
#ifdef YOLOS_PIPELINE_NVTX
        if (timeline) {
            nvtxNameCudaStreamA(async_->upload.get(),"pipeline upload");
            nvtxNameCudaStreamA(stream_.get(),"pipeline compute (single Context)");
            nvtxNameCudaStreamA(async_->download.get(),"pipeline download");
        }
#endif
    }

    // Blocks on a full ring. The image must stay unchanged until this returns;
    // after return no cv::Mat storage is retained, and the caller may release it.
    // Strictly increasing caller frame IDs identify every accepted frame.
    void submit(const cv::Mat& image,uint64_t frameId) {
        auto started=Clock::now();
        if (!async_) throw std::logic_error("enableAsync must precede submit");
        if (image.empty() || image.dims!=2 || image.type()!=CV_8UC3)
            throw std::invalid_argument("Async requires a non-empty 2D CV_8UC3 BGR image");
        auto& state=*async_;
        const size_t bytes=size_t(image.rows)*size_t(image.cols)*3;
        if (bytes>state.maxImageBytes) throw std::invalid_argument("Image exceeds reserved async capacity; reconfigure while idle, never allocate per frame");
        const auto geometry=cuda::letterboxGeometry(image.cols,image.rows,inputShape_.width,inputShape_.height);
        std::unique_lock<std::mutex> lock(state.mutex);
        rethrowFailure(state);
        if (state.producerBusy) throw std::logic_error("Concurrent submit calls are unsupported; use one producer");
        if (state.closed) throw std::logic_error("Async submissions are closed");
        if (state.hasId && frameId<=state.lastId) throw std::invalid_argument("Frame IDs must increase strictly");
        state.producerBusy=true;
        if (state.slots[state.nextSubmit]->phase!=Phase::Free) ++state.capacityWaits;
        state.changed.wait(lock,[&]{return state.closed || state.failure || state.slots[state.nextSubmit]->phase==Phase::Free;});
        if (state.closed || state.failure) {
            state.producerBusy=false;state.changed.notify_all();rethrowFailure(state);
            throw std::logic_error("Async submissions closed while waiting for capacity");
        }
        const size_t index=state.nextSubmit;
        auto& slot=*state.slots[index];slot.phase=Phase::Submitting;
        slot.started=started;slot.admitted=Clock::now();slot.frameId=frameId;slot.size=image.size();
        slot.gainX=float(geometry.newW)/image.cols;slot.gainY=float(geometry.newH)/image.rows;
        slot.padX=float(geometry.padLeft);slot.padY=float(geometry.padTop);
        lock.unlock();
        try {
            CUDA_CHECK(cudaSetDevice(state.device));
            auto copyStart=Clock::now();
            if (image.isContinuous()) std::memcpy(slot.hostSrc.data(),image.data,bytes);
            else for (int y=0;y<image.rows;++y)
                std::memcpy(static_cast<uint8_t*>(slot.hostSrc.data())+size_t(y)*image.cols*3,image.ptr(y),size_t(image.cols)*3);
            slot.stagingMs=milliseconds(Clock::now()-copyStart);
            CUDA_CHECK(cudaMemcpyAsync(slot.deviceSrc.data(),slot.hostSrc.data(),bytes,cudaMemcpyHostToDevice,state.upload.get()));
            slot.uploaded.record(state.upload.get());
            CUDA_CHECK(cudaStreamWaitEvent(stream_.get(),slot.uploaded.get(),0));
            cuda::letterboxPreprocess(static_cast<const uint8_t*>(slot.deviceSrc.data()),image.cols,image.rows,
                static_cast<float*>(inputTensors_[0].deviceBuf.data()),inputShape_.width,inputShape_.height,stream_.get());
            CUDA_CHECK(cudaGetLastError());
            if (isGraphEnabled()) CUDA_CHECK(cudaGraphLaunch(graphExec_,stream_.get()));
            else if (!context_->enqueueV3(stream_.get())) throw std::runtime_error("TensorRT async enqueueV3 failed");
            // The shared output is consumed before the NEXT inference on this stream.
            // D2H uses a per-slot candidate packet, never the shared Engine output.
            slot.post.filter(static_cast<const float*>(outputTensors_[0].deviceBuf.data()),int(classNames_.size()),state.conf,stream_.get());
            slot.computed.record(stream_.get());
            CUDA_CHECK(cudaStreamWaitEvent(state.download.get(),slot.computed.get(),0));
            slot.post.copyToHost(state.maxDet,state.download.get());
            slot.downloaded.record(state.download.get());
            slot.enqueued=Clock::now();
            lock.lock();
            slot.phase=Phase::Queued;
            state.order[state.tail]=index;state.tail=(state.tail+1)%state.slots.size();++state.pending;
            state.highWater=std::max(state.highWater,state.pending);
            state.nextSubmit=(state.nextSubmit+1)%state.slots.size();
            state.lastId=frameId;state.hasId=true;++state.accepted;state.producerBusy=false;
            lock.unlock();state.changed.notify_all();
        } catch (...) {
            if (!lock.owns_lock()) lock.lock();
            state.producerBusy=false;state.failure=std::current_exception();state.closed=true;
            lock.unlock();state.changed.notify_all();throw;
        }
    }

    // Blocking FIFO collection. nullopt only after closeSubmissions and full drain.
    // CPU NMS runs outside the mutex while the producer queues later GPU work.
    std::optional<Result> collect() {
        if (!async_) throw std::logic_error("enableAsync must precede collect");
        auto& state=*async_;
        std::unique_lock<std::mutex> lock(state.mutex);
        rethrowFailure(state);
        if (state.collectorBusy) throw std::logic_error("Concurrent collect calls are unsupported; use one collector");
        state.collectorBusy=true;
        state.changed.wait(lock,[&]{return state.failure || state.pending || (state.closed && !state.producerBusy);});
        if (state.failure) {state.collectorBusy=false;state.changed.notify_all();rethrowFailure(state);}
        if (!state.pending) {state.collectorBusy=false;state.changed.notify_all();return std::nullopt;}
        const size_t index=state.order[state.head];
        auto& slot=*state.slots[index];slot.phase=Phase::Collecting;
        auto collectStarted=Clock::now();
        lock.unlock();
        try {
            CUDA_CHECK(cudaSetDevice(state.device));
            slot.downloaded.synchronize(); // only this slot's LAST device user
            auto downloaded=Clock::now();
            const auto& header=slot.post.header();
            if (header.invalid || header.overflow || header.candidateCount>size_t(getOutputShape()[2]))
                throw std::runtime_error("Invalid/overflowed async candidate packet");
            TraceRange cpuPost(state.timeline,"CPU NMS and result packaging");
            std::vector<post::Candidate> candidates(slot.post.records(),slot.post.records()+header.candidateCount);
            auto kept=post::nms(std::move(candidates),state.iou,state.maxDet);
            Result result;
            result.frameId=slot.frameId;result.originalSize=slot.size;result.slot=index;
            result.candidates=header.candidateCount;result.d2hBytes=slot.post.transferBytes(state.maxDet);
            result.detections=finishCandidatesWithGeometry(kept,slot.size,slot.gainX,slot.gainY,slot.padX,slot.padY);
            auto ready=Clock::now();
            result.latencyMs=milliseconds(ready-slot.started);
            result.admittedLatencyMs=milliseconds(ready-slot.admitted);
            result.submitWaitMs=milliseconds(slot.admitted-slot.started);
            result.submitHostMs=milliseconds(slot.enqueued-slot.started);
            result.stagingMs=slot.stagingMs;
            result.collectionQueueMs=milliseconds(collectStarted-slot.enqueued);
            result.collectEventWaitMs=milliseconds(downloaded-collectStarted);
            result.cpuPostMs=milliseconds(ready-downloaded);
            lock.lock();
            state.head=(state.head+1)%state.slots.size();--state.pending;
            slot.phase=Phase::Free;++state.collected;state.collectorBusy=false;
            lock.unlock();state.changed.notify_all();
            return result; // owns its detections; no slot references escape
        } catch (...) {
            if (!lock.owns_lock()) lock.lock();
            state.collectorBusy=false;state.failure=std::current_exception();state.closed=true;
            lock.unlock();state.changed.notify_all();throw;
        }
    }

    void closeSubmissions() {
        if (!async_) return;
        {std::lock_guard<std::mutex> lock(async_->mutex);async_->closed=true;}
        async_->changed.notify_all();
    }
    std::vector<Result> drain() {
        closeSubmissions();std::vector<Result> pending;
        while (auto result=collect()) pending.push_back(std::move(*result));
        return pending;
    }
    // New closed-and-drained batch; preserves ALL CUDA addresses and allocations.
    void restartAsyncBatch() {
        if (!async_) throw std::logic_error("Async is not enabled");
        auto& state=*async_;std::lock_guard<std::mutex> lock(state.mutex);
        rethrowFailure(state);
        if (!state.closed || state.pending || state.producerBusy || state.collectorBusy)
            throw std::logic_error("Close, drain and join callers before restarting a batch");
        state.closed=false;state.hasId=false;
        state.accepted=state.collected=state.highWater=state.capacityWaits=0;
        state.head=state.tail=state.nextSubmit=0;
    }
    // Producer must be joined; returns every accepted, uncollected frame before reset.
    std::vector<Result> stopAsync() {
        if (!async_) return {};
        auto results=drain();
        {std::lock_guard<std::mutex> lock(async_->mutex);
         if (async_->producerBusy || async_->collectorBusy) throw std::logic_error("Join callers before stopAsync");}
        async_.reset();return results;
    }
    Counters counters() const {
        if (!async_) return {};
        std::lock_guard<std::mutex> lock(async_->mutex);
        return {async_->accepted,async_->collected,async_->highWater,async_->capacityWaits};
    }
    void writeAsyncResources(std::ostream& out) const {
        if (!async_) throw std::logic_error("Async is not enabled");
        std::lock_guard<std::mutex> lock(async_->mutex);
        out << "resource,index,host_source,device_source,host_candidates,upload_done,compute_done,download_done,source_bytes,candidate_device_bytes\n";
        for (size_t i=0;i<async_->slots.size();++i) {
            const auto& s=*async_->slots[i];
            out << "slot," << i << ',' << s.hostSrc.data() << ',' << s.deviceSrc.data() << ',' << s.post.records()
                << ',' << s.uploaded.get() << ',' << s.computed.get() << ',' << s.downloaded.get() << ','
                << async_->maxImageBytes << ',' << s.post.deviceBytes() << '\n';
        }
        out << "shared_input,0,," << inputTensors_[0].deviceBuf.data() << ",,,,," << inputTensors_[0].bytes << ",0\n";
        out << "shared_output,0,," << outputTensors_[0].deviceBuf.data() << ",,,,," << outputTensors_[0].bytes << ",0\n";
        out << "upload_stream,0,," << async_->upload.get() << ",,,,,0,0\n";
        out << "compute_stream,0,," << stream_.get() << ",,,,,0,0\n";
        out << "download_stream,0,," << async_->download.get() << ",,,,,0,0\n";
    }

private:
    class Event {
    public:
        Event() {CUDA_CHECK(cudaEventCreateWithFlags(&event_,cudaEventDisableTiming));}
        ~Event() {if (event_) cudaEventDestroy(event_);}
        Event(const Event&)=delete;Event& operator=(const Event&)=delete;
        void record(cudaStream_t stream) {CUDA_CHECK(cudaEventRecord(event_,stream));}
        void synchronize() {CUDA_CHECK(cudaEventSynchronize(event_));}
        cudaEvent_t get() const {return event_;}
    private: cudaEvent_t event_{nullptr};
    };
    struct TraceRange {
        bool enabled;
        TraceRange(bool on,const char* name):enabled(on) {
#ifdef YOLOS_PIPELINE_NVTX
            if (enabled) nvtxRangePushA(name);
#else
            (void)name;
#endif
        }
        ~TraceRange() {
#ifdef YOLOS_PIPELINE_NVTX
            if (enabled) nvtxRangePop();
#endif
        }
    };
    enum class Phase { Free,Submitting,Queued,Collecting };
    struct Slot {
        trt::PinnedBuffer hostSrc;
        trt::DeviceBuffer deviceSrc;
        cuda::DetectionPostprocessor post;
        Event uploaded,computed,downloaded;
        Phase phase{Phase::Free};
        uint64_t frameId{0};cv::Size size;
        float gainX{1},gainY{1},padX{0},padY{0};
        Clock::time_point started,admitted,enqueued;double stagingMs{0};
        Slot(size_t bytes,int predictions):hostSrc(bytes),deviceSrc(bytes),post(predictions,false) {}
    };
    struct State {
        std::vector<std::unique_ptr<Slot>> slots;
        trt::CudaStream upload,download;
        mutable std::mutex mutex;std::condition_variable changed;
        std::array<size_t,3> order{};
        size_t head{0},tail{0},pending{0},nextSubmit{0},maxImageBytes;
        size_t accepted{0},collected{0},highWater{0},capacityWaits{0};
        int device,maxDet;float conf,iou;
        bool closed{false},producerBusy{false},collectorBusy{false},hasId{false},timeline{false};uint64_t lastId{0};
        std::exception_ptr failure;
        State(size_t count,size_t bytes,int predictions,int deviceId,float confidence,float threshold,int limit,bool annotate)
            :maxImageBytes(bytes),device(deviceId),maxDet(limit),conf(confidence),iou(threshold),timeline(annotate) {
            slots.reserve(count);
            for (size_t i=0;i<count;++i) slots.push_back(std::make_unique<Slot>(bytes,predictions));
        }
    };
    std::unique_ptr<State> async_;
    static double milliseconds(Clock::duration delta) {return std::chrono::duration<double,std::milli>(delta).count();}
    void requireIdleConfiguration() const {
        if (async_) throw std::logic_error("Stop and drain the async pipeline before synchronous detect or configuration changes");
    }
    static void rethrowFailure(const State& state) {if (state.failure) std::rethrow_exception(state.failure);}
};
} // namespace yolos::det
