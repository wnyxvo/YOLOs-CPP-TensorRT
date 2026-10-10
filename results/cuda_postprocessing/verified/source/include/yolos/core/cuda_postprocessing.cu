#include "yolos/core/cuda_postprocessing.hpp"
#include "yolos/core/trt_utils.hpp"
#include <cub/cub.cuh>
#include <climits>

namespace yolos::cuda {
using post::Candidate;
using post::PacketHeader;

__global__ void filterCompact(const float* raw,int predictions,int classes,float conf,
                              PacketHeader* header,Candidate* records) {
    int d=blockIdx.x*blockDim.x+threadIdx.x;
    if (d>=predictions) return;
    int cls=0; float score=raw[4*predictions+d];
    for (int c=1;c<classes;++c) {
        float s=raw[(4+c)*predictions+d];
        if (s>score) {score=s;cls=c;}
    }
    if (!isfinite(score)) {atomicExch(&header->invalid,1u);return;}
    if (score<=conf) return;
    float cx=raw[d],cy=raw[predictions+d],w=raw[2*predictions+d],h=raw[3*predictions+d];
    Candidate a{cx-w*.5f,cy-h*.5f,cx+w*.5f,cy+h*.5f,score,cls,uint32_t(d),0};
    if (!isfinite(a.x1)||!isfinite(a.y1)||!isfinite(a.x2)||!isfinite(a.y2)) {
        atomicExch(&header->invalid,1u);return;
    }
    unsigned slot=atomicAdd(&header->candidateCount,1u);
    if (slot>=unsigned(predictions)) {atomicExch(&header->overflow,1u);return;}
    records[slot]=a;
}

__global__ void makeSortKeys(const PacketHeader* h,const Candidate* records,int capacity,
                             uint64_t* keys,uint32_t* values) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=capacity) return;
    values[i]=i;
    // Surviving scores are positive finite FP32. Inverted IEEE bits order by
    // score descending; low 32 bits order equal scores by source index ascending.
    keys[i]=i<int(h->candidateCount)
        ? (uint64_t(~__float_as_uint(records[i].score))<<32)|records[i].sourceIndex
        : UINT64_MAX;
}

__device__ float boxIou(const Candidate& a,const Candidate& b) {
    float inter=fmaxf(0.f,fminf(a.x2,b.x2)-fmaxf(a.x1,b.x1)) *
                fmaxf(0.f,fminf(a.y2,b.y2)-fmaxf(a.y1,b.y1));
    float area=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter;
    return area>0 ? inter/area : 0.f;
}

// Fixed-size launch with device-side iteration over only active tiles, avoiding
// a host read of candidateCount and avoiding a grid of 132*132 mostly empty tiles.
__global__ void suppressionMasks(const PacketHeader* header,const Candidate* records,
                                 const uint32_t* sorted,uint64_t* masks,int maskStride,float threshold) {
    __shared__ Candidate columns[64];
    int count=int(header->candidateCount), blocks=(count+63)/64;
    int lane=threadIdx.x;
    for (int tile=blockIdx.x;tile<blocks*blocks;tile+=gridDim.x) {
        int rowBlock=tile/blocks,colBlock=tile%blocks;
        if (colBlock<rowBlock) continue;
        int j=colBlock*64+lane,i=rowBlock*64+lane;
        if (j<count) columns[lane]=records[sorted[j]];
        __syncthreads();
        if (i<count) {
            Candidate a=records[sorted[i]]; uint64_t bits=0;
            int end=min(64,count-colBlock*64);
            int start=rowBlock==colBlock ? lane+1 : 0;
            for (int k=start;k<end;++k)
                if (a.classId==columns[k].classId && boxIou(a,columns[k])>threshold)
                    bits|=uint64_t(1)<<k;
            masks[size_t(i)*maskStride+colBlock]=bits;
        }
        __syncthreads();
    }
}

// Exact greedy NMS in the explicit total order. It is NOT the approximation
// that suppresses a box merely because any higher-scoring (possibly suppressed) box overlaps.
__global__ void selectGreedy(const PacketHeader* input,const Candidate* records,const uint32_t* sorted,
                             const uint64_t* masks,int stride,int maxDet,
                             PacketHeader* output,Candidate* results) {
    extern __shared__ uint64_t removed[];
    __shared__ int keep,chosen;
    int count=int(input->candidateCount),blocks=(count+63)/64;
    for (int c=threadIdx.x;c<blocks;c+=blockDim.x) removed[c]=0;
    if (threadIdx.x==0) {keep=0;chosen=0;}
    __syncthreads();
    for (int i=0;i<count && keep<maxDet;++i) {
        if (threadIdx.x==0) {
            chosen=!(removed[i/64]&(uint64_t(1)<<(i%64)));
            if (chosen) results[keep++]=records[sorted[i]];
        }
        __syncthreads();
        if (chosen)
            for (int c=i/64+threadIdx.x;c<blocks;c+=blockDim.x)
                removed[c]|=masks[size_t(i)*stride+c];
        __syncthreads();
    }
    if (threadIdx.x==0) *output={input->candidateCount,uint32_t(keep),input->overflow,input->invalid};
}

struct DetectionPostprocessor::Impl {
    int predictions,stride;
    bool gpuNms;
    trt::DeviceBuffer candidates,results,keysIn,keysOut,indicesIn,indicesOut,masks,scratch;
    trt::PinnedBuffer host;
    size_t scratchBytes{0};
    Impl(int n,bool useNms):predictions(n),stride((n+63)/64),gpuNms(useNms),
        candidates(sizeof(PacketHeader)+size_t(n)*sizeof(Candidate)),host(candidates.size()) {
        if (gpuNms) {
            results=trt::DeviceBuffer(candidates.size());
            keysIn=trt::DeviceBuffer(size_t(n)*sizeof(uint64_t)); keysOut=trt::DeviceBuffer(keysIn.size());
            indicesIn=trt::DeviceBuffer(size_t(n)*sizeof(uint32_t));indicesOut=trt::DeviceBuffer(indicesIn.size());
            masks=trt::DeviceBuffer(size_t(n)*stride*sizeof(uint64_t));
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,scratchBytes,
                static_cast<uint64_t*>(keysIn.data()),static_cast<uint64_t*>(keysOut.data()),
                static_cast<uint32_t*>(indicesIn.data()),static_cast<uint32_t*>(indicesOut.data()),n,0,64));
            scratch=trt::DeviceBuffer(scratchBytes);
        }
    }
    static PacketHeader* header(trt::DeviceBuffer& b) {return static_cast<PacketHeader*>(b.data());}
    static Candidate* rows(trt::DeviceBuffer& b) {return reinterpret_cast<Candidate*>(static_cast<char*>(b.data())+sizeof(PacketHeader));}
};

DetectionPostprocessor::DetectionPostprocessor(int predictions,bool gpuNms) {
    if (predictions<1||predictions>30000) throw std::invalid_argument("GPU postprocessing requires 1..30000 predictions; no implicit Top-K");
    impl_=std::make_unique<Impl>(predictions,gpuNms);
}
DetectionPostprocessor::~DetectionPostprocessor()=default;
void DetectionPostprocessor::filter(const float* raw,int classes,float conf,cudaStream_t stream) {
    if (!raw || classes<1 || !std::isfinite(conf) || conf<0 || conf>1)
        throw std::invalid_argument("GPU filter requires non-null output, positive classes, finite confidence in [0,1]");
    auto& p=*impl_;
    CUDA_CHECK(cudaMemsetAsync(p.candidates.data(),0,sizeof(PacketHeader),stream));
    filterCompact<<<(p.predictions+255)/256,256,0,stream>>>(raw,p.predictions,classes,conf,
        Impl::header(p.candidates),Impl::rows(p.candidates));
    CUDA_CHECK(cudaGetLastError());
}
void DetectionPostprocessor::nms(float iou,int maxDet,cudaStream_t stream) {
    if (!std::isfinite(iou)||iou<0||iou>1||maxDet<1||maxDet>30000)
        throw std::invalid_argument("Invalid GPU NMS settings");
    auto& p=*impl_;
    if (!p.gpuNms) throw std::logic_error("GPU NMS buffers were not configured");
    makeSortKeys<<<(p.predictions+255)/256,256,0,stream>>>(Impl::header(p.candidates),Impl::rows(p.candidates),p.predictions,
        static_cast<uint64_t*>(p.keysIn.data()),static_cast<uint32_t*>(p.indicesIn.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(p.scratch.data(),p.scratchBytes,
        static_cast<uint64_t*>(p.keysIn.data()),static_cast<uint64_t*>(p.keysOut.data()),
        static_cast<uint32_t*>(p.indicesIn.data()),static_cast<uint32_t*>(p.indicesOut.data()),p.predictions,0,64,stream));
    suppressionMasks<<<256,64,0,stream>>>(Impl::header(p.candidates),Impl::rows(p.candidates),
        static_cast<uint32_t*>(p.indicesOut.data()),static_cast<uint64_t*>(p.masks.data()),p.stride,iou);
    CUDA_CHECK(cudaGetLastError());
    selectGreedy<<<1,256,size_t(p.stride)*sizeof(uint64_t),stream>>>(Impl::header(p.candidates),Impl::rows(p.candidates),
        static_cast<uint32_t*>(p.indicesOut.data()),static_cast<uint64_t*>(p.masks.data()),p.stride,std::min(maxDet,p.predictions),
        Impl::header(p.results),Impl::rows(p.results));
    CUDA_CHECK(cudaGetLastError());
}
void DetectionPostprocessor::copyToHost(int maxDet,cudaStream_t stream) {
    auto& p=*impl_;
    CUDA_CHECK(cudaMemcpyAsync(p.host.data(),p.gpuNms?p.results.data():p.candidates.data(),
                              transferBytes(maxDet),cudaMemcpyDeviceToHost,stream));
}
const PacketHeader& DetectionPostprocessor::header() const {return *impl_->host.as<PacketHeader>();}
const Candidate* DetectionPostprocessor::records() const {
    return reinterpret_cast<const Candidate*>(static_cast<const char*>(impl_->host.data())+sizeof(PacketHeader));
}
size_t DetectionPostprocessor::transferBytes(int maxDet) const {
    return sizeof(PacketHeader)+size_t(impl_->gpuNms?std::min(maxDet,impl_->predictions):impl_->predictions)*sizeof(Candidate);
}
size_t DetectionPostprocessor::deviceBytes() const {
    const auto& p=*impl_;
    return p.candidates.size()+p.results.size()+p.keysIn.size()+p.keysOut.size()+p.indicesIn.size()+p.indicesOut.size()+p.masks.size()+p.scratch.size();
}
} // namespace yolos::cuda
