// Independent GPU-vs-CPU regression tests for ordering, greedy semantics and capacity.
#include "yolos/core/cuda_postprocessing.hpp"
#include "yolos/core/trt_utils.hpp"
#include <fstream>
#include <iomanip>
#include <random>
#include <limits>
#include <filesystem>

using namespace yolos;
static void require(bool value,const std::string& why) {if (!value) throw std::runtime_error(why);}
static void check(const std::string& name,const std::vector<float>& raw,int boxes,int classes,float conf,float iou,int maxDet,
                  std::ofstream& report,int expected=-1) {
    auto candidates=post::collect(raw.data(),boxes,classes,conf);
    auto wanted=post::nms(candidates,iou,maxDet);
    if (expected>=0) require(wanted.size()==size_t(expected),name+": CPU reference size");
    trt::DeviceBuffer output(raw.size()*sizeof(float));trt::CudaStream stream;
    CUDA_CHECK(cudaMemcpyAsync(output.data(),raw.data(),output.size(),cudaMemcpyHostToDevice,stream.get()));
    for (bool gpuNms : {false,true}) {
        cuda::DetectionPostprocessor p(boxes,gpuNms);
        // Repeat on the same buffers to catch stale counts/masks and atomic order changes.
        for (int repeat=0;repeat<3;++repeat) {
            p.filter(static_cast<float*>(output.data()),classes,conf,stream.get());
            if (gpuNms) p.nms(iou,maxDet,stream.get());
            p.copyToHost(maxDet,stream.get());stream.synchronize();
            const auto& h=p.header();
            require(!h.invalid&&!h.overflow&&h.candidateCount==candidates.size(),name+": candidate count/flags");
            size_t n=gpuNms?h.resultCount:h.candidateCount;
            std::vector<post::Candidate> got(p.records(),p.records()+n);
            if (!gpuNms) got=post::nms(std::move(got),iou,maxDet);
            require(got.size()==wanted.size(),name+": kept count");
            for (size_t i=0;i<got.size();++i) {
                const auto& a=got[i];const auto& b=wanted[i];
                require(a.sourceIndex==b.sourceIndex&&a.classId==b.classId&&a.score==b.score&&
                        a.x1==b.x1&&a.y1==b.y1&&a.x2==b.x2&&a.y2==b.y2,name+": record/order differs");
            }
        }
        report << name << ',' << (gpuNms?"gpu-nms":"gpu-filter") << ',' << boxes << ',' << candidates.size()
               << ',' << wanted.size() << ',' << p.transferBytes(maxDet) << ",PASS\n";
    }
}
int main(int argc,char** argv) try {
    std::ofstream report(argc>1?argv[1]:"cuda_postprocessing_tests.csv");
    report << "case,mode,predictions,candidates,results,d2h_bytes,status\n";
    const int n=32,c=4;
    std::vector<float> raw(size_t(n)*(4+c),0);
    auto clear=[&]{std::fill(raw.begin(),raw.end(),0.f);};
    auto put=[&](int i,int cls,float score,float x1,float y1,float x2,float y2) {
        raw[i]=(x1+x2)*.5f;raw[n+i]=(y1+y2)*.5f;raw[2*n+i]=x2-x1;raw[3*n+i]=y2-y1;raw[(4+cls)*n+i]=score;
    };
    check("empty",raw,n,c,.25f,.45f,300,report,0);
    put(3,0,.9f,0,0,100,100);put(1,0,.9f,0,0,100,100);put(9,1,.9f,0,0,100,100);
    raw[(4+2)*n+9]=.9f;
    check("score_tie_class_tie_class_aware",raw,n,c,.25f,.45f,300,report,2);
    check("max_det_1",raw,n,c,.25f,.45f,1,report,1);
    clear();put(0,0,.25f,0,0,10,10);put(1,0,std::nextafter(.25f,1.f),20,20,30,30);
    check("confidence_equal_boundary",raw,n,c,.25f,.45f,300,report,1);
    clear();put(0,0,.9f,0,0,3,2);put(1,0,.8f,1,0,4,2);
    check("iou_equal_boundary",raw,n,c,.25f,.5f,300,report,2);
    check("iou_nextafter_boundary",raw,n,c,.25f,std::nextafter(.5f,0.f),300,report,1);
    clear();put(0,0,.9f,-100,100,100,200);put(1,0,.8f,0,100,100,200);
    check("nms_before_clipping",raw,n,c,.25f,.6f,300,report,2);
    // Chain: A suppresses B, B overlaps C, A does not overlap C. Exact greedy keeps A,C.
    clear();put(0,0,.9f,0,0,10,10);put(1,0,.8f,4,0,14,10);put(2,0,.7f,8,0,18,10);
    check("greedy_chain_not_fast_nms",raw,n,c,.25f,.3f,300,report,2);
    clear();put(0,0,.9f,.4f,20,10.4f,30);put(1,0,.8f,.8f,20,10.8f,30);
    check("float_no_rounding",raw,n,c,.25f,.95f,300,report,2);
    clear();put(0,0,.9f,0,0,0,0);put(1,0,.8f,0,0,0,0);
    check("zero_area",raw,n,c,.25f,.45f,300,report,2);
    // More than one 64-bit mask tile and a partial final tile.
    for (int count : {65,127,129,513,8400}) {
        std::vector<float> all(size_t(count)*(4+c),0);
        for (int d=0;d<count;++d) {
            all[d]=float((d%100)*8);all[count+d]=float((d/100)*8);all[2*count+d]=4;all[3*count+d]=4;
            all[(4+d%c)*count+d]=.9f;
        }
        check("all_candidates_capacity_"+std::to_string(count),all,count,c,.25f,.45f,300,report,std::min(count,300));
    }
    std::mt19937 gen(1729);std::uniform_real_distribution<float> coord(-30,700),score(0,1),size(0,100);
    const int randomN=2049,randomC=80;
    std::vector<float> random(size_t(randomN)*(4+randomC));
    for (int d=0;d<randomN;++d) {
        random[d]=coord(gen);random[randomN+d]=coord(gen);random[2*randomN+d]=size(gen);random[3*randomN+d]=size(gen);
        for (int cl=0;cl<randomC;++cl) random[(4+cl)*randomN+d]=std::floor(score(gen)*16)/16;
    }
    check("random_dense_ties",random,randomN,randomC,.8f,.45f,300,report);
    // Finite-output guard and unsupported capacity must fail explicitly.
    bool rejected=false;try {cuda::DetectionPostprocessor p(30001,true);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Unsupported capacity silently accepted");
    clear();raw[4*n]=std::numeric_limits<float>::quiet_NaN();
    trt::DeviceBuffer dev(raw.size()*sizeof(float));trt::CudaStream st;
    CUDA_CHECK(cudaMemcpyAsync(dev.data(),raw.data(),dev.size(),cudaMemcpyHostToDevice,st.get()));
    cuda::DetectionPostprocessor p(n,true);p.filter(static_cast<float*>(dev.data()),c,.25f,st.get());p.nms(.45f,300,st.get());p.copyToHost(300,st.get());st.synchronize();
    require(p.header().invalid==1,"Non-finite score was not flagged");
    report << "nonfinite_and_capacity_guards,both,32,0,0,0,PASS\n";
    if (argc>2) {
        int files=0;
        for (const auto& entry:std::filesystem::directory_iterator(argv[2])) {
            auto name=entry.path().filename().string();
            if (name.size()<12 || name.substr(name.size()-12)!=".output0.f32") continue;
            std::vector<float> saved(84*8400);
            std::ifstream f(entry.path(),std::ios::binary);
            f.read(reinterpret_cast<char*>(saved.data()),saved.size()*sizeof(float));
            require(f.gcount()==std::streamsize(saved.size()*sizeof(float)) && f.peek()==EOF,"Unexpected replay tensor shape");
            for (float confidence:{.01f,.25f,.8f,.99f})
                for (int limit:{1,300})
                    check("replay_"+name+"_conf_"+std::to_string(confidence)+"_limit_"+std::to_string(limit),
                          saved,8400,80,confidence,.45f,limit,report);
            ++files;
        }
        require(files>0,"No saved [1,84,8400] tensors found for replay");
    }
    require(bool(report),"Cannot write test report");
    std::cout << "PASS CUDA postprocessing order, boundaries, full capacity and exact greedy NMS\n";
    return 0;
} catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
