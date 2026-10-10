// Standalone diagnostic; does not change the detector, Engine, or its Graph.
// All frame buffers/streams/events/Graphs are persistent before the measured range.
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e)); } while(0)
constexpr int frames=12;
constexpr size_t largeBytes=16*1024*1024, imageBytes=4922883, packetBytes=268816;
__global__ void compute(const unsigned char* input, unsigned char* output, unsigned long long cycles) {
    auto begin=clock64();
    while(clock64()-begin<cycles) { }
    if(blockIdx.x==0 && threadIdx.x==0) output[0]=input[0]+1;
}
struct Frame { unsigned char *input,*output,*host; cudaEvent_t up,done,down; cudaGraph_t graph; cudaGraphExec_t exec; };
int main(int argc,char** argv) try {
    if(argc!=2) throw std::runtime_error("Usage: copy_overlap_probe OUTPUT.csv");
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop,0));
    std::ofstream csv(argv[1]); csv<<"case,stream_flags,graph,h2d_bytes,d2h_bytes,frames,host_enqueue_ms,batch_ms,verified\n";
    unsigned char* hostInput; CHECK(cudaMallocHost(&hostInput,largeBytes)); memset(hostInput,8,largeBytes);
    unsigned long long cycles=static_cast<unsigned long long>(prop.clockRate)*2; // ~2ms at reported clock; trace measures actual durations.
    auto batch=nvtxDomainRegisterStringA(nullptr,"MEASURED_BATCH");
    nvtxEventAttributes_t batchAttr{}; batchAttr.version=NVTX_VERSION; batchAttr.size=NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    batchAttr.messageType=NVTX_MESSAGE_TYPE_REGISTERED; batchAttr.message.registered=batch;
    // Setup/warmup is deliberately outside each CASE range, so full-process capture is used.
    for(unsigned flags:{unsigned(cudaStreamDefault),unsigned(cudaStreamNonBlocking)}) {
        cudaStream_t upload,computeStream,download; CHECK(cudaStreamCreateWithFlags(&upload,flags));
        CHECK(cudaStreamCreateWithFlags(&computeStream,flags)); CHECK(cudaStreamCreateWithFlags(&download,flags));
        unsigned actual=99; CHECK(cudaStreamGetFlags(upload,&actual)); if(actual!=flags) throw std::runtime_error("stream flags mismatch");
        std::vector<Frame> f(frames);
        for(auto& s:f) {
            CHECK(cudaMalloc(&s.input,largeBytes)); CHECK(cudaMalloc(&s.output,largeBytes)); CHECK(cudaMallocHost(&s.host,largeBytes));
            CHECK(cudaMemset(s.input,8,largeBytes)); CHECK(cudaMemset(s.output,42,largeBytes));
            CHECK(cudaEventCreateWithFlags(&s.up,cudaEventDisableTiming)); CHECK(cudaEventCreateWithFlags(&s.done,cudaEventDisableTiming)); CHECK(cudaEventCreateWithFlags(&s.down,cudaEventDisableTiming));
            CHECK(cudaStreamBeginCapture(computeStream,cudaStreamCaptureModeThreadLocal));
            compute<<<24,128,0,computeStream>>>(s.input,s.output,cycles);
            CHECK(cudaStreamEndCapture(computeStream,&s.graph)); CHECK(cudaGraphInstantiate(&s.exec,s.graph,0));
        }
        for(int w=0;w<3;++w) compute<<<24,128,0,computeStream>>>(f[0].input,f[0].output,cycles);
        CHECK(cudaDeviceSynchronize());
        for(bool large:{false,true}) for(bool graph:{false,true}) {
            std::vector<std::string> orders={"interleaved","lookahead_one","batched_uploads"};
            if(!large && !graph && flags==cudaStreamNonBlocking) { orders.push_back("explicit_serial"); orders.push_back("host_gap"); }
            for(auto& order:orders) {
                const size_t inBytes=large?largeBytes:imageBytes,outBytes=large?largeBytes:packetBytes;
                const auto name=order+"_"+(large?"large":"project")+"_flags"+std::to_string(flags)+"_graph"+std::to_string(graph);
                auto doUpload=[&](int i) { CHECK(cudaMemcpyAsync(f[i].input,hostInput,inBytes,cudaMemcpyHostToDevice,upload)); CHECK(cudaEventRecord(f[i].up,upload)); };
                auto doCompute=[&](int i) { CHECK(cudaStreamWaitEvent(computeStream,f[i].up,0));
                    if(graph) CHECK(cudaGraphLaunch(f[i].exec,computeStream));
                    else compute<<<24,128,0,computeStream>>>(f[i].input,f[i].output,cycles);
                    CHECK(cudaGetLastError()); CHECK(cudaEventRecord(f[i].done,computeStream)); };
                auto doDownload=[&](int i) { CHECK(cudaStreamWaitEvent(download,f[i].done,0)); CHECK(cudaMemcpyAsync(f[i].host,f[i].output,outBytes,cudaMemcpyDeviceToHost,download)); CHECK(cudaEventRecord(f[i].down,download)); };
                nvtxRangePushEx(&batchAttr); nvtxRangePushA(name.c_str()); auto start=std::chrono::steady_clock::now();
                if(order=="batched_uploads") {
                    for(int i=0;i<frames;++i) doUpload(i);
                    for(int i=0;i<frames;++i) doCompute(i);
                    for(int i=0;i<frames;++i) doDownload(i);
                } else if(order=="lookahead_one") {
                    doUpload(0);
                    for(int i=0;i<frames;++i) {
                        doCompute(i);
                        if(i+1<frames) doUpload(i+1); // Issue H2D(i+1) before dependency-blocked D2H(i).
                        doDownload(i);
                    }
                } else {
                    for(int i=0;i<frames;++i) {
                        if(order=="explicit_serial" && i) CHECK(cudaStreamWaitEvent(upload,f[i-1].down,0));
                        if(order=="host_gap" && i) std::this_thread::sleep_for(std::chrono::milliseconds(6));
                        doUpload(i); doCompute(i); doDownload(i);
                    }
                }
                auto enqueued=std::chrono::steady_clock::now(); CHECK(cudaEventSynchronize(f.back().down)); auto end=std::chrono::steady_clock::now();
                nvtxRangePop(); nvtxRangePop();
                for(auto& s:f) if(s.host[0]!=9 || s.host[outBytes-1]!=42) throw std::runtime_error("copy/compute result mismatch");
                auto ms=[](auto duration){return std::chrono::duration<double,std::milli>(duration).count();};
                csv<<name<<','<<flags<<','<<graph<<','<<inBytes<<','<<outBytes<<','<<frames<<','<<ms(enqueued-start)<<','<<ms(end-start)<<",true\n";
                std::cout<<name<<" verified, batch "<<ms(end-start)<<" ms\n";
            }
        }
        for(auto& s:f) { CHECK(cudaGraphExecDestroy(s.exec)); CHECK(cudaGraphDestroy(s.graph)); CHECK(cudaEventDestroy(s.up)); CHECK(cudaEventDestroy(s.done)); CHECK(cudaEventDestroy(s.down)); CHECK(cudaFree(s.input)); CHECK(cudaFree(s.output)); CHECK(cudaFreeHost(s.host)); }
        CHECK(cudaStreamDestroy(upload)); CHECK(cudaStreamDestroy(computeStream)); CHECK(cudaStreamDestroy(download));
    }
    CHECK(cudaFreeHost(hostInput)); return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
