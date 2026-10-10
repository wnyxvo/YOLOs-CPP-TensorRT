#include "yolos/tasks/async_detection.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <future>
#include <thread>
#ifdef YOLOS_PIPELINE_NVTX
#include <nvToolsExt.h>
#endif

using Detector=yolos::det::AsyncYOLODetector;
using Result=Detector::Result;
using Clock=Detector::Clock;
namespace fs=std::filesystem;
static double ms(Clock::duration d) {return std::chrono::duration<double,std::milli>(d).count();}
static void require(bool condition,const std::string& message) {if (!condition) throw std::runtime_error(message);}
static std::string quote(const std::string& value) {
    std::string out="\"";for (char c:value) {out+=c;if (c=='"') out+=c;}return out+"\"";
}
struct Range {
    bool enabled;
    Range(bool on,const std::string& name):enabled(on) {
#ifdef YOLOS_PIPELINE_NVTX
        if (enabled) {
            // Registered messages support Nsight's default NVTX capture trigger.
            nvtxEventAttributes_t event{};
            event.version=NVTX_VERSION;event.size=NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            event.messageType=NVTX_MESSAGE_TYPE_REGISTERED;
            event.message.registered=nvtxDomainRegisterStringA(nullptr,name.c_str());
            nvtxRangePushEx(&event);
        }
#else
        (void)name;require(!enabled,"This build has no NVTX support");
#endif
    }
    ~Range() {
#ifdef YOLOS_PIPELINE_NVTX
        if (enabled) nvtxRangePop();
#endif
    }
};
static void equal(const std::vector<yolos::det::Detection>& actual,const std::vector<yolos::det::Detection>& expected) {
    require(actual.size()==expected.size(),"Frame result count differs");
    for (size_t i=0;i<actual.size();++i)
        require(actual[i].classId==expected[i].classId && actual[i].conf==expected[i].conf && actual[i].xyxy==expected[i].xyxy,
                "Frame class/score/float coordinates/order differs");
}
static std::vector<Result> batch(Detector& detector,const std::vector<cv::Mat>& images,size_t count,
                                  bool asynchronous,bool timeline=false,bool mutateAfterSubmit=false) {
    std::vector<Result> results;results.reserve(count);
    if (!asynchronous) {
        for (size_t i=0;i<count;++i) {
            auto started=Clock::now();
            auto detections=detector.detect(images[i%images.size()],.25f,.45f);
            auto ready=Clock::now();
            Result r;r.frameId=i;r.originalSize=images[i%images.size()].size();r.detections=std::move(detections);
            r.candidates=detector.lastCandidateCount();r.d2hBytes=detector.lastD2hBytes();
            r.latencyMs=r.admittedLatencyMs=ms(ready-started);results.push_back(std::move(r));
        }
        return results;
    }
    std::exception_ptr producerFailure;
    std::thread producer([&] {
        try {
            for (size_t i=0;i<count;++i) {
                Range range(timeline,"submit frame="+std::to_string(i));
                if (!mutateAfterSubmit) detector.submit(images[i%images.size()],i);
                else {
                    const auto& image=images[i%images.size()];
                    cv::Mat backing(image.rows+2,image.cols+7,CV_8UC3);
                    auto roi=backing(cv::Rect(3,1,image.cols,image.rows));image.copyTo(roi);
                    require(!roi.isContinuous(),"ROI unexpectedly contiguous");
                    detector.submit(roi,i);
                    roi.setTo(cv::Scalar(17,29,43)); // lifetime test: safe immediately after submit returns
                }
            }
        } catch (...) {producerFailure=std::current_exception();}
        detector.closeSubmissions();
    });
    try {
        while (true) {
            Range range(timeline,"collect frame="+std::to_string(results.size()));
            auto r=detector.collect();if (!r) break;results.push_back(std::move(*r));
        }
    } catch (...) {
        detector.closeSubmissions();producer.join();throw;
    }
    producer.join();
    if (producerFailure) std::rethrow_exception(producerFailure);
    require(results.size()==count,"Accepted/submitted/collected batch count differs");
    return results;
}
static void lifecycleChecks(Detector& detector,const std::vector<cv::Mat>& images,size_t slots,size_t bytes,
                             const std::vector<std::vector<yolos::det::Detection>>& reference,std::ofstream& log) {
    detector.enableAsync(slots,bytes);
    std::ostringstream before;detector.writeAsyncResources(before);
    auto results=batch(detector,images,images.size()*12,true,false,true);
    for (size_t i=0;i<results.size();++i) {
        require(results[i].frameId==i && results[i].slot==i%slots,"Frame ID or slot ring order differs");
        require(results[i].originalSize==images[i%images.size()].size(),"Frame geometry crossed slots");
        equal(results[i].detections,reference[i%images.size()]);
    }
    std::ostringstream after;detector.writeAsyncResources(after);
    require(before.str()==after.str(),"Persistent addresses changed while slots cycled");
    auto counters=detector.counters();require(counters.accepted==results.size() && counters.collected==results.size(),"Ring lost/duplicated frames");
    log << "PASS repeated ring reuse, mixed image sizes, noncontiguous ROI, mutation/release after submit, exact results and stable resources\n";
    detector.restartAsyncBatch();
    for (size_t i=0;i<slots;++i) detector.submit(images[i%images.size()],i);
    bool invalidId=false;try {detector.submit(images[0],slots-1);} catch (const std::invalid_argument&) {invalidId=true;}
    require(invalidId,"Duplicate/non-increasing frame ID accepted");
    std::promise<void> attempting;auto begun=attempting.get_future();
    auto extra=std::async(std::launch::async,[&]{attempting.set_value();detector.submit(images[slots%images.size()],slots);});
    begun.get();
    const auto deadline=Clock::now()+std::chrono::seconds(2);
    while (detector.counters().capacityWaits==0 && Clock::now()<deadline) std::this_thread::yield();
    bool blocked=detector.counters().capacityWaits==1 && extra.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready;
    if (!blocked) {
        detector.closeSubmissions();
        try {extra.get();} catch (...) {}
        throw std::runtime_error("Full queue did not apply observable backpressure");
    }
    auto first=detector.collect();require(first && first->frameId==0,"FIFO collection failed");
    equal(first->detections,reference[0]);extra.get();
    auto remaining=detector.stopAsync();
    require(remaining.size()==slots,"Exit drain count differs");
    for (size_t i=0;i<remaining.size();++i) {
        require(remaining[i].frameId==i+1,"Drain reordered/lost a frame");
        equal(remaining[i].detections,reference[(i+1)%images.size()]);
    }
    log << "PASS full ring blocks submit until collect, strict frame IDs, FIFO and explicit stop/drain returns every pending result\n";
    detector.enableAsync(slots,bytes);
    bool type=false,capacity=false,configuration=false,sync=false;
    try {detector.submit(cv::Mat(8,8,CV_8UC1),0);}catch(const std::invalid_argument&){type=true;}
    try {detector.submit(cv::Mat(1,int(bytes/3)+1,CV_8UC3),0);}catch(const std::invalid_argument&){capacity=true;}
    try {detector.setMaxDet(1);}catch(const std::logic_error&){configuration=true;}
    try {detector.detect(images[0]);}catch(const std::logic_error&){sync=true;}
    require(type&&capacity&&configuration&&sync,"Unsupported input/configuration was silently accepted");
    detector.closeSubmissions();require(!detector.collect(),"Empty closed queue did not terminate");detector.stopAsync();
    log << "PASS unsupported type/over-capacity input/configuration/synchronous overlap rejected; empty closed collection terminates\n";
}

int main(int argc,char** argv) try {
    if (argc<5) {std::cerr << "Usage: async_pipeline_benchmark ENGINE LABELS IMAGE_FOLDER OUTPUT_PREFIX [--slots 0|1|2|3] [--frames N] [--warmup N] [--graph 0|1] [--validate] [--timeline]\n";return 1;}
    size_t slots=0,frames=2800,warmup=30;int graph=1;bool validate=false,timeline=false;
    for (int i=5;i<argc;++i) {
        std::string option=argv[i];
        if (option=="--validate") validate=true;
        else if (option=="--timeline") timeline=true;
        else if (option=="--slots" && i+1<argc) slots=std::stoul(argv[++i]);
        else if (option=="--frames" && i+1<argc) frames=std::stoul(argv[++i]);
        else if (option=="--warmup" && i+1<argc) warmup=std::stoul(argv[++i]);
        else if (option=="--graph" && i+1<argc) graph=std::stoi(argv[++i]);
        else throw std::invalid_argument("Unknown/missing option: "+option);
    }
    require(slots<=3 && frames>0 && (graph==0||graph==1),"Invalid benchmark parameters");
    fs::create_directories(fs::absolute(argv[4]).parent_path());
    std::vector<fs::path> paths;
    for (const auto& entry:fs::directory_iterator(argv[3])) {
        auto ext=entry.path().extension().string();std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if (ext==".png"||ext==".jpg"||ext==".jpeg"||ext==".bmp") paths.push_back(entry.path());
    }
    std::sort(paths.begin(),paths.end());require(!paths.empty(),"No images");
    std::vector<cv::Mat> images;size_t capacity=0;
    for (const auto& path:paths) {
        images.push_back(cv::imread(path.string(),cv::IMREAD_COLOR));require(!images.back().empty(),"Cannot read "+path.string());
        capacity=std::max(capacity,images.back().total()*images.back().elemSize());
    }
    Detector detector(argv[1],argv[2],yolos::YOLOVersion::V11);
    detector.setPostprocessMode("gpu-filter");detector.setMaxDet(300);detector.setGraphEnabled(graph!=0);
    require(!graph || detector.isGraphEnabled(),"Graph requested but unavailable");
    std::ofstream checks(std::string(argv[4])+".checks.txt");
    std::vector<std::vector<yolos::det::Detection>> reference;
    if (validate) {
        for (const auto& image:images) reference.push_back(detector.detect(image,.25f,.45f));
        if (slots) lifecycleChecks(detector,images,slots,capacity,reference,checks);
    }
    if (slots) detector.enableAsync(slots,capacity,.25f,.45f,timeline);
    if (warmup) {
        batch(detector,images,warmup*images.size(),slots!=0);
        if (slots) detector.restartAsyncBatch();
    }
    std::ofstream resources(std::string(argv[4])+".resources.csv");
    if (slots) detector.writeAsyncResources(resources);
    auto start=Clock::now();std::vector<Result> results;
    {
        Range range(timeline,"MEASURED_BATCH");
        results=batch(detector,images,frames,slots!=0,timeline);
    }
    const double elapsed=ms(Clock::now()-start);
    Detector::Counters counters=slots?detector.counters():Detector::Counters{frames,frames,1,0};
    require(counters.accepted==frames && counters.collected==frames,"Batch was not completely drained");
    for (size_t i=0;i<results.size();++i) {
        require(results[i].frameId==i,"Missing/duplicate/out-of-order frame ID");
        if (validate) equal(results[i].detections,reference[i%images.size()]);
    }
    checks << "PASS measured batch FIFO IDs, exact synchronous parity=" << validate << ", accepted=" << counters.accepted
           << ", collected=" << counters.collected << '\n';
    std::ofstream data(std::string(argv[4])+".frames.csv"), detections(std::string(argv[4])+".detections.csv"), config(std::string(argv[4])+".config.csv");
    data << "frame_id,image,slot,latency_ms,admitted_latency_ms,submit_wait_ms,submit_host_ms,staging_ms,collection_queue_ms,collect_event_wait_ms,cpu_post_ms,candidate_count,result_count,d2h_bytes\n" << std::fixed << std::setprecision(6);
    detections << "frame_id,image,class_id,confidence,x1,y1,x2,y2\n" << std::setprecision(9);
    for (const auto& r:results) {
        const auto image=paths[r.frameId%paths.size()].filename().string();
        data << r.frameId << ',' << quote(image) << ',' << (slots?int(r.slot):-1) << ',' << r.latencyMs << ',' << r.admittedLatencyMs << ',' << r.submitWaitMs;
        if (slots) data << ',' << r.submitHostMs << ',' << r.stagingMs << ',' << r.collectionQueueMs << ',' << r.collectEventWaitMs << ',' << r.cpuPostMs;
        else data << ",,,,,";
        data << ',' << r.candidates << ',' << r.detections.size() << ',' << r.d2hBytes << '\n';
        for (const auto& d:r.detections) {
            detections << r.frameId << ',' << quote(image) << ',' << d.classId << ',' << d.conf;
            for (auto coordinate:d.xyxy) detections << ',' << coordinate;
            detections << '\n';
        }
    }
    config << "slots,frames,images,warmup_per_image,graph,postprocess,conf,iou,max_det,image_capacity_bytes,batch_elapsed_ms,completed_fps,accepted,collected,high_water,capacity_waits,timeline\n"
           << std::fixed << std::setprecision(9) << slots << ',' << frames << ',' << images.size() << ',' << warmup << ',' << detector.isGraphEnabled()
           << ",gpu-filter,0.25,0.45,300," << capacity << ',' << elapsed << ',' << (frames*1000./elapsed) << ',' << counters.accepted
           << ',' << counters.collected << ',' << counters.highWater << ',' << counters.capacityWaits << ',' << timeline << '\n';
    if (slots) require(detector.stopAsync().empty(),"Batch left uncollected frames");
    require(data&&detections&&config&&checks&&resources,"Artifact write failed");
    std::cout << "PASS slots=" << slots << " accepted=" << frames << " completed=" << frames << " batch_ms=" << elapsed
              << " completed_fps=" << frames*1000./elapsed << " high_water=" << counters.highWater << " waits=" << counters.capacityWaits << '\n';
    return 0;
} catch (const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
