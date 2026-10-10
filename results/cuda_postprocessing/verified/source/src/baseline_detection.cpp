// Headless correctness and latency entry point. Images stay in memory throughout timing.
#include "yolos/tasks/detection.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using yolos::det::Detection;

static std::string csv(const std::string& s) {
    std::string out = "\"";
    for (char c : s) { out += c; if (c == '"') out += c; }
    return out + "\"";
}

class BaselineDetector : public yolos::det::YOLODetector {
public:
    using YOLODetector::YOLODetector;
    double postMs{0};
    std::vector<Detection> measured(const cv::Mat& im, float conf, float iou, bool profile) {
        configurePostprocess(conf,iou);
        inferGpu(im);
        auto start = profile ? Clock::now() : Clock::time_point{};
        auto result = postprocess(im.size(), inputShape_, version_, conf, iou);
        if (profile) postMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return result;
    }
    void inferCpuForDiagnostic(const std::vector<float>& blob) { infer(blob.data(),blob.size()); }
    void checkPostprocessing() {
        auto* raw = outputTensors_[0].hostBuf.as<float>();
        std::vector<float> backup(raw, raw + outputTensors_[0].bytes/sizeof(float));
        auto savedX=cachedScaleX_, savedY=cachedScaleY_, savedPx=cachedPadX_, savedPy=cachedPadY_;
        cachedScaleX_=cachedScaleY_=1; cachedPadX_=cachedPadY_=0;
        auto clear = [&]() { std::fill(raw,raw+backup.size(),0.f); };
        auto put = [&](int i, int cls, float score, float x1,float y1,float x2,float y2) {
            raw[i]=(x1+x2)*.5f; raw[8400+i]=(y1+y2)*.5f;
            raw[16800+i]=x2-x1; raw[25200+i]=y2-y1; raw[(4+cls)*8400+i]=score;
        };
        clear();
        put(0,0,.9f,-100,100,100,200); put(1,0,.8f,0,100,100,200);
        put(2,1,.7f,-100,100,100,200); put(3,0,.85f,-100,100,100,200);
        auto r=postprocessStandard({640,640},1,0,0,.25f,.6f);
        if (r.size()!=3 || r[0].classId!=0 || r[1].classId!=0 || r[2].classId!=1 || r[0].xyxy[0]!=0)
            throw std::runtime_error("NMS-before-clipping/class-aware regression");
        clear(); put(0,0,.9f,.4f,20,10.4f,30); put(1,0,.8f,.8f,20,10.8f,30);
        r=postprocessStandard({640,640},1,0,0,.25f,.95f);
        if (r.size()!=2 || std::abs(r[0].xyxy[0]-.4f)>1e-5)
            throw std::runtime_error("Float NMS precision regression");
        setMaxDet(1); r=postprocessStandard({640,640},1,0,0,.25f,.95f);
        if (r.size()!=1 || r[0].conf!=.9f) throw std::runtime_error("max_det regression");
        setMaxDet(300); clear(); put(0,0,.25f,1,1,2,2);
        if (!postprocessStandard({640,640},1,0,0,.25f,.45f).empty())
            throw std::runtime_error("Confidence comparison regression");
        std::copy(backup.begin(),backup.end(),raw);
        cachedScaleX_=savedX; cachedScaleY_=savedY; cachedPadX_=savedPx; cachedPadY_=savedPy;
    }
    void checkOutput() const {
        const auto& s = getOutputShape();
        if (numOutputs() != 1 || s.size() != 3 || s[0] != 1 || s[1] != 84 || s[2] != 8400 ||
            getInputShape() != cv::Size(640,640) || getClassNames().size() != 80)
            throw std::runtime_error("This baseline requires COCO YOLO11 [1,3,640,640] -> [1,84,8400]");
    }
};

int main(int argc, char** argv) try {
    if (argc < 5) {
        std::cerr << "Usage: baseline_detection ENGINE LABELS IMAGE_OR_FOLDER OUTPUT_PREFIX "
                     "[--graph 0|1] [--profile] [--warmup N] [--runs N] [--dump] [--checks] "
                     "[--conf F] [--iou F] [--postprocess cpu|gpu-filter|gpu-nms] [--max-det N]\n";
        return 1;
    }
    const std::string prefix = argv[4];
    int warmup = 30, runs = 100, graph = 1, maxDet=300;
    std::string postprocessMode="cpu";
    float conf = .25f, iou = .45f;
    bool profile = false, dump = false, checks = false;
    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if (a=="--postprocess" && i+1<argc) postprocessMode=argv[++i];
        else if (a=="--max-det" && i+1<argc) maxDet=std::stoi(argv[++i]);
        else if (a == "--profile") profile = true;
        else if (a == "--dump") dump = true;
        else if (a == "--checks") checks = true;
        else if (i+1 < argc && a == "--warmup") warmup = std::stoi(argv[++i]);
        else if (i+1 < argc && a == "--runs") runs = std::stoi(argv[++i]);
        else if (i+1 < argc && a == "--graph") graph = std::stoi(argv[++i]);
        else if (i+1 < argc && a == "--conf") conf = std::stof(argv[++i]);
        else if (i+1 < argc && a == "--iou") iou = std::stof(argv[++i]);
        else throw std::runtime_error("Unknown/missing option: " + a);
    }
    if (warmup < 0 || runs < 1 || (graph != 0 && graph != 1) || conf < 0 || conf > 1 || iou < 0 || iou > 1)
        throw std::runtime_error("Invalid benchmark settings");
    fs::create_directories(fs::absolute(prefix).parent_path());
    std::vector<fs::path> paths;
    if (fs::is_directory(argv[3])) {
        for (const auto& e : fs::directory_iterator(argv[3])) {
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") paths.push_back(e.path());
        }
    } else paths.emplace_back(argv[3]);
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw std::runtime_error("No images");
    std::vector<cv::Mat> images;
    for (const auto& p : paths) {
        images.push_back(cv::imread(p.string(), cv::IMREAD_COLOR));
        if (images.back().empty()) throw std::runtime_error("Cannot read " + p.string());
    }
    BaselineDetector det(argv[1], argv[2], yolos::YOLOVersion::V11);
    det.checkOutput();
    det.setMaxDet(maxDet);
    det.setPostprocessMode(postprocessMode);
    det.setGraphEnabled(graph != 0);
    if (graph && !det.isGraphEnabled()) throw std::runtime_error("Graph requested but capture unavailable");
    det.setProfiling(profile);
    std::ofstream io(prefix + ".engine.csv"), config(prefix + ".config.csv");
    det.writeEngineInfo(io);
    config << "graph_requested,graph_actual,profile,warmup_per_image,runs_per_image,images,conf,iou,max_det,opencv,trt,postprocess,candidate_capacity,extra_device_bytes\n"
           << graph << ',' << det.isGraphEnabled() << ',' << profile << ',' << warmup << ',' << runs << ','
           << images.size() << ',' << conf << ',' << iou << ',' << maxDet << ',' << CV_VERSION << ','
           << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD
           << ',' << det.postprocessModeName() << ",8400," << det.postprocessDeviceBytes() << '\n';
    std::ofstream results(prefix + ".detections.csv"), times(prefix + ".timings.csv");
    results << "image,class_id,class_name,confidence,x1,y1,x2,y2\n" << std::setprecision(9);
    times << "image,iteration,e2e_ms,staging_ms,h2d_ms,preprocess_ms,inference_ms,filter_ms,nms_ms,d2h_ms,postprocess_ms,candidate_count,result_count,d2h_bytes\n"
          << std::fixed << std::setprecision(6);
    // Round-robin whole-set warmup and measurement to reduce image/order bias.
    for (int n = 0; n < warmup; ++n)
        for (const auto& im : images) det.measured(im, conf, iou, profile);
    for (int n = 0; n < runs; ++n) {
        for (size_t k = 0; k < images.size(); ++k) {
            auto start = Clock::now();
            auto ds = det.measured(images[k], conf, iou, profile);
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            times << csv(paths[k].filename().string()) << ',' << n << ',' << ms;
            if (profile) {
                const auto& t = det.stageTimes();
                times << ',' << t.stagingMs << ',' << t.h2dMs << ',' << t.preprocessMs << ',' << t.inferenceMs
                      << ',' << t.filterMs << ',' << t.nmsMs << ',' << t.d2hMs << ',' << det.postMs;
            } else times << ",,,,,,,,";
            times << ',' << det.lastCandidateCount() << ',' << ds.size() << ',' << det.lastD2hBytes() << '\n';
            if (n == 0) {
                for (const auto& d : ds) {
                    results << csv(paths[k].filename().string()) << ',' << d.classId << ','
                            << csv(det.getClassNames().at(d.classId)) << ',' << d.conf;
                    for (float x : d.xyxy) results << ',' << x;
                    results << '\n';
                }
                if (dump) {
                    const auto name = prefix + "." + paths[k].filename().string();
                    det.dumpTensors(name);
                    // Independent OpenCV CPU path, diagnostics only, outside measured regions.
                    const auto& im = images[k];
                    auto g = yolos::cuda::letterboxGeometry(im.cols,im.rows,640,640);
                    cv::Mat resized;
                    cv::resize(im,resized,cv::Size(g.newW,g.newH),0,0,cv::INTER_LINEAR);
                    std::vector<float> blob(3*640*640,114.f/255.f);
                    for (int y=0;y<g.newH;++y) for (int x=0;x<g.newW;++x)
                        for (int c=0;c<3;++c) blob[c*640*640+(y+g.padTop)*640+x+g.padLeft] =
                            resized.at<cv::Vec3b>(y,x)[2-c]/255.f;
                    det.inferCpuForDiagnostic(blob);
                    det.dumpTensors(name + ".opencv");
                }
            }
        }
    }
    if (checks) {
        std::ofstream check(prefix + ".checks.txt");
        for (size_t k = 0; k < images.size(); ++k) {
            const auto& im = images[k];
            cv::Mat backing(im.rows+2, im.cols+7, CV_8UC3);
            auto roi = backing(cv::Rect(3,1,im.cols,im.rows)); im.copyTo(roi);
            auto a = det.detect(im,conf,iou), b = det.detect(roi,conf,iou);
            if (roi.isContinuous() || a.size() != b.size()) throw std::runtime_error("ROI mismatch");
            for (size_t i=0;i<a.size();++i)
                if (a[i].xyxy != b[i].xyxy || a[i].conf != b[i].conf || a[i].classId != b[i].classId)
                    throw std::runtime_error("ROI results differ");
        }
        check << "PASS non-contiguous ROI exactly equals contiguous input for all images\n";
        for (int type : {CV_8UC1, CV_8UC4, CV_32FC3}) {
            bool rejected = false;
            try { det.detect(cv::Mat(16,16,type)); } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Unsupported cv::Mat type accepted");
        }
        bool rejected = false;
        try { det.detect(cv::Mat()); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Empty Mat accepted");
        check << "PASS empty/gray/BGRA/float input rejected\n";
        det.checkPostprocessing();
        det.setMaxDet(maxDet);
        check << "PASS float NMS, clipping after NMS, class-aware NMS, max_det and confidence boundary\n";
    }
    if (!io || !config || !results || !times) throw std::runtime_error("Output write failed");
    std::cout << "Saved " << prefix << ".{engine,config,detections,timings}.csv\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << '\n';
    return 1;
}
