// Streaming dataset export; accuracy only, no latency claims or dataset writes.
#include "yolos/tasks/async_detection.hpp"
#include <fstream>
#include <iomanip>
#include <thread>
int main(int argc,char** argv) try {
    if (argc!=6 && argc!=7) {std::cerr << "Usage: dataset_accuracy_detection ENGINE LABELS ID_PATH_TSV OUTPUT_CSV cpu|gpu-filter|async2 [MAX_IMAGE_BYTES]\n";return 1;}
    const std::string mode=argv[5];
    const bool asynchronous=mode=="async2";
    if (mode!="cpu" && mode!="gpu-filter" && !asynchronous) throw std::invalid_argument("Accuracy comparison requires cpu, gpu-filter or async2");
    if (asynchronous && argc!=7) throw std::invalid_argument("async2 requires a fixed MAX_IMAGE_BYTES from dataset dimensions");
    yolos::det::AsyncYOLODetector det(argv[1],argv[2],yolos::YOLOVersion::V11);
    det.setPostprocessMode("gpu-filter"); // validate standard layout for both paths
    if (mode=="cpu") det.setPostprocessMode(mode);
    det.setMaxDet(300);det.setGraphEnabled(true);
    if (!det.isGraphEnabled()) throw std::runtime_error("Graph requested but unavailable");
    if (det.getInputShape()!=cv::Size(640,640) || det.getClassNames().size()!=80)
        throw std::runtime_error("This dataset evaluator requires 640x640 COCO YOLO11 input");
    std::ifstream list(argv[3]);std::ofstream out(argv[4]),frames(std::string(argv[4])+".frames.csv");
    if (!list||!out||!frames) throw std::runtime_error("Cannot open dataset manifest/result file");
    out << "image_id,class_id,confidence,x1,y1,x2,y2\n" << std::setprecision(9);
    frames << "image_id,width,height,result_count,slot\n";
    struct Entry {uint64_t id;std::string path;};std::vector<Entry> entries;
    std::string line;size_t images=0,boxes=0;
    while (std::getline(list,line)) {
        auto tab=line.find('\t');if (tab==std::string::npos) throw std::runtime_error("Expected ID TAB image path");
        auto imageId=std::stoull(line.substr(0,tab));
        if (!entries.empty() && imageId<=entries.back().id) throw std::invalid_argument("Manifest IDs must increase strictly");
        entries.push_back({imageId,line.substr(tab+1)});
    }
    if (entries.empty()) throw std::runtime_error("Empty dataset manifest");
    auto save=[&](uint64_t imageId,const cv::Size& size,const std::vector<yolos::det::Detection>& result,int slot) {
        if (images>=entries.size() || imageId!=entries[images].id) throw std::runtime_error("Lost, duplicate or reordered image ID");
        for (const auto& d:result) {
            out << imageId << ',' << d.classId << ',' << d.conf;
            for (auto value:d.xyxy) out << ',' << value;
            out << '\n';++boxes;
        }
        frames << imageId << ',' << size.width << ',' << size.height << ',' << result.size() << ',' << slot << '\n';
        ++images;
    };
    if (!asynchronous) {
        for (const auto& entry:entries) {
            auto image=cv::imread(entry.path,cv::IMREAD_COLOR);
            if (image.empty()) throw std::runtime_error("Cannot read "+entry.path);
            save(entry.id,image.size(),det.detect(image,.25f,.45f),-1);
        }
    } else {
        det.enableAsync(2,std::stoull(argv[6]),.25f,.45f);
        std::exception_ptr producerFailure;
        std::thread producer([&] {
            try {
                for (const auto& entry:entries) {
                    auto image=cv::imread(entry.path,cv::IMREAD_COLOR);
                    if (image.empty()) throw std::runtime_error("Cannot read "+entry.path);
                    det.submit(image,entry.id); // image storage released after submit
                }
            } catch (...) {producerFailure=std::current_exception();}
            det.closeSubmissions();
        });
        try {
            while (auto result=det.collect()) save(result->frameId,result->originalSize,result->detections,int(result->slot));
        } catch (...) {det.closeSubmissions();producer.join();throw;}
        producer.join();
        if (producerFailure) std::rethrow_exception(producerFailure);
        auto counts=det.counters();
        if (counts.accepted!=entries.size() || counts.collected!=entries.size() || !det.stopAsync().empty())
            throw std::runtime_error("Async accuracy batch did not drain every image");
    }
    if (images!=entries.size() || !out || !frames) throw std::runtime_error("Image count or result write failure");
    std::cout << "Saved " << images << " images, " << boxes << " detections, conf=.25 iou=.45 max_det=300\n";
    return 0;
} catch (const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
