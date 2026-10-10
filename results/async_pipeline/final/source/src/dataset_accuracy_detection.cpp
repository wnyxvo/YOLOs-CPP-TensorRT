// Streaming dataset export; accuracy only, no latency claims or dataset writes.
#include "yolos/tasks/detection.hpp"
#include <fstream>
#include <iomanip>
int main(int argc,char** argv) try {
    if (argc!=6) {std::cerr << "Usage: dataset_accuracy_detection ENGINE LABELS ID_PATH_TSV OUTPUT_CSV cpu|gpu-filter\n";return 1;}
    const std::string mode=argv[5];
    if (mode!="cpu" && mode!="gpu-filter") throw std::invalid_argument("Accuracy comparison requires cpu or gpu-filter");
    yolos::det::YOLODetector det(argv[1],argv[2],yolos::YOLOVersion::V11);
    det.setPostprocessMode("gpu-filter"); // validate standard layout for both paths
    det.setPostprocessMode(mode);det.setMaxDet(300);det.setGraphEnabled(true);
    if (!det.isGraphEnabled()) throw std::runtime_error("Graph requested but unavailable");
    if (det.getInputShape()!=cv::Size(640,640) || det.getClassNames().size()!=80)
        throw std::runtime_error("This dataset evaluator requires 640x640 COCO YOLO11 input");
    std::ifstream list(argv[3]);std::ofstream out(argv[4]);
    if (!list||!out) throw std::runtime_error("Cannot open dataset manifest/result file");
    out << "image_id,class_id,confidence,x1,y1,x2,y2\n" << std::setprecision(9);
    std::string line;size_t images=0,boxes=0;
    while (std::getline(list,line)) {
        auto tab=line.find('\t');if (tab==std::string::npos) throw std::runtime_error("Expected ID TAB image path");
        auto imageId=std::stoull(line.substr(0,tab));const auto path=line.substr(tab+1);
        auto image=cv::imread(path,cv::IMREAD_COLOR);if (image.empty()) throw std::runtime_error("Cannot read "+path);
        auto result=det.detect(image,.25f,.45f);
        for (const auto& d:result) {
            out << imageId << ',' << d.classId << ',' << d.conf;
            for (auto value:d.xyxy) out << ',' << value;
            out << '\n';++boxes;
        }
        ++images;
    }
    if (!images || !out) throw std::runtime_error("No images or result write failure");
    std::cout << "Saved " << images << " images, " << boxes << " detections, conf=.25 iou=.45 max_det=300\n";
    return 0;
} catch (const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
