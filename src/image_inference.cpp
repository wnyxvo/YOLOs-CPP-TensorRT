/**
 * @file image_inference.cpp
 * @brief Object detection in a static image using YOLO TensorRT engines.
 *
 * This application loads a TensorRT engine, runs object detection on one or
 * more images, and displays the results with bounding boxes.
 *
 * Usage:
 *   ./image_inference <engine_path> [image_path_or_folder] [labels_path]
 *
 * Author: YOLOs-TRT Team
 */

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "yolos/tasks/detection.hpp"

using namespace yolos::det;

int main(int argc, char* argv[]) try {
    namespace fs = std::filesystem;

    // Paths to the model, labels, and test image
    std::string labelsPath = "../models/coco.names";
    std::string imagePath  = "../data/dog.jpg";
    std::string modelPath  = "../models/yolo11n.trt";
    std::vector<std::string> imageFiles;

    if (argc > 1) {
        modelPath = argv[1];
    }
    if (argc > 2) {
        imagePath = argv[2];
        if (fs::is_directory(imagePath)) {
            for (const auto& entry : fs::directory_iterator(imagePath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff" || ext == ".tif") {
                        imageFiles.push_back(fs::absolute(entry.path()).string());
                    }
                }
            }
            if (imageFiles.empty()) {
                std::cerr << "No image files found in directory: " << imagePath << std::endl;
                return -1;
            }
        } else if (fs::is_regular_file(imagePath)) {
            imageFiles.push_back(imagePath);
        } else {
            std::cerr << "Provided path is not a valid file or directory: " << imagePath << std::endl;
            return -1;
        }
    } else {
        std::cout << "Usage: " << argv[0] << " <engine_path> [image_path_or_folder] [labels_path]\n";
        std::cout << "No image path provided. Using default: " << imagePath << std::endl;
        imageFiles.push_back(imagePath);
    }
    if (argc > 3) {
        labelsPath = argv[3];
    }

    bool display = true, graph = true;
    float conf = .4f, iou = .45f;
    int maxDet = 300;
    std::string csvPath;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-display") display = false;
        else if (a == "--csv" && i+1 < argc) csvPath = argv[++i];
        else if (a == "--graph" && i+1 < argc) {
            std::string v = argv[++i];
            if (v != "0" && v != "1") throw std::invalid_argument("--graph expects 0 or 1");
            graph = v == "1";
        }
        else if (a == "--conf" && i+1 < argc) conf = std::stof(argv[++i]);
        else if (a == "--iou" && i+1 < argc) iou = std::stof(argv[++i]);
        else if (a == "--max-det" && i+1 < argc) maxDet = std::stoi(argv[++i]);
        else throw std::invalid_argument("Unknown/missing option: " + a);
    }
    if (conf < 0 || conf > 1 || iou < 0 || iou > 1) throw std::invalid_argument("Thresholds must be in [0,1]");
    std::sort(imageFiles.begin(), imageFiles.end());
    std::ofstream out;
    if (!csvPath.empty()) {
        out.open(csvPath);
        if (!out) throw std::runtime_error("Cannot open " + csvPath);
        out << "image,class_id,class_name,confidence,x1,y1,x2,y2\n" << std::setprecision(9);
    }
    auto quote = [](const std::string& t) {
        std::string q = "\"";
        for (char c : t) { q += c; if (c == '"') q += c; }
        return q + "\"";
    };

    // Initialize the YOLO detector with TensorRT engine
    YOLODetector detector(modelPath, labelsPath);
    detector.setGraphEnabled(graph);
    detector.setMaxDet(maxDet);

    for (const auto& imgPath : imageFiles) {
        std::cout << "\nProcessing: " << imgPath << std::endl;

        cv::Mat image = cv::imread(imgPath);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image!\n";
            return 1;
        }

        auto start = std::chrono::steady_clock::now();
        std::vector<Detection> results = detector.detect(image, conf, iou);
        auto duration = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start);

        std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Number of detections found: " << results.size() << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "Detection " << i << ": Class=" << results[i].classId
                      << ", Confidence=" << results[i].conf
                      << ", Box=(" << results[i].box.x << "," << results[i].box.y
                      << "," << results[i].box.width << "," << results[i].box.height << ")" << std::endl;
        }

        if (out.is_open()) {
            for (const auto& d : results) {
                out << quote(fs::path(imgPath).filename().string()) << ',' << d.classId << ','
                    << quote(detector.getClassNames().at(d.classId)) << ',' << d.conf;
                for (float x : d.xyxy) out << ',' << x;
                out << '\n';
            }
            if (!out) throw std::runtime_error("Cannot write " + csvPath);
        }
        if (display) {
            detector.drawDetections(image, results);
            cv::imshow("Detections", image);
            cv::waitKey(0);
        }
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << '\n';
    return 1;
}
