# CUDA 后处理与输出压缩

本轮实现两条可选检测路径，默认仍为 `cpu`。实际性能与正确性证据见 [本机报告](../results/cuda_postprocessing/REPORT.md)，原始 CSV/JSON 在 `results/cuda_postprocessing/final/`。沿用已有 Engine，没有重新导出、构建、量化或改变预处理。

| 模式 | Engine 输出后的路径 | 当前 Engine 每帧 D2H |
|---|---|---:|
| `cpu` | 完整输出回传 → CPU argmax/filter → CPU NMS | 2,822,400 bytes |
| `gpu-filter` | GPU argmax/filter/浮点框解码/压缩 → 候选包回传 → CPU NMS | 268,816 bytes |
| `gpu-nms` | GPU argmax/filter/压缩 → GPU 排序/精确贪心 NMS → 结果包回传 | 9,616 bytes（max_det=300） |

回传包固定大小，包含 16-byte header 与 32-byte Candidate 槽位。`gpu-filter` 传 N 个槽位，`gpu-nms` 传 min(max_det,N) 个槽位，CPU 仅解析 header 声明的有效部分。尾部无效槽位不保证内容。这样保留原单流一次同步；没有读取候选计数后再启动第二次拷贝的额外同步。**并非按有效候选数/最终框数精确变长传输**。GPU NMS 后，CPU 仍负责最终少量框的坐标还原、裁剪和 Detection 包装。

## 明确的语义与边界

- 每个预测只取最高类别分数；类别同分取最低 class ID；严格 `score > conf`。
- 候选以 `score` 降序、原始预测索引升序排序。原 CPU 按预测顺序收集后 stable_sort，与新的显式次键一致；原子压缩顺序不会改变最终结果。
- 精确 class-aware **贪心 NMS**：只有已经保留的同类框在 `IoU > threshold` 时抑制后续框。不是可能被已抑制框继续抑制的 Fast NMS 近似。
- NMS 使用模型空间的 FP32 xyxy；不预先取整/裁剪。NMS 后按现有 letterbox 的分别舍入 resize gain_x/gain_y 还原，裁剪浮点框；整数框只供绘制。
- 候选容量等于 Engine 的预测数 N。加速模式要求 `1 <= N <= 30000`，本机 N=8400。没有 Top-K，所有过阈值候选均保留；超过支持容量显式拒绝，不静默截断。原 CPU 兼容路径仍保留已有 max_nms=30000 的限制。
- `max_det` 为 NMS **后**的最大输出数，默认 300，支持 1..30000。GPU 的有效输出容量为 min(max_det,N)，不会先取最高 300 个候选再做 NMS。
- 仅支持标准 YOLOv8/v11 单输出 `[1,4+classes,N]`；当前公共运行时只接受固定形状 FP32 I/O。FP16 Engine 构建模式不表示 FP16 I/O。其他输出、输入类型、非有限最高分数/选中候选坐标以及非法阈值显式报错。
- 输入为二维 CV_8UC3 BGR；非连续 ROI 按行 staging copy。拒绝空、灰度、BGRA、float Mat，沿用已验证预处理。

实现分三处：`detection_postprocessing.hpp` 是 CPU 语义对照与 wire layout；`cuda_postprocessing.cu` 提供持久化 RAII 缓冲区和 kernel；`detection.hpp` 提供三模式切换与共同坐标还原。CUDA 类别扫描按预测索引并行，类别平面访问在相邻线程间合并；atomicAdd 压缩后以显式次键恢复确定顺序。GPU NMS 使用 CUB 64-bit radix key（逆序 FP32 分数位 + 原始索引），排序 N 个槽位并给无效槽位 sentinel；64-box 分块生成同类抑制位图，单 block 顺序选择框并并行 OR 掩码。NMS scratch 持久化，按实际候选数遍历活跃 tiles，但排序仍覆盖容量 N。

GPU 后处理编译禁用 FMA 合并（`--fmad=false`），保持与 CPU 的 FP32 IoU 算术在边界处一致。没有每帧 cudaMalloc、候选计数 D2H 或阶段同步。已有 CUDA Graph 仍仅捕获 TensorRT 推理，后处理 kernel 单独提交；Graph on/off 均验证。

## 复现

```bash
cd /home/wnyxvo/YOLOs-CPP-TensorRT
source /home/wnyxvo/miniconda3/etc/profile.d/conda.sh
conda activate yolos-trt

# --fresh 清除旧编译器隐式 include 缓存；当前 CMake 支持此选项。
cmake --fresh -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DTENSORRT_DIR="$CONDA_PREFIX/opt/TensorRT-10.7.0.23" \
  -DOpenCV_DIR="$CONDA_PREFIX/lib/cmake/opencv4"
cmake --build build -j4

# 小型数据集沿用前一轮基线。输出目录须为空，避免覆盖证据。
python scripts/cuda_postprocessing/run.py \
  --output results/cuda_postprocessing/reproduction \
  --original-prefix results/cuda_postprocessing/original/validation \
  --warmup 30 --runs 200 --rounds 2
python scripts/cuda_postprocessing/report.py results/cuda_postprocessing/reproduction \
  --output results/cuda_postprocessing/REPRODUCTION_REPORT.md \
  --original-prefix results/cuda_postprocessing/original/e2e

# 单独入口：事件 profiling 与 E2E 测量不要混用
./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/cuda_postprocessing/custom_e2e \
  --postprocess gpu-filter --graph 1 --warmup 30 --runs 200
./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/cuda_postprocessing/custom_profile \
  --postprocess gpu-filter --graph 1 --warmup 30 --runs 200 --profile
./build/image_inference models/yolo11n.trt data/dog.jpg models/coco.names \
  --postprocess gpu-filter

# 合成边界测试；第二参数可重放目录内的实际 *.output0.f32
./build/cuda_postprocessing_test results/cuda_postprocessing/custom_kernel_tests.csv \
  results/cuda_postprocessing/original
```

`run.py` 先做类别 + IoU 二分图匹配，再单独检查分数/坐标零差异和确定顺序；使用 tensor SHA256 验证预处理与原始输出没有变化。调用既有 Python 脚本，参考为 ONNX Runtime CPU FP32，附加 PT/PyTorch CPU FP32 对照。阈值 `.01/.25/.8/.99` 与 max_det=1 检查三路径一致性；真实输出重放与合成测试验证同分、阈值边界、跨类、零候选、满容量、部分 tile 和贪心链条。不会放宽阈值以通过检查。

## 测量范围与证据

E2E：`steady_clock` 测量内存 cv::Mat 输入到最终 CPU Detection 可用；排除读图、绘制、显示、dump 和 CSV 写入。每个图像 30 次 warmup，每轮 200 次测量，七图 round-robin；两轮反转三模式顺序，Graph on/off 均测量。报告均值、P50、P95，分别保留轮次及逐图结果；没有把阶段时间加总作为 E2E。

profiling 独立运行。CPU staging 与后处理用 steady_clock；GPU H2D、预处理、TensorRT、筛选压缩、NMS（含排序/掩码/选择）、D2H 使用同流 CUDA Events，仅在原有完成同步点后读取。CPU 模式的 GPU filter/NMS 是空 hook，事件会显示少量标记开销。最终 E2E 未开启事件。源文件和 executable 在全程前后 SHA256 校验，模型 SHA256、Engine 真实 I/O、环境、命令和工作区 patch 均保存；额外新文件有完整 source snapshot。

原始二进制/源码与对应 validation/e2e/profile 保存于 `results/cuda_postprocessing/original/`；本轮 `verified/` 保留 max_det=1 验证入口配置冲突时的日志，该轮未进入性能测量；`pre_race_fix/` 保留共享计数循环退出风险修复前的完整数据；最终证据使用 `final/`。验证入口已将合成测试的 max_det 与 CLI 配置分离，未通过改变检测阈值或结果掩盖差异。selectGreedy 的退出决定已限定在线程 0，并在 block 同步后统一分支，避免共享 keep 的跨 warp 读取与更新竞态。

已实际验证：本机 YOLO11n Engine、上述七图及合成/真实输出重放。静态检查：缓冲区容量/flags、同步、排序规则和未支持形状拒绝。尚未验证：完整 COCO mAP、其他 Engine/YOLOv8 实测、其他设备、长时间视频运行。修正后 synthetic test 的 memcheck 为 0 errors；selectGreedy 合成测试与 suppressionMasks 七图测试的 racecheck 为 0 errors/0 warnings；两个 kernel 合成测试的 synccheck 为 0 errors。sanitizer 检查仅覆盖上述具体路径。没有引入 Top-K，因此没有 Top-K 截断精度结论；没有实施 INT8、多帧流水线或完整后处理 Graph 捕获。
