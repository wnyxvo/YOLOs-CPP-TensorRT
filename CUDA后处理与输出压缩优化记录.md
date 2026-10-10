# CUDA 后处理与输出压缩：本次优化过程记录

记录日期：2026-10-10。项目：YOLOs-CPP-TensorRT 二次开发；设备：RTX 4060 Laptop。

本轮在现有、已跑通的 YOLO11n TensorRT 检测链路上完成两步优化：先将类别最大值、置信度筛选与候选压缩迁移到 GPU，保留 CPU NMS；再实现 GPU 排序和精确 class-aware 贪心 NMS。三条路径保留开关，使用同一个 Engine 做正确性对照和性能测量。

**最终建议使用 `gpu-filter` + CPU NMS 作为当前部署的性能基线。** 修复后最终测量中，Graph 开启时平均端到端延迟从 CPU 路径的 **2.2904 ms** 降至 **1.6032 ms**，降低 **30.00%**。GPU NMS 为 **1.7485 ms**，当前小候选场景比混合路径慢 **0.1452 ms**。这些数字仅适用于本次设备、Engine、样本与测量范围。

本记录同时保留实现中的取舍、验证入口问题、GPU 同步风险及其修正。修复前数据保留用于追溯，后续优化以修复后的 `final/` 为准。

## 1. 本轮目标与范围

目标数据流：

```text
模型输出留在 GPU
  → 每个预测的类别最大值与置信度筛选
  → 浮点框解码、候选整理与压缩
  → 按类别 NMS
  → 回传最终结果
```

采用两步实现，分别验证收益：

| 模式 | 工作划分 | 本轮定位 |
|---|---|---|
| `cpu` | 原始输出 D2H，CPU 筛选、解码和 NMS | 保留正确性与性能对照，仍为默认模式 |
| `gpu-filter` | GPU 筛选、解码、压缩，CPU 排序和 NMS | 当前推荐部署模式 |
| `gpu-nms` | GPU 筛选、压缩、排序、NMS，CPU 结果包装 | 完整 GPU NMS 的实测对照 |

本轮沿用既有 CUDA letterbox 和 TensorRT CUDA Graph 推理；没有重复实现它们。没有重新导出模型、重建实际推理 Engine、引入 Top-K、INT8 或多帧流水线。GPU NMS 后，少量框的坐标还原、裁剪和 C++ Detection 包装仍在 CPU 完成。

## 2. 起始实现与实验环境

### 2.1 先确认实际代码

起始 commit 为 `89e7da64fd3ee80e6f51cc447b4d018da04a1459`，修改前工作区状态保存在 `results/cuda_postprocessing/original/`。新增实现属于当前未提交工作区，不能只用起始 commit 代表最终代码；最终证据还保存了源码 SHA256、patch、完整新增文件快照和二进制 SHA256。

实际代码的原始链路为：

```text
内存 cv::Mat
  → CPU memcpy 到复用的 pinned staging
  → 单 stream H2D
  → CUDA 融合 letterbox / BGR→RGB / 归一化 / HWC→NCHW
  → TensorRT enqueueV3 或已捕获的 CUDA Graph 重放
  → 完整模型输出 D2H
  → 该 stream 一次完成同步
  → CPU argmax、置信度筛选、浮点框解码、排序、class-aware NMS
  → letterbox 坐标还原、裁剪与 Detection 包装
```

固定输入和输出缓冲区、图像暂存空间、Pinned Memory、CUDA Stream、TensorRT 推理 Graph 已存在。当前 Graph 仅捕获 TensorRT 推理；新增后处理 kernel 在同一个 stream 上独立提交。

### 2.2 实际环境

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 驱动 | 591.86 |
| CUDA Toolkit | 12.4.99 |
| TensorRT C++ | 10.7.0.23 |
| OpenCV C++ | 4.10.0 |
| OpenCV Python | 5.0.0 |
| Ultralytics | 8.4.174 |
| PyTorch | 2.14.1+cu130 |
| ONNX Runtime | 1.30.0 |
| 运行环境 | WSL/Linux，conda `yolos-trt` |
| 本次 CUDA 架构 | sm_89 |

TensorRT Python 包未安装；本轮通过 C++ TensorRT 库进行推理，不依赖 Python TensorRT。环境和链接库的实际采集结果见 [environment.json](results/cuda_postprocessing/final/environment.json)。未锁定 GPU 功耗/时钟，系统调度和设备状态会影响延迟；报告保留两轮和逐图统计。

### 2.3 模型和真实 I/O

模型为 YOLO11n，复用已有 FP16 构建的 Engine。

| Tensor | 实际 dtype | shape | bytes |
|---|---|---|---:|
| input `images` | float32 | `[1,3,640,640]` | 4,915,200 |
| output `output0` | float32 | `[1,84,8400]` | 2,822,400 |

**FP16 构建/计算模式与 I/O dtype 分别记录。** 当前 kernel 处理 FP32 输出；未将 FP32 输出按 half 解释，也没有把 FP16 构建标志写成 FP16 I/O。

模型文件标识：

| 文件 | SHA256 |
|---|---|
| models/yolo11n.pt | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| models/yolo11n.onnx | `86c32d35d058eaf9b84b4d159ac779fa4160fa735593112fcf07a1f278f70fc7` |
| models/yolo11n.trt | `de97ab4e249a501cd138e0cc2ce66d38c5b69615933bf11d30d2bfddae8b1270` |

已有命令：

```bash
python models/export_onnx.py --model yolo11n
./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16 --workspace 2048
```

上述命令来自用户已执行信息和既有导出脚本/模型元数据；本轮没有重新执行导出或构建该推理 Engine，不补造构建日志。

## 3. 为什么从输出压缩开始

原始输出共 `84 × 8400 = 705600` 个 FP32 数值。前四个通道为 cx、cy、w、h，后 80 个通道为类别分数。CPU 每帧需要回传整个 2.82 MB 张量，再遍历 `80 × 8400 = 672000` 个类别分数寻找最大值。

实际 `conf=.25` 的七张图片只有 10～47 个候选，最终仅 1～5 个框。输出张量的体量与最终结果规模差距很大，因此值得首先验证：GPU 直接扫描类别分数，CPU 只接收候选结构后，D2H 和 CPU 扫描成本能否共同下降。

修改前独立 profiling 的阶段均值约为：

| 阶段 | 原始均值 ms |
|---|---:|
| CPU staging | 0.1758 |
| H2D | 0.2089 |
| GPU 预处理 | 0.0687 |
| TensorRT 推理 | 0.9518 |
| 完整输出 D2H | 0.2735 |
| CPU 后处理 | 0.4445 |

这些值来自 `original/profile.timings.csv`，不相加估算端到端时间。CPU 后处理包含类别扫描、解码、排序、NMS 和包装，不能把该值全部称为 CPU NMS 耗时。

修改前无 profiling 的 1400 次测量平均端到端为 2.1513 ms。后续比较主要使用最终版本内同一二进制的 CPU/GPU 模式，避免混用不同时间或不同版本的测试。

## 4. 实验冻结与证据保存

开始改动前保存了：

- 起始 commit、工作区状态和模型 SHA256。
- 原始 `baseline_detection.binary`。
- 原始 `detection.hpp` 和 `trt_session_base.hpp`。
- 原始七图检测 CSV、中间输入/输出张量和检查结果。
- 原始不带事件的端到端数据，以及独立 profiling 数据。

当前目录的用途：

| 目录/文件 | 用途 | 是否为最终基线 |
|---|---|---|
| `original/` | 修改前源码/二进制与测量数据 | 原始对照 |
| `filter_smoke.*`、`nms_smoke.*` | 第一版两条 CUDA 路径的冒烟结果 | 否 |
| `verified/` | 首次完整验证；max_det=1 合成检查配置冲突导致中止，尚未性能测量 | 否 |
| `pre_race_fix/` | GPU 贪心循环同步风险修复前的完整数据 | 否，仅保留追溯 |
| `final/` | 同步修正后的完整验证、两轮性能和独立 profiling | **是** |
| `memcheck.log`、`racecheck_*.log`、`synccheck.log` | 修正后的 sanitizer 证据 | 辅助验证 |

中间张量 `.f32/.npy` 和原始二进制保留在本机，被该结果目录的 `.gitignore` 排除以控制仓库体积；CSV、JSON、源码快照与说明保留。中间张量可由验证入口 `--dump` 重新生成。

## 5. 修改点与职责划分

| 文件 | 本次职责 |
|---|---|
| [detection_postprocessing.hpp](include/yolos/core/detection_postprocessing.hpp) | Candidate/Header 布局、显式排序规则、CPU 筛选与精确贪心 NMS 对照 |
| [cuda_postprocessing.hpp](include/yolos/core/cuda_postprocessing.hpp) | GPU 后处理对象接口，持久化 buffer 的封装 |
| [cuda_postprocessing.cu](include/yolos/core/cuda_postprocessing.cu) | GPU 筛选压缩、排序 key、抑制 bitmask、精确贪心选择和压缩 D2H |
| [trt_session_base.hpp](include/yolos/core/trt_session_base.hpp) | 推理后 hooks、阶段 Events、统一完成点、诊断时显式回传原始输出 |
| [detection.hpp](include/yolos/tasks/detection.hpp) | 三模式切换、形状/容量约束、结果 flags 校验、共同坐标还原 |
| [baseline_detection.cpp](src/baseline_detection.cpp) | 模式/max_det CLI、候选/输出/字节统计、测量与正确性入口 |
| [image_inference.cpp](src/image_inference.cpp) | 图像入口增加 `--postprocess` |
| [cuda_postprocessing_test.cpp](tests/detection/cuda_postprocessing_test.cpp) | 合成边界用例、真实张量重放、缓冲区重复使用检查 |
| [run.py](scripts/cuda_postprocessing/run.py) | 正确性先行、模型/源码留档、Graph 对照、分离性能测量 |
| [report.py](scripts/cuda_postprocessing/report.py) | 从已保存 CSV/JSON 生成可阅读报告 |
| [CMakeLists.txt](CMakeLists.txt) | 新 CUDA 静态库、测试目标、精确 FP32 算术选项与链接 |

公共运行时提供默认空筛选/NMS hook 和原始完整 D2H；检测子类按显式模式覆盖这些 hooks。其他任务保留原始路径，未宣称已为分割、姿态等实现 GPU 后处理。

## 6. 第一步：GPU 筛选与候选压缩

### 6.1 输入布局和线程划分

标准输出按 `[1,4+classes,N]` 解释，索引形式为：

```cpp
cx = raw[0 * N + d];
cy = raw[1 * N + d];
w  = raw[2 * N + d];
h  = raw[3 * N + d];
score_c = raw[(4 + c) * N + d];
```

一个 CUDA 线程处理一个预测索引 `d`。线程内部扫描所有类别，相邻线程同时读取同一类别平面中相邻的预测，利于 GPU 合并访问。最高类别分数确定后，严格判断 `score > conf`，通过阈值的预测解码为浮点 xyxy。

各线程通过 `atomicAdd(candidateCount,1)` 获得候选槽位，写入记录。atomic 压缩后的物理顺序不确定，因此每个记录保留原始 `sourceIndex=d`，在 NMS 前恢复明确总顺序。

### 6.2 数据包布局

```cpp
struct Candidate {
    float x1, y1, x2, y2, score;
    int32_t classId;
    uint32_t sourceIndex;
    uint32_t reserved;
}; // 32 bytes

struct PacketHeader {
    uint32_t candidateCount;
    uint32_t resultCount;
    uint32_t overflow;
    uint32_t invalid;
}; // 16 bytes
```

候选容量为 Engine 预测数 N，当前 N=8400。即便 8400 个预测全部过阈值，也能全部保留；没有为节省传输而提前截断成 300 个候选。

`overflow` 和 `invalid` 随包回传，CPU 完成点后显式检查；非法结果不能被静默解释为正常框。非有限最高分数或选中框坐标触发错误。

### 6.3 为什么选择固定容量包

第一步实际拷贝：

```text
16 + 8400 × 32 = 268816 bytes / frame
```

相较原始 2,822,400 bytes，D2H 字节数减少约 90.48%。

这一步仍回传固定 N 个槽位，CPU 只读取 header 指定的有效记录。无效尾槽位内容未定义，不能作为结果解析。没有实现“先把计数回传到 CPU、同步、再按计数发起第二次 D2H”的两段回传；固定包保持既有一次完成同步。

因此，本轮“压缩”包含两件事：在 GPU 中将有效候选整理到连续前缀；将完整类别矩阵替换为更小的候选结构包。实际字节数包含无效尾槽位，报告没有把有效候选数乘以 32 当成真实传输量。

### 6.4 CPU 保留的工作

CPU 对有效候选排序，再做 class-aware 精确贪心 NMS，最后还原坐标并生成 Detection。候选少时，排序和 NMS 的 CPU 成本很小；这也是混合路径可能优于完整 GPU NMS 的原因。

## 7. 必须明确的排序、阈值和容量规则

| 项目 | 本轮规则 |
|---|---|
| 单预测类别选择 | 类别分数最大值；同分取最低 class ID |
| 置信度筛选 | 严格 `score > conf`，等于阈值不保留 |
| 全局候选顺序 | score 降序；同分按原始预测索引升序 |
| 抑制条件 | 同类且 `IoU > iou_threshold`，等于阈值不抑制 |
| NMS 坐标 | 模型空间 FP32 xyxy |
| 取整和裁剪 | NMS 后还原并裁剪浮点框；绘制整数框单独生成 |
| 加速模式最大候选数 | 与 Engine N 相同，支持 1..30000；本机 8400 |
| 最大输出数 | `max_det`，默认 300，支持 1..30000，在 NMS 后生效 |
| Top-K | 没有引入 |
| 不支持容量 | 显式拒绝，不静默 Top-K |

原 CPU 按预测索引顺序收集，并按分数 stable_sort；新 CPU 使用 score/sourceIndex 显式总顺序，两者在合法有限输出上的同分语义一致。GPU 原子收集顺序不影响最终选择。

原 CPU 兼容路径保留已有 max_nms=30000 限制。加速模式直接拒绝 N>30000 的 Engine，从而不对候选默默截断；这是当前实现的支持边界。

## 8. 第二步：GPU 精确贪心 NMS

### 8.1 GPU 排序

为有效候选生成 64-bit key：

```text
高 32 bit：正有限 FP32 分数的位表示取反
低 32 bit：原始 sourceIndex
```

过阈值分数为正有限数；正 FP32 的位模式与数值大小单调一致，取反后升序 radix sort 得到分数降序。同分通过低 32 bit 的 sourceIndex 打破平局。无效槽位使用 UINT64_MAX sentinel，排在最后。

使用 CUB `DeviceRadixSort::SortPairs` 对 key 和候选槽位索引排序，scratch 在初始化时分配并复用。排序容量仍为 N，避免先将候选数量回传给主机；这是减少同步与增加排序工作的取舍。

### 8.2 同类抑制位图

候选按 64 个框分块，每个 uint64_t 表示一组后续候选的抑制关系。kernel 在 GPU 上读取有效候选计数，只遍历活跃 tiles；不处理下三角重复比较。

每个有效框仅对后续同类框计算浮点 IoU，满足严格抑制条件后置位。shared memory 暂存列块，块内通过同步保证列数据加载和下一次重用的顺序。

位图 stride 为 `ceil(N/64)`，对应当前 N=8400 的位图容量为：

```text
8400 × ceil(8400 / 64) × 8 = 8870400 bytes
```

GPU NMS 新增持久化 device buffer 总量约 9.72 MB，包含候选/结果包、keys、indices、位图和 CUB scratch。它不是进程峰值显存，也不包含 Engine 与已有运行时缓冲区。

### 8.3 精确贪心选择

单 block 按已排序顺序决定是否保留框，其他线程并行 OR 被保留框的抑制 mask。只有“已经保留”的框会影响后续候选；被抑制的框不会继续抑制其他框。

合成测试专门使用 A/B/C 链：A 与 B 重叠，B 与 C 重叠，A 与 C 不达到抑制阈值。正确贪心结果为 A、C。若用所有高分框同时抑制低分框的近似，C 可能被已经抑制的 B 去掉。本次实现和测试保持精确贪心语义。

CUDA 后处理使用 `--fmad=false`，避免 IoU 乘加融合改变精确阈值边界；边界用例还覆盖等于阈值及 `nextafter` 的相邻浮点值。

### 8.4 最终结果回传

GPU NMS 模式传输固定结果容量：

```text
16 + min(max_det, N) × 32
```

默认 max_det=300 时为 9,616 bytes，相比完整输出减少约 99.66%。CPU 根据 resultCount 仅读取有效记录，再还原坐标和包装结果。尾部空槽位依然计入实际传输量。

## 9. 流执行、缓冲区与诊断张量

新路径保持同流顺序：

```text
H2D → GPU 预处理 → TensorRT
    → GPU 筛选 → 可选 GPU NMS → 结果 D2H
    → 原有 stream.synchronize()
    → CPU 解析/可选 CPU NMS/坐标还原
```

候选、结果、key、indices、mask、CUB scratch 和 pinned host 包均通过 RAII 管理，初始化分配、逐帧复用。每帧仅异步清零 header，没有逐帧 cudaMalloc，也没有每个阶段单独同步。

GPU 模式正常推理不再回传完整原始输出，因此旧 host raw-output buffer 可能是旧数据。为避免验证脚本导出陈旧张量，`dumpTensors()` 在诊断时显式从 device 回传原始输出，再写文件。此额外回传只用于 `--dump` 的验证运行，位于计时区间之外；最终 E2E 运行不开启 dump。

Graph 开关仍用于同一个 Engine 的推理提交对照；本轮没有把新增筛选/NMS kernel 捕获到 Graph 中。

## 10. 正确性验证如何建立

### 10.1 小型图片集

| 图片 | 尺寸 H×W | 候选数 | 最终框数 |
|---|---|---:|---:|
| bus.png | 1080×810 | 47 | 5 |
| dog.png | 329×302 | 10 | 1 |
| multi_mosaic.png | 1281×1281 | 41 | 4 |
| odd_landscape.png | 667×1001 | 10 | 1 |
| portrait.png | 901×577 | 10 | 1 |
| wide.png | 501×1203 | 10 | 1 |
| zidane.png | 720×1280 | 26 | 3 |

dog 的宽高比/奇数尺寸变体用于检查 letterbox 和边界，bus、zidane、mosaic 用于多目标与多类别结果。共七张图片，变体不等同于更多独立语义场景。

### 10.2 比较顺序

按下面的顺序定位差异：

1. 比较 GPU 预处理输入张量，检查 input.f32 的 SHA256。
2. 比较同一 Engine 原始输出，检查 output0.f32 的 SHA256。
3. 在相同原始输出上比较 CPU/GPU 后处理，分离数值计算与后处理差异。
4. 使用独立 Python 参考检查当前 C++ 完整结果，并记录跨后端差异。

C++ 检测 CSV 包含 image、class_id、class_name、confidence 和浮点 x1/y1/x2/y2。结果先按类别与 IoU 建立最大匹配，不按数组位置逐项配对；记录坐标差、分数差和未匹配结果。匹配通过后，再单独检查确定排序与逐值完全相等，以验证同分处理。

本轮同一 Engine 的三模式和 Graph 开关结果：**14 组对照、98 个图像比较通过；未匹配为零，坐标/分数差异为零，input/raw-output 逐位一致。** 新 CPU 路径也与保留的修改前二进制匹配。

### 10.3 Python 参考与预处理

主参考为 ONNX Runtime CPUExecutionProvider FP32；附加参考为 PT/PyTorch CPU FP32 fused eval。模型路径和执行后端写入 [comparison.json](results/cuda_postprocessing/final/python_reference/comparison.json)。

对齐设置为 imgsz=640、batch=1、auto=False、scale_fill=False、scaleup=True、center=True、padding=114、BGR→RGB、NCHW、float32/255、conf=.25、IoU=.45、class-aware、multi_label=False、max_det=300、max_nms=30000。COCO 类别 ID 固定，名字别名由已有参考脚本规范化。

沿用前一轮验证的 CUDA 插值配置，本轮没有改变插值算法。独立 Python/OpenCV uint8 插值与 CUDA 路径的输入差异、同后端预处理影响、同输入下 TRT/ONNX 原始输出差异分别记录。对同一 TRT 原始输出，Python/Ultralytics 后处理结构检查通过。

同一 Engine 的模式一致性采用零数值差异；FP16 TRT 与 FP32 ONNX/PT 的跨后端数值差异按报告保留，不要求逐位相等，也没有临时放宽检测阈值让检查通过。

### 10.4 合成与原始输出重放

1377 条测试结果覆盖：

- 零候选、同分数、同类别分数、跨类别重叠。
- conf 等于阈值和相邻浮点值，IoU 等于阈值和 `nextafter`。
- NMS 前不裁剪、不取整；零面积框。
- A/B/C 贪心链；max_det=1。
- 65/127/129/513/8400 全候选，部分 64-bit tile，确认不隐式 Top-K。
- 固定种子的密集随机框与大量同分。
- 非有限最高分数和不支持的候选容量保护。
- 实际 raw-output 文件重放，conf=.01/.25/.8/.99，max_det=1/300。

每个配置在同一组缓冲区上重复三次，检查旧计数/旧 mask 残留和原子顺序变化。测试条目包括模式、阈值和重复的实际文件，不代表 1377 张独立图片。

非连续 ROI 按行 copy，结果与连续输入完全一致；空/灰度/BGRA/float cv::Mat 明确拒绝。单独构建负向测试 Engine，实际验证 FP16 I/O 和 batch>1 被拒绝；这些小 Engine 不替换实际推理模型。

## 11. 实现过程中发现并修正的问题

### 11.1 构建缓存与 nvcc warning 参数

初次构建遇到旧 CMake CUDA 编译器隐式 include 缓存问题，C++ 编译找不到 cuda_runtime.h。使用 `cmake --fresh` 重新识别实际 CUDA Toolkit，显式设定 nvcc、sm_89、TensorRT 和 OpenCV 路径。

全局 GCC 的 `-Wall/-Wextra/-Wpedantic` 改为仅 CXX 编译使用，避免直接传给 nvcc。失败和成功配置/构建日志均保留。这是构建环境/参数修正，没有用模型或阈值调整绕过问题。

### 11.2 max_det=1 与合成验证入口冲突

第一次完整脚本在 max_det=1 配置下进入已有合成检查；其中早期用例断言返回三个框，却继承了 CLI 的最大输出数 1，导致验证入口报错。

修正方法：合成检查开始时显式设定自身预期的 max_det=300，内部再检查 max_det=1；调用者结束后恢复 CLI 配置。真实图片的 max_det=1 仍由三路径对照验证。首次中止日志留在 `verified/`，该轮没有进入最终性能测量。

### 11.3 GPU 贪心循环的跨 warp 同步风险

第一版选择 kernel 使用：

```cpp
for (int i = 0; i < count && keep < maxDet; ++i) {
    if (threadIdx.x == 0) {
        // 可能执行 keep++
    }
    __syncthreads();
    // 更新 removed mask
    __syncthreads();
}
```

`keep` 是 shared memory。各 warp 在循环条件中读取它，同时线程 0 可能在循环体更新它；如果观察到不同值，可能在 block barrier 附近出现不同退出决定。此前结果通过功能测试，不能据此排除该同步风险。

这是**静态审查发现的风险**，没有记录为修复前实测 racecheck 故障。修正后，只有线程 0 读取/更新 keep；它通过 shared chosen 发布是否退出的决定，block 同步后所有线程统一分支：

```cpp
for (int i = 0; i < count; ++i) {
    if (threadIdx.x == 0) {
        chosen = keep >= maxDet ? -1 : /* 当前框是否未被抑制 */;
        if (chosen > 0) results[keep++] = /* 当前候选 */;
    }
    __syncthreads();
    if (chosen < 0) break;
    // 被保留时，各线程并行 OR 对应的 removed mask
    __syncthreads();
}
```

保留 `pre_race_fix/` 及 [修复前报告](results/cuda_postprocessing/REPORT_PRE_RACE_FIX.md)，修正后重新完成全部正确性和性能测量。修复前后 CPU/混合路径也有延迟变化，因此没有将两轮时间差全部归因于这处修改。

### 11.4 补充 sanitizer

最初未从 PATH 找到工具，随后定位到 `/usr/local/cuda/bin/compute-sanitizer`，执行了以下检查：

| 检查 | 范围 | 实际结果 |
|---|---|---|
| memcheck | 修正后的完整合成测试 executable，含 CUB 调用 | 0 errors |
| racecheck | selectGreedy 的合成用例，包括满容量/部分 tile | 0 errors、0 warnings |
| racecheck | suppressionMasks 的七图 max_det=1 路径 | 0 errors、0 warnings |
| synccheck | selectGreedy 和 suppressionMasks 的合成用例 | 0 errors |

第一次未限定 kernel 的 TensorRT 入口 racecheck 在中断时尚未完成，其 `racecheck.log` 没有成功摘要，不计为通过。随后限定新增 kernel 进行检查，避免将 TensorRT 内部插桩耗时混入本轮检查范围。

sanitizer 仅证明上述运行覆盖中未检出对应错误，仍需更多 Engine/输入/长时间场景验证。

## 12. 性能测量方法

### 12.1 最终端到端

CPU 使用 steady_clock，从内存图像输入开始，到最终 Detection 可供 CPU 使用结束。范围包含 staging、H2D、GPU 预处理、推理、可选 GPU 后处理、D2H、等待完成和 CPU 后处理；排除读图、绘制、显示、结果写盘和诊断 dump。

测量参数：

| 参数 | 配置 |
|---|---|
| 输入 | 七张预先读入内存的图片 |
| warmup | 每图 30 次 |
| 每轮测量 | 每图 200 次，round-robin |
| 轮数 | 两轮，三模式顺序反转 |
| 单模式/Graph 配置样本量 | 7 × 200 × 2 = 2800 |
| Graph | on/off 均独立运行 |
| 最终 E2E CUDA Events | 关闭 |
| 统计 | 平均、P50、P95，保留两轮和逐图数据 |

缓冲区在初始化和预热中分配并复用，稳定测量期间没有为新增后处理逐帧分配显存。CSV 在单次计时结束后写入。源码与测量 executable SHA256 在运行前后检查一致；测量期间没有并行构建或运行 Python 模型参考。

### 12.2 分阶段 profiling

独立 `--profile` 运行，每模式每图 30 warmup、200 测量，共 1400 个样本。

- CPU staging copy、CPU 后处理：steady_clock。
- H2D、GPU 预处理、TensorRT、GPU 筛选压缩、GPU NMS、D2H：同流 CUDA Events。
- Events 在原有一次 stream 完成同步后读取，不在每个阶段新加同步。
- GPU NMS 阶段包括 key 生成、CUB 排序、mask 和 greedy 选择。
- CPU 模式的 GPU filter/NMS、混合模式的 GPU NMS 是空 hook，事件仍会有微小间隔。

阶段数据用于定位开销；GPU 执行和 CPU 提交/等待的时间关系不能用“阶段均值机械相加”代替 E2E。profiling 本身可能影响小模型，最终性能只采用关闭 profiling 的数据。

## 13. 修正后最终性能

### 13.1 无 Events 的 E2E

| Graph | 模式 | 平均 ms | P50 ms | P95 ms | 每帧 D2H bytes |
|---|---|---:|---:|---:|---:|
| on | cpu | 2.2904 | 2.2334 | 2.7899 | 2,822,400 |
| on | gpu-filter | 1.6032 | 1.5388 | 2.1296 | 268,816 |
| on | gpu-nms | 1.7485 | 1.6885 | 2.3150 | 9,616 |
| off | cpu | 3.1221 | 3.0520 | 4.1035 | 2,822,400 |
| off | gpu-filter | 2.2847 | 2.1325 | 3.2758 | 268,816 |
| off | gpu-nms | 2.6381 | 2.6221 | 3.6975 | 9,616 |

Graph on 混合路径平均减少 **30.00%**。完整 GPU NMS 相比混合路径增加 **0.1452 ms**，约 **9.06%**。Graph off 也保留真实结果，不能把 Graph 收益混到 GPU 后处理收益中。

两轮单独均值：

| Graph | 模式 | round 0 ms | round 1 ms |
|---|---|---:|---:|
| 1 | cpu | 2.2949 | 2.2860 |
| 1 | gpu-filter | 1.6166 | 1.5899 |
| 1 | gpu-nms | 1.7436 | 1.7533 |
| 0 | cpu | 3.2382 | 3.0060 |
| 0 | gpu-filter | 2.3470 | 2.2223 |
| 0 | gpu-nms | 2.6993 | 2.5769 |

所有模式均使用同一个实际 Engine、相同阈值、预处理和输出限制；结果一致后才进入性能比较。

### 13.2 Graph on 的独立阶段均值

单位为 ms；设备字节数为新增持久化分配。

| 模式 | staging | H2D | preprocess | TRT | GPU filter | GPU NMS | D2H | CPU post | 额外 device bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | 0.1840 | 0.2011 | 0.0678 | 0.9545 | 0.0015 | 0.0015 | 0.2648 | 0.5038 | 0 |
| gpu-filter | 0.1717 | 0.1958 | 0.0704 | 0.9624 | 0.0076 | 0.0015 | 0.0597 | 0.0018 | 268,816 |
| gpu-nms | 0.1717 | 0.1910 | 0.0640 | 0.9661 | 0.0076 | 0.1876 | 0.0309 | 0.0007 | 9,722,527 |

从这些数据可以解释收益来源：GPU 筛选替代了完整输出的 CPU 类别扫描，压缩包同时降低 D2H；混合路径的 CPU 后处理仅剩少量候选的排序、NMS 和包装。完整 GPU NMS 虽继续降低 D2H，却增加 radix sort、mask 和顺序选择的 GPU/提交开销。

候选只有 10～47 个时，GPU NMS 的额外开销足以抵消更小回传包的收益。因此决定采用混合路径有实测依据；没有预先假设全部迁移 GPU 必然更快。

## 14. 复现命令

### 14.1 环境和构建

```bash
cd /home/wnyxvo/YOLOs-CPP-TensorRT
source /home/wnyxvo/miniconda3/etc/profile.d/conda.sh
conda activate yolos-trt

cmake --fresh -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DTENSORRT_DIR="$CONDA_PREFIX/opt/TensorRT-10.7.0.23" \
  -DOpenCV_DIR="$CONDA_PREFIX/lib/cmake/opencv4"
cmake --build build -j4
```

本次 GPU 验证在解除执行沙箱 GPU 限制的命令环境中运行；项目无需写入“沙箱开关”或改变 CUDA 算法。

### 14.2 全套验证与测量

输出目录须为空，脚本拒绝覆盖旧证据。已有七图数据集位于 `results/baseline/samples/`。

```bash
python scripts/cuda_postprocessing/run.py \
  --output results/cuda_postprocessing/reproduction \
  --original-prefix results/cuda_postprocessing/original/validation \
  --warmup 30 --runs 200 --rounds 2

python scripts/cuda_postprocessing/report.py \
  results/cuda_postprocessing/reproduction \
  --output results/cuda_postprocessing/REPRODUCTION_REPORT.md \
  --original-prefix results/cuda_postprocessing/original/e2e
```

脚本先执行三模式/Graph 正确性、原始张量重放、Python 对照和阈值/max_det 检查，全部通过后才进行最终 E2E；最后独立 profiling。失败时保留日志，不放宽阈值继续测量。

### 14.3 单独 E2E 与 profiling

```bash
# 不带 Events，最终端到端
./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/cuda_postprocessing/custom_e2e \
  --postprocess gpu-filter --graph 1 --warmup 30 --runs 200

# 独立运行分阶段 profiling
./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/cuda_postprocessing/custom_profile \
  --postprocess gpu-filter --graph 1 --warmup 30 --runs 200 --profile

# 两次实验只修改 --postprocess/--graph，复用同一个 Engine
# --postprocess cpu | gpu-filter | gpu-nms
# --graph 0 | 1
```

### 14.4 正确性、图像入口和 sanitizer

```bash
./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/cuda_postprocessing/custom_check \
  --postprocess gpu-nms --graph 1 --warmup 5 --runs 1 --dump --checks

./build/image_inference models/yolo11n.trt data/dog.jpg models/coco.names \
  --postprocess gpu-filter --graph 1 --conf .25 --iou .45 --max-det 300 \
  --no-display --csv results/cuda_postprocessing/image_results.csv

./build/cuda_postprocessing_test results/cuda_postprocessing/custom_tests.csv \
  results/cuda_postprocessing/original

/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build/cuda_postprocessing_test results/cuda_postprocessing/custom_memcheck.csv

/usr/local/cuda/bin/compute-sanitizer --tool racecheck \
  --kernel-name kns=selectGreedy --error-exitcode 99 \
  ./build/cuda_postprocessing_test results/cuda_postprocessing/custom_racecheck.csv

/usr/local/cuda/bin/compute-sanitizer --tool racecheck \
  --kernel-name kns=suppressionMasks --error-exitcode 99 \
  ./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/cuda_postprocessing/custom_masks \
  --postprocess gpu-nms --graph 0 --warmup 0 --runs 1 --max-det 1

/usr/local/cuda/bin/compute-sanitizer --tool synccheck \
  --kernel-name kns=selectGreedy --kernel-name kns=suppressionMasks \
  --error-exitcode 99 ./build/cuda_postprocessing_test \
  results/cuda_postprocessing/custom_synccheck.csv
```

图像演示入口用于展示结果，性能数字以 baseline_detection 的明确计时范围为准。测试工具可以选第二参数重放目录内的 `[1,84,8400]` raw-output 文件。

## 15. 证据索引与已验证边界

| 证据 | 路径 |
|---|---|
| 最终简明报告 | [REPORT.md](results/cuda_postprocessing/REPORT.md) |
| 最终环境/模型信息 | [environment.json](results/cuda_postprocessing/final/environment.json) |
| 源码和二进制 SHA256 | [source-sha256.json](results/cuda_postprocessing/final/source-sha256.json) |
| 匹配框、差异、未匹配结果 | [parity.json](results/cuda_postprocessing/final/parity.json) |
| Python 三层差异定位 | [comparison.json](results/cuda_postprocessing/final/python_reference/comparison.json) |
| 合成与实际输出重放 | [kernel_tests.csv](results/cuda_postprocessing/final/kernel_tests.csv) |
| 两轮合并 E2E | [pooled.json](results/cuda_postprocessing/final/pooled.json) |
| 各轮/逐图/阶段统计 | [summary.json](results/cuda_postprocessing/final/summary.json) |
| 原始最终计时 CSV 示例 | [e2e_r0_g1_gpu-filter.timings.csv](results/cuda_postprocessing/final/e2e_r0_g1_gpu-filter.timings.csv) |
| 实际运行参数与退出码 | [commands.json](results/cuda_postprocessing/final/commands.json) |
| FP16 I/O 与 batch>1 拒绝 | [guards.json](results/cuda_postprocessing/final/engine_guards/guards.json) |
| CUDA 内存检查 | [memcheck.log](results/cuda_postprocessing/memcheck.log) |
| 贪心选择竞态检查 | [racecheck_greedy.log](results/cuda_postprocessing/racecheck_greedy.log) |
| mask 竞态检查 | [racecheck_masks.log](results/cuda_postprocessing/racecheck_masks.log) |
| block 同步检查 | [synccheck.log](results/cuda_postprocessing/synccheck.log) |
| 实现接口和复现说明 | [CUDA_POSTPROCESSING.md](doc/CUDA_POSTPROCESSING.md) |

**已实际验证：** 本机 YOLO11n FP32 I/O Engine 的三模式、Graph 开关、七图一致性、ONNX/PT 参考、阈值/最大输出数、非连续输入、容量边界、原始输出重放、限定范围 sanitizer、各入口编译兼容。

**静态检查发现并处理：** GPU 贪心选择共享计数循环退出风险；固定包边界、flags、排序键、NMS 前坐标精度、同步位置和不支持形状的拒绝逻辑。

**尚未验证：** 完整 COCO mAP，其他 Engine/YOLOv8 的实际推理，其他 GPU，长期视频稳定性，更高候选密度的真实数据集，以及完整后处理 Graph 捕获收益。小样本一致性不等于完整精度验证；若引入 Top-K，必须新增独立精度验证，不能复用本次“无截断”结论。

## 16. 下一步优先级

1. 使用 `--postprocess gpu-filter --graph 1`，以修正后的最终结果作为后续性能基线；CPU 和 GPU NMS 模式继续保留用于回归。
2. 当前 TRT 约 0.96 ms，staging/H2D 也占有可观成本。下一轮可评估扩展 Graph 捕获或减少提交与数据搬运成本，按阶段与 E2E 重新测量。本轮未实施，不预设提升幅度。
3. GPU NMS 对少候选没有 E2E 优势；后续应在真实密集候选场景分析全容量排序、bitmask 和贪心选择的成本，再判断部署价值。
4. 扩大图片/视频覆盖并验证完整 mAP，之后才考虑 Top-K、INT8 或多帧流水线等会改变精度或执行行为的优化。

本轮的工程决策来自正确性先行、保留原始证据、分阶段定位与同版本 E2E 对照。当前交付不仅包含更小的回传包，也包含可切换实现、明确边界、可复现实验，以及选择混合路径的实测依据。
