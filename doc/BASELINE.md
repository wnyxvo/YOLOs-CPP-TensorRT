# 可验证的 YOLO11n TensorRT 基线

本轮只包含正确性对照、最小修复、单帧串行性能测量。保留既有 pinned staging、融合 CUDA 预处理和仅捕获 `enqueueV3` 的 CUDA Graph；未实现 GPU NMS、多帧流水线或 INT8。

实测报告见 [实验报告](../results/baseline/REPORT.md)，原始数据见 `results/baseline/final_original/`，修正后基线见 `results/baseline/final_corrected/`。张量和保存的二进制保留在本地但不加入 Git，JSON、CSV、复现记录和原实现测量补丁可加入版本控制。

## 复现整套实验

从仓库根目录执行。下列路径是本机已经使用的依赖路径；其他机器需调整 TensorRT/OpenCV 路径和 GPU architecture。

```bash
source /home/wnyxvo/miniconda3/etc/profile.d/conda.sh
conda activate yolos-trt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DTENSORRT_DIR="$CONDA_PREFIX/opt/TensorRT-10.7.0.23" \
  -DOpenCV_DIR="$CONDA_PREFIX/lib/cmake/opencv4"
cmake --build build -j 4
python scripts/baseline/make_samples.py
python scripts/baseline/run.py --output results/baseline/my_run --warmup 30 --runs 200
```

`run.py` 拒绝覆盖已有实验目录。使用已经存在的 `models/yolo11n.{pt,onnx,trt}`；不重新导出或构建正式 Engine。GPU 命令应在允许访问 GPU 的执行环境中运行。依赖为本项目 C++ 依赖及已有 Python 的 PyTorch、Ultralytics、OpenCV、ONNX、ONNX Runtime、NumPy；无需 TensorRT Python 模块。

图片集由本地 dog 图片、安装包中 bus/zidane 及确定性变换产生，保存为 PNG 和 SHA256 manifest。7 图来自三个场景，包含奇数尺寸、宽图、竖图和四目标拼图。不是 mAP 数据集。

顺序执行环境记录、Graph 开关张量验证、Python 参考、Graph 精确对照、Engine 拒绝测试，然后单独运行四组性能测试。不要在性能运行期间同时编译、跑 Python 模型或运行其他 GPU 工作。

## 单独运行

机器可读结果包含图像 basename、类别 ID、名称、置信度和浮点 xyxy。类别 ID 是对照键；仓库旧 COCO 名称的六个别名映射记录在参考报告中。

```bash
./build/image_inference models/yolo11n.trt results/baseline/samples models/coco.names \
  --no-display --conf 0.25 --iou 0.45 --max-det 300 --graph 1 \
  --csv results/baseline/detections.csv

./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/baseline/check \
  --warmup 5 --runs 1 --dump --checks --graph 1

python scripts/baseline/reference.py --cpp-prefix results/baseline/check \
  --images results/baseline/samples --pt models/yolo11n.pt \
  --output results/baseline/check_reference --require-structural-pass

./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/baseline/e2e \
  --warmup 30 --runs 200 --graph 1

./build/baseline_detection models/yolo11n.trt models/coco.names \
  results/baseline/samples results/baseline/profile \
  --warmup 30 --runs 200 --graph 1 --profile

python scripts/baseline/summarize.py results/baseline/e2e results/baseline/profile \
  --output results/baseline/my_summary.json
```

将 `--graph 1` 改为 `--graph 0` 使用相同 Engine 的普通 enqueueV3 路径。两种模式都允许构造期间预热和捕获；关闭模式在所有图像测量中只使用 enqueueV3。默认推理仍开启 Graph。`--dump` 和 `--checks` 仅用于正确性实验，不用于最终端到端基准。

`image_inference` 保留默认置信度 0.4；参考与基准统一为 0.25，因此单图对照必须显式传阈值。基准固定 max_det=300，class-aware NMS、multi_label=false、IoU=0.45。默认参考使用 **ONNX Runtime CPUExecutionProvider FP32**，可选 PT 对照使用 **PyTorch CPU FP32、fused eval**。

## 正确性与支持边界

- 预处理：640×640、batch=1、center=true、auto=false、scale_fill=false、scaleup=true、padding=114、BGR→RGB、NCHW FP32 /255。CUDA 使用半像素双线性坐标并量化为 uint8 后归一化。OpenCV 使用自己的整数插值实现，因此还可能存在一个 uint8 code value 的差异；不是逐元素完全相同的预处理。
- 主输出 `[1,84,8400]` 用浮点模型输入坐标做 NMS，在 NMS 后逆变换和裁剪；只有画框表示使用整数。按本机 Ultralytics 8.4.174 的 `scale_boxes`，分别使用取整后 resized width/height 计算 gain_x/gain_y。升级 Ultralytics 后必须重新确认此契约。
- 匹配先要求同类别且 IoU≥0.9，再进行最大匹配数的二分图匹配；优先遍历较高 IoU 的边。该阈值只是建立对应关系，报告保留每一对坐标差、分数差和未匹配结果，不把匹配成功当作完整精度验证。
- 相同 TRT 原始输出的 Python/C++ 后处理检查使用坐标误差 <0.0005 px、分数误差 <1e-7、无未匹配结果。预处理检查的预声明上限为一个 uint8 灰度级（1/255+1e-6），并保留真实误差，未靠放宽置信度或 NMS 阈值通过检查。
- 支持非连续 `CV_8UC3` ROI：逐行拷贝到原有 pinned staging。空图、gray、BGRA、float 图像明确拒绝。
- Session 只支持一个正尺寸静态 FP32、device LINEAR、非向量化 Engine 输入 `[1,3,H,W]`，以及 FP32 输出。本基准入口另外要求 640×640 COCO YOLO11 标准输出。FP16 compute flag 与 I/O dtype 分开记录；FP16 I/O 不转换，明确拒绝。
- FP16 I/O 和 batch>1 拒绝已用真实小型 Engine 验证。动态形状、其他 I/O dtype/布局的拒绝为静态检查；没有为所有情形生成 Engine。其他 YOLO 版本及分类/分割/姿态等任务只检查编译，本轮不宣称其精度已验证。

定位链路：CUDA input vs Python CPU input → 相同 ONNX 后端分别运行两种 input → 同一 CUDA input 的 TRT vs ONNX raw output → 相同 TRT raw output 的 Python vs C++ 后处理。修正版本还保存独立 C++ OpenCV input，以及该 input 的 TRT 输出，区分 CUDA 插值和 Engine 精度差异。

`.input.f32` / `.output0.f32` 为 little-endian FP32 连续二进制；形状和字节数见 `.engine.csv`。`.opencv.*` 是仅在诊断中使用的 CPU 预处理路径，不进入测量区间。Python 张量为 `.npy`。

## 测量范围

端到端采用 `steady_clock`，范围从内存中的 `cv::Mat` 输入到最终检测结果可供 CPU 使用，包含 staging、上传、预处理、推理、下载、既有 stream 同步和 CPU 后处理；不包含读图、绘制、显示或写盘。热身和测量都轮流遍历全图片集，`runs`/`warmup` 均指每张图片的次数。

分阶段模式用 CPU steady_clock 测 staging/后处理，用同一 stream 上的五个 CUDA Event 边界测 H2D/预处理/TRT/D2H，仅在既有单次 stream synchronize 后读取 elapsed time，未增加逐阶段同步。Event 区间可包含提交间隙，不是纯内核忙碌时间。分阶段模式本身增加事件记录和读取开销，因此最终端到端以无事件组为准；两组不能机械相加或混用。

保留逐样本小数毫秒 CSV、配置、图片级和总体 mean/P50/P95。分位数使用 NumPy 默认 linear interpolation。本机 GPU 有桌面负载、未锁频/固定功耗，这是一轮测量结果，不是持续视频 FPS、功耗曲线或跨设备保证。

## 重建原实现测量入口

起始 commit 和工作区状态保存在 `results/baseline/original/`；起始工作区干净。`instrumentation.patch` 只补充诊断/测量功能，保留原插值和整数框 NMS。可在独立临时 checkout 中从该 commit 应用此补丁并构建 `baseline_detection`，或直接使用本机保存的原二进制：

```bash
python scripts/baseline/run.py --output results/baseline/original_retest \
  --binary results/baseline/original/baseline_detection.binary --legacy \
  --warmup 30 --runs 200
```

原版本未实现 max_det 限制；本次原版输出数量远小于 300，配置 CSV 的 max_det=300 是对照目标值。原版参考检查会保留失败证据而不阻断性能采集。正式 Engine 的 SHA256 在前后版本中必须相同。

收尾时出现来源未确认的额外按类别平面扫描 CPU 优化，用户确认未修改。其源码、二进制、差异补丁和独立实验已隔离保存于 `current_snapshot/`、`final_current/`；当前工作区恢复为已测过的最小修正基线。该额外优化不计入本轮交付变更。
