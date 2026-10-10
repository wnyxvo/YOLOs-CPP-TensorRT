> 保留的修复前数据，不能作为最终基线。静态审查发现 selectGreedy 的共享 keep 循环条件可能跨 warp 竞态；后续修复将退出决定限定在线程 0，并同步后统一退出。最终版本见 REPORT.md。

# CUDA 后处理本机实测报告

结论：当前小候选场景优先使用 `gpu-filter`（GPU 筛选/压缩 + CPU NMS）。GPU NMS 实现并验证通过，保留为对照模式。默认 CPU 模式不自动改变。

采集时间：2026-10-10T15:38:13.425922+08:00；起始 commit：`89e7da64fd3ee80e6f51cc447b4d018da04a1459`。当前修改及源码/executable SHA256 见 [环境](pre_race_fix/environment.json) 与 [源文件清单](pre_race_fix/source-sha256.json)；完整新文件快照位于 `final/source/`。

## 环境与模型

RTX 4060 Laptop GPU，驱动 591.86，CUDA Toolkit 12.4.99，TensorRT C++ 10.7.0.23，OpenCV C++ 4.10.0。WSL/Linux 本机运行；未固定功耗/时钟，尾延迟受系统调度和设备状态影响。

Python 包版本（实际采集）：`{"ultralytics": "8.4.174", "torch": "2.14.1+cu130", "cv2": "5.0.0", "onnx": "1.23.2", "onnxruntime": "1.30.0", "tensorrt": {"unavailable": "No module named 'tensorrt'"}}`。

复用 YOLO11n Engine；FP16 构建，实际 FP32 输入 `[1,3,640,640]`（4,915,200 bytes）、FP32 输出 `[1,84,8400]`（2,822,400 bytes）。导出命令 `python models/export_onnx.py --model yolo11n`；Engine 构建命令 `./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16 --workspace 2048` 为先前用户执行信息，本轮不重建。

| 模型文件 | SHA256 |
|---|---|
| models/yolo11n.pt | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| models/yolo11n.onnx | `86c32d35d058eaf9b84b4d159ac779fa4160fa735593112fcf07a1f278f70fc7` |
| models/yolo11n.trt | `de97ab4e249a501cd138e0cc2ce66d38c5b69615933bf11d30d2bfddae8b1270` |

## 正确性

- 14 组对照、98 个图像比较全部通过。按类别/IoU 匹配，未匹配结果为零，同一 Engine 的坐标/分数差异为零，并额外验证确定排序。Graph on/off、三模式的 input/raw-output SHA256 一致；新 CPU 模式与修改前二进制也一致。
- 1377 条 CUDA 测试结果通过（同一配置复用缓冲区 3 次）；含合成语义边界和真实张量重放，不代表 1377 张独立图片。实际图片 7 张，含不同宽高比、奇数尺寸与多目标；阈值 .01/.25/.8/.99、max_det=1/300。
- 非连续 ROI 与连续图像相等；空/灰度/BGRA/float 输入拒绝。所有模式的 checks.txt 保存原始检查结果。
- Python 主参考：ONNX Runtime CPU FP32；附加参考：PT/PyTorch CPU FP32 fused eval。对同一 TRT 原始输出，Ultralytics 后处理检查全通过。FP16 TRT 与 FP32 ONNX/PT 的数值差异记录在 Python 报告中，没有假定跨后端逐位一致。
- 未引入 Top-K：容量 N=8400，超过支持 N=30000 的 Engine 显式拒绝；max_det 在精确 class-aware NMS 后生效。同分取原始预测索引较小者，类别同分取 ID 较小者。

[逐框匹配](pre_race_fix/parity.json) · [CUDA 测试](pre_race_fix/kernel_tests.csv) · [Python 差异分解](pre_race_fix/python_reference/comparison.json)

## 无事件端到端性能

范围：内存 cv::Mat 到 CPU 最终检测结果，不含读图/绘制/显示/写盘。每图 warmup=30；每轮每图测量=200；七图 round-robin，两轮反转模式顺序。每种模式/Graph 设置共 2800 次；P50/P95 为线性插值分位数。

| Graph | 后处理 | 平均 ms | P50 ms | P95 ms | 实际 D2H bytes/帧 |
|---|---|---:|---:|---:|---:|
| on | cpu | 2.1783 | 2.1214 | 2.6914 | 2,822,400 |
| on | gpu-filter | 1.4601 | 1.4071 | 1.9464 | 268,816 |
| on | gpu-nms | 1.5867 | 1.5307 | 2.1240 | 9,616 |
| off | cpu | 3.1019 | 3.0893 | 4.0777 | 2,822,400 |
| off | gpu-filter | 2.3436 | 2.2542 | 3.3587 | 268,816 |
| off | gpu-nms | 2.5756 | 2.5666 | 3.6298 | 9,616 |

Graph on：混合路径比本轮 CPU 均值减少 32.97%，从 2.1783 到 1.4601 ms；GPU NMS 比混合路径增加 0.1266 ms（8.67%）。这是该 Engine/样本/设备下的观测，不推断普适加速倍数。

固定包的 D2H 相比完整输出减少 90.48%（gpu-filter）和 99.66%（gpu-nms，max_det=300）。这些 bytes **包含无效尾槽位**，不宣称只传有效框的精确字节数；没有为读取候选数额外同步。

| 图片 | 过阈值候选数（conf=.25） | 最终框数 |
|---|---:|---:|
| bus.png | 47 | 5 |
| dog.png | 10 | 1 |
| multi_mosaic.png | 41 | 4 |
| odd_landscape.png | 10 | 1 |
| portrait.png | 10 | 1 |
| wide.png | 10 | 1 |
| zidane.png | 26 | 3 |

两轮单独均值（避免只呈现 pooled 结果）：

| Graph | 模式 | round 0 ms | round 1 ms |
|---|---|---:|---:|
| 1 | cpu | 2.1764 | 2.1802 |
| 1 | gpu-filter | 1.4666 | 1.4535 |
| 1 | gpu-nms | 1.5986 | 1.5747 |
| 0 | cpu | 3.0580 | 3.1458 |
| 0 | gpu-filter | 2.2670 | 2.4203 |
| 0 | gpu-nms | 2.6586 | 2.4925 |

修改前原始二进制数据保留：1400 次，均值 2.1513 ms，P50 2.1013 ms，P95 2.6523 ms。该次运行时间不同、仅一轮；优化对比以本轮两轮同一新二进制的 CPU/GPU 模式为准，不混用历史测试。

## 独立 profiling

下表为 Graph on、独立 profiling run 的阶段均值，每模式 1400 次。GPU 用同流 Events，在既有一次完成同步后读取；CPU 用 steady_clock。NMS GPU 时间含 key生成/CUB排序/抑制mask/greedy选择。事件本身可影响小模型，**不用于替代上表 E2E，不相加估算 E2E**。

| 模式 | staging | H2D | preprocess | TRT | GPU filter | GPU NMS | D2H | CPU post | 额外 device bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | 0.1782 | 0.1924 | 0.0669 | 0.9527 | 0.0015 | 0.0015 | 0.2638 | 0.4871 | 0 |
| gpu-filter | 0.1698 | 0.1963 | 0.0626 | 0.9535 | 0.0077 | 0.0015 | 0.0546 | 0.0017 | 268,816 |
| gpu-nms | 0.1676 | 0.1971 | 0.0574 | 0.9615 | 0.0078 | 0.1617 | 0.0267 | 0.0007 | 9,722,527 |

CPU 模式 filter/NMS、混合模式 GPU NMS 是空 hook，记录值是事件间隔开销。CUDA 内存数字为新增持久化 buffer/scratch，不含 Engine/原运行时分配，也不是系统峰值显存。

[汇总与逐图数据](pre_race_fix/summary.json) · [两轮合并](pre_race_fix/pooled.json) · [运行命令](pre_race_fix/commands.json)。原始 `*.timings.csv` 均在同目录，包含小数毫秒、candidate/result count 与实际 D2H 字节数。

## 后续优先级与验证边界

1. 以 `gpu-filter` + CPU NMS 作为当前 YOLO11n 部署性能基线；该样本候选仅 10～47 个，CPU NMS 成本很小，先保持明确的模式开关。
2. 根据 TRT、预处理、H2D 的实测占比，再评估减少提交开销/扩展 Graph 捕获或更广的输入场景；这轮未实施，不预设收益。
3. GPU NMS 后续应针对更多候选的真实场景单独测量排序与 bitmask 开销；没有证据证明当前少候选场景值得全部迁移。若引入 Top-K，须重新验证精度，不能沿用本次结果。

已实际验证：本机 YOLO11n 的三路径、Graph 开关、七图匹配、ONNX/PT 参考、边界及实际原始输出重放。静态检查：容量与 flags、固定包布局、同步位置、未支持形状拒绝。尚未验证：完整 mAP、其他 Engine/设备、长视频稳定性、CUDA sanitizer（环境未提供命令）。七图一致性不是完整 mAP 结论。

原始版本保留于 `original/`；`verified/` 保留首次 max_det=1 测试入口配置冲突日志，未进入性能测量；修复后完整通过的数据在 `final/`。必要修复仅将验证入口合成用例的 max_det 与 CLI 独立，另为两模式明确确定排序、有限值/容量保护；沿用既有预处理与后 NMS 裁剪。

实现与复现详见 [CUDA_POSTPROCESSING.md](../../doc/CUDA_POSTPROCESSING.md)。
