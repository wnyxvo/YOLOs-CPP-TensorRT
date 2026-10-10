#!/usr/bin/env python3
"""Render the measured evidence into a root Markdown report; invent no metrics."""
import argparse, csv, json, re
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results/async_pipeline/final'))
    p.add_argument('--output',type=Path,default=Path('多帧异步流水线实现与实测.md'));a=p.parse_args()
    load=lambda name:json.loads((a.results/name).read_text())
    summary,env,completion=load('summary.json'),load('environment.json'),load('completion.json')
    if completion['status']!='completed':raise RuntimeError('Complete validation/performance data required')
    configs=summary['configurations'];base=configs[0]['completed_fps']
    table=['| 路径 | 完成吞吐 FPS | 相对同步 | 平均延迟 ms | P50 ms | P95 ms |','|---|---:|---:|---:|---:|---:|']
    queue=['| 槽位 | admitted 平均/P95 ms | submit wait 平均/P95 ms |','|---|---:|---:|']
    rounds=['| 路径 | round 0 FPS | round 1 FPS |','|---|---:|---:|']
    for r in configs:
        label='同步' if not r['slots'] else f"异步 {r['slots']} 槽"
        l=r['latency_ms'];q=r['submit_wait_ms'];ad=r['admitted_latency_ms']
        table.append(f"| {label} | {r['completed_fps']:.2f} | {(r['completed_fps']/base-1)*100:+.2f}% | {l['mean']:.4f} | {l['p50']:.4f} | {l['p95']:.4f} |")
        queue.append(f"| {r['slots']} | {ad['mean']:.4f} / {ad['p95']:.4f} | {q['mean']:.4f} / {q['p95']:.4f} |")
        raw=r['rounds_raw'];rounds.append(f"| {label} | "+' | '.join(f"{float(v['completed_fps']):.2f}" for v in raw)+' |')
    traces=[load(f'timeline_s{i}.analysis.json') for i in range(4)]
    trace_table=['| 槽位 | H2D∩kernel ms | D2H∩kernel ms | CPU NMS/包装∩kernel ms | kernel忙碌时间/批次 |','|---|---:|---:|---:|---:|']
    calls_table=['| 槽位 | EventSynchronize | StreamSynchronize | DeviceSynchronize | malloc/pinned分配/free/事件创建 |','|---|---:|---:|---:|---:|']
    for i,t in enumerate(traces):
        if t['status']!='executed':raise RuntimeError('Missing real GPU activity timeline')
        overlap=t['overlap_ms'];cpu='未标注' if i==0 else f"{overlap['cpu_post_kernels']:.6f}"
        trace_table.append(f"| {i} | {overlap['h2d_kernels']:.6f} | {overlap['d2h_kernels']:.6f} | {cpu} | {t['union_busy_ms']['kernels']/t['batch_ms']*100:.2f}% |")
        c=t['cuda_api_calls'];alloc=sum(v for k,v in c.items() if any(x in k for x in ('Malloc','Alloc','Free','EventCreate','EventDestroy','StreamCreate','StreamDestroy')))
        calls_table.append(f"| {i} | {c.get('cudaEventSynchronize',0)} | {c.get('cudaStreamSynchronize',0)} | {c.get('cudaDeviceSynchronize',0)} | {alloc} |")
    shapes=[]
    for row in (a.results/'sample_images.tsv').read_text().splitlines():
        _,path=row.split('\t',1)
        import cv2
        im=cv2.imread(path);shapes.append(f"`{Path(path).name}` {im.shape[1]}×{im.shape[0]}")
    cpp_trt=env['tensorrt_cpp_runtime'];cache=env['cmake_cache']
    opencv=re.search(r'FIND_PACKAGE_MESSAGE_DETAILS_OpenCV:INTERNAL=.*\[v([^()]*)\(',cache)
    toolkit=re.search(r'release ([^,]+), V([^\s]+)',env['commands']['cuda_toolkit']['stdout'])
    trt_version=cpp_trt['getInferLibVersion']
    frames=summary['frames_per_round'];warmup=summary['warmup_per_image'];images=summary['images']
    gpu=env['commands']['gpu']['stdout'].strip()
    text=f'''# 有界多帧异步流水线：实现、验证与真实测量

本轮完成了 **单 TensorRT Context、1/2/3 个有界槽位、submit/collect、事件依赖、背压与排空**，保留同步 detect 和 cpu/gpu-filter/gpu-nms 三路径。推荐当前吞吐场景采用 **2 槽**；低延迟单帧场景继续使用同步路径。

在相同 Engine、gpu-filter + CPU NMS、推理 Graph 开启条件下，2 槽完成吞吐提高 **{(configs[2]['completed_fps']/base-1)*100:.2f}%**。平均逐帧延迟也增大，不能把吞吐收益写成单帧延迟降低。Nsight **没有观测到 H2D/D2H 与 kernel 重叠**，因此不能宣布实现了已验证的 DMA/计算重叠加速。完整数据集 mAP 尚未执行：本地缺少真实标签。

## 1. 冻结旧版本与实验条件

- 原同步代码 commit：`{env['baseline_commit']}`，冻结时工作区干净。旧二进制、源归档、hash、复现结果独立保存在 `results/async_pipeline/baseline`，此前 `results/cuda_postprocessing/final` 原始结果也保留。
- 当前测量使用该 commit 之上的工作区变更，**尚未创建新 Git commit**。精确源快照在 `results/async_pipeline/final/source`；源 SHA256 和二进制 SHA256 在 `completion.json`。二进制：`{completion['binary_sha256']}`。正式运行前后检查源文件和二进制未改变。
- GPU、驱动及运行前状态原始查询：

```text
{gpu}
```

- 系统：`{env['platform']}`；Python `{env['python'].splitlines()[0]}`。
- CUDA Toolkit：`{toolkit.group(2)}`；TensorRT C++ runtime `{trt_version//10000}.{trt_version//100%100}.{trt_version%100}`，`getInferLibVersion={cpp_trt['getInferLibVersion']}`，加载 `{cpp_trt['path']}`。Python tensorrt 模块未安装，并不妨碍 C++ TensorRT 运行。
- Python 依赖：Ultralytics `{env['python_packages']['ultralytics']}`、PyTorch `{env['python_packages']['torch']}`、OpenCV `{env['python_packages']['cv2']}`。C++ OpenCV `{opencv.group(1)}`，Release、CUDA architecture=89；不能混用 Python cv2 版本描述 C++ 依赖。
- 模型为 YOLO11n，batch=1，imgsz=640。固定同一个现有 Engine，不重新导出或构建：

| 模型 | SHA256 |
|---|---|
'''
    for path,digest in completion['model_hashes'].items():text+=f'| `{path}` | `{digest}` |\n'
    text+='''
用户报告的导出/构建命令保存在 environment：`python models/export_onnx.py --model yolo11n`；`./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16 --workspace 2048`。构建 FP16 不等于 FP16 I/O，本轮实际 Engine I/O：

| Tensor | dtype | shape | bytes |
|---|---|---|---:|
| images | float32 | [1,3,640,640] | 4,915,200 |
| output0 | float32 | [1,84,8400] | 2,822,400 |

每次程序初始化均打印实际 I/O。基类构造拒绝非 FP32、非正静态形状、非 batch1 NCHW RGB 三通道 I/O 布局；GPU 后处理额外检查标准 YOLOv8/v11 单输出。没有静默解释 FP16/动态输入。

后处理保持 conf=.25、IoU=.45、class-aware、max_det=300；取最大类别时同分选较小类别 ID，候选按置信度降序，同分按原 prediction index 升序；IoU 严格大于阈值才抑制。最多保存全部 8400 个预测的有效候选，未引入 Top-K。坐标在 NMS 前保持模型空间浮点值，NMS 后按每帧 gain/pad 还原并裁剪。

## 2. 基线精度收尾

新增 `dataset_accuracy_detection` 与 `scripts/async_pipeline/dataset_accuracy.py`，在同一 Engine 上分别运行 CPU/gpu-filter 后处理，逐图按类别和 IoU 匹配，再用真实 COCO 标注执行 COCOeval。保留 .25 的部署阈值，并记录 COCOeval maxDets=[1,10,100]；此阈值下 AP 不等于模型官方低阈值完整召回设置的 mAP。

本地已检查配置的 datasets 目录和已知数据存放位置，没有找到配套 COCO 图片与 instances 标注。`accuracy_status.json` 中 metrics=null，**没有伪标签、没有虚构 mAP，也没有用这七张图片冒充验证集**。C++ 数据集导出入口已实际在七张图片上执行，CPU/gpu-filter 与冻结结果的类别、浮点框、分数完全一致，见 `sample_cpu_filter_parity.json`。有标签后的完整 AP 评估入口尚待真实数据执行。

## 3. 实现过程与资源所有权

首先保留旧同步二进制和源码快照，建立新异步类；仅将原 `finishCandidates` 的坐标还原逻辑提取为带显式 geometry 的静态函数，避免 CPU NMS 使用下一帧已覆盖的共享 cachedScale/cachedPad。原同步方法仍调用同一函数、维护原计数。GPU NMS、CUDA 预处理、GPU 筛选、CPU NMS 语义及原 Graph 捕获实现均未改动，逐文件 hash 与冻结版本相同。

每槽位独占 raw image pinned/device buffer、GPU 筛选候选 device/host 包、三个依赖事件、frame ID、原图尺寸、gain/pad。最大原图容量根据预读图片固定为 4,922,883 bytes；每槽位候选 device/host 包各 268,816 bytes。宿主和设备侧分别新增 5,191,699 bytes/槽（不含小型事件对象）；原同步持久化缓冲区继续保留。具体指针和地址见各运行 `resources.csv`。

计算流和 TensorRT Context 仍只有原来的一套。Graph 捕获原 Engine 的固定输入输出地址，异步没有让 replay 自动切换槽位；采用共享 Engine I/O、串行计算、逐槽候选输出的方案：

```text
upload stream:    slot_i H2D -> uploaded_i
compute stream:  wait(uploaded_i) -> 原CUDA预处理 -> 原TRT Graph
                 -> 筛选读取共享output并写slot_i候选 -> computed_i
                 -> 才允许下一帧的预处理/推理覆盖共享I/O
download stream: wait(computed_i) -> slot_i候选D2H -> downloaded_i
CPU collector:   wait(downloaded_i) -> CPU NMS -> slot_i的gain/pad还原 -> Result
                 -> slot_i Free，通知背压中的提交者
```

槽位状态为 `Free→Submitting→Queued→Collecting→Free`。图像提交返回后调用者可立即释放或改写；非连续 ROI 逐行复制。CPU NMS 不占 mutex，允许提交者继续排队后续 GPU 工作；返回结果有自己的存储，没有槽位指针。槽位复用同时等待 D2H 和 CPU 包读取完成；只等 H2D 或推理完成就复用候选包会造成串帧，因此没有这样做。

单提交者/单收集者可以并发，FIFO 按严格递增 frame ID 收集。满队列阻塞，不丢帧。`closeSubmissions()` 停止接收，`drain()/stopAsync()` 返回全部剩余结果；退出前 join 调用线程并排空。析构等待设备结束但不能替调用者消费结果，未收集结果会警告。热身后 `restartAsyncBatch()` 保持所有地址和分配不变。

同步 `detect()` 保留；异步运行中配置和同步调用明确报错。CPU/gpu-nms 的异步实现不在本轮范围。接口完整示例、错误处理与线程契约见 `doc/ASYNC_PIPELINE.md`。

## 4. 已实际完成的验证

样本：'''+', '.join(shapes)+'''. 所有图片预读，包含长宽比变化、奇数尺寸、多目标。它们重复用于 ring/stress 与性能，不代表真实视频或完整数据集。

- Graph on/off × 同步/1/2/3 槽，共 8 组，正式检查每组 84 帧。
- 异步每组额外 12 次七图槽位循环，检查 frame ID、槽位顺序、尺寸、non-contiguous ROI，并在 submit 返回后立即污染/释放原 ROI。逐帧结果与同步路径类别、分数、浮点框和顺序完全相同。
- 满槽后额外提交线程确实阻塞；collect 释放一个槽位才返回。检查重复/回退 frame ID 被拒绝，退出时排空返回每个剩余结果。
- 非 CV_8UC3、超容量、异步配置修改、同步与异步混用报错；空的已关闭队列 collect 正常返回 nullopt。
- 资源地址循环前后不变；每个测量批次 accepted=collected，ID 连续，无丢帧、重复、串帧。
- 正式测量所有逐帧导出结果也与冻结同步结果按类别和 IoU 匹配；重复且完全相同的输出复用已检查的匹配，所有 frame ID 和输出均检查。最大坐标/分数差异均为 0，无未匹配框。
- 全项目构建成功；现有 C++ GPU NMS 同步入口补充检查也与冻结结果完全一致。源码 diff whitespace 检查通过。

检查原文见 `validate_g*_s*.checks.txt`、对应 `.parity.json`、`additional_checks.json`。这些结果说明本次异步调度保持了已测样本结果，不说明完整 mAP 已验证。

## 5. 无 profiler 正式性能

每配置的轮次、帧数由 summary.json 记录，默认实验为两轮2800帧，七图循环。具体本次样本数如下表，预读与热身参数见 summary.json。顺序偶数轮0/1/2/3，奇数轮3/2/1/0，另有基类初始化10次模型热身。

CPU 使用 steady_clock。同步延迟从内存图像调用 detect 到 CPU 最终结果可用；异步从进入 submit 到 CPU 最终结果可用，**包含取得槽位的背压、计算队列和 FIFO 等待**。吞吐从整批开始到排空且提交线程 join，包含启动和尾部排空，不使用提交速率。读图、输出写盘、绘制和编码均在测量外。原始数据是 `.frames.csv/.detections.csv/.config.csv`，汇总为 summary.json。

'''+ f"本次每配置 {configs[0]['rounds']} 轮，每轮 {frames} 帧，{images} 张图片；每图热身 {warmup} 次，共 {warmup*images} 帧；每配置 {configs[0]['completed_frames']} 个正式测量样本。\n\n"+ '\n'.join(table)+ '\n\n'+ '\n'.join(queue)+ '\n\n'+ '\n'.join(rounds)+f'''

admitted latency 从获得槽位开始，仍包含 GPU/FIFO 排队，仅帮助解释背压，不能替代包含背压的真实逐帧延迟。submit wait 包含锁和等待槽位。原始 CSV 还记录 staging、host enqueue、collect event wait、CPU post 等 CPU 区间；collection_queue 指 enqueue 完毕到 FIFO collect 开始，并非纯 GPU queue time。不能机械相加这些重叠区间。

此前同步最终报告 1.6032 ms/P95 2.1296 ms 保留在旧目录；本轮同步重新实测为 {configs[0]['latency_ms']['mean']:.4f} ms/P95 {configs[0]['latency_ms']['p95']:.4f} ms。实验时间、GPU运行状态和统计方式不同，不把该差异归因于本轮异步优化；对照收益使用本轮同时测量的同步配置。

**解释与取舍：** 1 槽平均 admitted latency 接近同步，但从 submit 进入计时会先等待上一帧释放槽位，所以真实平均延迟更高，吞吐收益有限。2 槽形成有界队列，完成吞吐提升 {((configs[2]['completed_fps']/base)-1)*100:.2f}%，但平均延迟 {configs[2]['latency_ms']['mean']:.4f} ms、P95 {configs[2]['latency_ms']['p95']:.4f} ms。3 槽比2槽吞吐仅高 {(configs[3]['completed_fps']/configs[2]['completed_fps']-1)*100:.2f}%，变化小于轮次波动，平均延迟增加 {configs[3]['latency_ms']['mean']-configs[2]['latency_ms']['mean']:.4f} ms，因此没有理由在当前数据上默认使用3槽。两轮数据没有证明统计显著性或长期持续功耗稳定性。

## 6. 独立 Nsight 时间线与残留同步

每配置另外采集140帧，热身每图5次，仅捕获 MEASURED_BATCH。使用 CUDA/NVTX/osrt、关闭 CPU sampling/context-switch 采样、Graph node tracing；正式性能不使用这些时间线的延迟。生成 `.nsys-rep` 与 SQLite、`.analysis.json` 和前20ms可视化 SVG。

最初工具只采到 API/NVTX，没有 GPU kernel/memcpy。WSL 上临时设置 `CuptiUseRawGpuTimestamps=false` 后获得实际活动数据，采集结束恢复原配置。这是 [NVIDIA 官方论坛提供的 workaround](https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536)，官方明确表示该转换方式精度较低。不能把极小时间交集当成稳定硬件重叠收益。

'''+ '\n'.join(trace_table)+ '\n\n'+ '\n'.join(calls_table)+'''

每个异步时间线包含140 H2D、140 D2H、140 CPU后处理范围，kernel只在一个计算流上。每帧有3个 event record和2个 stream wait建立依赖，并在 collect 保留1个 downloaded event同步；这正是结果可供CPU读取前需要的完成点。稳定范围内没有 stream/device 全局同步，也没有 malloc/pinned allocation/free/event或stream创建销毁。原同步路径每帧一个 StreamSynchronize。

H2D/D2H/kernel和CPU后处理的实测时间交集见上表。本次H2D/D2H与kernel均未观测到重叠；极小的CPU交集远不足以宣称稳定后处理重叠，该值还受到WSL转换精度限制。同步CPU后处理没有新增NVTX标注，表中的“未标注”不表示CPU不存在。

2/3槽的kernel忙碌时间占批次比例约79%，同步约65%；这与减少主机提交间隙、提高设备工作连续性的解释相符，**属于推断，不能单凭时间线把吞吐收益精确归因于某阶段，也不能称为copy/compute重叠加速**。需在更精确的GPU时间戳环境与更长负载下继续验证原因。

可在 Nsight UI 打开原始报告检查上传/计算/回传三个流及 CPU submit/collect 范围；SQLite 解析不对缺失GPU活动返回虚假的零值，而是明确 blocked。

## 7. 复现命令与文件

```bash
conda activate yolos-trt
cmake --build build -j4
python scripts/async_pipeline/reproduce_sync.py \\
  --output-prefix results/async_pipeline/reproduce_new/sync --checks
python scripts/async_pipeline/run.py \\
  --output results/async_pipeline/rerun --frames 2800 --warmup 30 --rounds 2 \\
  --nsight --wsl-cupti-workaround
python scripts/async_pipeline/report.py \\
  --results results/async_pipeline/rerun --output 多帧异步流水线复测.md
```

输出目录必须为空，保留原始测量。更换 Engine、PT、ONNX 会触发hash检查；每次运行结束重新验证源码和二进制一致。程序可单独通过 `--slots 0|1|2|3 --graph 0|1 --validate` 运行，示例见接口文档。数据集复现需要真实 COCO 验证图片、匹配 instances JSON 和 pycocotools。

| 文件 | 用途 |
|---|---|
| baseline/version.json、source-sha256.json、sync_baseline.binary、source.tar.gz | 原同步版本与本地可执行复现 |
| final/environment.json、completion.json、source/ | 环境、工作区状态、代码/二进制/模型hash与测量源快照 |
| final/commands.json | 每条实际复现命令、返回码和日志 |
| final/accuracy_status.json、sample_cpu_filter_parity.json | 数据集阻塞与实际小样本CPU/filter一致性 |
| final/validate_g*_s*.checks.txt、parity.json | 生命周期、Graph开关、逐帧结果验证 |
| final/final_r*_s*.frames.csv、detections.csv、config.csv | 无profiler两轮原始结果、逐帧延迟、整批完成时间 |
| final/summary.json | 平均/P50/P95、完成吞吐及测量边界 |
| final/timeline_s*.nsys-rep、sqlite、analysis.json、analysis.svg | 独立时间线、原始数据库与分析；大型二进制报告不提交Git，保留本地 |
| include/yolos/tasks/async_detection.hpp | 有界异步API与生命周期 |
| src/async_pipeline_benchmark.cpp | 可运行的验证/对照基准 |
| scripts/async_pipeline/dataset_accuracy.py | 有真实标签后的数据集AP与CPU/filter对照 |

## 8. 已验证、静态结论、尚未验证

**已实际验证：** 1/2/3槽、Graph开关、逐帧精确一致性、混合尺寸/ROI/提交后图像污染、背压、顺序、循环地址稳定、退出排空、完整批次吞吐、包含排队的延迟、独立Nsight实际GPU活动和残留同步、当前同步三路径。

**静态检查并由时间线支持：** 单Context、共享Graph I/O固定地址、计算串行、候选包独占；未改变GPU NMS/Graph捕获范围；末次使用者完成后复用；稳定执行阶段没有CUDA/pinned资源分配。接口要求SPSC和配置/析构独占，不支持任意数量调用线程。

**尚未验证：** 完整数据集mAP、真实长期视频/摄像头处理与编解码集成、低功耗/散热稳定性、更大候选数场景、其它Engine/GPU、原生Linux精确时间戳下copy/compute重叠。当前任务交付是检测运行时和预读图像的流水线基准，未把旧视频/摄像头入口自动替换为异步。

## 9. 下一步优先级

1. 使用真实标签补齐同一Engine的CPU/gpu-filter AP对照；引入新精度变更前先关闭这一证据缺口。
2. 吞吐部署采用2槽、严格背压，接入真实视频并测从获取帧到结果的总延迟和队列行为；单帧低延迟保留同步路径。3槽当前没有明确实测优势。
3. 在更准确的Nsight环境核查拷贝与kernel为何未重叠，区分驱动/调度/硬件与主机提交间隙；本轮没有依据预设新的加速倍数。
4. 先改善已观察到的瓶颈和连续工作能力，再考虑多Context、完整Graph、GPU NMS变化或INT8；本轮未同时引入这些变量。
'''
    a.output.write_text(text)
    print(a.output)


if __name__=='__main__':main()
