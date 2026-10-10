#!/usr/bin/env python3
"""Create the delivery report only from completed full COCO and saved performance."""
import csv,hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')

def main():
    folder=ROOT/'results/async_pipeline/coco_release';accuracy=load(folder/'metrics.json')
    if accuracy['scope']!='full' or accuracy['images']!=5000 or not accuracy['exact_detection_parity']:
        raise RuntimeError('Require successful full 5000-image validation')
    if len(set(accuracy['prediction_sha256'].values()))!=1:raise RuntimeError('Three prediction exports differ')
    performance=load(ROOT/'results/async_pipeline/final/summary.json')
    previous=load(ROOT/'results/async_pipeline/final/completion.json')
    unchanged=[p for p in previous['source_sha256'] if p.startswith('include/')]+['src/async_pipeline_benchmark.cpp','CMakeLists.txt']
    if any(sha(ROOT/p)!=previous['source_sha256'][p] for p in unchanged):raise RuntimeError('Saved performance no longer describes current runtime')
    if sha(ROOT/'build/async_pipeline_benchmark')!=previous['binary_sha256']:raise RuntimeError('Performance binary changed')
    if any(sha(Path(p))!=digest for p,digest in accuracy['artifacts_sha256'].items()):raise RuntimeError('Accuracy source or executable changed')
    for mode in ('cpu','gpu-filter','async2'):
        with (folder/(mode+'.csv.frames.csv')).open() as f:frames=list(csv.DictReader(f))
        if len(frames)!=5000 or len({r['image_id'] for r in frames})!=5000:raise RuntimeError('Incomplete frame audit')
        if mode=='async2' and any(int(r['slot'])!=i%2 for i,r in enumerate(frames)):raise RuntimeError('Slot ring sequence changed')
    prepare=load(ROOT/'results/async_pipeline/coco_preparation.json')
    annotation=load(Path(prepare['annotation']['path']));cats=sorted(annotation['categories'],key=lambda r:r['id'])
    labels=(ROOT/'models/coco.names').read_text().splitlines()
    write(folder/'category_mapping.json',[{'class_id':i,'cpp_name':labels[i],'coco_category_id':cat['id'],'coco_name':cat['name']} for i,cat in enumerate(cats)])
    attribution=load(ROOT/'results/async_pipeline/attribution_delivery/analysis.json')
    aps=['| 路径 | 图片 | 检测数 | AP@[.50:.95] | AP50 | AP75 |','|---|---:|---:|---:|---:|---:|']
    for mode in ('cpu','gpu-filter','async2'):
        m=accuracy['metrics'][mode];count=sum(1 for _ in (folder/(mode+'.csv')).open())-1
        aps.append(f"| {mode} | 5000 | {count} | {100*m['AP']:.4f}% | {100*m['AP50']:.4f}% | {100*m['AP75']:.4f}% |")
    configs=performance['configurations'];perf=['| 路径 | 完成吞吐 FPS | 平均延迟 ms | P50 ms | P95 ms |','|---|---:|---:|---:|---:|']
    for s in (0,2):
        r=configs[s];l=r['latency_ms'];perf.append(f"| {'同步' if s==0 else '异步2槽'} | {r['completed_fps']:.2f} | {l['mean']:.4f} | {l['p50']:.4f} | {l['p95']:.4f} |")
    sync,dual=attribution['trace']['0'],attribution['trace']['2'];gain=(configs[2]['completed_fps']/configs[0]['completed_fps']-1)*100
    text=f'''# 部署基线交付与 COCO 精度验证

本轮补齐三项交付：**现有数据的延迟拆分与收益证据、完整 COCO val2017 三路径精度对照、可提交的源码/结果/复现命令**。上一轮运行时已提交为 `0f827e37a67b04b4cf3b198b995fd5c43815e9af`；本轮新 commit 以 `git rev-parse HEAD` 获取，源码与结果清单见 `results/async_pipeline/delivery_manifest.json`。

## 固定的模型与语义

RTX 4060 Laptop GPU，驱动591.86，CUDA Toolkit12.4.99，TensorRT C++10.7.0，OpenCV C++4.10.0，Release/sm_89。PyTorch、Ultralytics、系统与动态链接库记录在上一轮 `final/environment.json`，本轮评估新增 pycocotools **{accuracy['pycocotools']}**。

固定原 YOLO11n Engine，SHA256：`{accuracy['same_engine_sha256']}`。原构建命令：`./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16 --workspace 2048`。没有重建 Engine，也没有 INT8。

- FP16 builder 配置，**实际 I/O 均为 FP32**。输入 `[1,3,640,640]`，4,915,200 bytes；输出 `[1,84,8400]`，2,822,400 bytes。
- imgsz=640×640、batch=1；原 CUDA letterbox、双线性插值、114 padding、BGR→RGB、float32/255、NCHW；按同一每帧 gain/pad 还原坐标。
- conf=0.25，NMS IoU=0.45，class-aware、single-label，max_det=300，无 Top-K；最多8400候选，同分按原预测索引排序、类别同分选小ID。
- 原 inference-only CUDA Graph 开启；一个 TensorRT Context、一个串行计算流，Graph共享I/O地址固定，2槽独占原图与候选包，事件建立依赖，末次设备使用和CPU读取完成后复用。
- `cpu` 表示 **TensorRT CUDA推理 + CPU后处理**，不是CPU执行模型。`gpu-filter` 是同步GPU筛选+CPU NMS；`async2` 是相同后处理的有界2槽调度。

## 完整精度验证：已实际执行

下载 [COCO 官方 val2017](https://cocodataset.org/#download) 图片与 `instances_val2017.json`，对ZIP执行CRC检查，逐图记录SHA256与尺寸，核验5000个唯一ID和完整文件集。数据来源、两个ZIP和标注SHA256、5000张图片hash在 `coco_preparation.json`；图片/标注留在本地 `datasets/coco2017` 并被Git忽略。

COCOeval bbox 参数一致：IoU=.50:.05:.95、101 recall points、useCats=1、maxDets=[1,10,100]、完整5000图。检测阶段始终保持 conf=.25、IoU=.45、max_det=300。**以下是部署阈值下的完整验证集 AP，不是官方低置信度完整召回配置的模型宣传 mAP。**

{chr(10).join(aps)}

两次完整执行均通过；最终严格尺寸检查版本保存在 `coco_release`，第一次结果保留在 `coco_full`。三路径CSV的SHA256完全相同：`{next(iter(accuracy['prediction_sha256'].values()))}`。

逐图检查有序image ID、解码width/height与标签精确一致、结果数量；异步槽位序列0/1循环2500次，所有5000帧完整排空。按类别和IoU匹配，两个对照共 **{accuracy['matched_pairs_across_two_comparisons']} 对框**，坐标差/分数差最大值均为0，无未匹配结果。不是只比较AP相等，也不是按数组下标替代匹配。原始float坐标、分数和类别留在CSV，COCO映射在 `category_mapping.json`。

评估读取图像并流式执行，只有两帧槽位；这次数据集运行用于准确性，不从包含读盘的运行时长报告吞吐。复现指标和命令在 `coco_release/metrics.json`、`commands.json`；AP计算与完整匹配报告在 `comparison.json`。没有改变阈值以通过对照。

## 2槽为什么更快：基于已有测量

{chr(10).join(perf)}

上述性能仍是上一轮**七图预读、无profiler、每配置两轮2800帧、每图热身30次**的实测，不是COCO吞吐。runtime头文件、Graph实现、CUDA预处理/后处理、benchmark源码和二进制hash均未改变；原始版本及结果未覆盖。

2槽完成吞吐提升 **{gain:.2f}%**。独立140帧Nsight中，前帧D2H结束到下帧H2D开始的平均间隙从 **{sync['d2h_to_next_h2d_gap_ms']['mean']:.4f}ms** 降至 **{dual['d2h_to_next_h2d_gap_ms']['mean']:.4f}ms**；同步下一帧提前提交0/139次，2槽 **139/139次**。所有kernel/copy/memset活动求并集后的GPU空闲比例从 **{sync['gpu_idle_ms']/sync['batch_ms']*100:.2f}%** 降至 **{dual['gpu_idle_ms']/dual['batch_ms']*100:.2f}%**。

这支持减少host staging/提交造成的跨帧供给间隙：同步路径等前帧完成再准备下一帧，2槽允许另一CPU线程预先准备和入队，串行GPU工作更连续。**没有观测到H2D/D2H与kernel重叠，也没有TRT kernel自身加速证据。** CPU NMS仅微秒级，不能把收益归因于NMS加速。

2槽平均逐帧延迟的互斥CPU墙钟拆分：背压/取锁 **1.0071ms**，staging **0.1811ms**，其它host enqueue **0.0739ms**，FIFO collect等待 **0.9987ms**，完成事件等待 **1.2518ms**，CPU后处理 **0.0018ms**，合计 **3.5144ms**。完成事件等待包含GPU服务和依赖，并非同步API的纯开销；这些区间同一帧可相加，跨帧有重叠，不能加成批次时长。

具体数据、P50/P95与原始跨帧时间戳见 [attribution_delivery/REPORT.md](results/async_pipeline/attribution_delivery/REPORT.md)。原始Nsight报告保留本地；CSV导出关键CPU/GPU时间戳并提交Git。WSL采用已记录的CUPTI时间戳workaround，精度有限，profile和无profiler是不同批次；不能把最终24.15%精确分摊为某阶段贡献比例。

## 计时边界与入口

保留同步低延迟与异步2槽吞吐模式。同步逐帧延迟从内存图像进入detect到返回检测结果；异步从进入submit（包含背压）到collect构建完最终CPU检测结果的时间点，包含计算队列和FIFO等待。整批吞吐从批次开始到所有帧完成收集、排空并join提交线程，包含启动/尾部排空；不统计提交速率。

模型初始化、资源分配、热身、读图、绘制、显示、编码、写结果全部在性能计时外。Graph只捕获原TRT推理；不引入多Context、GPU NMS变更或扩大捕获范围。2槽稳定阶段每帧只等待自己的downloaded事件，没有全流/设备同步或CUDA/pinned资源分配。

```bash
conda activate yolos-trt
python -m pip install -r scripts/async_pipeline/requirements-accuracy.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \\
  -DTENSORRT_DIR="$CONDA_PREFIX/opt/TensorRT-10.7.0.23" \\
  -DOpenCV_DIR="$CONDA_PREFIX/lib/cmake/opencv4"
cmake --build build -j4

# 完整数据准备；不将下载的数据/ZIP提交Git
python scripts/async_pipeline/prepare_coco.py \\
  --manifest results/async_pipeline/coco_preparation.json
# 使用空输出目录，保留原始结果
python scripts/async_pipeline/dataset_accuracy.py \\
  --annotations datasets/coco2017/annotations/instances_val2017.json \\
  --images datasets/coco2017/val2017 --output results/async_pipeline/coco_rerun

# 同步与2槽；预读图片，不含解码/绘制/写盘
./build/async_pipeline_benchmark models/yolo11n.trt models/coco.names \\
  results/baseline/samples results/async_pipeline/manual_sync \\
  --slots 0 --frames 2800 --warmup 30 --graph 1 --validate
./build/async_pipeline_benchmark models/yolo11n.trt models/coco.names \\
  results/baseline/samples results/async_pipeline/manual_async2 \\
  --slots 2 --frames 2800 --warmup 30 --graph 1 --validate
# 全对照和独立时间线；WSL设置会在finally中恢复
python scripts/async_pipeline/run.py --output results/async_pipeline/perf_rerun \\
  --frames 2800 --warmup 30 --rounds 2 --nsight --wsl-cupti-workaround
# 分析已有时间线，选择空输出目录
python scripts/async_pipeline/attribution.py \\
  --results results/async_pipeline/final --output results/async_pipeline/attribution_rerun

# 交付commit及工作区状态
git rev-parse HEAD
git status --short
```

本机成功构建时的CMake cache和链接库版本在 `final/environment.json`；不同安装布局需相应调整两个依赖路径，Engine在其它GPU/TensorRT版本上通常需要重建，不能把本Engine的测量数字直接套用。

## 交付文件与验证边界

- 实现：`include/yolos/tasks/async_detection.hpp`，原同步detect继续保留；可运行基准 `src/async_pipeline_benchmark.cpp`；三路径数据集导出 `src/dataset_accuracy_detection.cpp`。
- 精度原始结果：`coco_release` 和保留的 `coco_full`；5000图CSV、逐帧ID/尺寸/槽位、COCO预测JSON、逐图匹配、12项AP/AR、精度参数与模型/二进制hash。
- 归因证据：`attribution_delivery/analysis.json`、`transitions_s*.csv`，原有 `final` 逐帧性能CSV与时间线分析。
- 版本：新源码、复现文档和结果一起提交；`delivery_manifest.json` 记录base commit、测量二进制、源hash、结果hash、依赖和5000图完整验证状态。大数据集、模型、构建二进制和Nsight二进制报告保留本地，不加入Git。

**已实际验证：** 完整COCO val2017三路径AP和逐框精确一致性；2槽全部5000帧顺序/尺寸/结果及槽位复用；上一轮Graph on/off与1/2/3槽生命周期检查、饱和输入吞吐和排队延迟；当前runtime与旧测量源hash一致。

最终复现脚本的附加检查保存在 `delivery_checks`：8组Graph开关/槽位生命周期验证通过，4组28帧短批次用于检查复现入口和完整COCO指标引用。这些短批次不替代上表的两轮正式性能测量。

**仍未验证：** COCO上的吞吐提升比例、长期真实视频/摄像头与编码总延迟、跨GPU/原生Linux时间线、官方低conf模型mAP或PT/ONNX完整数据集对照。当前结论是这一个Engine的CPU后处理、同步筛选和异步2槽精度等价，2槽在已测七图饱和负载上提高完成吞吐，并增加排队延迟。
'''
    (ROOT/'部署基线交付与COCO精度验证.md').write_text(text)
    sources=[ROOT/'CMakeLists.txt',ROOT/'.gitignore',ROOT/'results/async_pipeline/.gitignore',ROOT/'README.md',ROOT/'部署基线交付与COCO精度验证.md',ROOT/'多帧异步流水线实现与实测.md',ROOT/'doc/ASYNC_PIPELINE.md']
    sources += [p for base in ('include','src','scripts/async_pipeline') for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.cu','.hpp','.cpp','.py','.txt')]
    results=[folder/'metrics.json',folder/'comparison.json',folder/'category_mapping.json',folder/'environment_accuracy.json',folder/'evaluation.txt',folder/'dependency_install.txt',ROOT/'results/async_pipeline/coco_preparation.json',ROOT/'results/async_pipeline/attribution_delivery/analysis.json',ROOT/'results/async_pipeline/delivery_checks/completion.json',ROOT/'results/async_pipeline/delivery_checks/accuracy_status.json']
    results += list(folder.glob('*.csv'))+list(folder.glob('*.coco.json'))+list((ROOT/'results/async_pipeline/attribution_delivery').glob('*.csv'))
    write(ROOT/'results/async_pipeline/delivery_manifest.json',{
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'delivery_commit':'Resolve git rev-parse HEAD after committing this manifest; avoid self-referential commit hash',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources))},
        'results_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(results))},
        'performance_runtime_unchanged':True,'performance_binary_sha256':previous['binary_sha256'],
        'accuracy_binary_sha256':sha(ROOT/'build/dataset_accuracy_detection'),'engine_sha256':accuracy['same_engine_sha256'],
        'full_coco_images':5000,'exact_parity':True,'pycocotools':accuracy['pycocotools'],'input_shape':[1,3,640,640],
        'builder_precision':'FP16 enabled','io_dtype':'float32','conf':.25,'iou':.45,'max_det':300,
        'git_status_before_commit':subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True)})
    print(ROOT/'部署基线交付与COCO精度验证.md')

if __name__=='__main__':main()
