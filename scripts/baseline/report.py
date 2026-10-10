#!/usr/bin/env python3
"""Create a Chinese evidence report from saved measurements; no inferred speedups."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('results/baseline'))
    p.add_argument('--output',type=Path);a=p.parse_args();root=a.root
    def load(s):return json.loads((root/s).read_text())
    env=load('final_corrected/execution_provenance.json')
    ref=load('final_corrected/reference/comparison.json');old=load('final_original/reference/comparison.json')
    summaries={label:load(label+'/summary.json') for label in ('final_original','final_corrected')}
    c=ref['images'];o={i['image']:i for i in old['images']}
    lines=['# YOLO11n 可验证基线实测记录','',
      '本轮已完成可运行的验证入口、机器可读检测结果、独立性能基准和最小正确性修复。实测平台为 RTX 4060 Laptop GPU，沿用同一个既有 Engine。未实现 GPU NMS、多帧流水线或 INT8。后续优化以 `final_corrected` 为基线。','',
      '## 实验身份与环境','',
      f"- 起始 Git commit：`{(root/'original/git-commit.txt').read_text().strip()}`。起始 `git-status.txt` 为空，即工作区干净；交付修改尚未提交。",
      f"- 正式实验日期：{env['timestamp'][:10]}，Asia/Shanghai；每条复现命令及起止时间见两个正式目录的 `commands.json`。",
      '- 设备、驱动、Toolkit 与运行库分别记录，避免把驱动支持的 CUDA 上限当作 Toolkit 版本。','',
      '| 项目 | 实测记录 |','|---|---|',
      '| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，compute capability 8.9，CMake arch=89 |',
      '| 驱动 | 591.86；WSL nvidia-smi 程序版本 590.48.01，显示支持 CUDA 13.1 |',
      '| CUDA Toolkit / C++ cudart | nvcc 12.4.99 / libcudart.so.12 |',
      f"| TensorRT C++ | 编译头和库包 10.7.0.23，实际动态库 getInferLibVersion={env['tensorrt_cpp_runtime']['getInferLibVersion']} |",
      '| OpenCV C++ | 4.10.0（CMake 和实际链接库） |',
      f"| OpenCV Python | {env['python_packages']['cv2']} |",
      f"| Ultralytics | {env['python_packages']['ultralytics']} |",
      f"| PyTorch | {env['python_packages']['torch']}，其自带 CUDA runtime {env['torch_cuda_runtime']} |",
      f"| ONNX / ONNX Runtime | {env['python_packages']['onnx']} / {env['python_packages']['onnxruntime']} |",
      '', '- TensorRT Python 模块缺失；本轮的构建、Engine 查询、执行均使用已有 C++ 库，不构成阻塞。GPU 与桌面共享，未锁定频率/功耗，也未测量交流供电状态或持续热稳定性。','',
      '原始与修正版本的环境记录、实际二进制哈希、ldd 和工作区 diff 均已保存。`environment.json` 是运行前采样；`execution_provenance.json` 补充本轮实际执行二进制和 TensorRT 运行库，原二进制对应保存的 source snapshot 与 instrumentation.patch，而不是修正后的工作区源码。','',
      '## 模型与 Engine','',
      '| 文件 | 字节数 | SHA256 |','|---|---:|---|']
    for m in env['model_artifacts']:lines.append(f"| {m['path']} | {m['bytes']} | `{m['sha256']}` |")
    lines += ['',
      '导出命令为 `python models/export_onnx.py --model yolo11n`，脚本设置 static、imgsz=640、opset=11。现存 ONNX metadata 确认 batch=1、imgsz=[640,640]、dynamic=false、simplify=true、end2end=false。正式 Engine 构建命令由用户提供：`./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16 --workspace 2048`。原构建日志不可用；本轮记录其来源，没有重新导出或重建正式模型，也不声称从 Engine 反推出原 workspace 或所有层都为 FP16。前后模型哈希完全相同。','',
      '| Engine 实际 I/O | dtype | shape | 字节数 |','|---|---|---|---:|',
      '| images | float32 | [1,3,640,640] | 4,915,200 |',
      '| output0 | float32 | [1,84,8400] | 2,822,400 |','',
      '由 C++ TensorRT API 查询并打印，同时保存到每组 `.engine.csv`。FP16 builder flag 与 FP32 I/O 分开记录。','',
      '## 正确性对照','',
      '主参考为 **ONNX Runtime CPUExecutionProvider FP32**，执行现存 ONNX；另用 **PT / PyTorch CPU FP32 fused eval** 交叉核对。统一 640×640、batch=1、center=true、auto=false、scale_fill=false、scaleup=true、padding=114、BGR→RGB、NCHW /255、conf=0.25、IoU=0.45、class-aware NMS、multi_label=false、max_det=300、max_nms=30000。六个旧 COCO 名称别名显式映射，类别 ID 顺序经 PT/ONNX metadata 核对一致。','',
      '7 图来自本地 dog 场景及已安装 Ultralytics 的 bus/zidane，包含奇数长宽、竖图、宽图和四目标拼图。来源、尺寸、SHA256 见 `samples/manifest.json`。同类别且 IoU≥0.9 才建立匹配边，使用最大匹配数的二分图匹配，不按数组下标逐项比较。全部逐框坐标/分数差、未匹配结果见参考 JSON。','',
      '| 图片 | 修正前最大预处理差 | 修正后最大预处理差 | 原版最大坐标差 px | 修正版最大坐标差 px | 修正版最大分数差 | 匹配数 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for im in c:
        before=o[im['image']];m=im['final_cpp_vs_onnx'];bm=before['final_cpp_vs_onnx']
        lines.append(f"| {im['image']} | {before['preprocessing']['max_abs']:.8f} | {im['preprocessing']['max_abs']:.8f} | {bm['max_coordinate_abs_delta']:.6f} | {m['max_coordinate_abs_delta']:.6f} | {m['max_score_abs_delta']:.8f} | {len(m['matched'])} |")
    matches=sum(len(im['final_cpp_vs_onnx']['matched']) for im in c)
    unactual=sum(len(im['final_cpp_vs_onnx']['unmatched_actual']) for im in c);unref=sum(len(im['final_cpp_vs_onnx']['unmatched_reference']) for im in c)
    post=max(im['postprocess_cpp_vs_python_on_identical_trt_output']['max_coordinate_abs_delta'] for im in c)
    pt=max(im['pt_cpu_fp32_vs_onnx_cpu_fp32']['max_coordinate_abs_delta'] for im in c)
    lines += ['',f'修正后共匹配 {matches} 个检测框，C++ 未匹配 {unactual}、参考未匹配 {unref}。相同 TRT 原始输出上的后处理检查 7/7 通过，最大坐标差 {post:.9f} px（包含 CSV 文本精度误差），分数仅有文本精度误差。PT vs ONNX FP32 的匹配框最大坐标差 {pt:.9f} px。小样本一致性不等同于完整 mAP 验证，也不能排除临近 conf/IoU 边界的结果变化。','',
      '定位证据按以下链路保存：','',
      '1. 原 CUDA 使用 x*srcW/newW 插值，没有 OpenCV 的半像素位移；未做 uint8 量化。原版最大归一化像素差达 0.40000007。修正后使用半像素坐标、边界重复、uint8 量化，最大差约 1/255。padding、RGB 次序和归一化均经整张张量比较检查。',
      '2. 独立 C++ OpenCV 4.10 的 CPU 张量与 Python OpenCV 5.0 的 CPU 张量在本图片集上完全相同。CUDA 的剩余差异因此不是这两个 OpenCV 版本导致，而是 CUDA 浮点插值/舍入与 OpenCV 整数插值实现不同；仍保留该限制，不声称预处理 bit exact。',
      '3. ONNX 同后端分别执行 CPU input / CUDA input，保存预处理引起的 raw boxes/scores 差异；相同 CUDA input 的 TRT vs ONNX raw output 保存 Engine 数值差异。另保存同一个 TRT Engine 在 C++ CPU input 上的输出作诊断。FP16 与 FP32 的差异和低分候选的原始输出差异不混同于后处理错误。',
      '4. 原版先取整、再裁剪、再 NMS；相同原始输出对照失败。修正版先在 float 模型坐标上进行 class-aware NMS，之后再逆变换和裁剪；相同 raw output 的 Python/C++ 后处理对照通过。',
      '',
      '未改变置信度或 NMS 阈值让检查通过。预处理的一个 uint8 code value 上限、相同输出后处理的 <0.0005 px / <1e-7 分数上限写在脚本中；跨 FP16/FP32 的坐标/分数差直接报告，没有设置宽松数值阈值声称精度完全等同。','',
      '## 最小修复与验证范围','',
      '| 修改 | 原因与证据 |','|---|---|',
      '| CUDA half-pixel 插值、uint8 量化、统一 double 几何取整 | 消除原有空间位移，host/kernel 与 Python letterbox 几何一致；保留 ≤1 灰度级剩余差异 |',
      '| 标准检测 float xyxy NMS，延后裁剪，max_det | 原整数框/提前裁剪会改变 IoU；同 raw output 对照与边界定向检查通过 |',
      '| cv::Mat 非连续 ROI 逐行 staging | 原 memcpy 会把行间距当作像素；7 图 ROI 检测与连续输入完全相同 |',
      '| 输入类型、Engine dtype/布局/shape 拒绝 | 防止错误布局或把 FP16 buffer 当 float 使用；实际 half I/O / batch2 Engine 报错 |',
      '| CSV 导出、Graph 开关、可选分阶段计时 | 输出浮点结果，避免整数毫秒；事件只在 profiling 模式启用 |','',
      '**已实际验证**：整项目 C++ 构建；7 图 ONNX/PT/TRT 对照；Graph 开关 input/raw output bit exact；连续/非连续 ROI；空图/gray/BGRA/float Mat 拒绝；真实 half I/O 和 batch2 Engine 拒绝；NMS clipping/float/class-aware/max_det/conf 边界；`image_inference --no-display --csv` 与基准入口结果一致，max_det=2 生效。','',
      '**静态检查确认**：device LINEAR / 非向量化 I/O 限制、动态与非正形状拒绝、非 FP32 I/O 拒绝、单输入限制、enqueue 失败检查。动态/其他 dtype/其他布局没有逐一构建 Engine。','',
      '**尚未验证**：完整 COCO mAP、更多独立场景、视频连续帧与持续热稳定性、其他 YOLO 版本或其他任务的运行正确性、FP16 I/O 转换、GPU NMS、多帧流水线、INT8。没有为这些项目生成数据或推测加速倍数。','',
      '## 性能实测','',
      '端到端：内存 cv::Mat → 最终 CPU 检测结果；steady_clock 小数毫秒；不包含读图、绘制、显示或写盘。7 图各热身 30 次、测量 200 次，每组 1400 样本，另有构造时 10 次 TRT zero-input warmup。round-robin 全集执行。mean/P50/P95 使用原始逐样本 CSV，分位数为 NumPy linear interpolation。','',
      '| 版本与模式（无 CUDA Events） | mean ms | P50 ms | P95 ms |','|---|---:|---:|---:|']
    for label,title in [('final_original','原版'),('final_corrected','修正基线')]:
        for r in summaries[label]['runs'][:2]:
            e=r['aggregate']['e2e_ms'];g=r['configuration']['graph_actual']
            lines.append(f"| {title} Graph {'on' if g=='1' else 'off'} | {e['mean_ms']:.6f} | {e['p50_ms']:.6f} | {e['p95_ms']:.6f} |")
    lines += ['',
      '以下阶段计时是**另外运行**的 profiling 组。GPU 用同一 stream 上的 Events；在已有 stream synchronize 后读取，未新增逐阶段同步。CPU staging / CPU 后处理用 steady_clock。事件区间可能包括提交间隙，不能机械相加得到端到端耗时。','',
      '| 修正基线阶段 | Graph on mean / P50 / P95 ms | Graph off mean / P50 / P95 ms |','|---|---:|---:|']
    runs=summaries['final_corrected']['runs'];names={'staging_ms':'CPU staging copy','h2d_ms':'H2D','preprocess_ms':'GPU 预处理','inference_ms':'TensorRT 推理','d2h_ms':'D2H','postprocess_ms':'CPU 后处理'}
    for key,title in names.items():
        values=[]
        for r in runs[2:]:
            st=r['aggregate'][key];values.append(' / '.join(f'{st[k]:.6f}' for k in ('mean_ms','p50_ms','p95_ms')))
        lines.append(f'| {title} | {values[0]} | {values[1]} |')
    pe=[r['aggregate']['e2e_ms']['mean_ms'] for r in runs[2:]]
    lines += ['',f'profiling 组自身端到端均值：on={pe[0]:.6f} ms、off={pe[1]:.6f} ms，仅用于观察测量影响。正式无事件组均值 on={runs[0]["aggregate"]["e2e_ms"]["mean_ms"]:.6f} ms、off={runs[1]["aggregate"]["e2e_ms"]["mean_ms"]:.6f} ms。图片级 mean/P50/P95 见 summary.json；不是持续视频 FPS。前后版本分时运行，没有锁频、重复多轮或控制桌面干扰，不把前后微小差异归因于单项修复。','',
      '## 下一步优先级','',
      '1. 保持 Graph 开启作为后续正确基线。推理仍是最大阶段，先用更长稳定运行、层级 profiling 检查约 0.942 ms 的实际构成与抖动；Graph 已存在且实测有效，不重复实现。',
      '2. 优先定位 CPU 后处理约 0.485 ms 的全量分数扫描和输出布局访问。输出为 84×8400、D2H 为 2.82 MB / 约 0.269 ms；候选筛选/减少下载数据、CPU 遍历布局优化值得先评估。不能从“后处理耗时”直接判断 NMS 本身是瓶颈。',
      '3. staging 约 0.207 ms、H2D 约 0.214 ms，且随源尺寸变化。先检查图片级 P95 与真实视频分辨率，再决定重叠上传/推理的多帧流水线；本轮串行数据不足以承诺其加速倍数。',
      '4. GPU 预处理约 0.062 ms，当前优先级较低。增加真实独立验证图片并确认插值/阈值边界，之后再考虑进一步融合或精度路径；INT8 和 GPU NMS 尚未实施。','',
      '## 证据与复现入口','',
      '- [复现说明](../../doc/BASELINE.md)：构建、验证、基准、原版复测命令及支持边界。',
      '- [图片清单](samples/manifest.json)：来源、形状、SHA256。',
      '- [修正版对照](final_corrected/reference/comparison.json)、[原版对照](final_original/reference/comparison.json)：逐框匹配和三阶段定位。',
      '- [修正版性能](final_corrected/summary.json)、[原版性能](final_original/summary.json)：总体及图片级统计；各目录 `.timings.csv` 是原始样本。',
      '- [Graph 对照](final_corrected/graph_equivalence.json)、[Engine 拒绝测试](final_corrected/engine_guards/guards.json)、[图像入口检查](final_corrected/image_entry_checks.json)。',
      '- 原始图像入口二进制、原实现测量二进制、源码 snapshot / instrumentation.patch 保存于 `original/`；张量 `.f32` 和 Python `.npy` 全部留在本地。',
      '- 收尾时工作区出现来源无法确认的额外 CPU 扫描优化，用户确认不是其修改。本轮已恢复为 `final_corrected` 对应源码；额外版本的 source/binary/patch 和独立复测保存在 `current_snapshot/`、`final_current/`，不纳入本轮正式基线。',
      '- `original/` 和 `corrected/` 的首轮 5 图数据仍保留，仅为试运行。5 图修正版测量期间与编译相邻/部分重叠，未作为正式性能结论。正式对照仅用 `final_original/`、`final_corrected/` 的 7 图、200 次测量数据。',
      '',
      '不存在 GPU/运行依赖阻塞。剩余精度边界、场景覆盖和持续性能验证已列为尚未验证，不将它们包装成已完成优化。','']
    (a.output or root/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
