#!/usr/bin/env python3
"""Render a local report from saved measurements, never from estimated latency."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--original-prefix', type=Path)
    a = p.parse_args()
    root = a.directory
    pooled = json.loads((root / 'pooled.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    environment = json.loads((root / 'environment.json').read_text())
    parity = json.loads((root / 'parity.json').read_text())
    reference = json.loads((root / 'python_reference/comparison.json').read_text())
    tests = list(csv.DictReader((root / 'kernel_tests.csv').open()))
    if (not reference['structural_pass'] or any(t['status'] != 'PASS' for t in tests)
            or any(not item['pass'] for r in parity for item in r['images'])):
        raise RuntimeError('Correctness evidence is incomplete or failed')
    runs = {Path(r['prefix']).name:r for r in summary['runs']}
    cfg = runs['e2e_r0_g1_cpu']['configuration']
    round_ids = sorted({int(name.split('_')[1][1:]) for name in runs if name.startswith('e2e_r')})
    def link(path):
        import os
        return os.path.relpath(path, a.output.parent)
    lines = ['# CUDA 后处理本机实测报告', '',
        '结论：当前小候选场景优先使用 `gpu-filter`（GPU 筛选/压缩 + CPU NMS）。GPU NMS 实现并验证通过，保留为对照模式。默认 CPU 模式不自动改变。', '',
        f"采集时间：{environment['timestamp']}；起始 commit：`{environment['commands']['git_commit']['stdout'].strip()}`。当前修改及源码/executable SHA256 见 [环境](" + link(root/'environment.json') + ') 与 [源文件清单](' + link(root/'source-sha256.json') + ')；完整新文件快照位于 `final/source/`。', '',
        '## 环境与模型', '',
        'RTX 4060 Laptop GPU，驱动 591.86，CUDA Toolkit 12.4.99，TensorRT C++ 10.7.0.23，OpenCV C++ 4.10.0。WSL/Linux 本机运行；未固定功耗/时钟，尾延迟受系统调度和设备状态影响。', '',
        'Python 包版本（实际采集）：`' + json.dumps(environment['python_packages'], ensure_ascii=False) + '`。', '',
        '复用 YOLO11n Engine；FP16 构建，实际 FP32 输入 `[1,3,640,640]`（4,915,200 bytes）、FP32 输出 `[1,84,8400]`（2,822,400 bytes）。导出命令 `python models/export_onnx.py --model yolo11n`；Engine 构建命令 `./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16 --workspace 2048` 为先前用户执行信息，本轮不重建。', '',
        '| 模型文件 | SHA256 |', '|---|---|']
    lines += [f"| {x['path']} | `{x['sha256']}` |" for x in environment['model_artifacts']]
    lines += ['', '## 正确性', '',
        f"- {len(parity)} 组对照、{sum(len(r['images']) for r in parity)} 个图像比较全部通过。按类别/IoU 匹配，未匹配结果为零，同一 Engine 的坐标/分数差异为零，并额外验证确定排序。Graph on/off、三模式的 input/raw-output SHA256 一致；新 CPU 模式与修改前二进制也一致。",
        f"- {len(tests)} 条 CUDA 测试结果通过（同一配置复用缓冲区 3 次）；含合成语义边界和真实张量重放，不代表 {len(tests)} 张独立图片。实际图片 {cfg['images']} 张，含不同宽高比、奇数尺寸与多目标；阈值 .01/.25/.8/.99、max_det=1/300。",
        '- 非连续 ROI 与连续图像相等；空/灰度/BGRA/float 输入拒绝。所有模式的 checks.txt 保存原始检查结果。',
        '- Python 主参考：ONNX Runtime CPU FP32；附加参考：PT/PyTorch CPU FP32 fused eval。对同一 TRT 原始输出，Ultralytics 后处理检查全通过。FP16 TRT 与 FP32 ONNX/PT 的数值差异记录在 Python 报告中，没有假定跨后端逐位一致。',
        '- 未引入 Top-K：容量 N=8400，超过支持 N=30000 的 Engine 显式拒绝；max_det 在精确 class-aware NMS 后生效。同分取原始预测索引较小者，类别同分取 ID 较小者。',
        '', '[逐框匹配](' + link(root/'parity.json') + ') · [CUDA 测试](' + link(root/'kernel_tests.csv') + ') · [Python 差异分解](' + link(root/'python_reference/comparison.json') + ')', '',
        '## 无事件端到端性能', '',
        f"范围：内存 cv::Mat 到 CPU 最终检测结果，不含读图/绘制/显示/写盘。每图 warmup={cfg['warmup_per_image']}；每轮每图测量={cfg['runs_per_image']}；{cfg['images']} 图 round-robin，{len(round_ids)} 轮交替模式顺序。每种模式/Graph 设置共 {pooled['g1_cpu']['samples']} 次；P50/P95 为线性插值分位数。", '',
        '| Graph | 后处理 | 平均 ms | P50 ms | P95 ms | 实际 D2H bytes/帧 |', '|---|---|---:|---:|---:|---:|']
    for graph in (1, 0):
        for mode in ('cpu', 'gpu-filter', 'gpu-nms'):
            r = pooled[f'g{graph}_{mode}']
            lines.append(f"| {'on' if graph else 'off'} | {mode} | {r['mean_ms']:.4f} | {r['p50_ms']:.4f} | {r['p95_ms']:.4f} | {r['d2h_bytes'][0]:,} |")
    cpu, mixed, gpu = (pooled['g1_'+x]['mean_ms'] for x in ('cpu','gpu-filter','gpu-nms'))
    lines += ['', f'Graph on：混合路径比本轮 CPU 均值减少 {(cpu-mixed)/cpu*100:.2f}%，从 {cpu:.4f} 到 {mixed:.4f} ms；GPU NMS 比混合路径增加 {gpu-mixed:.4f} ms（{(gpu/mixed-1)*100:.2f}%）。这是该 Engine/样本/设备下的观测，不推断普适加速倍数。', '',
        '固定包的 D2H 相比完整输出减少 90.48%（gpu-filter）和 99.66%（gpu-nms，max_det=300）。这些 bytes **包含无效尾槽位**，不宣称只传有效框的精确字节数；没有为读取候选数额外同步。', '',
        '| 图片 | 过阈值候选数（conf=.25） | 最终框数 |', '|---|---:|---:|']
    rows = list(csv.DictReader((root/'e2e_r0_g1_cpu.timings.csv').open()))
    for image, counts in pooled['g1_cpu']['candidates_per_image'].items():
        result = sorted({int(r['result_count']) for r in rows if r['image']==image})
        lines.append(f"| {image} | {counts[0]} | {result[0]} |")
    lines += ['', '各轮单独均值（避免只呈现 pooled 结果）：', '',
              '| Graph | 模式 | ' + ' | '.join(f'round {n} ms' for n in round_ids) + ' |',
              '|---|---|' + '---:|' * len(round_ids)]
    for graph in (1,0):
        for mode in ('cpu','gpu-filter','gpu-nms'):
            x = [runs[f'e2e_r{n}_g{graph}_{mode}']['aggregate']['e2e_ms']['mean_ms'] for n in round_ids]
            lines.append(f"| {graph} | {mode} | " + ' | '.join(f'{v:.4f}' for v in x) + ' |')
    if a.original_prefix:
        with Path(str(a.original_prefix)+'.timings.csv').open() as f:
            values = np.array([float(r['e2e_ms']) for r in csv.DictReader(f)])
        lines += ['', f'修改前原始二进制数据保留：{len(values)} 次，均值 {values.mean():.4f} ms，P50 {np.percentile(values,50):.4f} ms，P95 {np.percentile(values,95):.4f} ms。该次运行时间不同、仅一轮；优化对比以本轮两轮同一新二进制的 CPU/GPU 模式为准，不混用历史测试。']
    lines += ['', '## 独立 profiling', '',
        '下表为 Graph on、独立 profiling run 的阶段均值，每模式 1400 次。GPU 用同流 Events，在既有一次完成同步后读取；CPU 用 steady_clock。NMS GPU 时间含 key生成/CUB排序/抑制mask/greedy选择。事件本身可影响小模型，**不用于替代上表 E2E，不相加估算 E2E**。', '',
        '| 模式 | staging | H2D | preprocess | TRT | GPU filter | GPU NMS | D2H | CPU post | 额外 device bytes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ('cpu','gpu-filter','gpu-nms'):
        r = runs['profile_g1_'+mode]
        cols = [r['aggregate'][x]['mean_ms'] for x in ('staging_ms','h2d_ms','preprocess_ms','inference_ms','filter_ms','nms_ms','d2h_ms','postprocess_ms')]
        lines.append('| '+mode+' | '+' | '.join(f'{v:.4f}' for v in cols)+f" | {int(r['configuration']['extra_device_bytes']):,} |")
    lines += ['', 'CPU 模式 filter/NMS、混合模式 GPU NMS 是空 hook，记录值是事件间隔开销。CUDA 内存数字为新增持久化 buffer/scratch，不含 Engine/原运行时分配，也不是系统峰值显存。', '',
        '[汇总与逐图数据](' + link(root/'summary.json') + ') · [两轮合并](' + link(root/'pooled.json') + ') · [运行命令](' + link(root/'commands.json') + ')。原始 `*.timings.csv` 均在同目录，包含小数毫秒、candidate/result count 与实际 D2H 字节数。', '',
        '## 后续优先级与验证边界', '',
        '1. 以 `gpu-filter` + CPU NMS 作为当前 YOLO11n 部署性能基线；该样本候选仅 10～47 个，CPU NMS 成本很小，先保持明确的模式开关。',
        '2. 根据 TRT、预处理、H2D 的实测占比，再评估减少提交开销/扩展 Graph 捕获或更广的输入场景；这轮未实施，不预设收益。',
        '3. GPU NMS 后续应针对更多候选的真实场景单独测量排序与 bitmask 开销；没有证据证明当前少候选场景值得全部迁移。若引入 Top-K，须重新验证精度，不能沿用本次结果。', '',
        '已实际验证：本机 YOLO11n 的三路径、Graph 开关、七图匹配、ONNX/PT 参考、边界及实际原始输出重放。静态检查：容量与 flags、固定包布局、同步位置、未支持形状拒绝。尚未验证：完整 mAP、其他 Engine/设备、长视频稳定性。七图一致性不是完整 mAP 结论。', '',
        '独立 sanitizer：修正后的 synthetic test 在 memcheck 下 0 errors；selectGreedy 的合成测试 racecheck 为 0 errors/0 warnings；suppressionMasks 的七图 max_det=1 测试 racecheck 为 0 errors/0 warnings；两个 kernel 的合成测试 synccheck 为 0 errors。检查限定这些路径，不覆盖所有设备/输入，也不替代长视频验证。日志保留于本目录上一级。', '',
        '原始版本保留于 `original/`；`verified/` 保留首次 max_det=1 测试入口配置冲突日志，未进入性能测量；`pre_race_fix/` 保留同步风险修复前的数据，最终基线使用 `final/`。静态审查发现 selectGreedy 共享计数作为跨 warp 循环条件可能竞态，已改为仅线程 0 读取/更新计数、同步后统一退出。该风险没有被记录为修复前实测 racecheck 故障；修复后的 sanitizer 与全套复测通过。另将验证入口合成用例的 max_det 与 CLI 独立，明确排序、有限值/容量保护；沿用既有预处理与后 NMS 裁剪。', '',
        '实现与复现详见 [CUDA_POSTPROCESSING.md](' + link(Path('doc/CUDA_POSTPROCESSING.md')) + ')。']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
