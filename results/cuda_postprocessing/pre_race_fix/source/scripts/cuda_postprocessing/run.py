#!/usr/bin/env python3
"""Validate exact CPU/GPU postprocessing parity, then measure without profiling events.
Run from the repository root using the existing Engine; no model is rebuilt.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/baseline'))
from reference import read_detections, match

MODES = ('cpu', 'gpu-filter', 'gpu-nms')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images', type=Path, default=Path('results/baseline/samples'))
    p.add_argument('--engine', type=Path, default=Path('models/yolo11n.trt'))
    p.add_argument('--onnx', type=Path, default=Path('models/yolo11n.onnx'))
    p.add_argument('--pt', type=Path, default=Path('models/yolo11n.pt'))
    p.add_argument('--original-prefix', type=Path, help='Optional saved original detections/tensors')
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--runs', type=int, default=200)
    p.add_argument('--rounds', type=int, default=2)
    a = p.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from repository root')
    if a.output.exists() and any(a.output.iterdir()):
        raise RuntimeError('Evidence directory must be empty; keep previous measurements')
    if a.warmup < 0 or a.runs < 1 or a.rounds < 1:
        raise ValueError('Invalid measurement counts')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / '.gitignore').write_text('*.f32\n*.npy\n*.binary\n')
    commands = []

    def run(argv, name):
        argv = [str(x) for x in argv]
        print(name, flush=True)
        with (a.output / (name + '.log')).open('w') as log:
            log.write('ARGV ' + json.dumps(argv) + '\n')
            log.flush()
            r = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        commands.append({'argv': argv, 'returncode': r.returncode, 'log': name + '.log'})
        (a.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if r.returncode:
            raise RuntimeError(f'{name} failed; inspect its log')

    run([sys.executable, 'scripts/baseline/collect_environment.py', '--output', a.output / 'environment.json'], 'environment')
    tracked = subprocess.check_output(['git', 'diff', '--binary'])
    (a.output / 'tracked.patch').write_bytes(tracked)
    sources = [Path(x) for x in (
        'CMakeLists.txt', 'include/yolos/core/cuda_postprocessing.cu',
        'include/yolos/core/cuda_postprocessing.hpp', 'include/yolos/core/detection_postprocessing.hpp',
        'include/yolos/core/trt_session_base.hpp', 'include/yolos/tasks/detection.hpp',
        'src/baseline_detection.cpp', 'src/image_inference.cpp',
        'tests/detection/cuda_postprocessing_test.cpp', 'scripts/cuda_postprocessing/run.py')]
    manifest = {}
    for source in sources:
        dest = a.output / 'source' / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest[str(source)] = sha(source)
    for binary in ('build/baseline_detection', 'build/cuda_postprocessing_test'):
        manifest[binary] = sha(Path(binary))
    (a.output / 'source-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def detect(name, mode, graph=1, validation=False, conf=.25, max_det=300, profile=False):
        prefix = a.output / name
        argv = ['build/baseline_detection', a.engine, 'models/coco.names', a.images, prefix,
                '--postprocess', mode, '--graph', graph, '--warmup', 5 if validation else a.warmup,
                '--runs', 1 if validation else a.runs, '--conf', conf, '--iou', .45, '--max-det', max_det]
        if validation:
            argv += ['--checks']
            if conf == .25 and max_det == 300:
                argv += ['--dump']
        if profile:
            argv += ['--profile']
        run(argv, name)
        return prefix

    images = sorted(p.name for p in a.images.glob('*.png'))
    if not images:
        raise RuntimeError('Expected PNG validation images')
    comparisons = []

    def compare(actual, reference, tensors=False):
        aa = read_detections(str(actual) + '.detections.csv')
        bb = read_detections(str(reference) + '.detections.csv')
        record = {'actual': str(actual), 'reference': str(reference), 'images': []}
        for image in images:
            x, y = aa.get(image, []), bb.get(image, [])
            m = match(x, y, min_iou=.99999)
            # A same-class IoU match assesses numeric parity; exact ordered equality
            # separately checks the declared score/source-index total order.
            m.update(image=image, exact_order_and_values=x == y)
            if tensors:
                m['tensor_bit_exact'] = {}
                for suffix in ('input.f32', 'output0.f32'):
                    xp = Path(str(actual) + '.' + image + '.' + suffix)
                    yp = Path(str(reference) + '.' + image + '.' + suffix)
                    m['tensor_bit_exact'][suffix] = sha(xp) == sha(yp)
            passed = (not m['unmatched_actual'] and not m['unmatched_reference']
                      and m['max_coordinate_abs_delta'] == 0 and m['max_score_abs_delta'] == 0
                      and m['exact_order_and_values'] and all(m.get('tensor_bit_exact', {}).values()))
            m['pass'] = passed
            record['images'].append(m)
        comparisons.append(record)
        (a.output / 'parity.json').write_text(json.dumps(comparisons, indent=2) + '\n')
        if not all(m['pass'] for m in record['images']):
            raise RuntimeError('Parity failed; inspect preprocessing -> raw output -> postprocessing, do not relax thresholds')

    cpu = None
    for graph in (1, 0):
        for mode in MODES:
            prefix = detect(f'validation_g{graph}_{mode}', mode, graph, validation=True)
            if cpu is None:
                cpu = prefix
                if a.original_prefix:
                    compare(cpu, a.original_prefix, tensors=True)
            else:
                compare(prefix, cpu, tensors=True)
    run(['build/cuda_postprocessing_test', a.output / 'kernel_tests.csv', a.output], 'kernel_tests_with_raw_replay')
    run([sys.executable, 'scripts/baseline/reference.py', '--onnx', a.onnx, '--pt', a.pt,
         '--images', a.images, '--cpp-prefix', cpu, '--output', a.output / 'python_reference',
         '--require-structural-pass'], 'python_reference')
    for conf, limit in ((.01, 300), (.8, 300), (.99, 300), (.25, 1)):
        reference = None
        for mode in MODES:
            prefix = detect(f'threshold_{conf}_limit_{limit}_{mode}', mode, validation=True, conf=conf, max_det=limit)
            if reference is None:
                reference = prefix
            else:
                compare(prefix, reference)

    # No tensor dumps/reference workloads/profile events during final E2E runs.
    prefixes = []
    for round_id in range(a.rounds):
        order = MODES if round_id % 2 == 0 else tuple(reversed(MODES))
        for graph in (1, 0):
            for mode in order:
                prefixes.append(detect(f'e2e_r{round_id}_g{graph}_{mode}', mode, graph))
    for graph in (1, 0):
        for mode in MODES:
            prefixes.append(detect(f'profile_g{graph}_{mode}', mode, graph, profile=True))
    run([sys.executable, 'scripts/baseline/summarize.py', *prefixes, '--output', a.output / 'summary.json'], 'summary')
    for source in sources:
        if sha(source) != manifest[str(source)]:
            raise RuntimeError(f'Source changed during measurement: {source}')
    for binary in ('build/baseline_detection', 'build/cuda_postprocessing_test'):
        if sha(Path(binary)) != manifest[binary]:
            raise RuntimeError(f'Executable changed during measurement: {binary}')
    summary = json.loads((a.output / 'summary.json').read_text())
    pooled = {}
    for graph in (1, 0):
        for mode in MODES:
            rows = []
            for round_id in range(a.rounds):
                path = a.output / f'e2e_r{round_id}_g{graph}_{mode}.timings.csv'
                with path.open() as f:
                    rows.extend(csv.DictReader(f))
            values = np.array([float(r['e2e_ms']) for r in rows])
            pooled[f'g{graph}_{mode}'] = {'samples': len(values), 'mean_ms': float(values.mean()),
                'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95)),
                'candidates_per_image': {image: sorted({int(r['candidate_count']) for r in rows if r['image'] == image}) for image in images},
                'd2h_bytes': sorted({int(r['d2h_bytes']) for r in rows})}
    (a.output / 'pooled.json').write_text(json.dumps(pooled, indent=2) + '\n')
    print(json.dumps(pooled, indent=2), flush=True)
    print('PASS correctness, Python reference, raw replay and separate E2E/profiling runs', flush=True)


if __name__ == '__main__':
    main()
