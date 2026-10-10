#!/usr/bin/env python3
"""Reproducible sync/1/2/3-slot validation, drained throughput and separate Nsight runs.
Run inside yolos-trt after building; never rebuild or export the Engine here.
"""
import argparse, collections, csv, hashlib, json, re, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baseline'))
from reference import match,read_detections
from timeline import analyze

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/async_pipeline/baseline'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(count=len(values),mean=float(values.mean()),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)))


def parity(prefix,reference,expected_frames):
    frames=rows(Path(str(prefix)+'.frames.csv'));groups=collections.defaultdict(list)
    for r in rows(Path(str(prefix)+'.detections.csv')):
        groups[int(r['frame_id'])].append({'class_id':int(r['class_id']),'confidence':float(r['confidence']),'xyxy':[float(r[k]) for k in ('x1','y1','x2','y2')]})
    if [int(r['frame_id']) for r in frames]!=list(range(expected_frames)):raise RuntimeError('Missing/duplicate/reordered frame IDs')
    comparisons={};seen=set()
    for r in frames:
        fid=int(r['frame_id']);image=r['image'];detections=groups[fid]
        if len(detections)!=int(r['result_count']):raise RuntimeError('Result count mismatch')
        key=(image,json.dumps(detections,sort_keys=True))
        if key in seen:continue
        seen.add(key);m=match(detections,reference.get(image,[]),.99999)
        if m['unmatched_actual'] or m['unmatched_reference'] or m['max_coordinate_abs_delta']!=0 or m['max_score_abs_delta']!=0:
            write(Path(str(prefix)+'.mismatch.json'),{'frame_id':fid,'comparison':m});raise RuntimeError('Frozen baseline parity failed')
        comparisons[image]={'matched':len(m['matched']),'max_coordinate_abs_delta':m['max_coordinate_abs_delta'],'max_score_abs_delta':m['max_score_abs_delta']}
    record={'frames':len(frames),'unique_outputs_checked':len(seen),'every_frame_checked':True,'matching':'same class and IoU; identical repetitions reuse verified match','all_exact':True,'images':comparisons}
    write(Path(str(prefix)+'.parity.json'),record);return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=2800);p.add_argument('--warmup',type=int,default=30);p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--nsight',action='store_true');p.add_argument('--wsl-cupti-workaround',action='store_true');a=p.parse_args()
    if a.frames<1 or a.warmup<1 or a.rounds<1:raise ValueError('Positive frames/warmup/rounds required')
    if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('Output must be empty; existing measurements are preserved')
    a.output.mkdir(parents=True,exist_ok=True)
    commands=[]
    def run(argv,log):
        with log.open('w') as f:r=subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        commands.append({'argv':list(map(str,argv)),'returncode':r.returncode,'log':str(log)})
        write(a.output/'commands.json',commands)
        if r.returncode:raise RuntimeError(f'Command failed; see {log}')
    frozen=json.loads((BASE/'version.json').read_text())
    for path,digest in frozen['model_artifacts'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Model artifact differs from frozen baseline')
    manifest=json.loads((BASE/'source-sha256.json').read_text())
    # These files define preprocessing, GPU filter/NMS and Graph capture; none changes in this task.
    unchanged=['include/yolos/core/cuda_preprocessing.cu','include/yolos/core/cuda_preprocessing.hpp','include/yolos/core/cuda_postprocessing.cu','include/yolos/core/cuda_postprocessing.hpp','include/yolos/core/trt_session_base.hpp','include/yolos/core/detection_postprocessing.hpp']
    for path in unchanged:
        if sha(ROOT/path)!=manifest[path]:raise RuntimeError('Unexpected core semantic or capture change: '+path)
    source_paths=sorted(set([p for folder in ('include','src','scripts/async_pipeline') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.hpp','.cu','.cpp','.py')]+[ROOT/'CMakeLists.txt']))
    hashes={str(path.relative_to(ROOT)):sha(path) for path in source_paths}
    for path in source_paths:
        dst=a.output/'source'/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes())
    binary=ROOT/'build/async_pipeline_benchmark';binary_hash=sha(binary)
    run([sys.executable,ROOT/'scripts/baseline/collect_environment.py','--output',a.output/'environment.json','--binary',binary],a.output/'environment.log')
    env=json.loads((a.output/'environment.json').read_text());env['source_files_role']='Current async worktree; frozen synchronous snapshot is separate'
    env.update(source_sha256=hashes,unchanged_core_files=unchanged,baseline_commit=frozen['commit']);write(a.output/'environment.json',env)
    write(a.output/'accuracy_status.json',json.loads((BASE/'accuracy_status.json').read_text()))
    reference=read_detections(str(BASE/'reproduction.detections.csv'))
    images=ROOT/'results/baseline/samples'
    # Exercise the new dataset streaming exporter with unlabeled samples, never compute AP.
    manifest_path=a.output/'sample_images.tsv';paths=sorted(images.glob('*.png'))
    manifest_path.write_text(''.join(f'{i}\t{path}\n' for i,path in enumerate(paths)))
    exported={}
    for mode in ('cpu','gpu-filter'):
        out=a.output/('sample_'+mode+'.csv')
        run([ROOT/'build/dataset_accuracy_detection',ROOT/'models/yolo11n.trt',ROOT/'models/coco.names',manifest_path,out,mode],a.output/('sample_'+mode+'.log'))
        exported[mode]=collections.defaultdict(list)
        for row in rows(out):
            exported[mode][int(row['image_id'])].append({'class_id':int(row['class_id']),'confidence':float(row['confidence']),'xyxy':[float(row[k]) for k in ('x1','y1','x2','y2')]})
    checks=[]
    for i,path in enumerate(paths):
        for label,actual in [(mode,exported[mode][i]) for mode in exported]:
            m=match(actual,reference.get(path.name,[]),.99999)
            if m['unmatched_actual'] or m['unmatched_reference'] or m['max_coordinate_abs_delta'] or m['max_score_abs_delta']:raise RuntimeError('Dataset exporter sample parity failed')
            checks.append({'image':path.name,'mode':label,'matched':len(m['matched']),'max_coordinate_abs_delta':0,'max_score_abs_delta':0})
    write(a.output/'sample_cpu_filter_parity.json',{'scope':'Seven unlabeled images; NOT mAP','checks':checks})
    common=[binary,ROOT/'models/yolo11n.trt',ROOT/'models/coco.names',images]
    for graph in (1,0):
        for slots in range(4):
            prefix=a.output/f'validate_g{graph}_s{slots}'
            run(common+[prefix,'--slots',slots,'--frames',84,'--warmup',5,'--graph',graph,'--validate'],prefix.with_suffix('.log'))
            parity(prefix,reference,84)
    # Final measurements are separate processes, no profiler and no background validation/build.
    measurements=[]
    for repeat in range(a.rounds):
        order=range(4) if repeat%2==0 else range(3,-1,-1)
        for slots in order:
            prefix=a.output/f'final_r{repeat}_s{slots}'
            print(f'Final run {repeat+1}/{a.rounds}, slots={slots}',flush=True)
            run(common+[prefix,'--slots',slots,'--frames',a.frames,'--warmup',a.warmup,'--graph',1],prefix.with_suffix('.log'))
            measurements.append((slots,prefix))
    summary=[]
    for slots in range(4):
        batches=[];frames=[]
        for s,prefix in measurements:
            if s!=slots:continue
            batches+=rows(Path(str(prefix)+'.config.csv'));frames+=rows(Path(str(prefix)+'.frames.csv'))
        summary.append({'slots':slots,'completed_frames':len(frames),'rounds':len(batches),'batch_elapsed_ms':sum(float(r['batch_elapsed_ms']) for r in batches),
                        'completed_fps':len(frames)*1000/sum(float(r['batch_elapsed_ms']) for r in batches),
                        'latency_ms':stats([r['latency_ms'] for r in frames]),'admitted_latency_ms':stats([r['admitted_latency_ms'] for r in frames]),
                        'submit_wait_ms':stats([r['submit_wait_ms'] for r in frames]),
                        'collection_queue_ms':None if slots==0 else stats([r['collection_queue_ms'] for r in frames]),
                        'rounds_raw':batches})
    # Check every final frame AFTER all performance runs.
    for _,prefix in measurements:parity(prefix,reference,a.frames)
    write(a.output/'summary.json',{'timestamp':datetime.now().isoformat(),'frames_per_round':a.frames,'warmup_per_image':a.warmup,'images':len(paths),'model_engine_sha256':frozen['model_artifacts']['models/yolo11n.trt'],
                                  'latency_definition':'submit entry (including capacity wait) to CPU final detections ready; synchronous detect entry to return',
                                  'admitted_latency_definition':'slot reservation to CPU final detections ready; GPU/FIFO queue still included',
                                  'throughput_definition':'completed frames / entire batch wall time including final drain and producer join; not submission rate',
                                  'preloaded_images':True,'profiler':False,'graph':True,'postprocess':'gpu-filter + CPU NMS','configurations':summary})
    if a.nsight:
        config_path=Path(subprocess.check_output(['nsys','-z'],text=True).strip());saved=config_path.read_bytes() if config_path.exists() else None
        try:
            if a.wsl_cupti_workaround:
                content=(saved or b'').decode();content=re.sub(r'^\s*CuptiUseRawGpuTimestamps\s*=.*$','',content,flags=re.M)+'\nCuptiUseRawGpuTimestamps=false\n'
                config_path.parent.mkdir(parents=True,exist_ok=True);config_path.write_text(content)
                write(a.output/'nsight_configuration.json',{'temporary_config_path':str(config_path),'override':'CuptiUseRawGpuTimestamps=false','original_existed':saved is not None,'restoration':'finally block','reference':'https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536'})
            for slots in range(4):
                prefix=a.output/f'timeline_s{slots}'
                run(['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--capture-range=nvtx','--nvtx-capture=MEASURED_BATCH','--capture-range-end=stop','--output='+str(prefix)]+common+[prefix,'--slots',slots,'--frames',140,'--warmup',5,'--graph',1,'--timeline'],prefix.with_suffix('.log'))
                run(['nsys','export','--type=sqlite','--output='+str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],Path(str(prefix)+'.export.log'))
                trace=analyze(str(prefix)+'.sqlite',str(prefix)+'.analysis.json',a.wsl_cupti_workaround)
                if trace['status']!='executed':raise RuntimeError('Nsight GPU activities missing')
        finally:
            if a.wsl_cupti_workaround:
                if saved is None:config_path.unlink(missing_ok=True)
                else:config_path.write_bytes(saved)
    if sha(binary)!=binary_hash or any(sha(ROOT/path)!=digest for path,digest in hashes.items()):raise RuntimeError('Sources/binary changed during measurement')
    write(a.output/'completion.json',{'status':'completed','binary_sha256':binary_hash,'source_sha256':hashes,'model_hashes':frozen['model_artifacts'],'accuracy_mAP':'not run: missing labeled dataset','nsight':a.nsight})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
