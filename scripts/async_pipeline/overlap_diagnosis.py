#!/usr/bin/env python3
"""Independent copy/compute diagnostic, with actual Nsight intervals as evidence.

Run from repository root in yolos-trt. Requires nvcc, nsys and matplotlib.
It temporarily applies NVIDIA's WSL CUPTI timestamp workaround and restores it.
Production detector sources, Engine, Graph scope and performance data are untouched.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timeline import union, length, overlap, analyze

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/async_pipeline/overlap_diagnosis'
NSYS='/usr/local/bin/nsys'

def run(args, name):
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    commands.append(args)

def case_analysis(path):
    c=sqlite3.connect(path); c.row_factory=sqlite3.Row
    strings={r['id']:r['value'] for r in c.execute('select * from StringIds')}
    nvtx=[dict(r) for r in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
    cases=[]; intervals_by_name={}
    for scope in nvtx:
        name=scope['text'] or strings.get(scope['textId'],'')
        if not name.startswith(('interleaved_','lookahead_one_','batched_uploads_','explicit_serial_','host_gap_')): continue
        start,end=scope['start'],scope['end']; bounds=(start,end)
        kernels=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',bounds)]
        copies=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',bounds)]
        h=sorted((r for r in copies if r['copyKind']==1),key=lambda r:r['start'])
        d=sorted((r for r in copies if r['copyKind']==2),key=lambda r:r['start'])
        if len(kernels)!=12 or len(h)!=12 or len(d)!=12: raise RuntimeError(f'{name}: incomplete GPU activity')
        calls=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',bounds)]
        api={r['correlationId']:r for r in calls}
        counts=collections.Counter(strings[r['nameId']].split('_v')[0] for r in calls)
        if any(counts[k] for k in ['cudaMalloc','cudaMallocHost','cudaFree','cudaFreeHost','cudaDeviceSynchronize']): raise RuntimeError('Unexpected allocation or device synchronization inside case')
        iv=lambda rows:[(r['start'],r['end']) for r in rows]
        hi,di,ki=iv(h),iv(d),iv(kernels)
        hk,dk=overlap(hi,ki),overlap(di,ki)
        stats={'case':name,'batch_ms':(end-start)/1e6,'kernels':len(kernels),'h2d_count':len(h),'d2h_count':len(d),
            'h2d_bytes':h[0]['bytes'],'d2h_bytes':d[0]['bytes'],'h2d_busy_ms':length(union(hi)),
            'd2h_busy_ms':length(union(di)),'kernel_busy_ms':length(union(ki)),
            'h2d_kernel_overlap_ms':hk,'d2h_kernel_overlap_ms':dk,'h2d_d2h_overlap_ms':overlap(hi,di),
            'h2d_overlap_fraction':hk/length(union(hi)),'d2h_overlap_fraction':dk/length(union(di)),
            'h2d_next_prequeued_before_previous_d2h_end':sum(api[h[i]['correlationId']]['end']<d[i-1]['end'] for i in range(1,12)),
            'h2d_next_starts_after_previous_d2h_end':sum(h[i]['start']>=d[i-1]['end'] for i in range(1,12)),
            'cuda_api_calls':dict(counts)}
        cases.append(stats); intervals_by_name[name]=(start,hi,ki,di)
    if len(cases)!=26: raise RuntimeError(f'Expected 26 controls, got {len(cases)}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['interleaved_project_flags0_graph1','lookahead_one_project_flags0_graph1','batched_uploads_project_flags0_graph1',
           'explicit_serial_project_flags1_graph0','host_gap_project_flags1_graph0']
    fig,axes=plt.subplots(len(names),1,figsize=(12,10),sharex=True)
    for ax,name in zip(axes,names):
        start,*lanes=intervals_by_name[name]
        for lane,(iv,color) in enumerate(zip(lanes,['#3388cc','#33aa77','#bb7733'])):
            ax.broken_barh([((a-start)/1e6,(b-a)/1e6) for a,b in iv],(lane-.3,.6),facecolors=color)
        ax.set(yticks=range(3),yticklabels=['H2D','compute','D2H'],title=name,xlim=(0,24)); ax.grid(axis='x',alpha=.25)
    axes[-1].set_xlabel('ms from case start (independent profiler run)')
    fig.tight_layout(); fig.savefig(OUT/'controls.svg'); plt.close(fig)
    result={'timestamp_note':'CuptiUseRawGpuTimestamps=false; NVIDIA notes reduced timestamp accuracy on WSL',
            'cases':cases,'stream_metadata':[dict(r) for r in c.execute('select * from TARGET_INFO_CUDA_STREAM')],
            'stream_flag_enum':[dict(r) for r in c.execute('select * from ENUM_CUPTI_STREAM_TYPE')]}
    c.close(); return result

def production_audit(path):
    c=sqlite3.connect(path); c.row_factory=sqlite3.Row
    strings={r['id']:r['value'] for r in c.execute('select * from StringIds')}
    scope=next(r for r in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null') if (r['text'] or strings.get(r['textId']))=='MEASURED_BATCH')
    start,end=scope['start'],scope['end']; api={r['correlationId']:dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(start,end))}
    h=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=1 and start>=? and end<=? order by start',(start,end))]
    d=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and start>=? and end<=? order by start',(start,end))]
    transitions=[{'frame':i,'next_h2d_api_end_ms':(api[h[i]['correlationId']]['end']-start)/1e6,
                  'previous_d2h_gpu_end_ms':(d[i-1]['end']-start)/1e6,
                  'next_h2d_gpu_start_ms':(h[i]['start']-start)/1e6,
                  'prequeued':api[h[i]['correlationId']]['end']<d[i-1]['end'],
                  'gpu_starts_after_previous_d2h':h[i]['start']>=d[i-1]['end']} for i in range(1,len(h))]
    activity_streams={}
    for table in ['KERNEL','MEMCPY','MEMSET']:
        activity_streams[table]=[dict(r) for r in c.execute(f'select streamId,count(*) as activities from CUPTI_ACTIVITY_KIND_{table} where start>=? and end<=? group by streamId',(start,end))]
    result={'stream_metadata':[dict(r) for r in c.execute('select * from TARGET_INFO_CUDA_STREAM')],
            'stream_flag_enum':[dict(r) for r in c.execute('select * from ENUM_CUPTI_STREAM_TYPE')],
            'activity_streams':activity_streams,'prequeued_transitions':sum(x['prequeued'] for x in transitions),
            'gpu_serial_transitions':sum(x['gpu_starts_after_previous_d2h'] for x in transitions),
            'transitions':transitions}
    c.close(); return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--analyze-only',action='store_true'); args=p.parse_args()
    os.chdir(ROOT); OUT.mkdir(parents=True,exist_ok=True); commands=[]
    if not args.analyze_only:
        run(['/usr/local/cuda/extras/demo_suite/deviceQuery'],'device_query')
        run(['nvidia-smi'],'nvidia_smi')
        run(['uname','-a'],'uname')
        run(['git','rev-parse','HEAD'],'commit')
        run(['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch=sm_89','scripts/async_pipeline/copy_overlap_probe.cu','-lnvToolsExt','-o',str(OUT/'probe.binary')],'build_probe')
        run([str(OUT/'probe.binary'),str(OUT/'unprofiled.csv')],'unprofiled')
        config=Path(subprocess.check_output([NSYS,'-z'],text=True).strip())
        original=config.read_bytes() if config.exists() else None
        config.parent.mkdir(parents=True,exist_ok=True)
        try:
            config.write_bytes((original or b'')+b'\nCuptiUseRawGpuTimestamps=false\n')
            run([NSYS,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                 '--force-overwrite=true','--output='+str(OUT/'controls'),str(OUT/'probe.binary'),str(OUT/'profiled.csv')],'profile_controls')
            run([NSYS,'export','--type=sqlite','--force-overwrite=true','--output='+str(OUT/'controls.sqlite'),str(OUT/'controls.nsys-rep')],'export_controls')
            for graph in [0,1]:
                prefix=OUT/f'pipeline_graph{graph}'
                run([NSYS,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                     '--capture-range=nvtx','--nvtx-capture=MEASURED_BATCH','--capture-range-end=stop','--force-overwrite=true','--output='+str(prefix),
                     './build/async_pipeline_benchmark','models/yolo11n.trt','models/coco.names','results/baseline/samples',str(prefix),
                     '--slots','2','--frames','140','--warmup','5','--graph',str(graph),'--timeline'],f'profile_graph{graph}')
                run([NSYS,'export','--type=sqlite','--force-overwrite=true','--output='+str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],f'export_graph{graph}')
        finally:
            if original is None: config.unlink(missing_ok=True)
            else: config.write_bytes(original)
        (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    results={'controls':case_analysis(OUT/'controls.sqlite'),'production':{}}
    for graph in [0,1]:
        path=OUT/f'pipeline_graph{graph}.sqlite'
        summary=analyze(path,OUT/f'pipeline_graph{graph}.analysis.json',True)
        results['production'][str(graph)]={'summary':summary,'audit':production_audit(path)}
    (OUT/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    tracked=['models/yolo11n.trt','build/async_pipeline_benchmark','include/yolos/core/trt_utils.hpp','include/yolos/tasks/async_detection.hpp',
             'scripts/async_pipeline/copy_overlap_probe.cu','scripts/async_pipeline/overlap_diagnosis.py']
    (OUT/'sha256.json').write_text(json.dumps({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in tracked},indent=2)+'\n')
    for case in results['controls']['cases']:
        print(case['case'],f"H2D overlap {case['h2d_kernel_overlap_ms']:.4f}ms ({case['h2d_overlap_fraction']:.1%}), D2H {case['d2h_kernel_overlap_ms']:.4f}ms, prequeued {case['h2d_next_prequeued_before_previous_d2h_end']}/11")
