#!/usr/bin/env python3
"""Summarize actual Nsight GPU intervals and CUDA calls within MEASURED_BATCH."""
import argparse, collections, json, sqlite3
from pathlib import Path


def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]: merged[-1][1]=max(b,merged[-1][1])
        else: merged.append([a,b])
    return merged


def length(intervals): return sum(b-a for a,b in intervals)/1e6


def overlap(a,b):
    a,b=union(a),union(b);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return total/1e6


def analyze(path,output,raw_timestamp_workaround=False):
    c=sqlite3.connect(path);c.row_factory=sqlite3.Row
    tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    strings={r['id']:r['value'] for r in c.execute('select * from StringIds')}
    nvtx=[dict(r) for r in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
    for r in nvtx:r['name']=r['text'] or strings.get(r['textId'],'')
    scope=[r for r in nvtx if r['name']=='MEASURED_BATCH']
    if len(scope)!=1:raise RuntimeError('Require exactly one measured NVTX batch')
    start,end=scope[0]['start'],scope[0]['end']
    calls=collections.Counter()
    for r in c.execute('select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=?',(start,end)):
        calls[strings[r['nameId']].split('_v')[0]]+=1
    result={'sqlite':str(path),'scope':'MEASURED_BATCH; warmup/setup excluded','batch_ms':(end-start)/1e6,
            'cuda_api_calls':dict(calls),'raw_gpu_timestamp_workaround':raw_timestamp_workaround,
            'timestamp_note':'CUPTI GPU/CPU conversion workaround for WSL; NVIDIA describes reduced timestamp accuracy' if raw_timestamp_workaround else None,
            'profiler_timings_are_not_final_performance':True}
    if not {'CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY'}<=tables:
        result.update(status='blocked_gpu_activity_missing',overlap_ms=None)
    else:
        kernels=[dict(r) for r in c.execute('select start,end,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(start,end))]
        copies=[dict(r) for r in c.execute('select start,end,streamId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(start,end))]
        for r in kernels:r['name']=strings[r['demangledName']]
        cpu=[r for r in nvtx if r['name']=='CPU NMS and result packaging' and r['start']>=start and r['end']<=end]
        intervals=lambda rows:[(r['start'],r['end']) for r in rows]
        k,h,d,p=intervals(kernels),intervals([r for r in copies if r['copyKind']==1]),intervals([r for r in copies if r['copyKind']==2]),intervals(cpu)
        result.update(status='executed',kernel_count=len(k),h2d_count=len(h),d2h_count=len(d),cpu_post_ranges=len(p),
                      kernel_stream_ids=sorted({r['streamId'] for r in kernels}),
                      copy_stream_ids=sorted({r['streamId'] for r in copies}),
                      union_busy_ms={'kernels':length(union(k)),'h2d':length(union(h)),'d2h':length(union(d)),'cpu_post':length(union(p))},
                      overlap_ms={'h2d_kernels':overlap(h,k),'d2h_kernels':overlap(d,k),'h2d_d2h':overlap(h,d),'cpu_post_kernels':overlap(p,k)},
                      h2d_bytes=sum(r['bytes'] for r in copies if r['copyKind']==1),d2h_bytes=sum(r['bytes'] for r in copies if r['copyKind']==2))
        # Standard plotting artifact: first 20 ms, with each interval drawn from raw trace.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,3.4))
        for index,(label,data,color) in enumerate([('H2D',h,'#3388cc'),('GPU kernels',k,'#33aa77'),('D2H',d,'#bb7733'),('CPU NMS / packaging',p,'#aa55aa')]):
            bars=[((a-start)/1e6,(b-a)/1e6) for a,b in data if a-start<20e6]
            ax.broken_barh(bars,(index-.3,.6),facecolors=color)
        ax.set(yticks=range(4),yticklabels=['H2D','GPU kernels','D2H','CPU NMS / packaging'],xlabel='ms from MEASURED_BATCH start',xlim=(0,min(20,(end-start)/1e6)))
        ax.grid(axis='x',alpha=.25);fig.tight_layout();fig.savefig(Path(output).with_suffix('.svg'));plt.close(fig)
    Path(output).write_text(json.dumps(result,indent=2)+'\n');c.close();return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('output',type=Path);p.add_argument('--wsl-cupti-workaround',action='store_true');a=p.parse_args()
    print(json.dumps(analyze(a.sqlite,a.output,a.wsl_cupti_workaround),indent=2))
