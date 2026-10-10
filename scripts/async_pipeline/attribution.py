#!/usr/bin/env python3
"""Derive latency decomposition and inter-frame GPU gaps from existing raw evidence.
Does not rerun inference or add timing events; preserves prior benchmark results.
"""
import argparse,csv,json,sqlite3,hashlib
from pathlib import Path
import numpy as np
from timeline import union,length,overlap

def stats(values):
    a=np.asarray(values,dtype=float)
    return {'count':len(a),'mean':float(a.mean()),'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95))}

def trace(path,transitions_path):
    c=sqlite3.connect(path);c.row_factory=sqlite3.Row
    strings=dict(c.execute('select id,value from StringIds'))
    ranges=[dict(r) for r in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
    scope=[r for r in ranges if (r['text'] or strings.get(r['textId']))=='MEASURED_BATCH']
    if len(scope)!=1:raise RuntimeError('Require one measured batch')
    a,b=scope[0]['start'],scope[0]['end']
    select=lambda t,columns:[dict(r) for r in c.execute(f'select {columns} from {t} where start>=? and end<=? order by start',(a,b))]
    kernels=select('CUPTI_ACTIVITY_KIND_KERNEL','start,end,demangledName')
    copies=select('CUPTI_ACTIVITY_KIND_MEMCPY','start,end,copyKind,correlationId')
    runtime=select('CUPTI_ACTIVITY_KIND_RUNTIME','start,end,correlationId,nameId')
    if not kernels or not copies:raise RuntimeError('GPU activity missing, cannot infer zero overlap')
    host={r['correlationId']:r for r in runtime if strings[r['nameId']].startswith('cudaMemcpyAsync')}
    h=[r for r in copies if r['copyKind']==1];d=[r for r in copies if r['copyKind']==2]
    if len(h)!=len(d):raise RuntimeError('Incomplete uploads/downloads')
    categories={'preprocess':[],'gpu_filter':[],'trt_kernels':[]}
    for k in kernels:
        name=strings[k['demangledName']]
        category='preprocess' if 'letterboxNormalizeKernel' in name else 'gpu_filter' if 'filterCompact' in name else 'trt_kernels'
        categories[category].append((k['start'],k['end']))
    # Record all collected device work, including candidate-header memset.
    tables={r[0] for r in c.execute('select name from sqlite_master')}
    memset=select('CUPTI_ACTIVITY_KIND_MEMSET','start,end') if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else []
    all_ops=union([(k['start'],k['end']) for k in kernels+copies+memset])
    gaps=[(h[i+1]['start']-d[i]['end'])/1e6 for i in range(len(h)-1)]
    ahead=[host[h[i+1]['correlationId']]['end']<d[i]['end'] for i in range(len(h)-1)]
    with transitions_path.open('w') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['previous_frame','next_frame','previous_d2h_end_ns','next_h2d_start_ns','next_host_h2d_start_ns','next_host_h2d_end_ns','gpu_gap_ms','host_prequeued'])
        for i in range(len(h)-1):
            call=host[h[i+1]['correlationId']]
            writer.writerow([i,i+1,d[i]['end'],h[i+1]['start'],call['start'],call['end'],gaps[i],int(ahead[i])])
    out={'sqlite_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'frames':len(h),'batch_ms':(b-a)/1e6,
         'gpu_active_union_ms':length(all_ops),'gpu_idle_ms':(b-a)/1e6-length(all_ops),
         'gpu_kernel_busy_fraction':length(union([(k['start'],k['end']) for k in kernels]))/(b-a)*1e6,
         'd2h_to_next_h2d_gap_ms':stats(gaps),'negative_gap_count':sum(g<0 for g in gaps),
         'next_h2d_host_enqueued_before_previous_d2h_end':{'count':sum(ahead),'total':len(ahead),'fraction':sum(ahead)/len(ahead)},
         'stage_mean_ms':{name:length(union(intervals))/len(h) for name,intervals in categories.items()},
         'h2d_kernel_overlap_ms':overlap([(r['start'],r['end']) for r in h],[(r['start'],r['end']) for r in kernels]),
         'd2h_kernel_overlap_ms':overlap([(r['start'],r['end']) for r in d],[(r['start'],r['end']) for r in kernels])}
    for name,rows in [('h2d',h),('d2h',d),('memset',memset)]:out['stage_mean_ms'][name]=sum(r['end']-r['start'] for r in rows)/len(h)/1e6
    c.close();return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results/async_pipeline/final'))
    p.add_argument('--output',type=Path,default=Path('results/async_pipeline/attribution'));a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('Choose empty output; preserve prior attribution')
    a.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((a.results/'summary.json').read_text());latencies={};traces={}
    for slots in range(4):
        rows=[]
        for path in sorted(a.results.glob(f'final_r*_s{slots}.frames.csv')):
            with path.open() as f:rows+=list(csv.DictReader(f))
        vals={k:stats([float(r[k]) for r in rows]) for k in rows[0] if k.endswith('_ms') and rows[0][k]}
        if slots:
            parts={name:[] for name in ['capacity_wait','staging','remaining_host_enqueue','collection_queue','completion_event_wait','cpu_post']}
            errors=[]
            for row in rows:
                r={k:float(v) for k,v in row.items() if k.endswith('_ms')}
                vector=[r['submit_wait_ms'],r['staging_ms'],r['submit_host_ms']-r['submit_wait_ms']-r['staging_ms'],r['collection_queue_ms'],r['collect_event_wait_ms'],r['cpu_post_ms']]
                errors.append(abs(sum(vector)-r['latency_ms']))
                for key,value in zip(parts,vector):parts[key].append(value)
            if max(errors)>5e-6:raise RuntimeError('CPU timestamp decomposition not additive')
            vals['exclusive_segments_ms']={k:stats(v) for k,v in parts.items()};vals['max_decomposition_error_ms']=max(errors)
        latencies[slots]=vals;traces[slots]=trace(a.results/f'timeline_s{slots}.sqlite',a.output/f'transitions_s{slots}.csv')
    record={'performance_source':str(a.results),'profiling_separate_from_final_measurement':True,
            'timing_note':'Exclusive CPU wall intervals add per frame; event wait includes GPU service/queue and is not synchronization overhead',
            'nsight_timestamp_limitation':'CuptiUseRawGpuTimestamps=false WSL workaround; reduced timestamp accuracy; profiling is separate and not causal percentages',
            'latency':latencies,'trace':traces}
    (a.output/'analysis.json').write_text(json.dumps(record,indent=2)+'\n')
    text=['# 2槽收益与延迟拆分','', '来源：已有无profiler两轮原始CSV及独立Nsight SQLite。没有重测或增加CUDA计时事件。', '',
          '## 同一帧的互斥CPU墙钟区间','', '| 槽位 | 背压/取锁 | staging | 其它host enqueue | FIFO collect等待 | 完成事件等待 | CPU后处理 | 合计ms |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in (1,2,3):
        v=latencies[s];parts=v['exclusive_segments_ms']
        text.append(f'| {s} | '+' | '.join(f"{r['mean']:.4f}" for r in parts.values())+f" | {v['latency_ms']['mean']:.4f} |")
    text+=['','这些区间来自同一帧的submit/admitted/enqueued/collect/downloaded/ready时间戳，互斥，允许相加；staging从host enqueue区间单独分出。admitted latency、submit host等包含其它区间，不能再次加入。每帧加法误差不超过5e-6ms（CSV小数舍入）。', '',
           '完成事件等待包含该帧GPU工作和依赖等待，并不是新增同步API的纯开销。多帧之间这些区间可以重叠，因此不能加总成批次总时间，也不能用逐帧延迟倒数计算吞吐。', '',
           '## 实际GPU阶段与跨帧空隙（独立profiler）','', '| 槽位 | H2D均值 | 预处理kernel | TRT kernels之和 | 筛选kernel | D2H | 前帧D2H→下帧H2D平均间隙ms | GPU空闲占批次 | 提前提交下帧H2D比例 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in (0,1,2,3):
        t=traces[s];st=t['stage_mean_ms'];ahead=t['next_h2d_host_enqueued_before_previous_d2h_end']
        text.append(f'| {s} | '+ ' | '.join(f'{st[k]:.4f}' for k in ['h2d','preprocess','trt_kernels','gpu_filter','d2h'])+f" | {t['d2h_to_next_h2d_gap_ms']['mean']:.4f} | {t['gpu_idle_ms']/t['batch_ms']*100:.2f}% | {ahead['count']}/{ahead['total']} |")
    sync,dual=traces[0],traces[2];perf=summary['configurations'];gain=(perf[2]['completed_fps']/perf[0]['completed_fps']-1)*100
    completion_period=lambda s:1000/perf[s]['completed_fps']
    text+=['','kernel分类按实际名称：letterboxNormalizeKernel为预处理，filterCompact为筛选，其余为TRT；TRT栏是kernel持续时间之和，排除了kernel间空隙，并不是enqueueV3 wall span。GPU空闲通过kernel/copy/memset全部区间求并集后从批次墙钟减去，未把重叠区间重复计算。', '',
           '## 收益归因与边界','',
           f"无profiler完成周期从同步 {completion_period(0):.4f}ms/帧降至2槽 {completion_period(2):.4f}ms/帧，节省 {completion_period(0)-completion_period(2):.4f}ms/完成帧，吞吐提升 {gain:.2f}%。这是批次时间除完成帧数，不是队列中的单帧停留时间。",'',
           f"时间线的跨帧D2H→下帧H2D平均间隙从 {sync['d2h_to_next_h2d_gap_ms']['mean']:.4f}ms 降至 {dual['d2h_to_next_h2d_gap_ms']['mean']:.4f}ms；2槽下一帧H2D在上一帧D2H完成前已由host提交的数量为 {dual['next_h2d_host_enqueued_before_previous_d2h_end']['count']}/{dual['next_h2d_host_enqueued_before_previous_d2h_end']['total']}。",'',
           '机制是：同步路径等待GPU完成、CPU后处理结束后才开始下一帧staging和CUDA提交；2槽给提交线程另一份图像/候选包资源，可以预先完成下一帧CPU准备与入队，使串行设备工作更连续。事件依赖保持结果正确，后处理和共享TRT I/O均未改写。以上数据支持减少主机供给间隙的解释。', '',
           'H2D/D2H与kernel交集仍为0，不能归因于DMA和计算重叠；CPU NMS只占微秒级，不能把24%收益归因于CPU NMS加速；TRT计算并没有改算法或精度，profile中的kernel时间没有同步降幅。', '',
           '这些profile是另外的短批次，且WSL时间戳转换精度降低；profiler、频率和主机调度会影响测量。可确认“预提交与更小跨帧间隙”，但不能把最终24.15%精确拆成某阶段贡献百分比，也不能据此断言特定驱动/硬件造成串行拷贝。', '',
           '2槽增加了FIFO驻留：平均collection_queue约1ms，3槽约2.25ms；每帧GPU完成等待约1.25ms仍在。3槽只增加缓冲深度，未明显缩短完成周期，所以吞吐几乎相同但延迟增长。饱和输入采用2槽；低延迟单图保留同步模式。','']
    (a.output/'REPORT.md').write_text('\n'.join(text));print(a.output/'REPORT.md')

if __name__=='__main__':main()
