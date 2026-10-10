#!/usr/bin/env python3
"""Summarize raw CSV latency; never sum stages to estimate end-to-end latency."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def stats(rows,key):
    x=np.array([float(r[key]) for r in rows if r[key]])
    if not len(x):return None
    return {'samples':len(x),'mean_ms':float(x.mean()),'p50_ms':float(np.percentile(x,50)),
            'p95_ms':float(np.percentile(x,95)),'min_ms':float(x.min()),'max_ms':float(x.max())}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('prefixes',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();summary=[]
    for prefix in a.prefixes:
        rows=list(csv.DictReader(open(str(prefix)+'.timings.csv')))
        cfg=next(csv.DictReader(open(str(prefix)+'.config.csv')))
        if len(rows)!=int(cfg['runs_per_image'])*int(cfg['images']):raise ValueError('Incomplete run '+str(prefix))
        metrics=[k for k in rows[0] if k.endswith('_ms')]
        r={'prefix':str(prefix),'configuration':cfg,'aggregate':{k:stats(rows,k) for k in metrics},
           'per_image':{im:{k:stats([r for r in rows if r['image']==im],k) for k in metrics}
                        for im in sorted({r['image'] for r in rows})}}
        summary.append(r)
        e=r['aggregate']['e2e_ms'];print(prefix,'mean/P50/P95',*[round(e[k],4) for k in ('mean_ms','p50_ms','p95_ms')])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'e2e_scope':'memory cv::Mat input -> final CPU detections, excluding decode/draw/display/write',
      'gpu_events':'single stream; read only after existing stream synchronization; stages are observations, not additive e2e estimates',
      'percentiles':'numpy.percentile linear interpolation; per-image and pooled equally weighted sample counts',
      'runs':summary},indent=2)+'\n')
if __name__=='__main__':main()
