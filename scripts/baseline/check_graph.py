#!/usr/bin/env python3
"""Compare Graph on/off outputs for the same Engine and images, with explicit deltas."""
import argparse,json
from pathlib import Path
import numpy as np
from reference import read_detections,match,differences

p=argparse.ArgumentParser(description=__doc__);p.add_argument('on',type=Path);p.add_argument('off',type=Path)
p.add_argument('--images',type=Path,default=Path('results/baseline/samples'));p.add_argument('--output',type=Path,required=True)
a=p.parse_args();on=read_detections(str(a.on)+'.detections.csv');off=read_detections(str(a.off)+'.detections.csv')
rows=[]
for im in sorted(a.images.glob('*.png')):
    m=match(on.get(im.name,[]),off.get(im.name,[]),min_iou=.999)
    inp=[np.fromfile(str(prefix)+'.'+im.name+'.input.f32',np.float32) for prefix in (a.on,a.off)]
    raw=[np.fromfile(str(prefix)+'.'+im.name+'.output0.f32',np.float32) for prefix in (a.on,a.off)]
    rows.append({'image':im.name,'detections':m,'input':differences(*inp),'raw_output':differences(*raw),
     'bit_exact_input':np.array_equal(*inp),'bit_exact_raw_output':np.array_equal(*raw)})
passed=bool(rows) and all(r['bit_exact_input'] and r['bit_exact_raw_output'] and not r['detections']['unmatched_actual']
 and not r['detections']['unmatched_reference'] and r['detections']['max_coordinate_abs_delta']==0
 and r['detections']['max_score_abs_delta']==0 for r in rows)
a.output.write_text(json.dumps({'pass':passed,'images':rows},indent=2)+'\n')
print('Graph equivalence:',passed)
raise SystemExit(0 if passed else 1)
