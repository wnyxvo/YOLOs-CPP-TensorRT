#!/usr/bin/env python3
"""Reproduce the frozen synchronous baseline; use the same existing Engine."""
import argparse,hashlib,json,subprocess
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-prefix',type=Path,required=True)
p.add_argument('--images',type=Path,default=Path('results/baseline/samples'))
p.add_argument('--mode',choices=['cpu','gpu-filter','gpu-nms'],default='gpu-filter')
p.add_argument('--graph',type=int,choices=[0,1],default=1)
p.add_argument('--warmup',type=int,default=30);p.add_argument('--runs',type=int,default=200)
p.add_argument('--checks',action='store_true')
a=p.parse_args()
frozen=Path('results/async_pipeline/baseline')
version=json.loads((frozen/'version.json').read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
binary=frozen/'sync_baseline.binary'
if sha(binary)!=version['binary_sha256']:raise RuntimeError('Frozen executable hash changed')
if sha('models/yolo11n.trt')!=version['model_artifacts']['models/yolo11n.trt']:raise RuntimeError('Engine changed')
if Path(str(a.output_prefix)+'.config.csv').exists():raise RuntimeError('Keep existing evidence; choose a new prefix')
a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
command=[str(binary),'models/yolo11n.trt','models/coco.names',str(a.images),str(a.output_prefix),
         '--postprocess',a.mode,'--graph',str(a.graph),'--warmup',str(a.warmup),'--runs',str(a.runs),
         '--conf','.25','--iou','.45','--max-det','300']
if a.checks:command+=['--checks']
with Path(str(a.output_prefix)+'.log').open('w') as log:
 result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
Path(str(a.output_prefix)+'.provenance.json').write_text(json.dumps({'frozen_version':version,'argv':command,'returncode':result.returncode,'gpu':subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout},indent=2)+'\n')
if result.returncode:raise RuntimeError('Synchronous baseline failed; inspect log')
print('Saved frozen synchronous baseline',a.output_prefix)
