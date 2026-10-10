#!/usr/bin/env python3
"""Run an isolated baseline experiment, preserving earlier evidence directories.

Activate yolos-trt first. GPU access is required for the C++ runner.
No models are re-exported or rebuilt. Images are loaded once per process.
"""
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--images',type=Path,default=Path('results/baseline/samples'))
    p.add_argument('--binary',type=Path,default=Path('build/baseline_detection'))
    p.add_argument('--warmup',type=int,default=30);p.add_argument('--runs',type=int,default=200)
    p.add_argument('--legacy',action='store_true',help='Saved original binary; skip corrected-only guards')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()): raise RuntimeError('Refusing to overwrite evidence directory: '+str(a.output))
    a.output.mkdir(parents=True,exist_ok=True)
    commands=[]
    def run(args,name):
        record={'argv':[str(x) for x in args],'start':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
        print('Running',name,flush=True)
        with (a.output/(name+'.log')).open('w') as f:
            r=subprocess.run(record['argv'],stdout=f,stderr=subprocess.STDOUT)
        record.update(returncode=r.returncode,end=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        commands.append(record);(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if r.returncode:raise RuntimeError('Failed '+name+'; see '+str(a.output/(name+'.log')))
    py=sys.executable
    run([py,'scripts/baseline/collect_environment.py','--output',a.output/'environment.json','--binary',a.binary.resolve()],'environment')
    binary=str(a.binary.resolve())
    common=[binary,'models/yolo11n.trt','models/coco.names',a.images]
    val=[]
    for g in (1,0):
        prefix=a.output/('validation_'+('on' if g else 'off'));val.append(prefix)
        run(common+[prefix,'--graph',g,'--warmup',5,'--runs',1,'--dump']+([] if a.legacy else ['--checks']),prefix.name)
    run([py,'scripts/baseline/reference.py','--cpp-prefix',val[0],'--images',a.images,
         '--pt','models/yolo11n.pt','--output',a.output/'reference']+([] if a.legacy else ['--require-structural-pass']),'reference')
    run([py,'scripts/baseline/check_graph.py',*val,'--images',a.images,'--output',a.output/'graph_equivalence.json'],'graph_equivalence')
    if not a.legacy:run([py,'scripts/baseline/check_engine_guards.py','--binary',binary,'--output',a.output/'engine_guards'],'engine_guards')
    prefixes=[]
    # Final E2E runs contain no event recording or dumps. Stage runs are separate processes.
    for profile in (False,True):
        for graph in (1,0):
            name=('profile_' if profile else 'graph_')+('on' if graph else 'off')
            prefix=a.output/name;prefixes.append(prefix)
            run(common+[prefix,'--graph',graph,'--warmup',a.warmup,'--runs',a.runs]+(['--profile'] if profile else []),name)
    run([py,'scripts/baseline/summarize.py',*prefixes,'--output',a.output/'summary.json'],'summary')
    print('Evidence saved to',a.output)
if __name__=='__main__':main()
