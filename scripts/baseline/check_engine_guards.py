#!/usr/bin/env python3
"""Build tiny negative-test engines; confirm unsupported I/O is rejected before inference."""
import argparse,json,subprocess
from pathlib import Path
import onnx
from onnx import TensorProto,helper

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
p.add_argument('--binary',default='build/baseline_detection');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
records=[]
for name,dtype,shape,expected in (
 ('half_io',TensorProto.FLOAT16,[1,3,640,640],'only FP32 I/O is implemented'),
 ('batch2',TensorProto.FLOAT,[2,3,640,640],'Expected static FP32 NCHW input [1,3,H,W]')):
    model=helper.make_model(helper.make_graph([helper.make_node('Identity',['images'],['output0'])],name,
     [helper.make_tensor_value_info('images',dtype,shape)],[helper.make_tensor_value_info('output0',dtype,shape)]),
     opset_imports=[helper.make_opsetid('',11)],ir_version=9)
    onnx_path=a.output/(name+'.onnx');engine=a.output/(name+'.trt');onnx.save(model,onnx_path)
    r=subprocess.run(['build/onnx2trt',str(onnx_path),str(engine),'--workspace','64'],capture_output=True,text=True)
    (a.output/(name+'.build.log')).write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('Cannot build '+name)
    r=subprocess.run([a.binary,str(engine),'models/coco.names','data/dog.jpg',str(a.output/(name+'.probe')),
                      '--runs','1','--warmup','0'],capture_output=True,text=True)
    (a.output/(name+'.probe.log')).write_text(r.stdout+r.stderr)
    ok=r.returncode!=0 and expected in r.stderr
    records.append({'case':name,'shape':shape,'onnx_dtype':dtype,'returncode':r.returncode,'expected_error':expected,'pass':ok})
(a.output/'guards.json').write_text(json.dumps(records,indent=2)+'\n')
if not all(r['pass'] for r in records):raise RuntimeError('An unsupported Engine was silently accepted')
print('PASS half I/O and batch>1 Engine rejection')
