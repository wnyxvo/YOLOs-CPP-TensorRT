#!/usr/bin/env python3
"""Collect reproducible local metadata without installing dependencies or rebuilding models."""
import argparse, ctypes, hashlib, importlib, json, platform, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def command(args):
    try:
        r=subprocess.run(args,text=True,capture_output=True,timeout=60)
        return {'argv':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except (OSError,subprocess.TimeoutExpired) as e: return {'argv':args,'error':str(e)}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,default=Path('build/baseline_detection'));a=p.parse_args()
    versions={}
    for name in ('ultralytics','torch','cv2','onnx','onnxruntime','tensorrt'):
        try: versions[name]=importlib.import_module(name).__version__
        except ImportError as e:versions[name]={'unavailable':str(e)}
    import onnx,torch
    model=onnx.load('models/yolo11n.onnx')
    obj={'timestamp':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'python':sys.version,'executable':sys.executable,
         'platform':platform.platform(),'python_packages':versions,'torch_cuda_runtime':torch.version.cuda,
         'torch_cuda_available':torch.cuda.is_available(),
         'commands':{name:command(args) for name,args in {
           'git_commit':['git','rev-parse','HEAD'],'git_status':['git','status','--porcelain=v1'],
           'gpu':['nvidia-smi','--query-gpu=name,driver_version,pci.bus_id,memory.total,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem','--format=csv'],
           'nvidia_smi':['nvidia-smi'],'cuda_toolkit':['/usr/local/cuda/bin/nvcc','--version'],
           'linked_libraries':['ldd',str(a.binary)]}.items()},
         'model_artifacts':[sha(Path('models')/name) for name in ('yolo11n.pt','yolo11n.onnx','yolo11n.trt')],
         'onnx':{'opsets':[{'domain':o.domain,'version':o.version} for o in model.opset_import],
                 'metadata':{m.key:m.value for m in model.metadata_props}},
         'export_command':'python models/export_onnx.py --model yolo11n',
         'export_settings':{'imgsz':640,'batch':1,'dynamic':False,'opset':11,'nms':False,'half':False,
                            'provenance':'command reported by user plus current export script and ONNX metadata; not re-exported in this run'},
         'engine_build_command':'./build/onnx2trt models/yolo11n.onnx models/yolo11n.trt --fp16 --workspace 2048',
         'engine_build_settings':{'fp16_builder_flag':True,'workspace_mib':2048,'int8':False,
                                 'provenance':'user-reported command; build log unavailable; same existing Engine reused; no rebuild'},
         'cmake_cache':Path('build/CMakeCache.txt').read_text(),
         'tracked_worktree_diff':command(['git','diff','--binary'])}
    if torch.cuda.is_available():
        prop=torch.cuda.get_device_properties(0)
        obj['torch_gpu_properties']={'name':prop.name,'compute_capability':[prop.major,prop.minor],'total_memory_bytes':prop.total_memory}
    obj['executable_artifact']=sha(a.binary)
    linked=obj['commands']['linked_libraries'].get('stdout','')
    nvinfer=next((line.split('=>')[1].split()[0] for line in linked.splitlines() if 'libnvinfer.so' in line and '=>' in line),None)
    if nvinfer:
        lib=ctypes.CDLL(nvinfer);lib.getInferLibVersion.restype=ctypes.c_int32
        obj['tensorrt_cpp_runtime']={'path':nvinfer,'getInferLibVersion':lib.getInferLibVersion()}
    obj['source_files_role']='current worktree sources; legacy executable uses saved original snapshot/instrumentation.patch'
    obj['source_files']=[sha(p) for p in sorted(Path('scripts/baseline').glob('*.py'))]+[sha(Path(p)) for p in
     ['include/yolos/core/cuda_preprocessing.cu','include/yolos/core/trt_session_base.hpp','include/yolos/tasks/detection.hpp','src/baseline_detection.cpp']]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(obj,indent=2)+'\n')
if __name__=='__main__':main()
