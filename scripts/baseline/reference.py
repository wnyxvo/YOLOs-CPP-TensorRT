#!/usr/bin/env python3
"""ONNX Runtime CPU FP32 reference with installed Ultralytics preprocessing/NMS.

Separates preprocessing, raw output and postprocessing. Cross-backend comparisons
are observations, not mAP or an assertion that FP16 and FP32 are bit-identical.
"""
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
import ultralytics
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes


def preprocess(im):
    lb = LetterBox(new_shape=(640,640), auto=False, scale_fill=False, scaleup=True, center=True)
    bgr = lb(image=im)
    return np.ascontiguousarray(bgr[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.


def postprocess(raw, shape, conf=.25, iou=.45):
    pred = non_max_suppression(torch.from_numpy(raw.copy()), conf_thres=conf, iou_thres=iou,
                              agnostic=False, multi_label=False, max_det=300, nc=80,
                              max_nms=30000, max_wh=7680, end2end=False)[0]
    scale_boxes((640,640),pred[:,:4],shape)
    return [{'class_id':int(x[5]),'confidence':float(x[4]),'xyxy':x[:4].tolist()} for x in pred.numpy()]


def box_iou(a,b):
    a,b = np.array(a),np.array(b)
    wh = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter=float(wh.prod()); union=float(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter)
    return inter/union if union>0 else 0.


def match(actual, reference, min_iou=.9):
    # Maximum-cardinality bipartite match; edges within the same class only.
    edges = {i:sorted([(j,box_iou(a['xyxy'],b['xyxy'])) for j,b in enumerate(reference)
                     if a['class_id']==b['class_id'] and box_iou(a['xyxy'],b['xyxy'])>=min_iou],
                    key=lambda v:-v[1]) for i,a in enumerate(actual)}
    owner={}
    def visit(i,seen):
        for j,_ in edges[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owner or visit(owner[j],seen): owner[j]=i; return True
        return False
    for i in sorted(edges,key=lambda k:len(edges[k])): visit(i,set())
    pairs=[]
    for j,i in sorted(owner.items()):
        a,b=actual[i],reference[j]
        pairs.append({'class_id':a['class_id'],'actual':a,'reference':b,'iou':box_iou(a['xyxy'],b['xyxy']),
                      'coordinate_delta':(np.array(a['xyxy'])-b['xyxy']).tolist(),
                      'score_delta':a['confidence']-b['confidence']})
    return {'matching':'maximum cardinality; same class; prefer higher IoU edges', 'min_match_iou':min_iou,
            'actual_count':len(actual),'reference_count':len(reference),'matched':pairs,
            'unmatched_actual':[a for i,a in enumerate(actual) if i not in owner.values()],
            'unmatched_reference':[b for j,b in enumerate(reference) if j not in owner],
            'max_coordinate_abs_delta':max([max(abs(x) for x in p['coordinate_delta']) for p in pairs],default=0),
            'max_score_abs_delta':max([abs(p['score_delta']) for p in pairs],default=0)}


def differences(a,b):
    d=np.abs(a.astype(np.float64)-b)
    return {'max_abs':float(d.max()),'mean_abs':float(d.mean()),'p95_abs':float(np.percentile(d,95)),
            'rms':float(np.sqrt(np.mean(d*d)))}


def read_detections(path):
    groups={}
    with open(path) as f:
        for row in csv.DictReader(f):
            groups.setdefault(row['image'],[]).append({'class_id':int(row['class_id']),
             'confidence':float(row['confidence']),'xyxy':[float(row[x]) for x in ('x1','y1','x2','y2')]})
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx',type=Path,default=Path('models/yolo11n.onnx'))
    p.add_argument('--pt',type=Path,help='Optional secondary PT CPU FP32 cross-check')
    p.add_argument('--images',type=Path,default=Path('results/baseline/samples'))
    p.add_argument('--cpp-prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-structural-pass',action='store_true',help='Fail on preprocessing bound, identical-output NMS, or unmatched detections')
    p.add_argument('--conf',type=float,default=.25);p.add_argument('--iou',type=float,default=.45)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    session=ort.InferenceSession(str(a.onnx),sess_options=opts,providers=['CPUExecutionProvider'])
    names=Path('models/coco.names').read_text().splitlines()
    if len(names)!=80: raise ValueError('Expected 80 COCO labels')
    aliases={'motorbike':'motorcycle','aeroplane':'airplane','sofa':'couch','pottedplant':'potted plant','diningtable':'dining table','tvmonitor':'tv'}
    canonical_names=[aliases.get(n,n) for n in names]
    pt=None
    if a.pt:
        pt=ultralytics.YOLO(str(a.pt)).model.float().eval().cpu();pt.fuse(verbose=False)
        if [pt.names[i] for i in range(80)]!=canonical_names: raise ValueError('PT class ID mapping differs')
    cpp=read_detections(str(a.cpp_prefix)+'.detections.csv')
    config=next(csv.DictReader(open(str(a.cpp_prefix)+'.config.csv')))
    if abs(float(config['conf'])-a.conf)>1e-6 or abs(float(config['iou'])-a.iou)>1e-6:
        raise ValueError('C++ and reference thresholds differ')
    records=[];reference={}
    for path in sorted(a.images.glob('*.png')):
        im=cv2.imread(str(path)); x=preprocess(im)
        gpu=np.fromfile(str(a.cpp_prefix)+'.'+path.name+'.input.f32',np.float32).reshape(1,3,640,640)
        trt=np.fromfile(str(a.cpp_prefix)+'.'+path.name+'.output0.f32',np.float32).reshape(1,84,8400)
        raw=session.run(None,{session.get_inputs()[0].name:x})[0]
        raw_gpu=session.run(None,{session.get_inputs()[0].name:gpu})[0]
        ref=postprocess(raw,im.shape[:2],a.conf,a.iou);reference[path.name]=ref
        cpp_raw_ref=postprocess(trt,im.shape[:2],a.conf,a.iou)
        xr=a.output/path.name
        np.save(str(xr)+'.input.npy',x);np.save(str(xr)+'.onnx.npy',raw)
        np.save(str(xr)+'.onnx_on_gpu_input.npy',raw_gpu)
        r={'image':path.name,'shape':list(im.shape),'preprocessing':differences(gpu,x),
           'raw_output':{
             'preprocessing_effect_same_onnx_backend':{'boxes':differences(raw_gpu[:,:4],raw[:,:4]),'scores':differences(raw_gpu[:,4:],raw[:,4:])},
             'trt_vs_onnx_same_gpu_input':{'boxes':differences(trt[:,:4],raw_gpu[:,:4]),'scores':differences(trt[:,4:],raw_gpu[:,4:])}},
           'final_cpp_vs_onnx':match(cpp.get(path.name,[]),ref),
           'postprocess_cpp_vs_python_on_identical_trt_output':match(cpp.get(path.name,[]),cpp_raw_ref)}
        same=r['postprocess_cpp_vs_python_on_identical_trt_output']
        r['postprocess_identical_output_pass']=(not same['unmatched_actual'] and not same['unmatched_reference'] and
             same['max_coordinate_abs_delta']<.0005 and same['max_score_abs_delta']<1e-7)
        # A single uint8 code value is a declared bound, not tuned against detector results.
        r['preprocessing_one_uint8_level_bound_pass']=r['preprocessing']['max_abs']<=1/255+1e-6
        if pt is not None:
            with torch.inference_mode(): pt_raw=pt(torch.from_numpy(x))[0].numpy()
            np.save(str(xr)+'.pt.npy',pt_raw)
            r['pt_cpu_fp32_vs_onnx_cpu_fp32']=match(postprocess(pt_raw,im.shape[:2],a.conf,a.iou),ref)
        cpu_path=Path(str(a.cpp_prefix)+'.'+path.name+'.opencv.input.f32')
        if cpu_path.exists():
            cpp_cpu=np.fromfile(cpu_path,np.float32).reshape(1,3,640,640)
            trt_cpu=np.fromfile(str(a.cpp_prefix)+'.'+path.name+'.opencv.output0.f32',np.float32).reshape(1,84,8400)
            r['opencv_cpp_vs_python_input']=differences(cpp_cpu,x)
            r['cuda_vs_opencv_cpp_input']=differences(gpu,cpp_cpu)
            r['trt_preprocessing_effect_same_engine']={'boxes':differences(trt[:,:4],trt_cpu[:,:4]),'scores':differences(trt[:,4:],trt_cpu[:,4:])}
            r['trt_cpp_opencv_input_vs_onnx_reference']=match(postprocess(trt_cpu,im.shape[:2],a.conf,a.iou),ref)
        records.append(r)
    if not records: raise RuntimeError('No samples checked')
    metadata={'primary_reference':{'model':str(a.onnx),'backend':'ONNX Runtime CPUExecutionProvider FP32',
               'version':ort.__version__,'providers':session.get_providers()},
              'secondary_reference':None if pt is None else {'model':str(a.pt),'backend':'PyTorch CPU FP32 fused eval','version':torch.__version__},
              'preprocessing':{'imgsz':640,'batch':1,'auto':False,'scale_fill':False,'scaleup':True,'center':True,
               'interpolation':'cv2.INTER_LINEAR uint8','padding':114,'color':'BGR->RGB','layout':'NCHW','normalization':'float32/255',
               'opencv_python':cv2.__version__,'ultralytics':ultralytics.__version__},
              'postprocessing':{'conf':a.conf,'iou':a.iou,'class_aware':True,'multi_label':False,'max_det':300,'max_nms':30000,
               'coordinate_inverse':'installed ultralytics scale_boxes: separately rounded resize gain_x/gain_y; clip after NMS', 'class_names':canonical_names,'cpp_label_aliases':aliases},
              'scope':'Small-image consistency check, not complete mAP validation',
              'structural_pass':all(r['preprocessing_one_uint8_level_bound_pass'] and r['postprocess_identical_output_pass']
                  and not r['final_cpp_vs_onnx']['unmatched_actual'] and not r['final_cpp_vs_onnx']['unmatched_reference'] for r in records),
              'images':records}
    (a.output/'reference.detections.json').write_text(json.dumps(reference,indent=2)+'\n')
    (a.output/'comparison.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for r in records:
        c=r['final_cpp_vs_onnx'];print(r['image'],'pre max',r['preprocessing']['max_abs'],'detections',c['actual_count'],c['reference_count'],
         'coord max',c['max_coordinate_abs_delta'],'score max',c['max_score_abs_delta'],'same-raw-postprocess',r['postprocess_identical_output_pass'])

    if a.require_structural_pass and not metadata['structural_pass']:
        raise RuntimeError('Structural correctness check failed; inspect comparison.json before benchmarking')

if __name__=='__main__':main()
