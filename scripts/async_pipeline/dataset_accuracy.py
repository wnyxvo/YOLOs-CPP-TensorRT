#!/usr/bin/env python3
"""CPU/synchronous gpu-filter/async2 parity and COCO bbox AP on a labeled dataset.
Detection conf=.25/IoU=.45/max_det=300 is retained; COCOeval uses standard [1,10,100].
No downloads, pseudo-labels, or automatic fallback to unlabeled consistency samples.
"""
import argparse,csv,hashlib,json,subprocess,sys,importlib.metadata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baseline'))
from reference import match

MODES=('cpu','gpu-filter','async2')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('Output must be empty')
    a.output.mkdir(parents=True,exist_ok=True)
    if not a.annotations.is_file() or not a.images.is_dir():
        (a.output/'status.json').write_text(json.dumps({'status':'blocked_missing_dataset','metrics':None,'annotations':str(a.annotations),'images':str(a.images)},indent=2)+'\n')
        raise RuntimeError('Images or ground-truth annotations missing; no mAP was generated')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco=COCO(str(a.annotations));ids=sorted(coco.getImgIds())
    if a.limit:
        if a.limit<1:raise ValueError('limit must be positive')
        ids=ids[:a.limit]
    cats=sorted(coco.getCatIds());names=Path('models/coco.names').read_text().splitlines()
    aliases={'motorbike':'motorcycle','aeroplane':'airplane','sofa':'couch','pottedplant':'potted plant','diningtable':'dining table','tvmonitor':'tv'}
    if len(cats)!=80 or [coco.cats[c]['name'] for c in cats]!=[aliases.get(n,n) for n in names]:
        raise ValueError('Dataset categories do not match canonical COCO IDs/names')
    paths=[a.images/coco.imgs[i]['file_name'] for i in ids]
    if not ids or any(not path.is_file() for path in paths):raise ValueError('Incomplete image set; no subset is silently skipped')
    if any('\t' in str(path) or '\n' in str(path) for path in paths):raise ValueError('Manifest paths cannot contain TAB/newline')
    manifest=a.output/'images.tsv';manifest.write_text(''.join(f'{i}\t{path.resolve()}\n' for i,path in zip(ids,paths)))
    max_bytes=max(coco.imgs[i]['width']*coco.imgs[i]['height']*3 for i in ids)
    identity={str(path):sha(path) for path in [Path('models/yolo11n.trt'),Path('models/coco.names'),Path('build/dataset_accuracy_detection'),Path('include/yolos/tasks/async_detection.hpp'),Path('src/dataset_accuracy_detection.cpp'),Path(__file__)]}
    all_detections={};metrics={};commands=[];frame_counts={}
    for mode in MODES:
        csv_path=a.output/(mode+'.csv')
        argv=['build/dataset_accuracy_detection','models/yolo11n.trt','models/coco.names',str(manifest),str(csv_path),mode,str(max_bytes)]
        with (a.output/(mode+'.log')).open('w') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'argv':argv,'returncode':r.returncode})
        (a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if r.returncode:raise RuntimeError('Inference failed; inspect log')
        with Path(str(csv_path)+'.frames.csv').open() as f:frames=list(csv.DictReader(f))
        if [int(r['image_id']) for r in frames]!=ids:raise RuntimeError('Lost/duplicate/reordered dataset frame IDs')
        for frame in frames:
            i=int(frame['image_id']);size=[int(frame['width']),int(frame['height'])];declared=[coco.imgs[i]['width'],coco.imgs[i]['height']]
            if size!=declared:raise RuntimeError('Decoded dimensions differ from annotations')
        frame_counts[mode]=len(frames)
        detections={i:[] for i in ids};coco_results=[]
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                image_id=int(row['image_id']);class_id=int(row['class_id']);score=float(row['confidence'])
                if image_id not in detections or not 0<=class_id<80:raise RuntimeError('Invalid result image/category ID')
                xyxy=[float(row[k]) for k in ('x1','y1','x2','y2')]
                detections[image_id].append({'class_id':class_id,'confidence':score,'xyxy':xyxy})
                coco_results.append({'image_id':image_id,'category_id':cats[class_id],'score':score,'bbox':[xyxy[0],xyxy[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]]})
        (a.output/(mode+'.coco.json')).write_text(json.dumps(coco_results)+'\n')
        if any(len(detections[int(frame['image_id'])])!=int(frame['result_count']) for frame in frames):raise RuntimeError('Per-frame output count differs')
        # COCO.loadRes cannot handle an empty list; create its equivalent empty DT.
        if coco_results:dt=coco.loadRes(coco_results)
        else:
            dt=COCO();dt.dataset={'images':[coco.imgs[i] for i in ids],'categories':[coco.cats[c] for c in cats],'annotations':[]};dt.createIndex()
        evaluator=COCOeval(coco,dt,'bbox');evaluator.params.imgIds=ids
        evaluator.params.maxDets=[1,10,100];evaluator.params.useCats=1
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
        metrics[mode]=dict(zip(('AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large'),map(float,evaluator.stats)))
        all_detections[mode]=detections
    comparisons=[];passed=True;matched=0
    for mode in MODES[1:]:
        for image_id in ids:
            result=match(all_detections[mode][image_id],all_detections['cpu'][image_id],.99999)
            exact=not result['unmatched_actual'] and not result['unmatched_reference'] and result['max_coordinate_abs_delta']==0 and result['max_score_abs_delta']==0
            passed=passed and exact;matched+=len(result['matched'])
            # All original boxes/scores are in CSV; avoid duplicating them three times.
            item={k:v for k,v in result.items() if k!='matched'}
            item.update(image_id=image_id,mode=mode,matched_count=len(result['matched']),exact=exact,
                        min_matched_iou=min((p['iou'] for p in result['matched']),default=None))
            comparisons.append(item)
            if not exact:
                (a.output/f'mismatch_{mode}_{image_id}.json').write_text(json.dumps(result,indent=2)+'\n')
    if any(sha(Path(path))!=digest for path,digest in identity.items()):raise RuntimeError('Engine/labels/source/executable changed during evaluation')
    record={'status':'executed','same_engine_sha256':sha(Path('models/yolo11n.trt')),
            'annotation_sha256':hashlib.sha256(a.annotations.read_bytes()).hexdigest(),'images':len(ids),'dataset_total_images':len(coco.getImgIds()),
            'scope':'full' if len(ids)==len(coco.getImgIds()) else 'explicit subset','conf':.25,'iou':.45,'max_det':300,
            'imgsz':[640,640],'batch':1,'class_aware':True,'multi_label':False,'max_candidates':8400,'top_k':None,
            'coco_eval_max_dets':[1,10,100],'coco_eval_iou_thresholds':list(map(float,evaluator.params.iouThrs)),
            'coco_eval_recall_thresholds':list(map(float,evaluator.params.recThrs)),
            'graph':True,'graph_capture':'existing inference only','precision':'same existing FP16-builder Engine; FP32 I/O',
            'backend':'C++ TensorRT CUDA; CPU/gpu-filter/async2 are postprocessing/scheduling variants',
            'preprocessing':{'letterbox':'existing fused CUDA; same resize/pad','color':'BGR to RGB','normalization':'float32 / 255','layout':'NCHW'},
            'deployment_threshold_ap':True,'pycocotools':importlib.metadata.version('pycocotools'),
            'metrics':metrics,'exact_detection_parity':passed,'matched_pairs_across_two_comparisons':matched,
            'frame_counts':frame_counts,'commands':commands,'artifacts_sha256':identity,
            'frame_validation':'exact ordered image IDs, width/height and detection counts; no missing/duplicate frames',
            'prediction_sha256':{mode:sha(a.output/(mode+'.csv')) for mode in MODES}}
    (a.output/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    (a.output/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
    if not passed:raise RuntimeError('Dataset CPU/gpu-filter outputs differ')

if __name__=='__main__':main()
