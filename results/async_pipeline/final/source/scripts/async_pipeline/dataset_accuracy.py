#!/usr/bin/env python3
"""CPU/gpu-filter parity and COCO bbox AP on an explicitly supplied labeled dataset.
Detection conf=.25/IoU=.45/max_det=300 is retained; COCOeval uses standard [1,10,100].
No downloads, pseudo-labels, or automatic fallback to unlabeled consistency samples.
"""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'baseline'))
from reference import match


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
    all_detections={};metrics={};commands=[]
    for mode in ('cpu','gpu-filter'):
        csv_path=a.output/(mode+'.csv')
        argv=['build/dataset_accuracy_detection','models/yolo11n.trt','models/coco.names',str(manifest),str(csv_path),mode]
        with (a.output/(mode+'.log')).open('w') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'argv':argv,'returncode':r.returncode})
        if r.returncode:raise RuntimeError('Inference failed; inspect log')
        detections={i:[] for i in ids};coco_results=[]
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                image_id=int(row['image_id']);class_id=int(row['class_id']);score=float(row['confidence'])
                xyxy=[float(row[k]) for k in ('x1','y1','x2','y2')]
                detections[image_id].append({'class_id':class_id,'confidence':score,'xyxy':xyxy})
                coco_results.append({'image_id':image_id,'category_id':cats[class_id],'score':score,'bbox':[xyxy[0],xyxy[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]]})
        (a.output/(mode+'.coco.json')).write_text(json.dumps(coco_results)+'\n')
        # COCO.loadRes cannot handle an empty list; create its equivalent empty DT.
        if coco_results:dt=coco.loadRes(coco_results)
        else:
            dt=COCO();dt.dataset={'images':[coco.imgs[i] for i in ids],'categories':[coco.cats[c] for c in cats],'annotations':[]};dt.createIndex()
        evaluator=COCOeval(coco,dt,'bbox');evaluator.params.imgIds=ids
        evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
        metrics[mode]=dict(zip(('AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large'),map(float,evaluator.stats)))
        all_detections[mode]=detections
    comparisons=[]
    for image_id in ids:
        result=match(all_detections['gpu-filter'][image_id],all_detections['cpu'][image_id],.99999)
        result['image_id']=image_id;comparisons.append(result)
    passed=all(not r['unmatched_actual'] and not r['unmatched_reference'] and r['max_coordinate_abs_delta']==0 and r['max_score_abs_delta']==0 for r in comparisons)
    record={'status':'executed','same_engine_sha256':hashlib.sha256(Path('models/yolo11n.trt').read_bytes()).hexdigest(),
            'annotation_sha256':hashlib.sha256(a.annotations.read_bytes()).hexdigest(),'images':len(ids),'dataset_total_images':len(coco.getImgIds()),
            'scope':'full' if len(ids)==len(coco.getImgIds()) else 'explicit subset','conf':.25,'iou':.45,'max_det':300,
            'coco_eval_max_dets':[1,10,100],'graph':True,'metrics':metrics,'exact_detection_parity':passed,'commands':commands}
    (a.output/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    (a.output/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
    if not passed:raise RuntimeError('Dataset CPU/gpu-filter outputs differ')

if __name__=='__main__':main()
