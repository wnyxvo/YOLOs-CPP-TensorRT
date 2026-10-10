#!/usr/bin/env python3
"""Deterministic local smoke set; transformed dog image, not an mAP dataset."""
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np
import ultralytics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, default=Path('results/baseline/samples'))
a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
im = cv2.imread('data/dog.jpg')
if im is None: raise RuntimeError('Missing data/dog.jpg')
portrait = np.full((901,577,3),114,np.uint8)
portrait[234:667] = cv2.resize(im,(577,433))
wide = np.full((501,1203,3),114,np.uint8)
wide[:,267:935] = cv2.resize(im,(668,501))
tile = cv2.resize(im,(640,640))
mosaic = np.full((1281,1281,3),114,np.uint8)
for y in (0,641):
    for x in (0,641): mosaic[y:y+640,x:x+640] = tile
samples = {'dog.png':im, 'odd_landscape.png':cv2.resize(im,(1001,667)),
           'portrait.png':portrait, 'wide.png':wide, 'multi_mosaic.png':mosaic}
asset_dir = Path(ultralytics.__file__).parent/'assets'
sources = [{'path':'data/dog.jpg','sha256':hashlib.sha256(Path('data/dog.jpg').read_bytes()).hexdigest()}]
for name in ('bus','zidane'):
    source=asset_dir/(name+'.jpg')
    image=cv2.imread(str(source))
    if image is None: raise RuntimeError('Missing installed Ultralytics sample: '+str(source))
    samples[name+'.png']=image
    sources.append({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
meta = []
for name,img in samples.items():
    path=a.output/name
    if not cv2.imwrite(str(path), img): raise RuntimeError(path)
    meta.append({'image':name,'shape':list(img.shape),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(a.output/'manifest.json').write_text(json.dumps({'source':'data/dog.jpg','source_sha256':hashlib.sha256(Path('data/dog.jpg').read_bytes()).hexdigest(),
 'scope':'7 images from 3 local scenes: 5 deterministic dog transformations plus installed Ultralytics bus/zidane samples; not mAP', 'sources':sources,'samples':meta},indent=2)+'\n')
