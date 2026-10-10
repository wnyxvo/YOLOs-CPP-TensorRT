#!/usr/bin/env python3
"""Download official COCO val2017 archives; verify ZIP CRC and all 5000 image IDs.
Official COCO links use HTTP. Record URLs, archive/annotation/image SHA256 and sizes.
Images and labels stay local; only the preparation manifest belongs in Git.
"""
import argparse, concurrent.futures, hashlib, json, shutil, time, zipfile
import requests
from pathlib import Path

URLS={'val2017.zip':'http://images.cocodataset.org/zips/val2017.zip',
      'annotations_trainval2017.zip':'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('datasets/coco2017'));p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);a.manifest.parent.mkdir(parents=True,exist_ok=True)
    def download(item):
        name,url=item;target=a.root/name
        if not target.exists():
            # Large HTTP connections can close early in this environment. Bounded
            # Range requests resume independently and verify exact Content-Range.
            head=requests.head(url,timeout=30);head.raise_for_status();total=int(head.headers['Content-Length'])
            chunk_size=8*1024*1024;parts=a.root/(name+'.parts');parts.mkdir(exist_ok=True)
            def fetch(index):
                first=index*chunk_size;last=min(total,first+chunk_size)-1;part=parts/str(index)
                if part.exists() and part.stat().st_size==last-first+1:return part
                for attempt in range(6):
                    try:
                        with requests.get(url,headers={'Range':f'bytes={first}-{last}'},stream=True,timeout=(15,30)) as response:
                            response.raise_for_status()
                            if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {first}-{last}/{total}':raise RuntimeError('Server did not honor exact Range')
                            with part.open('wb') as f:
                                for chunk in response.iter_content(1024*1024):f.write(chunk)
                        if part.stat().st_size!=last-first+1:raise RuntimeError('Truncated chunk')
                        print(f'{name}: chunk {index+1}/{(total+chunk_size-1)//chunk_size}',flush=True);return part
                    except Exception:
                        if attempt==5:raise
                        time.sleep(min(attempt+1,5))
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                ordered=list(pool.map(fetch,range((total+chunk_size-1)//chunk_size)))
            assembled=target.with_suffix('.zip.assembled')
            with assembled.open('wb') as out:
                for part in ordered:
                    with part.open('rb') as f:shutil.copyfileobj(f,out)
            if assembled.stat().st_size!=total:raise RuntimeError('Archive size differs')
            assembled.replace(target)
        with zipfile.ZipFile(target) as archive:
            bad=archive.testzip()
            if bad:raise RuntimeError('ZIP CRC failed: '+bad)
            members=archive.namelist() if name=='val2017.zip' else ['annotations/instances_val2017.json']
            for member in members:
                path=Path(member)
                if path.is_absolute() or '..' in path.parts:raise RuntimeError('Unsafe ZIP entry')
                if not (a.root/path).exists():archive.extract(member,a.root)
        return name,{'url':url,'bytes':target.stat().st_size,'sha256':sha(target),'zip_crc':'pass'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
        archives=dict(workers.map(download,URLS.items()))
    annotation=a.root/'annotations/instances_val2017.json';data=json.loads(annotation.read_text());images=[]
    for im in sorted(data['images'],key=lambda r:r['id']):
        path=a.root/'val2017'/im['file_name']
        if not path.is_file():raise RuntimeError('Missing image '+str(path))
        images.append({'id':im['id'],'file_name':im['file_name'],'width':im['width'],'height':im['height'],'bytes':path.stat().st_size,'sha256':sha(path)})
    if len(images)!=5000 or len({r['id'] for r in images})!=5000 or len(list((a.root/'val2017').glob('*.jpg')))!=5000:
        raise RuntimeError('Require full val2017: 5000 unique IDs and image files')
    record={'status':'prepared','dataset':'COCO val2017 full 5000 images','source_page':'https://cocodataset.org/#download',
            'root':str(a.root.resolve()),'transport':'official HTTP links; TLS hostname mismatch observed on images.cocodataset.org; no TLS verification bypass',
            'archives':archives,'annotation':{'path':str(annotation.resolve()),'sha256':sha(annotation),'images':len(images),'annotations':len(data['annotations']),'categories':len(data['categories'])},'images':images}
    a.manifest.write_text(json.dumps(record,indent=2)+'\n');print('Prepared full COCO val2017:',a.manifest)

if __name__=='__main__':main()
