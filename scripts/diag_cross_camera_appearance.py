"""實測:同一個人同時出現在兩台鏡頭上時,外觀向量匹配得起來嗎?"""
import sys, json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0,'src')
ROOT='D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA'
def phys(s): return s.split('_2023')[0].split('_2024')[0]
man=json.load(open('results/clips/clip_manifest.json',encoding='utf-8'))
L3=[c for c in man['clips'] if c['rule']=='L3S']
from m5_reid.dino_embedder import DINOv2Embedder
emb=DINOv2Embedder()

same_cam, cross_cam = [], []
n_sync = 0
for clip in L3:
    seq=clip['seq']; cams=clip['cameras']; gid=clip['gt_id']
    gt={}
    for f in sorted((Path(ROOT)/'annotations'/seq).glob('*.json')):
        cam=phys(f.stem)
        if cam not in cams: continue
        per=gt.setdefault(cam,{})
        for fr,dets in json.loads(f.read_text(encoding='utf-8')).items():
            for o in dets:
                if abs(int(o['id']))==gid: per[int(fr)]=tuple(map(float,o['BboxP']))
    # 兩台鏡頭「同時」看到目標的幀
    sync=sorted(set.intersection(*[set(gt.get(c,{})) for c in cams])) if all(c in gt for c in cams) else []
    sync=[f for f in sync if (f-1)%25==0][:6]       # 取樣避免太多
    if not sync: continue
    n_sync+=len(sync)
    feats=defaultdict(dict)
    for cam in cams:
        vids=sorted(glob.glob(f'{ROOT}/clips_singal_person_result/{seq}/{cam}_*.avi'))
        if not vids: continue
        cap=cv2.VideoCapture(vids[0]); fid=0; want=set(sync)
        while want:
            if not cap.grab(): break
            fid+=1
            if fid not in want: continue
            want.discard(fid)
            ok,img=cap.retrieve()
            if not ok: continue
            bb=gt[cam][fid]
            x1,y1,x2,y2=(int(max(0,bb[0])),int(max(0,bb[1])),int(min(img.shape[1],bb[2])),int(min(img.shape[0],bb[3])))
            if x2-x1<8 or y2-y1<16: continue
            feats[cam][fid]=emb.extract(img[y1:y2,x1:x2])
        cap.release()
    # 跨鏡頭:同一幀、同一人、兩台
    for fid in sync:
        if all(fid in feats[c] for c in cams[:2]):
            cross_cam.append(float(feats[cams[0]][fid] @ feats[cams[1]][fid]))
    # 同鏡頭:同一台、相鄰取樣
    for cam in cams:
        fs=sorted(feats[cam])
        for a,b in zip(fs,fs[1:]):
            same_cam.append(float(feats[cam][a] @ feats[cam][b]))

print(f'L3S 片段裡「兩台鏡頭同時看到同一個人」的取樣幀:{n_sync} 個\n')
for lab,v in [('同一人 · 同一台鏡頭(相鄰取樣)',same_cam),
              ('同一人 · 兩台鏡頭同時',cross_cam)]:
    if not v: print(f'  {lab}:無樣本'); continue
    v=np.asarray(v)
    print(f'  {lab}')
    print(f'     N={len(v)}  cos 中位 {np.median(v):+.4f}  [p10 {np.percentile(v,10):+.4f}, p90 {np.percentile(v,90):+.4f}]')
    print(f'     → app = 0.5x(1+cos) 中位 {0.5*(1+np.median(v)):.4f}')
print()
print('對照(先前在單一鏡頭上量到的):')
print('   同一人 相鄰格   cos +0.977   app 0.989')
print('   不同人 同一幀   cos +0.440   app 0.720')
