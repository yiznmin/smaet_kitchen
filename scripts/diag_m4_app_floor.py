"""診斷:外觀項的數值落點是否把 0.2 的關聯門檻廢掉了。

只讀真值標註與影片,不跑追蹤、不改任何檔案。
"""
import sys, json, glob, itertools, random
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
ROOT = 'D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA'
VDIR = 'clips_singal_person_result'
SEQS = ['seq_000', 'seq_001', 'seq_002']
CAMS = ['camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_7']
STRIDE = 5
random.seed(0)

def phys(stem): return stem.split('_2023')[0].split('_2024')[0]

def load_gt(seq):
    out = {}
    for f in sorted((Path(ROOT)/'annotations'/seq).glob('*.json')):
        cam = phys(f.stem); per = out.setdefault(cam, defaultdict(list))
        for fr, dets in json.loads(f.read_text(encoding='utf-8')).items():
            for o in dets:
                per[int(fr)].append((abs(int(o['id'])), tuple(map(float, o['BboxP']))))
    return out

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1); ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0.0,ix2-ix1), max(0.0,iy2-iy1); inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

# ── 蒐集裁圖:每個 (seq, cam) 取樣格點,存 (gt_id, fid, crop) ──
N_PER_UNIT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
samples = []   # (seq, cam, gt_id, fid, crop, bbox)
for seq in SEQS:
    gt = load_gt(seq)
    for cam in CAMS:
        if cam not in gt: continue
        vids = sorted(glob.glob(f'{ROOT}/{VDIR}/{seq}/{cam}_*.avi'))
        if not vids: continue
        frames = sorted(f for f in gt[cam] if (f-1) % STRIDE == 0 and gt[cam][f])
        if not frames: continue
        # 取連續區塊(不是均勻散佈),才拿得到「同一人相鄰取樣格」的配對
        if len(frames) <= N_PER_UNIT:
            pick = frames
        else:
            n_blk = 8; blk = max(2, N_PER_UNIT // n_blk)
            starts = np.linspace(0, len(frames) - blk, n_blk).astype(int)
            pick = sorted({frames[i] for s0 in starts for i in range(s0, s0 + blk)})
        want = set(pick)
        cap = cv2.VideoCapture(vids[0]); fid = 0
        while True:
            ok = cap.grab()
            if not ok: break
            fid += 1
            if fid not in want: continue
            ok, img = cap.retrieve()
            if not ok: continue
            h, w = img.shape[:2]
            for gid, bb in gt[cam][fid]:
                x1,y1,x2,y2 = (int(max(0,bb[0])), int(max(0,bb[1])),
                               int(min(w,bb[2])), int(min(h,bb[3])))
                if x2-x1 < 8 or y2-y1 < 16: continue
                samples.append((seq, cam, gid, fid, img[y1:y2, x1:x2].copy(), bb))
        cap.release()
print(f'裁圖數 {len(samples)}  (單元取樣 {N_PER_UNIT} 格)', flush=True)

def run(model_name):
    if model_name == 'dinov2':
        from m5_reid.dino_embedder import DINOv2Embedder; emb = DINOv2Embedder()
    else:
        from m5_reid.chirla_embedder import ChirlaEmbedder; emb = ChirlaEmbedder(model_name)
    feats = []
    B = 32
    for i in range(0, len(samples), B):
        feats.append(emb.extract_batch([s[4] for s in samples[i:i+B]]))
    F = np.concatenate(feats)
    print(f'[{model_name}] 特徵 {F.shape}', flush=True)

    # 分組
    by_unit_frame = defaultdict(list)     # (seq,cam,fid) -> [idx]
    by_unit_id    = defaultdict(list)     # (seq,cam,gid) -> [(fid,idx)]
    for i,(seq,cam,gid,fid,_,_) in enumerate(samples):
        by_unit_frame[(seq,cam,fid)].append(i)
        by_unit_id[(seq,cam,gid)].append((fid,i))

    same, diff_same_frame, diff_cross = [], [], []
    diff_same_frame_iou = []
    # (a) 同一人、相鄰取樣格(同鏡頭)
    for k,v in by_unit_id.items():
        v.sort()
        for (f1,i1),(f2,i2) in zip(v, v[1:]):
            if f2-f1 <= STRIDE*3:
                same.append(float(F[i1] @ F[i2]))
    # (b) 同幀不同人
    for k,v in by_unit_frame.items():
        for i1,i2 in itertools.combinations(v, 2):
            if samples[i1][2] != samples[i2][2]:
                diff_same_frame.append(float(F[i1] @ F[i2]))
                diff_same_frame_iou.append(iou(samples[i1][5], samples[i2][5]))
    # (c) 不同鏡頭/序列的不同人(隨機 4000 對)
    idx = list(range(len(samples)))
    for _ in range(4000):
        i1,i2 = random.sample(idx, 2)
        a,b = samples[i1], samples[i2]
        if (a[0],a[1]) != (b[0],b[1]) or a[2] != b[2]:
            diff_cross.append(float(F[i1] @ F[i2]))

    def stat(name, cs):
        cs = np.asarray(cs)
        if not len(cs): print(f'  {name}: 無樣本'); return None
        app = 0.5*(1+cs)
        print(f'  {name:<22} N={len(cs):<6} cos 中位 {np.median(cs):+.3f} '
              f'[p10 {np.percentile(cs,10):+.3f}, p90 {np.percentile(cs,90):+.3f}]  '
              f'→ app 中位 {np.median(app):.3f} [p10 {np.percentile(app,10):.3f}]')
        return app
    print(f'[{model_name}] app = 0.5*(1+cos) 的分佈:')
    a_same = stat('同一人相鄰格', same)
    a_dsf  = stat('同幀不同人', diff_same_frame)
    a_dx   = stat('跨單元不同人', diff_cross)

    GATE = 0.2
    print(f'[{model_name}] 「外觀項單獨」能否越過 {GATE} 門檻(即框重疊 0 也能配上):')
    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
        for nm, a in (('同幀不同人', a_dsf), ('跨單元不同人', a_dx)):
            if a is None: continue
            share = float(np.mean(w*a >= GATE))
            print(f'  w={w:.1f}  {nm:<14} w*app 中位 {w*np.median(a):.3f}  '
                  f'越過門檻比例 {share*100:5.1f}%')
    if a_dsf is not None:
        print(f'[{model_name}] 同幀不同人的框重疊:中位 {np.median(diff_same_frame_iou):.3f}，'
              f'IoU<0.2 佔 {np.mean(np.asarray(diff_same_frame_iou)<0.2)*100:.1f}%')
    return dict(model=model_name,
                same=dict(n=len(same), cos_median=float(np.median(same)) if same else None),
                diff_same_frame=dict(n=len(diff_same_frame),
                    cos_median=float(np.median(diff_same_frame)) if diff_same_frame else None,
                    app_median=float(np.median(a_dsf)) if a_dsf is not None else None),
                diff_cross=dict(n=len(diff_cross),
                    cos_median=float(np.median(diff_cross)) if diff_cross else None,
                    app_median=float(np.median(a_dx)) if a_dx is not None else None),
                gate_cross={f'w={w}': float(np.mean(w*a_dsf >= 0.2)) for w in (0.1,0.2,0.3,0.4,0.5)}
                    if a_dsf is not None else {})

import os
MODELS = os.environ.get('MODELS','dinov2').split(',')
out = [run(m) for m in MODELS]
json.dump(out, open('results/m4_round4/diag_app_floor.json','w',encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('已存 results/m4_round4/diag_app_floor.json')
