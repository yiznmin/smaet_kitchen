import sys, glob
from collections import defaultdict, Counter
from pathlib import Path
import cv2
sys.path.insert(0, "scripts")
from diag_m4_ghosts import load_rows, per_frame_assign, features, CATS
from eval_m4m5_chirla import load_gt, match_tracks
R = "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
SP = sys.argv[1]
SEQS = ["seq_004", "seq_006", "seq_007", "seq_020", "seq_024", "seq_025", "seq_026"]
for cam in ("camera_5", "camera_6"):
    best = None
    for seq in SEQS:
        gt = load_gt(R, seq)
        rows = [r for r in load_rows(f"results/m5_step3/m5_cbiou/m4dump/{seq}/tracks.csv")]
        tg, *_ = match_tracks(gt, rows)
        info, _ = per_frame_assign(gt, rows)
        by = defaultdict(list)
        for r in rows:
            by[(r["cam"], r["tid"])].append(r)
        one = set()
        for k, rs in by.items():
            if k[0] != cam or tg[k] is not None:
                continue
            f = features(sorted(info[k]), [r["bbox"] for r in rs], [r["conf"] for r in rs])
            if f["cat"] == CATS[0]:
                one.add(k)
        cnt = Counter(r["fid"] for r in rows if (r["cam"], r["tid"]) in one)
        if cnt:
            fid, n = cnt.most_common(1)[0]
            if best is None or n > best[2]:
                best = (seq, fid, n, [r for r in rows if r["cam"] == cam and r["fid"] == fid], one, gt)
    seq, fid, n, rs, one, gt = best
    vid = glob.glob(f"{R}/videos/{seq}/{cam}_*.avi")[0]
    sys.path.insert(0, "src")
    from common.video_io import iter_frames
    img = None
    for f, _t, frame in iter_frames(vid):
        if f == fid:
            img = frame
            break
    for _g, b in gt[cam].get(fid + 1, []):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    for r in rs:
        b = r["bbox"]
        col = (0, 0, 255) if (cam, r["tid"]) in one else (255, 160, 0)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        cv2.putText(img, f"{r['conf']:.2f}", (int(b[0]), max(12, int(b[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    out = f"{SP}/ghost_{cam}_{seq}_{fid}.png"
    cv2.imwrite(out, img)
    print(cam, seq, "fid", fid, "① boxes", n, "->", out)
