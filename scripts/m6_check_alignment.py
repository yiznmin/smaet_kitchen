"""M6 第 1 步:驗 EPFL 動作標註的時間與影片幀是否對齊(幀號 = 秒 × fps)。

取幾段標註的起、中、訖三個時間點,從數個鏡頭各取一幀拼成對照圖,人工目視確認動作吻合。
不吻合就停,先查 meta_data/timestamps.txt。

⚠ EPFL 是 CC-BY-NC,輸出的對照圖只供內部檢查,不進 git(results/ 已被 .gitignore 擋掉)。

用法:
    python scripts/m6_check_alignment.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import imwrite_unicode, video_meta   # noqa: E402

SESSION = ROOT / "data" / "epfl_ann" / "YH2040" / "2023_09_12_09_50_41"
# 要檢查的標註段:(動詞, 名詞, 第幾段符合的)
DEFAULT_SEGS = [("Wash", "Hand", 0), ("Grab", "Knife", 0), ("Cut", "Tomatoes", 0)]


def load_actions(xlsx):
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:] if r[0] is not None]


def read_frame(video, fid):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"讀不到 {video} 的第 {fid} 幀")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data" / "epfl"))
    ap.add_argument("--cams", nargs="+", default=["Boutput0", "Aoutput1", "output0"])
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "m6" / "alignment"))
    ap.add_argument("--width", type=int, default=480, help="每格縮放後的寬度")
    args = ap.parse_args()

    acts = load_actions(SESSION / "annotations" / "actions_annotations.xlsx")
    out_dir = Path(args.out_dir)
    for verb, noun, k in DEFAULT_SEGS:
        hits = [a for a in acts if a["Verbs"] == verb and a["Nouns"] == noun]
        seg = hits[k]
        t0, t1 = float(seg["Start"]), float(seg["End"])
        times = [t0, (t0 + t1) / 2, t1]
        rows = []
        for cam in args.cams:
            video = Path(args.videos_dir) / f"{cam}.mp4"
            fps = video_meta(video)["fps"]
            cells = []
            for t in times:
                fid = int(round(t * fps))
                img = read_frame(video, fid)
                h = int(img.shape[0] * args.width / img.shape[1])
                img = cv2.resize(img, (args.width, h))
                cv2.putText(img, f"{cam} t={t:.2f}s f={fid}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cells.append(img)
            rows.append(np.hstack(cells))
        grid = np.vstack(rows)
        bar = np.zeros((36, grid.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{verb} {noun}  {t0:.2f}-{t1:.2f}s  (start / mid / end)", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        path = out_dir / f"{verb}_{noun}_{t0:.2f}.jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(path, np.vstack([bar, grid]))
        print(f"{verb} {noun} {t0:.2f}-{t1:.2f}s -> {path}")


if __name__ == "__main__":
    main()
