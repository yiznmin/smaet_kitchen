"""M6 多鏡頭觸發前置:用 SAM 3 在 9 台鏡頭各自找洗手台、垃圾桶,提案 zone 給使用者確認。

由 scripts/m6_sink_zone_proposals.py 擴充(多類別、9 台)。

做法同垃圾桶 zone(docs/M7_事件測試_預先登記_20261020.md 前情):每個場次取一支片段的第一幀,
SAM 3 文字提示(洗手台 "sink"、垃圾桶區 "trash can";信心 > 0.4),各鏡頭把所有場次的框疊起來,提案 = 框的中位數外擴 5%。
輸出 results/m6/multiview/zones/(本機;EPFL 影像不進 git):
  proposals.json            每個框(場次、鏡頭、信心)與各鏡頭提案矩形
  <區域>_<鏡頭>_overlay.jpg        各場次框(細線)+ 提案(粗黃框)畫在一張代表畫面上
  <區域>_<鏡頭>_sessions.jpg       各場次第一幀縮圖 + 提案,檢查鏡頭有沒有移動

用法:
    python scripts/m6_zone_proposals_multiview.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor

ROOT = Path(__file__).resolve().parent.parent
CLIPS = Path("D:/yizhen/event_clips")
WEIGHTS = ROOT / "integrated/weights/sam3"
OUT = ROOT / "results/m6/multiview/zones"
VIEWS = ["output0", "Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3", "Boutput0", "Boutput1", "Boutput2", "Boutput3"]
PROMPTS = {"洗手台": "sink", "垃圾桶區": "trash can"}
TAG = {"洗手台": "sink", "垃圾桶區": "trash"}               # 檔名與圖上文字用(cv2 不能畫中文)
THR, PAD = 0.4, 0.05


def first_frame(path):
    cap = cv2.VideoCapture(str(path))
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = Sam3Model.from_pretrained(WEIGHTS).to("cuda").eval()
    p = Sam3Processor.from_pretrained(WEIGHTS)
    clips = {}
    for r in csv.DictReader(open(CLIPS / "clips_index.csv", encoding="utf-8-sig")):
        if r["view"] in VIEWS:
            clips.setdefault((r["session_id"], r["view"]), r["clip_file"])
    boxes, frames = [], {}
    for (sess, v), fn in sorted(clips.items()):
        f = first_frame(CLIPS / fn)
        if f is None:
            continue
        frames.setdefault(v, []).append((sess, f))
        for zname, text in PROMPTS.items():
            inp = p(images=f[:, :, ::-1].copy(), text=text, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = m(**inp)
            rr = p.post_process_instance_segmentation(out, threshold=THR, mask_threshold=0.5,
                                                      target_sizes=inp["original_sizes"].tolist())[0]
            for sc, b in zip(rr["scores"], rr["boxes"]):
                boxes.append(dict(zone=zname, session=sess, view=v, score=round(float(sc), 3),
                                  box=[round(x) for x in b.tolist()]))
    prop = {}
    for zname in PROMPTS:
        tag = TAG[zname]
        for v in VIEWS:
            sel = [b for b in boxes if b["view"] == v and b["zone"] == zname]
            n_sess = len(frames.get(v, []))
            if not sel:
                print(f"{zname} {v}:沒有找到")
                continue
            # 每個場次只取信心最高的框,避免同場次多框拉偏中位數
            best = {}
            for b in sel:
                if b["session"] not in best or b["score"] > best[b["session"]]["score"]:
                    best[b["session"]] = b
            bb = np.array([b["box"] for b in best.values()], float)
            med = np.median(bb, 0)
            w, h = med[2] - med[0], med[3] - med[1]
            rect = [int(med[0] - PAD * w), int(med[1] - PAD * h), int(med[2] + PAD * w), int(med[3] + PAD * h)]
            hit = len(best)
            prop.setdefault(zname, {})[v] = dict(rect=rect, n_boxes=len(sel), sessions_with_hit=hit, sessions=n_sess)
            print(f"{zname} {v}:{hit}/{n_sess} 個場次找到,提案 {rect}")
            base = frames[v][0][1].copy()
            for b in sel:
                cv2.rectangle(base, tuple(b["box"][:2]), tuple(b["box"][2:]), (0, 200, 255), 1)
            cv2.rectangle(base, tuple(rect[:2]), tuple(rect[2:]), (0, 255, 255), 4)
            cv2.putText(base, f"{v} {tag}: {hit}/{n_sess} sessions", (10, 40), 0, 1.2, (0, 255, 255), 3)
            cv2.imencode(".jpg", base)[1].tofile(str(OUT / f"{tag}_{v}_overlay.jpg"))
            cells = []
            for sess, f in frames[v]:
                g = f.copy()
                cv2.rectangle(g, tuple(rect[:2]), tuple(rect[2:]), (0, 255, 255), 4)
                cv2.putText(g, sess.split("/")[1] + " " + sess.split("/")[2][:10], (10, 50), 0, 1.5, (0, 255, 255), 3)
                cells.append(cv2.resize(g, (320, 180)))
            while len(cells) % 6:
                cells.append(np.zeros_like(cells[0]))
            grid = np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)])
            cv2.imencode(".jpg", grid)[1].tofile(str(OUT / f"{tag}_{v}_sessions.jpg"))
    (OUT / "proposals.json").write_text(json.dumps(dict(boxes=boxes, proposals=prop), ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"輸出 {OUT}")


if __name__ == "__main__":
    main()
