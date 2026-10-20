"""事件測試的 SAM 3 比較組:先把 SAM 3 框存下來,再由 scripts/eval_events.py 讀取。

預先登記:docs/M7_事件測試_預先登記_20261020.md §4.1、§4.2
  抹布:近拍鏡頭(Aoutput1–3、Boutput1–2)每 1 秒取一幀,提示 "sponge"、"towel",信心 > 0.5
  垃圾桶:output0、Aoutput1 片段第一幀,提示 "trash can",取信心最高的框(信心 > 0.5)

輸出 results/m7/events/sam3/<事件 id>.json(本機):
  {"cloth": {片段檔名: {"<步>": [[x1,y1,x2,y2], ...]}}, "trash": {"output0": [x1,y1,x2,y2] 或 null, ...}}
  步 = 0.1 秒一步(與偵測快取 stride 3 對齊),取樣幀為 0、10、20…步

用法:
    python scripts/m7_event_sam3.py --split dev
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from transformers import Sam3Model, Sam3Processor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_events import CLIPS, CLOSE, SAM3_DIR, load_events               # noqa: E402

WEIGHTS = ROOT / "integrated/weights/sam3"
CLOTH_PROMPTS = ["sponge", "towel"]
THR = 0.5
EVERY = 30                  # 幀;30 fps → 1 秒


def detect(m, p, rgb, text):
    inp = p(images=rgb, text=text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = m(**inp)
    r = p.post_process_instance_segmentation(out, threshold=THR, mask_threshold=0.5,
                                             target_sizes=inp["original_sizes"].tolist())[0]
    return [([round(x, 1) for x in b.tolist()], float(s)) for b, s in zip(r["boxes"], r["scores"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()
    SAM3_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = Sam3Model.from_pretrained(WEIGHTS).to("cuda").eval()
    p = Sam3Processor.from_pretrained(WEIGHTS)
    events = load_events(args.split)
    todo = [e for e in events if not (SAM3_DIR / f"{e['id']}.json").exists()]
    print(f"{args.split}:{len(events)} 個事件,待跑 {len(todo)}", flush=True)
    for n, e in enumerate(todo, 1):
        res = {"cloth": {}, "trash": {}}
        for v in CLOSE["A"] + CLOSE["B"]:
            stem = e["clips"][v]
            cap = cv2.VideoCapture(str(CLIPS / f"{stem}.mp4"))
            boxes, fid = {}, 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if fid % EVERY == 0:
                    rgb = f[:, :, ::-1].copy()
                    bb = [b for t in CLOTH_PROMPTS for b, _ in detect(m, p, rgb, t)]
                    if bb:
                        boxes[str(fid // 3)] = bb
                fid += 1
            cap.release()
            res["cloth"][stem] = boxes
        for v in ("output0", "Aoutput1"):
            cap = cv2.VideoCapture(str(CLIPS / f"{e['clips'][v]}.mp4"))
            ok, f = cap.read()
            cap.release()
            hits = detect(m, p, f[:, :, ::-1].copy(), "trash can") if ok else []
            res["trash"][v] = max(hits, key=lambda h: h[1])[0] if hits else None
        (SAM3_DIR / f"{e['id']}.json").write_text(json.dumps(res), encoding="utf-8")
        if n % 5 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
