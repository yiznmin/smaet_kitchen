"""事件測試第 3 輪(吃東西):Qwen3-VL-8B 看片段回答是否把東西放進嘴裡。

預先登記:docs/M7_事件測試_預先登記_20261020.md §9.7
每 0.5 秒取一幀(最多 32 幀)、短邊縮到 360;分數 = 第一個輸出 token 的 P(yes) / (P(yes) + P(no))。
輸出 results/m7/events/vlm/<事件 id>.json:{"Aoutput1": 分數, "output0": 分數}

用法:
    python scripts/m7_event_vlm.py --split dev
    python scripts/m7_event_vlm.py --split dev --limit 4      # 冒煙
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_events import CLIPS, load_events                                  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
OUT = ROOT / "results/m7/events/vlm"
VIEWS = ["Aoutput1", "output0"]
PROMPT = ("Watch this kitchen video. Does the person put food or anything into their own mouth "
          "(eating or tasting)? Answer only yes or no.")
EVERY_S, MAX_FRAMES, SHORT = 0.5, 32, 360


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(round(fps * EVERY_S))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            s = SHORT / min(h, w)
            frames.append(cv2.resize(f, (int(w * s) // 32 * 32, int(h * s) // 32 * 32))[:, :, ::-1])
        i += 1
    cap.release()
    if len(frames) > MAX_FRAMES:
        frames = [frames[k] for k in np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)]
    if len(frames) % 2:                                  # Qwen 影片以 2 幀為一組
        frames.append(frames[-1])
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    events = load_events(args.split)
    todo = [e for e in events if not (OUT / f"{e['id']}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.split}:{len(events)} 個事件,待跑 {len(todo)}", flush=True)
    for n, e in enumerate(todo, 1):
        res = {}
        for v in VIEWS:
            fr = read_frames(CLIPS / f"{e['clips'][v]}.mp4")
            meta = dict(fps=1 / EVERY_S, total_num_frames=len(fr), frames_indices=list(range(len(fr))))
            inp = proc(text=[text], videos=[fr], video_metadata=[meta], do_sample_frames=False,
                       cap_pixels_per_frame=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits = model(**inp).logits[0, -1].float()
            p = torch.softmax(logits[yes + no], 0)
            res[v] = round(float(p[:len(yes)].sum()), 4)
        (OUT / f"{e['id']}.json").write_text(json.dumps(res), encoding="utf-8")
        if args.limit or n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} {e['cat']} {res}  {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
