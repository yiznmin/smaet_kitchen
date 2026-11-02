"""NG4 反例候選搜尋:洗好食材 → 放到砧板 / 容器(給人工檢視)。

流程:
  1. EPFL 動作標註「Wash <食材>」後 30 秒內第一個「Put <同一食材>」→ 一件候選(片段 = 清洗起點 → 放下終點)。
  2. M3(不重訓)在放下前後取幀,看「新出現、不在手上」的食材框落在砧板 / 容器上,還是桌面 → 自動初判。
  3. 9 視角都剪片,檔名與中繼檔欄位照團隊範例(D:/yizhen/event_clips_20261102)。

輸出 D:/yizhen/NG4_OK_candidates_20261102/(EPFL 衍生物,CC BY-NC 4.0 僅內部使用,不進 git):
  01_砧板或容器_候選反例/、02_桌面_可能NG4/、03_無法判斷/、ng_clips_index.csv、ng_instances.csv、人工檢視表.xlsx

用法:
    python scripts/mine_ng4_ok_candidates.py --check      # 只搜尋 + 初判,印召回率與初判準確度,不剪片
    python scripts/mine_ng4_ok_candidates.py              # 全部(已存在的片段跳過)
"""
import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "integrated/src"))
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from common.zones import get_zone                                         # noqa: E402

VID = Path("D:/yizhen/Public_release_videos")
ANN = ROOT / "data/epfl_ann"
EXAMPLE = Path("D:/yizhen/event_clips_20261102")
TEAM = Path("D:/yizhen/NG_clips/ng_instances.csv")
OUT = Path("D:/yizhen/NG4_OK_candidates_20261102")
WEIGHTS = ROOT / "integrated/weights/m3_nano_r2_armA/checkpoint_best_total.pth"
ZONES = ROOT / "integrated/configs/zones"
VIEWS = ["Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3", "Boutput0", "Boutput1", "Boutput2", "Boutput3", "output0"]
JUDGE_VIEWS = ["Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2"]
USER = "系統候選"
FPS = 30
MAX_GAP_S = 30.0                       # 清洗結束到放下開始的最長間隔
BEFORE_S = (-1.0, -0.5)                # 相對 Put 起點:手上拿著時(比對用)
AFTER_S = (0.3, 0.8, 1.3)              # 相對 Put 終點:放下後
THR_OBJ, THR_HAND, SUP_IOA, NEW_IOU = 0.4, 0.2, 0.3, 0.3
M3_TRAIN = ("YH2040/2023_09_12_09_50_41",)
# 非食材名詞(器具、手、清潔用品、環境);其餘名詞視為食材
NONFOOD = {"Hand", "Sponge", "Towel", "Knife", "Spatula", "Spoon", "Pot", "Pan", "Pot_lid", "Cutting_board", "Bowl",
           "Salad_bowl", "Plate", "Cup", "Colander", "Grater", "Peeler", "Whip", "Pasta_Spoon", "Doser_Glass", "Bottle",
           "Box", "Package", "Water", "nan", "Tissue", "Trash", "Recipe", "Button", "Cupboard", "Fridge", "Fork",
           "Glass", "Lid", "Tray", "Sink", "Tap", "Faucet", "Oven", "Stove", "Ladle", "Tongs", "Strainer", "Jar",
           "Can", "Bag", "Glove", "Gloves", "Container", "Scale", "Measuring_cup", "Brush", "Soap", "Phone", "Paper", "Foil"}
# EPFL 名詞 → 中文(檢視表用;沒有的保留英文)
ZH = {"Tomatoes": "番茄", "Zucchini": "櫛瓜", "Onions": "洋蔥", "Avocado": "酪梨", "Mushrooms": "蘑菇", "Eggplant": "茄子",
      "Bell_pepper": "甜椒", "Radish": "櫻桃蘿蔔", "Lemon": "檸檬", "Cucumber": "黃瓜", "Carrots": "胡蘿蔔",
      "Shallots": "紅蔥頭", "Fresh_thyme": "百里香", "Bean_sprouts": "豆芽", "Tofu": "豆腐", "Rice": "米",
      "Eggs": "蛋", "Pasta": "義大利麵", "Noodles": "麵", "Surimi": "蟹肉棒", "Peanuts": "花生", "Potatoes": "馬鈴薯",
      "Garlic": "大蒜", "Basil": "羅勒", "Parsley": "巴西里", "Spring_onions": "青蔥", "Ginger": "薑", "Lime": "萊姆",
      "Apple": "蘋果", "Salad": "生菜", "Lettuce": "生菜", "Broccoli": "青花菜", "Cauliflower": "花椰菜"}
FOLDER = {"砧板或容器": ("01_砧板或容器_候選反例", "NG4-OK", "清洗後放到砧板或容器"),
          "桌面": ("02_桌面_可能NG4", "NG4", "清洗後放到桌面"),
          "無法判斷": ("03_無法判斷", "NG4-UNK", "無法判斷放置位置")}
INDEX_COLS = ["clip_file", "source_file", "view", "ng_type", "result", "ng_index", "reason", "cause", "effect", "vehicle",
              "session_time", "session_id", "annotated_view", "category", "note", "user", "annotation_id",
              "event_start_sec", "event_end_sec", "outside_event", "event_timeline", "detail_ids", "clip_start_sec",
              "clip_end_sec", "clip_start_hms", "clip_end_hms", "clip_duration_sec"]
INST_COLS = ["annotation_id", "session_id", "vehicle", "session_time", "annotated_view", "category", "note", "user",
             "ng_type", "result", "ng_index", "reason", "cause", "effect", "start_sec", "end_sec", "start_hms", "end_hms",
             "duration_sec", "event_start_sec", "event_end_sec", "outside_event", "event_timeline", "detail_ids"]


def hms(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def zh(noun):
    return ZH.get(noun.strip(), noun.strip())


def sessions():
    """[(session_id, 受試者, 場次時間)];同一受試者 / 場次在 test 與 train 都有時只留第一個。"""
    out, seen = [], set()
    for d in sorted(VID.glob("*/*/*")):
        split, veh, st = d.parts[-3:]
        if (veh, st) in seen:
            continue
        seen.add((veh, st))
        out.append((f"{split}/{veh}/{st}", veh, st))
    return out


def find_candidates():
    cands, no_ann = [], []
    for sid, veh, st in sessions():
        f = ANN / veh / st / "annotations/actions_annotations.xlsx"
        if not f.exists():
            no_ann.append(sid)
            continue
        d = pd.read_excel(f).sort_values("Start").reset_index(drop=True)
        d["Verbs"] = d["Verbs"].astype(str).str.strip()
        d["Nouns"] = d["Nouns"].astype(str).str.strip()
        washes = [(w.Nouns, float(w.Start), float(w.End), "Wash")
                  for _, w in d[d.Verbs == "Wash"].iterrows() if w.Nouns not in NONFOOD]
        # 標註沒寫 Wash、但手拿著食材期間開水 → 關水,也視為清洗(例:YH2040 07_28 15:23 番茄)
        for _, o in d[(d.Verbs == "Open") & (d.Nouns == "Water")].iterrows():
            cl_ = d[(d.Verbs == "Close") & (d.Nouns == "Water") & (d.Start >= o.Start)]
            end = float(cl_.iloc[0].End) if len(cl_) else float(o.End)
            held = d[d.Verbs.isin(["Carry", "Hold"]) & ~d.Nouns.isin(NONFOOD) & (d.Start <= o.Start) & (d.End >= end)]
            washes += [(n, float(o.Start), end, "開水時手持") for n in held.Nouns.unique()]
        for noun, ws, we, src in washes:
            p = d[(d.Verbs == "Put") & (d.Nouns == noun) & (d.Start >= ws) & (d.Start <= we + MAX_GAP_S)]
            if not len(p):
                continue
            p = p.iloc[0]
            cands.append(dict(session_id=sid, vehicle=veh, session_time=st, noun=noun, src=src,
                              wash_start=ws, wash_end=we, put_start=float(p.Start), put_end=float(p.End)))
    # 同一場次、同一食材、同一次放下(標註重複)只留一件
    uniq = {}
    for c in cands:
        uniq.setdefault((c["session_id"], c["noun"], round(c["put_end"], 1)), c)
    return list(uniq.values()), no_ann


def overlaps(a0, a1, b0, b1, tol=0.0):
    return min(a1, b1) + tol >= max(a0, b0)


def mark_known(cands):
    """範例已有 / 團隊已標。回傳範例清單(召回率用)。"""
    ex = list(csv.DictReader(open(EXAMPLE / "ng_instances.csv", encoding="utf-8-sig")))
    team = [r for r in csv.DictReader(open(TEAM, encoding="utf-8-sig")) if r["ng_type"] == "NG4"]
    for c in cands:
        c["example"] = ""
        c["team"] = ""
        for r in ex:
            w0, w1 = [sum(float(x) * k for x, k in zip(t.split(":"), (3600, 60, 1)))
                      for t in r["cause"].split(" ")[0].split("–")]
            if r["session_id"] == c["session_id"] and overlaps(w0, w1, c["wash_start"], c["wash_end"], 1.0):
                c["example"] = (c["example"] + " " if c["example"] else "") + f"{r['reason']}"
        for r in team:
            if r["session_id"] == c["session_id"] and overlaps(float(r["start_sec"]), float(r["end_sec"]),
                                                                c["wash_start"], c["put_end"]):
                c["team"] = r["reason"]
    return ex


def read_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * FPS))))
    ok, fr = cap.read()
    return fr if ok else None


def detect(model, fr):
    from PIL import Image
    det = model.predict(Image.fromarray(fr[:, :, ::-1]), threshold=0.2)
    if len(det) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return det.xyxy, det.confidence, det.class_id.astype(int)


def ioa(a, b):
    """a 落在 b 內的比例,[len(a), len(b)]。"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / np.maximum(area[:, None], 1e-6)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None] - inter, 1e-6)


def food_boxes(xy, cf, cl):
    """(食材框, 是否與手框重疊, 砧板 / 容器框)。"""
    F = xy[(cl == NAME_TO_CLASS["食材"]) & (cf > THR_OBJ)]
    H = xy[(cl == NAME_TO_CLASS["手"]) & (cf > THR_HAND)]
    S = xy[np.isin(cl, [NAME_TO_CLASS["砧板"], NAME_TO_CLASS["容器"]]) & (cf > THR_OBJ)]
    held = (ioa(F, H) > 0).any(1) if len(F) and len(H) else np.zeros(len(F), bool)
    return F, held, S


def judge(model, c):
    """回傳 (初判, 說明)。只看放下後「新出現」的食材框(放下前沒有的)。"""
    votes = {"sup": [], "counter": []}
    for v in JUDGE_VIEWS:
        cap = cv2.VideoCapture(str(VID / c["session_id"] / "videos" / f"{v}.mp4"))
        sink = get_zone(ZONES / f"epfl_{v}.json", "洗手台") if (ZONES / f"epfl_{v}.json").exists() else None
        before = []
        for dt in BEFORE_S:
            fr = read_frame(cap, c["put_start"] + dt)
            if fr is not None:
                before.append(food_boxes(*detect(model, fr))[0])
        before = np.concatenate(before) if before else np.zeros((0, 4))
        n_sup = n_cnt = 0
        for dt in AFTER_S:
            fr = read_frame(cap, c["put_end"] + dt)
            if fr is None:
                continue
            F, held, S = food_boxes(*detect(model, fr))
            if len(F) and len(before):
                new = ~(iou(F, before) >= NEW_IOU).any(1)
                F, held = F[new], held[new]
            if not len(F):
                continue
            on = (ioa(F, S) >= SUP_IOA).any(1) if len(S) else np.zeros(len(F), bool)
            if sink is not None:
                p = np.asarray(sink, np.float32).reshape(-1, 1, 2)
                ctr = np.c_[(F[:, 0] + F[:, 2]) / 2, (F[:, 1] + F[:, 3]) / 2]
                in_sink = np.array([cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for x, y in ctr])
            else:
                in_sink = np.zeros(len(F), bool)
            n_sup += int(on.any())                                # 砧板 / 容器:手還按著也算(放下後常馬上切)
            n_cnt += int((~on & ~in_sink & ~held).any())          # 桌面:手要離開,避免拿在半空中被當成放下
        cap.release()
        if n_sup >= 2:
            votes["sup"].append(v)
        if n_cnt >= 2:
            votes["counter"].append(v)
    why = f"砧板/容器:{'、'.join(votes['sup']) or '無'};桌面:{'、'.join(votes['counter']) or '無'}"
    if votes["sup"] and not votes["counter"]:
        return "砧板或容器", why
    if votes["counter"] and not votes["sup"]:
        return "桌面", why
    return "無法判斷", why + ("(兩者都有)" if votes["sup"] else "")


def cut(job):
    src, out, t0, dur = job
    if out.exists():
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(FPS), "-pix_fmt", "yuv420p",
                    "-an", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只搜尋 + 初判,不剪片、不寫交付檔")
    args = ap.parse_args()

    cands, no_ann = find_candidates()
    ex = mark_known(cands)
    print(f"場次 {len(sessions())},缺標註 {len(no_ann)}:{no_ann}")
    print(f"候選 {len(cands)}")

    # 召回率:範例每件是否被找到
    print("範例召回:")
    for r in ex:
        w0, w1 = [sum(float(x) * k for x, k in zip(t.split(":"), (3600, 60, 1)))
                  for t in r["cause"].split(" ")[0].split("–")]
        hit = [c for c in cands if c["session_id"] == r["session_id"] and overlaps(w0, w1, c["wash_start"], c["wash_end"], 1.0)]
        print(f"  {r['session_id']} {r['cause'][:30]} {r['reason']}:{'找到 ' + '、'.join(c['noun'] for c in hit) if hit else '沒找到'}")

    from m3.model import load_rfdetr
    model = load_rfdetr("nano", WEIGHTS)
    for n, c in enumerate(cands, 1):
        c["auto"], c["why"] = judge(model, c)
        if n % 10 == 0:
            print(f"  初判 {n}/{len(cands)}", flush=True)
    print("範例上的初判:")
    for c in cands:
        if c["example"]:
            print(f"  {c['session_id']} {c['noun']} 範例:{c['example']} → 系統:{c['auto']}({c['why']})")
    print("初判分布:", {k: sum(c["auto"] == k for c in cands) for k in FOLDER})
    if args.check:
        return

    # 編號:每場次依清洗時間從 001 起
    cands.sort(key=lambda c: (c["session_id"], c["wash_start"], c["noun"]))
    k = {}
    idx_rows, inst_rows, review, jobs = [], [], [], []
    for c in cands:
        k[c["session_id"]] = k.get(c["session_id"], 0) + 1
        no = k[c["session_id"]]
        sub, tag, reason = FOLDER[c["auto"]]
        t0, t1 = c["wash_start"], c["put_end"]
        aid = f"cand_{c['vehicle']}_{c['session_time']}_{no:03d}"
        cause = f"{hms(c['wash_start'])}–{hms(c['wash_end'])} 食材或廚具清洗（{zh(c['noun'])}）"
        effect = f"{hms(c['put_end'])} 食材放下（{zh(c['noun'])}）"
        how = f"Wash {c['noun']}" if c["src"] == "Wash" else f"手持 {c['noun']} 時開水"
        note = f"EPFL 標註:{how} → Put {c['noun']};系統初判 {c['auto']}"
        base = dict(session_id=c["session_id"], vehicle=c["vehicle"], session_time=c["session_time"],
                    annotated_view="output0", category="食材衛生", note=note, user=USER, annotation_id=aid,
                    ng_type="NG4", result="待人工確認", ng_index=no, reason=reason, cause=cause, effect=effect,
                    event_start_sec=f"{t0:.3f}", event_end_sec=f"{t1:.3f}", outside_event="否",
                    event_timeline=f"＊{cause}\n＊{effect}", detail_ids="")
        inst_rows.append(dict(base, start_sec=f"{t0:.3f}", end_sec=f"{t1:.3f}", start_hms=hms(t0), end_hms=hms(t1),
                              duration_sec=f"{t1 - t0:.3f}"))
        for v in VIEWS:
            name = f"{c['vehicle']}_{c['session_time']}_ng{no:03d}_{tag}_{v}_{USER}.mp4"
            src = VID / c["session_id"] / "videos" / f"{v}.mp4"
            idx_rows.append(dict(base, clip_file=f"{sub}/{name}", source_file=f"{c['session_id']}/videos/{v}.mp4",
                                 view=v, clip_start_sec=f"{t0:.3f}", clip_end_sec=f"{t1:.3f}", clip_start_hms=hms(t0),
                                 clip_end_hms=hms(t1), clip_duration_sec=f"{t1 - t0:.3f}"))
            if src.exists():
                jobs.append((src, OUT / sub / name, t0, t1 - t0))
        review.append({"編號": f"{c['vehicle']}_{c['session_time']}_ng{no:03d}", "資料夾": sub, "場次": c["session_id"],
                       "食材": zh(c["noun"]), "清洗": f"{hms(c['wash_start'])}–{hms(c['wash_end'])}",
                       "放下": hms(c["put_end"]), "片段長度(秒)": round(t1 - t0, 1), "系統初判": c["auto"],
                       "初判依據(哪些鏡頭)": c["why"],
                       "清洗依據": "標註 Wash" if c["src"] == "Wash" else "標註無 Wash,手持食材時開水", "範例已有": c["example"], "團隊已標": c["team"],
                       "M3訓練場次": "是" if any(m in c["session_id"] for m in M3_TRAIN) else "",
                       "檢視結果(反例/NG4/都不是)": "", "實際放置位置": "", "備註": ""})

    for sub, _, _ in FOLDER.values():
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex_:
        list(ex_.map(cut, jobs))
    for name, cols, rows in (("ng_clips_index.csv", INDEX_COLS, idx_rows), ("ng_instances.csv", INST_COLS, inst_rows)):
        with open(OUT / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    pd.DataFrame(review).to_excel(OUT / "人工檢視表.xlsx", index=False)
    # 欄位與範例一致性檢查
    for name in ("ng_clips_index.csv", "ng_instances.csv"):
        a = next(csv.reader(open(EXAMPLE / name, encoding="utf-8-sig")))
        b = next(csv.reader(open(OUT / name, encoding="utf-8-sig")))
        print(f"{name} 欄位與範例相同:{a == b}")
    print(f"完成:候選 {len(cands)},片段 {len(jobs)} → {OUT}")


if __name__ == "__main__":
    main()
