"""NG3 碰臉第 3 輪(§8.85)誤報片段匯出,交給他人人工檢視是否其實有碰臉。

規則 = LOPO 選到的 r0.5_hand_m0_any(手框到臉中心 ≤ 0.5 × 肩寬,連續 ≥ 0.3 秒)。
每件輸出:疊圖影片(手框、臉中心與判定半徑、系統判碰臉時畫面紅框)、每次觸發的截圖、人工檢視表.xlsx。
輸出到 D:/yizhen/NG3_碰臉誤報人工檢視_20261105/(含 EPFL 影像,不進 git)。

用法:
    python scripts/export_ng3_fp_review.py
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "integrated/src"))
import eval_ng3_face as F                                                 # noqa: E402
import eval_ng_clips as E                                                 # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.violation import HAND, at, steps                           # noqa: E402

OUT = Path("D:/yizhen/NG3_碰臉誤報人工檢視_20261105")
PARAM = (0.5, "hand", 0.0, False)
FONT = "C:/Windows/Fonts/msjh.ttc"
W, H = 640, 360


def text_img(lines, width, size=20):
    font = ImageFont.truetype(FONT, size)
    im = Image.new("RGB", (width, (size + 8) * len(lines) + 8), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for k, t in enumerate(lines):
        d.text((8, 4 + k * (size + 8)), t, font=font, fill=(255, 255, 0))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def faces_at(pz, i):
    """第 i 步(或往前最近一個關鍵點幀)的臉中心與肩寬。"""
    st = pz["step"]
    j = st[st <= i]
    if not len(j):
        return []
    out = []
    for k in pz["kp"][st == j.max()]:
        head = k[:5][k[:5, 2] >= F.KP_THR]
        if len(head) and k[5, 2] >= F.KP_THR and k[6, 2] >= F.KP_THR:
            sw = np.linalg.norm(k[5, :2] - k[6, :2])
            if sw >= 1:
                out.append((head[:, :2].mean(0), sw))
    return out


def main():
    d = pd.read_csv(ROOT / "results/m6/ng/ng3_face_r3/per_clip.csv", encoding="utf-8-sig")
    fp = d[(d.y == 0) & (d.cv == 1)].reset_index(drop=True)
    C = E.load_clips()
    (OUT / "影片").mkdir(parents=True, exist_ok=True)
    (OUT / "截圖").mkdir(exist_ok=True)
    rows = []
    for n, r in fp.iterrows():
        c = C[r.key]
        vs = [v for v in F.POSE_VIEWS if v in c["caches"] and F.pose_path(c, v).exists()]
        dcs = {v: DetCache(c["caches"][v]) for v in vs}
        pzs = {v: np.load(F.pose_path(c, v)) for v in vs}
        T = min(len(steps(x)[0]) for x in dcs.values())
        raw = np.any([F.face_series3(dcs[v], pzs[v], T)[PARAM] for v in vs], 0)
        segs = F.touches(raw, 0.3)
        seg_txt = "、".join(f"{a * F.STEP:.1f}–{b * F.STEP:.1f} 秒" for a, b in segs)
        name = f"{n + 1:02d}_{r.key}"
        frames_all = {}
        for v in vs:                                       # 逐鏡頭讀進記憶體(片段短)
            cap = cv2.VideoCapture(F.short_path(c["files"][v]))
            fs = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                fs.append(f)
            cap.release()
            frames_all[v] = fs
        fps = 30.0
        stride = int(dcs[vs[0]].meta["cache_stride"])
        nf = min(len(x) for x in frames_all.values())
        head = text_img([f"#{n + 1}  {r.key}", f"團隊類別:{r.ng} {r['cat']}  備註:{r.note if isinstance(r.note, str) else ''}",
                         f"系統判碰臉:{seg_txt}(畫面紅框 = 系統判定此刻手碰到臉)",
                         "黃圈 = 臉中心與判定範圍,綠框 = 手"], W * len(vs))
        tmp = ROOT / "results/m6/ng/_tmp_ng3.mp4"               # OpenCV 寫不了中文路徑:先寫 ASCII 暫存再搬
        wr = cv2.VideoWriter(str(tmp),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * len(vs), H + head.shape[0]))
        shot_at = {next(j for j in range(a, b) if raw[j]) for a, b in segs}   # 每段第一個判碰臉的步
        for fi in range(nf):
            i = min(fi // stride, T - 1)
            tiles = []
            for v in vs:
                f = frames_all[v][fi].copy()
                xy, cf, cl = at(dcs[v], steps(dcs[v]), i)
                for x in xy[(cl == HAND) & (cf >= F.HAND_THR2)]:
                    cv2.rectangle(f, tuple(map(int, x[:2])), tuple(map(int, x[2:4])), (0, 255, 0), 2)
                for fc, sw in faces_at(pzs[v], i):
                    cv2.circle(f, tuple(map(int, fc)), int(PARAM[0] * sw), (0, 255, 255), 2)
                    cv2.circle(f, tuple(map(int, fc)), 4, (0, 255, 255), -1)
                f = cv2.resize(f, (W, H))
                cv2.putText(f, f"{v}  t={fi / fps:.1f}s", (8, 26), 0, 0.7, (255, 255, 255), 2)
                tiles.append(f)
            body = np.hstack(tiles)
            if raw[i]:
                cv2.rectangle(body, (0, 0), (body.shape[1] - 1, body.shape[0] - 1), (0, 0, 255), 8)
                cv2.putText(body, "TOUCH", (body.shape[1] - 130, 34), 0, 1.0, (0, 0, 255), 3)
            img = np.vstack([head, body])
            wr.write(img)
            if fi % stride == 0 and i in shot_at:
                (OUT / "截圖" / f"{name}_{i * F.STEP:.1f}s.jpg").write_bytes(cv2.imencode(".jpg", img)[1].tobytes())
        wr.release()
        shutil.move(str(tmp), OUT / "影片" / f"{name}.mp4")
        rows.append({"編號": n + 1, "片段": r.key, "受試者": r.pid, "團隊類別": f"{r.ng} {r['cat']}",
                     "團隊備註": r.note if isinstance(r.note, str) else "", "片段長度(秒)": round(r.dur_s, 1),
                     "系統判碰臉時間": seg_txt, "影片": f"影片/{name}.mp4",
                     "檢視結果(1 有碰臉 / 2 沒碰臉 / 3 看不出來)": "", "碰臉方式或時間(選填)": "", "備註": ""})
        print(f"  {n + 1}/{len(fp)} {r.key}", flush=True)
    guide = pd.DataFrame({"說明": [
        "這 27 個片段是系統判定「手碰到臉」、但團隊標註沒有標成碰臉的片段,請看影片判斷是否其實有碰到臉。",
        "算碰臉:手直接碰臉(摸臉、嘴、鼻、揉眼、擦汗)、手拿食物入口、用手臂或手背擦臉。",
        "不算碰臉:摸頭髮 / 頭部、隔著衛生紙擦鼻子、手只是靠近臉但沒碰到。",
        "影片上方有系統判碰臉的時間;畫面出現紅框(右上 TOUCH)= 系統判定此刻手碰到臉。黃圈是臉的位置與判定範圍,綠框是偵測到的手。",
        "請在「人工檢視表」的檢視結果欄填 1(有碰臉)/ 2(沒碰臉)/ 3(看不出來);有碰臉時可在下一欄寫碰臉方式或時間。",
        "截圖資料夾是每次系統判碰臉當下的畫面,可輔助快速判斷。"]})
    with pd.ExcelWriter(OUT / "人工檢視表.xlsx") as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="人工檢視表", index=False)
        guide.to_excel(xw, sheet_name="說明", index=False)
    print("完成", OUT)


if __name__ == "__main__":
    main()
