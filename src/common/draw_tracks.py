"""標註畫面的繪製 —— 線上 runner 與離線渲染器共用同一份實作。

2026-09-16 從 `scripts/m5_track_video.py` **原樣**搬過來(邏輯一個字沒改)。
搬的理由:難度分級交付要一支離線渲染器(重放 `tracks.csv` 產出結果影片),
而它沒有活著的 `IdentityManager` 可以問「這條 track 綁給誰」。
兩邊若各畫各的,影片與指標就可能對不上而且沒人會發現 —— 所以共用一份。

唯一的介面調整:`draw_panel` 的 chef 對照參數可以是 manager,也可以是純 dict
(鍵一律是 `(camera_id, track_id)`)。行為不變是驗收條件:EPFL 九台重跑,
`chef_events.jsonl` / `tracks.csv` / `track_events.csv` 必須與搬動前逐位相同。

⚠ 顏色由 **chef_id** 決定,不是 track_id —— track_id 每台鏡頭各自編號,
  用它上色的話「跨鏡頭認出同一個人」這件事在畫面上就看不出來。
"""
from collections import namedtuple

import numpy as np

# 離線渲染器用:從 tracks.csv 的一列做出 draw_panel 吃得下的最小物件。
# 用同一個型別,兩條路徑才不會在「bbox 是 None 怎麼辦」這種地方各自處理。
DrawTrack = namedtuple("DrawTrack", "track_id bbox")

_PALETTE = [(80, 200, 120), (80, 160, 255), (200, 120, 255), (60, 220, 240),
            (255, 170, 80), (140, 220, 90), (255, 120, 170), (110, 190, 255)]


def chef_color(chef_id):
    return _PALETTE[(int(chef_id) - 1) % len(_PALETTE)] if chef_id else (140, 140, 140)


def _chef_of(chefs):
    """取得 (camera_id, track_id) -> chef_id 的查詢函式。

    chefs 可以是 SpatioTemporalIdentityManager(線上)或 dict(離線重放)。
    """
    table = getattr(chefs, "track_to_chef", chefs)
    return table.get


def draw_panel(frame_bgr, tracks, cam, chefs, t_sec, width=640):
    """畫一台鏡頭:每個 active track 標上它的 chef_id(不是 track_id)。"""
    import cv2
    img = frame_bgr.copy()
    look = _chef_of(chefs)
    # ⚠ 標題列必須**先**畫。原本畫在最後,會把貼齊上緣的人的標籤整個蓋掉
    #   —— cam3 那格看起來像沒認出人,實際上只是標籤被蓋住了。
    cv2.rectangle(img, (0, 0), (img.shape[1], 42), (28, 28, 28), -1)
    cv2.putText(img, f"{cam}   t={t_sec:.2f}s", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2)
    for tr in tracks:
        if tr.bbox is None:
            continue
        cid = look((cam, tr.track_id))
        col = chef_color(cid)
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        label = f"chef {cid}" if cid else f"track {tr.track_id} unbound"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # 標籤預設畫在框上方,但人貼齊畫面上緣時會被切掉(頂部還有鏡頭名的橫條)
        # → 放不下就翻到框內側。第一版沒做這件事,cam3 的標籤整個看不到。
        top = y1 - th - 10
        ty1 = top if top >= 46 else max(y1, 46)
        ty2 = y1 if top >= 46 else ty1 + th + 10
        lx = min(x1, img.shape[1] - tw - 12)          # 靠右邊界時往左收
        cv2.rectangle(img, (lx, ty1), (lx + tw + 10, ty2), col, -1)
        cv2.putText(img, label, (lx + 5, ty2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    h = int(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h))


def stitch(panels):
    """多鏡頭橫向拼接(高度不同就補黑),讓同一時刻的各鏡頭並排。"""
    import cv2
    if not panels:
        return None
    hmax = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < hmax:
            p = cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(20, 20, 20))
        padded.append(p)
    return np.hstack(padded)
