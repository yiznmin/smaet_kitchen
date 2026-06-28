"""
通用 zone 標註工具:載入鏡頭畫面 → 滑鼠點多邊形 → 命名 → 存 zones.json。

固定鏡頭裝機時標一次、開機載入。靜態線地板 ROI 與 M6 各 zone 共用同一檔。

互動操作(需在有顯示器的本機執行):
  滑鼠左鍵 : 加一個點
  c        : 完成目前多邊形(>=3 點)→ 到終端機輸入名稱(如 floor / zoneA_raw / wash)
  z        : 退回上一個點
  d        : 刪除最後一個已完成的多邊形
  s        : 存檔(zones.json)
  q / ESC  : 存檔並離開

用法:
  python scripts/zone_annotator.py data/epfl/Boutput0.mp4
  python scripts/zone_annotator.py data/epfl/Boutput0.mp4 -o configs/zones.json --frame 0
  python scripts/zone_annotator.py data/epfl/Boutput0.mp4 --preview configs/zones.json
        ↑ 只渲染預覽圖(不需 GUI),輸出 <zones>.preview.png

注意:--preview 不需顯示器;互動標註需本機有視窗環境(cv2.imshow)。
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames, imwrite_unicode    # noqa: E402
from common.zones import load_zones, save_zones, draw_zones  # noqa: E402


def grab_frame(path, idx):
    """抓影片第 idx 幀;若為圖片則直接讀。"""
    p = Path(path)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        import numpy as np
        data = np.fromfile(str(p), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    for fid, ts, fr in iter_frames(p):
        if fid >= idx:
            return fr
    return None


def preview(video, zones_path):
    data = load_zones(zones_path)
    frame = grab_frame(video, data.get("frame", 0))
    if frame is None:
        sys.exit(f"無法取得畫面:{video}")
    vis = draw_zones(frame, data.get("zones", []))
    out = Path(zones_path).with_suffix(".preview.png")
    imwrite_unicode(out, vis)
    print(f"預覽已存:{out}")
    for z in data.get("zones", []):
        print(f"  - {z.get('name')}: {len(z.get('points', []))} 點")


def annotate(video, out_path, frame_idx):
    frame = grab_frame(video, frame_idx)
    if frame is None:
        sys.exit(f"無法取得畫面:{video}")
    H, W = frame.shape[:2]

    zones, current = [], []
    if Path(out_path).exists():           # 可接續編輯既有檔
        zones = load_zones(out_path).get("zones", [])
        print(f"載入既有 {len(zones)} 個 zone,可接續新增")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([x, y])

    win = "zone annotator  (L:加點  c:完成  z:退點  d:刪區  s:存檔  q:離開)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = draw_zones(frame, zones, current)
        cv2.putText(vis, f"zones={len(zones)}  current_pts={len(current)}",
                    (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('z') and current:
            current.pop()
        elif k == ord('d') and zones:
            removed = zones.pop()
            print(f"刪除 zone: {removed.get('name')}")
        elif k == ord('c') and len(current) >= 3:
            name = input("這個多邊形的名稱(如 floor / zoneA_raw / wash / trash):").strip()
            if name:
                zones.append({"name": name, "points": [list(p) for p in current]})
                print(f"  加入 zone: {name}（{len(current)} 點）")
            current.clear()
        elif k == ord('s'):
            save_zones(out_path, zones, source=video, width=W, height=H, frame=frame_idx)
            print(f"已存:{out_path}（{len(zones)} 個 zone）")

    save_zones(out_path, zones, source=video, width=W, height=H, frame=frame_idx)
    cv2.destroyAllWindows()
    print(f"離開,已存:{out_path}（{len(zones)} 個 zone）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="鏡頭影片或單張畫面")
    ap.add_argument("-o", "--out", default=str(ROOT / "configs" / "zones.json"))
    ap.add_argument("--frame", type=int, default=0, help="用影片第幾幀來標")
    ap.add_argument("--preview", metavar="ZONES_JSON", help="只渲染既有 zones.json 預覽(不需 GUI)")
    args = ap.parse_args()

    if args.preview:
        preview(args.source, args.preview)
    else:
        annotate(args.source, args.out, args.frame)


if __name__ == "__main__":
    main()
