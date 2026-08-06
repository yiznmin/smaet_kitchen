"""
時間對齊的多視角抽幀:9 個鏡頭抽「同一批 frame id」→ 同一時間點、不同角度。

作法:先用一個參考鏡頭(預設 output0)挑 N 個「有動作、全片分散」的 frame id,
再從 9 個鏡頭各自抽出「同一批 frame id」→ 每個時間點就有 9 個角度。

可跑多個場次(session):各場次用不同 --tag,輸出檔名帶場次標籤,不會撞名。

輸出:<out>/<tag>_<view>_f<fid>.jpg
用法:
  # 場次資料夾內要有 9 支 <view>.mp4(Aoutput0..3, Boutput0..3, output0)
  python scripts/extract_multiview_synced.py --session data/epfl     --tag s1 --per 30
  python scripts/extract_multiview_synced.py --session data/epfl_s2  --tag s2 --per 30
  # 兩場次跑完 = 30 x 9 x 2 = 540 張,全部在同一個 mv_images 資料夾
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import video_meta, imwrite_unicode   # noqa: E402

VIEWS = ["Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3",
         "Boutput0", "Boutput1", "Boutput2", "Boutput3", "output0"]


def pick_keyframes(ref_video, per):
    """挑 per 個『全片均勻分散』的 frame id(跳過頭尾 5%,避免空景/暖機)。

    單人連續烹飪的影片,均勻取樣幾乎張張有內容,不必逐幀掃動作(那太慢)。
    這些 frame id 之後會套用到全部 9 個鏡頭 → 同一時間點、不同角度。
    """
    total = video_meta(ref_video)["nb_frames"]
    start, end = int(total * 0.05), int(total * 0.95)
    if per <= 1:
        return [(start + end) // 2]
    step = (end - start) / (per - 1)
    return [int(start + step * i) for i in range(per)]


def grab(video, fids, out_dir, tag, view):
    """從指定影片抽出指定的一批 frame id(seek,快)。"""
    cap = cv2.VideoCapture(str(video))
    n = 0
    for fid in fids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if ok:
            fp = out_dir / f"{tag}_{view}_f{fid:05d}.jpg"
            if imwrite_unicode(fp, frame):
                n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="含 9 支 <view>.mp4 的場次資料夾")
    ap.add_argument("--tag", required=True, help="場次標籤(檔名前綴,如 s1/s2);不同場次要不同")
    ap.add_argument("--per", type=int, default=30, help="每視角抽幾張(= 挑幾個時間點)")
    ap.add_argument("--ref", default="output0", help="用哪個視角挑時間點(預設 output0)")
    ap.add_argument("--out", default=str(ROOT / "data" / "m3_finetune" / "mv_images"),
                    help="輸出資料夾(預設 data/m3_finetune/mv_images)")
    args = ap.parse_args()

    sess = Path(args.session)
    out_dir = Path(args.out)
    ref = sess / f"{args.ref}.mp4"
    if not ref.exists():
        sys.exit(f"找不到參考視角:{ref}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.tag}] 用 {args.ref} 挑 {args.per} 個有動作的時間點 …")
    fids = pick_keyframes(ref, args.per)
    print(f"  選定 {len(fids)} 個 frame id:{fids}")

    tot = 0
    for v in VIEWS:
        vid = sess / f"{v}.mp4"
        if not vid.exists():
            print(f"  跳過(找不到){v}")
            continue
        c = grab(vid, fids, out_dir, args.tag, v)
        print(f"  {v}: {c} 張")
        tot += c
    ideal = len(fids) * len(VIEWS)
    print(f"\n[{args.tag}] 完成:共 {tot} 張(理想 {len(fids)}x9={ideal})→ {out_dir}")


if __name__ == "__main__":
    main()
