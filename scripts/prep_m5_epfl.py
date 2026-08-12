"""為 M5 Re-ID 驗證準備 EPFL 多身份同步幀:下載 K 個「不同參與者」場次、抽同步幀、刪影片。

每個場次 = 一位不同參與者 = 一個廚房身份;每場 9 視角同步 → 同一時刻不同視角=同一人。
輸出 data/m5_reid_epfl/frames/<sN>_<view>_f<fid>.jpg,供 scripts/reid_eval_epfl.py 使用。

⚠ 每場 9 視角約 1–3 GB,抽完即刪(只留幾十張幀)。K=6 需下載數 GB,建議背景執行。
授權:EPFL CC-BY-NC,僅研究驗證、不出貨。

用法:
  python scripts/prep_m5_epfl.py --k 6 --per 20
  python scripts/prep_m5_epfl.py --k 6 --views output0 Aoutput0 Aoutput2 Boutput0 Boutput2   # 少視角省下載
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_session import read_central_directory, sessions_map     # noqa: E402
from fetch_sample import extract_one                               # noqa: E402
from extract_multiview_synced import pick_keyframes, grab, VIEWS   # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="要幾個不同參與者(身份)")
    ap.add_argument("--per", type=int, default=20, help="每視角每場抽幾張同步幀")
    ap.add_argument("--views", nargs="*", default=VIEWS, help="要下載的視角(預設全部 9)")
    ap.add_argument("--out", default=str(ROOT / "data" / "m5_reid_epfl" / "frames"))
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    url = cfg["dataset"]["zip_url"]
    print("讀 zip central directory …")
    total, entries = read_central_directory(url)
    m = sessions_map(entries)
    full = {s: v for s, v in m.items() if all(view in v for view in args.views)}

    picked, seen = [], set()
    for s in sorted(full):
        pid = s.split("/")[2] if len(s.split("/")) > 2 else s
        if pid in seen:
            continue
        seen.add(pid)
        picked.append((pid, s))
        if len(picked) >= args.k:
            break
    print(f"挑到 {len(picked)} 位參與者:{[p for p, _ in picked]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "data" / "_m5_tmp"
    for i, (pid, s) in enumerate(picked, start=1):
        tag = f"s{i}"
        print(f"\n[{tag}] {pid}  下載 {len(args.views)} 視角 …")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for v in args.views:
            e = full[s][v]
            print(f"  {v} ({e['usize'] / 1e6:.0f}MB) …", end=" ", flush=True)
            extract_one(url, e, tmp / f"{v}.mp4")
            print("OK")
        ref = tmp / ("output0.mp4" if "output0" in args.views else f"{args.views[0]}.mp4")
        fids = pick_keyframes(ref, args.per)
        for v in args.views:
            grab(tmp / f"{v}.mp4", fids, out, tag, v)
        print(f"  [{tag}] 抽 {len(fids)}×{len(args.views)} 同步幀")
        shutil.rmtree(tmp)
    print(f"\n完成 → {out}(身份 {len(picked)})")


if __name__ == "__main__":
    main()
