"""
從 EPFL zip 中,抓「與 default_sample 同一 session 資料夾」的所有視角。
同一資料夾 = 同步錄製 = **同一時間點、不同相機**(多視角)。

沿用 fetch_sample 的 HTTP Range 抽檔:只傳輸這幾支(各 ~80MB),不下整包 192.7 GB。
授權:EPFL 為 CC BY-NC 4.0,僅作 benchmark,使用須引用 arXiv 2506.01608。

用法:
  python scripts/fetch_multiview.py --list     # 只列同 session 的視角(不下載)
  python scripts/fetch_multiview.py            # 下載全部(跳過已存在的)
  python scripts/fetch_multiview.py --only Aoutput0.mp4 Aoutput1.mp4   # 只下指定幾支
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sample import read_central_directory, extract_one   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "benchmark.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只列同 session 視角,不下載")
    ap.add_argument("--only", nargs="*", help="只下載指定檔名(如 Aoutput0.mp4)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    url = ds["zip_url"]
    sample = ds["default_sample"]
    session_dir = sample.rsplit("/", 1)[0] + "/"      # .../<session>/videos/

    print(f"讀取 zip central directory …")
    total, entries = read_central_directory(url)
    views = [e for e in entries
             if e["name"].startswith(session_dir)
             and e["name"].lower().endswith(".mp4")
             and "depth" not in e["name"].lower()]
    views.sort(key=lambda e: e["name"])

    print(f"\nsession(同一時間點):{session_dir}")
    print(f"同步視角(非深度 mp4):{len(views)} 支")
    for e in views:
        print(f"  {e['usize'] / 1e6:7.1f} MB   {Path(e['name']).name}")

    if args.list:
        return

    targets = views
    if args.only:
        want = set(args.only)
        targets = [e for e in views if Path(e["name"]).name in want]

    print()
    for e in targets:
        dest = ROOT / ds["sample_dir"] / Path(e["name"]).name
        if dest.exists():
            print(f"跳過(已存在):{dest.name}")
            continue
        print(f"下載 {Path(e['name']).name}（{e['usize'] / 1e6:.1f} MB）…")
        n = extract_one(url, e, dest)
        print(f"  完成 {n / 1e6:.1f} MB → {dest.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
