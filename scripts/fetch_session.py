"""
列出 / 下載 EPFL 其他場次(session)的 9 個同步視角,存到指定資料夾。
用 HTTP Range 只抓那 9 支(各 ~85MB),不下整包 192GB。

授權:EPFL 為 CC BY-NC 4.0(禁商用),僅作 benchmark;使用須引用 arXiv 2506.01608。

用法:
  python scripts/fetch_session.py --list                    # 列出有齊 9 視角的場次
  python scripts/fetch_session.py --auto --out data/epfl_s2 # 自動挑一個與現用不同(盡量不同人)的場次下載
  python scripts/fetch_session.py --session <zip內session路徑/> --out data/epfl_s2
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_sample import read_central_directory, extract_one   # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"
VIEWS = ["Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3",
         "Boutput0", "Boutput1", "Boutput2", "Boutput3", "output0"]


def sessions_map(entries):
    """回傳 {session_dir: {view: entry}},只收非深度、檔名為 9 視角之一者。"""
    m = defaultdict(dict)
    for e in entries:
        n = e["name"]
        if not n.lower().endswith(".mp4") or "depth" in n.lower():
            continue
        p = PurePosixPath(n)
        if p.stem in VIEWS and p.parent.name == "videos":
            m[str(p.parent) + "/"][p.stem] = e
    return m


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    url = cfg["dataset"]["zip_url"]
    cur = cfg["dataset"]["default_sample"].rsplit("/", 1)[0] + "/"

    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--session")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "epfl_s2"))
    args = ap.parse_args()

    print("讀取 zip central directory …")
    total, entries = read_central_directory(url)
    m = sessions_map(entries)
    full = {s: v for s, v in m.items() if len(v) >= 9}
    print(f"共 {len(m)} 個場次,有齊 9 視角的 {len(full)} 個。現用:{cur}")

    if args.list:
        for s in sorted(full):
            print(f"  {'★現用' if s == cur else '     '}  {s}")
        return

    if args.auto:
        cand = sorted(s for s in full if s != cur)
        if not cand:
            sys.exit("沒有其他有齊 9 視角的場次")
        cur_pid = cur.split("/")[2] if len(cur.split("/")) > 2 else ""
        diff_pid = [s for s in cand if (s.split("/")[2] if len(s.split("/")) > 2 else "") != cur_pid]
        target = (diff_pid or cand)[0]
    elif args.session:
        target = args.session if args.session.endswith("/") else args.session + "/"
        if target not in full:
            sys.exit(f"找不到或視角不齊:{target}")
    else:
        sys.exit("請用 --list / --auto / --session")

    out = Path(args.out)
    print(f"\n下載場次:{target}\n → {out}")
    tot = 0
    for v in VIEWS:
        e = full[target][v]
        dest = out / f"{v}.mp4"
        print(f"  {v} ({e['usize'] / 1e6:.0f}MB) …", end=" ", flush=True)
        n = extract_one(url, e, dest)
        print(f"{n / 1e6:.0f}MB OK")
        tot += n
    print(f"\n完成:9 視角,共 {tot / 1e6:.0f}MB → {out}")


if __name__ == "__main__":
    main()
