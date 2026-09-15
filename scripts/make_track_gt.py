"""產生四層歸因診斷用的真值對照表 `{"<camera>|<track_id>": gt_id}`。

配對邏輯**直接 import `eval_m4m5_chirla.match_tracks`**,不另外實作 ——
兩邊若分歧,診斷分解出來的數字就對不上誤併率與碎裂率,整份分析失去意義。

⚠ 為什麼可以用 base 那一格算出來的表去餵其他格:
  `m5_track_video.py` 的 tracker 只吃偵測結果,**M5 從不回饋到 M4**,
  所以 `(camera, track_id)` 在所有 M5 變體之間逐條相同。
  這也避免了「oracle 改變了 track id」這種循環依賴。

⚠ 但**換追蹤器(門檻、backend)就會換 track id**,真值表必須重算。
  2026-09-15 起加 `--tracks-dir`:直接吃 `eval_m4_chirla.py --dump-dir` 匯出的 tracks.csv
  (幾秒),不必先完整跑一次 M5 只為了拿 track。前提是匯出的 track 與 M5 runner
  產生的 track 逐列相同 —— 那是步驟 3 的硬性驗收,不是假設。

用法:
    # 從 M5 輸出目錄(原本的用法)
    python scripts/make_track_gt.py --root <CHIRLA根> \
        --run-dir results/chirla_p1grid/base/seq_004 ... --out-dir results/track_gt

    # 從 eval_m4_chirla.py 的匯出目錄(內含 <seq>/tracks.csv)
    python scripts/make_track_gt.py --root <CHIRLA根> \
        --tracks-dir <匯出目錄> --seqs seq_004 ... --out-dir results/track_gt_cbiou
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_m4m5_chirla import load_gt, load_run, match_tracks   # noqa: E402


def load_tracks_csv(path):
    """與 `eval_m4m5_chirla.load_run` 讀 tracks.csv 的欄位與型別逐一相同。"""
    with open(path, encoding="utf-8") as f:
        return [dict(fid=int(r["video_fid"]), cam=r["camera_id"], tid=int(r["track_id"]),
                     bbox=(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
                for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CHIRLA 根目錄")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", nargs="+", help="m5_track_video.py 的輸出目錄,每個對應一個序列")
    src.add_argument("--tracks-dir", help="內含 <seq>/tracks.csv 的目錄(需搭配 --seqs)")
    ap.add_argument("--seqs", nargs="+", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "track_gt"))
    args = ap.parse_args()
    if args.tracks_dir and not args.seqs:
        raise SystemExit("--tracks-dir 需要搭配 --seqs")

    units = []
    if args.run_dir:
        for rd in args.run_dir:
            meta, tracks, _events = load_run(rd)
            seq = next(s for s in Path(next(iter(meta["camera_video_map"].values()))).parts
                       if s.startswith("seq_"))
            units.append((seq, tracks))
    else:
        for seq in args.seqs:
            units.append((seq, load_tracks_csv(Path(args.tracks_dir) / seq / "tracks.csv")))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'序列':<10}{'track':>8}{'有真值':>8}{'誤偵':>8}{'誤偵率':>9}   輸出")
    print("-" * 72)
    for seq, tracks in units:
        track_gt, tstats, _gtm, _gtt = match_tracks(load_gt(args.root, seq), tracks)

        # ⚠ 只寫**有真值**的 track。誤偵 track 不出現在表裡,runner 那邊
        #   `track_gt.get(key)` 回 None 就直接判成 L1_ghost —— 與 eval 的
        #   `gid is None → ghost_bind` 走同一個判準。
        payload = {f"{cam}|{tid}": gid for (cam, tid), gid in track_gt.items()
                   if gid is not None}
        out = out_dir / f"{seq}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

        n, ghost = len(tstats), sum(s["is_ghost"] for s in tstats.values())
        print(f"{seq:<10}{n:>8}{len(payload):>8}{ghost:>8}{ghost / n * 100:>8.1f}%   {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
