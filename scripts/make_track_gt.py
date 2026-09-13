"""產生四層歸因診斷用的真值對照表 `{"<camera>|<track_id>": gt_id}`。

配對邏輯**直接 import `eval_m4m5_chirla.match_tracks`**,不另外實作 ——
兩邊若分歧,診斷分解出來的數字就對不上誤併率與碎裂率,整份分析失去意義。

⚠ 為什麼可以用 base 那一格算出來的表去餵其他格:
  `m5_track_video.py` 的 tracker 只吃偵測結果,**M5 從不回饋到 M4**,
  所以 `(camera, track_id)` 在所有 M5 變體之間逐條相同。
  這也避免了「oracle 改變了 track id」這種循環依賴。

用法:
    python scripts/make_track_gt.py \
        --root "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA" \
        --run-dir results/chirla_p1grid/base/seq_004 ... \
        --out-dir results/track_gt
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_m4m5_chirla import load_gt, load_run, match_tracks   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CHIRLA 根目錄")
    ap.add_argument("--run-dir", nargs="+", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "track_gt"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'序列':<10}{'track':>8}{'有真值':>8}{'誤偵':>8}{'誤偵率':>9}   輸出")
    print("-" * 72)
    for rd in args.run_dir:
        meta, tracks, _events = load_run(rd)
        seq = next(s for s in Path(next(iter(meta["camera_video_map"].values()))).parts
                   if s.startswith("seq_"))
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
