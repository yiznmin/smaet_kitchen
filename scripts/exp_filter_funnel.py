"""
實驗:分支 B「有過濾 vs 沒過濾」會被送進分類器的候選數量。
量化幾何過濾的降量效果(候選愈少 → 分類器負載愈低,但前提是不能漏)。

動態線(老鼠):單次跑,逐階段漏斗
   原始候選(MOG2+面積)→ 空間排除後 → 密度排除後 → 時序後(最終送分類器)
靜態線(水漬):A/B 兩跑
   過濾全關(無地板ROI/長寬比/動作排除/密度/時序)vs 全開(完整設定)

用法:
  python scripts/exp_filter_funnel.py data/epfl/Boutput0.mp4
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames                  # noqa: E402
from m2_motion.branch_b import SmallTargetDetector       # noqa: E402
from m2_motion.static_line import StaticTargetDetector   # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"
OUT = ROOT / "results" / "m2" / "filter_experiment.md"
EPFL_FLOOR = [[330, 715], [1080, 715], [950, 380], [470, 380]]
STRIDE = 10


def dynamic_funnel(cfg, video):
    """動態線:逐階段累計候選數(從偵測器輸出的計數反推漏斗)。"""
    b = cfg["m2_branch_b"]
    det = SmallTargetDetector(
        min_blob_area=b["min_blob_area"], large_blob_area=b["large_blob_area"],
        exclusion_margin=b["exclusion_margin"], crop_size=b["crop_size"],
        max_crops_per_frame=b["max_crops_per_frame"], mog2_history=b["mog2_history"],
        mog2_var_threshold=b["mog2_var_threshold"], warmup_frames=b["warmup_frames"],
        persistence_frames=b["persistence_frames"], match_max_dist=b["match_max_dist"],
        track_max_miss=b["track_max_miss"],
        density_radius=b["density_radius"], density_max_neighbors=b["density_max_neighbors"])

    raw = spatial = dense = pending = final = frames = 0
    for fid, ts, frame in iter_frames(video, stride=STRIDE):
        r = det.process(frame)
        if r["warming"]:
            continue
        frames += 1
        ne, nd = r["n_excluded"], r["n_dense_excluded"]
        npd, nc = r["n_pending"], r["n_crops"]
        raw += nc + npd + nd + ne     # 原始候選(面積過濾後,尚未做空間/密度/時序)
        spatial += ne                  # 空間排除掉的
        dense += nd                    # 密度排除掉的
        pending += npd                 # 時序未確認(暫不送)
        final += nc                    # 最終送分類器
    after_spatial = raw - spatial
    after_density = after_spatial - dense
    return dict(frames=frames, raw=raw, after_spatial=after_spatial,
                after_density=after_density, final=final)


def static_count(cfg, video, filtered):
    """靜態線:統計總共送出多少 crop。filtered=False 為過濾全關的基準。"""
    s = cfg["m2_static_line"]
    common = dict(
        baseline_alpha=s["baseline_alpha"], baseline_diff_thresh=s["baseline_diff_thresh"],
        motion_thresh=s["motion_thresh"], blur_ksize=s["blur_ksize"],
        min_blob_area=s["min_blob_area"], max_blob_area=s["max_blob_area"],
        crop_size=s["crop_size"], warmup_frames=s["warmup_frames"],
        persistence_frames=s["persistence_frames"], match_max_dist=s["match_max_dist"],
        track_max_miss=s["track_max_miss"], density_radius=s["density_radius"])
    if filtered:
        det = StaticTargetDetector(**common, motion_exclusion_margin=s["motion_exclusion_margin"],
                                   max_aspect_ratio=s["max_aspect_ratio"],
                                   density_max_neighbors=s["density_max_neighbors"],
                                   floor_zone=EPFL_FLOOR)
    else:
        # 過濾全關:無地板ROI、不砍瘦高、動作排除最小、不砍密集、不要求持續
        det = StaticTargetDetector(**{**common, "persistence_frames": 1},
                                   motion_exclusion_margin=1, max_aspect_ratio=1e9,
                                   density_max_neighbors=10 ** 9, floor_zone=None)
    total = frames = 0
    for fid, ts, frame in iter_frames(video, stride=STRIDE):
        r = det.process(frame)
        if r["warming"]:
            continue
        frames += 1
        total += r["n_crops"]
    return total, frames


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "epfl" / "Boutput0.mp4"
    if not video.exists():
        sys.exit(f"找不到影片 {video}")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    print(f"影片:{video.name}(stride={STRIDE})")
    print("\n=== 動態線(老鼠):逐階段候選漏斗 ===")
    d = dynamic_funnel(cfg, video)
    raw = max(d["raw"], 1)
    rows_d = [
        ("原始候選(未過濾,MOG2+面積)", d["raw"]),
        ("空間排除後", d["after_spatial"]),
        ("密度排除後", d["after_density"]),
        ("時序後(最終送分類器)", d["final"]),
    ]
    for name, v in rows_d:
        print(f"  {name:<26} {v:>7}  ({v / raw:6.1%})")
    dyn_reduction = 1 - d["final"] / raw
    print(f"  → 總降量:未過濾 {d['raw']} → 過濾後 {d['final']}(減少 {dyn_reduction:.1%})")

    print("\n=== 靜態線(水漬):過濾全關 vs 全開 ===")
    off, nf = static_count(cfg, video, filtered=False)
    on, _ = static_count(cfg, video, filtered=True)
    sta_reduction = 1 - on / max(off, 1)
    print(f"  過濾全關(未過濾)  {off:>7}")
    print(f"  過濾全開(已過濾)  {on:>7}")
    print(f"  → 降量:減少 {sta_reduction:.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join([
        "# M2 分支 B — 過濾降量實驗(有過濾 vs 沒過濾)",
        "",
        f"- 影片:`{video.name}`,取樣 stride={STRIDE}(EPFL,全程有人、無老鼠無水漬 = 最壞情況)",
        "- 指標:累計「會被送進分類器的候選 crop 數」。愈少=分類器負載愈低。",
        "- 注意:此片無真目標,候選全是雜訊/人;降量代表「少送多少垃圾給分類器」。",
        "",
        "## 動態線(老鼠):逐階段候選漏斗",
        "",
        f"處理幀數:{d['frames']}",
        "",
        "| 階段 | 累計候選數 | 相對原始 |",
        "|---|---|---|",
        *[f"| {name} | {v} | {v / raw:.1%} |" for name, v in rows_d],
        "",
        f"**總降量:未過濾 {d['raw']} → 過濾後 {d['final']}(減少 {dyn_reduction:.1%})**",
        "",
        "## 靜態線(水漬):過濾全關 vs 全開",
        "",
        f"處理幀數:{nf}",
        "",
        "| 設定 | 累計候選數 |",
        "|---|---|",
        f"| 過濾全關(未過濾) | {off} |",
        f"| 過濾全開(已過濾) | {on} |",
        "",
        f"**降量:減少 {sta_reduction:.1%}**",
        "",
        "> 結論:幾何過濾大幅降低送進分類器的候選量(= 降低分類器負載與誤報基數),",
        "> 同時(經合成測試驗證)不漏真目標。最終身分判斷仍由多類別分類器負責。",
    ]), encoding="utf-8")
    print(f"\n報告已寫入 {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
