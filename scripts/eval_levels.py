"""難度分級評估的編排 —— 自己不實作任何指標。

## 為什麼是「編排」

repo 裡已經有兩種 IDF1:`m5_reid.metrics.idf1`(跨鏡頭、以綁定決策為單位)與
`trackers.eval`(單鏡頭、逐幀,TrackEval 對齊)。**再寫第三種就是負債** ——
兩個數字不一致時沒有人知道哪個才對。所以這支只做三件事:
讀 manifest、把既有函式套到窗上、把結果排成表。

  L1 / L2  單鏡頭 → `eval_m4_idf1.unit_inputs(..., fid_range=)` + `trackers.eval`
  L3       跨鏡頭 → `eval_m4m5_chirla` 的 `match_tracks` / `build_records` + `metrics.summarize`

## ⚠ match_tracks 一律在整段序列上跑

`match_tracks` 是用 IoU 多數決決定「這條 track 是誰」的。若先把 track 截到窗內再配對,
同一條 track 可能被判成不同的人,窗內的數字就不再能和已提交的全序列基線比較。
所以:**配對用整段序列,只有事件被濾到窗內。**

## ⚠ L1/L2 的誤併率結構上不存在

窗內只有一個身份,`binding_outcomes` 的「綁到別人身上」那條分支是死碼,
所以它回 `None` 而不是 0(那是本專案踩過最嚴重的坑的形狀)。
這支把它印成「不可量測」,而且若 L1/L2 真的算出數字就**讓整輪失敗** ——
那表示挑窗時的獨佔過濾漏了。

用法:
    python scripts/eval_levels.py --root "D:/.../CHIRLA" \\
        --manifest results/levels/level_manifest.json \\
        --run-root results/m5_step3/m5_cbiou --label m5_cbiou \\
        --out-dir results/levels/m5_cbiou
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_m4m5_chirla import build_records, load_gt, load_run, match_tracks   # noqa: E402
from m5_reid import metrics                                                  # noqa: E402
from m5_reid.spatiotemporal import CameraTopology                            # noqa: E402

UNMEASURABLE = "不可量測(窗內只有 1 個身份)"


def fmt(v, ok):
    """指標格式化。⚠ 量不到就印理由,不印 0%、不印破折號 —— 破折號會被當成 0。"""
    if not ok or v is None:
        return UNMEASURABLE
    return f"{v * 100:.2f}%"


class SeqCache:
    """一個序列讀一次:run_meta、tracks、events、真值、track→GT 配對。"""

    def __init__(self, root, run_root, seq, topo):
        rd = Path(run_root) / seq
        self.run_dir = rd if rd.exists() else Path(run_root)
        self.meta, self.tracks, self.events = load_run(self.run_dir)
        self.stride = int(self.meta["stride"])
        self.gt = load_gt(root, seq)
        # ⚠ 整段序列配對,不截窗 —— 見檔頭
        self.track_gt, self.tstats, _gtm, _gtt = match_tracks(self.gt, self.tracks)
        self.topo, self.seq = topo, seq
        self.by_cam_fid = defaultdict(lambda: defaultdict(list))
        for t in self.tracks:
            self.by_cam_fid[t["cam"]][t["fid"]].append((t["tid"], t["bbox"]))

    def window_records(self, win):
        ev = [e for e in self.events
              if win["start_fid"] <= e["video_fid"] <= win["end_fid"]
              and e["camera_id"] in win["cameras"]]
        recs, ghost, path = build_records(ev, self.track_gt, self.topo.overlapping, self.seq)
        return recs, ghost, path, len(ev)


def l1l2_metrics(cache, win):
    """單鏡頭 IDF1 / IDSW。⚠ 需要 trackers 套件(遠端已裝;本機沒有)。"""
    from eval_m4_idf1 import unit_inputs
    from trackers.eval.clear import compute_clear_metrics
    from trackers.eval.identity import compute_identity_metrics

    cam = win["cameras"][0]
    gids, tids, sims = unit_inputs(cache.gt[cam], cache.by_cam_fid[cam], cache.stride,
                                   fid_range=(win["start_fid"], win["end_fid"]))
    i_m, c_m = compute_identity_metrics(gids, tids, sims), compute_clear_metrics(gids, tids, sims)
    return dict(IDF1=i_m["IDF1"], IDR=i_m["IDR"], IDP=i_m["IDP"],
                IDSW=c_m["IDSW"], MOTA=c_m["MOTA"]), (i_m, c_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CHIRLA 根目錄")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-root", required=True, help="含 <seq>/tracks.csv 的 M5 輸出")
    ap.add_argument("--topology", default=str(ROOT / "configs" / "camera_topology.chirla.yaml"))
    ap.add_argument("--label", default="run")
    ap.add_argument("--levels", nargs="+", default=["L1", "L2", "L3"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    wins = [w for w in man["windows"] if w["level"] in args.levels]
    if not wins:
        raise SystemExit("manifest 裡沒有符合的窗")
    topo = CameraTopology.from_yaml(args.topology)

    caches, rows, fails = {}, [], []
    single_cam_agg = defaultdict(lambda: ([], []))     # (level, split) -> (identity, clear)
    rec_pool = defaultdict(list)                       # (level, split) -> records
    path_pool = defaultdict(lambda: {"overlap": [], "transit": []})

    for w in sorted(wins, key=lambda x: (x["level"], x["split"], x["seq"], x["start_frame"])):
        key = (w["split"], w["seq"])
        if key not in caches:
            caches[key] = SeqCache(args.root, args.run_root, w["seq"], topo)
        c = caches[key]
        recs, ghost, path, n_ev = c.window_records(w)
        s = metrics.summarize(recs) if recs else dict(measurable={}, n_transitions=0)
        row = dict(window_id=w["window_id"], level=w["level"], split=w["split"],
                   seq=w["seq"], cameras=w["cameras"], gt_id=w["gt_id"],
                   duration_s=w["duration_s"], n_sampled_frames=w["n_sampled_frames"],
                   gap_band=w.get("gap_band"), rendered=w["rendered"],
                   exclusive=w["exclusive"], n_other_ids=w["n_other_ids"],
                   n_bindings=n_ev, n_ghost_bindings=ghost,
                   n_transitions=s.get("n_transitions"),
                   p_break=s.get("p_break"), p_false_merge=s.get("p_false_merge"),
                   p_correct=s.get("p_correct"),
                   fragmentation=(s.get("fragmentation") or {}).get("mean"),
                   measurable=s.get("measurable", {}))

        # 風險 1 的硬性防護:L1/L2 若算出數字,表示獨佔過濾漏了
        if w["level"] in ("L1", "L2") and s.get("p_false_merge") is not None:
            fails.append(f"{w['window_id']}:L1/L2 竟然算得出誤併率 —— 窗內不只一個身份")

        if w["level"] in ("L1", "L2"):
            try:
                m, (i_m, c_m) = l1l2_metrics(c, w)
                row.update(m)
                single_cam_agg[(w["level"], w["split"])][0].append(i_m)
                single_cam_agg[(w["level"], w["split"])][1].append(c_m)
            except ImportError as e:
                row["IDF1"] = None
                fails.append(f"{w['window_id']}:算不了單鏡頭 IDF1({e})")
        else:
            for k in path_pool[(w["level"], w["split"])]:
                path_pool[(w["level"], w["split"])][k] += path[k]
        rec_pool[(w["level"], w["split"])] += recs
        rows.append(row)

    # ── 彙總:記錄直接彙集後算一次(與逐窗平均不同,分母才正確)──
    agg = {}
    for (level, split), recs in sorted(rec_pool.items()):
        a = dict(n_windows=sum(1 for r in rows if r["level"] == level and r["split"] == split),
                 n_bindings=len(recs))
        a.update({k: v for k, v in (metrics.summarize(recs) if recs else {}).items()
                  if k in ("n_transitions", "p_break", "p_false_merge", "p_correct",
                           "idf1", "id_switches", "fragmentation", "measurable",
                           "fm_unmeasurable_reason")})
        if (level, split) in single_cam_agg and single_cam_agg[(level, split)][0]:
            from trackers.eval.clear import aggregate_clear_metrics
            from trackers.eval.identity import aggregate_identity_metrics
            i_l, c_l = single_cam_agg[(level, split)]
            ai, ac = aggregate_identity_metrics(i_l), aggregate_clear_metrics(c_l)
            a["single_cam"] = dict(IDF1=ai["IDF1"], IDR=ai["IDR"], IDP=ai["IDP"],
                                   IDSW=ac["IDSW"], MOTA=ac["MOTA"])
        if level == "L3":
            a["paths"] = {k: (metrics.summarize(v) if v else None)
                          for k, v in path_pool[(level, split)].items()}
        agg[f"{level}|{split}"] = a

    # ── 報表 ──
    L = [f"# 難度分級評估:{args.label}", "",
         f"manifest `{args.manifest}` · run `{args.run_root}`", "",
         "⚠ 評估只看窗內選中的鏡頭,但**系統執行時 7 台鏡頭都在線** —— "
         "這些數字不代表「只裝 2~3 台」的部署。", ""]
    for k, a in agg.items():
        level, split = k.split("|")
        L.append(f"## {level} · {split}(窗 {a['n_windows']} 個,綁定 {a['n_bindings']} 次)")
        meas = a.get("measurable", {})
        L.append("")
        L.append("| 指標 | 值 |")
        L.append("|---|---|")
        if "single_cam" in a:
            sc = a["single_cam"]
            L.append(f"| 單鏡頭 IDF1 | {sc['IDF1']:.4f} |")
            L.append(f"| ID 切換 | {sc['IDSW']} |")
        L.append(f"| 正確率 | {fmt(a.get('p_correct'), meas.get('p_break', True))} |")
        L.append(f"| 碎裂率 | {fmt(a.get('p_break'), meas.get('p_break', True))} |")
        L.append(f"| **誤併率** | {fmt(a.get('p_false_merge'), meas.get('p_false_merge', False))} |")
        L.append(f"| 每人被拆成幾個身份 | {(a.get('fragmentation') or {}).get('mean')} |")
        if a.get("paths"):
            for pk, pv in a["paths"].items():
                if pv:
                    L.append(f"| {pk} 路徑:碎裂 / 誤併 | "
                             f"{fmt(pv.get('p_break'), True)} / "
                             f"{fmt(pv.get('p_false_merge'), pv['measurable']['p_false_merge'])} |")
        L.append("")
        # 風險 2 的防護:被拍成影片的那幾個窗,自己的數字要印在平均旁邊
        shown = [r for r in rows if r["rendered"] and r["level"] == level and r["split"] == split]
        if shown:
            L.append("被渲染成影片的窗(供對照,判斷它是否典型):")
            L.append("")
            L.append("| 窗 | 秒 | 正確率 | 碎裂率 | 誤併率 |")
            L.append("|---|---|---|---|---|")
            for r in shown:
                L.append(f"| `{r['window_id']}` | {r['duration_s']:.0f} | "
                         f"{fmt(r['p_correct'], True)} | {fmt(r['p_break'], True)} | "
                         f"{fmt(r['p_false_merge'], r['measurable'].get('p_false_merge', False))} |")
            L.append("")

    out_dir = Path(args.out_dir or ROOT / "results" / "levels" / args.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(
        dict(label=args.label, manifest=args.manifest, run_root=args.run_root,
             aggregate=agg, windows=rows, failures=fails),
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n".join(L))
    print(f"已存 {out_dir}/summary.{{md,json}}")
    if fails:
        print(f"\n[FAIL] {len(fails)} 項:")
        for f in fails[:10]:
            print(f"  - {f}")
        return 1
    print("\n[OK] 沒有結構性問題")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
