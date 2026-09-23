"""逐片段獨立推論的硬性驗收 V1–V10(預先登記 §8)。

⚠ 順序是硬的:V6 / V6b / V6c 沒過之前,不看任何真實數字 ——
  指標模組沒被證明「會失敗」之前,它給出的好看數字沒有意義。

用法:
    python scripts/verify_clip_runs.py --root "D:/.../CHIRLA" --checks V6 V6b V6c
    python scripts/verify_clip_runs.py --root "D:/.../CHIRLA" --checks all
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.clip_metrics import clip_metrics                 # noqa: E402
from common.video_io import iter_frames                      # noqa: E402
from eval_m4m5_chirla import load_gt                         # noqa: E402

DELIVERY = "results/m5_step3/m5_cbiou"


def synth(clip, stride, gt, mode="identity"):
    """把標註當成追蹤輸出。mode:
       identity — track_id = chef_id = gt_id(完美系統)
       permuted — 片段中點之後把目標的 chef_id 換掉(碎裂)
       merged   — 窗內所有人共用同一個 chef_id(全部併成一個)
    """
    tracks, chefs = defaultdict(list), {}
    fids = list(range(clip["start_fid"], clip["end_fid"] + 1, stride))
    mid = fids[len(fids) // 2] if fids else 0
    for fid in fids:
        for cam in clip["cameras"]:
            for gid, box in gt.get(cam, {}).get(fid + 1, []):
                tid = gid if mode != "permuted" or fid < mid or gid != clip["gt_id"] else gid + 900
                tracks[(cam, fid)].append((tid, box))
                chefs[(cam, tid)] = 1 if mode == "merged" else tid
    return tracks, chefs


def v6_family(man, root, mode):
    gts, bad = {}, []
    for clip in man["clips"]:
        if clip["seq"] not in gts:
            gts[clip["seq"]] = load_gt(root, clip["seq"])
        gt = gts[clip["seq"]]
        tracks, chefs = synth(clip, 5, gt, mode)
        _rows, m = clip_metrics(clip, 5, gt, tracks, chefs)
        cid = clip["clip_id"]
        if mode == "identity":
            if m["recall_target"] != 1.0 or m["mean_iou"] != 1.0:
                bad.append(f"{cid}:召回 {m['recall_target']} / IoU {m['mean_iou']} 不是 1.0")
            if m["p_continuity"] != 1.0 or m["p_majority"] != 1.0:
                bad.append(f"{cid}:連續 {m['p_continuity']} / 多數 {m['p_majority']} 不是 1.0")
            if m["chef_switches"] or m["n_ghost_boxes"]:
                bad.append(f"{cid}:切換 {m['chef_switches']} / 誤偵 {m['n_ghost_boxes']} 不是 0")
            if clip["level"] == "L2" and m["l2_n_pairs_k5"] and m["l2_p_reacquire_k5"] != 1.0:
                bad.append(f"{cid}:L2 接回 {m['l2_p_reacquire_k5']} 不是 1.0")
            if clip["level"] == "L3" and m["l3_n_pairs_usable"] and m["l3_p_handoff"] != 1.0:
                bad.append(f"{cid}:L3 一致 {m['l3_p_handoff']} 不是 1.0")
        elif mode == "permuted":
            if m["chef_switches"] != 1:
                bad.append(f"{cid}:切換 {m['chef_switches']} 不是 1")
            if m["p_continuity"] == 1.0:
                bad.append(f"{cid}:連續率仍是 1.0 —— 指標無法失敗")
            if m["p_majority"] < m["p_continuity"]:
                bad.append(f"{cid}:多數率 {m['p_majority']} < 連續率 {m['p_continuity']}")
        elif mode == "merged":
            if clip["exclusive"]:
                if m["p_exclusive"] is not None:
                    bad.append(f"{cid}:獨佔窗的排他率應為 None,得到 {m['p_exclusive']}")
            else:
                if m["p_exclusive"] != 0.0:
                    bad.append(f"{cid}:併成一個編號時排他率應為 0,得到 {m['p_exclusive']}")
                if m["p_continuity"] != 1.0:
                    bad.append(f"{cid}:併成一個編號時連續率應為 1.0")
    return bad


def v6d_injected_second_person(man, root, n=5):
    """⚠ 這批 67 段**全部獨佔**,所以 V6c 的「併成一個編號 → 排他率 0」分支
    在真實片段上永遠走不到。這裡**合成**一個第二人(把目標的框平移),
    讓兩人共用同一個編號,證明排他率真的會歸零 —— 否則 V6c 等於沒驗。
    """
    gts, bad = {}, []
    for clip in man["clips"][:n]:
        if clip["seq"] not in gts:
            gts[clip["seq"]] = load_gt(root, clip["seq"])
        gt = {cam: {f: list(v) for f, v in d.items()}
              for cam, d in gts[clip["seq"]].items()}
        fake = 9999
        for cam in clip["cameras"]:
            for f, dets in gt.get(cam, {}).items():
                for gid, (x1, y1, x2, y2) in list(dets):
                    if gid == clip["gt_id"]:
                        w = x2 - x1
                        dets.append((fake, (x1 + 2 * w, y1, x2 + 2 * w, y2)))
        c2 = dict(clip, exclusive=False, n_other_ids=1)
        tracks, chefs = synth(c2, 5, gt, "merged")
        _rows, m = clip_metrics(c2, 5, gt, tracks, chefs)
        if m["p_exclusive"] != 0.0:
            bad.append(f"{clip['clip_id']}:注入第二人並共用編號後,排他率應為 0,"
                       f"得到 {m['p_exclusive']}")
        if m["p_continuity"] != 1.0 or m["p_majority"] != 1.0:
            bad.append(f"{clip['clip_id']}:連續 / 多數率應仍為 1.0("
                       f"{m['p_continuity']} / {m['p_majority']})—— 這正是它們無法否證的證明")
    return bad


def v1_manifest(man, root):
    bad, gts = [], {}
    ids = set()
    for c in man["clips"]:
        if c["clip_id"] in ids:
            bad.append(f"{c['clip_id']}:重複")
        ids.add(c["clip_id"])
        if (c["start_frame"] - 1) % man["grid"]:
            bad.append(f"{c['clip_id']}:起始幀不在共用格點")
        if c["seq"] not in gts:
            gts[c["seq"]] = load_gt(root, c["seq"])
        for cam in c["cameras"]:
            g = gts[c["seq"]].get(cam, {})
            hit = any(c["gt_id"] in {i for i, _b in g.get(f, [])}
                      for f in range(c["start_frame"], c["end_frame"] + 1))
            if not hit:
                bad.append(f"{c['clip_id']}:目標在 {cam} 窗內無標註")
            meta = Path(c["det_cache"]) / f"{cam}.npz"
            if not meta.exists():
                bad.append(f"{c['clip_id']}:缺快取 {cam}")
    if len(man["clips"]) != 67:
        bad.append(f"段數 {len(man['clips'])} 不是 67")
    return bad


def v2_v3_runs(man, run_root, strides):
    bad, ref = [], None
    for c in man["clips"]:
        for s in strides:
            d = Path(run_root) / c["clip_id"] / f"s{s}"
            if not (d / "run_meta.json").exists():
                bad.append(f"{c['clip_id']} s{s}:缺 run_meta")
                continue
            m = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
            exp = (c["end_fid"] - c["start_fid"]) // s + 1
            if m["n_loops"] != exp:
                bad.append(f"{c['clip_id']} s{s}:迴圈 {m['n_loops']} ≠ 預期 {exp}")
            if m["truncated"] or abs(m["coverage"] - 1.0) > 1e-9:
                bad.append(f"{c['clip_id']} s{s}:coverage {m['coverage']} truncated {m['truncated']}")
            key = {k: m[k] for k in ("tracker_cfg", "thr", "embedder", "ttl_loops",
                                     "topology_path", "person_cls")}
            key["ttl_loops"] = 600      # 次要對照會不同,主網格才比
            if ref is None:
                ref = key
            elif key != ref and s in strides:
                diff = [k for k in key if key[k] != ref[k]]
                bad.append(f"{c['clip_id']} s{s}:設定與其他段不同 {diff}")
            # V3:幀號格點
            with open(d / "tracks.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    fid, loop = int(r["video_fid"]), int(r["loop_i"])
                    if fid % s or not (c["start_fid"] <= fid <= c["end_fid"]) \
                            or fid != c["start_fid"] + loop * s:
                        bad.append(f"{c['clip_id']} s{s}:fid {fid} loop {loop} 不合格點")
                        break
    return bad


def v4_seek(man, n=5, seed=0):
    import hashlib
    import random
    bad = []
    rnd = random.Random(seed)
    for c in rnd.sample(man["clips"], min(n, len(man["clips"]))):
        for cam in c["cameras"]:
            v = c["videos"][cam]
            want = None
            for fid, _t, fr in iter_frames(v, stride=1, end=c["start_fid"]):
                if fid == c["start_fid"]:
                    want = hashlib.sha256(fr.tobytes()).hexdigest()
            got = None
            for fid, _t, fr in iter_frames(v, stride=5, start=c["start_fid"]):
                got = hashlib.sha256(fr.tobytes()).hexdigest()
                break
            if want != got:
                bad.append(f"{c['clip_id']} {cam}:快轉到 {c['start_fid']} 的幀不一致")
    return bad


def v5_cache(man, run_root, strides):
    bad = []
    deliv = {}
    for c in man["clips"]:
        for s in strides:
            p = Path(run_root) / c["clip_id"] / f"s{s}" / "run_meta.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            dp = Path(DELIVERY) / c["seq"] / "run_meta.json"
            if not dp.exists():
                continue
            if c["seq"] not in deliv:
                deliv[c["seq"]] = json.loads(dp.read_text(encoding="utf-8"))["det_cache_meta"]
            for cam in c["cameras"]:
                a = {k: v for k, v in deliv[c["seq"]][cam].items() if k != "video"}
                b = {k: v for k, v in m["det_cache_meta"][cam].items() if k != "video"}
                if a != b:
                    bad.append(f"{c['clip_id']} s{s} {cam}:快取中繼資料與交付版不同")
    return bad


def v9_redlines():
    bad = []
    r = subprocess.run(["git", "status", "--porcelain",
                        "results/m5_step3", "results/levels/videos", "src/m5_sim/world.py"],
                       capture_output=True, text=True, cwd=ROOT)
    if r.stdout.strip():
        bad.append("紅線目錄有變更:\n" + r.stdout.strip())
    return bad


def v10_determinism(man, run_root, clip_ids, stride=5):
    """重跑幾段,四個輸出檔必須逐位相同。"""
    import hashlib
    import shutil
    bad = []
    for cid in clip_ids:
        c = next(x for x in man["clips"] if x["clip_id"] == cid)
        src = Path(run_root) / cid / f"s{stride}"
        tmp = Path("results/clips/regress/determinism") / cid
        tmp.mkdir(parents=True, exist_ok=True)
        cmd = [str(ROOT / ".venv/Scripts/python.exe"), "scripts/run_clip_grid.py",
               "--clips", cid, "--strides", str(stride), "--out-root", str(tmp),
               "--index", str(tmp / "idx.csv")]
        subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
        # ⚠ 不比 resident.csv:它逐列記 rss_mb(行程記憶體),每次執行本來就不同;
        #   決定性要看的是決策輸出。V0 也是同一個處置。
        for f in ("tracks.csv", "chef_events.jsonl", "track_events.csv"):
            a, b = src / f, tmp / cid / f"s{stride}" / f
            if not b.exists():
                bad.append(f"{cid}:重跑缺 {f}")
                continue
            if hashlib.sha256(a.read_bytes()).hexdigest() != \
               hashlib.sha256(b.read_bytes()).hexdigest():
                bad.append(f"{cid}:{f} 重跑不一致")
        shutil.rmtree(tmp, ignore_errors=True)
    return bad


CHECKS = ("V1", "V2V3", "V4", "V5", "V6", "V6b", "V6c", "V6d", "V9", "V10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", default="results/clips/clip_manifest.json")
    ap.add_argument("--run-root", default="results/clips/runs")
    ap.add_argument("--strides", nargs="+", type=int, default=[5, 1])
    ap.add_argument("--checks", nargs="+", default=["all"])
    ap.add_argument("--out", default="results/clips/agg/verification.json")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    want = set(CHECKS) if "all" in args.checks else set(args.checks)
    res = {}
    runner = {
        "V1": lambda: v1_manifest(man, args.root),
        "V2V3": lambda: v2_v3_runs(man, args.run_root, args.strides),
        "V4": lambda: v4_seek(man),
        "V5": lambda: v5_cache(man, args.run_root, args.strides),
        "V6": lambda: v6_family(man, args.root, "identity"),
        "V6b": lambda: v6_family(man, args.root, "permuted"),
        "V6c": lambda: v6_family(man, args.root, "merged"),
        "V6d": lambda: v6d_injected_second_person(man, args.root),
        "V9": v9_redlines,
        "V10": lambda: v10_determinism(man, args.run_root,
                                       [c["clip_id"] for c in man["clips"][:3]]),
    }
    for name in CHECKS:
        if name not in want:
            continue
        bad = runner[name]()
        res[name] = dict(passed=not bad, n_failures=len(bad), failures=bad[:20])
        print(f"{name:5} {'PASS' if not bad else f'FAIL ({len(bad)} 項)'}")
        for b in bad[:5]:
            print("   -", b)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已存 {out}")
    return 0 if all(v["passed"] for v in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
