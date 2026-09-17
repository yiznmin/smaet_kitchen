"""B 套 Level 定義:把 CHIRLA 資料夾裡的 Level CSV 轉成與 `chirla_select_levels.py` 同結構的 manifest。

## 為什麼要轉換,而不是另寫評估

A 套(`chirla_select_levels.py`)與 B 套(資料夾 CSV,2026-09-17 出現)定義不同,
使用者決定兩套都測、分開報。評估一律走同一支 `eval_levels.py` ——
再寫第二套指標,兩個數字不一致時沒有人知道哪個才對。
所以這支只做結構轉換,判定函式全部 import 自 `chirla_select_levels`。

## ⚠ 不信任 CSV 自己的欄位

`exclusive` / `n_other_ids` 用標註在取樣幀上重算;CSV 只提供「哪個人、哪幾台、哪一段」。

## ⚠ rendered 由人工確認決定

`manual_check/review_sheet.csv` 的 `result` 判為符合的片段才標 `rendered=True`;
沒填或不符合一律 False(使用者 2026-09-17 決定:先人工確認片段符合條件)。

用法見 `docs/難度分級交付_補充登記_20260917.md` §7.1。
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chirla_select_levels import (FPS, build_tables, make_window, pair_class,   # noqa: E402
                                  selfcheck)
from eval_m4m5_chirla import load_gt                                            # noqa: E402

CSV_FILES = dict(L1="level1_single_cam_continuous.csv",
                 L2="level2_single_cam_reappear.csv",
                 L3="level3_cross_camera.csv")
PASS_VALUES = {"符合", "通過", "是", "ok", "pass", "yes", "y", "v", "o"}
FAIL_VALUES = {"不符合", "不通過", "否", "ng", "fail", "no", "n", "x"}


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def align_in(lo, hi, stride):
    """起點往後、終點往前對齊到取樣格點((frame−1) % stride == 0);窗只縮不擴。"""
    lo2 = lo + ((1 - lo) % stride)
    hi2 = hi - ((hi - 1) % stride)
    return lo2, hi2


def spans(text):
    return [(int(a), int(b)) for a, b in re.findall(r"\[(\d+)-(\d+)\]", text)]


def cam_spans(text):
    return [(f"camera_{c}", int(a), int(b))
            for c, a, b in re.findall(r"c(\d+)\[(\d+)-(\d+)\]", text)]


def load_review(path):
    """{(level, seq, 鏡頭組, person_id, start, end): (sample_id, 判定)};判定 ∈ {True, False, None}。"""
    out, bad = {}, []
    if not path:
        return out, bad
    for r in read_csv(path):
        cams = tuple(sorted(f"camera_{c[1:]}" for c in r["cameras"].split("+")))
        v = (r.get("result") or "").strip().lower()
        verdict = True if v in PASS_VALUES else False if v in FAIL_VALUES else None
        if v and verdict is None:
            bad.append(f"{r['sample_id']}:看不懂 result「{r['result']}」")
        key = (r["sample_id"][:2], r["seq"], cams, int(r["person_id"]),
               int(r["start_frame"]), int(r["end_frame"]))
        out[key] = (r["sample_id"], verdict)
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CHIRLA 根目錄(讀標註)")
    ap.add_argument("--csv-dir", required=True, help="Level CSV 所在目錄")
    ap.add_argument("--review-sheet", default=None, help="manual_check/review_sheet.csv")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--topology", default=str(ROOT / "configs" / "camera_topology.chirla.yaml"))
    ap.add_argument("--train-seqs", nargs="+", default=[])
    ap.add_argument("--test-seqs", nargs="+", default=[])
    ap.add_argument("--extra-seqs", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits = dict(train=args.train_seqs, test=args.test_seqs, extra=args.extra_seqs)
    split_of = {s: k for k, v in splits.items() for s in v}
    st = args.stride
    with open(args.topology, encoding="utf-8") as f:
        topo = (yaml.safe_load(f) or {}).get("camera_topology", {})
    overlapping = {frozenset(x) for x in topo.get("overlapping", [])}
    review, review_bad = load_review(args.review_sheet)

    rows = {lv: read_csv(Path(args.csv_dir) / name) for lv, name in CSV_FILES.items()}
    tables, windows, skipped, matched = {}, [], [], set()

    def tables_for(seq):
        if seq not in tables:
            gt = {c: v for c, v in load_gt(args.root, seq).items() if v}
            tables[seq] = build_tables(gt, st)
        return tables[seq]

    for lv, rs in rows.items():
        for i, r in enumerate(rs):
            seq, gid = r["seq"], int(r["person_id"])
            if seq not in split_of:
                skipped.append(f"{lv}#{i}:{seq} 不在任何分桶")
                continue
            split = split_of[seq]
            occ, _pres = tables_for(seq)
            lo, hi = align_in(int(r["start_frame"]), int(r["end_frame"]), st)
            common = dict(source="csv", csv_row=dict(level=lv, index=i, **r))

            if lv in ("L1", "L2"):
                cams = [r["camera"]]
                if lv == "L1":
                    rule = "L1"
                    extra = dict(segments_s=[round((int(r["end_frame"]) - int(r["start_frame"])) / FPS, 3)],
                                 gaps_s=[], gap_band=None, trimmed=False)
                else:
                    rule = "L2"
                    ap_ = spans(r["appearances"])
                    gaps = [round((b[0] - a[1]) / FPS, 3) for a, b in zip(ap_, ap_[1:])]
                    extra = dict(segments_s=[round((e - s) / FPS, 3) for s, e in ap_],
                                 gaps_s=gaps, trimmed=False,
                                 gap_band="within_buffer" if max(gaps) <= 5.0 else "beyond_buffer")
            else:
                segs = cam_spans(r["segments"])
                cams = sorted({c for c, _, _ in segs})
                if r["handoff_type"] == "overlapping":
                    rule = "L3S"
                    best = 0
                    for (ca, sa, ea), (cb, sb, eb) in combinations(segs, 2):
                        if ca != cb:
                            best = max(best, min(ea, eb) - max(sa, sb))
                    extra = dict(segments_s=[], gaps_s=[], gap_band=None, transitions=[],
                                 overlap_s=round(best / FPS, 3), trimmed=False,
                                 pair_classes=sorted({pair_class(a, b, overlapping)
                                                      for a, b in combinations(cams, 2)}))
                elif r["handoff_type"] == "sequential":
                    rule = "L3T"
                    order = sorted(segs, key=lambda x: (x[1], x[2]))
                    tr = next(((a, b) for a, b in zip(order, order[1:])
                               if a[0] != b[0] and b[1] > a[2]), None)
                    if tr is None:
                        skipped.append(f"{lv}#{i}:sequential 但找不到不重疊的轉場")
                        continue
                    (c1, _s1, e1), (c2, s2, _e2) = tr
                    dt = round((s2 - e1) / FPS, 3)
                    pc = pair_class(c1, c2, overlapping)
                    extra = dict(segments_s=[], gaps_s=[dt], gap_band=None, overlap_s=None,
                                 trimmed=False,
                                 transitions=[dict(**{"from": c1}, to=c2, dt_s=dt, pair_class=pc)],
                                 pair_classes=[pc])
                else:
                    skipped.append(f"{lv}#{i}:未知 handoff_type {r['handoff_type']}")
                    continue

            key = (lv, seq, tuple(sorted(cams)), gid, int(r["start_frame"]), int(r["end_frame"]))
            sample_id, verdict = review.get(key, (None, None))
            if sample_id:
                matched.add(sample_id)
            w = make_window("L3" if lv == "L3" else lv, rule, split, seq, cams, gid, lo, hi,
                            st, occ, **extra, **common)
            w["manual_sample_id"] = sample_id
            w["manual_verdict"] = verdict
            w["rendered"] = bool(verdict)
            windows.append(w)

    bad = selfcheck(windows, splits, dict(stride=st))
    ids = Counter(w["window_id"] for w in windows)
    bad += [f"{k}:window_id 重複 {n} 次" for k, n in ids.items() if n > 1]
    unmatched = sorted({v[0] for v in review.values()} - matched)
    bad += [f"{s}:review_sheet 的片段在 CSV 找不到" for s in unmatched] + review_bad

    counts = Counter(f"{w['selection_rule']}|{w['split']}" for w in windows)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    out = dict(definition="B(CHIRLA 資料夾 CSV)", params=dict(stride=st), splits=splits,
               topology=str(args.topology), csv_dir=str(args.csv_dir),
               review_sheet=args.review_sheet, git_sha=sha, argv=sys.argv[1:],
               counts=dict(counts), n_rendered=sum(w["rendered"] for w in windows),
               skipped=skipped, selfcheck_failures=bad, windows=windows)

    print("每個規則的產量:")
    for rule in ("L1", "L2", "L3S", "L3T"):
        print(f"  {rule:<4} " + "  ".join(f"{s}={counts.get(f'{rule}|{s}', 0)}" for s in splits))
    print(f"  非獨佔的窗:{sum(not w['exclusive'] for w in windows)}")
    print(f"  manual_check 對上 {len(matched)} 段;判為符合 {out['n_rendered']} 段")
    if skipped:
        print(f"  略過 {len(skipped)} 列:", *skipped[:10], sep="\n    ")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {path}")
    if bad:
        print(f"[自檢] {len(bad)} 項:")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("[OK] 自檢全部通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
