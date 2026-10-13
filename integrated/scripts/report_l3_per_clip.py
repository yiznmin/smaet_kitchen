"""L3 逐段報表:每個鏡頭對標明走哪條路徑、是否牽涉倒影鏡頭、各格是否一致。

只讀 eval 產出的 clip_metrics.json 與拓撲設定,不重跑任何東西。

用法:
    python scripts/report_l3_per_clip.py \
        --cells "原拓撲 none=results/single_person/eval/clip_app_dino_w0.7" \
                "修拓撲 none=results/l3_full/eval/fix17_none" \
                "修拓撲 dinov2=results/l3_full/eval/fix17_dino" \
        --topology configs/fix_grid/base_overlapfix.yaml
"""
import argparse
import glob
import itertools
import json

import yaml

MIRROR = {"camera_5", "camera_6"}


def load_topo(path):
    f = yaml.safe_load(open(path, encoding="utf-8"))["camera_topology"]
    ov = {tuple(sorted(p)) for p in f.get("overlapping", [])}
    lk = set()
    for l in f.get("links", []):
        lk.add(tuple(sorted((l["from"], l["to"]))))
    return ov, lk


def path_of(a, b, ov, lk):
    """這一對鏡頭在拓撲裡走哪條路徑。⚠ 重疊優先 —— M5 先查重疊再查轉場。"""
    k = tuple(sorted((a, b)))
    if k in ov and k in lk:
        return "重疊+連結"
    if k in ov:
        return "重疊"
    if k in lk:
        return "轉場連結"
    return "未建模"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help='"標題=eval 目錄"')
    ap.add_argument("--topology", default="configs/fix_grid/base_overlapfix.yaml")
    ap.add_argument("--base-topology", default="configs/fix_grid/base.yaml",
                    help="第一格用的原始拓撲(用來標示哪些對是新增的)")
    ap.add_argument("--manifest", default="results/clips/clip_manifest.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = [c.rsplit("=", 1) for c in args.cells]
    ov, lk = load_topo(args.topology)
    ov0, lk0 = load_topo(args.base_topology)
    man = json.loads(open(args.manifest, encoding="utf-8").read())
    info = {c["clip_id"]: c for c in man["clips"]}

    D = {}
    for lab, root in cells:
        for f in glob.glob(f"{root}/metrics/L3*/s5/clip_metrics.json"):
            m = json.loads(open(f, encoding="utf-8").read())
            D.setdefault(lab, {})[m["clip_id"]] = m
    clips = sorted(D[cells[0][0]])

    rows = []
    for rule in ("L3S", "L3T"):
        sel = [c for c in clips if info[c]["rule"] == rule]
        if not sel:
            continue
        print(f"\n{'='*96}\n{rule}({len(sel)} 段)\n{'='*96}")
        for c in sel:
            cams = sorted(info[c]["cameras"])
            mirror = set(cams) & MIRROR
            tag = f"  [含倒影 {','.join(sorted(mirror))}]" if mirror else ""
            print(f"\n  {c.replace(rule + '-', ''):<30}{'+'.join(x.replace('camera_', 'c') for x in cams)}"
                  f"  {info[c]['duration_s']}s{tag}")
            for a, b in itertools.combinations(cams, 2):
                p = path_of(a, b, ov, lk)
                p0 = path_of(a, b, ov0, lk0)
                new = "  <- 本次新增" if p != p0 else ""
                line = f"      {a.replace('camera_','c')}+{b.replace('camera_','c')}  {p:<10}"
                for lab, _ in cells:
                    m = D[lab].get(c)
                    per = (m or {}).get("l3_chef_per_camera") or {}
                    if per.get(a) is None or per.get(b) is None:
                        line += f"{'(一台未綁定)':>16}"
                    else:
                        line += f"{('一致' if per[a] == per[b] else '不一致'):>16}"
                print(line + new)
                rows.append(dict(rule=rule, clip=c, pair=f"{a}+{b}", path=p, new=bool(new),
                                 mirror=bool(mirror)))
    # 彙總
    print(f"\n{'='*96}\n彙總:依路徑分\n{'='*96}")
    print(f"{'路徑':<12}{'鏡頭對':>8}" + "".join(f"{lab:>18}" for lab, _ in cells))
    for p in ("重疊", "重疊+連結", "轉場連結", "未建模"):
        sub = [r for r in rows if r["path"] == p]
        if not sub:
            continue
        line = f"{p:<12}{len(sub):>8}"
        for lab, _ in cells:
            ok = n = 0
            for r in sub:
                m = D[lab].get(r["clip"])
                per = (m or {}).get("l3_chef_per_camera") or {}
                a, b = r["pair"].split("+")
                if per.get(a) is None or per.get(b) is None:
                    continue
                n += 1
                ok += per[a] == per[b]
            line += f"{f'{ok}/{n}' if n else '—':>18}"
        print(line)
    if args.out:
        json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n已存 {args.out}")


if __name__ == "__main__":
    main()
