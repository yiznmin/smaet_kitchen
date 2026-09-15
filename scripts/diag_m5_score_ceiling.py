"""M5 各條候選路徑在數學上能拿到的最高分 —— 探索性診斷(2026-09-15)。

`diag_m5_binding_rules.py` 在 CHIRLA 上量到:所有成功的綁定都來自重疊路徑的常數分數,
非重疊路徑的第一名最高只有 −0.815,從未過門檻。這支回答「是資料剛好如此,還是結構上不可能」。

算的東西(全部直接呼叫 CameraTopology 的實作,不另外推公式):
  1. 門檻、外觀項在 cosine = 0 時的值(embedder=none 的情況)、重疊常數
  2. 各路徑轉場項在最有利 Δt 下的最高值:同鏡頭 / 每條有連結的鏡頭對 / 無連結鏡頭對
  3. 同鏡頭位置項的最高值(離場框與進場框完全相同)
  4. 同鏡頭路徑隨 Δt 的總分表,以及 Δt ≥ 跟丟緩衝時的最高總分

⚠ 為什麼要看 Δt ≥ 跟丟緩衝:`identity_st.on_track_removed` 的離場時間戳取自 lost 當下
  (identity_st.py 的 docstring 與 `_pending_exit`),而廚師要到名下 track 全部 removed 才進 `gone`,
  也就是跟丟緩衝到期之後。所以轉場路徑最早也要在 Δt ≥ 緩衝時才會評分到這位廚師。

用法:
    python scripts/diag_m5_score_ceiling.py --topology configs/fix_grid/base.yaml --lost-buffer-seconds 5
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--lost-buffer-seconds", type=float, default=5.0)
    args = ap.parse_args()

    from m5_reid.spatiotemporal import CameraTopology
    topo = CameraTopology.from_yaml(args.topology)
    f = topo.fusion
    thr, app0 = topo.llr_threshold, topo.app_lr.llr(0.0)
    const = float(f["overlap_llr"]) + app0

    print(f"門檻 llr_threshold                 {thr:.4f}")
    print(f"外觀項 app_lr.llr(cos = 0)          {app0:.4f}   (profile={f.get('appearance_profile')};"
          f"對照 llr(μ_diff 0.465) = {topo.app_lr.llr(0.465):.4f}、llr(μ_same 0.490) = {topo.app_lr.llr(0.490):.4f})")
    print(f"重疊常數分數 = {f['overlap_llr']} − ln k + ({app0:.4f}) = {const:.4f} − ln k"
          f"   → k ≤ {math.floor(math.exp(const - thr))} 必過門檻")

    dts = [x / 100 for x in range(1, 20001)]

    def best(a, b, lo=0.0):
        vals = [(topo.transit_llr(a, 0.0, b, dt), dt) for dt in dts if dt >= lo]
        ok = [(v[1], dt) for v, dt in vals if v[0]]
        return max(ok) if ok else (None, None)

    cams = sorted(topo.all_cameras())
    rows = []
    s_best, s_dt = best(cams[0], cams[0])
    rows.append(("同鏡頭", s_best, s_dt))
    for (a, b) in sorted(topo.links):
        m, dt = best(a, b)
        rows.append((f"有連結 {a}->{b}", m, dt))
    unl = [(a, b) for a in cams for b in cams
           if a != b and (a, b) not in topo.links and not topo.is_overlapping(a, b)]
    if unl:
        m, dt = best(*unl[0])
        rows.append((f"無連結 {unl[0][0]}->{unl[0][1]}", m, dt))

    box = (100.0, 100.0, 160.0, 280.0)
    pmax = max(topo.pos_lr.llr(box, box, d) for d in (0.01, 0.1, 0.5, 1, 2, 5)) if topo.pos_lr else 0.0

    print("\n各路徑轉場項最高值(掃 Δt 0.01~200 秒)與理論上限(+ 外觀項;同鏡頭另加位置項最高值)")
    print(f"   {'路徑':<34}{'轉場項':>9}{'@Δt':>8}{'上限':>9}{'過門檻':>7}{'外觀中性時':>11}")
    for name, m, dt in rows:
        if m is None:
            print(f"   {name:<34}{'永遠拒絕':>9}")
            continue
        extra = pmax if name == "同鏡頭" else 0.0
        top = m + app0 + extra
        print(f"   {name:<34}{m:>9.4f}{dt:>8}{top:>9.4f}{'是' if top >= thr else '否':>7}{top - app0:>11.4f}")

    print(f"\n同鏡頭路徑隨 Δt 的總分(轉場項 + 外觀項 + 位置項,位置完全相同的最有利情況)")
    print(f"   {'Δt(秒)':>8}{'轉場項':>9}{'位置項':>9}{'總分':>9}{'過門檻':>7}")
    worst_after = None
    for dt in (0.5, 1, 2, 3, 4, args.lost_buffer_seconds, 6, 8, 10, 12, 15):
        ok, t = topo.transit_llr(cams[0], 0.0, cams[0], dt)
        if not ok:
            print(f"   {dt:>8}{'拒絕':>9}")
            continue
        p = topo.pos_lr.llr(box, box, dt) if topo.pos_lr else 0.0
        tot = t + app0 + p
        if dt >= args.lost_buffer_seconds:
            worst_after = tot if worst_after is None else max(worst_after, tot)
        print(f"   {dt:>8}{t:>9.4f}{p:>9.4f}{tot:>9.4f}{'是' if tot >= thr else '否':>7}")
    if worst_after is not None:
        print(f"\n   Δt ≥ 跟丟緩衝 {args.lost_buffer_seconds} 秒時的最高總分:{worst_after:.4f} → "
              f"{'可能過門檻' if worst_after >= thr else '結構上不可能過門檻'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
