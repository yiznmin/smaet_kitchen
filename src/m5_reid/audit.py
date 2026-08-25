"""M5 部署前自檢 —— 把「靜默失效」變成「開機就報錯」。

現況的問題:transition_gate / transit_llr 擋掉候選時是**靜默回 False**。業主只會
看到「chef_id 一直開新的」,查不出原因 —— 而原因可能是拓撲填錯、σ 不合理、
M4 的 lost_track_buffer 與轉場時間衝突,或根本是鏡頭時鐘沒同步。這些都是
config 層面就能事先抓到的,不該讓人在現場猜。

用法:
    from m5_reid.audit import audit_topology, format_findings
    print(format_findings(audit_topology(topo, tracker_cfg)))
或直接跑 scripts/audit_m5_config.py。
"""
import numpy as np

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"


class Finding:
    def __init__(self, level, code, message, fix=None):
        self.level, self.code, self.message, self.fix = level, code, message, fix

    def __repr__(self):
        return f"<{self.level} {self.code}>"


def effective_window(topo, cam_from, cam_to, app_llr=0.0, span=40.0, step=0.02):
    """該連結實際會接受的 Δt 區間(秒)。與融合模式無關,直接掃實際的評分函式。

    app_llr = 外觀能提供的證據量。傳 0 代表「外觀完全中性」,
    傳 topo.app_lr.max_abs_llr() 代表「外觀最有利」→ 兩者夾出窗的上下界。
    """
    mu, sd = topo.links[(cam_from, cam_to)]
    llr_mode = topo.fusion["mode"] == "llr"
    lo = hi = None
    for dt in np.arange(step, mu + span, step):
        if llr_mode:
            ok, llr = topo.transit_llr(cam_from, 0.0, cam_to, dt)
            passed = ok and (llr + app_llr) >= topo.llr_threshold
        else:
            f = topo.fusion
            ok, sp = topo.transition_gate(cam_from, 0.0, cam_to, dt)
            passed = ok and (f["w_st"] * sp + f["w_app"] * 1.0 * (app_llr > 0)) \
                >= f["combined_threshold"]
        if passed:
            lo = dt if lo is None else lo
            hi = dt
    return (lo, hi)


def audit_topology(topo, tracker_cfg=None, expected_headcount=None):
    """檢查拓撲 config 的內在一致性,以及與 M4 設定的跨模組約束。

    tracker_cfg = configs/tracker.yaml 的 tracker 區段(取 lost_track_buffer / frame_rate)。
    """
    out = []
    f = topo.fusion
    cams = topo.all_cameras()

    # ── A1 單向連結 ────────────────────────────────────────────────────
    for (a, b) in sorted(topo.links):
        if (b, a) not in topo.links:
            out.append(Finding(
                WARN, "A1_ONE_WAY_LINK",
                f"{a}→{b} 有連結但 {b}→{a} 沒有 —— 廚師走回頭路時會被判為新人。",
                f"若該路徑可雙向通行,補一條 {{from: {b}, to: {a}, ...}}(回程時間可不同)。"))

    # ── A2 孤立鏡頭 ────────────────────────────────────────────────────
    linked = {c for pair in topo.links for c in pair}
    overlapped = {c for pair in topo.overlapping for c in pair}
    for c in sorted(cams - linked - overlapped):
        out.append(Finding(
            ERROR, "A2_ISOLATED_CAMERA",
            f"{c} 沒有任何連結也不與其他鏡頭重疊 —— 進入該鏡頭的人一律開新 chef_id。",
            "補 links,或確認該鏡頭本來就不參與跨鏡頭身份追蹤。"))

    # ── A3 與 M4 的跨模組約束 ──────────────────────────────────────────
    if tracker_cfg:
        buf = float(tracker_cfg.get("lost_track_buffer", 30))
        fps = float(tracker_cfg.get("frame_rate", 30))
        gallery_delay = buf / fps
        for (a, b) in sorted(topo.links):
            lo, _ = effective_window(topo, a, b, app_llr=topo.app_lr.max_abs_llr())
            if lo is not None and lo < gallery_delay:
                out.append(Finding(
                    ERROR, "A3_M4_COUPLING",
                    f"{a}→{b} 最短可接受轉場 {lo:.2f}s < M4 進入 gone gallery 所需的 "
                    f"{gallery_delay:.2f}s(lost_track_buffer={buf:.0f} @ {fps:.0f}fps)"
                    " —— 走得快的廚師在候選名單建立前就抵達,必定開新 chef_id。",
                    f"降低 lost_track_buffer 到 < {lo*fps:.0f} 幀,"
                    f"或把 {a}/{b} 改列為 overlapping。"))

    # ── A4 σ 合理性 ────────────────────────────────────────────────────
    for (a, b), (mu, sd) in sorted(topo.links.items()):
        if sd <= 0:
            out.append(Finding(ERROR, "A4_BAD_SIGMA", f"{a}→{b} 的 std_s={sd} 不合法。",
                               "std_s 必須 > 0;沒有實測值時用 0.35 × mean_s。"))
            continue
        if sd > mu:
            out.append(Finding(
                WARN, "A4_SIGMA_GT_MEAN",
                f"{a}→{b} σ={sd:.1f}s > μ={mu:.1f}s —— 變異大於平均,時間幾乎不提供資訊。",
                "重新量測步行路徑距離,或改用 scripts/calibrate_topology.py 從影片估。"))
        lo, hi = effective_window(topo, a, b)
        if lo is not None and (hi - lo) < 1.0:
            out.append(Finding(
                WARN, "A4_WINDOW_TOO_NARROW",
                f"{a}→{b} 有效時間窗只有 {hi-lo:.2f} 秒({lo:.2f}~{hi:.2f}s)"
                " —— 稍微走慢一點就會被判為新人。",
                "放大 std_s,或降低 cost_false_merge_over_break。"))

    # ── A5/A6 時鐘同步 ─────────────────────────────────────────────────
    if not topo.clock_offset or all(v == 0 for v in topo.clock_offset.values()):
        out.append(Finding(
            INFO, "A5_NO_CLOCK_OFFSET",
            "所有相機的 clock_offset_s 都是 0(或未設定)。",
            "若 NVR 未跑 NTP,請先用 scripts/audit_m5_config.py --estimate-skew "
            "從重疊鏡頭的同時觀測反推殘餘偏移,填回 cameras.*.clock_offset_s。"))
    max_skew = float(topo.clock["max_skew_s"])
    sds = [sd for _, sd in topo.links.values()]
    if sds:
        min_sd = min(sds)
        ratio = max_skew / min_sd
        extra = _skew_breakage(topo, ratio)
        level = ERROR if ratio > 0.5 else (WARN if ratio > 0.25 else INFO)
        out.append(Finding(
            level, "A6_CLOCK_TOLERANCE",
            f"容許殘餘 skew {max_skew:.2f}s = 最小 σ({min_sd:.1f}s)的 {ratio:.2f} 倍"
            f" → 額外碎裂率約 +{extra*100:.1f} 個百分點。",
            "收緊 NTP(建議 skew ≤ 0.2s),或加大 std_s 讓時間窗容納漂移。"))

    # ── A7 融合模式 ────────────────────────────────────────────────────
    if f["mode"] == "weighted_sum":
        out.append(Finding(
            WARN, "A7_LEGACY_FUSION",
            "mode=weighted_sum 是 v2 加權和,實測在最有利假設下碎裂率 11.6%(預算 5%)。",
            "改用 mode: llr。weighted_sum 僅供消融對照,不建議出貨。"))

    # ── A8 外觀發言權 ──────────────────────────────────────────────────
    if f["mode"] == "llr":
        max_app = topo.app_lr.max_abs_llr()
        if max_app >= topo.llr_threshold and f["appearance_clip"] is None:
            out.append(Finding(
                WARN, "A8_APPEARANCE_DOMINATES",
                f"外觀最大證據 {max_app:.2f} nats ≥ 判定門檻 {topo.llr_threshold:.2f} nats"
                f" —— {f['appearance_profile']} 可以單獨推翻中等強度的時間證據。",
                f"設 appearance_clip: {topo.llr_threshold*0.6:.2f} 夾住外觀發言權,"
                "或重新推導 cost_false_merge_over_break。"))

    # ── A9 headcount 自檢提示 ──────────────────────────────────────────
    if expected_headcount:
        out.append(Finding(
            INFO, "A9_HEADCOUNT",
            f"排班人數 {expected_headcount} —— 執行時若同時 active 的 chef_id 數超過它,"
            "代表發生碎裂(同一人被拆成多個身份)。",
            "碎裂可偵測、誤併不可偵測,所以這個檢查是唯一能自動抓到的失效訊號。"))
    return out


def _skew_breakage(topo, ratio):
    """時鐘偏移 ratio(以 σ 為單位)造成的額外碎裂率(近似)。"""
    from math import erf, sqrt

    def phi(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    (a, b) = next(iter(topo.links))
    lo, hi = effective_window(topo, a, b, app_llr=0.0)
    if lo is None:
        return 1.0
    mu, sd = topo.links[(a, b)]
    z_lo, z_hi = (lo - mu) / sd, (hi - mu) / sd
    base = phi(z_hi) - phi(z_lo)
    shifted = phi(z_hi - ratio) - phi(z_lo - ratio)
    return max(0.0, base - shifted)


def estimate_clock_skew(coobservations, min_samples=5):
    """從重疊鏡頭的同時觀測反推殘餘時鐘偏移。

    coobservations = [(cam_a, t_a, cam_b, t_b), ...] —— 同一個人在兩台**視野重疊**
    的鏡頭上被同時看到的時間戳配對。理論上 t_a == t_b,實測差值的中位數就是
    兩台的相對偏移。用中位數而非平均,以免被偶發的錯誤配對拉走。

    回傳 {(cam_a, cam_b): (偏移中位數, 樣本數, 四分位距)}。
    """
    from collections import defaultdict
    diffs = defaultdict(list)
    for ca, ta, cb, tb in coobservations:
        key = (ca, cb) if ca <= cb else (cb, ca)
        d = (ta - tb) if ca <= cb else (tb - ta)
        diffs[key].append(float(d))
    out = {}
    for key, vals in diffs.items():
        if len(vals) < min_samples:
            continue
        arr = np.asarray(vals)
        out[key] = (float(np.median(arr)), len(arr),
                    float(np.percentile(arr, 75) - np.percentile(arr, 25)))
    return out


def format_findings(findings):
    if not findings:
        return "自檢通過,沒有發現問題。"
    order = {ERROR: 0, WARN: 1, INFO: 2}
    icon = {ERROR: "✗", WARN: "⚠", INFO: "·"}
    lines = []
    for fd in sorted(findings, key=lambda x: order[x.level]):
        lines.append(f"{icon[fd.level]} [{fd.level}] {fd.code}")
        lines.append(f"    {fd.message}")
        if fd.fix:
            lines.append(f"    → 修法:{fd.fix}")
    n_err = sum(1 for f in findings if f.level == ERROR)
    n_warn = sum(1 for f in findings if f.level == WARN)
    lines.append("")
    lines.append(f"合計:{n_err} 個 ERROR、{n_warn} 個 WARN、"
                 f"{len(findings)-n_err-n_warn} 個 INFO")
    return "\n".join(lines)
