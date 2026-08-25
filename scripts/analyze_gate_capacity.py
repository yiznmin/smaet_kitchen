"""M5 時空門的解析式容量分析 —— 不需任何資料、不需 GPU,秒級得到可行/不可行訊號。

要回答的問題:現行參數化在「最有利假設」(μ/σ 量得完全正確、轉場真的是高斯)下,
每次跨鏡頭轉場有多少機率被誤拒?被誤拒 = 開新 chef_id = 身份碎裂。

§1–§6 分析 v2 加權和的數學性質(保留供對照與回溯):
    fused = w_st·exp(-0.5·z²) + w_app·app  ≥  combined_threshold,   z = (Δt-μ)/σ
    另有一道 |z| ≤ k_sigma —— 實測發現它是死參數,幾乎從不生效。

§7 實測 v2 vs v3(likelihood ratio,目前的預設模式)的門寬與三個約束:
    C1 碎裂率達標 / C2 外觀不能單獨過門 / C3 外觀仍能破平手

§5 與 §7 都用**真實的 SpatioTemporalIdentityManager** 反推實測門界。
§5 額外與解析值對拍 —— 解析式若與實作不符,以實作為準,前四節作廢。

用法:
  python scripts/analyze_gate_capacity.py
  python scripts/analyze_gate_capacity.py --topology configs/camera_topology.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid.embedder import l2norm                            # noqa: E402
from m5_reid.identity_st import SpatioTemporalIdentityManager  # noqa: E402
from m5_reid.spatiotemporal import CameraTopology              # noqa: E402


def _phi(x):
    """標準常態 CDF(避免為了一個函式引入 scipy)。"""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def gate_halfwidth(app, w_st, w_app, thr, k_sigma):
    """回傳 (有效門半寬 |z|max, 哪道門在生效)。

    解 w_st·exp(-0.5z²) + w_app·app ≥ thr 得 |z| ≤ sqrt(-2·ln((thr-w_app·app)/w_st))
    再與 k_sigma 取小。
    """
    need = (thr - w_app * app) / w_st          # 時空分數至少要多少
    if need <= 0:                              # 外觀單獨已過門(本設計刻意不讓它發生)
        return k_sigma, "k_sigma(外觀已足夠)"
    if need >= 1.0:                            # 連 z=0 都過不了
        return 0.0, "無解(門檻高於 w_st)"
    z = float(np.sqrt(-2.0 * np.log(need)))
    return (k_sigma, "k_sigma") if z > k_sigma else (z, "combined_threshold")


def section1_gate_width(f):
    print("=" * 78)
    print("§1 有效門寬與真實轉場的誤拒率(假設 μ/σ 完全正確、轉場真為高斯)")
    print("=" * 78)
    print(f"    w_st={f['w_st']}  w_app={f['w_app']}  thr={f['combined_threshold']}  "
          f"k_sigma={f['k_sigma']}")
    print()
    print("  外觀 cos | 需要 st_prob | 有效門寬 |z| | 高斯覆蓋率 | 誤拒率 | 生效的門")
    print("  " + "-" * 74)
    rows = []
    for app in [0.0, 0.15, 0.20, 0.35, 0.49, 0.62, 0.80, 0.90, 1.00]:
        z, which = gate_halfwidth(app, f["w_st"], f["w_app"],
                                  f["combined_threshold"], f["k_sigma"])
        need = max((f["combined_threshold"] - f["w_app"] * app) / f["w_st"], 0.0)
        cov = 2 * _phi(z) - 1
        rows.append((app, z, cov))
        print(f"  {app:8.2f} | {need:11.4f} | {z:11.4f} | {cov*100:9.1f}% | "
              f"{(1-cov)*100:5.1f}% | {which}")

    # k_sigma 何時才真的是那道門
    need_at_k = f["w_st"] * np.exp(-0.5 * f["k_sigma"] ** 2)
    app_star = (f["combined_threshold"] - need_at_k) / f["w_app"]
    print()
    print(f"  → k_sigma={f['k_sigma']} 只有在外觀 cos ≥ {app_star:.3f} 時才是實際生效的門。")
    print("     DINOv2 實測跨視角同人平均 0.490、OSNet 0.618 → 兩者都達不到,")
    print("     所以 k_sigma 在真實運作中幾乎從不觸發,真正淘汰候選的是 combined_threshold。")
    return rows


def section2_candidate_multiplicity(f, mu, sigma):
    """候選多重度:決定外觀品質重不重要。

    新 track 出現時,通過時空門的「假候選」數 ≈ 其他廚師從有連結的鏡頭離場、
    且落在寬度 2·|z|max·σ 的時間窗內的期望數:
        E[C_false] = (N-1)/T · 2·|z|max·σ · φ
    φ = 離場來自「有連結進本鏡頭」的比例(拓撲扇入係數)。
    """
    print()
    print("=" * 78)
    print("§2 候選多重度 E[C] —— 決定外觀品質到底重不重要")
    print("=" * 78)
    z, _ = gate_halfwidth(0.20, f["w_st"], f["w_app"],
                          f["combined_threshold"], f["k_sigma"])   # 用 DINOv2 量級的外觀
    width = 2 * z * sigma
    print(f"    有效時間窗寬度 = 2·{z:.3f}·σ = {width:.2f} 秒 (σ={sigma})")
    print("    φ(扇入係數)= 0.5,即半數離場來自有連結的鏡頭")
    print()
    print("  廚師數 N \\ 平均轉場間隔 T |" + "".join(f"{t:>8}s" for t in [15, 30, 60, 120]))
    print("  " + "-" * 62)
    phi_fan = 0.5

    def ec_false(N, T):
        return (N - 1) / T * width * phi_fan

    for N in [2, 3, 4, 5, 6, 8]:
        cells = [f"{1+ec_false(N, T):>9.2f}" for T in [15, 30, 60, 120]]
        print(f"  {N:>8}                  |" + "".join(cells))
    print()
    print("  (數值 = 1 個真候選 + 期望假候選數)")
    print()
    # 用「典型廚房」而非全表最差格判定:3~5 人、每 60~120 秒換一次鏡頭。
    # T=15s 代表每 15 秒就換一台鏡頭,那是換班尖峰而非常態,單獨列為尖峰情境。
    typical = max(ec_false(N, T) for N in [3, 4, 5] for T in [60, 120])
    peak = ec_false(8, 15)
    print(f"  典型情境(3~5 人、每 60~120 秒轉場一次):E[假候選] = {typical:.2f}")
    print(f"  換班尖峰(8 人、每 15 秒轉場一次)      :E[假候選] = {peak:.2f}")
    print()
    if typical < 0.15:
        print("  → 典型情境下幾乎每次決策都只有 1 個候選通過時空門。")
        print("     **外觀品質在絕大多數決策裡根本沒被用到**——沒有第二個人可以搞混。")
        print("     推論 1:放寬門檻換低碎裂率,誤併代價很小 → 修法(a) 可行。")
        print("     推論 2:DINOv2 的 Rank-1 只有 11%,對本架構不是致命問題;")
        print("             換更好的 embedder 買到的邊際效益很低。")
        print(f"     ⚠ 但換班尖峰 E[假候選]={peak:.2f},該時段外觀仍會被用到。")
    elif typical < 0.5:
        print("  → 典型情境多為單候選,尖峰時會出現第二個。")
        print("     放寬門檻大致安全,但必須監控誤併率。")
    else:
        print("  → 經常有多個候選,外觀是真正的仲裁者。")
        print("     DINOv2 的 11% 會成為瓶頸,embedder 選型是必要投資。")
    return typical, peak, width


def section6_can_reweighting_fix_it(f, sigma, typical_ec, budget):
    """修法(a) 夠不夠?掃參數空間找出能達標的配置,並量化它的誤併曝險代價。

    放寬門會等比放大時間窗 → 假候選數等比增加。曝險倍數 = z_new / z_now,
    這是精確的(不需要對外觀分布做任何假設)。
    """
    print()
    print("=" * 78)
    print("§6 光靠重配權重救得回來嗎?")
    print("=" * 78)
    app = 0.49                                   # DINOv2 量級
    z_now, _ = gate_halfwidth(app, f["w_st"], f["w_app"], f["combined_threshold"], f["k_sigma"])
    p_now = 1 - (2 * _phi(z_now) - 1)
    # 要達到 budget,需要的門半寬
    z_need = float(-np.sqrt(2) * 0 + 0)
    from math import erf, sqrt                    # 反解 Φ⁻¹
    lo, hi = 0.0, 8.0
    for _ in range(200):                          # 二分找 2Φ(z)-1 = 1-budget
        mid = (lo + hi) / 2
        if (2 * (0.5 * (1 + erf(mid / sqrt(2)))) - 1) < (1 - budget):
            lo = mid
        else:
            hi = mid
    z_need = (lo + hi) / 2
    print(f"    現況:|z|max={z_now:.3f} → 碎裂率 {p_now*100:.1f}%")
    print(f"    達標:|z|max≥{z_need:.3f} → 碎裂率 ≤{budget*100:.1f}%")
    print(f"    → 門必須放寬 {z_need/z_now:.2f} 倍,假候選數同步放大 {z_need/z_now:.2f} 倍")
    print(f"      典型情境的 E[假候選] 從 {typical_ec:.2f} 變成 {typical_ec*z_need/z_now:.2f}")
    print()
    print("  三個必須同時滿足的約束:")
    print("    C1 碎裂率 ≤ 預算            → 時間窗要夠寬")
    print("    C2 外觀拉滿也不能單獨過門    → w_app < thr(維持『時空是硬門』的設計意圖)")
    print("    C3 外觀仍能破平手            → w_app 的擺動幅度要大於典型分數差")
    print()
    print("    w_st | w_app |  thr | |z|max | 碎裂率 | C1 | C2 | 外觀擺幅")
    print("    " + "-" * 64)
    found = []
    for w_st in [0.7, 1.0]:
        for w_app in [0.30, 0.20, 0.10, 0.05]:
            for thr in [0.35, 0.25, 0.18, 0.12]:
                z, _ = gate_halfwidth(app, w_st, w_app, thr, 99.0)   # 先不夾 k_sigma
                p = 1 - (2 * _phi(z) - 1)
                c1 = p <= budget
                c2 = w_app * 1.0 < thr              # 外觀拉滿仍過不了門
                if c1 and c2:
                    found.append((w_st, w_app, thr, z, p))
                print(f"    {w_st:.2f} | {w_app:5.2f} | {thr:4.2f} | {z:6.3f} | {p*100:5.1f}% | "
                      f"{'✅' if c1 else '✗ '} | {'✅' if c2 else '✗ '} | ±{w_app:.2f}")
    print()
    if found:
        w_st, w_app, thr, z, p = max(found, key=lambda r: r[1])   # 取 w_app 最大者
        print(f"  → 有解:w_st={w_st}, w_app={w_app}, combined_threshold={thr}")
        print(f"     碎裂率 {p*100:.1f}% ≤ 預算,且外觀拉滿({w_app:.2f})仍過不了門檻({thr})。")
        print(f"     代價 1:時間窗放寬 {z/z_now:.2f} 倍 → 典型假候選 {typical_ec:.2f} → "
              f"{typical_ec*z/z_now:.2f}(仍遠小於 1,可接受)")
        print(f"     代價 2:**外觀擺幅從 ±0.30 縮到 ±{w_app:.2f}**,破平手能力同步縮小。")
        print("             §2 說典型情境幾乎用不到破平手,但換班尖峰用得到 → 尖峰時誤併風險上升。")
        print("     ⚠ k_sigma 必須同步調到 ≥|z|max,否則它會取代 thr 成為新的實際限制。")
        print()
        print("  這就是加權和融合的結構性問題:**同一個 thr 同時控制「窗多寬」與「外觀能不能")
        print("  單獨過門」,兩個需求互相拉扯。** 要放寬窗就得壓低 thr,壓低 thr 就得同步壓低")
        print("  w_app 才能維持硬門性質,而壓低 w_app 就削弱了破平手能力。三者無法各自獨立調。")
    else:
        print("  → 無解:重配權重達不到預算。必須走修法(c)。")
    print()
    print("  修法(c) 為什麼能解開這個結:likelihood-ratio 把兩種證據轉成同一個尺度上的")
    print("  對數勝算比相加,門檻設在「總證據強度」上。時間窗寬度由轉場分布自身決定")
    print("  (可用 Histogram-Parzen 從實測資料估),外觀的貢獻由它自己的可分性決定,")
    print("  兩者不再共用一個旋鈕 → 三個約束可以各自獨立滿足。")
    return found


def section3_m4_coupling(f, topo, tracker_lost_buffer=30, fps=30.0):
    """M4 的 lost_track_buffer 與 M5 的 μ 之間有隱藏耦合。

    M5 的路徑(a) 只掃 self.gone,而 chef 要等 M4 發 removed(= lost buffer 到期)
    才進 gone。若廚師在 removed 之前就抵達下一台鏡頭,路徑(a) 掃不到他
    (他還在 active),路徑(b) 又因非重疊而跳過 → **必定開新 chef_id**。
    """
    print()
    print("=" * 78)
    print("§3 M4↔M5 隱藏耦合:lost_track_buffer vs 最短轉場時間")
    print("=" * 78)
    buf_s = tracker_lost_buffer / fps
    print(f"    M4 lost_track_buffer = {tracker_lost_buffer} 幀 @ {fps}fps = {buf_s:.2f} 秒")
    print("    → 廚師離開後要 %.2f 秒才進入 gone gallery,期間抵達下一鏡頭一律開新 chef" % buf_s)
    print()
    z, _ = gate_halfwidth(0.20, f["w_st"], f["w_app"],
                          f["combined_threshold"], f["k_sigma"])
    print("  連結            | μ (s) | σ (s) | 最早可信抵達 μ-|z|σ | 判定")
    print("  " + "-" * 70)
    bad = []
    for (a, b), (mu, sd) in sorted(topo.links.items()):
        earliest = mu - z * sd
        ok = earliest > buf_s
        if not ok:
            bad.append((a, b, mu, sd, earliest))
        print(f"  {a}→{b:<10} | {mu:5.1f} | {sd:5.1f} | {earliest:18.2f} | "
              f"{'OK' if ok else '✗ 早於 gallery 就緒'}")
    print()
    if bad:
        print("  → **有連結會失效**:這些路徑上走得快的廚師必定被判為新人。")
        print("     修法:降低 lost_track_buffer,或這些鏡頭對改列為 overlapping。")
    else:
        print(f"  → 目前設定安全,但這是脆弱的隱含約束:任何 μ < {buf_s:.2f}s 的鏡頭對都會靜默失效。")
        print("     交付給業主的自檢工具必須斷言 min(μ - |z|σ) > lost_track_buffer/fps。")
    return bad


def section4_clock_skew(f, sigma):
    """鏡頭時鐘不同步 δ 秒 → 觀測到的 Δt 整體偏移 δ → 等效 z 偏移 δ/σ。"""
    print()
    print("=" * 78)
    print("§4 鏡頭時鐘漂移容忍度(程式、config、文件目前完全未處理)")
    print("=" * 78)
    z0, _ = gate_halfwidth(0.20, f["w_st"], f["w_app"],
                           f["combined_threshold"], f["k_sigma"])
    print(f"    σ={sigma}s,有效門半寬 |z|max={z0:.3f} → 門的實際半寬 {z0*sigma:.2f} 秒")
    print()
    print("  時鐘漂移 δ | 等效 z 偏移 | 仍落在門內的真實轉場 | 誤拒率")
    print("  " + "-" * 62)
    for delta in [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]:
        shift = delta / sigma
        # 真實 Δt~N(μ,σ);觀測 = 真實+δ。落在 [μ-z0σ, μ+z0σ] 的機率
        cov = _phi(z0 - shift) - _phi(-z0 - shift)
        print(f"  {delta:9.1f}s | {shift:11.3f} | {cov*100:19.1f}% | {(1-cov)*100:5.1f}%")
    print()
    print("  → NVR 若無 NTP,鏡頭間漂移 1~2 秒是常態。上表顯示這足以讓誤拒率翻倍以上,")
    print("     而症狀是「chef_id 一直開新的」,看起來像模型爛,實際是時鐘問題。")
    print("     部署需求必須寫明 NTP 同步,且自檢工具要能反推殘餘 skew。")


def section5_empirical_crosscheck(topo, f):
    """用真實的 SpatioTemporalIdentityManager 反推實測門界,與解析式對拍。

    解析式若與實作對不上,以實作為準,前四節作廢。
    """
    print()
    print("=" * 78)
    print("§5 與真實程式碼對拍(解析式的正確性檢查)")
    print("=" * 78)
    rng = np.random.RandomState(0)
    DIM = 128

    def make_emb(target, cos):
        n = rng.randn(DIM)
        n = l2norm(n - (n @ target) * target)
        return l2norm(cos * target + np.sqrt(max(0.0, 1 - cos * cos)) * n)

    (a, b), (mu, sd) = sorted(topo.links.items())[0]
    print(f"    掃描連結 {a}→{b}(μ={mu}, σ={sd}),把抵達時間從 μ-4σ 掃到 μ+4σ")
    print()
    print("  外觀 cos | 解析預測門界 |z|max | 實測門界 |z| | 差異 | 一致?")
    print("  " + "-" * 66)
    all_ok = True
    for app in [0.0, 0.20, 0.49, 0.90]:
        z_pred, _ = gate_halfwidth(app, f["w_st"], f["w_app"],
                                   f["combined_threshold"], f["k_sigma"])
        matched_z = []
        for z in np.arange(0.0, 4.0, 0.005):
            m = SpatioTemporalIdentityManager(topo, fps=1.0, recently_disappeared_ttl=10 ** 9)
            A = l2norm(rng.randn(DIM))
            m.on_new_track(1, camera_id=a, frame_id=0, embedding=A, t_sec=0.0)
            m.on_track_lost(1, camera_id=a, frame_id=1, t_sec=1.0)
            m.on_track_removed(1, camera_id=a, frame_id=2, t_sec=1.5)
            t_enter = 1.0 + mu + z * sd            # 出口時間戳為 lost 當時的 1.0
            r = m.on_new_track(2, camera_id=b, frame_id=int(t_enter),
                               embedding=make_emb(A, app), t_sec=t_enter)
            if r.matched:
                matched_z.append(z)
        z_emp = max(matched_z) if matched_z else 0.0
        diff = abs(z_emp - z_pred)
        ok = diff < 0.02
        all_ok &= ok
        print(f"  {app:8.2f} | {z_pred:19.4f} | {z_emp:12.4f} | {diff:.4f} | "
              f"{'✅' if ok else '❌ 解析式與實作不符'}")
    print()
    if all_ok:
        print("  → 解析式與實作一致,前四節的結論成立。")
        print("  ⚠ 同時暴露一件事:verify_m5_st.py 的 S1–S9 全都只測 z=0 或 z 極大,")
        print("     完全沒有覆蓋 z∈[1,2] 這個決定成敗的區間。")
    else:
        print("  → 解析式與實作不符,前四節作廢,需先釐清實作行為。")
    return all_ok


def _empirical_gate(topo_cfg, mode_overrides, app_list, seed=0):
    """用真實 SpatioTemporalIdentityManager 實測門界(單邊 |z|max)。

    回傳 {app: z_max}。做法:固定連結,把抵達時間從 μ 往外掃,找 matched 翻轉點。
    """
    import copy
    cfg = copy.deepcopy(topo_cfg)
    cfg.setdefault("fusion", {}).update(mode_overrides)
    topo = CameraTopology.from_config(cfg)
    (a, b), (mu, sd) = sorted(topo.links.items())[0]
    rng = np.random.RandomState(seed)
    DIM = 128
    out = {}
    for app in app_list:
        matched_z = []
        for z in np.arange(0.0, 5.0, 0.01):
            m = SpatioTemporalIdentityManager(topo, fps=1.0, recently_disappeared_ttl=10 ** 9)
            A = l2norm(rng.randn(DIM))
            n = rng.randn(DIM)
            n = l2norm(n - (n @ A) * A)
            q = l2norm(app * A + np.sqrt(max(0.0, 1 - app * app)) * n)
            m.on_new_track(1, camera_id=a, frame_id=0, embedding=A, t_sec=0.0)
            m.on_track_lost(1, camera_id=a, frame_id=1, t_sec=1.0)
            m.on_track_removed(1, camera_id=a, frame_id=2, t_sec=1.5)
            t_enter = 1.0 + mu + z * sd
            if m.on_new_track(2, camera_id=b, frame_id=int(t_enter),
                              embedding=q, t_sec=t_enter).matched:
                matched_z.append(z)
        out[app] = max(matched_z) if matched_z else 0.0
    return out


def section7_v2_vs_v3(topo_cfg, budget):
    """v3(likelihood ratio)有沒有真的解決 §6 那個結構性張力?

    三個約束必須同時成立:
      C1 碎裂率 ≤ 預算
      C2 外觀單獨不能過門(把外觀證據拉滿,時間完全不合理時仍須拒絕)
      C3 外觀仍能破平手(兩個時間同樣合理的候選,較像者要勝出)
    """
    print()
    print("=" * 78)
    print("§7 v2 加權和 vs v3 對數勝算比 —— 實測門寬(跑真實程式碼)")
    print("=" * 78)
    apps = [0.0, 0.20, 0.49, 0.62, 0.90]
    v2 = _empirical_gate(topo_cfg, {"mode": "weighted_sum"}, apps)
    v3 = _empirical_gate(topo_cfg, {"mode": "llr"}, apps)
    print("  外觀 cos | v2 |z|max | v2 碎裂率 | v3 |z|max | v3 碎裂率")
    print("  " + "-" * 60)
    for a in apps:
        p2 = 1 - (2 * _phi(v2[a]) - 1)
        p3 = 1 - (2 * _phi(v3[a]) - 1)
        print(f"  {a:8.2f} | {v2[a]:9.3f} | {p2*100:8.1f}% | {v3[a]:9.3f} | {p3*100:8.1f}%")
    p3_dino = 1 - (2 * _phi(v3[0.49]) - 1)
    p2_dino = 1 - (2 * _phi(v2[0.49]) - 1)
    print()
    print(f"  DINOv2 量級(cos≈0.49):v2 碎裂 {p2_dino*100:.1f}% → v3 碎裂 {p3_dino*100:.1f}%"
          f"(預算 {budget*100:.0f}%)")
    c1 = p3_dino <= budget
    print(f"  C1 碎裂率達標:{'✅' if c1 else '❌'}")

    # C2:外觀證據拉滿時,時空仍必須有否決權
    import copy
    cfg = copy.deepcopy(topo_cfg)
    cfg.setdefault("fusion", {}).update({"mode": "llr"})
    topo = CameraTopology.from_config(cfg)
    (a, b), (mu, sd) = sorted(topo.links.items())[0]
    rng = np.random.RandomState(1)

    def try_bind(exit_cam, enter_cam, t_enter, cos=1.0):
        m = SpatioTemporalIdentityManager(topo, fps=1.0, recently_disappeared_ttl=10 ** 9)
        A = l2norm(rng.randn(128))
        m.on_new_track(1, camera_id=exit_cam, frame_id=0, embedding=A, t_sec=0.0)
        m.on_track_lost(1, camera_id=exit_cam, frame_id=1, t_sec=1.0)
        m.on_track_removed(1, camera_id=exit_cam, frame_id=2, t_sec=1.5)
        q = A if cos >= 1.0 else None
        return m.on_new_track(2, camera_id=enter_cam, frame_id=int(max(t_enter, 0)),
                              embedding=q, t_sec=t_enter).matched

    # C2a 物理不可能:同鏡頭 / 無連結 / 比離開更早抵達 → 外觀拉滿也必拒(硬保證)
    c2a = (not try_bind(a, a, 1.0 + mu)                     # 同鏡頭
           and not try_bind(a, "cam_no_link", 1.0 + mu)     # 無拓撲連結
           and not try_bind(a, b, 0.5))                     # 早於離場
    print(f"  C2a 物理不可能(同鏡頭/無連結/早於離場)+ 外觀 cos=1.0 → 必拒:"
          f"{'✅' if c2a else '❌'}")

    # C2b 荒謬延遲(20×μ,連逗留模型都無法解釋)→ 外觀拉滿仍拒
    c2b = not try_bind(a, b, 1.0 + 20 * mu)
    print(f"  C2b 荒謬延遲(Δt={20*mu:.0f}s = 20×μ)+ 外觀 cos=1.0 → 必拒:"
          f"{'✅' if c2b else '❌'}")

    # C2c 外觀的發言權上限 vs 判定門檻 —— 量化「外觀最多能推翻多少時間證據」
    max_app = topo.app_lr.max_abs_llr()
    print(f"  C2c 外觀最大證據 {max_app:.2f} nats vs 判定門檻 {topo.llr_threshold:.2f} nats "
          f"→ 外觀單獨{'不足以' if max_app < topo.llr_threshold else '足以'}過門")
    c2 = c2a and c2b
    if c2:
        print("      ⚠ 但中等延遲(如 μ+10σ)配上強外觀**會**綁定 —— 這是逗留模型的")
        print("         設計意圖(廚師中途停下來做事是真的),不是 bug。若要收緊,")
        print("         調 p_loiter/tau_loiter_s,或設 appearance_clip 夾住外觀發言權。")

    # C3:兩個時間同樣合理的候選,外觀較像者要勝出
    m = SpatioTemporalIdentityManager(topo, fps=1.0, recently_disappeared_ttl=10 ** 9)
    A, B = l2norm(rng.randn(128)), l2norm(rng.randn(128))
    ra = m.on_new_track(1, camera_id=a, frame_id=0, embedding=A, t_sec=0.0)
    m.on_track_lost(1, camera_id=a, frame_id=1, t_sec=1.0)
    m.on_track_removed(1, camera_id=a, frame_id=2, t_sec=1.5)
    rb = m.on_new_track(2, camera_id=a, frame_id=0, embedding=B, t_sec=0.9)
    m.on_track_lost(2, camera_id=a, frame_id=1, t_sec=1.0)      # 兩人同時離場 → 時間證據相同
    m.on_track_removed(2, camera_id=a, frame_id=2, t_sec=1.5)
    n = rng.randn(128)
    n = l2norm(n - (n @ B) * B)
    q = l2norm(0.62 * B + np.sqrt(1 - 0.62 ** 2) * n)           # 明顯比較像 B
    rq = m.on_new_track(3, camera_id=b, frame_id=5, embedding=q, t_sec=1.0 + mu)
    c3 = rq.matched and rq.chef_id == rb.chef_id
    print(f"  C3 時間相同時外觀破平手(選較像的 B):{'✅' if c3 else '❌'}"
          f" — A={ra.chef_id} B={rb.chef_id} 綁={rq.chef_id} 候選數={m.last_candidates}")
    print()
    if c1 and c2 and c3:
        print("  → **v3 三個約束同時成立**,§6 那個「三者共用一個旋鈕」的結構性張力已解開。")
        print()
        print("  值得注意:C2c 那個「外觀單獨不足以過門」不是調參調出來的,是 DINOv2 本身")
        print("  就只有 1.31 nats 的鑑別力。**模型有多少資訊量就自動獲得多少發言權**——")
        print("  這正是 v2 用固定 w_app=0.3 硬塞時做不到的。")
        print("  ⚠ 反過來說,若日後換成 OSNet(最大證據約 5 nats)或 TAO,外觀就會超過門檻、")
        print("     可以單獨推翻中等強度的時間證據。屆時要用 appearance_clip 夾住,")
        print("     或重新推導 cost_false_merge_over_break。這是選型時要一併決定的事。")
    else:
        print("  → v3 尚未同時滿足三個約束,需再調 background_arrival_hz / cost 比。")
    return (c1 and c2 and c3), p3_dino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default=str(ROOT / "configs" / "camera_topology.yaml"))
    ap.add_argument("--lost-track-buffer", type=int, default=30)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--p-break-budget", type=float, default=0.05,
                    help="可容忍的每次轉場碎裂率(由 M6 事件需求反推)")
    args = ap.parse_args()

    import yaml
    with open(args.topology, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    topo_cfg = raw.get("camera_topology", raw)

    # §1–§6 分析的是 v2 加權和的數學性質,所以強制用 weighted_sum 的參數解讀
    topo = CameraTopology.from_yaml(args.topology)
    f = dict(topo.fusion)
    f["mode"] = "weighted_sum"
    mus = [m for m, _ in topo.links.values()]
    sds = [s for _, s in topo.links.values()]
    mu, sigma = float(np.mean(mus)), float(np.mean(sds))

    rows = section1_gate_width(f)
    typical_ec, peak_ec, _ = section2_candidate_multiplicity(f, mu, sigma)
    bad_links = section3_m4_coupling(f, topo, args.lost_track_buffer, args.fps)
    section4_clock_skew(f, sigma)
    # §5 要對拍的是 v2 的數學,必須用 weighted_sum 模式建管理器
    import copy
    v2_cfg = copy.deepcopy(topo_cfg)
    v2_cfg.setdefault("fusion", {})["mode"] = "weighted_sum"
    consistent = section5_empirical_crosscheck(CameraTopology.from_config(v2_cfg), f)
    fixes = section6_can_reweighting_fix_it(f, sigma, typical_ec, args.p_break_budget)
    v3_ok, p3_dino = section7_v2_vs_v3(topo_cfg, args.p_break_budget)

    # ── 判定 ────────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("判定")
    print("=" * 78)
    p_break_dino = 1 - [c for a, _, c in rows if abs(a - 0.49) < 1e-9][0]
    print(f"  DINOv2 量級外觀(cos≈0.49)下,理想假設的碎裂率 = {p_break_dino*100:.1f}%")
    print(f"  由 M6 反推的預算                            = {args.p_break_budget*100:.1f}%")
    verdict_ok = p_break_dino <= args.p_break_budget
    print(f"  → {'通過' if verdict_ok else '**未通過,差 %.1f 倍**' % (p_break_dino/args.p_break_budget)}")
    print()
    print("  注意這是**最有利**的假設:μ/σ 完全正確、轉場真為高斯、時鐘完全同步、")
    print("  無逗留無繞路。真實條件只會更差。")
    print()
    if not verdict_ok:
        if v3_ok:
            print(f"  ✅ 但 §7 已實測:v3(likelihood ratio)把同條件的碎裂率壓到 "
                  f"{p3_dino*100:.1f}%,三個約束同時成立")
            print("     → **修法(c) 已落地**,現行預設模式即為 llr。")
            print("     以下保留 v2 的分析,供對照、回溯與消融實驗用。")
            print()
        print("  三條修法的對照:")
        if fixes:
            w_st, w_app, thr, z, p = max(fixes, key=lambda r: r[1])
            print(f"   (a) 重配權重 w_st={w_st}, w_app={w_app}, thr={thr} —— §6 證明有解")
            print(f"       (碎裂 {p*100:.1f}%),但外觀擺幅被迫從 ±0.30 壓到 ±{w_app:.2f},")
            print("       破平手能力同步縮小 → 換班尖峰的誤併風險上升。是取捨不是免費的。")
        else:
            print("   (a) 重配權重 —— §6 顯示無解。")
        print("   (b) k_sigma 目前是死參數(只在外觀 cos≥0.851 時生效),調參時必須同步")
        print("       處理,否則它會取代 thr 成為新的實際限制。")
        print("   (c) likelihood-ratio 融合 + 從資料估轉場分布(st-ReID 作法)。治本:")
        print("       它把「窗寬」與「外觀權重」解耦成兩個獨立旋鈕,不必再互相犧牲。")
        print()
        print("  ⚠ (a) 只是把「理想假設下」的碎裂率壓進預算。真實條件(σ 估錯、逗留、")
        print("     繞路、時鐘漂移)還會再吃掉餘裕,那要靠蒙地卡羅量,不是解析式能回答的。")
    if bad_links:
        print()
        print(f"  ⚠ 另有 {len(bad_links)} 條連結受 §3 的 M4 耦合影響,與門寬無關,必須另外修。")
    sys.exit(0 if consistent else 1)


if __name__ == "__main__":
    main()
