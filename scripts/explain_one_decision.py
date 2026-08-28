"""把「一次身分綁定決策」攤開來看 —— 系統到底在算什麼、憑什麼決定。

七輪實驗的結論都建立在「重複做這個決策幾千次、數錯幾次」上,
但那些百分比看不出裡面發生什麼事。這支腳本只做一次決策,印出全部細節。

三件事對應報告裡的三句話:
  §1 會不會出錯      → 我們知道正確答案(世界是我們造的),比對系統的答案
  §2 什麼條件下出錯  → 只改一個條件(時鐘、校正、旁邊有沒有人),看答案會不會翻
  §3 加哪條證據有用  → 把某條證據關掉,看同一個決策的分數少多少

用法:python scripts/explain_one_decision.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from m5_reid.evidence import AppearanceLR, GroundPlaneLR, VelocityLR   # noqa: E402
from m5_reid.spatiotemporal import CameraTopology                      # noqa: E402

LINKS = [{"from": "cam1", "to": "cam2", "mean_s": 4.0, "std_s": 1.5},
         {"from": "cam2", "to": "cam1", "mean_s": 4.0, "std_s": 1.5}]


def topo(**fusion):
    return CameraTopology(links=LINKS, overlapping=[],
                          fusion={"mode": "llr",
                                  "background_arrival_hz": 1 / 600.0,
                                  "cost_false_merge_over_break": 5.0,
                                  "transit_model": "loiter",
                                  "p_loiter": 0.15, "tau_loiter_s": 20.0,
                                  **fusion})


def rule(title=""):
    print("\n" + "─" * 74)
    if title:
        print(title)
        print("─" * 74)


def main():
    T = topo()
    thr = T.llr_threshold

    print("=" * 74)
    print("一次身分綁定決策的完整追蹤")
    print("=" * 74)
    print(f"""
情境(這是我們造的世界,所以我們知道正確答案):

    t = 100.0s   張師傅 走出 cam1 的畫面
    t = 104.2s   cam2 出現一個人

  系統要回答:這個人是剛才那位張師傅,還是一個新的人?
  正確答案:  是張師傅(我們造世界時就是這樣安排的)

  判定門檻 = {thr:.2f} nats。證據總和超過它才綁定,否則開新身分。
  門檻不是拍腦袋訂的,是由兩個可量測的量算出來的:
    · 真新人出現的速率 1/600 秒
    · 誤併比碎裂嚴重 5 倍
""")

    rule("① 系統怎麼算 —— 每條證據值多少 nats")
    dt = 4.2
    ok, llr_t = T.transit_llr("cam1", 100.0, "cam2", 104.2)
    app_lr = AppearanceLR(**AppearanceLR.MEASURED["dinov2"])
    llr_a = app_lr.llr(0.49)                      # 外觀:同人的實測平均值
    print(f"""
  轉場時間   走 cam1→cam2 通常要 4.0±1.5 秒,他走了 {dt:.1f} 秒
             → 很合理,這條證據給  {llr_t:+.2f} nats

  外觀       餘弦相似度 0.49(DINOv2 對「同一人」的實測平均)
             → 但同人 0.490 / 不同人 0.465 幾乎分不開
             → 這條證據只給         {llr_a:+.2f} nats  ← 幾乎沒發言權

  合計 {llr_t + llr_a:+.2f} nats  vs  門檻 {thr:.2f}
  → {'綁定' if llr_t + llr_a >= thr else '開新身分'}(正確答案:綁定){'  ✅ 對了' if llr_t + llr_a >= thr else '  ❌ 錯了'}
""")
    print("  這就是「身分綁定邏輯」。整套系統做的就是這件事,一天幾千次。")

    rule("② 什麼條件下會出錯 —— 只改一個條件,看答案翻不翻")
    print(f"\n  {'條件':<34}{'時間證據':>10}{'總分':>9}{'決策':>10}{'對錯':>7}")
    print("  " + "-" * 70)
    cases = [
        ("正常(走了 4.2 秒)", 4.2, 0.0),
        ("中途停下來洗手(走了 25 秒)", 25.0, 0.0),
        ("拓撲距離填錯(實走 12 秒)", 12.0, 0.0),
        ("── 以下看時鐘偏移的影響 ──", None, None),
        ("走得快的人(2.5 秒),時鐘準", 2.5, 0.0),
        ("同上,但時鐘差 2 秒", 2.5, 2.0),
        ("同上,但時鐘差 4 秒", 2.5, 4.0),
        ("走得慢的人(7 秒),時鐘準", 7.0, 0.0),
        ("同上,但時鐘差 2 秒", 7.0, 2.0),
    ]
    for name, gap, skew in cases:
        if gap is None:
            print(f"  {name}")
            continue
        ok, lt = T.transit_llr("cam1", 100.0, "cam2", 100.0 + gap + skew)
        total = (lt if ok else -99) + llr_a
        bind = ok and total >= thr
        print(f"  {name:<34}{(lt if ok else float('nan')):>+10.2f}"
              f"{total:>9.2f}{'綁定' if bind else '開新身分':>10}"
              f"{'✅' if bind else '❌ 碎裂':>7}")
    print("""
  這張表就是「什麼條件下出錯」。

  ⚠ 注意時鐘那幾列:走 4.2 秒(正中轉場時間分布的峰值)的人,
    餘裕很大,差 4 秒也還撐得住。真正被時鐘偏移害死的是**本來就在邊緣**的人
    —— 走得比較快或比較慢的那些。所以單看一次決策會低估時鐘的殺傷力,
    要跑過**整個母體**才看得出來:實測 2 秒偏移讓誤拒率從 18% 跳到 49%。

  這正是為什麼要跑幾千次而不是看一個案例。而佈署規格
  「鏡頭時鐘 NTP 同步、殘餘偏移 ≤ 0.2 秒」就是從那個母體數字反推的。""")

    rule("③ 加哪條證據有用 —— 兩人同時在場時,時間證據不夠用")
    print("""
  上面都只有一個候選人。真正難的情況是**兩位廚師同時符合時間窗**:

    t = 100.0s   張師傅 走出 cam1
    t = 100.3s   李師傅 也走出 cam1
    t = 104.2s   cam2 出現一個人 —— 是誰?
""")
    ok_a, lt_a = T.transit_llr("cam1", 100.0, "cam2", 104.2)
    ok_b, lt_b = T.transit_llr("cam1", 100.3, "cam2", 104.2)
    print(f"  只用時間:張師傅 {lt_a:+.2f} nats  vs  李師傅 {lt_b:+.2f} nats"
          f"  → 差 {abs(lt_a - lt_b):.2f},幾乎分不出來")
    print("  兩人都過門檻 → 系統只能挑分數高的,**有一半機率挑錯**。\n")

    gp = GroundPlaneLR(sigma_m=0.1, area_m2=30.0, clip=8.0)
    vel = VelocityLR(sigma_pos_m=0.1, window_s=1.0, max_speed_mps=1.5, clip=6.0)
    print(f"  {'再加上……':<24}{'張師傅':>10}{'李師傅':>10}{'差距':>10}")
    print("  " + "-" * 56)
    # 張師傅就在該位置;李師傅在 2.5 公尺外
    g_a, g_b = gp.llr((5.0, 3.0), (5.02, 3.03)), gp.llr((5.0, 3.0), (7.5, 3.0))
    print(f"  {'地面校正(站在哪)':<24}{g_a:>+10.2f}{g_b:>+10.2f}{g_a - g_b:>10.2f}")
    # 兩人速度不同
    v_a, v_b = vel.llr((0.9, 0.1), (0.88, 0.14)), vel.llr((0.9, 0.1), (-0.6, 0.7))
    print(f"  {'軌跡(往哪走、多快)':<24}{v_a:>+10.2f}{v_b:>+10.2f}{v_a - v_b:>10.2f}")
    print(f"""
  加了這兩條之後差距拉開到 {(g_a + v_a) - (g_b + v_b):.1f} nats —— 這才分得出誰是誰。

  這就是「加哪條證據有用」。七輪實驗做的就是反覆問這個問題:
  把某條證據開/關,跑幾千次決策,看誤併率差多少。
  結論是加新的證據軸(位置、方向、全景鏡頭、地面校正、軌跡)都有效,
  而在既有證據上調權重、換分布、改門檻,一次都沒有用。""")

    rule("為什麼這件事不需要「人的影像」")
    print("""
  從頭到尾,系統拿到的只有:哪台鏡頭、幾點幾分、框在哪、世界座標、速度。
  **它沒有看過任何一張圖。** 影像在上游就被 M3 偵測與 M4 追蹤消化成這些數字了。

  所以模擬能測的:上面這套判斷邏輯,以及它在什麼條件下會壞。
  模擬**測不到**的:偵測準不準、追蹤會不會斷、真實遮擋、真實制服下的外觀
  —— 那些必須用真實影片,而多人的真實影片我們還沒有。
""")


if __name__ == "__main__":
    main()
