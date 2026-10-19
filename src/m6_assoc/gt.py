"""EPFL 動作標註 → M6 人—物關聯的真值(每個 M3 類別一組「接觸中」區間)。

標註檔:<場次>/annotations/actions_annotations.xlsx(⚠ 檔名是 actions,官方 README 寫成 action)
欄位:Start、End(秒)、Verbs、Nouns、Confusion(標註者把握度)。秒 × 30 = 影片幀號
(2026-10-19 以 3 段標註 × 3 視角目視驗證對齊)。

⚠ 沒對到的動詞 / 名詞一律回報計數,不默默丟掉 —— 其他場次可能有 YH2040 沒有的詞。
"""
from collections import Counter

# 計分的 M3 類別(順序 = 報表順序)
SCORED = ["刀具", "砧板", "食材", "鍋鏟", "鍋子", "容器", "抹布"]

NOUN_TO_CLASS = {
    "Knife": "刀具",
    "Cutting_board": "砧板",
    "Tomatoes": "食材", "Eggs": "食材", "Omelet": "食材", "Tomato_salad": "食材", "Sauce": "食材",
    "Spatula": "鍋鏟",
    "Pan": "鍋子",
    "Bowl": "容器", "Salad_bowl": "容器", "Plate": "容器", "Box": "容器", "Bottle": "容器",
    "Towel": "抹布", "Tissue": "抹布", "Sponge": "抹布",
}

# 已知、刻意不計分的名詞與理由
EXCLUDED_NOUNS = {
    "Hand": "洗手事件真值,下一輪用",
    "Water": "水槽 zone 輔助真值,下一輪用",
    "Fridge": "zone 事件,非 M3 類", "Cupboard": "zone 事件,非 M3 類", "Trash": "zone 事件,非 M3 類",
    "Seasoning": "外觀是瓶子,容器/食材有歧義", "Frying_Oil": "外觀是瓶子,容器/食材有歧義",
    "Spoon": "M3 無此類", "Whip": "M3 無此類", "Recipe": "M3 無此類",
    "Button": "M3 無此類", "Soap": "M3 無此類",
}

# 不算「人與物件接觸」的動詞
NON_CONTACT_VERBS = {"Wait", "Squat", "Read"}
# YH2040 出現過、算接觸的動詞;不在兩個集合裡的動詞視為未知 → 不計分並回報
CONTACT_VERBS = {"Grab", "Carry", "Put", "Move", "Wash", "Hold", "Add", "Open", "Cut", "Close",
                 "Shake", "Stir", "Dry", "Switch", "Sauté", "Touch", "Pour", "Tap", "Press",
                 "Clean", "Split", "Slide"}


def load_actions(xlsx):
    """讀標註檔 → [{start, end, verb, noun, confusion}](依原檔順序)。"""
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h).strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        d = dict(zip(head, r))
        if d.get("Start") is None:
            continue
        out.append(dict(start=float(d["Start"]), end=float(d["End"]),
                        verb=(str(d["Verbs"]).strip() if d.get("Verbs") else None),
                        noun=(str(d["Nouns"]).strip() if d.get("Nouns") else None),
                        confusion=d.get("Confusion")))
    return out


def union(intervals):
    """區間聯集(輸入 [(s, e)],輸出已排序、不重疊)。"""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def build_gt(actions):
    """回傳 (gt, report)。

    gt[cls] = {"segments": 原始段 [(s, e, verb, noun)], "union": 聯集區間 [(s, e)]}
    report  = 各種被排除的計數(未知詞要人工看過)
    """
    segs = {c: [] for c in SCORED}
    rep = dict(n_rows=len(actions), used=Counter(), excluded_noun=Counter(),
               non_contact_verb=Counter(), no_noun=0,
               unknown_noun=Counter(), unknown_verb=Counter())
    for a in actions:
        verb, noun = a["verb"], a["noun"]
        if verb in NON_CONTACT_VERBS:
            rep["non_contact_verb"][f"{verb} {noun}"] += 1
            continue
        if noun is None:
            rep["no_noun"] += 1
            continue
        if verb not in CONTACT_VERBS:
            rep["unknown_verb"][f"{verb} {noun}"] += 1
            continue
        if noun in EXCLUDED_NOUNS:
            rep["excluded_noun"][noun] += 1
            continue
        cls = NOUN_TO_CLASS.get(noun)
        if cls is None:
            rep["unknown_noun"][f"{verb} {noun}"] += 1
            continue
        segs[cls].append((a["start"], a["end"], verb, noun))
        rep["used"][noun] += 1
    gt = {c: {"segments": segs[c], "union": union([(s, e) for s, e, _, _ in segs[c]])}
          for c in SCORED}
    return gt, rep


def print_report(gt, rep):
    print(f"標註共 {rep['n_rows']} 段")
    for c in SCORED:
        u = gt[c]["union"]
        print(f"  {c}: 原始 {len(gt[c]['segments'])} 段,聯集後 {len(u)} 段,"
              f"共 {sum(e - s for s, e in u):.1f} 秒")
    print(f"  已用名詞:{dict(rep['used'])}")
    print(f"  刻意排除的名詞:{dict(rep['excluded_noun'])}")
    print(f"  不算接觸的動詞:{sum(rep['non_contact_verb'].values())} 段;沒有名詞:{rep['no_noun']} 段")
    for k in ("unknown_noun", "unknown_verb"):
        if rep[k]:
            print(f"  ⚠ {k}(未計分,須人工確認):{dict(rep[k])}")
