# -*- coding: utf-8 -*-
"""工单 M1-1b 四档验收判定（纯函数，档位是算出来的、不是自称的）。

门槛逐条取自《M1-1b工单-特征设计与优化.md》"四、验收标准"：
  及格(D)：5 特征表能跑通；代码规范率≥70%
  良好(C)：≥10 特征且字典完整
  优秀(B)：特征 −30% 性能不降；决策记录完整
  优(A)  ：新特征进 Top3 且有物理解释
数字外置在 config/rules.yaml 的 tiers: 段（gates() 读出，缺失才走
DEFAULT_GATES 兜底——兜底值同样逐字取自工单，不许自创更高的线）。
输入全部来自 pipeline 的实测值（CV 指标、字典自检、消融差值、榜名次），
verdict 只做比较，不产生新数据——与 style 同口径，可复核。
"物理解释/口头答辩"类人工项进 MANUAL：程序实测"解释成文且字数达标"，
说服力另计现场答辩。
"""

KEYS = ("D", "C", "B", "A")
LABELS = {"D": "及格(D)", "C": "良好(C)", "B": "优秀(B)", "A": "优(A)"}

#: rules.yaml 缺 tiers: 段时的兜底——数字逐字取自工单验收标准表
DEFAULT_GATES = {
    "D": {"min_features": 5, "min_code_style_pct": 70.0},
    "C": {"min_features": 10, "dict_required_cols": 5},
    "B": {"cut_frac": 0.30, "max_acc_drop": 0.03, "min_decision_chars": 120},
    "A": {"top_n": 3, "min_new_in_top": 1, "min_explain_chars": 40},
}

NEEDS = {
    "D": "5 特征表能跑通；代码规范率≥70%",
    "C": "≥10 特征且字典完整（名称/公式/含义/单位/范围五栏成文）",
    "B": "特征砍 30% 性能不降（消融验证）；决策记录完整",
    "A": "新特征进 Top3 且有物理解释",
}

#: 程序判不了、必须现场人工核验的项（如实列出，不冒充实测）
MANUAL = [
    "口头答辩：为什么比值型特征对工况免疫（企业元师傅签认）",
    "解释说服力：新特征物理解释是否讲得通（程序只判成文+字数达标）",
    "工单末页学习反思单三问（计入过程分 10%）",
]


def gates(raw_cfg):
    """从 config/rules.yaml 的 tiers: 段读四档门槛（逐键兜底 DEFAULT_GATES）。"""
    raw = (raw_cfg or {}).get("tiers") or {}
    out = {}
    for k, dflt in DEFAULT_GATES.items():
        got = raw.get(k) or {}
        merged = dict(dflt)
        if isinstance(got, dict):
            for name, val in got.items():
                if name in dflt:
                    merged[name] = val
        out[k] = merged
    return out


def fold(biz, st, order=KEYS):
    """把各档"功能实测达标"布尔与规范率引擎折叠成档位判定。

    工单只把"代码规范率≥70%"钉在 D 档，故仅 D 叠加规范率门槛；
    claimed 取"从 D 起连续达标"的最高档——低档没过不许凭空承诺高档。
    """
    final = {t: bool(biz.get(t)) for t in order}
    final["D"] = final["D"] and bool(st["passes"])
    claimed = "未达 D 档"
    claimed_key = None
    for t in order:
        if final[t]:
            claimed = LABELS[t]
            claimed_key = t
        else:
            break
    return {"final": final, "claimed": claimed, "claimed_key": claimed_key,
            "order": list(order)}


def _functional(m, g):
    """四档功能门：只拿实测值跟工单门槛比，不产生新数据。"""
    return {
        "D": (m["min5_n_features"] == g["D"]["min_features"]
              and m["min5_acc"] > 0.0),
        "C": (m["n_features"] >= g["C"]["min_features"]
              and m["dict_missing_cells"] == 0
              and m["dict_cols"] >= g["C"]["dict_required_cols"]),
        "B": (m["cut_pct_actual"] >= 100.0 * g["B"]["cut_frac"] - 1e-9
              and m["tail_drop"] <= g["B"]["max_acc_drop"] + 1e-9
              and m["decision_chars"] >= g["B"]["min_decision_chars"]),
        "A": (m["n_new_in_top"] >= g["A"]["min_new_in_top"]
              and m["explain_chars"] >= g["A"]["min_explain_chars"]),
    }


def verdict(m, st, g):
    """按实测值 m（CV 指标/字典自检/消融差值/新特征榜名次）判四档。"""
    biz = _functional(m, g)
    folded = fold(biz, st)
    out = []
    for key in KEYS:
        gate = g["D"]["min_code_style_pct"] if key == "D" else None
        out.append({"tier": LABELS[key], "key": key, "need": NEEDS[key],
                    "gate": gate, "functional": bool(biz[key]),
                    "style_ok": True if gate is None
                    else st["rate"] >= gate,
                    "pass": bool(folded["final"][key]),
                    "evidence": m["evidence"][key]})
    return {"tiers": out, "final": folded["final"],
            "claimed": folded["claimed"],
            "claimed_key": folded["claimed_key"],
            "measured": {k: v for k, v in m.items() if k != "evidence"},
            "evidence": m["evidence"], "gates": g,
            "style_rate": st["rate"], "manual": list(MANUAL)}


def render_lines(v):
    """把判定渲染成报告行（实测 + 档位判定，进 output/report.txt）。"""
    m = v["measured"]
    lines = ["实测达成档位: %s ｜ 代码规范率 %.2f%%（工单把≥%.0f%%钉在 D 档）｜ "
             "特征 %d 个（最小集 acc %.4f）｜ 消融砍 %.1f%% 后 acc %+.4f ｜ "
             "Top%d 内新特征 %d 个"
             % (v["claimed"], v["style_rate"],
                v["gates"]["D"]["min_code_style_pct"], m["n_features"],
                m["min5_acc"], m["cut_pct_actual"], m["tail_drop"],
                v["gates"]["A"]["top_n"], m["n_new_in_top"])]
    for t in v["tiers"]:
        if t["gate"] is None:
            lines.append("  %-8s 规范率不另设线（工单仅 D 档设线，实测 %.2f%%）"
                         " [%s] ｜ 功能%s ｜ %s ｜ 实测 %s"
                         % (t["tier"], v["style_rate"],
                            "达标" if t["pass"] else "未达标",
                            "OK" if t["functional"] else "缺",
                            t["need"], t["evidence"]))
        else:
            lines.append("  %-8s 规范率门槛≥%.0f%% [%s] ｜ 功能%s ｜ %s ｜ 实测 %s"
                         % (t["tier"], t["gate"],
                            "达标" if t["pass"] else "未达标",
                            "OK" if t["functional"] else "缺",
                            t["need"], t["evidence"]))
    lines.append("  人工答辩项（程序不判、不许自称达标）：%s"
                 % "；".join(v["manual"]))
    return lines
