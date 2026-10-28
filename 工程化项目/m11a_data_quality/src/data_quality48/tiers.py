# -*- coding: utf-8 -*-
"""工单 M1-1a 四档验收判定（纯函数，档位是算出来的、不是自称的）。

门槛逐条取自《M1-1a工单-数据清洗与质量评估.md》"四、验收标准"：
  及格(D)：温度列缺失率降为0且行数不变；代码规范率≥70%
  良好(C)：缺失处理率≥95% + 三清单齐全
  优秀(B)：评分≥85且理由完整；决策记录完整
  优(A)  ：预测填充优于均值且论证清楚
数字外置在 config/rules.yaml 的 tiers: 段（gates() 读出，缺失才走
DEFAULT_GATES 兜底——兜底值同样逐字取自工单，不许自创更高的线）。
输入全部来自 pipeline 的实测值，verdict 只做比较，不产生新数据。
"论证清楚/口头答辩"类人工项进 MANUAL：程序只实测"段落存在且字数达标"，
说服力另计现场答辩，不在实测里许诺。
"""

KEYS = ("D", "C", "B", "A")
LABELS = {"D": "及格(D)", "C": "良好(C)", "B": "优秀(B)", "A": "优(A)"}

#: rules.yaml 缺 tiers: 段时的兜底——数字逐字取自工单验收标准表
DEFAULT_GATES = {
    "D": {"max_temp_missing_cells": 0, "require_rows_unchanged": True,
          "min_code_style_pct": 70.0},
    "C": {"min_handle_rate_pct": 95.0, "min_lists": 3},
    "B": {"min_quality_score": 85.0, "min_reason_entries": 4,
          "min_decision_chars": 120},
    "A": {"require_pred_better": True, "min_argument_chars": 200},
}

NEEDS = {
    "D": "温度列缺失率降为0且行数不变；代码规范率≥70%",
    "C": "缺失处理率≥95%+三清单齐全",
    "B": "评分≥85且理由完整；决策记录完整",
    "A": "预测填充优于均值且论证清楚",
}

#: 程序判不了、必须现场人工核验的项（如实列出，不冒充实测）
MANUAL = [
    "口头答辩：逐列缺失机制判断与填充策略是否讲得清（企业元师傅签认）",
    "论证说服力：A 档'预测优于均值'的机理解释（程序只判段落存在+字数达标）",
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
    reaches_b = (m["quality_after"] >= g["B"]["min_quality_score"]
                 and m["n_reasons"] >= g["B"]["min_reason_entries"]
                 and m["decision_chars"] >= g["B"]["min_decision_chars"])
    return {
        "D": (m["temp_missing_after"] <= g["D"]["max_temp_missing_cells"]
              and (m["rows_unchanged"]
                   if g["D"]["require_rows_unchanged"] else True)),
        "C": (m["handle_rate_pct"] >= g["C"]["min_handle_rate_pct"]
              and m["n_lists"] >= g["C"]["min_lists"]),
        "B": reaches_b,
        "A": (m["pred_better"] if g["A"]["require_pred_better"] else True),
    }


def verdict(m, st, g):
    """按实测值 m（缺失/处理率/清单/质量分/填充实验/报告段落字数）判四档。"""
    biz = _functional(m, g)
    # A 档"论证清楚"的可实测部分：报告论证段落真实存在且字数达标
    biz["A"] = biz["A"] and m["argument_chars"] >= g["A"]["min_argument_chars"]
    folded = fold(biz, st)
    out = []
    for key in KEYS:
        gate = g["D"]["min_code_style_pct"] if key == "D" else None
        ok = bool(folded["final"][key])
        out.append({"tier": LABELS[key], "key": key, "need": NEEDS[key],
                    "gate": gate, "functional": bool(biz[key]),
                    "style_ok": True if gate is None
                    else st["rate"] >= gate, "pass": ok,
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
             "temp 缺失 %d 格 ｜ 缺失处理率 %.2f%% ｜ 三清单 %d 张 ｜ 质量分 %.2f ｜ "
             "预测填充 MAE %.4f < 均值 %.4f（%s）"
             % (v["claimed"], v["style_rate"],
                v["gates"]["D"]["min_code_style_pct"],
                m["temp_missing_after"], m["handle_rate_pct"], m["n_lists"],
                m["quality_after"], m["mae_pred"], m["mae_mean"],
                "优于" if m["pred_better"] else "未优于")]
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
