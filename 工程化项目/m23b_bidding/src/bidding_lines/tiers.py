# -*- coding: utf-8 -*-
"""工单 M2-3b 四档验收判定（纯函数，档位是算出来的、不是自称的）。

门槛逐条取自《M2-3b工单-CNN或LSTM二选一（竞标）》"四、验收标准"：
  及格(D)：指标可复现；代码规范率≥70%
  良好(C)：CNN≥75% 或 LSTM 误差达标（兜底臂数字不认）
  优秀(B)：CNN/LSTM 两轮实测对比 + 决策记录完整（竞标胜负/评委加分属人工）
  优(A)  ：给出选型决策矩阵（候选×权重×实测得分→结论）
数字外置在 config/rules.yaml 的 tiers: 段（gates() 读出，缺失才走
DEFAULT_GATES 兜底——兜底值同样逐字取自工单，不许自创更高的线）。
输入全部来自 pipeline 的实测值（复现比对、真引擎 acc/误差、矩阵行数与
权重和、成文字数），verdict 只做比较，不产生新数据——与 style 同口径。
"赢得竞标或评委加分"程序判不了，进 MANUAL：如实列出，不冒充实测。
"""

KEYS = ("D", "C", "B", "A")
LABELS = {"D": "及格(D)", "C": "良好(C)", "B": "优秀(B)", "A": "优(A)"}

#: rules.yaml 缺 tiers: 段时的兜底——数字逐字取自工单验收标准表
DEFAULT_GATES = {
    "D": {"require_reproducible": True, "min_code_style_pct": 70.0},
    "C": {"cnn_min_acc": 0.75, "lstm_max_err": 0.25,
          "require_real_engine": True},
    "B": {"require_two_arm_compare": True, "min_decision_chars": 120},
    "A": {"min_matrix_rows": 4, "require_weights": True,
          "min_conclusion_chars": 20,
          "weight_acc": 0.5, "weight_speed": 0.3, "weight_env": 0.2},
}

NEEDS = {
    "D": "指标可复现；代码规范率≥70%",
    "C": "CNN≥75% 或 LSTM 误差≤25%（仅认真卷积/真 LSTM 臂的实测数字）",
    "B": "CNN vs LSTM 两轮实测对比齐备；决策记录完整（竞标胜负/评委加分另人工）",
    "A": "选型决策矩阵：候选×权重×实测得分→结论",
}

#: 程序判不了、必须现场人工核验的项（如实列出，不冒充实测）
MANUAL = [
    "赢得竞标或评委加分：现场评委裁决，程序无实测依据（工单 B 档人评半，"
    "证据包=选型决策矩阵+两轮对比+泄漏核查）",
    "企业元验收栏师傅签认：功能可用/参数留痕/数据可信（学校元70%+企业元30%）",
    "工单末页学习反思与答辩陈述（解释说服力人工评）",
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
    engine_ok = (not g["C"]["require_real_engine"]) or \
        (m["cnn_engine"] == "torch" or m["lstm_engine"] == "torch")
    c_cnn = m["cnn_acc"] >= g["C"]["cnn_min_acc"] - 1e-9 and \
        (not g["C"]["require_real_engine"] or m["cnn_engine"] == "torch")
    c_lstm = m["lstm_err"] <= g["C"]["lstm_max_err"] + 1e-9 and \
        (not g["C"]["require_real_engine"] or m["lstm_engine"] == "torch")
    b_two_arm = (not g["B"]["require_two_arm_compare"]) \
        or bool(m["two_arm_real"])
    a_w = (not g["A"]["require_weights"]) or \
        abs(m["weights_sum"] - 1.0) < 1e-6
    d_req = g["D"]["require_reproducible"]
    return {
        "D": bool(m["reproducible"]) if d_req else True,
        "C": bool((c_cnn or c_lstm) and engine_ok),
        "B": bool(b_two_arm
                  and m["decision_chars"] >= g["B"]["min_decision_chars"]),
        "A": bool(m["matrix_rows"] >= g["A"]["min_matrix_rows"] and a_w
                  and m["conclusion_chars"] >= g["A"]["min_conclusion_chars"]),
    }


def verdict(m, st, g):
    """按实测值 m（复现比对/引擎与指标/矩阵结构/成文字数）判四档。"""
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
             "复现比对 %s ｜ CNN(%s) acc %.4f ｜ LSTM 误差 %.4f ｜ 决策矩阵 %d 行"
             % (v["claimed"], v["style_rate"],
                v["gates"]["D"]["min_code_style_pct"],
                "一致" if m["reproducible"] else "漂移",
                m["cnn_engine"], m["cnn_acc"], m["lstm_err"],
                m["matrix_rows"])]
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
