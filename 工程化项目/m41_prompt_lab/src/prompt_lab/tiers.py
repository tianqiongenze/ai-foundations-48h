# -*- coding: utf-8 -*-
"""工单 M4-1《LLM故障诊断》四档验收判定（档位是算出来的、不是自称的）。

门槛逐字取自工单验收标准（数字外置在 config/rules.yaml 的 tiers: 段）：
  及格(D)：能出回答；代码规范率≥70%
  良好(C)：提升≥20% 加对比表
  优秀(B)：断服不崩溃有降级话术；决策记录完整
  优(A) ：提升≥35% 或多轮对话
输入全部来自 pipeline 的实测值（三版对比分数、故障注入降级、ADR 文档反读），
verdict 只做比较，不产生新数据；档位按 D→C→B→A 逐级累加。
"""
import os

TIERS = ["D", "C", "B", "A"]
KEYS = tuple(TIERS)

#: rules.yaml 缺 tiers 段时的兜底门槛（与工单数字一致，显式标注来源）
DEFAULT_GATES = {
    "D": {"name": "及格(D)", "need": "能出回答", "style_min": 70.0},
    "C": {"name": "良好(C)", "need": "提升≥20%加对比表",
          "lift_min": 20.0, "table_rows_min": 3},
    "B": {"name": "优秀(B)",
          "need": "断服不崩溃有降级话术；决策记录完整"},
    "A": {"name": "优(A)", "need": "提升≥35%或多轮对话",
          "lift_a_min": 35.0, "turns_min": 2},
}

#: 代码里无法实测、须人工答辩确认的项（工单"企业元验收栏"师傅签认）
MANUAL = ["企业元验收栏：功能可用□ 参数留痕□ 数据可信□（师傅签字）"]


def gates(cfg_tiers=None):
    """合并 rules.yaml 门槛与兜底默认——数字单一出处在配置文件。"""
    out = {}
    for key in KEYS:
        merged = dict(DEFAULT_GATES[key])
        got = (cfg_tiers or {}).get(key)
        if isinstance(got, dict):
            merged.update(got)
        out[key] = merged
    return out


def decision_record_status(root):
    """实测"决策记录完整"：docs 下 ADR 文档存在且背景/备选/决策/后果四节齐全。"""
    docs = os.path.join(root, "docs")
    if not os.path.isdir(docs):
        return {"exists": False, "n_sections": 0, "missing": ["docs 目录"]}
    names = sorted(n for n in os.listdir(docs)
                   if n.startswith("07") and n.endswith(".md"))
    if not names:
        return {"exists": False, "n_sections": 0, "missing": ["ADR 文件"]}
    with open(os.path.join(docs, names[0]), encoding="utf-8") as f:
        text = f.read()
    need = ["背景", "备选", "决策", "后果"]
    missing = [k for k in need if k not in text]
    return {"exists": bool(names) and not missing, "n_sections": len(need)
            - len(missing), "missing": missing, "file": names[0]}


def verdict(gates_, rate, answers_ok, table_rows, lift_pct, fallback_hit,
            decision_ok, n_turns, context_retained):
    """按实测值给四档判定；每档 = 功能门（+D 档规范率门），档位逐级累加。

    lift_pct: 同一题库裸提示 vs 优化提示两次真实运行分数相减的实测提升；
    fallback_hit: 注入断服后代码确实输出降级话术且零异常逃逸（实测命中）；
    context_retained: 第 2 轮应答复述了第 1 轮关键量（多轮上下文实测）。
    """
    g = gates_
    style_gate = float(g["D"].get("style_min", 70.0))
    style_ok = rate >= style_gate
    functional = {
        "D": bool(answers_ok),
        "C": (lift_pct >= float(g["C"].get("lift_min", 20.0))
              and table_rows >= int(g["C"].get("table_rows_min", 3))),
        "B": bool(fallback_hit) and bool(decision_ok),
        "A": (lift_pct >= float(g["A"].get("lift_a_min", 35.0))
              or (n_turns >= int(g["A"].get("turns_min", 2))
                  and bool(context_retained))),
    }
    out, prev = [], True
    for key in KEYS:
        spec = functional[key] and prev
        if key == "D":
            spec = spec and style_ok
        row = {"tier": g[key]["name"], "need": g[key]["need"],
               "functional": bool(functional[key]),
               "style_ok": style_ok, "pass": bool(spec)}
        out.append(row)
        prev = bool(spec)
    claimed = next((t["tier"] for t in reversed(out) if t["pass"]),
                   "未达 D 档")
    return {"tiers": out, "claimed": claimed, "manual": list(MANUAL),
            "style_gate": style_gate,
            "lift_gate_c": float(g["C"].get("lift_min", 20.0)),
            "lift_gate_a": float(g["A"].get("lift_a_min", 35.0)),
            "measured": {"rate": rate, "answers_ok": bool(answers_ok),
                         "table_rows": table_rows, "lift_pct": lift_pct,
                         "fallback_hit": bool(fallback_hit),
                         "decision_ok": bool(decision_ok),
                         "n_turns": n_turns,
                         "context_retained": bool(context_retained)}}


def render_lines(v):
    """把判定渲染成报告行（实测 + 档位判定，进 output/report.txt）。"""
    m = v["measured"]
    lines = ["实测达成档位: %s ｜ 实测提升 %+.0f%% ｜ 对比表 %d 行 ｜ "
             "断服降级话术命中=%s ｜ 决策记录完整=%s ｜ 多轮 %d 轮且上下文保持=%s"
             " ｜ 代码规范率实测 %.1f%%"
             % (v["claimed"], m["lift_pct"], m["table_rows"],
                "是" if m["fallback_hit"] else "否",
                "是" if m["decision_ok"] else "否", m["n_turns"],
                "是" if m["context_retained"] else "否", m["rate"])]
    for t in v["tiers"]:
        if t["tier"] == "及格(D)":
            gate = "规范率门槛≥%.0f%% [%s] ｜ " % (
                v["style_gate"], "达标" if t["style_ok"] else "未达标")
        elif t["tier"] == "良好(C)":
            gate = "提升门槛≥%.0f%% ｜ " % v["lift_gate_c"]
        elif t["tier"] == "优(A)":
            gate = "提升门槛≥%.0f%%（或多轮）｜ " % v["lift_gate_a"]
        else:
            gate = ""
        lines.append("  %-8s %s功能%s ｜ %s"
                     % (t["tier"], gate,
                        "OK" if t["functional"] else "缺", t["need"]))
    lines.append("人工答辩项 %d 条，教师答辩时确认：%s"
                 % (len(v["manual"]), "；".join(v["manual"]) or "无"))
    return lines
