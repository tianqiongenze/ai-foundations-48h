# -*- coding: utf-8 -*-
"""工单 M4-2《RAG知识库问答》四档验收判定（档位是算出来的、不是自称的）。

门槛逐字取自工单验收标准（数字外置在 config/rules.yaml 的 tiers: 段）：
  及格(D)：幻觉案例≥2条；代码规范率≥70%
  良好(C)：检索≥80% 有引用
  优秀(B)：检索≥85% 可溯源；决策记录完整
  优(A) ：检索≥90% 加融合说明
输入全部来自 pipeline 的实测值（no-RAG 基线产出的幻觉案例、RRF 融合通道
recall@3 实测、引用覆盖率、引用→doc+段落可解析比例、报告融合段落存在性），
verdict 只做比较，不产生新数据；档位按 D→C→B→A 逐级累加。
"""
import os

KEYS = ("D", "C", "B", "A")
TIERS = KEYS  # 与判档函数同一出处，供 cli/测试引用

#: rules.yaml 缺 tiers 段时的兜底门槛（与工单数字一致，显式标注来源）
DEFAULT_GATES = {
    "D": {"name": "及格(D)", "need": "幻觉案例≥2条；代码规范率≥70%",
          "style_min": 70.0, "halluc_min": 2},
    "C": {"name": "良好(C)", "need": "检索≥80%有引用",
          "retrieve_min": 0.80, "cite_min": 1.0},
    "B": {"name": "优秀(B)", "need": "检索≥85%可溯源；决策记录完整",
          "retrieve_min": 0.85, "trace_min": 1.0},
    "A": {"name": "优(A)", "need": "检索≥90%加融合说明",
          "retrieve_min": 0.90, "fusion_lines_min": 2},
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
    return {"exists": not missing, "n_sections": len(need) - len(missing),
            "missing": missing, "file": names[0]}


def traceable_rate(citations, kb):
    """可溯源实测口径：引用 id 能解析回知识库"文档类目+段落原文"的比例。"""
    idx = {k["id"]: k for k in kb}
    ok = sum(1 for c in citations
             if c["id"] in idx and idx[c["id"]].get("category")
             and idx[c["id"]].get("text"))
    return (ok / len(citations)) if citations else 0.0


def fusion_explain_lines(sections):
    """融合说明实测：报告分节中真实存在"融合"解释的行数（不是口头宣称）。"""
    n = 0
    for _title, lines in sections:
        for ln in lines:
            if "融合" in str(ln):
                n += 1
    return n


def verdict(gates_, rate, halluc_cases, retrieve_rate, cite_cov, trace_rate,
            fusion_lines, decision_ok):
    """按实测值给四档判定；每档 = 功能门（+D 档规范率门），档位逐级累加。"""
    g = gates_
    style_ok = rate >= float(g["D"].get("style_min", 70.0))
    functional = {
        "D": halluc_cases >= int(g["D"].get("halluc_min", 2)),
        "C": (retrieve_rate >= float(g["C"].get("retrieve_min", 0.80))
              and cite_cov >= float(g["C"].get("cite_min", 1.0))),
        "B": (retrieve_rate >= float(g["B"].get("retrieve_min", 0.85))
              and trace_rate >= float(g["B"].get("trace_min", 1.0))
              and bool(decision_ok)),
        "A": (retrieve_rate >= float(g["A"].get("retrieve_min", 0.90))
              and fusion_lines >= int(g["A"].get("fusion_lines_min", 2))),
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
            "style_gate": float(g["D"].get("style_min", 70.0)),
            "measured": {"rate": rate, "halluc_cases": halluc_cases,
                         "retrieve_rate": retrieve_rate,
                         "cite_cov": cite_cov, "trace_rate": trace_rate,
                         "fusion_lines": fusion_lines,
                         "decision_ok": bool(decision_ok)}}


def render_lines(v):
    """把判定渲染成报告行（实测 + 档位判定，进 output/report.txt）。"""
    m = v["measured"]
    lines = ["实测达成档位: %s ｜ 幻觉案例实测 %d 条 ｜ 检索命中率（RRF 融合"
             "通道 recall@3）%.1f%% ｜ 引用覆盖率 %.0f%% ｜ 可溯源比例 %.0f%%"
             " ｜ 融合说明 %d 行 ｜ 决策记录完整=%s ｜ 代码规范率实测 %.1f%%"
             % (v["claimed"], m["halluc_cases"], m["retrieve_rate"] * 100,
                m["cite_cov"] * 100, m["trace_rate"] * 100,
                m["fusion_lines"], "是" if m["decision_ok"] else "否",
                m["rate"])]
    for t in v["tiers"]:
        if t["tier"] == "及格(D)":
            gate = "规范率门槛≥%.0f%% [%s] ｜ " % (
                v["style_gate"], "达标" if t["style_ok"] else "未达标")
        else:
            gate = ""
        lines.append("  %-8s %s功能%s ｜ %s"
                     % (t["tier"], gate,
                        "OK" if t["functional"] else "缺", t["need"]))
    lines.append("人工答辩项 %d 条，教师答辩时确认：%s"
                 % (len(v["manual"]), "；".join(v["manual"]) or "无"))
    return lines
