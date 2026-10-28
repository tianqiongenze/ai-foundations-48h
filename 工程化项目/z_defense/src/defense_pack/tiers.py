# -*- coding: utf-8 -*-
"""工单 Z《综合项目答辩》四档验收判定（档位是算出来的、不是自称的）。

门槛逐字取自工单验收标准（数字外置在 config/rules.yaml 的 tiers: 段）：
  及格(D)：预答辩评委放行；代码规范率≥70%
  良好(C)：链路完整 基本问题全答
  优秀(B)：评委评分≥85；决策记录完整
  优(A) ：创新亮点加部署演示
可实测部分：规范率（pycodestyle）、证据链 5/5（档案袋目录逐层反读）、
基本问题逐题命中（离线规则回答器+关键词核对）、决策记录（docs/ADR 反读）、
部署演示（真实起本地 HTTP 服务、实测端口/状态码/字节数）。
评委放行、评委评分≥85、创新亮点属人工答辩项——判不到就列进 manual，
不冒充实测：这三档的报告行标"机器OK｜人工待确认"，档位行首也写明
"仅机器可判部分，未经答辩确认不得据此宣称"。档位按 D→C→B→A 逐级累加。
"""
import os

KEYS = ("D", "C", "B", "A")
TIERS = KEYS  # 与判档函数同一出处，供 cli/测试引用

#: rules.yaml 缺 tiers 段时的兜底门槛（与工单数字一致，显式标注来源）
DEFAULT_GATES = {
    "D": {"name": "及格(D)", "need": "预答辩评委放行；代码规范率≥70%",
          "style_min": 70.0},
    "C": {"name": "良好(C)", "need": "链路完整基本问题全答",
          "chain_layers_min": 5, "qa_min": 3},
    "B": {"name": "优秀(B)", "need": "评委评分≥85；决策记录完整"},
    "A": {"name": "优(A)", "need": "创新亮点加部署演示"},
}

#: 代码里无法实测、须人工答辩确认的项（档位字母 → 事项）
MANUAL = [
    ("D", "预答辩评委放行（企业导师+组内互评 2/3 票，人工答辩确认）"),
    ("B", "评委评分≥85（答辩当天人工打分）"),
    ("A", "创新亮点（答辩评委判定）"),
]
MANUAL_BY_KEY = dict(MANUAL)


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


def verdict(gates_, rate, files_scanned, chain_ready, chain_total,
            qa_hit, qa_total, decision_ok, demo_ok):
    """按实测值给四档判定；每档 = 可实测功能门（+D 档规范率门），逐级累加。

    chain_ready: 档案袋目录逐层反读实测就绪数（证据链 5/5 才算链路完整）；
    qa_hit: 真实题库逐题生成回答并核对概念词的实测命中数；
    demo_ok: 本地 HTTP 部署演示真实起服务、GET 实测状态码 200 且字节数一致。
    """
    g = gates_
    style_ok = rate >= float(g["D"].get("style_min", 70.0))
    functional = {
        "D": int(files_scanned) > 0,
        "C": (chain_ready >= int(g["C"].get("chain_layers_min", 5))
              and qa_hit >= int(g["C"].get("qa_min", 3))
              and qa_hit == qa_total),
        "B": bool(decision_ok),
        "A": bool(demo_ok),
    }
    out, prev = [], True
    for key in KEYS:
        spec = functional[key] and prev
        if key == "D":
            spec = spec and style_ok
        pending = MANUAL_BY_KEY.get(key)
        row = {"tier": g[key]["name"], "need": g[key]["need"],
               "functional": bool(functional[key]),
               "style_ok": style_ok, "machine_ok": bool(spec),
               "human_pending": pending, "pass": bool(spec)}
        out.append(row)
        prev = bool(spec)
    claimed = next((t["tier"] for t in reversed(out) if t["pass"]),
                   "未达 D 档")
    ceiling = next((t["tier"] for t in reversed(out) if t["machine_ok"]),
                   "未达 D 档")
    return {"tiers": out, "claimed": claimed, "ceiling": ceiling,
            "manual": list(MANUAL),
            "style_gate": float(g["D"].get("style_min", 70.0)),
            "measured": {"rate": rate, "files_scanned": int(files_scanned),
                         "chain_ready": chain_ready,
                         "chain_total": chain_total, "qa_hit": qa_hit,
                         "qa_total": qa_total,
                         "decision_ok": bool(decision_ok),
                         "demo_ok": bool(demo_ok)}}


def render_lines(v):
    """把判定渲染成报告行（实测 + 档位判定，进 output/report.txt）。

    档位只认机器可判部分：含人工项的 D/B/A 档一律标"待答辩确认"，不冒充达标。
    """
    m = v["measured"]
    lines = ["实测达成档位（仅机器可判部分，未经答辩确认不得据此宣称）: %s ｜ "
             "%d 项须人工答辩现场确认（见下行清单）｜ 证据链实测 %d/%d 层就绪 ｜ "
             "基本问题实测命中 "
             "%d/%d ｜ 决策记录完整=%s ｜ 本地 HTTP 部署演示实测=%s ｜ "
             "代码规范率实测 %.1f%%"
             % (v["claimed"], len(v["manual"]),
                m["chain_ready"], m["chain_total"],
                m["qa_hit"], m["qa_total"],
                "是" if m["decision_ok"] else "否",
                "跑通（端口/状态码/字节数见部署演示节）" if m["demo_ok"]
                else "未跑通",
                m["rate"])]
    for t in v["tiers"]:
        if t["tier"] == "及格(D)":
            gate = "规范率门槛≥%.0f%% [%s] ｜ " % (
                v["style_gate"], "达标" if t["style_ok"] else "未达标")
        else:
            gate = ""
        if not t["pass"]:
            state = "缺"
        elif t["human_pending"]:
            state = "机器OK｜人工待确认（%s）" % t["human_pending"]
        else:
            state = "OK"
        lines.append("  %-8s %s功能%s ｜ %s"
                     % (t["tier"], gate, state, t["need"]))
    lines.append("人工答辩项 %d 条，教师答辩时确认：%s"
                 % (len(v["manual"]),
                    "；".join("%s 档：%s" % kv for kv in v["manual"])))
    return lines
