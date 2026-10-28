# -*- coding: utf-8 -*-
"""工单 M3-2 四档验收判定（纯函数，档位是算出来的、不是自称的）。

门槛逐条取自《M3-2工单-YOLO实时缺陷检测.md》「四、验收标准」
（数字外置在 config/rules.yaml 的 tiers: 段，此处 DEFAULT_GATES 只是缺省回落）：
  及格(D)：基线mAP有数；代码规范率≥70%
  良好(C)：mAP≥0.60 表完整
  优秀(B)：给出推荐点及理由；决策记录完整
  优(A)：mAP≥0.75 或 加速≥50%
输入全部来自 pipeline 的实测值（AP@0.5 由 IoU 贪心匹配+插值 AP 真算、
扫描表逐格实测、PEP8 体检、docs/ 决策记录条目数），verdict 只做比较，
不产生新数据。高档累计 D 档规范率底线：工单只在 D 档写了这条线，档位是
递进制，高档必然包含低档底线——数字仍逐字取自工单，未新增任何门槛。
"""
import os
import re

TIERS = [
    ("及格(D)", "基线mAP有数；代码规范率≥70%", "D"),
    ("良好(C)", "mAP≥0.60表完整", "C"),
    ("优秀(B)", "给出推荐点及理由；决策记录完整", "B"),
    ("优(A)", "mAP≥0.75或加速≥50%", "A"),
]
KEYS = ("D", "C", "B", "A")

#: 无法机器判定、必须教师答辩现场确认的条目（报告如实列出，不冒充达标）
MANUAL = [
    "推荐理由的口头陈述与答辩问答（工单：方案对比周必答）",
    "企业元验收栏签认（功能可用/参数留痕/数据可信，值班师傅签字）",
    "ONNX 通道未运行时的兜底口径是否可接受（见报告 ONNX 段）",
]

DEFAULT_GATES = {
    "D": {"style_rate": 70.0},
    "C": {"ap": 0.60},
    "B": {"decisions": 1},
    "A": {"ap": 0.75, "speedup_pct": 50.0},
}

_MEASURED_RE = re.compile(r"\d\.\d{2,}")


def merge_gates(overrides):
    """rules.yaml tiers: 段覆盖缺省门槛——数字外置，改配置不改代码。"""
    out = {k: dict(v) for k, v in DEFAULT_GATES.items()}
    for k, v in (overrides or {}).items():
        if k in out and isinstance(v, dict):
            for kk, vv in v.items():
                if kk in out[k]:
                    out[k][kk] = float(vv)
    return out


def count_decision_entries(root):
    """实测 docs/ 里的决策记录条目数：'## 决策' 标题后须有非空正文。"""
    docs = os.path.join(root, "docs")
    n = 0
    if not os.path.isdir(docs):
        return 0
    for fn in sorted(os.listdir(docs)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(docs, fn), encoding="utf-8") as f:
            text = f.read()
        for seg in re.split(r"^##\s*决策\s*$", text, flags=re.M)[1:]:
            body = next((ln.strip() for ln in seg.splitlines()
                         if ln.strip() and not ln.startswith("#")), "")
            if len(body) >= 8:                       # 空标题不算一条决策
                n += 1
    return n


def verdict(ap, style_rate, sweep_rows, sweep_cells, sweep_cells_expected,
            rec_ap, reason, speedup, decisions, gates=None, miss_note=None):
    """按实测值给四档判定；每档 = 工单功能条件 + D 档规范率底线，缺一即未达标。

    ap=None 表示基线 mAP 未算出（无 GT/未运行）——一律未达标，绝不把"没测"
    报成"达标"。表完整＝扫描行数和格子数都到齐（格子数由 config 的 conf×IoU
    组合现场算出，不写死）。理由"挂得住"＝推荐理由文本里带着实测数字，
    空话不算理由。speedup 是 win16 vs win4 同口径实测加速百分比。
    """
    g = merge_gates(gates)
    has_ap = ap is not None
    table_complete = (sweep_rows >= 1
                      and sweep_cells_expected >= 1
                      and sweep_cells >= sweep_cells_expected)
    reason_ok = bool(reason) and bool(_MEASURED_RE.search(str(reason)))
    functional = {
        "D": has_ap,
        "C": has_ap and ap >= g["C"]["ap"] and table_complete,
        "B": (rec_ap is not None and reason_ok
              and decisions >= g["B"]["decisions"]),
        "A": ((has_ap and ap >= g["A"]["ap"])
              or speedup >= g["A"]["speedup_pct"]),
    }
    style_ok = bool(style_rate is not None
                    and style_rate >= g["D"]["style_rate"])
    out = []
    for (name, need, key), _ in zip(TIERS, KEYS):
        out.append({"tier": name, "need": need, "key": key,
                    "gate": g["D"]["style_rate"],
                    "functional": bool(functional[key]),
                    "style_ok": bool(style_ok),
                    "pass": bool(functional[key]) and style_ok})
    claimed = next((t["tier"] for t in reversed(out) if t["pass"]),
                   "未达 D 档")
    return {"tiers": out, "claimed": claimed, "manual": list(MANUAL),
            "miss_note": miss_note, "gates": g,
            "measured": {"ap": ap, "style_rate": style_rate,
                         "sweep_rows": sweep_rows, "sweep_cells": sweep_cells,
                         "sweep_cells_expected": sweep_cells_expected,
                         "table_complete": table_complete,
                         "rec_ap": rec_ap, "reason_ok": reason_ok,
                         "speedup": speedup, "decisions": decisions}}


def render_lines(v):
    """把判定渲染成报告行（实测 + 档位判定，进 output/report.txt）。"""
    m = v["measured"]
    g = v.get("gates") or DEFAULT_GATES
    ap = "MISS（%s）" % (v["miss_note"] or "基线 mAP 未算出") \
        if m["ap"] is None else "%.4f" % m["ap"]
    rec = "MISS" if m["rec_ap"] is None else "%.4f" % m["rec_ap"]
    rate = "MISS" if m["style_rate"] is None else "%.2f%%" % m["style_rate"]
    lines = ["实测达成档位: %s ｜ 实测 AP@0.5 %s ｜ 扫描表 %d 行/%d 格（应 %d 格）｜ "
             "推荐点实测 AP %s ｜ win16 实测加速 %+.1f%% ｜ 决策记录 %d 条 ｜ 规范率 %s"
             % (v["claimed"], ap, m["sweep_rows"], m["sweep_cells"],
                m["sweep_cells_expected"], rec, m["speedup"],
                m["decisions"], rate)]
    detail = {
        "D": "基线 mAP 有数（实测 %s）" % ap,
        "C": "AP@0.5≥%g 且表完整（实测 %s / %d 行 %d 格，应 %d 格）"
             % (g["C"]["ap"], ap, m["sweep_rows"], m["sweep_cells"],
                m["sweep_cells_expected"]),
        "B": "推荐点+理由挂在实测数字+决策记录≥%d（实测推荐 AP %s，"
             "理由含实测数字=%s，决策记录 %d 条）"
             % (g["B"]["decisions"], rec, m["reason_ok"], m["decisions"]),
        "A": "AP@0.5≥%g 或 加速≥%g%%（实测 %s / %+.1f%%）"
             % (g["A"]["ap"], g["A"]["speedup_pct"], ap, m["speedup"]),
    }
    for t in v["tiers"]:
        lines.append("  %-8s 规范率门槛≥%.0f%% [%s] ｜ 功能%s ｜ %s"
                     % (t["tier"], t["gate"],
                        "达标" if t["pass"] else "未达标",
                        "OK" if t["functional"] else "缺", detail[t["key"]]))
    lines.append("人工答辩项 %d 条，教师答辩时确认：" % len(v["manual"]))
    lines.extend("  - %s" % x for x in v["manual"])
    return lines
