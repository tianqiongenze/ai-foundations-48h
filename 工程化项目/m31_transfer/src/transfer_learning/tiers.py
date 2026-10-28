# -*- coding: utf-8 -*-
"""工单 M3-1 四档验收判定（纯函数，档位是算出来的、不是自称的）。

门槛逐条取自《M3-1工单-ResNet迁移学习.md》验收标准（数字外置在
config/rules.yaml 的 tiers: 段，此处 DEFAULT_GATES 只是缺省回落）：
  及格(D)：准确率≥70%；代码规范率≥70%
  良好(C)：准确率≥75% 加对照表
  优秀(B)：≥80% 或增强有效论证；决策记录完整
  优(A)：准确率≥85% 加曲线
输入全部来自 pipeline 的实测值（2×2 真训实测准确率、PEP8 体检、对照表
行数、docs/ 决策记录条目数、曲线落盘），verdict 只做比较，不产生新数据。
高档累计 D 档规范率底线：工单只在 D 档写了这条线，档位是递进制，
高档必然包含低档底线——数字仍逐字取自工单，未新增任何门槛。
"""
import os
import re

TIERS = [
    ("及格(D)", "准确率≥70%；代码规范率≥70%", "D"),
    ("良好(C)", "准确率≥75%加对照表", "C"),
    ("优秀(B)", "≥80%或增强有效论证；决策记录完整", "B"),
    ("优(A)", "准确率≥85%加曲线", "A"),
]
KEYS = ("D", "C", "B", "A")

#: 无法机器判定、必须教师答辩现场确认的条目（报告如实列出，不冒充达标）
MANUAL = [
    "决策结论与理由的口头陈述（工单：答辩必答）",
    "企业元验收栏签认（功能可用/参数留痕/数据可信，值班师傅签字）",
    "底座口径是否可接受（真ResNet=本地缓存离线载入 / 兜底=本地CNN，实测证据见报告底座口径段）",
]

DEFAULT_GATES = {
    "D": {"acc": 0.70, "style_rate": 70.0},
    "C": {"acc": 0.75, "compare_rows": 2},
    "B": {"acc": 0.80, "decisions": 1},
    "A": {"acc": 0.85, "curves": 1},
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


def count_numeric_rows(lines):
    """对照表"完整"的机器口径：逐行数——带实测数字的行才算对照行。"""
    return sum(1 for ln in lines if _MEASURED_RE.search(str(ln)))


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


def verdict(acc, style_rate, compare_rows, aug_gains, decisions, curves,
            gates=None, miss_note=None):
    """按实测值给四档判定；每档 = 工单功能条件 + 规范率底线，缺一即未达标。

    acc/curves 等来自真实训练评测；acc=None 表示图像通道未跑通——一律
    MISS/未达标，绝不把"没测"报成"达标"。aug_gains: {"frozen": +x,
    "finetune": +y}，两路皆正才算"增强有效论证"（与结论段同一口径）。
    """
    g = merge_gates(gates)
    gd, gc, gb, ga = g["D"], g["C"], g["B"], g["A"]
    has_acc = acc is not None
    aug_effective = bool(aug_gains) and all(x > 0
                                            for x in aug_gains.values())
    f_b_acc = has_acc and acc >= gb["acc"]
    functional = {
        "D": has_acc and acc >= gd["acc"],
        "C": has_acc and acc >= gc["acc"],
        "B": f_b_acc or aug_effective,
        "A": has_acc and acc >= ga["acc"],
    }
    functional["C"] = (functional["C"]
                       and compare_rows >= gc["compare_rows"])
    functional["B"] = functional["B"] and decisions >= gb["decisions"]
    functional["A"] = functional["A"] and curves >= ga["curves"]
    style_ok = bool(style_rate is not None
                    and style_rate >= gd["style_rate"])
    out = []
    for (name, need, key), _ in zip(TIERS, KEYS):
        ok = bool(functional[key]) and style_ok
        out.append({"tier": name, "need": need, "key": key,
                    "gate": g["D"]["style_rate"],
                    "functional": bool(functional[key]),
                    "style_ok": bool(style_ok), "pass": ok})
    claimed = next((t["tier"] for t in reversed(out) if t["pass"]),
                   "未达 D 档")
    return {"tiers": out, "claimed": claimed, "manual": list(MANUAL),
            "miss_note": miss_note, "gates": g,
            "measured": {"acc": acc, "style_rate": style_rate,
                         "compare_rows": compare_rows,
                         "aug_gains": dict(aug_gains or {}),
                         "aug_effective": aug_effective,
                         "decisions": decisions, "curves": curves}}


def render_lines(v):
    """把判定渲染成报告行（实测 + 档位判定，进 output/report.txt）。"""
    m = v["measured"]
    acc = "MISS（%s）" % (v["miss_note"] or "图像迁移通道未跑通") \
        if m["acc"] is None else "%.4f" % m["acc"]
    rate = "MISS" if m["style_rate"] is None \
        else "%.2f%%" % m["style_rate"]
    gains = m["aug_gains"]
    aug_txt = "MISS" if not gains else "冻结 %+.4f/微调 %+.4f" % (
        gains.get("frozen", 0.0), gains.get("finetune", 0.0))
    lines = ["实测达成档位: %s ｜ 实测最优准确率 %s ｜ 规范率 %s ｜ "
             "对照表实测行 %d 条 ｜ 增强增益 %s ｜ 决策记录 %d 条 ｜ 曲线 %d 张"
             % (v["claimed"], acc, rate, m["compare_rows"], aug_txt,
                m["decisions"], m["curves"])]
    g = v.get("gates") or DEFAULT_GATES

    def fmt(key, name):
        """门槛渲染：规范率按百分比、其余按数值。"""
        gate = g[key][name]
        return "%g" % gate if name != "style_rate" else "%.2f%%" % gate

    detail = {
        "D": "准确率门槛≥%s（实测 %s）" % (fmt("D", "acc"), acc),
        "C": "准确率≥%s 且对照行≥%d（实测 %s / %d 行）"
             % (fmt("C", "acc"), g["C"]["compare_rows"], acc,
                m["compare_rows"]),
        "B": "准确率≥%s 或增强有效（增益见上行）且决策记录≥%d（实测 %d）"
             % (fmt("B", "acc"), g["B"]["decisions"], m["decisions"]),
        "A": "准确率≥%s 且有训练曲线（实测 %s / %d 张）"
             % (fmt("A", "acc"), acc, m["curves"]),
    }
    for t in v["tiers"]:
        lines.append("  %-8s 规范率门槛≥%.0f%% [%s] ｜ 功能%s ｜ %s"
                     % (t["tier"], t["gate"],
                        "达标" if t["pass"] else "未达标",
                        "OK" if t["functional"] else "缺", detail[t["key"]]))
    lines.append("人工答辩项 %d 条，教师答辩时确认：" % len(v["manual"]))
    lines.extend("  - %s" % x for x in v["manual"])
    return lines
