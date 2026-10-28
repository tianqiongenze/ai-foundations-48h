# -*- coding: utf-8 -*-
"""工单 M5-1《多智能体检修调度》四档验收判定（档位是算出来的、不是自称的）。

门槛逐字取自工单验收标准（数字外置在 config/rules.yaml 的 tiers: 段）：
  及格(D)：基线数据可复现；代码规范率≥70%
  良好(C)：正确率≥75% 加说明书
  优秀(B)：修复1个漏洞有证据；决策记录完整
  优(A) ：正确率≥90% 或泛化实验
输入全部来自 pipeline 的实测值（同种子二次运行逐键比对、调度正确性校验、
docs 运行手册字数反读、漏洞 before/after 成对实测输出、未见种子泛化评估），
verdict 只做比较，不产生新数据；档位按 D→C→B→A 逐级累加。
"""
import os

KEYS = ("D", "C", "B", "A")
TIERS = KEYS  # 与判档函数同一出处，供 cli/测试引用

#: rules.yaml 缺 tiers 段时的兜底门槛（与工单数字一致，显式标注来源）
DEFAULT_GATES = {
    "D": {"name": "及格(D)", "need": "基线数据可复现；代码规范率≥70%",
          "style_min": 70.0},
    "C": {"name": "良好(C)", "need": "正确率≥75%加说明书",
          "acc_min": 0.75, "manual_chars_min": 200},
    "B": {"name": "优秀(B)", "need": "修复1个漏洞有证据；决策记录完整",
          "wrong_before_min": 1},
    "A": {"name": "优(A)", "need": "正确率≥90%或泛化实验",
          "acc_a_min": 0.90, "gen_episodes_min": 5},
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


def manual_status(root):
    """使用说明书实测：docs/05-运行手册.md 真实存在，并统计非空正文字数。"""
    docs = os.path.join(root, "docs")
    names = sorted(n for n in os.listdir(docs)
                   if n.startswith("05") and n.endswith(".md")) \
        if os.path.isdir(docs) else []
    if not names:
        return {"exists": False, "chars": 0, "file": ""}
    with open(os.path.join(docs, names[0]), encoding="utf-8") as f:
        text = f.read()
    return {"exists": True, "chars": len(text.replace(" ", "").replace(
        "\n", "")), "file": names[0]}


def verdict(gates_, rate, reproducible, acc_best, manual_chars,
            wrong_before, reward_delta, decision_ok, unseen_acc,
            gen_episodes):
    """按实测值给四档判定；每档 = 功能门（+D 档规范率门），档位逐级累加。

    reproducible: 同种子两次真实运行逐键相等（基线可复现实测）；
    reward_delta: 漏洞修复后同一策略总奖励变化（负值=套利被剥掉，成对实测）；
    unseen_acc/gen_episodes: 未见种子泛化实验的实测正确率与集合数。
    """
    g = gates_
    style_ok = rate >= float(g["D"].get("style_min", 70.0))
    functional = {
        "D": bool(reproducible),
        "C": (acc_best >= float(g["C"].get("acc_min", 0.75))
              and manual_chars >= int(g["C"].get("manual_chars_min", 200))),
        "B": (wrong_before >= int(g["B"].get("wrong_before_min", 1))
              and reward_delta < 0 and bool(decision_ok)),
        "A": (max(acc_best, unseen_acc)
              >= float(g["A"].get("acc_a_min", 0.90))
              or gen_episodes >= int(g["A"].get("gen_episodes_min", 5))),
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
            "measured": {"rate": rate, "reproducible": bool(reproducible),
                         "acc_best": acc_best, "manual_chars": manual_chars,
                         "wrong_before": wrong_before,
                         "reward_delta": reward_delta,
                         "decision_ok": bool(decision_ok),
                         "unseen_acc": unseen_acc,
                         "gen_episodes": gen_episodes}}


def render_lines(v):
    """把判定渲染成报告行（实测 + 档位判定，进 output/report.txt）。"""
    m = v["measured"]
    lines = ["实测达成档位: %s ｜ 同种子二次运行逐键相等=%s ｜ 手写基线最高"
             "正确率 %.1f%% ｜ 说明书实测 %d 字 ｜ 漏洞修复成对证据：修复前假"
             "派检 %d 次且修复后总奖励 Δ%.1f（套利被剥掉）｜ 未见种子泛化实测"
             "正确率 %.1f%%（%d 集）｜ 决策记录完整=%s ｜ 代码规范率实测 %.1f%%"
             % (v["claimed"], "是" if m["reproducible"] else "否",
                m["acc_best"] * 100, m["manual_chars"], m["wrong_before"],
                m["reward_delta"], m["unseen_acc"] * 100, m["gen_episodes"],
                "是" if m["decision_ok"] else "否", m["rate"])]
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
