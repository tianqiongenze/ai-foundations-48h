# -*- coding: utf-8 -*-
"""工单 M2-3a 四档验收判定（纯函数，档位是算出来的、不是自称的）。

门槛逐条取自《M2-3a工单-MLP神经网络.md》"四、验收标准"：
  及格(D)：模板可运行出指标；代码规范率≥70%
  良好(C)：准确率≥75%图完整
  优秀(B)：gap收窄且指标不降；决策记录完整
  优(A)  ：找到性价比最优结构
数字外置在 config/rules.yaml 的 tiers: 段（config.py 读出后传入本模块）。
本模块只比较 pipeline 实测值：sklearn 基线真实出的指标、结构搜索逐候选
实测的参数量/耗时/acc、torch 双臂逐 epoch 损失算出的 train-val gap、
matplotlib 落盘 PNG 的真实字节数、result.json 回读的决策记录、
pycodestyle 实测规范率。"指标不降"按工单字面执行（严格 ≥），不为凑档
放容差；无真 Dropout 的兜底臂不得声称 B 线。
"""

TIERS = [
    ("及格(D)", "模板可运行出指标；代码规范率≥70%"),
    ("良好(C)", "准确率≥75%图完整"),
    ("优秀(B)", "gap收窄且指标不降；决策记录完整"),
    ("优(A)", "找到性价比最优结构"),
]
KEYS = ("D", "C", "B", "A")
#: 代码判不了、必须人工答辩/签认的验收项（进报告"人工答辩项"段）
MANUAL = [
    "损失曲线判读口述：什么形状算过拟合、早停为何定在那一轮——答辩必答",
    "决策矩阵小组打分（可行性/达标概率/工时）：课堂 3 分钟人工环节",
    "企业元验收栏：值班师傅签认（功能可用/参数留痕/数据可信）",
]


def _best_arm(arms, gap_base, acc_base):
    """从实测臂里找"gap 收窄且指标不降（严格 ≥ 基线）"的真 Dropout 臂。"""
    ok = [a for a in arms if a.get("engine") == "torch"
          and float(a["gap"]) < float(gap_base)
          and float(a["acc"]) >= float(acc_base)]
    if not ok:
        return None
    return min(ok, key=lambda a: float(a["gap"]))


def verdict(gates, metrics_ok, acc_best, png_bytes, gap_base, acc_base,
            arms, ss_candidates, best_value_ok, decision_ok, style_rate,
            summary=""):
    """按实测值给四档判定；每档 = 功能门 + 规范率门槛（工单钉在 D，高档继承）。

    arms：Dropout 扫描臂的逐臂实测 [{engine, dropout, acc, gap}, ...]；
    png_bytes：output/learning_curve.png 的真实字节数（os.path.getsize）；
    decision_ok：result.json 落盘回读（结构表/双臂 gap/诊断结论留痕）。
    """
    style_min = float(gates["D"]["style_min"])
    style_ok = float(style_rate) >= style_min
    best_arm = _best_arm(arms, gap_base, acc_base)
    functional = {
        "D": bool(metrics_ok),
        "C": (float(acc_best) >= float(gates["C"]["acc_min"])
              and int(png_bytes) >= int(gates["C"]["png_min_bytes"])),
        "B": best_arm is not None and bool(decision_ok),
        "A": int(ss_candidates) >= int(
            gates["A"]["candidates_min"]) and bool(best_value_ok),
    }
    out = []
    prev = True
    for (name, need), key in zip(TIERS, KEYS):
        ok = prev and functional[key] and style_ok
        out.append({"tier": name, "gate": style_min, "need": need,
                    "functional": bool(functional[key]),
                    "style_ok": style_ok, "pass": bool(ok)})
        prev = ok
    claimed = next((t["tier"] for t in reversed(out) if t["pass"]), "未达 D 档")
    return {"tiers": out, "claimed": claimed, "summary": summary,
            "measured": {"metrics_ok": bool(metrics_ok),
                         "acc_best": float(acc_best),
                         "png_bytes": int(png_bytes),
                         "gap_base": float(gap_base),
                         "acc_base": float(acc_base),
                         "arms": [dict(a) for a in arms],
                         "best_arm_dropout": (best_arm["dropout"]
                                              if best_arm else None),
                         "gap_best": (float(best_arm["gap"])
                                      if best_arm else None),
                         "acc_best_arm": (float(best_arm["acc"])
                                          if best_arm else None),
                         "ss_candidates": int(ss_candidates),
                         "best_value_ok": bool(best_value_ok),
                         "decision_ok": bool(decision_ok),
                         "style_rate": float(style_rate)}}


def render_lines(v):
    """把判定渲染成报告行（进 output/report.txt 的 [验收档位判定（实测）] 段）。"""
    lines = ["实测达成档位: %s ｜ %s" % (v["claimed"], v["summary"])]
    for t in v["tiers"]:
        lines.append("  %-8s 规范率门槛≥%.0f%% [%s] ｜ 功能%s ｜ %s"
                     % (t["tier"], t["gate"],
                        "达标" if t["pass"] else "未达标",
                        "OK" if t["functional"] else "缺", t["need"]))
    return lines
