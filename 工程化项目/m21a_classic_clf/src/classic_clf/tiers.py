# -*- coding: utf-8 -*-
"""工单 M2-1a 四档验收判定（纯函数，档位是算出来的、不是自称的）。

门槛逐条取自《M2-1a工单-传统分类：逻辑回归+决策树.md》"四、验收标准"：
  及格(D)：基线accuracy输出；代码规范率≥70%
  良好(C)：F1≥0.70口径正确
  优秀(B)：给出阈值方案与理由；决策记录完整
  优(A)  ：F1≥0.85或方案论证充分
数字外置在 config/rules.yaml 的 tiers: 段（config.py 读出后传入本模块），
本模块只比较 pipeline 传进来的实测值（真训练真评估跑出来的 acc/F1/阈值
曲线点数/落盘回读的决策记录/pycodestyle 实测规范率），不产生新数据。
高档包含低档：C 达标必须 D 也达标（档位是"档"，不是四个独立开关）；
"方案论证充分"这类代码判不了的分支进 MANUAL，不冒充机器达标。
"""

TIERS = [
    ("及格(D)", "基线accuracy输出；代码规范率≥70%"),
    ("良好(C)", "F1≥0.70口径正确"),
    ("优秀(B)", "给出阈值方案与理由；决策记录完整"),
    ("优(A)", "F1≥0.85或方案论证充分"),
]
KEYS = ("D", "C", "B", "A")
#: 代码判不了、必须人工答辩/签认的验收项（进报告"人工答辩项"段）
MANUAL = [
    "优(A) 备选路「方案论证充分」：口头答辩 + 评审评委签认，代码不代判",
    "决策矩阵小组打分（可行性/达标概率/工时）：课堂 3 分钟人工环节",
    "企业元验收栏：值班师傅签认（功能可用/参数留痕/数据可信）",
]


def _acc_out_ok(lr_acc, dt_acc):
    """D 档功能门：基线 accuracy 必须真实输出（两模型都拿到 [0,1] 内的实测数）。"""
    return all(a is not None and 0.0 <= float(a) <= 1.0
               for a in (lr_acc, dt_acc))


def verdict(gates, lr_acc, dt_acc, macro_f1, caliber_ok,
            sweep_min_points, sweep_total_points, decision_ok, style_rate,
            summary=""):
    """按实测值给四档判定；每档 = 功能门 + 规范率门槛（工单钉在 D，高档继承）。

    gates: config.Settings.tiers（rules.yaml tiers: 段读出，非硬编码）；
    caliber_ok: C 档"口径正确"实测——报告打印过 scoring=macro 的来源与 OvR
    pos_label（由 core 的真实调用参数回读，不是口头声明）；
    decision_ok: B 档"决策记录完整"实测——result.json 落盘后回读校验。
    """
    style_min = float(gates["D"]["style_min"])
    style_ok = float(style_rate) >= style_min
    functional = {
        "D": _acc_out_ok(lr_acc, dt_acc),
        "C": bool(caliber_ok) and (
            float(macro_f1) >= float(gates["C"]["f1_min"])),
        "B": int(sweep_min_points) >= int(
            gates["B"]["sweep_points_min"]) and bool(decision_ok),
        "A": float(macro_f1) >= float(gates["A"]["f1_min"]),
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
            "measured": {"lr_acc": lr_acc, "dt_acc": dt_acc,
                         "macro_f1": macro_f1, "caliber_ok": bool(caliber_ok),
                         "sweep_min_points": int(sweep_min_points),
                         "sweep_total_points": int(sweep_total_points),
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
