# -*- coding: utf-8 -*-
"""工单 M2-2 四档验收判定（纯函数，档位是算出来的、不是自称的）。

门槛逐条取自《M2-2工单-XGBoost参数调优.md》"四、验收标准"：
  及格(D)：基线可复现；代码规范率≥70%
  良好(C)：F1≥0.72限时内
  优秀(B)：曲线完整拐点论证；决策记录完整
  优(A)  ：F1≥0.88或快速方案论证
数字外置在 config/rules.yaml 的 tiers: 段（config.py 读出后传入本模块）。
本模块只比较 pipeline 实测值：同种子两次重跑是否逐位一致、perf_counter
墙钟对 2h 限时、学习曲线逐档实测 F1 与程序化拐点、result.json 回读的
决策记录、pycodestyle 实测规范率。高档包含低档；"拐点论证/快速方案论证"
的口头部分进 MANUAL，不冒充机器达标。
"""

TIERS = [
    ("及格(D)", "基线可复现；代码规范率≥70%"),
    ("良好(C)", "F1≥0.72限时内"),
    ("优秀(B)", "曲线完整拐点论证；决策记录完整"),
    ("优(A)", "F1≥0.88或快速方案论证"),
]
KEYS = ("D", "C", "B", "A")
#: 代码判不了、必须人工答辩/签认的验收项（进报告"人工答辩项"段）
MANUAL = [
    "优秀(B)「拐点论证」口头部分：为什么在拐点收手——答辩必答",
    "优(A)「快速方案论证」答辩：随机预算近似穷举的工时账解释",
    "决策矩阵小组打分（可行性/达标概率/工时）：课堂 3 分钟人工环节",
    "企业元验收栏：值班师傅签认（功能可用/参数留痕/数据可信）",
]


def verdict(gates, reproducible, grid_f1, random_f1, boost_f1, run_sec,
            curve_points, knee_found, decision_ok, style_rate, summary=""):
    """按实测值给四档判定；每档 = 功能门 + 规范率门槛（工单钉在 D，高档继承）。

    run_sec：pipeline 的 perf_counter 墙钟实测（对时限时用）；
    knee_found：拐点=相邻 val F1 增益跌破 gates B knee_gain_max 的点；
    decision_ok：result.json 落盘回读（网格/随机/曲线/工时账/引擎留痕）；
    A 档机器判据：F1≥0.88，或快速方案成立（预算搜索与穷举差≤容差
    且提升树臂过 C 线）——口径随 gates 写入报告。
    """
    style_min = float(gates["D"]["style_min"])
    style_ok = float(style_rate) >= style_min
    f1_c = float(gates["C"]["f1_min"])
    fast_ok = (abs(float(grid_f1) - float(random_f1))
               <= float(gates["A"]["fast_gap_max"])
               and float(boost_f1) >= f1_c)
    functional = {
        "D": bool(reproducible),
        "C": float(grid_f1) >= f1_c and float(run_sec) <= float(
            gates["C"]["limit_sec"]),
        "B": (int(curve_points) >= int(gates["B"]["curve_points_min"])
              and bool(knee_found) and bool(decision_ok)),
        "A": float(grid_f1) >= float(gates["A"]["f1_min"]) or fast_ok,
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
            "measured": {"reproducible": bool(reproducible),
                         "grid_f1": float(grid_f1),
                         "random_f1": float(random_f1),
                         "boost_f1": float(boost_f1), "fast_ok": bool(fast_ok),
                         "run_sec": float(run_sec),
                         "curve_points": int(curve_points),
                         "knee_found": bool(knee_found),
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
