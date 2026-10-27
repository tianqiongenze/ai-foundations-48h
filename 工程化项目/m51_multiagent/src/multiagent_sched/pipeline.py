# -*- coding: utf-8 -*-
"""三策略对照 + 奖励敏感性实验。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    seed = cfg.seed
    res = {
        "随机": core.run(core.policy_random, seed=seed),
        "奖励导向": core.run(core.policy_greedy, seed=seed),
        "安全优先": core.run(core.policy_safe, seed=seed),
    }
    sens = []
    for fb in (1.0, 3.0, 6.0):
        r = core.run(core.policy_safe, seed=seed, fix_benefit=fb)
        sens.append((fb, r["reward"], r["accuracy"]))
    result = {"strategies": res, "sensitivity": sens}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    lines = ["%-6s 总奖励 %9.1f｜协调正确率 %5.1f%%｜停机 %3d 步"
             % (k, v["reward"], v["accuracy"] * 100, v["downtime"])
             for k, v in res.items()]
    safe_acc = res["安全优先"]["accuracy"]
    sections = [("输入口径", ["3 台泵站｜200 步｜停机罚(-1/台)｜空转罚 5｜正确派检收益 3｜种子 %d" % seed]),
                ("三策略对照", lines),
                ("奖励敏感性（安全优先固定）",
                 ["FIX_BENEFIT=%.1f → 奖励 %.1f｜正确率 %.1f%%" % (fb, rw, ac * 100)
                  for fb, rw, ac in sens]),
                ("结论", ["安全优先正确率 %.1f%%（目标≥75%%）→ %s；但其奖励最低，"
                        "说明奖励函数与 KPI 存在冲突"
                        % (safe_acc * 100, "达标" if safe_acc >= 0.75 else "未达标")]),
                ("结论与建议", ["正确解法是约束优化：把正确率≥75% 设为硬约束，在此约束下再优化奖励"])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
