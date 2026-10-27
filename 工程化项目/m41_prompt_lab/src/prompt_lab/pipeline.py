# -*- coding: utf-8 -*-
"""三版 Prompt → 打分 → 提升幅度。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    ch = core.MockChannel()
    out, base = core.compare(ch)
    best = max(v["score"] for v in out.values())
    lift = (best - base) * 100
    result = {"question": core.QUESTION, "base": base, "results": out,
              "best_score": best, "lift_pct": round(lift, 1)}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    lines = []
    for tag, v in out.items():
        lines.append("%-4s 质量分=%.1f（较朴素 %+.0f%%）→ %s"
                     % (tag, v["score"], v["delta"] * 100, v["answer"][:30]))
    sections = [("输入口径", ["问题（仅给症状，不含诊断关键词）: %s" % core.QUESTION]),
                ("三版对比", lines),
                ("结论", ["最优 %.1f，较朴素 %+.0f%% → %s"
                        % (best, lift, "达标（≥+50%）" if lift >= 50 else "未达标")]),
                ("说明", ["本演示走离线规则通道；设 OPENAI_BASE_URL 或起 mock LLM 服务可切真机通道"]),
                ("结论与建议", ["角色给领域先验、示例给输出格式——两者叠加才让 LLM 照单办事"])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
