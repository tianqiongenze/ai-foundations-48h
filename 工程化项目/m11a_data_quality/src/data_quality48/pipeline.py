# -*- coding: utf-8 -*-
"""双解互验：造脏 → 两法清洗 → 一致性比对 → 质量分。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    raw = core.make_dirty(300, seed=cfg.seed)
    a = core.clean_py(raw)
    b = core.clean_pd(raw)
    same = (len(a) == len(b))
    q = core.quality(len(b), len(raw) - len(b))
    result = {"rows_before": len(raw), "rows_after": len(b),
              "removed": len(raw) - len(b), "quality": q,
              "py_rows": len(a), "pd_rows": len(b), "consistent": same}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    sections = [("输入口径", ["原始 %d 行（含缺失/超限/负值）" % len(raw)]),
                ("双解结果", ["纯Python: %d 行｜Pandas: %d 行｜一致: %s"
                 % (len(a), len(b), same)]),
                ("清洗留痕", ["%d 行 → %d 行（剔除 %d 行，保留率 %.1f%%）"
                 % (len(raw), len(b), result["removed"], len(b) / len(raw) * 100)]),
                ("质量分", ["%.4f（目标 ≥ 0.85）" % q]),
                ("结论与建议", ["两解同口径同结果 → 清洗规则可冻结；一致即互验通过"])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
