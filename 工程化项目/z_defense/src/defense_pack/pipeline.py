# -*- coding: utf-8 -*-
"""档案袋自检 → 提纲生成 → 评委三问。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    fw = str(cfg.raw.get("fw_dir", "../学生代码框架"))
    if not os.path.isabs(fw):
        fw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", fw)
    rows = core.check(fw)
    lines, n_ok = core.outline(rows)
    missing = [r[0] for r in rows if not r[1]]
    result = {"layers": [{"layer": r[0], "ok": r[1], "talk": r[2]} for r in rows],
              "ready": n_ok, "total": len(rows), "missing": missing}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    sections = [("输入口径", ["档案袋目录：%s" % os.path.relpath(fw, os.getcwd())]),
                ("自检结果", ["%s %s" % ("OK" if r[1] else "--", r[0]) for r in rows]),
                ("答辩提纲", lines),
                ("评委三问预演", ["%s：%s" % (t, q) for t, q in core.THREE_Q]),
                ("结论与建议", ["证据链 %d/%d；%s"
                 % (n_ok, len(rows),
                    "可上台" if not missing else "补齐：" + "、".join(missing))])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
