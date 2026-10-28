# -*- coding: utf-8 -*-
"""把 evidence/<slug>/ 的真跑捕获文本渲染成终端窗口截图（1280x560 PNG）。

shot_run.png     = python run.py 的完整终端输出（末段优先）
shot_test.png    = pytest -q 输出
shot_report.png  = output/report.txt 的「验收档位判定（实测）」段
文本逐字取自捕获文件，只在排版时折行，不改写任何数字。
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terminal_png import render, ANSI  # noqa: E402

EV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")
W, H = 1280, 560


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [ANSI.sub("", ln) for ln in fh.read().splitlines()]


def main():
    for d in sorted(glob.glob(os.path.join(EV, "*/"))):
        slug = os.path.basename(d.rstrip("/"))
        jp = os.path.join(d, "evidence.json")
        if not os.path.exists(jp):          # 采集尚未完成的项目
            continue
        ev = json.load(open(jp, encoding="utf-8"))
        run_lines = read(os.path.join(d, "run.txt"))
        rc = ev.get("run_rc")
        if run_lines:
            tail = "本次真跑，退出码 %s" % rc if rc == 0 else \
                   "本次真跑，退出码 %s（收尾期进程异常，输出与报告已完整产出）" % rc
            render(run_lines, os.path.join(d, "shot_run.png"),
                   "python run.py — %s（%s）" % (slug, tail), W, H)
        test_lines = read(os.path.join(d, "pytest.txt"))
        if test_lines:
            state = "全绿" if ev.get("test_rc") == 0 else \
                    "退出码 %s" % ev.get("test_rc")
            render(test_lines, os.path.join(d, "shot_test.png"),
                   "python -m pytest -q — %s（%s）" % (slug, state), W, H)
        render(ev.get("report_tier_seg") or [],
               os.path.join(d, "shot_report.png"),
               "output/report.txt — 验收档位判定（实测）段", W, H)
        have = [f for f in ("shot_run.png", "shot_test.png", "shot_report.png")
                if os.path.exists(os.path.join(d, f))]
        kb = sum(os.path.getsize(os.path.join(d, f)) for f in have) // 1024
        print("%-24s shots=%d (%dKB) run_rc=%s test_rc=%s" % (
            slug, len(have), kb, ev.get("run_rc"), ev.get("test_rc")))


if __name__ == "__main__":
    main()
