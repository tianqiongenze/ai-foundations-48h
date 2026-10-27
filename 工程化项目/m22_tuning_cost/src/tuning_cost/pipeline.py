# -*- coding: utf-8 -*-
"""造数 → 两种搜索 → 工时账。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    from sklearn.model_selection import train_test_split
    X, y = core.make_data(400, seed=cfg.seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    grid = {"max_depth": [3, 4, 5], "n_estimators": [50, 100]}
    g = core.search("grid", Xtr, ytr, grid)
    r = core.search("random", Xtr, ytr, grid, n_iter=3)
    saved = round(g["sec"] - r["sec"], 3)
    gap = abs(g["cv"] - r["cv"])
    result = {"grid": g, "random": r, "time_saved_sec": saved, "cv_gap": round(gap, 4),
              "n_combos": len(grid["max_depth"]) * len(grid["n_estimators"])}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    sections = [("输入口径", ["搜索空间 %d 组（max_depth×n_estimators）" % result["n_combos"]]),
                ("GridSearch（穷举）", ["cv=%.4f  耗时 %.3fs  最优 %s" % (g["cv"], g["sec"], g["best"])]),
                ("RandomSearch（预算）", ["cv=%.4f  耗时 %.3fs  最优 %s" % (r["cv"], r["sec"], r["best"])]),
                ("工时账", ["节省 %.3fs（%.0f%%）｜CV 代价 %.4f（目标 ≤0.02）"
                 % (saved, saved * 100 / max(g["sec"], 1e-9), gap)]),
                ("结论与建议", ["小空间穷举更稳；大空间用预算搜索——工时账写进工单决策矩阵"])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
