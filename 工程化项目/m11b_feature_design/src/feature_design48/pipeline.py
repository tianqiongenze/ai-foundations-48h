# -*- coding: utf-8 -*-
"""多窗口数据集 → 特征重要性 → Top-N 结论。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    n = int(cfg.raw.get("n_windows", 40))
    X, y = core.build_dataset(n, seed=cfg.seed)
    ranked, clf = core.rank(X, y)
    result = {"n_windows": n * 2, "n_features": len(core.FEAT_NAMES),
              "ranked": ranked, "top1": ranked[0][0],
              "train_acc": round(float(clf.score(X, y)), 4)}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    sections = [("输入口径", ["窗口 %d 个（每类 %d）｜特征 %d 维｜树 300 棵"
                % (n * 2, n, len(core.FEAT_NAMES))]),
                ("特征重要性", ["%d. %s = %.3f" % (k + 1, nm, v)
                 for k, (nm, v) in enumerate(ranked)]),
                ("结论与建议", ["Top1 = %s（16Hz 二倍频指纹）；特征设计决定模型上限，"
                 "比换模型更能提升效果" % ranked[0][0]])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
