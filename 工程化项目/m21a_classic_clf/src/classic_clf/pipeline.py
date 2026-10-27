# -*- coding: utf-8 -*-
"""三态数据 → 双模型 → 混淆矩阵 → 漏报账。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    labels = ["正常", "轴承磨损", "叶轮堵塞"]
    X, y = core.make_data(300, seed=cfg.seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    out = {}
    for nm, m in [("LR", LogisticRegression(max_iter=500)),
                  ("DT", DecisionTreeClassifier(max_depth=6, random_state=0))]:
        out[nm] = core.evaluate(m, Xtr, ytr, Xte, yte, labels)
    gap = abs(out["LR"]["acc"] - out["DT"]["acc"])
    result = {"labels": labels, "n_train": len(Xtr), "n_test": len(Xte),
              "models": {k: {"acc": v["acc"], "missed": v["missed"], "cm": v["cm"]}
                         for k, v in out.items()}, "acc_gap": round(gap, 4)}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    lines = []
    for nm, r in out.items():
        lines.append("%s acc=%.3f 漏报=%s" % (nm, r["acc"], r["missed"]))
        for a in labels:
            lines.append("    %s→%s" % (a, {b: r["cm"][a][b] for b in labels}))
    sections = [("输入口径", ["样本 %d 条（三态）｜训练 %d / 测试 %d" % (len(X), len(Xtr), len(Xte))]),
                ("双模型 + 混淆矩阵", lines),
                ("结论与建议", ["双解 acc 差距 %.3f（目标 ≤0.10）；漏报（故障→正常）需在决策矩阵加权"
                 % gap])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
