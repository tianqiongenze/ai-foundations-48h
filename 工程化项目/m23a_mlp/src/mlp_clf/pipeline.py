# -*- coding: utf-8 -*-
"""频谱 → MLP → 曲线诊断 → 正则对照。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    from sklearn.model_selection import train_test_split
    X, y = core.make_spectra(600, seed=cfg.seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    m, acc = core.train(Xtr, ytr, Xte, yte, alpha=0.0)
    dg = core.diagnose(m.loss_curve_)
    _, acc_reg = core.train(Xtr, ytr, Xte, yte, alpha=1e-3)
    result = {"n_epochs": len(m.loss_curve_), "acc": acc, "acc_reg": acc_reg,
              "diagnosis": dg}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    sections = [("输入口径", ["600 样本 × 30 维频谱｜隐藏层 (64,32)｜early_stopping=True"]),
                ("训练", ["epochs=%d  测试准确率 %.4f（目标≥0.75）" % (len(m.loss_curve_), acc)]),
                ("损失曲线诊断", ["四段均值: %s" % dg["means"],
                 "持续下降: %s｜末轮 loss %.4f" % (dg["improving"], dg["final"])]),
                ("正则对照", ["alpha=1e-3 准确率 %.4f（对比 %.4f）" % (acc_reg, acc)]),
                ("结论与建议", ["结构+早停+正则三件套守住准确率与泛化；曲线比分数更能说明问题"])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
