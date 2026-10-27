# -*- coding: utf-8 -*-
"""造世界 → 冻结提取 → 两组训练 → 增益结论。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    big, small, test, ys, yt = core.make_world(seed=cfg.seed)
    ext = core.FrozenExtractor(big, dim=8)
    _, raw_pred = core.train_head(small, ys, test)
    raw_acc = float((raw_pred == yt).mean())
    _, tf_pred = core.train_head(ext.transform(small), ys, ext.transform(test))
    tf_acc = float((tf_pred == yt).mean())
    evr = float(ext.pca.explained_variance_ratio_.sum())
    gain = tf_acc - raw_acc
    result = {"dim_raw": small.shape[1], "dim_frozen": ext.n,
              "raw_acc": round(raw_acc, 4), "transfer_acc": round(tf_acc, 4),
              "gain": round(gain, 4), "evr": round(evr, 4)}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    sections = [("输入口径", ["海量无标注 %d 条｜少量标注 %d 条｜测试 %d 条"
                % (len(big), len(small), len(test))]),
                ("对照组（16条直连192维）", ["测试准确率 %.4f" % raw_acc]),
                ("迁移组（冻结8维+训头）", ["测试准确率 %.4f（解释方差 %.3f）" % (tf_acc, evr)]),
                ("增益", ["%+.4f → %s" % (gain, "迁移胜出" if gain > 0 else "本轮未见增益")]),
                ("结论与建议", ["真实规律仅藏在少数测点维；冻结+小样本头是 ResNet 微调思想的最小可运行模型"])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
