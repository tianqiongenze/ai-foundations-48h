# -*- coding: utf-8 -*-
"""两路线基线 → CV 对比 → 竞标建议。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    Xi, yi = core.make_images(180, seed=cfg.seed)
    Xs, ys = core.make_series(180, seed=cfg.seed)
    a = core.cv([core.img_feats(im) for im in Xi], yi)
    b = core.cv([core.seq_feats(x) for x in Xs], ys)
    pick = "CNN 焊缝线" if a >= b else "LSTM 风机线"
    result = {"image_line_cv": a, "series_line_cv": b, "pick": pick,
              "gap": round(abs(a - b), 4)}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    sections = [("输入口径", ["图像线 180 张×3类｜时序线 180 条×2类｜3 折交叉验证"]),
                ("两路线基线", ["图像线(CNN) CV = %.4f" % a, "时序线(LSTM) CV = %.4f" % b]),
                ("竞标建议", ["轻量基线支持选「%s」（差 %.4f）" % (pick, abs(a - b))]),
                ("结论与建议", ["基线≠上限——真模型的增益需用数据量/算力/标注成本三项证据支撑"])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
