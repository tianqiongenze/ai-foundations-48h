# -*- coding: utf-8 -*-
"""造图 → 检测 → win 取舍实验。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    img, gt = core.make_scene(64, seed=cfg.seed)
    box, conf = core.detect(img)
    v = core.iou(box, gt) if box else 0.0
    rows, best = [], (None, -1)
    for win in (4, 8, 16):
        b, _ = core.detect(img, win=win)
        vv = core.iou(b, gt) if b else 0.0
        ff = core.fps(lambda im, w=win: core.detect(im, win=w), img, n=60)
        rows.append((win, vv, ff))
        if vv > best[1]:
            best = (win, vv)
    result = {"detected_box": box, "conf": round(conf, 4), "iou": round(v, 4),
              "gt": list(gt), "bench": [{"win": w, "iou": round(a, 3), "fps": b}
                                        for w, a, b in rows],
              "best_win": best[0]}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    sections = [("输入口径", ["64×64 产线图｜真值框 %s｜阈值 0.5" % (gt,)]),
                ("检测结果", ["框=%s 缺陷分=%.3f IoU=%.2f（目标≥0.5）"
                 % (box, conf, v)]),
                ("win 取舍表", ["win=%-3d IoU=%.2f  FPS=%.0f" % (w, a, b) for w, a, b in rows]),
                ("结论与建议", ["实测最优 win=%d；窗口过小信噪比低易误检、过大定位粗化——"
                 "这就是 YOLO 锚框/步长调参的同一问题" % best[0]])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
