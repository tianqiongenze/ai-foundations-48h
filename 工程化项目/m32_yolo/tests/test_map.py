# -*- coding: utf-8 -*-
"""mAP/检测/测速/ONNX 通道测试——任一能力（mAP、扫描、标注、固定口径FPS）被删除时必然失败。"""
import numpy as np

from yolo_detect import core, onnx_channel, pipeline


def _scene_with_block(x, y, w, size=64):
    rng = np.random.RandomState(0)
    img = rng.normal(0.4, 0.05, (size, size))
    img[y:y + w, x:x + w] += 0.4
    return img, (x, y, w, w)


def test_make_dataset_has_positives_and_negatives():
    scenes = core.make_dataset(n_scenes=4, n_neg=2, seed=3)
    assert len(scenes) == 6
    negs = [g for _, g in scenes if not g]
    poss = [g for _, g in scenes if g]
    assert len(negs) == 2 and len(poss) == 4
    for im, gs in scenes:
        for (x, y, w, h) in gs:
            assert (0 <= x and x + w <= im.shape[1]
                    and 0 <= y and y + h <= im.shape[0])


def test_refine_box_recovers_true_extent():
    # 缺陷在 (31,42) 8×8：窗口检测只给 (32,40) 粗框，精修应回到真值附近
    img, gt = _scene_with_block(31, 42, 8)
    thr = core.refine_thr(img)
    rough = (32, 40, 8, 8)
    fine = core.refine_box(img, rough, thr)
    assert core.iou(fine, gt) > core.iou(rough, gt)
    assert core.iou(fine, gt) >= 0.75


def test_detect_all_iou_beats_raw_window():
    img, gt = _scene_with_block(31, 42, 8)
    dets = core.detect_all(img, win=8, thresh=0.55)
    assert dets, "应至少检出一框"
    best = max(core.iou(b, gt) for b, _ in dets)
    raw = core.iou(core.detect(img)[0], gt)
    assert best > raw  # 精修+全候选通道定位必须优于 argmax 单框


def test_nms_keeps_higher_conf_and_suppresses_overlap():
    dets = [((0, 0, 8, 8), 0.9), ((1, 0, 8, 8), 0.5), ((40, 40, 8, 8), 0.7)]
    kept = core.nms(dets, iou_thr=0.5)
    assert len(kept) == 2 and kept[0][1] == 0.9


def test_average_precision_perfect_and_degenerate():
    assert core.average_precision([(1.0, 1), (0.9, 1)], 2) == 1.0
    assert core.average_precision([(1.0, 0)], 2) == 0.0
    assert core.average_precision([(0.9, 1)], 0) is None  # 无GT不编数


def test_average_precision_interpolated_value():
    # 手工可验证的经典例：AP = 0.5*1 + 0.5*0.6667 = 0.8333
    ap = core.average_precision([(0.9, 1), (0.8, 0), (0.7, 1)], 2)
    assert abs(ap - 5.0 / 6.0) < 1e-6


def test_match_scenes_counts_tp_fp():
    scenes = [(np.zeros((8, 8)), [(0, 0, 4, 4)]), (np.zeros((8, 8)), [])]
    dets = [[((0, 0, 4, 4), 0.9), ((5, 5, 2, 2), 0.6)], []]
    ms, n_gt = core.match_scenes(scenes, dets, iou_thr=0.5)
    assert n_gt == 1 and sorted(ms) == [(0.6, 0), (0.9, 1)]


def test_sweep_map_grid_complete():
    scenes = core.make_dataset(n_scenes=3, n_neg=1, seed=5)
    grid = core.sweep_map(scenes, (0.5, 0.7), (0.5, 0.9))
    assert set(grid) == {(0.5, 0.5), (0.5, 0.9), (0.7, 0.5), (0.7, 0.9)}
    assert any(v is not None and v > 0 for v in grid.values())


def test_bench_fixed_protocol_returns_fps_and_ms():
    imgs = [np.zeros((16, 16))]
    f, ms = core.bench(lambda im: core.detect_all(im, win=4, thresh=0.55),
                       imgs, warmup=2, n=5)
    assert f > 0 and ms > 0


def test_annotate_writes_png(tmp_path):
    img, gt = _scene_with_block(10, 10, 8)
    p = core.annotate(img, [(gt, 0.8)], [gt], str(tmp_path / "a.png"))
    import os
    assert os.path.exists(p) and os.path.getsize(p) > 500


def test_onnx_probe_reports_honestly():
    scenes = core.make_dataset(n_scenes=2, n_neg=0, seed=4)
    info = onnx_channel.probe([scenes[0][0]], warmup=0, n=1)
    assert info["available"] or "兜底" in info["note"]
    if info["available"]:
        assert 0.0 <= info["max_conf"] <= 1.0
        assert info["boxes_over_thr"] >= 0


def test_pipeline_report_contains_map_sweep_fps_and_onnx(tmp_path):
    from yolo_detect.config import Settings
    cfg = Settings(seed=7)
    pipeline.run(cfg, str(tmp_path))
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    for key in ("mAP 评估", "conf × IoU 阈值扫描", "固定口径 FPS", "预热",
                "ONNX YOLOv8n 通道实测", "标注图", "代码规范率",
                "验收档位判定（实测）"):
        assert key in text
    import os
    assert os.path.exists(str(tmp_path / "annotated_scene1.png"))
    rj = (tmp_path / "result.json")
    import json
    data = json.loads(rj.read_text(encoding="utf-8"))
    assert data["ap@0.5"] is None or 0.0 <= data["ap@0.5"] <= 1.0
