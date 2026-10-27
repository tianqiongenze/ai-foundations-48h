# -*- coding: utf-8 -*-

from yolo_detect import core


def test_make_scene_returns_gt():
    img, gt = core.make_scene(64, 7)
    assert img.shape == (64, 64) and gt == (30, 40, 8, 8)


def test_detect_returns_box_and_conf():
    img, _ = core.make_scene(64, 7)
    box, conf = core.detect(img)
    assert box is not None and conf > 0.5


def test_iou_identical_and_disjoint():
    a = (10, 10, 8, 8)
    assert core.iou(a, a) == 1.0
    assert core.iou((0, 0, 5, 5), (100, 100, 5, 5)) == 0.0


def test_score_map_shape():
    img, _ = core.make_scene(64, 7)
    assert core.score_map(img, 8).shape == (8, 8)


def test_fps_positive():
    img, _ = core.make_scene(64, 7)
    assert core.fps(lambda im: core.detect(im), img, n=10) > 0
