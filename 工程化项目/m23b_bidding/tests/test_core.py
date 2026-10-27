# -*- coding: utf-8 -*-
import numpy as np

from bidding_lines import core


def test_img_feats_three():
    f = core.img_feats(np.zeros((32, 32)))
    assert len(f) == 3


def test_seq_feats_one_value():
    assert len(core.seq_feats(np.arange(64, dtype=float))) == 1


def test_make_images_classes():
    X, y = core.make_images(60, 3)
    assert X.shape[1:] == (32, 32) and set(y) == {"正常", "气孔", "裂纹"}


def test_cv_between_0_and_1():
    X, y = core.make_images(90, 2)
    v = core.cv([core.img_feats(im) for im in X], y)
    assert 0.0 <= v <= 1.0
