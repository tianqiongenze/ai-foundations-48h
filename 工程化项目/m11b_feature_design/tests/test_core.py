# -*- coding: utf-8 -*-
import numpy as np

from feature_design48 import core


def test_feats_time_three_values():
    f = core.feats_time(np.arange(30, dtype=float))
    assert len(f) == 3


def test_feats_freq_two_values():
    t, x = core.make_signal("normal", 200, 3)
    assert len(core.feats_freq(x)) == 2


def test_make_signal_wear_has_higher_a16():
    _, xn = core.make_signal("normal", 400, 5)
    _, xw = core.make_signal("wear", 400, 5)
    assert core.feats_freq(xw)[1] > core.feats_freq(xn)[1]


def test_build_dataset_shape():
    X, y = core.build_dataset(10, 1)
    assert X.shape == (20, 5) and set(y) == {0, 1}


def test_rank_orders_by_importance():
    X, y = core.build_dataset(10, 1)
    ranked, _ = core.rank(X, y)
    vals = [v for _, v in ranked]
    assert vals == sorted(vals, reverse=True) and len(ranked) == 5
