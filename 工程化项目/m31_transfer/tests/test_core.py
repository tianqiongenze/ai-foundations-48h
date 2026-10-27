# -*- coding: utf-8 -*-
import numpy as np

from transfer_learning import core


def test_make_world_shapes():
    big, small, test, ys, yt = core.make_world(7)
    assert big.shape[1] == 192 and len(small) == 16 and len(test) == 200
    assert set(np.unique(ys)) <= {0, 1}


def test_frozen_extractor_dims():
    big, small, test, ys, yt = core.make_world(7)
    ext = core.FrozenExtractor(big, dim=8)
    assert ext.transform(small).shape == (16, 8)


def test_train_head_predicts_binary():
    big, small, test, ys, yt = core.make_world(7)
    ext = core.FrozenExtractor(big, dim=8)
    _, pred = core.train_head(ext.transform(small), ys, ext.transform(test))
    assert set(np.unique(pred)) <= {0, 1}


def test_transfer_beats_raw():
    big, small, test, ys, yt = core.make_world(7)
    ext = core.FrozenExtractor(big, dim=8)
    _, raw = core.train_head(small, ys, test)
    _, tf = core.train_head(ext.transform(small), ys, ext.transform(test))
    assert (tf == yt).mean() >= (raw == yt).mean()
