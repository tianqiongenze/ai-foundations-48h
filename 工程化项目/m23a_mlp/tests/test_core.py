# -*- coding: utf-8 -*-
import numpy as np

from mlp_clf import core


def test_make_spectra_shape():
    X, y = core.make_spectra(200, 1)
    assert X.shape == (200, 30) and set(y) <= {0, 1}


def test_train_returns_acc():
    X, y = core.make_spectra(300, 2)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    m, acc = core.train(Xtr, ytr, Xte, yte)
    assert 0.0 <= acc <= 1.0 and len(m.loss_curve_) > 0


def test_diagnose_four_segments():
    _, dg = None, core.diagnose(list(np.linspace(1.0, 0.1, 40)))
    assert dg["segments"] == 4 and dg["improving"] and dg["means"][0] > dg["means"][3]


def test_diagnose_short_curve():
    dg = core.diagnose([0.5, 0.4, 0.3])
    assert dg["segments"] == 1
