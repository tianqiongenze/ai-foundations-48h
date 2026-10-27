# -*- coding: utf-8 -*-
from tuning_cost import core


def test_make_data_shape():
    X, y = core.make_data(200, 1)
    assert X.shape == (200, 6) and len(y) == 200


def test_search_grid_returns_metrics():
    X, y = core.make_data(120, 2)
    r = core.search("grid", X[:90], y[:90], {"max_depth": [2, 3]})
    assert 0.0 <= r["cv"] <= 1.0 and r["sec"] > 0


def test_search_random_returns_metrics():
    X, y = core.make_data(120, 2)
    r = core.search("random", X[:90], y[:90], {"max_depth": [2, 3, 4]}, n_iter=2)
    assert 0.0 <= r["cv"] <= 1.0


def test_random_no_slower_than_grid_on_big_space():
    X, y = core.make_data(150, 2)
    grid = {"max_depth": [2, 3, 4, 5], "n_estimators": [20, 50, 100]}
    g = core.search("grid", X[:100], y[:100], grid)
    r = core.search("random", X[:100], y[:100], grid, n_iter=2)
    assert r["sec"] <= g["sec"] + 1e-6
