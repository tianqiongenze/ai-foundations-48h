# -*- coding: utf-8 -*-
from tuning_cost import core


def test_make_data_shape():
    X, y = core.make_data(200, 1)
    assert X.shape == (200, 6) and len(y) == 200


def test_search_grid_returns_f1_and_candidates():
    X, y = core.make_data(120, 2)
    r = core.search("grid", X[:90], y[:90], {"max_depth": [2, 3]})
    assert 0.0 <= r["f1"] <= 1.0 and r["sec"] > 0
    # 工时账凭证：逐候选参数/F1/墙钟秒，组合数必须等于网格大小
    assert len(r["candidates"]) == 2
    for c in r["candidates"]:
        assert set(c) == {"params", "f1", "sec"} and c["sec"] >= 0


def test_search_random_returns_f1():
    X, y = core.make_data(120, 2)
    r = core.search("random", X[:90], y[:90],
                    {"max_depth": [2, 3, 4]}, n_iter=2)
    assert 0.0 <= r["f1"] <= 1.0


def test_random_no_slower_than_grid_on_big_space():
    X, y = core.make_data(150, 2)
    grid = {"max_depth": [2, 3, 4, 5], "n_estimators": [20, 50, 100]}
    g = core.search("grid", X[:100], y[:100], grid)
    r = core.search("random", X[:100], y[:100], grid, n_iter=2)
    assert r["sec"] <= g["sec"] + 1e-6


def test_search_optimizes_f1_not_default_accuracy():
    """删掉 scoring='f1' 此测试即红——工单良好线口径就是 F1。"""
    import inspect
    src = inspect.getsource(core.search)
    assert src.count('scoring="f1"') >= 2   # grid 与 random 两路都要带


def test_learning_curve_f1_rows_and_knee():
    X, y = core.make_data(300, 2)
    lc = core.learning_curve_f1(X, y, {"max_depth": 3})
    assert lc["scoring"] == "f1" and len(lc["rows"]) == 4
    ns = [r["n_train"] for r in lc["rows"]]
    assert ns == sorted(ns)
    for r in lc["rows"]:
        assert 0.0 <= r["val_f1"] <= 1.0
        assert abs(r["gap"] - (r["train_f1"] - r["val_f1"])) < 1e-3
    assert lc["knee_n"] is None or lc["knee_n"] in ns


def test_boost_arm_engine_or_honest_fallback():
    """xgboost 装不上时必须走 GradientBoostingClassifier 兜底并在 note 里明说。"""
    X, y = core.make_data(300, 2)
    b = core.boost_arm(X[:220], y[:220], X[220:], y[220:])
    assert 0.0 <= b["f1"] <= 1.0 and b["best_iter"] >= 1 and b["sec"] > 0
    try:
        import xgboost  # noqa: F401
        assert b["engine"] == "xgboost" and b["fallback"] is False
    except ImportError:
        assert b["fallback"] is True
        assert "离线兜底" in b["note"] and "GradientBoosting" in b["engine"]


def test_worklog_conclusion_ranks_by_f1():
    cands = [{"params": {"a": 1}, "f1": 0.80, "sec": 1.0},
             {"params": {"a": 2}, "f1": 0.85, "sec": 2.5},
             {"params": {"a": 3}, "f1": 0.70, "sec": 0.5}]
    w = core.worklog_conclusion(cands)
    assert w["best"]["params"] == {"a": 2} and w["total_sec"] == 4.0
    assert "边际收益" in w["message"]


def test_tier_verdict_honest_below_line():
    """分级判定铁律：数字不过线/不可复现时档位必须掉下来（tiers.verdict 同源）。"""
    from tuning_cost import tiers as T

    gates = {"D": {"style_min": 70.0},
             "C": {"f1_min": 0.72, "limit_sec": 7200},
             "B": {"curve_points_min": 4, "knee_gain_max": 0.01},
             "A": {"f1_min": 0.88, "fast_gap_max": 0.02}}
    base = dict(gates=gates, reproducible=True, random_f1=0.83,
                boost_f1=0.85, run_sec=5.0, curve_points=4,
                knee_found=True, decision_ok=True, style_rate=95.0)
    low = T.verdict(grid_f1=0.60, **base)
    by_key = {t["tier"][-2]: t for t in low["tiers"]}
    assert not by_key["C"]["pass"] and low["claimed"] == "及格(D)"
    drift = T.verdict(**dict(base, grid_f1=0.90, reproducible=False))
    assert drift["claimed"] == "未达 D 档"
    high = T.verdict(grid_f1=0.90, **base)
    assert high["claimed"] == "优(A)"
