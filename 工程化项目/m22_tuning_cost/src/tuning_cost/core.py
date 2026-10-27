# -*- coding: utf-8 -*-
"""调参与工时账：穷举 vs 预算搜索（纯函数）。"""
import time

import numpy as np


def make_data(n=400, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 6))
    y = rng.binomial(1, 0.4, n)
    X[y == 1, :3] += 1.3
    return X, y


def search(kind, Xtr, ytr, grid, n_iter=3, seed=0):
    """两种搜索：grid（穷举）/ random（预算）。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    base = RandomForestClassifier(random_state=0)
    t0 = time.time()
    if kind == "grid":
        res = GridSearchCV(base, grid, cv=3).fit(Xtr, ytr)
    else:
        res = RandomizedSearchCV(base, grid, n_iter=n_iter, random_state=seed,
                                 cv=3).fit(Xtr, ytr)
    return {"best": dict(res.best_params_),
            "cv": round(float(res.best_score_), 4),
            "sec": round(time.time() - t0, 3)}
