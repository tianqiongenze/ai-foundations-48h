# -*- coding: utf-8 -*-
"""MLP：造数、训练、损失曲线诊断与正则对照（纯函数）。"""
import numpy as np


def make_spectra(n=600, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 30))
    y = rng.binomial(1, .5, n)
    X[y == 1, 2:8] += rng.uniform(.8, 1.6)
    return X, y


def train(Xtr, ytr, Xte, yte, alpha=0.0, seed=0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    m = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=alpha,
                      early_stopping=True, random_state=seed, max_iter=300).fit(Xtr, ytr)
    return m, round(float(m.score(Xte, yte)), 4)


def diagnose(loss_curve):
    """把损失曲线分成 4 段，判断是否持续下降。"""
    n = len(loss_curve)
    if n < 4:
        return {"segments": 1, "improving": False}
    idx = [(0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)]
    vals = [round(float(np.mean(loss_curve[a:b])), 4) for a, b in idx]
    return {"segments": 4, "means": vals,
            "improving": all(vals[i] > vals[i + 1] for i in range(3)),
            "final": round(float(loss_curve[-1]), 4)}
