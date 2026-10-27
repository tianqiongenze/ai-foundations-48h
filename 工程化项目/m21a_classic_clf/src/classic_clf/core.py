# -*- coding: utf-8 -*-
"""传统分类：造数、双模型、手写混淆矩阵与漏报统计（纯函数）。"""
import numpy as np


def make_data(n=300, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 4))
    third = n // 3
    y = np.array(["正常"] * third + ["轴承磨损"] * third + ["叶轮堵塞"] * (n - 2 * third),
                 dtype=object)
    X[y == "轴承磨损", 0] += 1.5
    X[y == "叶轮堵塞", 1] -= 1.4
    X[y == "叶轮堵塞", 2] += 1.0
    return X, y


def confusion(y_true, y_pred, labels):
    cm = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def missed(cm, labels):
    """每类漏报数（真实为 a 但预测成别的）。"""
    return {a: sum(cm[a][b] for b in labels if b != a) for a in labels}


def evaluate(model, Xtr, ytr, Xte, yte, labels):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    cm = confusion(yte, pred, labels)
    return {"acc": round(float((pred == yte).mean()), 4), "cm": cm,
            "missed": missed(cm, labels)}
