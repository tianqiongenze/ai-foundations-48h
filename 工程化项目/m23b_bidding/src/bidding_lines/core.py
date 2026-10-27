# -*- coding: utf-8 -*-
"""竞标二选一：图像线（CNN）vs 时序线（LSTM）轻量基线。"""
import numpy as np


def make_images(n=180, seed=7, size=32):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i in range(n):
        img = rng.normal(0.5, 0.08, (size, size))
        label = i % 3
        if label == 1:
            r0, c0 = rng.randint(4, size - 4, 2)
            img[r0:r0 + 3, c0:c0 + 3] += 0.5
        if label == 2:
            r0 = rng.randint(6, size - 6)
            img[r0:r0 + 1, 4:size - 4] -= 0.35
        X.append(img)
        y.append(["正常", "气孔", "裂纹"][label])
    return np.array(X), np.array(y)


def img_feats(img):
    """块统计：亮块比例 / 暗线比例 / 最大亮值。"""
    return [float((img > 0.8).mean()), float((img < 0.2).mean()), float(img.max())]


def make_series(n=180, seed=7):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i in range(n):
        t = np.linspace(0, 2, 64)
        amp = 0.3 + 0.25 * (i % 2) * t / 2
        X.append(amp * np.sin(2 * np.pi * 6 * t) + rng.randn(64) * 0.05)
        y.append(["健康", "退化"][i % 2])
    return np.array(X), np.array(y)


def seq_feats(x):
    """时序线：前/后半 RMS 之比（退化斜率指纹）。"""
    return [float(np.sqrt(np.mean(x[:32] ** 2)) / (np.sqrt(np.mean(x[32:] ** 2)) + 1e-9))]


def cv(F, y, folds=3):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    s = cross_val_score(RandomForestClassifier(random_state=0), F, y, cv=folds)
    return round(float(s.mean()), 4)
