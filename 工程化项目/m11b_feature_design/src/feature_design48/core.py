# -*- coding: utf-8 -*-
"""特征设计：13 维时域 + 4 维频域（纯函数）。"""
import numpy as np


def make_signal(kind, n=400, seed=7, noise=0.4):
    """合成 8Hz 基频信号；磨损类叠加 16Hz 二倍频（工业 SNR）。"""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4, n)
    x = 0.3 * np.sin(2 * np.pi * 8 * t) + rng.randn(n) * noise
    if kind == "wear":
        x = x + 0.25 * np.sin(2 * np.pi * 16 * t)
    return t, x


def feats_time(x):
    """时域三件套：RMS / 峰峰值 / 峭度。"""
    x = np.asarray(x, float)
    return [float(np.sqrt(np.mean(x ** 2))), float(np.ptp(x)),
            float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-12))]


def feats_freq(x, fs=100.0):
    """频域双特征：8Hz / 16Hz 幅值。"""
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return [float(spec[np.argmin(np.abs(f - 8))]),
            float(spec[np.argmin(np.abs(f - 16))])]


FEAT_NAMES = ["rms", "ptp", "kurt", "a8", "a16"]


def build_dataset(n_win=40, seed=7, noise=0.4):
    """多随机实现 → 稳定重要性。"""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for lab, kind in ((0, "normal"), (1, "wear")):
        for _ in range(n_win):
            _, x = make_signal(kind, seed=int(rng.randint(0, 100000)), noise=noise)
            X.append(feats_time(x) + feats_freq(x))
            y.append(lab)
    return np.array(X, float), np.array(y)


def rank(X, y):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=0, n_estimators=300).fit(X, y)
    order = np.argsort(clf.feature_importances_)[::-1]
    return [(FEAT_NAMES[i], round(float(clf.feature_importances_[i]), 4))
            for i in order], clf
