# -*- coding: utf-8 -*-
"""迁移学习概念版：冻结预训练特征 + 只训分类头。"""
import numpy as np


def make_world(seed=7, n_dim=192):
    """8 种工况原型，仅前 8 个测点携带信息，其余为环境噪声维。"""
    rng = np.random.RandomState(seed)
    centers = rng.normal(0, 3, (8, n_dim))
    centers[:, 8:] = 0
    big = np.vstack([c + rng.normal(0, 1, (250, n_dim)) for c in centers])
    small = np.vstack([c + rng.normal(0, 1, (2, n_dim)) for c in centers])
    test = np.vstack([c + rng.normal(0, 1, (25, n_dim)) for c in centers])

    def label(X):
        return (X[:, 0] > 0).astype(int)

    return big, small, test, label(small), label(test)


class FrozenExtractor:
    """'预训练特征提取器'：海量数据上一次性学成，之后冻结。"""

    def __init__(self, big, dim=8):
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=dim, random_state=0).fit(big)
        self.n = dim

    def transform(self, X):
        return self.pca.transform(X)


def train_head(X, y, Xte):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=500).fit(X, y)
    return m, m.predict(Xte)
