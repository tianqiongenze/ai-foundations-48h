# -*- coding: utf-8 -*-
# [教学脚手架] 下列导入/变量供 TODO 完成后使用，故标注 noqa 而非删除
"""M2-2 调参与工时账 starter（教学周4）：GridSearch穷举 vs 随机预算——限时体验"调参要计工时"

运行: /opt/anaconda3/bin/python W20_tuning_starter.py
说明: 机房无xgboost时自动回退随机森林（决策矩阵：环境也是选型维度）
"""
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


def make_data(n=600, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 6))
    y = rng.binomial(1, 0.4, n)
    X[y == 1, :3] += 1.3
    return X, y


def try_xgboost():
    try:
        from xgboost import XGBClassifier      # 工单主线：有xgboost用xgboost
        return XGBClassifier(eval_metric="logloss", random_state=0), "XGBoost"
    except ImportError:
        return RandomForestClassifier(random_state=0), "RandomForest(回退)"


def main():
    X, y = make_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=1)
    base, name = try_xgboost()
    grid = {"max_depth": [3, 4, 5], "n_estimators": [50, 100]}  # noqa: F841
    results = {}
    for mode, searcher in [("grid", GridSearchCV), ("random", RandomizedSearchCV)]:
        t0 = time.time()  # noqa: F841
        # TODO-W4-1: 用searcher(base, grid, cv=3[, n_iter=3, random_state=0])拟合Xtr,ytr
        #   记录 best_score_ 与 耗时 time.time()-t0 到 results[mode]
        pass
    if results:
        for m, r in results.items():
            print(f"{m:7s} best_cv={r['score']:.3f} 工时={r['sec']:.1f}s")
        print("目标：random工时<grid且cv差距≤0.02 —— 工时账写进工单")
    else:
        print("完成TODO-W4-1后可见穷举vs随机的精度-工时对比（模型:", name, "）")


if __name__ == "__main__":
    main()
