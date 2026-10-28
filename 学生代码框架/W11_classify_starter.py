# -*- coding: utf-8 -*-
"""M2-1a 传统分类双解 starter（教学周3）：逻辑回归 vs 决策树同台，手写混淆矩阵

运行: /opt/anaconda3/bin/python W11_classify_starter.py
数据: 合成泵站三态（正常/轴承磨损/叶轮堵塞），与工单M2-1a同源
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def make_three(n=300, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 4))
    # 注意（避坑）：不要写 y = np.array(["正常"]*n) —— 其 dtype 为 <U2，
    #   随后 y[...] = "轴承磨损" 会被 numpy **静默截断**成 "轴承"，标签静默出错。
    #   正确做法：一次性构造完整标签列表（dtype 自动取 <U4）。
    third = n // 3
    y = np.array(["正常"] * third + ["轴承磨损"] * third + ["叶轮堵塞"] * (n - 2 * third))
    X[y == "轴承磨损", 0] += 1.5          # 振动均值抬升
    X[y == "叶轮堵塞", 1] -= 1.4          # 压力波动下沉
    X[y == "叶轮堵塞", 2] += 1.0
    return X, y


def confusion(y_true, y_pred, labels):
    """手写混淆矩阵（工单要求：不许调库，练理解）"""
    # TODO-W3-1: 建 {label: {label: 0}} 二维字典，双重循环计数
    return {}


def run():
    X, y = make_three()
    labels = ["正常", "轴承磨损", "叶轮堵塞"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
    acc = {}
    for name, model in [("解A 逻辑回归", LogisticRegression(max_iter=500)),
                        ("解B 决策树", DecisionTreeClassifier(max_depth=6, random_state=0))]:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc[name] = (pred == yte).mean()
        cm = confusion(yte, pred, labels)
        if cm:
            print(f"{name} acc={acc[name]:.2f}")
            for r in labels:
                print("  ", r, {c: cm[r].get(c, 0) for c in labels})
        else:
            print(f"{name} acc={acc[name]:.2f}（完成TODO-W3-1可看混淆矩阵——漏报就藏在里面）")
    return acc


if __name__ == "__main__":
    acc = run()
    print("目标：双解 acc≥0.85 且差距≤0.10 —— 把对比写进工单决策矩阵")
