# -*- coding: utf-8 -*-
# [教学脚手架] 下列导入/变量供 TODO 完成后使用，故标注 noqa 而非删除
"""W5 MLP神经网络 —— 学生starter（M2-3a）
数据: 合成振动频谱二分类(正常/磨损)；目标: 准确率≥75%+损失曲线诊断报告
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # noqa: F401
from sklearn.preprocessing import StandardScaler


def make_spectra(n=600, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (n, 30))
    y = rng.binomial(1, .5, n)
    X[y == 1, 2:8] += rng.uniform(.8, 1.6)      # 磨损: 中频段整体抬升
    return X, y


def main():
    X, y = make_spectra()
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=42)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    # TODO-1: 训练MLP(hidden_layer_sizes=(64,32), early_stopping=True)
    # TODO-2: 打印测试准确率(目标≥0.75)
    # TODO-3: 取 mlp.loss_curve_ 画损失曲线, 标注过拟合区间, 存 diagnosis.png
    # TODO-4(优秀): Dropout替代实验——sklearn无Dropout, 用alpha=1e-3正则对比
    print("（先完成TODO-W5再跑——现在只是占位提示，不影响烟测）")


if __name__ == "__main__":
    main()
