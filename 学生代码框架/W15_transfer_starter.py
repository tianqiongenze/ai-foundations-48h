# -*- coding: utf-8 -*-
# [教学脚手架] 下列导入/变量供 TODO 完成后使用，故标注 noqa 而非删除
"""M3-1 迁移学习 starter（教学周7）：冻结预训练特征提取器+只训分类头——零torch理解ResNet精髓

思想实验（离线CPU版）：
  "预训练" = 用2000条无标注数据学出紧凑特征(PCA)——学完冻结，不随任务更新
  迁移 = 冻结它，只用16条标注样本训顶端逻辑回归头（小样本逆袭）
  对照 = 不用预训练特征，16条样本直接在192维(多为噪声维)训头——维度灾难大概率崩
真机版( torchvision ResNet18冻结实验 )见教师GPU演示——概念在本文件。
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def make_world(seed=7, n_dim=192):
    """合成工业特征：8种工况原型仅前8个测点携带信息，其余为环境噪声维"""
    rng = np.random.RandomState(seed)
    centers = rng.normal(0, 3, (8, n_dim))
    centers[:, 8:] = 0
    big = np.vstack([c + rng.normal(0, 1, (250, n_dim)) for c in centers])   # 海量"无标注"
    small = np.vstack([c + rng.normal(0, 1, (2, n_dim)) for c in centers])   # 仅16条有标注
    test = np.vstack([c + rng.normal(0, 1, (25, n_dim)) for c in centers])   # 200条测试
    label = lambda X: (X[:, 0] > 0).astype(int)      # noqa: E731  # 真实规律藏在第0维
    return big, small, test, label(small), label(test)


class FrozenExtractor:
    """'预训练特征提取器'：在海量数据上一次学成(PCA)，之后冻结不再更新"""

    def __init__(self, big, dim=8):
        self.pca = PCA(n_components=dim, random_state=0).fit(big)  # 预训练=在海量数据上学结构

    def transform(self, X):
        # TODO-W7-1: 返回冻结特征 self.pca.transform(X)（只前传、不更新权重）
        return X


def main():
    big, small, test, y_small, y_test = make_world()
    ext = FrozenExtractor(big)  # noqa: F841
    # 对照组：16条样本直接在192维原始空间训头
    s_raw = LogisticRegression(max_iter=500).fit(small, y_small).score(test, y_test)
    # 迁移组：冻结特征 → 训头
    # TODO-W7-2: Xf = ext.transform(small) 训头，测试集评估，得 s_transfer
    s_transfer = None
    print(f"对照(16条直连192维训头): 测试={s_raw:.2f}")
    if s_transfer is not None:
        print(f"迁移(冻结特征+训头):     测试={s_transfer:.2f}")
        print("目标：迁移组≥对照组 —— 预训练特征是维度灾难的解药（写进冻结/微调取舍实验）")
    else:
        print("完成TODO-W7-1/2看对比——两个TODO各一行")


if __name__ == "__main__":
    main()
