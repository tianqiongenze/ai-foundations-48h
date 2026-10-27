# -*- coding: utf-8 -*-
"""M2-3b 竞标二选一 starter（教学周6）：焊缝CNN vs 风机LSTM——先用轻量基线拿到竞标数据

竞标周玩法：真模型(教师GPU演示)之前，先用本starter跑出"两条路线的轻量基线对比表"，
带着数据上答辩台——竞标书要的是证据不是形容词。
基线做法（离线CPU、零torch）：
  图像线: 合成焊缝灰度图 → 块统计特征 → 随机森林（模拟"CNN看图"的下限）
  时序线: 合成风机振动 → 滑窗特征 → 随机森林（模拟"LSTM看序列"的下限）
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def make_images(n=200, seed=7):
    """合成32×32焊缝图：正常/气孔（局部亮点）/裂纹（细线）"""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i in range(n):
        img = rng.normal(0.5, 0.08, (32, 32))
        label = i % 3
        if label == 1:                      # 气孔：3×3亮点
            r0, c0 = rng.randint(4, 28, 2)
            img[r0:r0 + 3, c0:c0 + 3] += 0.5
        if label == 2:                      # 裂纹：一条暗线
            r0 = rng.randint(6, 26)
            img[r0:r0 + 1, 4:28] -= 0.35
        X.append(img)
        y.append(["正常", "气孔", "裂纹"][label])
    return np.array(X), np.array(y)


def img_feats(img):
    """图像线特征：亮块比例/暗线比例/最大亮值（CNN的先验雏形）"""
    # TODO-W6-1: 返回 [float((img>0.8).mean()), float((img<0.2).mean()), float(img.max())]
    return []


def make_series(n=200, seed=7):
    """合成风机振动序列：健康/轴承退化（幅值渐增）"""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i in range(n):
        t = np.linspace(0, 2, 64)
        deg = i % 2
        amp = 0.3 + 0.25 * deg * t / 2      # 退化类幅值随时间爬升
        x = amp * np.sin(2 * np.pi * 6 * t) + rng.randn(64) * 0.05
        X.append(x)
        y.append(["健康", "退化"][deg])
    return np.array(X), np.array(y)


def seq_feats(x):
    """时序线特征：前半/后半RMS之比（退化的斜率指纹，LSTM的先验雏形）"""
    # TODO-W6-2: 返回 [float(np.sqrt(np.mean(x[:32]**2)) / (np.sqrt(np.mean(x[32:]**2)) + 1e-9))]
    return []


def line_score(F, y, tag):
    if F and F[0]:
        s = cross_val_score(RandomForestClassifier(random_state=0), F, y, cv=3).mean()
        print(f"{tag} 轻量基线 cv={s:.2f}")
        return s
    print(f"{tag} 先完成TODO再跑（特征为空）")
    return None


if __name__ == "__main__":
    Xi, yi = make_images()
    Fi = [img_feats(im) for im in Xi]
    Xs, ys = make_series()
    Fs = [seq_feats(x) for x in Xs]
    a = line_score(Fi, yi, "图像线(CNN路线)")
    b = line_score(Fs, ys, "时序线(LSTM路线)")
    if a and b:
        pick = "CNN焊缝" if a >= b else "LSTM风机"
        print(f"竞标建议：轻量基线支持选 {pick} —— 但答辩要说明基线≠上限，真模型预期增益在哪")
