# -*- coding: utf-8 -*-
"""M1-1a 数据清洗 starter（大一友好版：先手写循环，再用Pandas对照——多解法启蒙）

运行: /opt/anaconda3/bin/python W07_clean_starter.py
数据: 自动合成泵站日志（缺测/超限/重复三类脏数据）
"""
import numpy as np
import pandas as pd


def make_dirty(n=300, seed=7):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "ts": pd.date_range("2026-03-01", periods=n, freq="h"),
        "vib": rng.normal(4.0, 0.5, n).round(3),
        "temp": rng.normal(55, 3, n).round(1)})
    df.loc[rng.choice(n, 15), "vib"] = np.nan          # 缺测
    df.loc[rng.choice(n, 6), "temp"] = 135.0            # 超限(传感器毛刺)
    df.loc[rng.choice(n, 5), "vib"] = -1.0              # 非法负值
    return df


def clean_a(df):
    """解A 纯Python思路逐行处理（第一学期刚学的循环/条件）"""
    rows = df.to_dict("records")
    out = []
    for r in rows:
        # TODO-W1-1: 跳过 vib 缺测/负值、temp>120 的行（三行if）
        out.append(r)
    return pd.DataFrame(out)


def clean_b(df):
    """解B Pandas布尔掩码链（一行一个条件，可读性强）"""
    # TODO-W1-2: 用 df[ (条件1) & (条件2) & ... ] 实现同口径清洗
    return df


def quality(df):
    """质量评分：完整率×合理率（工单口径）"""
    # TODO-W1-3: 完整率=非缺测比例；合理率=vib∈[0,15]且temp∈[0,120]比例
    return 0.0


if __name__ == "__main__":
    raw = make_dirty()
    for tag, fn in [("解A纯Py", clean_a), ("解B Pandas", clean_b)]:
        c = fn(raw)
        print(f"{tag}: 清洗后{len(c)}行 质量分={quality(c):.3f}（目标≥0.85）")
    print("两解行数应一致（同口径）——这就是《多解法手册》说的'双解互验'")
