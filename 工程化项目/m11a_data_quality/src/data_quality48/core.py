# -*- coding: utf-8 -*-
"""数据清洗与质量评估：双解互验（纯函数）。"""
import numpy as np


def make_dirty(n=300, seed=7):
    import pandas as pd
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "ts": pd.date_range("2026-03-01", periods=n, freq="h"),
        "vib": rng.normal(4.0, 0.5, n).round(3),
        "temp": rng.normal(55, 3, n).round(1)})
    df.loc[rng.choice(n, 15, replace=False), "vib"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "temp"] = 135.0
    df.loc[rng.choice(n, 5, replace=False), "vib"] = -1.0
    return df


def clean_py(df, temp_max=120.0, vib_min=0.0, vib_max=15.0):
    """解A：纯 Python 逐行判断。"""
    out = []
    for r in df.to_dict("records"):
        ok = (r["vib"] == r["vib"] and vib_min <= r["vib"] <= vib_max
              and r["temp"] <= temp_max)
        if ok:
            out.append(r)
    return out


def clean_pd(df, temp_max=120.0, vib_min=0.0, vib_max=15.0):
    """解B：Pandas 布尔掩码链。"""
    return df[(df["vib"].notna()) & (df["vib"].between(vib_min, vib_max))
              & (df["temp"] <= temp_max)]


def quality(n_rows, n_bad, temp_max=120.0):
    """质量分 = 完整率 × 合理率（清洗后口径）。"""
    if n_rows + n_bad == 0:
        return 0.0
    return round((n_rows / (n_rows + n_bad)) ** 2, 4)
