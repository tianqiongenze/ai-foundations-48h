# -*- coding: utf-8 -*-
"""数据清洗与质量评估：三清单体检 + 缺失填充 + IQR/3σ 异常 + 质量分（纯函数）。

对齐工单 M1-1a 的四档验收（三、任务要求 + 四、验收标准）：
  D 温度列缺失率降为 0 且**行数不变**（禁止用 dropna 删行"作弊"）；
  C 缺失处理率 ≥ 95% + 三清单齐全（缺失清单 / 异常清单 / 重复清单）；
  B 质量分 = 完整率 / 准确率(合规率) / 一致率(唯一率) 三项加权 ≥ 85，
    且每列的"缺失机制判断 → 策略 → 理由"决策记录完整；
  A 预测填充优于均值填充——同一留出实验给 MAE/RMSE 数字，不靠嘴说。

口径依据（工单知识卡）：
  B01 三清单 = isna().sum() / 越量程与 IQR / duplicated().sum()；
  B02 缺失四策略 = 删 / 数值插补 / 类别兜底 / 时序插值——成片缺失（断电）
     不能假装插出趋势，标记"待复核"更诚实；
  B03 异常两把尺 = 3σ（正态）与 IQR（偏态稳健），再加物理量程硬闸；
  B05 质量评分 = 完整性 + 合规性 + 唯一性 加权，权重按业务定并写明理由。
双解互验（纯 Python 逐行 vs Pandas 掩码）保留为规则一致性证据。
"""
import numpy as np

# 泵站日志字段（合成但按现场口径造）：温度/振动/电流/压力 + 设备编号 + 时间戳
TIME_COL = "ts"
NUMERIC_COLS = ["temp", "vib", "current", "pressure"]
# 物理量程硬闸（传感器 datasheet 口径；越界即"毛刺/断线"，不是真工况）
PHYS_RANGE = {"temp": (-20.0, 200.0), "vib": (0.0, 30.0),
              "current": (0.0, 120.0), "pressure": (0.0, 2.5)}
DEVICE_OK = ("P-01", "P-02", "P-03")
BLACKOUT = 12          # 断电段长度：成片缺失，插值会造假日趋势
OUTLIER_SIGMA = 3.0    # 尺一：3σ
IQR_K = 1.5            # 尺二：IQR 系数（箱线图须）


def make_dirty(n=300, seed=7, n_dup=8, blackout=BLACKOUT):
    """合成"脏"泵站日志：真实温度规律 + 断线缺失 + 毛刺 + 重复上报 + 断电段。

    temp 含 24 小时日周期与负载耦合（vib/current），使 A 层的"预测填充"
    有真实结构可学——否则任何填充法都只是在填噪声，实验无意义。
    """
    import pandas as pd
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2026-03-01 00:00", periods=n, freq="h")
    hour = np.array([t.hour for t in ts], float)
    device = np.array([DEVICE_OK[i % 3] for i in range(n)], dtype=object)
    vib = rng.normal(4.0, 0.5, n)
    current = rng.normal(45.0, 6.0, n) + 3.0 * (vib - 4.0)
    temp = 55.0 + 4.0 * np.sin(2 * np.pi * hour / 24.0) + 1.2 * (vib - 4.0) \
        + rng.normal(0, 0.8, n)
    pressure = 1.2 + 0.02 * (current - 45.0) + rng.normal(0, 0.05, n)
    df = pd.DataFrame({
        "ts": ts, "device": device,
        "temp": np.round(temp, 2), "vib": np.round(vib, 3),
        "current": np.round(current, 2), "pressure": np.round(pressure, 3)})

    # ① 断线丢点：温度散点缺失 25 处、振动 18 处、电流 6 处
    df.loc[rng.choice(n, 25, replace=False), "temp"] = np.nan
    df.loc[rng.choice(n, 18, replace=False), "vib"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "current"] = np.nan
    # ② 成片缺失（采集柜断电 blackout 小时）——压力通道整列缺失率 35%
    b0 = int(rng.randint(40, n - blackout - 40))
    df.loc[b0:b0 + blackout - 1, "pressure"] = np.nan
    rest = np.setdiff1d(np.arange(n), np.arange(b0, b0 + blackout))
    df.loc[rng.choice(rest, int(0.35 * n) - blackout, replace=False), "pressure"] = np.nan
    # ③ 毛刺/超量程：温度 135℃、振动 -1.0（负值不可能）、压力 9.9
    df.loc[rng.choice(n, 6, replace=False), "temp"] = 135.0
    df.loc[rng.choice(n, 5, replace=False), "vib"] = -1.0
    df.loc[rng.choice(n, 4, replace=False), "pressure"] = 9.9
    # ④ 重复上报：把 n_dup 行整行复制成它前面某行的副本（PLC 重发双录）
    src = rng.choice(n - 20, n_dup, replace=False)
    for s in src:
        df.iloc[s + 20] = df.iloc[s]
    df.attrs["blackout_slice"] = (b0, b0 + blackout)
    df.attrs["dup_src"] = sorted(int(s) + 20 for s in src)
    return df


# ---------------------------------------------------------------- 三清单
def missing_profile(df, cols=NUMERIC_COLS, topk=3, blackout=BLACKOUT):
    """缺失清单：每列空值数 / 缺失率 / 缺失机制判断（散点 vs 成片）。

    topk：缺失率最高的前 topk 列在报告中标为"重点关注列"（config/rules.yaml）。
    """
    n = len(df)
    rows = []
    for c in cols:
        miss = df[c].isna()
        k = int(miss.sum())
        rate = k / max(n, 1) * 100.0
        # 连续缺失最长段——区分"随机散点丢点"与"断电成片"
        vals = miss.to_numpy(dtype=bool) if hasattr(miss, "to_numpy") else np.asarray(miss)
        run = best = 0
        for v in vals:
            run = run + 1 if v else 0
            best = max(best, run)
        mech = "成片缺失(断电/断线)" if best >= blackout else "随机散点缺失"
        rows.append({"column": c, "n_missing": k, "missing_rate_pct": round(rate, 2),
                     "max_consecutive": int(best), "mechanism": mech})
    rows.sort(key=lambda r: (-r["n_missing"], r["column"]))
    for i, r in enumerate(rows):
        r["watch"] = bool(i < topk and r["n_missing"] > 0)
    return rows


def anomaly_profile(df, cols=NUMERIC_COLS):
    """异常清单：物理量程 / 3σ / IQR 三把尺各扫一遍（工单 B03）。"""
    rows = []
    for c in cols:
        v = df[c].astype(float)
        ok = v.dropna()
        if len(ok) < 4:
            continue
        lo, hi = PHYS_RANGE[c]
        mean, std = float(ok.mean()), float(ok.std())
        q1, q3 = float(ok.quantile(0.25)), float(ok.quantile(0.75))
        iqr = q3 - q1
        iqr_lo, iqr_hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        phys = ((v < lo) | (v > hi)) & v.notna()
        sigma = ((v - mean).abs() > OUTLIER_SIGMA * std) & v.notna()
        iqr_m = ((v < iqr_lo) | (v > iqr_hi)) & v.notna()
        union = (phys | sigma | iqr_m)
        rows.append({"column": c, "n_phys": int(phys.sum()), "n_sigma3": int(sigma.sum()),
                     "n_iqr": int(iqr_m.sum()), "n_union": int(union.sum()),
                     "phys_range": "[%g, %g]" % (lo, hi),
                     "iqr_bounds": "[%.3f, %.3f]" % (iqr_lo, iqr_hi),
                     "sigma_bounds": "[%.3f, %.3f]" % (mean - OUTLIER_SIGMA * std,
                                                       mean + OUTLIER_SIGMA * std),
                     "ruler": "IQR（右偏/有毛刺时更稳健）"})
    return rows


def duplicate_profile(df):
    """重复清单：duplicated() 命中行（工单 B01 误区：只查缺失不查重复）。"""
    dup_mask = df.duplicated(keep="first")
    idx = [int(i) for i in np.where(np.asarray(dup_mask))[0]]
    rows = [{"row": i,
             "ts": str(df.iloc[i]["ts"]),
             "device": str(df.iloc[i]["device"]),
             "temp": df.iloc[i]["temp"], "vib": df.iloc[i]["vib"]}
            for i in idx]
    return rows, idx


# ---------------------------------------------------------------- 双解互验
def flags_py(df, cols=NUMERIC_COLS):
    """解 A：纯 Python 逐行判断异常（可读、可断点）。"""
    out = []
    recs = df.to_dict("records")
    for i, r in enumerate(recs):
        bad = False
        for c in cols:
            v = r[c]
            if v != v:                      # NaN
                continue
            lo, hi = PHYS_RANGE[c]
            if v < lo or v > hi:
                bad = True
        out.append(bad)
    return out


def flags_pd(df, cols=NUMERIC_COLS):
    """解 B：Pandas 布尔掩码链（与解 A 同规则的向量化写法）。"""
    import pandas as pd
    m = pd.Series(False, index=df.index)
    for c in cols:
        lo, hi = PHYS_RANGE[c]
        m = m | (((df[c] < lo) | (df[c] > hi)) & df[c].notna())
    return [bool(x) for x in m]


def agree(a, b):
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


# ---------------------------------------------------------------- 清洗
def _ols(X, y):
    """最小二乘闭式解（numpy，零新增依赖）——A 层"预测填充"的模型。"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _design(df, idx):
    """预测温度用的设计矩阵：日内周期(sin/cos) + 振动 + 电流 + 设备编号。

    只允许用"该时刻已可得"的量（工单 B06 防泄漏），不含事后录的列。
    """
    import pandas as pd
    sub = df.loc[idx]
    hour = pd.to_datetime(sub["ts"]).dt.hour.to_numpy(float)
    vib = sub["vib"].to_numpy(float)
    cur = sub["current"].to_numpy(float)
    dev = np.column_stack([
        [1.0 if d == "P-01" else 0.0 for d in sub["device"]],
        [1.0 if d == "P-02" else 0.0 for d in sub["device"]]])
    ones = np.ones((len(idx), 1))
    return np.column_stack([ones, np.sin(2 * np.pi * hour / 24.0),
                            np.cos(2 * np.pi * hour / 24.0), vib, cur, dev])


def impute(df, seed=7):
    """缺失填充（行数不变）：温度=时序插值 / 振动=中位数 / 电流=前向填充 /
    压力=整列策略（中位数 + 缺失标记，成片断电段标"待复核"）。

    返回 (clean_df, log, stats)：
      log   = 留痕台账（step/before/after/reason），工单 B06 双轨留痕
      stats = 缺失单元格总数、已处理数、待复核数、原始观测掩码
    """
    import pandas as pd
    out = df.copy()
    n = len(out)
    log = []
    total_cells = 0
    missing_cells = 0
    handled = 0
    review = []

    orig_missing = {c: out[c].isna().copy() for c in NUMERIC_COLS}
    for c in NUMERIC_COLS:
        total_cells += n
        missing_cells += int(orig_missing[c].sum())

    # 温度：散点缺失走线性时序插值（工单 B02：interpolate 最合理）
    t = out["temp"].astype(float)
    t_interp = t.interpolate(method="linear", limit_area="inside").ffill().bfill()
    out["temp"] = np.round(t_interp.astype(float), 3)
    handled += int(orig_missing["temp"].sum())
    log.append({"step": "temp: 时序插值+两端兜底", "before_na": int(orig_missing["temp"].sum()),
                "after_na": int(out["temp"].isna().sum()),
                "reason": "散点断线（最长段 < 断电阈值），按相邻趋势补最贴近物理"})

    # 振动：中位数（偏态稳健）；电流：前向填充（缓慢变量）
    out["vib"] = out["vib"].astype(float).fillna(float(out["vib"].median()))
    handled += int(orig_missing["vib"].sum())
    log.append({"step": "vib: 中位数填充", "before_na": int(orig_missing["vib"].sum()),
                "after_na": int(out["vib"].isna().sum()),
                "reason": "振动右偏有长尾，中位数比均值稳；不删行以守住"
                          "『行数不变』口径"})
    out["current"] = out["current"].astype(float).ffill().bfill()
    handled += int(orig_missing["current"].sum())
    log.append({"step": "current: 前向填充", "before_na": int(orig_missing["current"].sum()),
                "after_na": int(out["current"].isna().sum()),
                "reason": "负载电流变化缓慢，ffill 比全局均值更贴现场"})

    # 压力：缺失率 35% > 30% → 整列处理策略（不删行，标记 + 中位数）
    pr = orig_missing["pressure"]
    out["pressure_missing_flag"] = pr.astype(int)
    out["temp_missing_flag"] = orig_missing["temp"].astype(int)   # 缺失即特征
    out["pressure"] = out["pressure"].astype(float).fillna(float(out["pressure"].median()))
    handled += int(pr.sum())
    bl = df.attrs.get("blackout_slice")
    if bl:
        for i in range(bl[0], min(bl[1], n)):
            review.append(i)
    log.append({"step": "pressure: 整列策略（标记+中位数）",
                "before_na": int(pr.sum()), "after_na": int(out["pressure"].isna().sum()),
                "reason": "缺失率 %.1f%%>30%%，其中 %d 小时成片断电——插值会造出"
                          "假趋势，故不假装恢复，加缺失标记列并计入待复核"
                          % (pr.sum() / n * 100.0, len(review))})

    stats = {"n_rows": n, "cells": total_cells, "missing_cells": missing_cells,
             "handled": handled, "n_review": len(review),
             "orig_missing": {c: int(orig_missing[c].sum()) for c in NUMERIC_COLS},
             "temp_missing_after": int(out["temp"].isna().sum()),
             "review_rows": review}
    return out, log, stats


def correct_anomalies(df, stats, cols=NUMERIC_COLS):
    """异常处置：越量程毛刺按趋势重插（不删行）；断电待复核段保留并标记。"""
    out = df.copy()
    fixed = {}
    for c in cols:
        lo, hi = PHYS_RANGE[c]
        v = out[c].astype(float)
        bad = (v < lo) | (v > hi)
        if not bool(bad.any()):
            fixed[c] = 0
            continue
        vv = v.mask(bad)
        vv = vv.interpolate(method="linear", limit_area="inside").ffill().bfill()
        out[c] = np.round(vv.astype(float), 3)
        fixed[c] = int(bad.sum())
    out["review_flag"] = 0
    for i in stats["review_rows"]:
        out.loc[i, "review_flag"] = 1
    return out, fixed


def value_valid_mask(df, col):
    """可信观测掩码：原始有值 且 通过三把尺（物理量程 / 3σ / IQR）。

    注入的毛刺（如 135℃）不是真工况，不能当填充实验的"考卷"——否则
    均值法与预测法都在比谁更能拟合毛刺，实验失真。
    """
    v = df[col].astype(float)
    lo, hi = PHYS_RANGE[col]
    ok = v.notna() & v.between(lo, hi)
    obs = v[ok]
    if len(obs) < 4:
        return ok
    mean, std = float(obs.mean()), float(obs.std())
    q1, q3 = float(obs.quantile(0.25)), float(obs.quantile(0.75))
    iqr = q3 - q1
    stat = ((v - mean).abs() <= OUTLIER_SIGMA * std) & \
           (v.between(q1 - IQR_K * iqr, q3 + IQR_K * iqr))
    return ok & stat


def remove_duplicates(df, drop_rows=None):
    """重复处置：删除"重复清单"里命中的那些行（保留首条）。

    传 drop_rows（来自 duplicate_profile）可让"清单 ↔ 处置"一一对应，
    审计可追溯；不传则现场重算 duplicated()。
    本阶段行数必然下降，与"缺失阶段行数不变"分开记台账：
    D 层看缺失阶段（300→300），唯一率看本阶段。
    """
    if drop_rows is None:
        d = df.drop_duplicates(keep="first")
        return d, int(len(df) - len(d))
    keep = [i for i in df.index if i not in set(drop_rows)]
    return df.loc[keep], len(df) - len(keep)


# ---------------------------------------------------------------- 质量分
def rates(n_cells, n_missing, n_bad, n_rows, n_dup):
    """三率：完整率 / 合规率(准确率) / 唯一率(一致率)，均为百分数。"""
    comp = (1.0 - n_missing / max(n_cells, 1)) * 100.0
    acc = (1.0 - n_bad / max(n_cells, 1)) * 100.0
    uniq = (1.0 - n_dup / max(n_rows, 1)) * 100.0
    return round(comp, 2), round(acc, 2), round(uniq, 2)


def quality(comp, acc, uniq, w=(0.4, 0.4, 0.2)):
    """质量分 = w1×完整率 + w2×准确率 + w3×一致率（工单 B05）。

    权重理由：缺陷/异常场景下"数据对不对"(合规)与"全不全"(完整)同等要命，
    重复上报只影响样本权重、危害最小 → 0.4/0.4/0.2，权重写进配置不硬编码。
    """
    w = np.asarray(w, float)
    if np.any(w < 0) or abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError("质量分权重必须非负且和为 1，收到 %s" % (w.tolist(),))
    return round(float(w[0] * comp + w[1] * acc + w[2] * uniq), 2)


# ---------------------------------------------------------------- A 层实验
def imputation_experiment(df_raw, df_feat, target="temp", seed=7, n_repeat=5,
                          hold_frac=0.25):
    """同一留出实验对比两种温度填充：全局均值 vs 预测填充（OLS 回归）。

    df_raw  提供"真值"（仅取通过三把尺的可信观测，见 value_valid_mask）；
    df_feat 提供填充特征（已无缺失的 vib/current 等，行序与 df_raw 对齐）。
    被留出的格子在两法中都不可见；特征只用该时刻已可得的列（工单 B06 防泄漏）。
    返回两法 MAE/RMSE 与改善幅度——A 层要求"预测填充优于均值"有数字。
    """
    import pandas as pd
    t_raw = pd.to_numeric(df_raw[target], errors="coerce")
    idx = np.where(value_valid_mask(df_raw, target).to_numpy())[0]
    if len(idx) < 40:
        raise ValueError("可信观测太少（%d），填充实验不具统计意义" % len(idx))
    Xall = _design(df_feat, df_feat.index)
    rows, errs_mean, errs_ols = [], [], []
    for rep in range(n_repeat):
        rng = np.random.RandomState(seed * 100 + rep)
        hold = rng.choice(idx, int(round(hold_frac * len(idx))), replace=False)
        keep = np.setdiff1d(idx, hold)
        y_true = t_raw.loc[hold].to_numpy(float)
        # 基线：均值填充（仅用保留集已知温度，两法信息量相同）
        mean_fill = float(np.mean(t_raw.loc[keep].to_numpy(float)))
        e_mean = y_true - mean_fill
        # 预测填充：在保留集上拟合 OLS，再预测留出格
        beta = _ols(Xall[keep], t_raw.loc[keep].to_numpy(float))
        e_ols = y_true - Xall[hold] @ beta
        rows.append({"repeat": rep + 1, "n_holdout": int(len(hold)),
                     "mae_mean": round(float(np.mean(np.abs(e_mean))), 4),
                     "mae_pred": round(float(np.mean(np.abs(e_ols))), 4),
                     "rmse_mean": round(float(np.sqrt(np.mean(e_mean ** 2))), 4),
                     "rmse_pred": round(float(np.sqrt(np.mean(e_ols ** 2))), 4)})
        errs_mean.append(e_mean)
        errs_ols.append(e_ols)
    mae_mean = round(float(np.mean([r["mae_mean"] for r in rows])), 4)
    mae_pred = round(float(np.mean([r["mae_pred"] for r in rows])), 4)
    rmse_mean = round(float(np.mean([r["rmse_mean"] for r in rows])), 4)
    rmse_pred = round(float(np.mean([r["rmse_pred"] for r in rows])), 4)
    lift = round((mae_mean - mae_pred) / max(mae_mean, 1e-9) * 100.0, 2)
    # 配对比较（免正态假设的符号检验），论证"不是运气好"
    am, ao = np.abs(np.concatenate(errs_mean)), np.abs(np.concatenate(errs_ols))
    wins = int((ao < am).sum())
    n_tot = int(len(am))
    return {"repeats": n_repeat, "hold_frac": hold_frac, "per_repeat": rows,
            "n_valid_obs": int(len(idx)),
            "mae_mean": mae_mean, "mae_pred": mae_pred,
            "rmse_mean": rmse_mean, "rmse_pred": rmse_pred,
            "mae_lift_pct": lift, "pred_win_rate": round(wins / max(n_tot, 1), 4),
            "n_compared": n_tot,
            "better": bool(mae_pred < mae_mean)}
