# -*- coding: utf-8 -*-
"""M1-1a core 单元测试：三清单 / 缺失填充 / 异常识别 / 质量分 / 填充实验。"""
import numpy as np
import pytest

from data_quality48 import core


# ---------------------------------------------------------------- 脏数据与三清单
def test_make_dirty_has_all_three_defects():
    df = core.make_dirty(300, 7)
    assert df["temp"].isna().any() and df["vib"].isna().any()
    assert ((df["temp"] > 200) | (df["vib"] < 0)).any()
    assert int(df.duplicated().sum()) > 0            # 重复清单有货


def test_missing_profile_reports_rate_and_mechanism():
    df = core.make_dirty(300, 7)
    rows = core.missing_profile(df, topk=3)
    assert len(rows) == len(core.NUMERIC_COLS)
    by = {r["column"]: r for r in rows}
    assert by["temp"]["n_missing"] == int(df["temp"].isna().sum())
    assert by["temp"]["missing_rate_pct"] > 0.0
    assert "成片" in by["pressure"]["mechanism"]      # 断电段被识别为成片缺失
    assert sum(r["watch"] for r in rows) <= 3         # topk 生效


def test_anomaly_profile_uses_iqr_and_sigma_and_physical():
    df = core.make_dirty(300, 7)
    rows = {r["column"]: r for r in core.anomaly_profile(df)}
    v = df["vib"].astype(float).dropna()
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    assert rows["vib"]["iqr_bounds"].startswith("[%.3f" % (q1 - 1.5 * (q3 - q1)))
    assert rows["vib"]["n_phys"] >= 5                 # -1.0 的负振动必被抓
    assert rows["temp"]["n_sigma3"] > 0               # 135℃ 尖峰
    assert all(r["n_union"] >= r["n_phys"] for r in rows.values())


def test_duplicate_profile_lists_rows():
    df = core.make_dirty(300, 7)
    rows, idx = core.duplicate_profile(df)
    assert len(rows) == len(idx) == int(df.duplicated().sum())
    assert all(df.duplicated(keep=False).iloc[i] for i in idx)


# ---------------------------------------------------------------- 双解互验
def test_two_solutions_agree():
    df = core.make_dirty(300, 7)
    assert core.agree(core.flags_py(df), core.flags_pd(df))
    assert sum(core.flags_py(df)) > 0                 # 非全零才算真互验


# ---------------------------------------------------------------- 缺失填充（D 层）
def test_impute_keeps_row_count_and_clears_temp_missing():
    df = core.make_dirty(300, 7)
    out, log, stats = core.impute(df)
    assert len(out) == len(df)                        # 行数不变
    assert stats["temp_missing_after"] == 0           # 温度列缺失率降为 0
    assert all(out[c].notna().all() for c in core.NUMERIC_COLS)
    assert len(log) == len(core.NUMERIC_COLS)         # 每列都留痕


def test_impute_handles_every_missing_cell():
    df = core.make_dirty(300, 7)
    out, log, stats = core.impute(df)
    assert stats["handled"] == stats["missing_cells"]
    rate = stats["handled"] / max(stats["missing_cells"], 1) * 100.0
    assert rate >= 95.0                               # C 层红线
    assert "pressure_missing_flag" in out.columns     # 缺失即特征


def test_blackout_segment_marked_not_faked():
    df = core.make_dirty(300, 7)
    out, log, stats = core.impute(df)
    clean, fixed = core.correct_anomalies(out, stats)
    assert stats["n_review"] == core.BLACKOUT
    assert int(clean["review_flag"].sum()) == core.BLACKOUT


def test_anomaly_correction_removes_physical_violations_only():
    df = core.make_dirty(300, 7)
    filled, log, stats = core.impute(df)
    clean, fixed = core.correct_anomalies(filled, stats)
    assert len(clean) == len(df)                      # 不删行
    assert sum(fixed.values()) > 0
    for c in core.NUMERIC_COLS:
        lo, hi = core.PHYS_RANGE[c]
        assert clean[c].between(lo, hi).all()


# ---------------------------------------------------------------- 质量分（B 层）
def test_quality_is_weighted_three_rates():
    # 完整 100 / 合规 90 / 唯一 80，权重 0.4/0.4/0.2 → 40+36+16 = 92
    assert core.quality(100.0, 90.0, 80.0, (0.4, 0.4, 0.2)) == 92.0


def test_quality_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        core.quality(100.0, 100.0, 100.0, (0.5, 0.3, 0.1))


def test_quality_monotone_in_each_rate():
    base = core.quality(90.0, 90.0, 90.0)
    assert core.quality(95.0, 90.0, 90.0) > base
    assert core.quality(90.0, 95.0, 90.0) > base
    assert core.quality(90.0, 90.0, 95.0) > base


def test_rates_matches_manual_counts():
    comp, acc, uniq = core.rates(1000, 100, 50, 200, 10)
    assert (comp, acc, uniq) == (90.0, 95.0, 95.0)


# ---------------------------------------------------------------- A 层实验
@pytest.fixture(scope="module")
def _exp():
    raw = core.make_dirty(300, 7)
    filled, log, stats = core.impute(raw)
    clean, _ = core.correct_anomalies(filled, stats)
    return core.imputation_experiment(raw, clean, seed=7)


def test_predictive_fill_beats_mean_fill(_exp):
    assert _exp["better"] is True
    assert _exp["mae_pred"] < _exp["mae_mean"]
    assert _exp["rmse_pred"] < _exp["rmse_mean"]


def test_predictive_fill_lift_is_material(_exp):
    """改善必须"值得写进工单"——至少 20% MAE，且逐格胜率过半。"""
    assert _exp["mae_lift_pct"] >= 20.0
    assert _exp["pred_win_rate"] > 0.5
    assert _exp["n_compared"] >= 100


def test_experiment_excludes_injected_spikes(_exp):
    """毛刺（135℃）不是真值：考卷只收通过三把尺的可信观测。"""
    raw = core.make_dirty(300, 7)
    n_spike = int((raw["temp"] == 135.0).sum())
    observed = int(raw["temp"].notna().sum())
    assert _exp["n_valid_obs"] <= observed - n_spike


def test_value_valid_mask_rejects_out_of_range():
    import pandas as pd
    df = pd.DataFrame({"temp": [50.0, 51.0, 135.0, np.nan, 52.0]})
    m = core.value_valid_mask(df, "temp")
    assert bool(m.iloc[2]) is False and bool(m.iloc[3]) is False
