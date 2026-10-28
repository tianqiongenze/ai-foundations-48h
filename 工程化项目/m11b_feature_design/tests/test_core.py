# -*- coding: utf-8 -*-
"""特征设计核心算法单测（不跑全流水线，只验单函数口径）。"""
import numpy as np
import pytest

from feature_design48 import core


# --------------------------------------------------------------- 特征字典
def test_feature_dict_has_at_least_10_features():
    assert len(core.FEATURE_DICT) >= 10          # 工单 C 档硬线
    assert len(core.FEAT_NAMES) == len(set(core.FEAT_NAMES))   # 名字不许重复


def test_feature_dict_five_columns_complete():
    assert core.dict_missing_columns() == []     # 五栏一个都不能空


def test_feature_dict_domain_split_13_time_5_freq():
    assert len(core.TIME_FEATS) == 13 and len(core.FREQ_FEATS) == 5
    assert set(core.TIME_FEATS) | set(core.FREQ_FEATS) == set(core.FEAT_NAMES)


def test_feature_dict_new_features_marked():
    assert core.NEW_FEATS and set(core.NEW_FEATS) <= set(core.FEAT_NAMES)
    assert {"f_ratio2x", "f_hf_ratio"} <= set(core.NEW_FEATS)


def test_set_tags_follows_dict_length():
    tg = core.set_tags(topk=6, n_screened=14)
    assert tg["topN"] == "D·Top6 精选" and tg["full"] == "C·全集(%d)" % len(core.FEAT_NAMES)
    assert tg["screened"] == "F·相关性初筛后(14)"


# --------------------------------------------------------------- 信号与特征
def test_feats_time_returns_all_13_documented_names():
    f = core.feats_time(np.sin(np.linspace(0, 20, 400)))
    assert set(f) == set(core.TIME_FEATS) and len(f) == 13


def test_feats_freq_returns_all_5_documented_names():
    f = core.feats_freq(np.sin(np.linspace(0, 20, 400)))
    assert set(f) == set(core.FREQ_FEATS) and len(f) == 5


def test_feats_time_no_broadcast_bug_on_positive_signal():
    """回归测试：过零率曾按 diff 长度取 sign 导致 (398,) vs (399,) 广播报错。"""
    f = core.feats_time(np.abs(np.sin(np.linspace(0, 20, 400))) + 1.0)
    assert f["t_zcr"] == 0.0


def test_feats_time_finite_on_all_inputs():
    t, x = core.make_signal("wear", n=400, seed=3)
    f = core.feats_time(x)
    assert all(np.isfinite(v) for v in f.values())


def test_make_signal_wear_ratio_higher_than_normal():
    """标签只走"二倍频比"这一条通道——特征设计的因果锚点。"""
    rn = [core.feats_freq(core.make_signal("normal", seed=s)[1])["f_ratio2x"]
          for s in range(20)]
    rw = [core.feats_freq(core.make_signal("wear", seed=s)[1])["f_ratio2x"]
          for s in range(20)]
    assert np.mean(rw) > 3 * np.mean(rn)


def test_make_signal_absolute_amplitude_is_not_the_label_cue():
    """工况随机到"绝对幅值分不开两类"——这正是本工单要教的坑。"""
    rms_n = [core.feats_time(core.make_signal("normal", seed=s)[1])["t_rms"]
             for s in range(30)]
    rms_w = [core.feats_time(core.make_signal("wear", seed=s)[1])["t_rms"]
             for s in range(30)]
    assert abs(np.mean(rms_n) - np.mean(rms_w)) < 1.5 * np.std(rms_n)


def test_extract_length_matches_dict():
    assert len(core.extract(np.random.RandomState(0).randn(400))) == len(core.FEAT_NAMES)


def test_build_dataset_shape_and_balance():
    X, y = core.build_dataset(10, seed=1)
    assert X.shape == (20, len(core.FEAT_NAMES))
    assert list(np.bincount(y)) == [10, 10]
    assert np.isfinite(X).all()


def test_subset_picks_named_columns():
    X = np.arange(36, dtype=float).reshape(2, 18)
    s = core.subset(X, ["f_a8", "t_mean"])
    assert s.shape == (2, 2)
    i_a8 = core.FEAT_NAMES.index("f_a8")
    i_mean = core.FEAT_NAMES.index("t_mean")
    assert np.array_equal(s[:, 0], X[:, i_a8])       # 列按给定名字顺序取
    assert np.array_equal(s[:, 1], X[:, i_mean])


# --------------------------------------------------------------- 评价与筛选
def test_evaluate_returns_acc_and_f1():
    X, y = core.build_dataset(8, seed=2)
    m = core.evaluate(X, y, folds=4, seed=0)
    assert 0.0 <= m["acc"] <= 1.0 and 0.0 <= m["f1"] <= 1.0
    assert set(m) == {"acc", "acc_std", "f1", "f1_std", "folds"}


def test_evaluate_multi_averages_seeds():
    X, y = core.build_dataset(8, seed=2)
    m = core.evaluate_multi(X, y, folds=4, seeds=(0, 1))
    assert m["seeds"] == [0, 1] and m["acc_spread"] >= 0.0
    single = core.evaluate(X, y, folds=4, seed=0)["acc"]
    assert abs(m["acc"] - single) <= 1.0


def test_rank_orders_by_importance_desc():
    X, y = core.build_dataset(10, seed=3)
    ranked, clf = core.rank(X, y)
    vals = [v for _, v in ranked]
    assert len(ranked) == len(core.FEAT_NAMES)
    assert vals == sorted(vals, reverse=True)
    assert abs(sum(vals) - 1.0) < 1e-3            # Gini 重要性总和为 1（4 位小数舍入后）


def test_rank_stable_carries_domain_and_version():
    X, y = core.build_dataset(10, seed=3)
    board = core.rank_stable(X, y, seeds=(0, 1))
    assert len(board) == len(core.FEAT_NAMES)
    for r in board:
        assert r["domain"] in ("时域", "频域") and r["since"] in ("v1.0", "v1.1")
        assert 1 <= r["rank_seed_min"] <= r["rank_seed_max"] <= len(board)


def test_correlation_screen_drops_redundant_clone():
    """构造一个"第 4 列 = 第 1 列的克隆"的矩阵：初筛必须砍掉克隆、留下原件。"""
    base = np.random.RandomState(0).randn(30, 3)
    Xc = np.column_stack([base, base[:, 0]])          # 第 4 列完全复制第 1 列
    names_bak = list(core.FEAT_NAMES)
    core.FEAT_NAMES.append("clone_of_first")
    try:
        pairs, keep = core.correlation_screen(Xc, threshold=0.95)
        assert len(pairs) == 1
        assert pairs[0]["a"] == core.FEAT_NAMES[0]
        assert pairs[0]["keep"] == core.FEAT_NAMES[0]
        assert keep == [0, 1, 2]                      # 克隆列 3 被剔除
    finally:
        core.FEAT_NAMES[:] = names_bak


def test_correlation_screen_single_column_safe():
    pairs, keep = core.correlation_screen(np.zeros((5, 1)))
    assert pairs == [] and keep == [0]


def test_ablate_cuts_at_least_the_requested_fraction():
    X, y = core.build_dataset(10, seed=5)
    ranked, _ = core.rank(X, y)
    ab = core.ablate(X, y, ranked, cut_frac=0.30, folds=4, seeds=(0,), n_random=1)
    assert ab["n_cut"] >= int(0.30 * ab["n_total"])
    assert ab["cut_pct_actual"] >= 30.0
    assert ab["keep"] == ab["n_total"] - ab["n_cut"]
    assert isinstance(ab["pass"], bool)


def test_feature_set_ab_has_seven_arms_with_ids():
    X, y = core.build_dataset(8, seed=6)
    out, best, top = core.feature_set_ab(X, y, topk=6, seeds=(0,))
    assert len(out) == 7 and len(top) == 6
    assert {v["id"] for v in out.values()} == {
        "min5", "time", "freq", "full", "topN", "top3", "screened"}
    assert best in out


def test_new_feature_check_detects_v11_in_top3():
    ranked = [("f_ratio2x", 0.4), ("f_a16", 0.3), ("f_hf_ratio", 0.2)]
    chk = core.new_feature_check(ranked, topn=3)
    assert chk["pass"] is True and chk["n_new_in_top"] == 2
    assert all(chk["meanings"][n] for n, _ in chk["new_in_top"])


@pytest.mark.parametrize("kind", ["normal", "wear"])
def test_pipeline_primitives_run_on_both_kinds(kind):
    t, x = core.make_signal(kind, n=400, seed=9)
    assert len(t) == len(x) == 400
    assert np.isfinite(core.extract(x)).all()
