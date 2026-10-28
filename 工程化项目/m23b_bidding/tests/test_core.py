# -*- coding: utf-8 -*-
import os

import numpy as np

from bidding_lines import core


def test_img_feats_three():
    f = core.img_feats(np.zeros((32, 32)))
    assert len(f) == 3


def test_seq_feats_one_value():
    assert len(core.seq_feats(np.arange(64, dtype=float))) == 1


def test_make_images_classes():
    X, y = core.make_images(60, 3)
    assert X.shape[1:] == (32, 32) and set(y) == {"正常", "气孔", "裂纹"}


def test_cv_between_0_and_1():
    X, y = core.make_images(90, 2)
    v = core.cv([core.img_feats(im) for im in X], y)
    assert 0.0 <= v <= 1.0


# ---------- 标签泄漏核查（审计整改核心，删功能即红） ----------

def test_make_series_fixed_has_class_overlap():
    """修复版：隐退化强度连续+过阈值打标 → 两类单特征值域必须可观重叠。"""
    X, y = core.make_series(180, seed=7)
    assert set(y) == {"健康", "退化"}
    f = np.array([core.seq_feats(x)[0] for x in X])
    lo, hi = [f[np.asarray(y) == c] for c in ("健康", "退化")]
    ov = min(lo.max(), hi.max()) - max(lo.min(), hi.min())
    assert ov / (f.max() - f.min()) > 0.05


def test_leakage_audit_flags_legacy_and_clears_fixed():
    """旧构造必被判泄漏（overlap≈0 且单特征 CV≈1），修复构造必须干净。"""
    leak = core.leakage_audit(core.make_series_leaky, core.seq_feats)
    fixed = core.leakage_audit(core.make_series, core.seq_feats)
    assert leak["leak_detected"] is True
    assert leak["single_feat_cv"] > 0.98 and leak["overlap_frac"] < 0.05
    assert fixed["leak_detected"] is False
    assert fixed["overlap_frac"] >= 0.05 or fixed["single_feat_cv"] <= 0.98


# ---------- 窗口数据集 + 四臂选型矩阵 ----------

def test_make_windows_shape_and_labels():
    X, y = core.make_series(20, 5)
    W, yw = core.make_windows(X, y, win=16, stride=8)
    assert W.shape[1:] == (16,) and len(W) == 20 * 7 and set(yw) <= {"健康", "退化"}
    # 窗口的"趋势"继承自序列：前半/后半 RMS 比与父序列一致方向
    assert core.win_feats(W[0])[1] > 0


def test_fit_eval_measures_acc_and_wallclock():
    from sklearn.ensemble import RandomForestClassifier
    X, y = core.make_images(120, 2)
    F = np.array([core.img_feats(im) for im in X])
    r = core.fit_eval(RandomForestClassifier(random_state=0),
                      F[:80], y[:80], F[80:], y[80:])
    assert 0.0 <= r["acc"] <= 1.0 and r["sec"] > 0


def test_torch_arms_declare_conv2d_and_lstm():
    """工单点名 nn.Conv2d / nn.LSTM；兜底路径必须明写且不得冒充真臂。"""
    import inspect
    src_cnn, src_lstm = (inspect.getsource(core.torch_cnn),
                         inspect.getsource(core.torch_lstm))
    assert "torch.nn.Conv2d" in src_cnn and "离线兜底" in src_cnn
    assert "torch.nn.LSTM" in src_lstm and "离线兜底" in src_lstm


def test_torch_cnn_and_lstm_real_metrics():
    X, y = core.make_images(120, 2)
    Itr, Ite = X[:80], X[80:]
    cnn = core.torch_cnn(Itr, y[:80], Ite, y[80:], ["正常", "气孔", "裂纹"], epochs=12)
    assert 0.0 <= cnn["acc"] <= 1.0 and cnn["sec"] > 0
    # 要么真 torch 臂，要么兜底且 note 明写——不许无声冒充
    assert cnn["engine"] == "torch" or "离线兜底" in cnn["note"]
    Xs, ys = core.make_series(60, 3)
    W, yw = core.make_windows(Xs, ys)
    lstm = core.torch_lstm(W[:300], yw[:300], W[300:], yw[300:], ["健康", "退化"], epochs=10)
    assert 0.0 <= lstm["acc"] <= 1.0 and lstm["sec"] > 0
    assert lstm["engine"] == "torch" or "离线兜底" in lstm["note"]


def test_tier_verdict_no_达标_for_human_tier_and_fallback_arms():
    """兜底臂数字不得替真卷积/真 LSTM 声明达标；竞标胜负只列人工不冒充实测。"""
    from bidding_lines import config as C
    from bidding_lines import tiers as T
    cfg = C.load_config(os.path.join(os.path.dirname(__file__), "..",
                                     "config", "rules.yaml"))
    g = cfg.tier_gates()
    st = {"rate": 90.0, "passes": True, "min_pass": 70.0}
    m = {"reproducible": True, "cnn_engine": "torch", "cnn_acc": 0.9,
         "lstm_engine": "torch", "lstm_err": 0.30, "two_arm_real": True,
         "decision_chars": 200, "matrix_rows": 6, "weights_sum": 1.0,
         "conclusion_chars": 60,
         "evidence": {k: "x" for k in ("D", "C", "B", "A")}}
    v = T.verdict(m, st, g)
    assert v["final"]["C"] is True                      # CNN 真臂过线即 C
    assert v["claimed"] == "优(A)"
    fb = dict(m, cnn_engine="sklearn-fallback", cnn_acc=0.99,
              lstm_engine="sklearn-fallback", lstm_err=0.01,
              two_arm_real=False)
    v2 = T.verdict(fb, st, g)
    assert v2["final"]["C"] is False                   # 兜底臂高分也不算能力达标
    assert v2["final"]["B"] is False                   # 兜底臂两轮对比不成立
    assert v2["claimed"] == "及格(D)"
    assert any("赢得竞标" in x for x in T.MANUAL)      # 人工环节如实列，不自动达标
