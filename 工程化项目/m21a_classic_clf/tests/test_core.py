# -*- coding: utf-8 -*-

from classic_clf import core


def test_make_data_labels_complete():
    X, y = core.make_data(9, 1)
    assert set(y) == {"正常", "轴承磨损", "叶轮堵塞"}
    assert "轴承" not in set(y)


def test_confusion_diagonal():
    cm = core.confusion(["a", "b"], ["a", "b"], ["a", "b"])
    assert cm["a"]["a"] == 1 and cm["b"]["b"] == 1


def test_missed_counts_off_diagonal():
    cm = {"a": {"a": 2, "b": 1}, "b": {"a": 0, "b": 3}}
    assert core.missed(cm, ["a", "b"]) == {"a": 1, "b": 0}


def test_evaluate_returns_acc_and_cm():
    from sklearn.tree import DecisionTreeClassifier
    X, y = core.make_data(150, 3)
    r = core.evaluate(DecisionTreeClassifier(max_depth=3, random_state=0),
                      X[:100], y[:100], X[100:], y[100:],
                      ["正常", "轴承磨损", "叶轮堵塞"])
    assert 0.0 <= r["acc"] <= 1.0
    assert set(r["missed"]) == {"正常", "轴承磨损", "叶轮堵塞"}


# ---------- 良好(C)/优秀(B) 线：P/R/F1 与阈值扫描（删功能即红） ----------

def test_prf_values_match_hand_computation():
    r = core.prf(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert r["per_class"]["a"]["precision"] == 1.0      # 预测 a 全对
    assert r["per_class"]["a"]["recall"] == 0.5          # 真 a 漏一半
    assert r["per_class"]["b"]["precision"] == 0.5
    assert r["per_class"]["b"]["recall"] == 1.0
    assert abs(r["per_class"]["a"]["f1"] - 0.6667) < 1e-3
    assert abs(r["macro"]["f1"] - 0.6667) < 1e-3


def test_threshold_sweep_shows_pr_tradeoff():
    # 可分分数：正类高分、负类低分——阈值升高必精确率不降、召回率不升
    sw = core.threshold_sweep(
        ["a", "a", "b", "b"], [0.9, 0.8, 0.2, 0.1], "a", n_points=9)
    rows = sw["rows"]
    assert len(rows) == 9
    # 末档阈值下无正预测，精确率按 0 记，不参与单调比较
    precs = [x["precision"] for x in rows[:-1]]
    recs = [x["recall"] for x in rows]
    # 精确率单调不降 / 召回率单调不升
    assert all(precs[i] <= precs[i + 1] + 1e-12
               for i in range(len(precs) - 1))
    assert all(recs[i] >= recs[i + 1] - 1e-12 for i in range(8))
    assert sw["best"]["f1"] >= 0.8 and 0.05 <= sw["best"]["threshold"] <= 0.95


def test_evaluate_reports_prf_and_sweep():
    """分级判定（良好C线）依赖 evaluate 返回 prf 与 threshold_sweep——功能被删此测试即红。"""
    from sklearn.linear_model import LogisticRegression
    X, y = core.make_data(150, 3)
    # 三类都进训练集，OvR 扫描才有三类
    r = core.evaluate(LogisticRegression(max_iter=200),
                      X[:120], y[:120], X[120:], y[120:],
                      ["正常", "轴承磨损", "叶轮堵塞"])
    assert set(r["prf"]["per_class"]) == {"正常", "轴承磨损", "叶轮堵塞"}
    assert 0.0 <= r["prf"]["macro"]["f1"] <= 1.0
    assert set(r["threshold_sweep"]) == {"正常", "轴承磨损", "叶轮堵塞"}
    assert r["threshold_sweep"]["轴承磨损"]["best"]["f1"] >= min(
        x["f1"] for x in r["threshold_sweep"]["轴承磨损"]["rows"])
    assert any("precision" in ln.lower() for ln in r["report_lines"])


def test_tier_verdict_never_claim_达标_on_failing_f1():
    """课程铁律：实测 F1 低于线时判定必须写"未达标"，不得虚标。"""
    from classic_clf import pipeline as P
    from classic_clf import tiers as T

    gates = {"D": {"style_min": 70.0}, "C": {"f1_min": 0.70},
             "B": {"sweep_points_min": 19}, "A": {"f1_min": 0.85}}

    def v(macro_f1):
        return T.verdict(gates, lr_acc=0.7, dt_acc=0.7, macro_f1=macro_f1,
                         caliber_ok=True, sweep_min_points=19,
                         sweep_total_points=57, decision_ok=True,
                         style_rate=95.0)

    low, high = v(0.50), v(0.90)
    by_key = {t["tier"][-2]: t for t in low["tiers"]}
    assert not by_key["C"]["functional"] and low["claimed"] == "及格(D)"
    by_key = {t["tier"][-2]: t for t in high["tiers"]}
    assert by_key["C"]["pass"] and by_key["A"]["pass"]
    assert high["claimed"] == "优(A)"
    assert hasattr(P, "run")  # 判定逻辑已收敛到 tiers.py，pipeline 只做编排
