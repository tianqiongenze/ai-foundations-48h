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
    assert 0.0 <= r["acc"] <= 1.0 and set(r["missed"]) == {"正常", "轴承磨损", "叶轮堵塞"}
