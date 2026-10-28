# -*- coding: utf-8 -*-
import os

import numpy as np

from mlp_clf import core


def test_make_spectra_shape():
    X, y = core.make_spectra(200, 1)
    assert X.shape == (200, 30) and set(y) <= {0, 1}


def test_train_returns_acc():
    X, y = core.make_spectra(300, 2)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    m, acc = core.train(Xtr, ytr, Xte, yte)
    assert 0.0 <= acc <= 1.0 and len(m.loss_curve_) > 0


def test_diagnose_four_segments():
    _, dg = None, core.diagnose(list(np.linspace(1.0, 0.1, 40)))
    assert (dg["segments"] == 4 and dg["improving"]
            and dg["means"][0] > dg["means"][3])


def test_diagnose_short_curve():
    dg = core.diagnose([0.5, 0.4, 0.3])
    assert dg["segments"] == 1


# ---------- 优(A)线：真·隐层结构搜索（删功能即红） ----------

def test_structure_search_two_candidates_real_numbers():
    X, y = core.make_spectra(300, 2)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    ss = core.structure_search(
        Xtr, ytr, Xte, yte, configs=((32, 16), (64, 32, 16)))
    assert [tuple(r["hidden"]) for r in ss["rows"]] == [(32, 16), (64, 32, 16)]
    for r in ss["rows"]:
        assert (0.0 <= r["acc"] <= 1.0 and r["params"] > 0
                and r["epochs"] > 0 and r["sec"] > 0)
    # 参数量口径：(64,32,16) 必比 (32,16) 大——真在搜"结构"而不是抄答案
    assert ss["rows"][1]["params"] > ss["rows"][0]["params"]
    assert ss["best_value"] in ss["rows"]


def test_n_params_hand_check():
    # 30→16→2：30*16+16 + 16*2+2 = 512
    assert core.n_params((16,), 30, 2) == 30 * 16 + 16 + 16 * 2 + 2


# ---------- torch 臂 + Dropout + 曲线图 + 过拟合诊断（良好C/优秀B线） ----------

def test_torch_arm_uses_linear_and_dropout():
    """源码级防删：工单点名 nn.Linear + nn.Dropout，兜底路径必须明写。"""
    import inspect
    src = inspect.getsource(core.torch_mlp)
    assert "torch.nn.Linear" in src and "torch.nn.Dropout" in src
    assert "离线兜底" in src


def test_torch_arm_returns_curves_and_acc():
    X, y = core.make_spectra(300, 2)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    arm = core.torch_mlp(Xtr, ytr, Xte, yte, dropout=0.3, epochs=15)
    assert arm["engine"] in ("torch", "sklearn-fallback")
    assert (0.0 <= arm["acc"] <= 1.0
            and len(arm["train_losses"]) == len(arm["val_losses"]))
    if arm["engine"] == "torch":
        assert (len(arm["train_losses"]) == 15
                and arm["train_losses"][0] > arm["train_losses"][-1])
    else:
        assert "未安装 torch" in arm["note"]   # 兜底必须诚实声明，不得冒充 torch 臂


def test_overfit_diagnosis_verdicts():
    tr = list(np.linspace(0.6, 0.05, 40))
    va_flat = list(np.linspace(0.6, 0.25, 40))     # val 持续下降：无过拟合
    va_up = (list(np.linspace(0.6, 0.3, 30))
             + list(np.linspace(0.3, 0.5, 10)))    # 后期回升
    d1 = core.overfit_diagnosis(tr, va_flat)
    d2 = core.overfit_diagnosis(tr, va_up)
    assert d2["val_rising"] and d2["gap"] > d1["gap"]
    assert "疑似过拟合" in d2["verdict"] and "疑似过拟合" not in d1["verdict"]


def test_plot_losses_writes_png(tmp_path):
    import inspect
    src = inspect.getsource(core.plot_losses)
    assert 'matplotlib.use("Agg")' in src          # 无头环境铁律
    p = core.plot_losses([0.6, 0.4, 0.3, 0.25], [0.7, 0.5, 0.45, 0.5],
                         str(tmp_path / "curve.png"))
    assert os.path.exists(p) and os.path.getsize(p) > 1024


def test_verdict_requires_png_and_compliant_arm():
    """C 线"图完整"要真 PNG 字节；无合规臂（gap 收窄且指标不降）B 线不算达标。"""
    from mlp_clf import tiers as T
    gates = {"D": {"style_min": 70.0},
             "C": {"acc_min": 0.75, "png_min_bytes": 1000},
             "B": {"dropout_arms": [0.2]},
             "A": {"candidates_min": 2}}
    ok_arms = [{"engine": "torch", "dropout": 0.2, "acc": 0.93, "gap": 0.30}]
    v = T.verdict(gates, True, 0.93, 5000, 0.50, 0.93, ok_arms,
                  3, True, True, 95.0)
    t = {x["tier"][-2]: x for x in v["tiers"]}
    assert t["C"]["pass"] and t["B"]["pass"] and t["A"]["pass"]
    no_png = T.verdict(gates, True, 0.93, 0, 0.50, 0.93, ok_arms,
                       3, True, True, 95.0)
    assert no_png["claimed"] == "及格(D)"        # 图缺失 → C 不成立
    wider = T.verdict(gates, True, 0.93, 5000, 0.50, 0.93,
                      [{"engine": "torch", "dropout": 0.2,
                        "acc": 0.93, "gap": 0.60}],
                      3, True, True, 95.0)
    assert wider["claimed"] == "良好(C)"         # gap 未收窄 → B 不成立
    dropped = T.verdict(gates, True, 0.93, 5000, 0.50, 0.93,
                        [{"engine": "torch", "dropout": 0.2,
                          "acc": 0.70, "gap": 0.30}],
                        3, True, True, 95.0)
    assert dropped["claimed"] == "良好(C)"       # 指标降了 → 严格口径不放行
    no_ss = T.verdict(gates, True, 0.93, 5000, 0.50, 0.93, ok_arms,
                      0, False, True, 95.0)
    assert no_ss["claimed"] == "优秀(B)"         # 无结构候选 → A 不成立
