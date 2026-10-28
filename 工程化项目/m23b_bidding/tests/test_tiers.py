# -*- coding: utf-8 -*-
"""工单 M2-3b 验收门禁测试：把"四、验收标准"四档写成可执行断言。

  及格(D)：指标可复现（同种子两次拟合 acc 一致）+ 代码规范率≥70%
  良好(C)：CNN≥75% 或 LSTM 误差≤25%——仅认真卷积/真 LSTM 臂的实测数字
  优秀(B)：CNN vs LSTM 两轮实测对比 + 决策记录完整（竞标胜负/评委加分另人工）
  优  (A)：给出选型决策矩阵（候选×权重×实测得分→结论）
用真实 config/rules.yaml 全量跑一遍（约 30 秒），断言的是**实测数字**，
不是"代码没报错"。任何一条破了，对应档位当场掉级——CI 上直接红。
"""
import json
import os

import pytest

from bidding_lines import config as C
from bidding_lines import pipeline as P
from bidding_lines import tiers as T

CFG = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("m23b"))
    P.run(C.load_config(CFG), out_dir=out)
    with open(os.path.join(out, "result.json"), encoding="utf-8") as f:
        return json.load(f), out


def _weaken(r, **over):
    """拿真实实测值造"削弱版"——只改被质疑的那一个测量，其余原样。"""
    m = dict(r["tier_verdict"]["measured"])
    m["evidence"] = r["tier_verdict"]["evidence"]
    m.update(over)
    return T.verdict(m, r["code_style"], C.load_config(CFG).tier_gates())


# ---------------------------------------------------------------- 及格 D
def test_D_same_seed_refit_is_reproducible(res):
    r, _ = res
    assert r["reproducible"] is True
    assert r["tiers"]["D"]["达标"] is True


def test_D_measured_style_rate_is_at_least_70(res):
    r, out = res
    st = r["code_style"]
    assert st["rate"] >= 70.0, (
        "D 档规范率门槛≥70%%，实测 %.2f%%（引擎 %s）" % (st["rate"], st["engine"]))
    assert st["engine"] and st["n_files"] > 0 and st["code_lines"] > 0
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        assert "检查引擎" in f.read()


# ---------------------------------------------------------------- 良好 C
def test_C_real_engine_arm_meets_the_line(res):
    """过线的数字必须来自真卷积/真 LSTM 臂——兜底臂高分也不算能力达标。"""
    r, _ = res
    g = r["tier_gates"]["C"]
    cnn, lstm = r["cnn_arm"], r["lstm_arm"]
    c_cnn = cnn["engine"] == "torch" and cnn["acc"] >= g["cnn_min_acc"] - 1e-9
    ok_lstm = (lstm["engine"] == "torch"
               and (1.0 - lstm["acc"]) <= g["lstm_max_err"] + 1e-9)
    assert c_cnn or ok_lstm, (
        "CNN(%s) acc=%.4f（线 %.2f）；LSTM(%s) 错误率=%.4f（线 %.2f）" % (
            cnn["engine"], cnn["acc"], g["cnn_min_acc"],
            lstm["engine"], round(1.0 - lstm["acc"], 4), g["lstm_max_err"]))
    assert r["tiers"]["C"]["达标"] is True


# ---------------------------------------------------------------- 优秀 B
def test_B_two_arm_comparison_and_decision_record(res):
    r, out = res
    assert r["two_arm_real"] is True             # 两轮均为真引擎，指标同口径可比
    assert r["decision_record_chars"] >= \
        r["tier_gates"]["B"]["min_decision_chars"]
    assert r["tiers"]["B"]["达标"] is True
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "[两轮对比（CNN vs LSTM，工单'二选一'的实测根据）]" in text
    assert "[决策记录]" in text


def test_B_human_half_never_self_certified(res):
    """竞标胜负/评委加分是人工环节：报告只列'人工答辩项'，不许写达标。"""
    r, out = res
    assert any("赢得竞标" in x for x in r["tier_verdict"]["manual"])
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "人工答辩项（程序不判、不许自称达标）" in text
    assert "赢得竞标" in text


# ---------------------------------------------------------------- 优 A
def test_A_decision_matrix_is_measured_and_conclusive(res):
    r, out = res
    dm = r["decision_matrix"]
    assert len(dm["rows"]) >= r["tier_gates"]["A"]["min_matrix_rows"]
    assert abs(dm["weights_sum"] - 1.0) < 1e-6   # 权重合法，否则矩阵是数字游戏
    assert dm["top"] and dm["conclusion"]
    for row in dm["rows"]:
        assert 0.0 <= row["acc"] <= 1.0 and row["sec"] > 0.0   # 全是跑出来的
    assert r["tiers"]["A"]["达标"] is True
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "[选型决策矩阵（候选×权重×实测得分→结论）]" in text
    assert dm["conclusion"] in text              # 结论真进了报告，不是只在代码里
    docs = os.path.join(ROOT, "docs", "09-选型决策矩阵.md")
    assert os.path.exists(docs) and os.path.getsize(docs) > 200


# ------------------------------------------------ 门槛外置 + 标准段渲染
def test_tier_gates_come_from_rules_yaml_and_match_workorder(res):
    """四档门槛出自 rules.yaml tiers: 段，数字与工单《M2-3b》验收表逐字一致。"""
    g = C.load_config(CFG).tier_gates()
    assert g["D"]["min_code_style_pct"] == 70.0   # 工单 D：代码规范率≥70%
    assert g["D"]["require_reproducible"] is True  # 工单 D：指标可复现
    assert g["C"]["cnn_min_acc"] == 0.75           # 工单 C：CNN≥75%
    assert g["C"]["lstm_max_err"] == 0.25          # 工单 C：LSTM 误差达标线
    assert g["C"]["require_real_engine"] is True   # 兜底臂不认
    assert g["A"]["min_matrix_rows"] == 4          # 工单 A：选型决策矩阵
    # 与展示线（顶层 cnn_target/lstm_err_target）不许两套口径
    r, _ = res
    assert g["C"]["cnn_min_acc"] == C.load_config(CFG).raw["cnn_target"]
    assert g["C"]["lstm_max_err"] == C.load_config(CFG).raw["lstm_err_target"]
    assert r["min_code_style"] == 70.0


def test_standard_tier_section_landed_in_report(res):
    """report.txt 必须出现标准段 [验收档位判定（实测）] 与逐档判定行。"""
    r, out = res
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "[验收档位判定（实测）]" in text
    assert "实测达成档位: %s" % r["claimed_tier"] in text
    for name in T.LABELS.values():
        assert name in text
    assert "人工答辩项" in text                   # manual 不许冒充实测


def test_render_lines_uses_verdict_not_constants(res):
    """渲染与判定同源：--tier 与报告行内容一致（同一 verdict 对象的产物）。"""
    r, _ = res
    lines = T.render_lines(r["tier_verdict"])
    assert lines[0].startswith("实测达成档位: %s" % r["claimed_tier"])
    assert len(lines) == 1 + 4 + 1                # 表头 + 四档 + 人工项


# ------------------------------------------------ 反向证明：削弱实测必掉级
def test_D_gate_bites_when_style_rate_low():
    """闸门会咬：规范率实测不足 70%，即便业务全过，D 也必须判不达标。"""
    biz = {"D": True, "C": True, "B": True, "A": True}
    st = {"rate": 65.0, "passes": False, "min_pass": 70.0}
    v = P.tier_verdict(biz, st)
    assert v["final"]["D"] is False and v["claimed"] == "未达 D 档"


def test_tier_drops_when_measured_not_reproducible(res):
    """反向证明：同种子复现比对改判'漂移'，D 直接掉到未达 D 档。"""
    r, _ = res
    v = _weaken(r, reproducible=False)
    assert v["final"]["D"] is False
    assert v["claimed"] == "未达 D 档"


def test_tier_drops_when_arms_fall_back(res):
    """反向证明：两臂换成兜底引擎（哪怕高分），C 必须掉级到 D。"""
    r, _ = res
    v = _weaken(r, cnn_engine="sklearn-fallback",
                lstm_engine="sklearn-fallback", two_arm_real=False)
    assert v["final"]["D"] and not v["final"]["C"]
    assert v["claimed"] == "及格(D)"


def test_tier_drops_when_decision_record_thinned(res):
    """反向证明：决策记录字数削到线下，B 掉级到 C——记录真在实测里。"""
    r, _ = res
    v = _weaken(r, decision_chars=10)
    assert v["final"]["C"] and not v["final"]["B"]
    assert v["claimed"] == "良好(C)"


def test_tier_drops_when_matrix_conclusion_missing(res):
    """反向证明：矩阵结论不成文（0 字），A 掉级到 B。"""
    r, _ = res
    v = _weaken(r, conclusion_chars=0)
    assert v["final"]["B"] and not v["final"]["A"]
    assert v["claimed"] == "优秀(B)"


def test_claimed_tier_recorded(res):
    r, _ = res
    assert r["claimed_tier"] in list(T.LABELS.values()) + ["未达 D 档"]


# ---------------------------------------------------------------- 产物
def test_artifacts_real_and_substantial(res):
    """真实产物存在且字节达标：报告/结果/矩阵 CSV 不许是空壳。"""
    r, out = res
    for fn in ("report.txt", "result.json"):
        assert os.path.getsize(os.path.join(out, fn)) > 1000, fn
    for fn in ("decision_matrix.csv",):
        assert os.path.exists(os.path.join(out, fn)) and \
            os.path.getsize(os.path.join(out, fn)) > 0, fn
