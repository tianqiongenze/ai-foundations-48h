# -*- coding: utf-8 -*-
"""工单 M1-1a 验收门禁测试：把"四、验收标准"的四档数值线写成可执行断言。

  及格(D)：温度列缺失率降为 0 且行数不变
  良好(C)：缺失处理率 ≥ 95% 且三清单齐全
  优秀(B)：质量分 ≥ 85 且三率明细/策略理由/决策记录完整
  优  (A)：预测填充优于均值填充（MAE 更低）
任何一条破了，工单对应档位就掉级——CI 上直接红。
"""
import json
import os

import pytest

from data_quality48 import config as C
from data_quality48 import pipeline as P
from data_quality48 import tiers as T

CFG = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("m11a"))
    cfg = C.load_config(CFG)
    P.run(cfg, out_dir=out)
    with open(os.path.join(out, "result.json"), encoding="utf-8") as f:
        return json.load(f), out


def test_D_temp_missing_zero_and_rows_unchanged(res):
    r, _ = res
    assert r["temp_missing_after"] == 0
    assert r["rows_after_missing_stage"] == r["rows_before"]      # 不许删行凑数
    assert r["tiers"]["D"]["达标"] is True


def test_D_measured_style_rate_is_at_least_70(res):
    """工单 D 档把"代码规范率≥70%"钉死——用的是真跑出来的数，不是常量。"""
    r, out = res
    st = r["code_style"]
    assert st["rate"] >= 70.0, (
        "D 档规范率门槛≥70%%，实测 %.2f%%（引擎 %s）" % (st["rate"], st["engine"]))
    assert st["engine"] and st["n_files"] > 0 and st["code_lines"] > 0
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "代码规范率" in text and "检查引擎" in text      # 引擎/明细落进报告
    assert ("达标" if st["passes"] else "未达标") in text


def test_C_missing_handle_rate_ge_95_and_three_lists(res):
    r, out = res
    assert r["missing_handle_rate_pct"] >= 95.0
    assert r["tiers"]["C"]["达标"] is True
    for fn in ("missing_list.csv", "anomaly_list.csv", "duplicate_list.csv"):
        p = os.path.join(out, fn)
        assert os.path.exists(p) and os.path.getsize(p) > 0        # 三清单齐全
    assert r["lists"]["missing"] and r["lists"]["anomaly"]


def test_C_missing_list_covers_every_numeric_column(res):
    r, _ = res
    assert {row["column"] for row in r["lists"]["missing"]} == set(
        ["temp", "vib", "current", "pressure"])


def test_B_quality_score_ge_85_with_three_rate_ledger(res):
    r, _ = res
    assert r["quality_after"] >= 85.0
    assert r["tiers"]["B"]["达标"] is True
    for k in ("complete", "accurate", "consistent"):
        assert k in r["rates_after"] and 0.0 <= r["rates_after"][k] <= 100.0
    assert abs(sum(r["weights"]) - 1.0) < 1e-9                    # 权重合法
    assert len(r["audit_log"]) == 4 and all(e["reason"] for e in r["audit_log"])


def test_B_decision_record_present(res):
    r, out = res
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "决策记录" in text and "候选方案" in text and "理由" in text


def test_A_predictive_fill_beats_mean_with_numbers(res):
    r, _ = res
    e = r["imputation_experiment"]
    assert e["mae_pred"] < e["mae_mean"] and e["rmse_pred"] < e["rmse_mean"]
    assert e["mae_lift_pct"] > 0 and e["better"] is True
    assert r["tiers"]["A"]["达标"] is True


def test_quality_before_is_lower_than_after(res):
    """清洗必须真的有用：分数上升，否则"清洗"没有证据。"""
    r, _ = res
    assert r["quality_gain"] > 0
    assert r["rates_after"]["complete"] > r["rates_before"]["complete"]


def test_duplicates_all_removed_and_row_trace(res):
    r, _ = res
    assert r["duplicates_after"] == 0
    assert r["rows_final"] == r["rows_before"] - r["duplicates_removed"]


def test_dual_implementations_agree(res):
    r, _ = res
    assert r["dual_check"]["consistent"] is True
    assert r["dual_check"]["py_rows_flagged"] > 0


def _fake_st(rate, min_pass=70.0):
    """给 tier_verdict 造一个"规范率引擎"结果——只填它读的几个键。"""
    return {"rate": rate, "passes": rate >= min_pass, "min_pass": min_pass}


def test_D_gate_bites_when_style_rate_low():
    """闸门会咬：规范率实测不足 70%，即便业务全过，D 档也必须判不达标。"""
    biz = {"D": True, "C": True, "B": True, "A": True}
    v = P.tier_verdict(biz, _fake_st(65.0))
    assert v["final"]["D"] is False
    assert v["claimed"] == "未达 D 档"                          # D 没过不许承诺


def test_claimed_tier_drops_when_a_capability_removed():
    """反向证明：拿掉 A 档能力，承诺档位从 A 掉到 B——档位真由实测决定。"""
    biz = {"D": True, "C": True, "B": True, "A": False}
    v = P.tier_verdict(biz, _fake_st(99.0))
    assert v["final"]["D"] and v["final"]["C"] and v["final"]["B"]
    assert not v["final"]["A"] and v["claimed"] == "优秀(B)"


def test_all_four_tiers_present_and_marked(res):
    r, _ = res
    assert set(r["tiers"]) == {"D", "C", "B", "A"}
    assert all(v["要求"] and v["实测"] for v in r["tiers"].values())
    assert r["claimed_tier"] in list(T.LABELS.values()) + ["未达 D 档"]


# ------------------------------------------- 门槛外置 + 标准段渲染 + 产物大小
def test_tier_gates_come_from_rules_yaml_and_match_workorder(res):
    """四档门槛出自 rules.yaml tiers: 段，数字与工单《M1-1a》验收表逐字一致。"""
    cfg = C.load_config(CFG)
    g = cfg.tier_gates()
    assert g["D"]["min_code_style_pct"] == 70.0      # 工单 D：代码规范率≥70%
    assert g["D"]["max_temp_missing_cells"] == 0     # 工单 D：缺失率降为 0
    assert g["D"]["require_rows_unchanged"] is True  # 工单 D：行数不变
    assert g["C"]["min_handle_rate_pct"] == 95.0     # 工单 C：处理率≥95%
    assert g["C"]["min_lists"] == 3                  # 工单 C：三清单齐全
    assert g["B"]["min_quality_score"] == 85.0       # 工单 B：评分≥85
    # 与报告展示线（danger/quality_target）不许两套口径
    assert g["C"]["min_handle_rate_pct"] == cfg.danger
    assert g["B"]["min_quality_score"] == cfg.raw["quality_target"]


def test_standard_tier_section_landed_in_report(res):
    """report.txt 必须出现标准段 [验收档位判定（实测）] 与逐档判定行。"""
    r, out = res
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "[验收档位判定（实测）]" in text
    assert "实测达成档位: %s" % r["claimed_tier"] in text
    for name in T.LABELS.values():
        assert name in text
    assert "人工答辩项" in text                       # manual 不许冒充实测


def test_render_lines_uses_verdict_not_constants(res):
    """渲染与判定同源：--tier 与报告行内容一致（同一 verdict 对象的产物）。"""
    r, _ = res
    lines = T.render_lines(r["tier_verdict"])
    assert lines[0].startswith("实测达成档位: %s" % r["claimed_tier"])
    assert len(lines) == 1 + 4 + 1                   # 表头 + 四档 + 人工项


def test_tier_drops_when_measured_handle_rate_weakened(res):
    """反向证明（实测值削弱）：处理率压到 95% 线下，C 必须掉级到 D。"""
    r, _ = res
    m = dict(r["tier_verdict"]["measured"])
    m["evidence"] = r["tier_verdict"]["evidence"]
    m["handle_rate_pct"] = 90.0                      # 低于工单 C 线
    g = C.load_config(CFG).tier_gates()
    st = r["code_style"]
    v = T.verdict(m, st, g)
    assert not v["final"]["C"] and v["final"]["D"]
    assert v["claimed"] == "及格(D)"                  # 不许继续承诺 C 及以上


def test_tier_drops_when_quality_score_weakened(res):
    """反向证明：质量分压到 85 线下，B 掉到 C——门槛真是从实测算出来的。"""
    r, _ = res
    m = dict(r["tier_verdict"]["measured"])
    m["evidence"] = r["tier_verdict"]["evidence"]
    m["quality_after"] = 80.0                        # 低于工单 B 线
    g = C.load_config(CFG).tier_gates()
    v = T.verdict(m, r["code_style"], g)
    assert v["final"]["D"] and v["final"]["C"]
    assert not v["final"]["B"] and v["claimed"] == "良好(C)"


def test_artifacts_real_and_substantial(res):
    """真实产物存在且字节达标：报告不许是空壳。"""
    r, out = res
    for fn in ("report.txt", "result.json"):
        assert os.path.getsize(os.path.join(out, fn)) > 1000, fn
    for fn in ("missing_list.csv", "anomaly_list.csv", "duplicate_list.csv",
               "audit_log.csv", "cleaned.csv"):
        assert os.path.exists(os.path.join(out, fn)), fn
        assert os.path.getsize(os.path.join(out, fn)) > 0, fn
