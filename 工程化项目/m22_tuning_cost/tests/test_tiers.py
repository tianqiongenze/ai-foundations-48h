# -*- coding: utf-8 -*-
"""工单 M2-2 四档验收断言——每档钉在本次真跑产生的实测值上。

档位判据与 tiers.verdict / cli --tier 同源（工单"四、验收标准"表）：
D 基线可复现（同种子两次重跑逐位一致）+规范率≥70%；C F1≥0.72 且
perf_counter 墙钟≤2h；B 曲线≥4 档实测+程序化拐点+决策记录回读完整；
A F1≥0.88 或快速方案机器判据成立。反向断言证明闸门会咬：F1 调低、
拐点缺失、墙钟超时、不可复现时承诺档位必须下降。
"""
import json
import os

import pytest

from tuning_cost import config as C
from tuning_cost import pipeline, tiers

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "config", "rules.yaml")


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    """实跑一次完整调参流水线（真网格搜索+真学习曲线），共用同一批实测值。"""
    out = str(tmp_path_factory.mktemp("m22_tiers"))
    pipeline.run(C.load_config(CFG), out_dir=out)
    with open(os.path.join(out, "result.json"), encoding="utf-8") as f:
        return json.load(f), out


def tiers_of(v):
    """tiers.verdict 的返回 → {档位字母: 判定}（实测与手工构造同一口径）。"""
    return {t["tier"][-2]: t for t in v["tiers"]}


def _verdict(res, **over):
    """以实测值为底、按 over 覆盖单项——反向断言与正向实测走同一判据。"""
    r = res[0]
    m = r["tier_verdict"]["measured"]
    kw = dict(gates=r["gates"], reproducible=m["reproducible"],
              grid_f1=m["grid_f1"], random_f1=m["random_f1"],
              boost_f1=m["boost_f1"], run_sec=m["run_sec"],
              curve_points=m["curve_points"], knee_found=m["knee_found"],
              decision_ok=m["decision_ok"], style_rate=m["style_rate"])
    kw.update(over)
    return tiers.verdict(**kw)


def test_D_reproducible_and_measured_style_pass(res):
    r, _ = res
    st = r["code_style"]
    assert st["rate"] >= r["gates"]["D"]["style_min"], (
        "D 档规范率门槛≥%.0f%%，实测 %.2f%%（引擎 %s）" % (
            r["gates"]["D"]["style_min"], st["rate"], st["engine"]))
    assert r["reproducible"], "同种子两次重跑的最优参数与 F1 必须逐位一致"
    assert tiers_of(r["tier_verdict"])["D"]["pass"], "D 档应达标"


def test_C_f1_and_time_limit_are_measured(res):
    r, _ = res
    m = r["tier_verdict"]["measured"]
    assert m["grid_f1"] >= r["gates"]["C"]["f1_min"], "grid F1 实测须过 0.72 线"
    assert m["run_sec"] <= float(r["gates"]["C"]["limit_sec"]), (
        "perf_counter 墙钟实测 %.3fs 须 ≤ 工单限时 %ds" % (
            m["run_sec"], int(r["gates"]["C"]["limit_sec"])))
    assert tiers_of(r["tier_verdict"])["C"]["pass"]


def test_B_curve_knee_and_decision_record(res):
    r, _ = res
    m = r["tier_verdict"]["measured"]
    assert m["curve_points"] >= int(r["gates"]["B"]["curve_points_min"])
    assert m["knee_found"], "拐点（相邻增益跌破阈值）必须由程序化扫描实测"
    assert m["decision_ok"], "决策记录完整性由 result.json 落盘回读实测"
    assert m["decision_ok"] and tiers_of(r["tier_verdict"])["B"]["pass"]


def test_A_gate_tracks_measurement_not_faking(res):
    """A 档不许虚标：达标=实测 F1 过 0.88 或快速方案机器判据成立。"""
    r, _ = res
    v = r["tier_verdict"]
    got = tiers_of(v)
    assert got["A"]["functional"] == (
        v["measured"]["fast_ok"] or
        v["measured"]["grid_f1"] >= r["gates"]["A"]["f1_min"])
    highest = next((t["tier"] for t in reversed(v["tiers"]) if t["pass"]),
                   "未达 D 档")
    assert v["claimed"] == highest


def test_xgboost_miss_is_marked_honestly(res):
    r, _ = res
    boost = r["boost_arm"]
    if boost["fallback"]:
        assert "xgboost MISS → 兜底引擎" in r["tier_verdict"]["summary"]
        assert "离线兜底" in boost["note"], "缺库必须明写，不许谎称跑了 XGBoost"


def test_report_contains_measured_tier_block(res):
    _, out = res
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "[验收档位判定（实测）]" in text and "实测达成档位" in text
    for name, _need in tiers.TIERS:
        assert name in text, "四档判定块缺档: %s" % name


def test_C_gate_bites_when_f1_lowered(res):
    """反向1：grid F1 跌到 0.55，C 必须不达标、承诺掉到 D。"""
    v = _verdict(res, grid_f1=0.55, random_f1=0.55, boost_f1=0.55)
    got = tiers_of(v)
    assert not got["C"]["pass"] and v["claimed"] == "及格(D)"


def test_C_gate_bites_when_over_time_limit(res):
    """反向2：墙钟实测超过 2h 限时，F1 再高 C 也不算限时内。"""
    v = _verdict(res, run_sec=float(res[0]["gates"]["C"]["limit_sec"]) + 1)
    got = tiers_of(v)
    assert not got["C"]["functional"] and v["claimed"] == "及格(D)"


def test_B_gate_bites_when_knee_missing(res):
    """反向3：拐点未收敛（knee=None），B 必须掉、承诺停在 C。"""
    v = _verdict(res, knee_found=False)
    got = tiers_of(v)
    assert not got["B"]["pass"] and got["C"]["pass"]
    assert v["claimed"] == "良好(C)"


def test_D_gate_bites_when_not_reproducible(res):
    """反向4：同种子重跑漂移，D 直接不达标、承诺=未达 D 档。"""
    v = _verdict(res, reproducible=False)
    assert v["claimed"] == "未达 D 档"
