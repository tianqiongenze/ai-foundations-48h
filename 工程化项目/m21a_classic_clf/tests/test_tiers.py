# -*- coding: utf-8 -*-
"""工单 M2-1a 四档验收断言——每档钉在本次真跑产生的实测值上。

档位判据与 tiers.verdict / cli --tier 同源（工单"四、验收标准"表）：
D 基线accuracy输出+规范率≥70%；C F1≥0.70口径正确；B 阈值方案+决策记录完整；
A F1≥0.85或方案论证充分（论证充分归人工，机器只认 F1 实测）。
正向断言看流水线实测；反向断言证明闸门会咬：F1 调低/决策记录缺失/
规范率不足时，承诺档位必须掉下来。
"""
import json
import os

import pytest

from classic_clf import config as C
from classic_clf import pipeline, tiers

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "config", "rules.yaml")


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    """实跑一次完整流水线（真训练真评估），模块内共用同一批实测值。"""
    out = str(tmp_path_factory.mktemp("m21a_tiers"))
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
    kw = dict(gates=r["gates"], lr_acc=m["lr_acc"], dt_acc=m["dt_acc"],
              macro_f1=m["macro_f1"], caliber_ok=m["caliber_ok"],
              sweep_min_points=m["sweep_min_points"],
              sweep_total_points=m["sweep_total_points"],
              decision_ok=m["decision_ok"], style_rate=m["style_rate"])
    kw.update(over)
    return tiers.verdict(**kw)


def test_D_baseline_accuracy_and_measured_style_pass(res):
    r, _ = res
    got = tiers_of(r["tier_verdict"])
    st = r["code_style"]
    assert st["rate"] >= r["gates"]["D"]["style_min"], (
        "D 档规范率门槛≥%.0f%%，实测 %.2f%%（引擎 %s）" % (
            r["gates"]["D"]["style_min"], st["rate"], st["engine"]))
    assert got["D"]["pass"], "基线 LR/DT accuracy 已实测输出，D 档应达标"


def test_C_macro_f1_gate_tracks_measurement(res):
    r, _ = res
    m = r["tier_verdict"]["measured"]
    got = tiers_of(r["tier_verdict"])
    assert m["caliber_ok"], "C 档口径来源（average=macro、OvR pos_label）必须实测登记"
    assert got["C"]["functional"] == (
        m["macro_f1"] >= r["gates"]["C"]["f1_min"]), "C 档只认实测 macro F1 对线"


def test_B_and_claimed_tier_are_measured(res):
    r, _ = res
    m = r["tier_verdict"]["measured"]
    v = r["tier_verdict"]
    got = tiers_of(v)
    assert m["decision_ok"], "B 档决策记录完整性由 result.json 落盘回读实测"
    assert got["B"]["pass"], "阈值曲线每类≥%d 点实测，B 档应达标" % r["gates"][
        "B"]["sweep_points_min"]
    assert got["A"]["pass"] == (m["macro_f1"] >= r["gates"]["A"]["f1_min"]), (
        "A 档不许虚标：达标与否必须与实测 macro F1 对线一致")
    highest = next((t["tier"] for t in reversed(v["tiers"]) if t["pass"]),
                   "未达 D 档")
    assert v["claimed"] == highest


def test_report_contains_measured_tier_block(res):
    _, out = res
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "[验收档位判定（实测）]" in text
    assert "实测达成档位" in text
    for name, _need in tiers.TIERS:
        assert name in text, "四档判定块缺档: %s" % name


def test_D_gate_bites_when_style_rate_low(res):
    """反向1：规范率跌到门槛下，四档全部不达标（承诺=未达 D 档）。"""
    v = _verdict(res, style_rate=res[0]["gates"]["D"]["style_min"] - 5.0)
    got = tiers_of(v)
    for k in ("D", "C", "B", "A"):
        assert not got[k]["pass"]
    assert v["claimed"] == "未达 D 档"


def test_C_gate_bites_when_f1_lowered(res):
    """反向2：把实测 macro F1 调到 0.55，C 档功能门必须翻否、承诺掉到 D。"""
    v = _verdict(res, macro_f1=0.55)
    got = tiers_of(v)
    assert not got["C"]["functional"] and not got["C"]["pass"]
    assert v["claimed"] == "及格(D)"


def test_B_gate_bites_when_decision_record_missing(res):
    """反向3：决策记录回读失败（缺 best），B 档必须掉、承诺停在 C。"""
    v = _verdict(res, decision_ok=False)
    got = tiers_of(v)
    assert not got["B"]["pass"] and got["C"]["pass"]
    assert v["claimed"] == "良好(C)"
