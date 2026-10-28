# -*- coding: utf-8 -*-
"""工单 M3-1 四档验收判定测试：正向 1 + 反向 ≥2（档位是算出来的）。

判据与 src/transfer_learning/tiers.py 同源（工单验收标准表 + rules.yaml
tiers: 段）；反向用例证明：准确率调低、缺对照表、缺决策记录、缺曲线时，
承诺档位必须下降——糊弄不过去。
"""
import json

from transfer_learning import pipeline, tiers
from transfer_learning.config import Settings


def tiers_of(v):
    """tiers.verdict 的返回 → {档位字母: 判定}（口径与 cli --tier 一致）。"""
    return {t["tier"][-2]: t for t in v["tiers"]}


FULL = dict(acc=0.87, style_rate=92.0, compare_rows=4,
            aug_gains={"frozen": 0.01, "finetune": 0.02},
            decisions=1, curves=1)


def test_full_measured_claims_top_tier():
    """实测四档条件齐备（0.87≥0.85、有表有曲线）→ 承诺 优(A)。"""
    v = tiers.verdict(**FULL)
    assert all(t["pass"] for t in v["tiers"])
    assert v["claimed"] == "优(A)"


def test_acc_dropped_claims_lower():
    """准确率 0.87→0.72：C(≥75%)/A(≥85%) 必掉；B 因两路正增益仍在。"""
    v = tiers.verdict(**dict(FULL, acc=0.72))
    got = tiers_of(v)
    assert got["D"]["pass"] and not got["C"]["pass"]
    assert not got["A"]["pass"] and v["claimed"] == "优秀(B)"


def test_no_table_rows_fails_c():
    """对照表被清空（0 行实测数字）→ 良好(C)『加对照表』必判缺。"""
    v = tiers.verdict(**dict(FULL, compare_rows=0))
    assert tiers_of(v)["C"]["pass"] is False


def test_no_decision_records_fails_b():
    """docs/ 决策记录实测 0 条 → 优秀(B)『决策记录完整』必判缺，档位回落。"""
    v = tiers.verdict(**dict(FULL, acc=0.78, decisions=0))
    got = tiers_of(v)
    assert got["B"]["pass"] is False and v["claimed"] == "良好(C)"


def test_style_below_d_floor_fails_all():
    """规范率跌破工单 D 档 70% 底线：四档累计，全部未达标。"""
    v = tiers.verdict(**dict(FULL, style_rate=60.0))
    assert not any(t["pass"] for t in v["tiers"])
    assert v["claimed"] == "未达 D 档"


def test_missing_acc_reports_miss():
    """图像通道没跑成（acc=None，增强增益也无从实测）→ 判缺不判达标。"""
    v = tiers.verdict(**dict(FULL, acc=None, aug_gains={}, curves=0,
                             miss_note="torch 缺失"))
    assert not any(t["functional"] for t in v["tiers"])
    txt = "\n".join(tiers.render_lines(v))
    assert "MISS" in txt and "未达 D 档" in txt


def test_gates_come_from_config_not_hardcode():
    """rules.yaml tiers: 段改门槛 → 判定随之变化（数字外置可复核）。"""
    v = tiers.verdict(**dict(FULL, acc=0.82), gates={"C": {"acc": 0.85}})
    assert tiers_of(v)["C"]["pass"] is False   # 门槛抬到0.85后 0.82 不达标
    assert tiers.verdict(**dict(FULL, acc=0.82))["claimed"] == "优秀(B)"


def test_pipeline_report_contains_measured_tier_section(tmp_path):
    """端到端：流水线小预算实跑一次，报告必须带四档实测判定段。"""
    cfg = Settings(seed=7, raw={"pretrain_epochs": 1, "head_epochs": 2,
                                "finetune_epochs": 2, "aug_repeats": 2})
    pipeline.run(cfg, str(tmp_path))
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "[验收档位判定（实测）]" in text
    assert "实测达成档位" in text
    for name, _need, _k in tiers.TIERS:
        assert name in text
    assert "人工答辩项" in text
    rj = tmp_path / "result.json"
    result = json.loads(rj.read_text(encoding="utf-8"))
    names = [t[0] for t in tiers.TIERS] + ["未达 D 档"]
    assert result["tiers"]["claimed"] in names
    assert result["decision_entries"] >= 1        # 决策条目真实存在才敢承诺 B
    assert result["compare_rows"] >= 2


def test_decision_entries_actually_counted():
    """count_decision_entries 数的是磁盘上的真条目：空目录=0，不吹数。"""
    import os
    assert tiers.count_decision_entries("/no/such/root") == 0
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert tiers.count_decision_entries(root) >= 1
