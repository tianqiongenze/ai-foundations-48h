# -*- coding: utf-8 -*-
"""工单 M3-2 四档验收判定测试：正向 1 + 反向 ≥4（档位是算出来的）。

判据与 src/yolo_detect/tiers.py 同源（工单"四、验收标准"表 + rules.yaml
tiers: 段）；反向用例证明：mAP 调低、扫描表残缺、理由不挂实测数字、
决策记录清空、规范率跌破底线时，承诺档位必须下降——糊弄不过去。
"""
import json

from yolo_detect import pipeline, tiers
from yolo_detect.config import Settings


def tiers_of(v):
    """tiers.verdict 的返回 → {档位字母: 判定}（口径与报告渲染一致）。"""
    return {t["tier"][-2]: t for t in v["tiers"]}


FULL = dict(ap=0.8272, style_rate=92.0, sweep_rows=3, sweep_cells=30,
            sweep_cells_expected=30, rec_ap=0.9130,
            reason="漏检代价高→conf 取网格实测 AP 最高的 0.55（0.9130）",
            speedup=182.9, decisions=1)


def test_full_measured_claims_top_tier():
    """实测四档条件齐备（AP≥0.75 且表完整且有推荐点/决策记录）→ 承诺 优(A)。"""
    v = tiers.verdict(**FULL)
    assert all(t["pass"] for t in v["tiers"])
    assert v["claimed"] == "优(A)"


def test_ap_and_speedup_both_low_fails_a():
    """AP 0.8272→0.55 且加速 182.9%→10%：A『mAP≥0.75或加速≥50%』两条都不成立。"""
    v = tiers.verdict(**dict(FULL, ap=0.55, speedup=10.0))
    got = tiers_of(v)
    assert got["D"]["pass"] and not got["C"]["pass"]
    assert got["B"]["pass"] and not got["A"]["pass"]
    assert v["claimed"] == "优秀(B)"


def test_incomplete_sweep_table_fails_c():
    """扫描表格子数不足（12/30 格）→ 良好(C)『表完整』必判缺。"""
    v = tiers.verdict(**dict(FULL, sweep_cells=12))
    assert tiers_of(v)["C"]["pass"] is False


def test_reason_without_measured_number_fails_b():
    """推荐理由里一个实测数字都没有 → 优秀(B)『给出推荐点及理由』判缺。

    工单 M3-2 各档写的是各自条件（与 m31 同口径），故 B 判缺不牵动 A：
    报告里 B 行会明确显示"功能缺"，承诺档位仍按实测最高档给。
    """
    v = tiers.verdict(**dict(FULL, reason="凭经验把置信度调高一点"))
    got = tiers_of(v)
    assert got["B"]["pass"] is False and got["A"]["pass"] is True


def test_no_decision_records_fails_b():
    """docs/ 决策记录实测 0 条 → 优秀(B)『决策记录完整』必判缺。"""
    v = tiers.verdict(**dict(FULL, decisions=0))
    assert tiers_of(v)["B"]["pass"] is False


def test_style_below_d_floor_fails_all():
    """规范率跌破工单 D 档 70% 底线：档位递进，四档全部未达标。"""
    v = tiers.verdict(**dict(FULL, style_rate=60.0))
    assert not any(t["pass"] for t in v["tiers"])
    assert v["claimed"] == "未达 D 档"


def test_unmeasured_reports_miss_not_pass():
    """基线 mAP 没算出（无 GT）且加速也没跑够 → 一律判缺，绝不把没测报成达标。"""
    v = tiers.verdict(**dict(FULL, ap=None, rec_ap=None, sweep_cells=0,
                             speedup=0.0, miss_note="无 GT 可评"))
    assert not any(t["functional"] for t in v["tiers"])
    txt = "\n".join(tiers.render_lines(v))
    assert "MISS" in txt and "未达 D 档" in txt


def test_gates_come_from_config_not_hardcode():
    """rules.yaml tiers: 段改门槛 → 判定随之变化（数字外置可复核）。"""
    v = tiers.verdict(**FULL, gates={"C": {"ap": 0.90}})
    assert tiers_of(v)["C"]["pass"] is False      # 门槛抬到 0.90 后 0.8272 不达标
    assert tiers.verdict(**FULL)["claimed"] == "优(A)"


def test_pipeline_report_contains_measured_tier_section(tmp_path):
    """端到端：流水线实跑一次，报告必须带四档实测判定段与规范率段。"""
    pipeline.run(Settings(seed=7), str(tmp_path))
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "[验收档位判定（实测）]" in text
    assert "实测达成档位" in text
    for name, _need, _k in tiers.TIERS:
        assert name in text
    assert "人工答辩项" in text
    assert "代码规范率" in text
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    names = [t[0] for t in tiers.TIERS] + ["未达 D 档"]
    assert result["tiers"]["claimed"] in names
    assert result["style_rate"] is not None       # 规范率是真测出来的


def test_decision_entries_actually_counted():
    """count_decision_entries 数的是磁盘上的真条目：空目录=0，不吹数。"""
    import os
    assert tiers.count_decision_entries("/no/such/root") == 0
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert tiers.count_decision_entries(root) >= 1
