# -*- coding: utf-8 -*-
"""工单 M2-3a 四档验收门禁测试：报告里每一句"达标"背后必须有实测值。

  及格(D) 模板可运行出指标 + 代码规范率≥70%（pycodestyle 实量）
  良好(C) 准确率≥75% 且图完整（PNG 实测字节 ≥ 门槛）
  优秀(B) gap 收窄且指标不降（Dropout 扫描臂实测，严格口径不放容差）
  优  (A) 找到性价比最优结构（结构搜索实测 ≥2 候选 + 机器选型）

正测试用流水线**真跑一遍**的结果断言档位与实测一致；负测试故意把入参
改坏——规范率不达标必须连 D 都不给、PNG 拿掉掉回 D、各臂指标全掉/记录
缺失掉回 C、结构候选清零掉回 B——证明"闸门会咬"。
"""
import json
import os

import pytest

from mlp_clf import config as C
from mlp_clf import pipeline as P
from mlp_clf import tiers as T

CFG = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    """模块级真跑一遍流水线：返回 (result.json 内容, 输出目录)。"""
    out = str(tmp_path_factory.mktemp("m23a"))
    P.run(C.load_config(CFG), out_dir=out)
    with open(os.path.join(out, "result.json"), encoding="utf-8") as f:
        return json.load(f), out


def tiers_of(v):
    """{档位键 D/C/B/A: 该档判定行}，省去测试里到处反查 tier 名。"""
    return {t["tier"][-2]: t for t in v["tiers"]}


def _verdict(res, **over):
    """以流水线实测值为底，替换个别字段重算判定——负测试专用。"""
    m = res[0]["tier_verdict"]["measured"]
    base = dict(gates=res[0]["gates"], metrics_ok=m["metrics_ok"],
                acc_best=m["acc_best"], png_bytes=m["png_bytes"],
                gap_base=m["gap_base"], acc_base=m["acc_base"],
                arms=m["arms"], ss_candidates=m["ss_candidates"],
                best_value_ok=m["best_value_ok"],
                decision_ok=m["decision_ok"], style_rate=m["style_rate"])
    base.update(over)
    return T.verdict(**base)


# ---------- 正测试：档位跟着实测走 ----------

def test_D_metrics_and_style_are_measured(res):
    r, _ = res
    v = r["tier_verdict"]
    assert v["measured"]["metrics_ok"] is True
    assert 0.0 <= r["acc"] <= 1.0 and r["n_epochs"] > 0
    st = r["code_style"]
    assert st["engine"] and st["n_files"] > 0 and st["code_lines"] > 0
    assert float(st["rate"]) >= float(r["gates"]["D"]["style_min"])
    assert tiers_of(v)["D"]["pass"] is True


def test_C_tracks_accuracy_and_real_png_bytes(res):
    r, out = res
    v = r["tier_verdict"]
    m = v["measured"]
    need = (float(m["acc_best"]) >= float(r["gates"]["C"]["acc_min"])
            and int(m["png_bytes"]) >= int(r["gates"]["C"]["png_min_bytes"]))
    assert tiers_of(v)["C"]["functional"] == need
    disk = os.path.getsize(os.path.join(out, "learning_curve.png"))
    assert disk == int(m["png_bytes"])          # 判定吃的字节=磁盘真实字节
    assert disk >= int(r["gates"]["C"]["png_min_bytes"])


def test_B_compliant_arm_is_strict(res):
    r, _ = res
    m = r["tier_verdict"]["measured"]
    assert m["decision_ok"] is True
    assert all(a.get("engine") for a in m["arms"])
    if m["best_arm_dropout"] is not None:
        assert float(m["gap_best"]) < float(m["gap_base"])
        assert float(m["acc_best_arm"]) >= float(m["acc_base"])
        assert tiers_of(r["tier_verdict"])["B"]["pass"] is True
    else:
        assert tiers_of(r["tier_verdict"])["B"]["functional"] is False


def test_claimed_is_highest_passing_and_report_has_block(res):
    r, out = res
    v = r["tier_verdict"]
    t = tiers_of(v)
    claimed = next((t[k]["tier"] for k in reversed(T.KEYS) if t[k]["pass"]),
                   "未达 D 档")
    assert v["claimed"] == claimed
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    assert "[验收档位判定（实测）]" in text
    assert "实测达成档位: %s" % v["claimed"] in text
    for name, _need in T.TIERS:
        assert name in text


# ---------- 负测试：闸门会咬 ----------

def test_negative_style_below_gate_drops_to_none(res):
    v = _verdict(res, style_rate=float(res[0]["gates"]["D"]["style_min"]) - 5)
    assert v["claimed"] == "未达 D 档"


def test_negative_png_missing_drops_to_D(res):
    v = _verdict(res, png_bytes=0)
    assert tiers_of(v)["C"]["functional"] is False
    assert v["claimed"] == "及格(D)"


def test_negative_all_arms_drop_acc_drops_to_C(res):
    arms = [dict(a, acc=0.0) for a in
            res[0]["tier_verdict"]["measured"]["arms"]]
    v = _verdict(res, arms=arms)
    assert v["measured"]["best_arm_dropout"] is None
    assert v["claimed"] == "良好(C)"


def test_negative_decision_record_missing_drops_to_C(res):
    v = _verdict(res, decision_ok=False)
    assert tiers_of(v)["B"]["functional"] is False
    assert v["claimed"] == "良好(C)"


def test_negative_no_structure_candidates_drops_to_B(res):
    v = _verdict(res, ss_candidates=0, best_value_ok=False)
    assert v["claimed"] == "优秀(B)"


def test_negative_gap_not_narrowing_drops_to_C(res):
    arms = [dict(a, gap=float(res[0]["tier_verdict"]["measured"]["gap_base"])
                 + 0.5)
            for a in res[0]["tier_verdict"]["measured"]["arms"]]
    v = _verdict(res, arms=arms)
    assert v["measured"]["best_arm_dropout"] is None
    assert v["claimed"] == "良好(C)"
