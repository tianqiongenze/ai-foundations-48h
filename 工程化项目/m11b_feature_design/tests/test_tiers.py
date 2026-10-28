# -*- coding: utf-8 -*-
"""工单 M1-1b 验收门禁测试：把"四、验收标准"四档写成可执行断言。

  及格(D)：5 特征表能跑通（最小集出指标）
  良好(C)：≥10 个特征且特征字典完整（五栏成文）
  优秀(B)：特征砍掉 30% 后性能不降（消融验证，红线 = rules.yaml danger）
  优  (A)：新特征挤进 Top3 且有物理解释

用真实 config/rules.yaml 全量跑一遍（约 1 分钟），断言的是**实测数字**，
不是"代码没报错"。任何一条破了，对应档位当场掉级。
"""
import json
import os

import pytest

from feature_design48 import config as C
from feature_design48 import core
from feature_design48 import pipeline as P
from feature_design48 import tiers as T

CFG = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("m11b"))
    P.run(C.load_config(CFG), out_dir=out)
    with open(os.path.join(out, "result.json"), encoding="utf-8") as f:
        return json.load(f), out


def _arm(r, sid):
    for v in r["feature_sets"].values():
        if v["id"] == sid:
            return v
    raise KeyError(sid)


# ---------------------------------------------------------------- 及格 D
def test_D_minimal_five_feature_table_runs(res):
    r, _ = res
    m = _arm(r, "min5")
    assert len(m["features"]) == 5
    assert m["n_features"] == 5 and 0.0 < m["acc"] <= 1.0
    assert r["tiers"]["D"]["达标"] is True


def test_D_top3_extreme_arm_also_runs(res):
    """Top3 极限集：证明"少而对的特征"够用——这是 D/C 档之间最重要的对照。"""
    r, _ = res
    t3 = _arm(r, "top3")
    assert t3["n_features"] == 3 and t3["acc"] > 0.9


# ---------------------------------------------------------------- 良好 C
def test_C_at_least_10_features_and_dict_complete(res):
    r, _ = res
    assert r["n_features"] >= 10
    assert r["feature_dict_rows"] == r["n_features"]
    assert r["feature_dict_missing_cells"] == []      # 五栏零缺空
    assert r["tiers"]["C"]["达标"] is True


def test_C_dict_landed_as_csv(res):
    r, out = res
    p = os.path.join(out, "feature_dict.csv")
    assert os.path.exists(p) and os.path.getsize(p) > 0
    with open(p, encoding="utf-8") as f:
        head = f.readline().strip()
    for col in ("name", "formula", "meaning", "unit", "range"):
        assert col in head


def test_C_correlation_screen_reported(res):
    r, _ = res
    assert r["correlation_threshold"] == pytest.approx(core.REDUNDANT_R, abs=1e-9)
    assert len(r["screened_features"]) <= r["n_features"]
    for p in r["correlation_pairs"]:
        assert abs(p["r"]) > r["correlation_threshold"] and p["keep"]


def test_C_freq_domain_beats_time_domain(res):
    """"好特征胜过换模型"的数字证据：同模型只换特征列。"""
    r, _ = res
    assert _arm(r, "freq")["acc"] > _arm(r, "time")["acc"] + 0.20
    assert _arm(r, "min5")["acc"] < _arm(r, "full")["acc"] - 0.30


# ---------------------------------------------------------------- 优秀 B
def test_B_ablation_cut_30pct_without_performance_loss(res):
    r, _ = res
    ab = r["ablation"]
    assert ab["cut_pct_actual"] >= 30.0                 # 真的砍够 30%
    assert ab["tail_drop"] <= r["ablation_redline"]     # 不降（红线来自 rules.yaml）
    assert ab["pass"] is True
    assert r["tiers"]["B"]["达标"] is True


def test_B_ablation_has_three_arms_for_credit(res):
    """尾部 / 随机 / 头部三臂齐备，否则"不降"没有对照，属于自说自话。"""
    r, _ = res
    ab = r["ablation"]
    assert ab["base"]["acc"] > 0 and ab["tail"]["acc"] > 0 and ab["random"]["acc"] > 0
    assert ab["head"]["acc"] < ab["base"]["acc"] - r["ablation_redline"]
    assert ab["head_collapses"] is True                 # 榜可信的反证
    assert ab["tail_better_than_random"] is True


def test_B_ablation_and_ab_use_multiseed_cv(res):
    """120 样本单跑一次不算证据：必须多种子 + 多随机组。"""
    r, _ = res
    assert len(r["seeds"]) >= 2 and r["folds"] >= 3
    assert r["ablation"]["random"]["n_sets"] >= 2
    for v in r["feature_sets"].values():
        assert "acc_spread" in v


# ---------------------------------------------------------------- 优 A
def test_A_new_feature_in_top3_with_physics_explanation(res):
    r, out = res
    chk = r["new_feature_check"]
    assert chk["n_new_in_top"] >= 1
    assert chk["pass"] is True
    assert r["tiers"]["A"]["达标"] is True
    for name, _imp in chk["new_in_top"]:
        assert chk["meanings"][name].strip()            # 物理解释成文
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        text = f.read()
    for name, _ in chk["new_in_top"]:
        assert name in text and "物理解释" in text


def test_A_ranking_is_seed_stable(res):
    """Top3 必须跨种子名次不抖，否则"A 档"只是运气。"""
    r, _ = res
    for row in r["importance_board"][:3]:
        assert row["rank_seed_max"] - row["rank_seed_min"] <= 1


def test_all_four_tiers_marked_pass(res):
    r, _ = res
    assert set(r["tiers"]) == {"D", "C", "B", "A"}
    assert all(v["达标"] for v in r["tiers"].values())
    assert all(v["要求"] and v["实测"] for v in r["tiers"].values())


def test_artifacts_written(res):
    r, out = res
    for fn in ("report.txt", "result.json", "feature_dict.csv", "importance_rank.csv",
               "correlation_screen.csv", "ab_feature_sets.csv", "ablation.csv"):
        assert os.path.exists(os.path.join(out, fn)), fn
        assert os.path.getsize(os.path.join(out, fn)) > 0, fn


def test_leakage_statement_present(res):
    r, out = res
    assert "窗口" in r["leak_check"]
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        assert "泄漏预警" in f.read()


# ------------------------------------------------ 规范率闸门 + 档位判定（纯函数）
def _fake_st(rate, min_pass=70.0):
    return {"rate": rate, "passes": rate >= min_pass, "min_pass": min_pass}


def test_D_measured_style_rate_is_at_least_70(res):
    r, out = res
    st = r["code_style"]
    assert st["rate"] >= 70.0, (
        "D 档规范率门槛≥70%%，实测 %.2f%%（引擎 %s）" % (st["rate"], st["engine"]))
    assert st["engine"] and st["n_files"] > 0 and st["code_lines"] > 0
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        assert "检查引擎" in f.read()


def test_D_gate_bites_when_style_rate_low():
    """闸门会咬：规范率实测不足 70%，即便业务全过，D 也必须判不达标。"""
    v = P.tier_verdict({"D": True, "C": True, "B": True, "A": True}, _fake_st(65.0))
    assert v["final"]["D"] is False and v["claimed"] == "未达 D 档"


def test_claimed_tier_drops_when_a_capability_removed():
    """反向证明：拿掉 A 能力，承诺档位从 A 掉到 B——档位由实测决定。"""
    v = P.tier_verdict({"D": True, "C": True, "B": True, "A": False}, _fake_st(99.0))
    assert v["final"]["B"] and not v["final"]["A"] and v["claimed"] == "优秀(B)"


def test_claimed_tier_recorded(res):
    r, _ = res
    assert r["claimed_tier"] in list(T.LABELS.values()) + ["未达 D 档"]


# ------------------------------------------- 门槛外置 + 标准段渲染 + 产物大小
def test_tier_gates_come_from_rules_yaml_and_match_workorder(res):
    """四档门槛出自 rules.yaml tiers: 段，数字与工单《M1-1b》验收表逐字一致。"""
    g = C.load_config(CFG).tier_gates()
    assert g["D"]["min_code_style_pct"] == 70.0      # 工单 D：代码规范率≥70%
    assert g["D"]["min_features"] == 5               # 工单 D：5 特征表能跑通
    assert g["C"]["min_features"] == 10              # 工单 C：≥10 特征
    assert g["C"]["dict_required_cols"] == 5         # 工单 C：字典五栏成文
    assert g["B"]["cut_frac"] == 0.30                # 工单 B：特征砍 30%
    assert g["B"]["max_acc_drop"] == 0.03            # 工单 B：性能不降红线
    assert g["A"]["top_n"] == 3                      # 工单 A：进 Top3
    assert g["A"]["min_new_in_top"] == 1             # 工单 A：≥1 个新特征
    # 与报告展示线（danger/min_features/cut_frac）不许两套口径
    r, _ = res
    assert r["ablation_redline"] == g["B"]["max_acc_drop"]
    assert r["min_features"] == g["C"]["min_features"]
    assert r["cut_frac"] == g["B"]["cut_frac"]


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


def test_tier_drops_when_measured_tail_drop_weakened(res):
    """反向证明（实测值削弱）：消融降幅压到红线之上，B 必须掉级到 C。"""
    r, _ = res
    m = dict(r["tier_verdict"]["measured"])
    m["evidence"] = r["tier_verdict"]["evidence"]
    m["tail_drop"] = 0.10                            # 高于工单 B 红线 0.03
    g = C.load_config(CFG).tier_gates()
    v = T.verdict(m, r["code_style"], g)
    assert v["final"]["D"] and v["final"]["C"]
    assert not v["final"]["B"] and v["claimed"] == "良好(C)"


def test_tier_drops_when_new_feature_not_in_top3(res):
    """反向证明：新特征退出 Top3（实测 0 个），A 必须掉级到 B。"""
    r, _ = res
    m = dict(r["tier_verdict"]["measured"])
    m["evidence"] = r["tier_verdict"]["evidence"]
    m["n_new_in_top"] = 0                            # 低于工单 A 下限 1
    g = C.load_config(CFG).tier_gates()
    v = T.verdict(m, r["code_style"], g)
    assert v["final"]["B"] and not v["final"]["A"]
    assert v["claimed"] == "优秀(B)"


def test_artifacts_real_and_substantial(res):
    """真实产物存在且字节达标：报告/结果不许是空壳。"""
    r, out = res
    for fn in ("report.txt", "result.json"):
        assert os.path.getsize(os.path.join(out, fn)) > 1000, fn
