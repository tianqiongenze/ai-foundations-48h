# -*- coding: utf-8 -*-
"""工单 M5-1 四档验收断言——每档都钉在"实测值"上，能力删掉就必须红。

档位判据与 tiers.verdict / cli --tier 同源（工单验收标准 + rules.yaml tiers:）：
D 基线同种子二次运行逐键相等+规范率≥70%；C 实测正确率≥75% 且说明书实测存在；
B 漏洞修复成对证据（修复前假派检真实发生、修复后套利被剥掉）+决策记录完整；
A 实测正确率≥90% 或真实跑过的未见种子泛化实验。
反向用例削弱任一实测值后，承诺档位必须下降——证明档位不是常量。
"""
import os

import pytest

from multiagent_sched import config as C
from multiagent_sched import core, pipeline, tiers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "rules.yaml")
OUT = os.path.join(ROOT, "output", "test_run")


@pytest.fixture(scope="module")
def built():
    """跑一次完整调度实验，模块内所有断言共用同一批实测值。"""
    sections, result = pipeline.run(C.load_config(CFG), out_dir=OUT)
    return sections, result


def tiers_of(v):
    """verdict 返回 → {档位字母: 判定}（实测与手工构造走同一口径）。"""
    return {t["tier"][-2]: t for t in v["tiers"]}


def test_gates_come_from_rules_yaml(built):
    """门槛数字外置在 rules.yaml：配置读出值与工单逐字一致。"""
    g = tiers.gates(C.load_config(CFG).tiers)
    assert float(g["D"]["style_min"]) == 70.0
    assert float(g["C"]["acc_min"]) == 0.75
    assert float(g["A"]["acc_a_min"]) == 0.90


def test_D_baseline_reproducible_by_double_run(built):
    """D 档实测：同种子两次真实运行逐键相等（可复现），规范率真测。"""
    _, r = built
    again = core.run(core.policy_random, seed=C.load_config(CFG).seed)
    assert again == r["strategies"]["随机"], "基线必须逐键可复现"
    assert r["style"]["rate"] >= 70.0


def test_C_accuracy_measured_with_manual(built):
    """C 档实测：三策略跑出的最高正确率≥75%，且运行手册实测字数≥200。"""
    _, r = built
    assert r["tiers"]["measured"]["acc_best"] >= 0.75
    assert r["manual_doc"]["exists"] and r["manual_doc"]["chars"] >= 200


def test_B_loophole_pairwise_evidence(built):
    """B 档成对证据：修复前假派检真实发生，修复后同一策略套利收益被剥掉。"""
    _, r = built
    b = r["loophole"]["before"]
    a = r["loophole"]["after"]
    assert b["wrong"] >= 1, "修复前必须实测到刷分行为"
    assert a["reward"] < b["reward"], "修复后套利必须实测消失（总奖励下降）"
    text = open(os.path.join(OUT, "report.txt"), encoding="utf-8").read()
    assert "修复前（罚差 1.0）" in text and "修复后（过度检修罚提至 8.0" in text


def test_all_four_tiers_pass_and_claim_is_highest(built):
    """四档全达标且报告落盘含验收档位判定段——档位由实测推出。"""
    _, r = built
    got = tiers_of(r["tiers"])
    for key in ("D", "C", "B", "A"):
        assert got[key]["pass"], "%s 档实测未达标: %s" % (key, got[key])
    assert r["tiers"]["claimed"] == "优(A)"
    text = open(os.path.join(OUT, "report.txt"), encoding="utf-8").read()
    assert "验收档位判定（实测）" in text and "人工答辩项" in text


def test_claim_drops_when_repro_broken(built):
    """反向证明①：基线不可复现（二次运行结果不同），D 红则全档红。"""
    _, r = built
    m = r["tiers"]["measured"]
    g = tiers.gates(C.load_config(CFG).tiers)
    down = tiers.verdict(g, rate=m["rate"], reproducible=False,
                         acc_best=m["acc_best"],
                         manual_chars=m["manual_chars"],
                         wrong_before=m["wrong_before"],
                         reward_delta=m["reward_delta"],
                         decision_ok=m["decision_ok"],
                         unseen_acc=m["unseen_acc"],
                         gen_episodes=m["gen_episodes"])
    assert down["claimed"] == "未达 D 档"


def test_claim_drops_when_arbitrage_survives(built):
    """反向证明②：漏洞修复后套利没被剥掉（Δ奖励≥0），B/A 必须红到 C。"""
    _, r = built
    m = r["tiers"]["measured"]
    g = tiers.gates(C.load_config(CFG).tiers)
    down = tiers.verdict(g, rate=m["rate"], reproducible=m["reproducible"],
                         acc_best=m["acc_best"],
                         manual_chars=m["manual_chars"],
                         wrong_before=m["wrong_before"], reward_delta=0.0,
                         decision_ok=m["decision_ok"],
                         unseen_acc=m["unseen_acc"],
                         gen_episodes=m["gen_episodes"])
    got = tiers_of(down)
    assert got["C"]["pass"] and not got["B"]["pass"] and not got["A"]["pass"]
    assert down["claimed"] == "良好(C)"


def test_claim_drops_when_generalization_vanishes(built):
    """反向证明③：正确率削弱到 0.80 且泛化实验未跑（0 集），A 必须掉档。"""
    _, r = built
    m = r["tiers"]["measured"]
    g = tiers.gates(C.load_config(CFG).tiers)
    down = tiers.verdict(g, rate=m["rate"], reproducible=m["reproducible"],
                         acc_best=0.80, manual_chars=m["manual_chars"],
                         wrong_before=m["wrong_before"],
                         reward_delta=m["reward_delta"],
                         decision_ok=m["decision_ok"],
                         unseen_acc=0.80, gen_episodes=0)
    got = tiers_of(down)
    assert got["B"]["pass"] and not got["A"]["pass"]
