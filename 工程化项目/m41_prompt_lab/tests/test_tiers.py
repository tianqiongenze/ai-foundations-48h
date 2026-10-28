# -*- coding: utf-8 -*-
"""工单 M4-1 四档验收断言——每档都钉在"实测值"上，能力删掉就必须红。

档位判据与 tiers.verdict / cli --tier 同源（工单验收标准 + rules.yaml tiers:）：
D 能出回答+规范率≥70%；C 实测提升≥20% 且有对比表；
B 断服注入实测命中降级话术+决策记录（docs/ADR 反读）完整；
A 实测提升≥35% 或多轮对话实测上下文保持。
反向用例削弱任一实测值后，承诺档位必须下降——证明档位不是常量。
"""
import os

import pytest

from prompt_lab import config as C
from prompt_lab import pipeline, tiers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "rules.yaml")
OUT = os.path.join(ROOT, "output", "test_run")


@pytest.fixture(scope="module")
def built():
    """跑一次完整流水线，模块内所有断言共用同一批实测值。"""
    sections, result = pipeline.run(C.load_config(CFG), out_dir=OUT)
    return sections, result


def tiers_of(v):
    """verdict 返回 → {档位字母: 判定}（实测与手工构造走同一口径）。"""
    return {t["tier"][-2]: t for t in v["tiers"]}


def test_gates_come_from_rules_yaml(built):
    """门槛数字外置在 rules.yaml：配置读出的 style_min/lift 与工单一致。"""
    cfg = C.load_config(CFG)
    g = tiers.gates(cfg.tiers)
    assert float(g["D"]["style_min"]) == 70.0
    assert float(g["C"]["lift_min"]) == 20.0
    assert float(g["A"]["lift_a_min"]) == 35.0


def test_D_answers_are_real_and_style_rate_measured(built):
    """D 档实测：三版应答非空且规范率实测≥70%。"""
    _, r = built
    assert all(v["answer"] for v in r["results"].values()), "三版应答须非空"
    assert r["style"]["rate"] >= 70.0, (
        "工单 D 档规范率≥70%%，实测 %.1f%%（引擎 %s）"
        % (r["style"]["rate"], r["style"]["engine"]))


def test_C_measured_lift_at_least_20_with_table(built):
    """C 档实测：两次真实运行分数相减的提升≥20% 且对比表≥3 行。"""
    _, r = built
    assert r["lift_pct"] >= 20.0, "实测提升来自两次真实运行相减，须≥20%"
    assert len(r["results"]) >= 3, "对比表须有≥3 行（三版同题集）"


def test_B_outage_injection_hits_fallback_text(built):
    """B 档降级实测：注入断服后确实走了降级话术，且决策记录文档反读齐全。"""
    _, r = built
    assert r["tiers"]["measured"]["fallback_hit"], "降级话术须实测命中"
    assert r["dead_stats"]["fallbacks"] == 1
    assert r["decision_record"]["n_sections"] == 4


def test_A_lift_or_multiturn_measured(built):
    """A 档实测：提升≥35% 或多轮≥2 轮且上下文保持，二者必居其一。"""
    _, r = built
    mt = r["multiturn"]
    assert (r["lift_pct"] >= 35.0
            or (mt["completed"] >= 2 and mt["context_retained"]))


def test_all_four_tiers_pass_and_claim_is_highest(built):
    """四档全达标且报告落盘含验收档位判定段——档位由实测推出。"""
    _, r = built
    got = tiers_of(r["tiers"])
    for key in ("D", "C", "B", "A"):
        assert got[key]["pass"], "%s 档实测未达标: %s" % (key, got[key])
    assert r["tiers"]["claimed"] == "优(A)"
    report = os.path.join(OUT, "report.txt")
    text = open(report, encoding="utf-8").read()
    assert "验收档位判定（实测）" in text and "实测达成档位" in text


def test_claim_drops_when_lift_weakened(built):
    """反向证明①：实测提升削弱到 15%（<20），C 及以上档必须全红。"""
    _, r = built
    m = r["tiers"]["measured"]
    down = tiers.verdict(tiers.gates(C.load_config(CFG).tiers),
                         rate=m["rate"], answers_ok=m["answers_ok"],
                         table_rows=m["table_rows"], lift_pct=15.0,
                         fallback_hit=m["fallback_hit"],
                         decision_ok=m["decision_ok"],
                         n_turns=0, context_retained=False)
    got = tiers_of(down)
    assert got["D"]["pass"] and not got["C"]["pass"]
    assert down["claimed"] == "及格(D)"


def test_claim_drops_when_fallback_missing(built):
    """反向证明②：降级话术没命中（断服直接崩），B/A 必须红、档位跌到 C。"""
    _, r = built
    m = r["tiers"]["measured"]
    g = tiers.gates(C.load_config(CFG).tiers)
    down = tiers.verdict(g, rate=m["rate"], answers_ok=m["answers_ok"],
                         table_rows=m["table_rows"], lift_pct=m["lift_pct"],
                         fallback_hit=False, decision_ok=m["decision_ok"],
                         n_turns=m["n_turns"],
                         context_retained=m["context_retained"])
    got = tiers_of(down)
    assert got["C"]["pass"] and not got["B"]["pass"] and not got["A"]["pass"]


def test_claim_drops_when_style_rate_low(built):
    """反向证明③：规范率掉到 69%，连 D 都不达标——档位不是常量。"""
    _, r = built
    m = r["tiers"]["measured"]
    g = tiers.gates(C.load_config(CFG).tiers)
    down = tiers.verdict(g, rate=69.0, answers_ok=m["answers_ok"],
                         table_rows=m["table_rows"], lift_pct=m["lift_pct"],
                         fallback_hit=m["fallback_hit"],
                         decision_ok=m["decision_ok"],
                         n_turns=m["n_turns"],
                         context_retained=m["context_retained"])
    assert down["claimed"] == "未达 D 档"
