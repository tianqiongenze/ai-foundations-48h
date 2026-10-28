# -*- coding: utf-8 -*-
"""工单 M4-2 四档验收断言——每档都钉在"实测值"上，能力删掉就必须红。

档位判据与 tiers.verdict / cli --tier 同源（工单验收标准 + rules.yaml tiers:）：
D 幻觉案例≥2 条+规范率≥70%；C 检索≥80% 且引用覆盖率实测达标；
B 检索≥85% 且引用可溯源（doc+段落实测解析）+决策记录完整；
A 检索≥90% 且报告内融合说明段落实测存在（≥2 行）。
反向用例削弱任一实测值后，承诺档位必须下降——证明档位不是常量。
"""
import os

import pytest

from rag48 import config as C
from rag48 import core, pipeline, tiers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "rules.yaml")
OUT = os.path.join(ROOT, "output", "test_run")


@pytest.fixture(scope="module")
def built():
    """跑一次完整问答评测流水线，模块内所有断言共用同一批实测值。"""
    sections, result = pipeline.run(C.load_config(CFG), out_dir=OUT)
    return sections, result


def tiers_of(v):
    """verdict 返回 → {档位字母: 判定}（实测与手工构造走同一口径）。"""
    return {t["tier"][-2]: t for t in v["tiers"]}


def test_gates_come_from_rules_yaml(built):
    """门槛数字外置在 rules.yaml：配置读出值与工单逐字一致。"""
    cfg = C.load_config(CFG)
    g = tiers.gates(cfg.tiers)
    assert float(g["D"]["style_min"]) == 70.0
    assert float(g["C"]["retrieve_min"]) == 0.80
    assert float(g["B"]["retrieve_min"]) == 0.85
    assert float(g["A"]["retrieve_min"]) == 0.90


def test_D_halluc_cases_are_real_productions(built):
    """D 档实测：no-RAG 基线真实跑出的幻觉案例≥2 条，且规范率实测≥70%。"""
    _, r = built
    assert r["contrast"]["no_rag_hallucination"] >= 2, "幻觉案例须真实产出计数"
    assert r["style"]["rate"] >= 70.0, (
        "工单 D 档规范率≥70%%，实测 %.1f%%（引擎 %s）"
        % (r["style"]["rate"], r["style"]["engine"]))


def test_C_recall_and_citation_measured(built):
    """C 档实测：RRF 融合通道 recall@3 真跑，每条库内答案带引用标注。"""
    _, r = built
    assert r["recall@3"]["rrf_fused"] >= 0.80
    assert r["contrast"]["rag_citations"] == r["recall@3"]["n"], "引用覆盖须 100%"


def test_B_traceable_to_doc_and_paragraph(built):
    """B 档实测：每条引用 id 都能解析回知识库 category(文档)+text(段落)。"""
    _, r = built
    kb, _ = core.load_kb()
    cites = []
    for row in r["rows"]:
        cites.extend(row["citations"])
    assert cites and tiers.traceable_rate(cites, kb) == 1.0
    assert r["decision_record"]["n_sections"] == 4


def test_all_four_tiers_pass_and_claim_is_highest(built):
    """四档全达标且报告落盘含验收档位判定段——档位由实测推出。"""
    _, r = built
    got = tiers_of(r["tiers"])
    for key in ("D", "C", "B", "A"):
        assert got[key]["pass"], "%s 档实测未达标: %s" % (key, got[key])
    assert r["tiers"]["claimed"] == "优(A)"
    text = open(os.path.join(OUT, "report.txt"),
                encoding="utf-8").read()
    assert "验收档位判定（实测）" in text and "实测达成档位" in text
    assert "人工答辩项" in text


def _measured(r):
    """把实测值喂回 verdict（可逐个削弱）——与 pipeline 同一判据函数。"""
    m = r["tiers"]["measured"]
    return tiers.verdict(tiers.gates(C.load_config(CFG).tiers),
                         rate=m["rate"], halluc_cases=m["halluc_cases"],
                         retrieve_rate=m["retrieve_rate"],
                         cite_cov=m["cite_cov"], trace_rate=m["trace_rate"],
                         fusion_lines=m["fusion_lines"],
                         decision_ok=m["decision_ok"])


def test_claim_drops_when_halluc_cases_short(built):
    """反向证明①：幻觉案例只有 1 条（<2），D 档必须红、档位直接判未达 D。"""
    _, r = built
    m = dict(r["tiers"]["measured"])
    m["halluc_cases"] = 1
    v = tiers.verdict(tiers.gates(C.load_config(CFG).tiers),
                      rate=m["rate"], halluc_cases=1,
                      retrieve_rate=m["retrieve_rate"],
                      cite_cov=m["cite_cov"], trace_rate=m["trace_rate"],
                      fusion_lines=m["fusion_lines"],
                      decision_ok=m["decision_ok"])
    assert v["claimed"] == "未达 D 档"


def test_claim_drops_when_retrieval_weakened(built):
    """反向证明②：检索命中率削弱到 86%，B 达标但 A（≥90%）必须红。"""
    v = _measured(built[1])
    m = built[1]["tiers"]["measured"]
    g = tiers.gates(C.load_config(CFG).tiers)
    down = tiers.verdict(g, rate=m["rate"], halluc_cases=m["halluc_cases"],
                         retrieve_rate=0.86, cite_cov=m["cite_cov"],
                         trace_rate=m["trace_rate"],
                         fusion_lines=m["fusion_lines"],
                         decision_ok=m["decision_ok"])
    got = tiers_of(down)
    assert v["claimed"] == "优(A)"          # 基线实测本身达 A
    assert got["B"]["pass"] and not got["A"]["pass"]
    assert down["claimed"] == "优秀(B)"


def test_claim_drops_when_fusion_note_missing(built):
    """反向证明③：报告里没有任何融合说明行（0 行），A 必须掉档到 B。"""
    _, r = built
    m = r["tiers"]["measured"]
    g = tiers.gates(C.load_config(CFG).tiers)
    down = tiers.verdict(g, rate=m["rate"], halluc_cases=m["halluc_cases"],
                         retrieve_rate=m["retrieve_rate"],
                         cite_cov=m["cite_cov"], trace_rate=m["trace_rate"],
                         fusion_lines=0, decision_ok=m["decision_ok"])
    got = tiers_of(down)
    assert got["B"]["pass"] and not got["A"]["pass"]
