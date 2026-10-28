# -*- coding: utf-8 -*-
"""工单 Z 四档验收断言——可实测部分钉在实测值上，人工项只列出不冒充。

档位判据与 tiers.verdict / cli --tier 同源（工单验收标准 + rules.yaml tiers:）：
D 规范率实测≥70%（评委放行=人工项）；C 证据链实测 5/5 且基本问题实测全命中；
B 决策记录反读完整（评委评分=人工项）；A 本地 HTTP 部署演示真实跑通。
反向用例削弱某实测值后，承诺档位必须下降——证明档位不是常量。
"""
import os

import pytest

from defense_pack import config as C
from defense_pack import core, pipeline, tiers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "rules.yaml")
OUT = os.path.join(ROOT, "output", "test_run")


@pytest.fixture(scope="module")
def built():
    """跑一次完整答辩自检（含部署演示），模块内共用同一批实测值。"""
    sections, result = pipeline.run(C.load_config(CFG), out_dir=OUT)
    return sections, result


def tiers_of(v):
    """verdict 返回 → {档位字母: 判定}（实测与手工构造走同一口径）。"""
    return {t["tier"][-2]: t for t in v["tiers"]}


def _down(r, **over):
    """拿本次实测值重跑 verdict，只削弱指定实测输入——档位必须跟着掉。"""
    m = r["tiers"]["measured"]
    args = {"rate": m["rate"], "files_scanned": m["files_scanned"],
            "chain_ready": m["chain_ready"], "chain_total": m["chain_total"],
            "qa_hit": m["qa_hit"], "qa_total": m["qa_total"],
            "decision_ok": m["decision_ok"], "demo_ok": m["demo_ok"]}
    args.update(over)
    return tiers.verdict(tiers.gates(C.load_config(CFG).tiers), **args)


def test_gates_come_from_rules_yaml(built):
    """门槛数字外置在 rules.yaml：配置读出值与工单逐字一致。"""
    g = tiers.gates(C.load_config(CFG).tiers)
    assert float(g["D"]["style_min"]) == 70.0
    assert int(g["C"]["chain_layers_min"]) == 5
    assert int(g["C"]["qa_min"]) == 3


def test_C_chain_measured_five_of_five(built):
    """C 档实测：证据链逐层反读必须真实测到 5/5（framework_dir 不许退回 0/5）。"""
    _, r = built
    assert r["ready"] == r["total"] == 5, (
        "档案袋定位实测仅 %d/%d，缺: %s"
        % (r["ready"], r["total"], r["missing"]))


def test_C_qa_bank_all_hit_measured(built):
    """基本问题全答实测：题库逐题生成回答并核对概念词，命中数逐题计。"""
    _, r = built
    assert r["qa"]["hit"] == r["qa"]["total"] == 3
    assert all(d["hit"] for d in r["qa"]["detail"])


def test_A_deploy_demo_actually_served(built):
    """A 档部署演示实测：本地 HTTP 真起服务，GET 状态码/字节数与磁盘一致。"""
    _, r = built
    demo = r["deploy_demo"]
    assert demo["ok"] and demo["status"] == 200 and demo["port"] > 0
    assert demo["bytes"] == demo["file_bytes"] > 0


def test_D_style_rate_measured_not_self_claimed(built):
    """D 档实测：规范率由 pycodestyle 真扫出来，不是常量。"""
    _, r = built
    assert r["style"]["code_lines"] > 300
    assert r["style"]["rate"] >= 70.0


def test_all_measurable_tiers_pass_and_claim_is_highest(built):
    """四档可实测部分全绿 → 承诺优(A)；报告含判定段与人工项清单。"""
    _, r = built
    got = tiers_of(r["tiers"])
    for key in ("D", "C", "B", "A"):
        assert got[key]["pass"], "%s 档实测未达标: %s" % (key, got[key])
    assert r["tiers"]["claimed"] == "优(A)"
    text = open(os.path.join(OUT, "report.txt"), encoding="utf-8").read()
    assert "验收档位判定（实测）" in text and "人工答辩项 3 条" in text


def test_claim_drops_when_chain_broken(built):
    """反向证明①：证据链实测掉到 4/5（缺一层），C 及以上必须红到 D。"""
    _, r = built
    down = _down(r, chain_ready=4)
    got = tiers_of(down)
    assert got["D"]["pass"] and not got["C"]["pass"]
    assert down["claimed"] == "及格(D)"


def test_claim_drops_when_demo_fails(built):
    """反向证明②：部署演示未跑通，A 必须红、档位跌到 B。"""
    _, r = built
    down = _down(r, demo_ok=False)
    got = tiers_of(down)
    assert got["B"]["pass"] and not got["A"]["pass"]
    assert down["claimed"] == "优秀(B)"


def test_claim_drops_when_style_rate_low(built):
    """反向证明③：规范率掉到 69%，连 D 都不达标——档位不是常量。"""
    _, r = built
    down = _down(r, rate=69.0)
    assert down["claimed"] == "未达 D 档"


def test_qa_answer_depends_on_chain(built):
    """回答器与证据链耦合：档案袋缺层时②题答不出该层、命中实测下降。"""
    rows = [(lay, i < 4, t) for i, (lay, _f, t) in enumerate(core.LAYERS)]
    hit, total, detail = core.qa_hits(rows, 4, 5)
    assert total == len(core.QA_BANK) == 3
    assert hit == 2 and not detail[1]["hit"]


def test_human_items_are_not_auto_claimed(built):
    """人工答辩项不得被机器判成"已达标"：D/B/A 行必须标人工待确认，
    且报告首行写明"仅机器可判部分，未经答辩确认不得据此宣称"。"""
    _, r = built
    got = tiers_of(r["tiers"])
    for key in ("D", "B", "A"):
        assert got[key]["human_pending"], "%s 档含人工项却未标注" % key
    assert not got["C"]["human_pending"]
    text = open(os.path.join(OUT, "report.txt"), encoding="utf-8").read()
    assert "仅机器可判部分，未经答辩确认不得据此宣称" in text
    assert "功能机器OK｜人工待确认" in text
