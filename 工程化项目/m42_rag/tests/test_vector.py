# -*- coding: utf-8 -*-
"""向量检索通道/OOD/有库无库对照测试——删掉 jieba+TF-IDF 检索或 OOD 能力时必失败。"""
from rag48 import core, pipeline
from rag48.config import Settings

KB, _ = core.load_kb()
RET = core.TfidfRetriever(KB)


def test_channel_label_is_honest():
    assert "TF-IDF" in RET.channel
    # 向量库通道二选一：chromadb 实跑（note为空）或如实写"离线兜底"，不许沉默
    assert RET.store_note == "" or "离线兜底" in RET.store_note


def test_tfidf_recall_at_least_good_tier():
    hit, n = core.recall_at_k(RET, core.EVAL_IN, topk=3)
    assert n >= 8
    assert hit / n >= 0.80  # 工单良好(C)线：检索≥80%


def test_ood_questions_exist_and_all_refused():
    assert len(core.EVAL_OOD) >= 2  # 工单及格线：幻觉（超纲）案例≥2
    thr = core.pick_refuse_threshold(RET, core.EVAL_IN, core.EVAL_OOD)
    for q in core.EVAL_OOD:
        a = core.answer_rag(q, RET, min_score=thr["threshold"])
        assert not a["grounded"] and "查无依据" in a["text"]


def test_refuse_threshold_measured_from_two_clusters():
    thr = core.pick_refuse_threshold(RET, core.EVAL_IN, core.EVAL_OOD)
    assert thr["separable"] and thr["ood_max"] < thr["threshold"] \
        < thr["in_min"]


def test_rrf_fuse_prefers_consensus():
    fused = core.rrf_fuse([["A", "B", "C"], ["B", "A", "D"]])
    assert set(fused[:2]) == {"A", "B"}


def test_no_rag_baseline_hallucinates_by_construction():
    # 无库基线的"有依据"是假象：它给不出任何可溯源引用——这正是对照实验测的差
    a = core.answer_no_rag("无人机螺旋桨间距参数标准是多少")
    assert a["citations"] == []


def test_rag_citations_are_traceable_ids():
    ids = {k["id"] for k in KB}
    for q, gold in core.EVAL_IN[:4]:
        a = core.answer_rag(q, RET)
        for c in a["citations"]:
            assert c["id"] in ids


def test_pipeline_reports_new_sections(tmp_path):
    pipeline.run(Settings(seed=7, topk=3), str(tmp_path))
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    for key in ("recall@3", "OOD/超纲拒答实测", "有库/无库对照", "验收档位判定", "RRF"):
        assert key in text
    import json
    data = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert data["ood_refused"] == data["ood_total"]
    assert (data["contrast"]["no_rag_hallucination"]
            > data["contrast"]["rag_hallucination"])
