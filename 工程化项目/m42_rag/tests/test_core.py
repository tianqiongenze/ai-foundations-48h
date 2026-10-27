# -*- coding: utf-8 -*-
from rag48 import core

KB, _SRC = core.load_kb()


def test_kb_loaded_with_ids():
    assert KB and KB[0]["id"] == "R1"


def test_kb_comes_from_project_copy_not_fallback():
    """知识库应来自项目自带副本，而非内置兜底（否则说明文件丢失）。"""
    assert _SRC != "内置兜底(4条)" and len(KB) >= 4


def test_bigrams_has_full_segment():
    assert "振动速度" in core.bigrams("振动速度")


def test_retrieve_hits_relevant():
    hits = core.retrieve("泵振动超标标准", KB, topk=2)
    assert hits and "4.5mm/s" in hits[0]["text"]


def test_retrieve_topk():
    assert len(core.retrieve("振动 电机 轴承", KB, topk=1)) == 1


def test_retrieve_empty_on_high_threshold():
    assert core.retrieve("午饭吃什么", KB, min_score=0.9) == []


def test_answer_with_citations():
    a = core.answer("电机电流偏高", KB)
    assert a["grounded"] and a["citations"] and "[R" in a["text"]


def test_answer_refuses_without_evidence():
    a = core.answer("今天午饭吃什么", KB, min_score=0.9)
    assert not a["grounded"] and "查无依据" in a["text"]
