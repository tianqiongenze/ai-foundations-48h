# -*- coding: utf-8 -*-
from prompt_lab import core


def test_question_has_no_diagnostic_keyword():
    assert not any(k in core.QUESTION for k in ("轴承", "特征频率"))


def test_mock_channel_gives_advice_for_role_prompt():
    ans = core.call(core.MockChannel(), core.PROMPTS["角色"], core.QUESTION)
    assert "轴承" in ans


def test_mock_channel_generic_for_naive():
    ans = core.call(core.MockChannel(), core.PROMPTS["朴素"], core.QUESTION)
    assert "轴承" not in ans


def test_score_tiers():
    assert core.score("建议：更换轴承（…") == 1.0
    assert core.score("建议：检查电源。") == 0.5
    assert core.score("不知道") == 0.0


def test_compare_lift_at_least_50():
    out, base = core.compare(core.MockChannel())
    best = max(v["score"] for v in out.values())
    assert (best - base) * 100 >= 50


def test_compare_deltas_relative_to_first():
    out, _ = core.compare(core.MockChannel())
    assert out["朴素"]["delta"] == 0.0
