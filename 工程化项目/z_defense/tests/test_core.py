# -*- coding: utf-8 -*-
from defense_pack import core


def test_layers_has_five():
    assert len(core.LAYERS) == 5


def test_three_questions_present():
    assert len(core.THREE_Q) == 3 and "需求" in core.THREE_Q[0][0]


def test_check_returns_flag_per_layer(tmp_path):
    rows = core.check(str(tmp_path))
    assert len(rows) == 5 and all(isinstance(r[1], bool) for r in rows)


def test_outline_counts_ready():
    rows = [("A", True, "t"), ("B", False, "t")]
    lines, n = core.outline(rows)
    assert n == 1 and "1/2" in lines[-1]


def test_outline_all_ready_message():
    rows = [("A", True, "t"), ("B", True, "t")]
    lines, n = core.outline(rows)
    assert n == 2 and "可以上台" in lines[-1]
