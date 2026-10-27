# -*- coding: utf-8 -*-
from data_quality48 import core


def test_make_dirty_has_all_defects():
    df = core.make_dirty(300, 7)
    assert df["vib"].isna().any() and (df["temp"] > 120).any() and (df["vib"] < 0).any()


def test_two_solutions_agree():
    df = core.make_dirty(300, 7)
    assert len(core.clean_py(df)) == len(core.clean_pd(df))


def test_clean_removes_all_defects():
    df = core.make_dirty(300, 7)
    c = core.clean_pd(df)
    assert c["vib"].notna().all() and c["temp"].max() <= 120.0 and c["vib"].min() >= 0.0


def test_quality_between_0_and_1():
    q = core.quality(275, 25)
    assert 0.0 <= q <= 1.0


def test_quality_empty():
    assert core.quality(0, 0) == 0.0
