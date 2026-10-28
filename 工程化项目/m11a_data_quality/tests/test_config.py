# -*- coding: utf-8 -*-
"""配置层测试：加载、默认值兜底、异常路径，以及"配置真的被流水线读取"。"""
import json
import os

import pytest

from data_quality48 import config as C
from data_quality48 import pipeline as P

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")
CFG_TEXT = "warn: 10" + chr(10) + "flag: true" + chr(10) + 'name: "x"' + chr(10)


def test_load_default_config():
    cfg = C.load_config(CFG_PATH)
    assert cfg.warn < cfg.danger
    assert cfg.seed > 0


def test_yaml_lite_basic():
    d = C.yaml_lite(CFG_TEXT)
    assert d.get("warn") == 10
    assert d.get("flag") is True
    assert d.get("name") == "x"


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        C.load_config("/no/such/rules.yaml")


def test_settings_defaults_fill():
    s = C.Settings.from_dict({})
    assert s.warn == 80.0 and s.topk == 3
    assert (s.raw["w_complete"], s.raw["w_accurate"], s.raw["w_consistent"]) == (0.4, 0.4, 0.2)


def test_block_list_parsing():
    text = "items:" + chr(10) + "  - a" + chr(10) + "  - b" + chr(10)
    assert C.yaml_lite(text)["items"] == ["a", "b"]


def test_nested_dict_parsing():
    text = "outer:" + chr(10) + "  inner: 5" + chr(10)
    assert C.yaml_lite(text)["outer"]["inner"] == 5


# ------------------------------------------------ 规则外置：改配置确实改结果
_SEQ = [0]


def _run(tmp_path, **over):
    """按配置覆盖项跑一遍流水线，返回 result.json（每次落独立目录，可比可复算）。"""
    base = C.load_config(CFG_PATH)
    d = dict(base.raw)
    d.update(over)
    cfg = C.Settings.from_dict(d)
    _SEQ[0] += 1
    out = str(tmp_path / ("run_%03d" % _SEQ[0]))
    P.run(cfg, out_dir=out)
    with open(os.path.join(out, "result.json"), encoding="utf-8") as f:
        return json.load(f)


def test_danger_line_is_read(tmp_path):
    """danger 是报告展示红线；C 档判定线外置在 tiers: 段——抬到 101 必须咬。"""
    r = _run(tmp_path, danger=101.0)
    assert r["missing_handle_redline"] == 101.0
    r2 = _run(tmp_path, tiers={"C": {"min_handle_rate_pct": 101.0}})
    assert r2["tier_gates"]["C"]["min_handle_rate_pct"] == 101.0
    assert r2["tiers"]["C"]["达标"] is False


def test_warn_line_is_read(tmp_path):
    r = _run(tmp_path, warn=70.0)
    assert r["warn_line"] == 70.0


def test_quality_target_is_read(tmp_path):
    """B 档判定线走 tiers: 段：抬到 99.5 本轮达不到，档位必须掉。"""
    r = _run(tmp_path, quality_target=99.5)
    assert r["quality_target"] == 99.5
    r2 = _run(tmp_path, tiers={"B": {"min_quality_score": 99.5}})
    assert r2["tier_gates"]["B"]["min_quality_score"] == 99.5
    assert r2["tiers"]["B"]["达标"] is False         # 线抬太高，本轮达不到


def test_weights_are_read(tmp_path):
    r = _run(tmp_path, w_complete=0.2, w_accurate=0.2, w_consistent=0.6)
    assert r["weights"] == [0.2, 0.2, 0.6]
    manual = round(0.2 * r["rates_after"]["complete"] + 0.2 * r["rates_after"]["accurate"]
                   + 0.6 * r["rates_after"]["consistent"], 2)
    assert abs(r["quality_after"] - manual) < 0.02


def test_topk_is_read(tmp_path):
    r = _run(tmp_path, topk=1)
    assert r["topk"] == 1
    assert sum(1 for m in r["lists"]["missing"] if m["watch"]) == 1


def test_seed_is_read(tmp_path):
    a = _run(tmp_path, seed=11)
    b = _run(tmp_path, seed=12)
    assert a["missing_cells"] != b["missing_cells"] or \
        a["quality_after"] != b["quality_after"]


def test_run_is_reproducible(tmp_path):
    a = _run(tmp_path, seed=7)
    b = _run(tmp_path, seed=7)
    assert a["quality_after"] == b["quality_after"]
    assert a["imputation_experiment"]["mae_pred"] == b["imputation_experiment"]["mae_pred"]
