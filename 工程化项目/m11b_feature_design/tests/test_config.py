# -*- coding: utf-8 -*-
"""配置层测试：加载、默认值兜底、异常路径，以及"rules.yaml 每个键真的被读"。

约定：本文件的流水线跑动一律用小规模覆盖参数（n_windows/seeds/随机组数），
只验"配置 → 行为"的接线，不验档位达不达标（档位看 test_tiers.py）。
"""
import json
import os

import pytest

from feature_design48 import config as C
from feature_design48 import core
from feature_design48 import pipeline as P

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "rules.yaml")
CFG_TEXT = "warn: 10" + chr(10) + "flag: true" + chr(10) + 'name: "x"' + chr(10)
FAST = {"n_windows": 5, "seeds": [0], "ablation_random_sets": 1, "folds": 3}

_SEQ = [0]


def _cfg(**over):
    d = dict(C.load_config(CFG_PATH).raw)
    d.update(over)
    return C.Settings.from_dict(d)


def _run(tmp_path, **over):
    """小规模跑一遍，返回 (result.json, 输出目录)。"""
    base = dict(FAST)
    base.update(over)
    _SEQ[0] += 1
    out = str(tmp_path / ("run_%03d" % _SEQ[0]))
    P.run(_cfg(**base), out_dir=out)
    with open(os.path.join(out, "result.json"), encoding="utf-8") as f:
        return json.load(f), out


# --------------------------------------------------------- 加载与兜底
def test_load_default_config():
    cfg = C.load_config(CFG_PATH)
    assert 0 < cfg.warn < 1.0                    # Top-N 准确率警戒线（0~1）
    assert 0 < cfg.danger < 0.5                  # 消融红线：允许的降幅（百分点）
    assert cfg.seed > 0 and cfg.topk > 0


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
    assert s.warn == 0.93 and s.danger == 0.03 and s.topk == 8
    assert s.raw["min_features"] == 10 and s.raw["cut_frac"] == 0.30
    assert s.raw["seeds"] == [0, 1, 2] and s.raw["redundant_r"] == 0.95


def test_block_list_parsing():
    text = "items:" + chr(10) + "  - a" + chr(10) + "  - b" + chr(10)
    assert C.yaml_lite(text)["items"] == ["a", "b"]


def test_nested_dict_parsing():
    text = "outer:" + chr(10) + "  inner: 5" + chr(10)
    assert C.yaml_lite(text)["outer"]["inner"] == 5


def test_shipped_yaml_matches_code_defaults():
    """rules.yaml 与 config.DEFAULTS 不许两套口径（配置漂移是事故源）。"""
    shipped = C.load_config(CFG_PATH).raw
    for k, v in C.DEFAULTS.items():
        assert k in shipped, "rules.yaml 缺键 %s" % k
        assert shipped[k] == v, "键 %s 配置 %r ≠ 代码默认 %r" % (k, shipped[k], v)


def test_garbage_numbers_degrade_not_crash():
    """坏配置不炸流水线：非数走默认值（课堂手改配置必踩）。"""
    p = P._read(_cfg(cut_frac="abc", folds="x", seeds="0,1,2",
                     redundant_r="", min_features="oops"))
    assert p["cut_frac"] == 0.30 and p["folds"] == 4
    assert p["seeds"] == (0, 1, 2) and p["redundant_r"] == core.REDUNDANT_R
    assert p["min_features"] == 10


# --------------------------------------------------------- 接线：改配置改行为
def test_topk_is_read(tmp_path):
    r, _ = _run(tmp_path, topk=4)
    assert r["topk"] == 4
    assert any(v["id"] == "topN" and v["n_features"] == 4
               for v in r["feature_sets"].values())


def test_danger_redline_is_read(tmp_path):
    """B 档红线唯一出处 = tiers.B.max_acc_drop：压到 -1（任何降幅都不行），
    消融必须判不达标——证明门槛真的被读，而不是代码里写死 0.03。"""
    r, _ = _run(tmp_path, tiers={"B": {"max_acc_drop": -1.0}})
    assert r["tier_gates"]["B"]["max_acc_drop"] == -1.0
    assert r["ablation_redline"] == -1.0
    assert r["ablation"]["max_drop"] == -1.0
    assert r["tiers"]["B"]["达标"] is False


def test_warn_line_is_read(tmp_path):
    r, _ = _run(tmp_path, warn=0.5)
    assert r["warn_line"] == 0.5


def test_cut_frac_is_read(tmp_path):
    """消融砍除比例出自 tiers.B.cut_frac：抬到 0.5 → 实际必须砍够 50%。"""
    r, _ = _run(tmp_path, tiers={"B": {"cut_frac": 0.5}})
    assert r["tier_gates"]["B"]["cut_frac"] == 0.5
    assert r["cut_frac"] == 0.5
    assert r["ablation"]["n_cut"] >= int(0.5 * r["ablation"]["n_total"])
    assert r["ablation"]["cut_pct_actual"] >= 50.0


def test_min_features_gate_is_read(tmp_path):
    """C 档特征数线出自 tiers.C.min_features：抬到 99 个必然不达标。"""
    r, _ = _run(tmp_path, tiers={"C": {"min_features": 99}})
    assert r["tier_gates"]["C"]["min_features"] == 99
    assert r["min_features"] == 99
    assert r["tiers"]["C"]["达标"] is False


def test_redundant_r_is_read(tmp_path):
    """阈值放宽到 0.7 → 判为冗余的对必须变多。"""
    strict, _ = _run(tmp_path, redundant_r=0.999)
    loose, _ = _run(tmp_path, redundant_r=0.70)
    assert len(loose["correlation_pairs"]) > len(strict["correlation_pairs"])
    assert len(loose["screened_features"]) < len(strict["screened_features"])


def test_seeds_and_folds_are_read(tmp_path):
    r, _ = _run(tmp_path, seeds=[5, 6], folds=3)
    assert r["seeds"] == [5, 6] and r["folds"] == 3
    assert r["ablation"]["base"]["seeds"] == [5, 6]
    assert r["ablation"]["base"]["folds"] == 3


def test_seeds_as_string_is_parsed(tmp_path):
    r, _ = _run(tmp_path, seeds="1,2,3")
    assert r["seeds"] == [1, 2, 3]


def test_n_windows_is_read(tmp_path):
    r, _ = _run(tmp_path, n_windows=6)
    assert r["n_samples"] == 12 and r["n_windows_per_class"] == 6


def test_unit_is_read(tmp_path):
    r, out = _run(tmp_path, unit="g")
    assert r["unit"] == "g"
    with open(os.path.join(out, "report.txt"), encoding="utf-8") as f:
        assert "振动单位 g" in f.read()


def test_seed_changes_dataset(tmp_path):
    a, _ = _run(tmp_path, seed=7)
    b, _ = _run(tmp_path, seed=21)
    assert a["importance_board"][0]["importance"] != \
        b["importance_board"][0]["importance"] or \
        a["ablation"]["base"]["acc"] != b["ablation"]["base"]["acc"]


def test_run_is_reproducible(tmp_path):
    a, _ = _run(tmp_path)
    b, _ = _run(tmp_path)
    assert a["ablation"]["tail"]["acc"] == b["ablation"]["tail"]["acc"]
    assert [x["name"] for x in a["importance_board"]] == \
           [x["name"] for x in b["importance_board"]]
