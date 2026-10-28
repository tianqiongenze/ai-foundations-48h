# -*- coding: utf-8 -*-
"""图像迁移通道测试：增强真实改变像素、2×2 对比可运行、曲线PNG落盘。

这些断言在"增强/2×2/曲线"任一功能被删除时必然失败（工单B/A档能力护栏）。
"""
import numpy as np

from transfer_learning import pipeline, transfer
from transfer_learning.config import Settings


def test_hflip_actually_flips():
    x = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.array_equal(transfer.hflip(x), x[:, ::-1])


def test_rotate_is_deterministic_and_shape_preserved():
    x = np.random.RandomState(0).normal(size=(8, 8)).astype(np.float32)
    a = transfer.rotate(x, 15)
    b = transfer.rotate(x, 15)
    assert a.shape == x.shape and np.allclose(a, b)
    assert not np.allclose(a, x)  # 旋转确实改变了图像


def test_noise_changes_pixels_deterministically():
    x = np.zeros((6, 6), dtype=np.float32) + 0.5
    a = transfer.add_noise(x, 0.03, np.random.RandomState(1))
    b = transfer.add_noise(x, 0.03, np.random.RandomState(1))
    assert np.allclose(a, b) and not np.allclose(a, x)


def test_augment_dataset_expands_and_preserves_labels():
    X = np.random.RandomState(2).normal(size=(10, 16, 16)).astype(np.float32)
    y = np.arange(10) % 2
    Xa, ya = transfer.augment_dataset(X, y, seed=3, repeats=4)
    assert Xa.shape == (40, 16, 16) and ya.shape == (40,)
    assert np.array_equal(ya[:10], y)                       # 原图保留在前
    assert not np.allclose(Xa[10:20], X)                    # 增强变体确有改变


def test_make_image_world_balanced_and_separable_shapes():
    big, small, test = transfer.make_image_world(
        seed=5, n_big=20, n_small=8, n_test=12)
    assert big[0].shape[1:] == (32, 32)
    assert set(np.unique(test[1])) == {0, 1}
    assert len(test[0]) == len(test[1]) == 12


def test_run_experiment_returns_full_2x2(tmp_path):
    raw = {"pretrain_epochs": 1, "head_epochs": 2, "finetune_epochs": 2,
           "aug_repeats": 2, "use_resnet": False}
    exp = transfer.run_experiment(seed=7, raw=raw, out_dir=str(tmp_path))
    assert exp["mode"] == "smallcnn-fallback"
    pairs = {(True, True), (True, False), (False, True), (False, False)}
    assert set(exp["cells"]) == pairs
    for c in exp["cells"].values():
        assert 0.0 <= c["acc"] <= 1.0 and len(c["curve"]) >= 2
    # 增强格子的训练样本数应严格大于不增强格子（增强确实进了训练）
    aug_n = exp["cells"][(True, True)]["n_train"]
    no_n = exp["cells"][(True, False)]["n_train"]
    assert aug_n > no_n
    png = str(tmp_path / "training_curves.png")
    assert exp["png"] == png and np.asarray(exp["pre_curve"]).shape[1] == 2


def test_pipeline_writes_aug_sections_and_curves(tmp_path):
    cfg = Settings(seed=7, raw={"pretrain_epochs": 1, "head_epochs": 2,
                                "finetune_epochs": 2, "aug_repeats": 2})
    sections = pipeline.run(cfg, str(tmp_path))
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "数据增强" in text and "2×2对比" in text and "训练曲线" in text
    # 底座口径段必须二选一并写明实测证据（真ResNet 或 兜底），不许含糊
    assert ("真·ResNet 口径" in text) or ("兜底口径" in text)
    assert (tmp_path / "training_curves.png").stat().st_size > 1000
    rj = (tmp_path / "result.json")
    import json
    assert "image2x2" in json.loads(rj.read_text(encoding="utf-8"))
    assert sections


def test_to_rgb_matches_resnet_input_contract():
    x = np.ones((3, 8, 8), dtype=np.float32) * 0.5
    r = transfer.to_rgb(x)
    assert r.shape == (3, 3, 8, 8) and np.allclose(r[:, 0], 0.5)


def test_load_resnet18_offline_uses_only_local_cache():
    """真·ResNet 通道护栏：只能读本地缓存；缓存缺时如实返回原因，不触网。"""
    info = {"cache_dir": "/nonexistent-dir-xyz", "cached": "resnet18-fake.pth"}
    net, reason = transfer.load_resnet18_offline(info, seed=0)
    assert net is None and ("失败" in reason or "不存在" in reason)
    net, reason = transfer.load_resnet18_offline({"cached": ""}, seed=0)
    assert net is None and "缓存无" in reason


def test_resnet_channel_runs_when_cache_available(tmp_path):
    """本地缓存有权重 → 2×2 必须走真·ResNet（工单主线'载ResNet权重'）。

    缓存缺失时跳过：兜底通道由 test_run_experiment_returns_full_2x2 覆盖。
    """
    import pytest
    probe = transfer.probe_pretrain_weights()
    if not probe["cached"]:
        pytest.skip("本机无 resnet 权重缓存——真ResNet通道不可测，兜底已另行覆盖")
    raw = {"head_epochs": 2, "finetune_epochs": 2, "aug_repeats": 2}
    exp = transfer.run_experiment(seed=7, raw=raw, out_dir=str(tmp_path))
    assert exp["mode"] == "resnet18-imagenet-local-cache"
    pairs = {(True, True), (True, False), (False, True), (False, False)}
    assert set(exp["cells"]) == pairs
    # 底座=ImageNet，非本地预训练
    assert exp["pre_acc"] is None and exp["pre_curve"] is None
    assert exp["resnet_info"]["weights"].startswith("resnet")
    assert (tmp_path / "training_curves.png").stat().st_size > 1000


def test_tier_claim_drops_when_curves_missing():
    """曲线缺失（A 档功能项判缺）→ 承诺档位必须从 优(A) 降到 优秀(B)。"""
    from transfer_learning import tiers
    v = tiers.verdict(acc=0.99, style_rate=95.0, compare_rows=4,
                      aug_gains={"frozen": 0.1, "finetune": 0.1},
                      decisions=1, curves=0)
    got = {t["tier"]: t for t in v["tiers"]}
    assert got["优(A)"]["pass"] is False and got["优秀(B)"]["pass"] is True
    assert v["claimed"] == "优秀(B)"


def test_probe_is_offline_and_reports_cache_evidence():
    """兜底口径必须零网络：探测函数只许查版本+本地缓存，谁再引 socket 谁失败。"""
    import inspect
    import re
    src = inspect.getsource(transfer.probe_pretrain_weights)
    assert not re.search(r"socket|urllib|requests|urlopen", src)
    info = transfer.probe_pretrain_weights()
    assert set(info) == {"torchvision", "cache_dir", "cached", "note"}
    # 笔记必须把"走哪条通道、为什么"说清楚：缓存有权重=启用，没有=兜底
    assert ("兜底" in info["note"] or "启用" in info["note"]) and info["note"]
    assert bool(info["cached"]) == ("启用" in info["note"])
