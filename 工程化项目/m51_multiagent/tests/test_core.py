# -*- coding: utf-8 -*-
from multiagent_sched import core


def test_run_returns_three_metrics():
    r = core.run(core.policy_random, seed=1, horizon=50)
    assert "reward" in r and 0.0 <= r["accuracy"] <= 1.0 and r["downtime"] >= 0


def test_run_deterministic():
    assert core.run(core.policy_random, seed=7) == core.run(core.policy_random, seed=7)


def test_safe_first_accuracy_high():
    r = core.run(core.policy_safe, seed=42)
    assert r["accuracy"] >= 0.75


def test_greedy_reward_higher_than_safe():
    g = core.run(core.policy_greedy, seed=42)
    s = core.run(core.policy_safe, seed=42)
    assert g["reward"] > s["reward"]


def test_greedy_accuracy_low_than_safe():
    g = core.run(core.policy_greedy, seed=42)
    s = core.run(core.policy_safe, seed=42)
    assert g["accuracy"] < s["accuracy"]


def test_policy_safe_waits_when_no_alarm():
    st = {"alarm": [False, False, False], "degrade": [0.1, 0.2, 0.3], "queue": 5}
    assert core.policy_safe(st, None) == 0


def test_policy_greedy_dispatches_with_queue():
    st = {"alarm": [False, False, False], "degrade": [0.1, 0.9, 0.3], "queue": 5}
    assert core.policy_greedy(st, None) == 2


def test_fix_benefit_increases_reward():
    a = core.run(core.policy_safe, seed=42, fix_benefit=1.0)
    b = core.run(core.policy_safe, seed=42, fix_benefit=6.0)
    assert b["reward"] > a["reward"]
