# -*- coding: utf-8 -*-
"""学习型策略与漏洞堵漏测试——bandit 或漏洞证据能力被删除时必然失败。"""
from multiagent_sched import core, pipeline
from multiagent_sched.config import Settings


def test_bandit_learns_q_table_online():
    sched = core.EpsilonGreedyBandit(seed=1)
    policy = core.policy_of(sched)
    core.run(policy, seed=1000, horizon=50, wrong_penalty=core.WRONG_PENALTY_FIX,
             learner=sched)
    assert sched.stats["pulls"] == 50 and sched.stats["explores"] >= 0
    assert sched.Q, "学习后价值表必须非空——bandit 没学就是没实现"
    assert len({k[0] for k in sched.Q}) >= 2  # 至少见过两种状态桶


def test_bandit_training_is_deterministic():
    _, c1, _, u1 = core.train_bandit(episodes=6, seed=3)
    _, c2, _, u2 = core.train_bandit(episodes=6, seed=3)
    assert c1 == c2 and u1 == u2


def test_bandit_eval_beats_random_on_accuracy():
    sched, _, tr, un = core.train_bandit(episodes=12, seed=5)
    rnd = core.run(core.policy_random, seed=2000, wrong_penalty=core.WRONG_PENALTY_FIX)
    assert un["episodes"] == 10
    assert un["accuracy"] >= tr["accuracy"]  # ε=0 纯利用评估不应更差
    assert un["accuracy"] > rnd["accuracy"]  # 学出来的策略必须赢过随机基线


def test_loophole_before_after_evidence():
    leak = core.loophole_evidence(seed=7)
    assert leak["hack_gain_before"] > 0 > leak["hack_gain_after"]
    b, a = leak["before"], leak["after"]
    assert b["wrong"] > 0                      # 旧奖励下刷分行为真实发生
    assert a["reward"] < b["reward"]           # 堵漏后同一策略套利被剥走，奖励必须下降
    assert a["wrong"] == b["wrong"]            # 策略行为未变——下降只能归因于奖励结构修复


def test_default_reward_keeps_legacy_numbers():
    old = core.run(core.policy_greedy, seed=7)
    explicit = core.run(core.policy_greedy, seed=7,
                        wrong_penalty=core.WRONG_PENALTY_OLD)
    assert old == explicit


def test_pipeline_reports_bandit_and_loophole(tmp_path):
    pipeline.run(Settings(seed=7, raw={"bandit_episodes": 10}), str(tmp_path))
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    for key in ("漏洞机理与堵漏证据", "学习型调度策略", "ε-greedy",
                "学习曲线", "泛化评估", "验收档位判定", "编码规范率"):
        assert key in text
    import json
    data = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert data["bandit"]["unseen_eval"]["accuracy"] >= 0.9
    assert data["loophole"]["before"]["wrong"] > 0
    assert data["reproducible"] is True
