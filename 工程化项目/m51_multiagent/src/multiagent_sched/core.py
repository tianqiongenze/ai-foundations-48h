# -*- coding: utf-8 -*-
"""多智能体调度：三策略对照 + 奖励敏感性（纯函数）。"""
import random

N = 3
THRESH = 1.0
IDLE_PENALTY = 5.0
FIX_BENEFIT = 3.0


def policy_random(state, rng):
    return rng.randint(0, N)


def policy_greedy(state, rng):
    """奖励导向：有告警派告警泵；无告警但有队列→派退化最高泵（躲空转罚款）。"""
    al, dg = state["alarm"], state["degrade"]
    cand = [(dg[i], i + 1) for i in range(len(al)) if al[i]]
    if cand:
        return max(cand)[1]
    if state["queue"] > 0:
        return max((dg[i], i + 1) for i in range(len(dg)))[1]
    return 0


def policy_safe(state, rng):
    """安全优先：仅对告警泵派检，无告警则等待（正确率导向）。"""
    al, dg = state["alarm"], state["degrade"]
    cand = [(dg[i], i + 1) for i in range(len(al)) if al[i]]
    return max(cand)[1] if cand else 0


def run(policy, seed=42, horizon=200, fix_benefit=FIX_BENEFIT):
    rng = random.Random(seed)
    degrade = [0.0] * N
    queue = 0
    total = 0.0
    correct = wrong = 0
    downtime = 0
    for _ in range(horizon):
        for i in range(N):
            degrade[i] += rng.uniform(0.02, 0.12)
        if rng.random() < 0.35:
            queue += 1
        dt = sum(1 for d in degrade if d >= THRESH)
        downtime += dt
        r = -float(dt)
        state = {"alarm": [degrade[i] >= THRESH for i in range(N)],
                 "degrade": degrade, "queue": queue}
        a = policy(state, rng)
        if a == 0:
            if queue > 0:
                r -= IDLE_PENALTY
        else:
            i = a - 1
            if degrade[i] >= THRESH:
                r += fix_benefit
                correct += 1
                degrade[i] = 0.0
                queue = max(0, queue - 1)
            else:
                wrong += 1
                r -= 1.0
        total += r
    acc = correct / max(correct + wrong, 1)
    return {"reward": round(total, 1), "accuracy": round(acc, 4),
            "downtime": downtime}
