# -*- coding: utf-8 -*-
"""多智能体调度：三策略对照 + 奖励敏感性 + ε-greedy bandit 学习策略 + 奖励漏洞堵漏前后证据（纯函数）。"""
import random

N = 3
THRESH = 1.0
IDLE_PENALTY = 5.0
FIX_BENEFIT = 3.0
WRONG_PENALTY_OLD = 1.0   # 旧奖励：假修复只罚 1 —— 低于空转罚 5，漏洞所在（工单B04）
WRONG_PENALTY_FIX = 8.0   # 堵漏奖励：过度检修罚 8 > 空转罚 5，"乱派检"由套利变成亏本
ACTIONS = list(range(N + 1))  # 0=等待；1..N=派检对应泵


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


def run(policy, seed=42, horizon=200, fix_benefit=FIX_BENEFIT,
        wrong_penalty=WRONG_PENALTY_OLD, learner=None):
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
                r -= wrong_penalty
        total += r
        if learner is not None:
            learner.step(state, a, r)  # bandit 在线学习：只吃"状态桶-动作-即时奖励"反馈
    acc = correct / max(correct + wrong, 1)
    return {"reward": round(total, 1), "accuracy": round(acc, 4),
            "downtime": downtime, "correct": correct, "wrong": wrong}


class EpsilonGreedyBandit:
    """ε-greedy 上下文 bandit：在(告警模式×队列忙闲)状态桶上学动作价值表 Q。

    工单B01 三要素的最小可运行实现：状态=告警向量模式、动作=等待/派检谁、
    奖励=环境即时分——没有任何手写规则，策略完全由 reward 反馈学出来（学习型调度）。
    更新律 Q←Q+α(r−Q)，种子固定可复现。
    """

    def __init__(self, alpha=0.2, eps=0.10, seed=0):
        self.alpha = alpha
        self.eps = eps
        self.rng = random.Random(seed)
        self.Q = {}
        self.stats = {"pulls": 0, "explores": 0, "exploits": 0}

    @staticmethod
    def bucket(state):
        return (tuple(1 if x else 0 for x in state["alarm"]),
                1 if state["queue"] > 0 else 0)

    def value(self, s, a):
        return self.Q.get((s, a), 0.0)

    def choose(self, state, rng=None):
        s = self.bucket(state)
        if self.rng.random() < self.eps:
            self.stats["explores"] += 1
            a = self.rng.choice(ACTIONS)
        else:
            self.stats["exploits"] += 1
            best = max(self.value(s, x) for x in ACTIONS)
            cands = [x for x in ACTIONS if self.value(s, x) >= best]
            a = self.rng.choice(cands)
        self.stats["pulls"] += 1
        return a

    def step(self, state, a, r):
        s = self.bucket(state)
        q = self.Q.get((s, a), 0.0)
        self.Q[(s, a)] = q + self.alpha * (r - q)


def train_bandit(episodes=40, seed=1, horizon=200, fix_benefit=FIX_BENEFIT,
                 wrong_penalty=WRONG_PENALTY_FIX, eval_seeds=range(2000, 2010)):
    """训练+评估一把闭环：curve 为每集 (回报, 正确率)；评估用未见过种子且 ε=0（纯利用）。

    返回 (sched, curve, eval_trainseeds, eval_unseen) ——全部实测数字，供报告引用。
    """
    sched = EpsilonGreedyBandit(seed=seed)
    curve = []
    for ep in range(episodes):
        # ε 随集数衰减（0.30×0.92^ep，下限 0.02）：前期多探索、后期多利用——学习曲线的常态配方
        sched.eps = max(0.02, 0.30 * (0.92 ** ep))
        r = run(policy_of(sched), seed=1000 + ep, horizon=horizon,
                fix_benefit=fix_benefit, wrong_penalty=wrong_penalty, learner=sched)
        curve.append((r["reward"], r["accuracy"]))
    eps_backup = sched.eps
    sched.eps = 0.0
    train_eval = evaluate(sched, seeds=range(1000, 1005), horizon=horizon,
                          fix_benefit=fix_benefit, wrong_penalty=wrong_penalty)
    unseen_eval = evaluate(sched, seeds=eval_seeds, horizon=horizon,
                           fix_benefit=fix_benefit, wrong_penalty=wrong_penalty)
    sched.eps = eps_backup
    return sched, curve, train_eval, unseen_eval


def policy_of(sched):
    """把 bandit 适配成 run() 需要的 policy(state, rng) 接口。"""
    return lambda state, rng: sched.choose(state)


def evaluate(sched, seeds, horizon=200, fix_benefit=FIX_BENEFIT,
             wrong_penalty=WRONG_PENALTY_FIX):
    """一组种子上评估（须在 ε=0 下调用）：返回平均奖励/正确率/停机。"""
    rs = [run(policy_of(sched), seed=s, horizon=horizon, fix_benefit=fix_benefit,
              wrong_penalty=wrong_penalty) for s in seeds]
    n = len(rs)
    return {"reward": round(sum(r["reward"] for r in rs) / n, 1),
            "accuracy": round(sum(r["accuracy"] for r in rs) / n, 4),
            "downtime": sum(r["downtime"] for r in rs),
            "correct": sum(r["correct"] for r in rs),
            "wrong": sum(r["wrong"] for r in rs), "episodes": n}


def loophole_evidence(seed=42, horizon=200):
    """漏洞前后证据（全实测）：旧奖励下'无告警假派检'是套利，堵漏后变成亏本。

    漏洞机理：空转罚 5 > 假修复罚 1 ⇒ 队列忙时'派错泵'反而净赚 4 分——
    智能体（和照此写死的奖励导向策略）必然学会'用假检修刷掉空转罚'，
    表现即：奖励导向总奖励 -90 远高于安全优先 -738，但协调正确率只有 ~19%。
    堵漏：过度检修罚提到 8（>空转罚5），同样策略的刷分收益由 +4 变 -3。
    """
    before = run(policy_greedy, seed=seed, horizon=horizon,
                 wrong_penalty=WRONG_PENALTY_OLD)
    after = run(policy_greedy, seed=seed, horizon=horizon,
               wrong_penalty=WRONG_PENALTY_FIX)
    hack_gain_before = IDLE_PENALTY - WRONG_PENALTY_OLD   # 罚差口径，均用模块常数
    hack_gain_after = IDLE_PENALTY - WRONG_PENALTY_FIX
    return {"before": before, "after": after,
            "hack_gain_before": hack_gain_before,
            "hack_gain_after": hack_gain_after}
