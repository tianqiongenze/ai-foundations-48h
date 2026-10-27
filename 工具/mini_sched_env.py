# -*- coding: utf-8 -*-
"""mini_sched_env —— W11 多智能体检修调度 自包含教学仿真环境（纯标准库，无gym依赖）
场景：两江智造运维中心3台泵站，工单随机到达队列；调度智能体每步派1人检修或等待。
状态: (队列长度, 各泵退化度)  动作: 0等待 1-3派检对应泵站  奖励: -停机时间-空转惩罚
用法:
    from mini_sched_env import SchedEnv
    env = SchedEnv(seed=42)
    s = env.reset()
    s, r, done, info = env.step(action)
口径: 协调正确率 = 派检时该泵确在告警(退化>=1)的比例（越高越好, 目标≥75%）
"""
import random


class SchedEnv:
    DEGRADE_THRESHOLD = 1.0        # 退化达此值视为告警, 需检修
    IDLE_PENALTY = 5.0             # 有队列却等待(空转刷分)的惩罚
    FIX_BENEFIT = 3.0              # 正确派检收益

    def __init__(self, seed=42, n_pumps=3, horizon=200):
        self.seed = seed
        self.n = n_pumps
        self.horizon = horizon

    def reset(self):
        self.rng = random.Random(self.seed)
        self.degrade = [0.0] * self.n
        self.queue = 0
        self.t = 0
        self.stats = dict(correct=0, wrong=0, idle_with_queue=0, downtime=0)
        return self._state()

    def _state(self):
        return {"queue": self.queue, "degrade": tuple(round(d, 2) for d in self.degrade),
                "alarm": tuple(d >= self.DEGRADE_THRESHOLD for d in self.degrade)}

    def step(self, action):
        self.t += 1
        # 1) 自然退化+随机新工单(世界演化)
        for i in range(self.n):
            self.degrade[i] += self.rng.uniform(0.02, 0.12)
        if self.rng.random() < 0.35:
            self.queue += 1
        # 2) 停机时间累计(告警未修的泵每步计1)
        downtime = sum(1 for d in self.degrade if d >= self.DEGRADE_THRESHOLD)
        self.stats["downtime"] += downtime
        # 3) 动作结算
        r = -float(downtime)
        if action == 0:
            if self.queue > 0:
                r -= self.IDLE_PENALTY
                self.stats["idle_with_queue"] += 1
        elif 1 <= action <= self.n:
            i = action - 1
            if self.degrade[i] >= self.DEGRADE_THRESHOLD:
                r += self.FIX_BENEFIT
                self.stats["correct"] += 1
                self.degrade[i] = 0.0          # 修复复位
                self.queue = max(0, self.queue - 1)
            else:
                self.stats["wrong"] += 1        # 派检无告警泵=误派
                r -= 1.0
        done = self.t >= self.horizon
        info = dict(**self.stats)
        return self._state(), r, done, info

    def accuracy(self):
        c, w = self.stats["correct"], self.stats["wrong"]
        return c / max(c + w, 1)


if __name__ == "__main__":
    # 随机基线演示
    env = SchedEnv(seed=7)
    s = env.reset()
    total = 0.0
    for _ in range(200):
        s, r, done, info = env.step(random.randint(0, 3))
        total += r
        if done:
            break
    print("随机策略: 总奖励", round(total, 1),
          "| 协调正确率", f"{env.accuracy():.1%}",
          "| 停机", info["downtime"], "步")
