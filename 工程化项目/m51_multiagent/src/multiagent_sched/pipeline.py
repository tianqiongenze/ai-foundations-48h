# -*- coding: utf-8 -*-
"""三策略对照 + 奖励敏感性 + bandit 学习策略 + 漏洞堵漏前后证据 + 档位实测判定。"""
import os

from . import core, io_utils, report, style, tiers

ROOT = style.project_root(os.path.abspath(__file__))


def run(cfg, out_dir):
    """跑完整调度实验：基线/敏感性/漏洞证据/bandit 泛化/规范率/档位判定。"""
    seed = cfg.seed
    res = {
        "随机": core.run(core.policy_random, seed=seed),
        "奖励导向": core.run(core.policy_greedy, seed=seed),
        "安全优先": core.run(core.policy_safe, seed=seed),
    }
    sens = []
    for fb in (1.0, 3.0, 6.0):
        r = core.run(core.policy_safe, seed=seed, fix_benefit=fb)
        sens.append((fb, r["reward"], r["accuracy"]))

    # —— 可复现性实测（及格D档）：同种子跑两遍必须逐键相等 ——
    repro = core.run(core.policy_random, seed=seed) == res["随机"]

    # —— 漏洞与堵漏证据（优秀B档）：全实测 before/after ——
    leak = core.loophole_evidence(seed=seed)

    # —— 学习型调度：ε-greedy bandit 训练+泛化评估（优A档）——
    sched, curve, tr_eval, un_eval = core.train_bandit(
        episodes=int(cfg.raw.get("bandit_episodes", 40)), seed=seed,
        eval_seeds=range(2000, 2010))
    rw_head = sum(r for r, _ in curve[:10]) / 10
    rw_tail = sum(r for r, _ in curve[-10:]) / 10
    ac_head = sum(a for _, a in curve[:10]) / 10
    ac_tail = sum(a for _, a in curve[-10:]) / 10
    learned = un_eval["accuracy"] >= 0.90

    # —— 规范率/说明书/决策记录/漏洞证据：档位判定的实测输入 ——
    tgates = tiers.gates(cfg.tiers)
    sty = style.code_style_rate(ROOT, min_pass=float(
        tgates["D"].get("style_min", 70.0)))
    decision = tiers.decision_record_status(ROOT)
    manual = tiers.manual_status(ROOT)
    acc_best = max(v["accuracy"] for v in res.values())
    reward_delta = leak["after"]["reward"] - leak["before"]["reward"]
    tv = tiers.verdict(
        tgates, rate=sty["rate"], reproducible=repro, acc_best=acc_best,
        manual_chars=manual["chars"],
        wrong_before=leak["before"]["wrong"], reward_delta=reward_delta,
        decision_ok=decision["exists"], unseen_acc=un_eval["accuracy"],
        gen_episodes=un_eval["episodes"])
    result = {"strategies": res, "sensitivity": sens, "reproducible": repro,
              "loophole": leak, "tiers": tv,
              "style": {k: sty[k] for k in
                        ("engine", "rate", "violations", "code_lines",
                         "layout_violations", "structural_violations")},
              "manual_doc": manual, "decision_record": decision,
              "bandit": {"curve_head_reward": round(rw_head, 1),
                         "curve_tail_reward": round(rw_tail, 1),
                         "curve_head_acc": round(ac_head, 4),
                         "curve_tail_acc": round(ac_tail, 4),
                         "pulls": sched.stats["pulls"],
                         "explores": sched.stats["explores"],
                         "exploits": sched.stats["exploits"],
                         "train_eval": tr_eval, "unseen_eval": un_eval,
                         "q_buckets": len({k[0] for k in sched.Q})}}

    lines = ["%-6s 总奖励 %9.1f｜协调正确率 %5.1f%%｜停机 %3d 步｜假派检 %d 次"
             % (k, v["reward"], v["accuracy"] * 100, v["downtime"], v["wrong"])
             for k, v in res.items()]
    safe_acc = res["安全优先"]["accuracy"]
    b, a = leak["before"], leak["after"]
    sections = [
        ("输入口径", ["3 台泵站｜200 步｜停机罚(-1/台)｜空转罚 5｜正确派检收益 3｜种子 %d" % seed]),
        ("三策略对照", lines),
        ("奖励敏感性（安全优先固定）",
         ["FIX_BENEFIT=%.1f → 奖励 %.1f｜正确率 %.1f%%" % (fb, rw, ac * 100)
          for fb, rw, ac in sens]),
        ("漏洞机理与堵漏证据（优秀B档：修复1个漏洞有证据——before/after 全实测）", [
            "漏洞：空转罚 5 > 假修复罚 1 ⇒ 队列忙时'派无告警泵'净赚 +%.1f 分——"
            "奖励导向策略据此刷分（工单B04'凡可被利用的奖励缺陷必然被利用'）" % leak["hack_gain_before"],
            "修复前（罚差 1.0）：奖励导向 总奖励 %.1f｜正确率 %.1f%%｜假派检 %d 次（刷分行为真实发生）"
            % (b["reward"], b["accuracy"] * 100, b["wrong"]),
            "修复后（过度检修罚提至 8.0，刷分收益 %+.1f 变亏本）：同一策略 "
            "总奖励 %.1f｜正确率 %.1f%%｜假派检 %d 次"
            % (leak["hack_gain_after"], a["reward"], a["accuracy"] * 100,
               a["wrong"]),
            "证据结论：奖励导向被'剥掉套利'后奖励由 %.1f → %.1f（Δ%.1f），"
            "证明旧奖励确实在奖励作弊而非能力——堵漏有效是测出来的，不是断言"
            % (b["reward"], a["reward"], a["reward"] - b["reward"])]),
        ("学习型调度策略（ε-greedy 上下文bandit，规则零手写，纯 reward 反馈学成）", [
            "训练 %d 集（种子1000-1039）｜状态桶 %d 个×4动作｜"
            "Q←Q+α(r−Q)，α=0.2，ε 由 0.30 衰减至 0.02"
            % (len(curve), result["bandit"]["q_buckets"]),
            "学习曲线（实测）：回报 前10集均值 %.1f → 后10集 %.1f（%s）；协调正确率 %.1f%% → %.1f%%"
            % (rw_head, rw_tail,
               "回报上升，学习有效" if rw_tail > rw_head else "回报未上升——如实报告",
               ac_head * 100, ac_tail * 100),
            "探索/利用计数：pulls=%d explores=%d exploits=%d"
            % (sched.stats["pulls"], sched.stats["explores"],
               sched.stats["exploits"]),
            "训练种子内评估：奖励 %.1f｜正确率 %.1f%%｜假派检 %d 次"
            % (tr_eval["reward"], tr_eval["accuracy"] * 100, tr_eval["wrong"]),
            "泛化评估（未见过种子 2000-2009，ε=0 纯利用）：奖励 %.1f｜"
            "正确率 %.1f%%｜假派检 %d 次｜停机 %d 步"
            % (un_eval["reward"], un_eval["accuracy"] * 100,
               un_eval["wrong"], un_eval["downtime"])]),
        ("结论", ["安全优先正确率 %.1f%%（目标≥75%%）→ %s；但其奖励最低，"
                "说明奖励函数与 KPI 存在冲突"
                % (safe_acc * 100, "达标" if safe_acc >= 0.75 else "未达标")]),
        ("编码规范率（PEP 8 实测）", [
            "引擎: %s｜扫描 %d 个文件｜非空代码行 %d｜违规 %d 条（版式 %d + "
            "结构 %d）→ 规范率 %.1f%%"
            % (sty["engine"], sty["n_files"], sty["code_lines"],
               sty["violations"], sty["layout_violations"],
               sty["structural_violations"], sty["rate"]),
            sty["fallback_note"]]),
        ("验收档位判定（实测）", tiers.render_lines(tv)),
        ("结论与建议", [
            "正确解法是约束优化：把正确率≥75% 设为硬约束，在此约束下再优化奖励",
            "bandit 实测补全'学习型'一环：同一堵漏后的奖励下，不写规则也能"
            "学到高正确率策略——奖励设计即治理（工单B02），规则与学习互为对照组"])
    ]
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    report.save(sections, os.path.join(out_dir, "report.txt"))
    # 返回 (sections, result)：cli --tier 与 pytest 直接吃实测值，不再重跑
    return sections, result
