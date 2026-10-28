# -*- coding: utf-8 -*-
"""造数 → scoring='f1' 双搜索 → 逐候选工时账 → 学习曲线 → 提升树兜底臂 → 验收档位判定（实测）。

四档判定吃的全部是本文件真跑真评估产生的实测值（tiers.verdict 只比较）：
同种子两次重跑逐位比对（D 可复现）、perf_counter 墙钟对 2h 限时（C）、
学习曲线逐档实测 F1 + 程序化拐点 + result.json 回读决策记录（B）、
F1 对线或快速方案机器判据（A）。门槛数字来自 rules.yaml 的 tiers: 段。
"""
import os
import time

from . import core, io_utils, report, style, tiers


def _style_lines(st):
    """规范率引擎结果 → 报告行（引擎/口径/文件行数违规数/达标判定，全实测）。"""
    return [
        "检查引擎：%s｜口径：%s" % (st["engine"], st["definition"]),
        "%d 个 .py 文件 / %d 行非空代码 / %d 条违规 → 规范率 %.2f%% → %s" % (
            st["n_files"], st["code_lines"], st["violations"], st["rate"],
            "达标" if st["passes"] else "未达标"),
        "违规分布：%s" % ("、".join(
            "%s×%d（%s）" % (b["code"], b["count"], b["meaning"])
            for b in st["by_rule"]) or "无"),
        "结构规范（pycodestyle 不管，本模块另算）：%s → %s" % (
            "、".join("%s %d 条" % (k, v) for k, v in st["structural"].items()),
            "全 0" if st["structural_clean"] else "有违规，需整改"),
        "%s——环境也是选型维度，但不是缺证据的借口" % st["fallback_note"],
    ]


def decision_record_ok(path):
    """B 档"决策记录完整"实测：落盘 result.json 回读——网格/随机最优、
    学习曲线逐档 F1、工时账、提升树引擎与兜底口径全部留痕才算完整。"""
    try:
        r = io_utils.read_json(path)
    except (OSError, ValueError):
        return False
    keys_ok = all(r.get(k) for k in
                  ("grid", "random", "learning_curve", "boost_arm", "worklog"))
    if not keys_ok:
        return False
    if not r["grid"].get("best") or not r["random"].get("best"):
        return False
    if not r["learning_curve"].get("rows") or not r["boost_arm"].get("note"):
        return False
    return bool(r["worklog"].get("best"))


def run(cfg, out_dir):
    t_start = time.perf_counter()  # C 档"限时内"的墙钟实测起点
    from sklearn.model_selection import train_test_split
    gates = cfg.tiers
    X, y = core.make_data(400, seed=cfg.seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=1)
    grid = {"max_depth": [3, 4, 5], "n_estimators": [50, 100]}
    # scoring='f1' 的穷举 vs 预算搜索（工单口径不是 acc）
    g = core.search("grid", Xtr, ytr, grid)
    r = core.search("random", Xtr, ytr, grid, n_iter=3)
    saved = round(g["sec"] - r["sec"], 3)
    gap = abs(g["f1"] - r["f1"])
    # 学习曲线（拐点论证，优秀B线）与提升树臂（xgboost→GBC 文档化兜底）
    lc = core.learning_curve_f1(Xtr, ytr, g["best"],
                                gain_max=float(gates["B"]["knee_gain_max"]))
    boost = core.boost_arm(Xtr, ytr, Xte, yte)
    wl = core.worklog_conclusion(g["candidates"])
    # 及格D"基线可复现"：同种子重跑小网格比对（真跑两遍，不是口头承诺）
    rp1 = core.search("grid", Xtr, ytr, {"max_depth": [3, 4]})
    rp2 = core.search("grid", Xtr, ytr, {"max_depth": [3, 4]})
    reproducible = rp1["best"] == rp2["best"] and rp1["f1"] == rp2["f1"]
    run_sec = round(time.perf_counter() - t_start, 3)
    # ---------- D 档次条：代码规范率（工单 M2-2 D 档 ≥70%，必须是量出来的数）
    st = style.code_style_rate(
        style.project_root(os.path.abspath(__file__)),
        min_pass=float(gates["D"]["style_min"]))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "result.json")
    result = {"grid": g, "random": r, "time_saved_sec": saved,
              "f1_gap": round(gap, 4),
              "n_combos": len(grid["max_depth"]) * len(grid["n_estimators"]),
              "learning_curve": lc, "boost_arm": boost,
              "worklog": {"best": wl["best"], "total_sec": wl["total_sec"]},
              "reproducible": reproducible, "run_sec": run_sec,
              "gates": gates, "code_style": st}
    io_utils.write_json(json_path, result)
    decision_ok = decision_record_ok(json_path)
    engine_note = ("%s MISS → 兜底引擎=%s，口径已记录" % (
        "xgboost", boost["engine"])) if boost["fallback"] else "xgboost 原生"
    v = tiers.verdict(
        gates, reproducible=reproducible, grid_f1=g["f1"], random_f1=r["f1"],
        boost_f1=boost["f1"], run_sec=run_sec, curve_points=len(lc["rows"]),
        knee_found=lc["knee_n"] is not None, decision_ok=decision_ok,
        style_rate=st["rate"],
        summary="同种子重跑两次%s｜grid F1 实测 %.4f｜perf_counter 墙钟 %.3fs"
                "（限时 %ds）｜曲线 %d 档拐点 n=%s（增益阈值 %.2f）｜决策记录%s｜"
                "%s｜实测规范率 %.2f%%" % (
                    "逐位一致" if reproducible else "漂移", g["f1"], run_sec,
                    int(gates["C"]["limit_sec"]), len(lc["rows"]),
                    lc["knee_n"], float(gates["B"]["knee_gain_max"]),
                    "完整（result.json 回读校验）" if decision_ok else "缺失",
                    engine_note, st["rate"]))
    result["tier_verdict"] = v
    io_utils.write_json(json_path, result)
    cand_tbl = io_utils.fmt_table(
        [[str(c["params"]), "%.4f" % c["f1"], "%.3f" % c["sec"]]
         for c in sorted(g["candidates"],
                         key=lambda c: c["f1"], reverse=True)],
        ["参数组", "F1", "墙钟秒"])
    lc_tbl = io_utils.fmt_table(
        [[row["n_train"], "%.4f" % row["train_f1"], "%.4f" % row["val_f1"],
          "%.4f" % row["gap"]] for row in lc["rows"]],
        ["训练量n", "train F1", "val F1", "差距"])
    conclusion = ["小空间穷举更稳；大空间用预算搜索——每组候选 F1/秒都记入"
                  "工时账；调参目标用 F1 与验收线同口径，acc 口径会选错模型"]
    input_lines = ["搜索空间 %d 组（max_depth×n_estimators）｜"
                   "scoring='f1'（工单口径，非默认 acc）" % result["n_combos"],
                   "提升树臂：%s（兜底=%s）" % (
                       boost["engine"], boost["fallback"])]
    worklog_lines = [
        cand_tbl, wl["message"],
        "随机省 %.3fs（%.0f%%）｜F1 代价 %.4f（快速方案容差 ≤%.2f）"
        % (saved, saved * 100 / max(g["sec"], 1e-9), gap,
           float(gates["A"]["fast_gap_max"]))]
    sections = [
        ("输入口径", input_lines),
        ("GridSearch（穷举·优化F1）",
         ["cv-F1=%.4f  耗时 %.3fs  最优 %s" % (g["f1"], g["sec"], g["best"])]),
        ("RandomSearch（预算·优化F1）",
         ["cv-F1=%.4f  耗时 %.3fs  最优 %s" % (r["f1"], r["sec"], r["best"])]),
        ("工时账（逐候选墙钟实测）", worklog_lines),
        ("学习曲线（scoring='f1'，拐点=相邻增益跌破 %.2f）"
         % float(gates["B"]["knee_gain_max"]),
         [lc_tbl,
          "拐点：n=%s 起 val F1 增益 <%.2f——数据再加收益趋平，早停在拐点收手"
          % (lc["knee_n"], float(gates["B"]["knee_gain_max"])),
          "train-val 差距 %s（末档）——差距大=过拟合信号，配早停/降深度"
          % lc["rows"][-1]["gap"]]),
        ("提升树臂（XGBoost 口径 + 离线兜底）", [
            "引擎：%s｜%s" % (boost["engine"], boost["note"]),
            "实测 val F1=%.4f｜早停点 best_iteration=%d｜墙钟 %.3fs"
            % (boost["f1"], boost["best_iter"], boost["sec"])]),
        ("验收档位判定（实测）", tiers.render_lines(v)),
        ("人工答辩项（代码判不了，进档位判定不算达标）", tiers.MANUAL),
        ("代码规范率（D 档次条：≥%.0f%%）" % gates["D"]["style_min"],
         _style_lines(st)),
        ("结论与建议", conclusion)]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
