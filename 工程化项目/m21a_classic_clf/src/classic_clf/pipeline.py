# -*- coding: utf-8 -*-
"""三态数据 → 双模型 → 混淆矩阵 + P/R/F1 → 阈值扫描 → 漏报账 → 验收档位判定（实测）。

四档判定吃的全部是本文件真跑真评估产生的实测值（tiers.verdict 只比较），
门槛数字来自 rules.yaml 的 tiers: 段——报告里每一个"达标"都能按行复算。
"""
import os

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


def decision_record_ok(path, labels):
    """B 档"决策记录完整"实测：落盘 result.json 回读——选型结论在、
    每个模型×每个类都有阈值曲线与推荐阈值才算记录完整，缺一条判不达标。"""
    try:
        r = io_utils.read_json(path)
    except (OSError, ValueError):
        return False
    if not r.get("best_by_f1") or not r.get("models"):
        return False
    for m in r["models"].values():
        sweep = m.get("threshold_sweep") or {}
        for a in labels:
            entry = sweep.get(a) or {}
            if not entry.get("rows") or not entry.get("best"):
                return False
    return True


def run(cfg, out_dir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    labels = ["正常", "轴承磨损", "叶轮堵塞"]
    gates = cfg.tiers
    X, y = core.make_data(300, seed=cfg.seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    out = {}
    for nm, m in [("LR", LogisticRegression(max_iter=500)),
                  ("DT", DecisionTreeClassifier(max_depth=6, random_state=0))]:
        out[nm] = core.evaluate(m, Xtr, ytr, Xte, yte, labels)
    gap = abs(out["LR"]["acc"] - out["DT"]["acc"])
    # 选型以 macro F1 为准（accuracy 会被类不平衡骗）
    best_name = max(out, key=lambda k: out[k]["prf"]["macro"]["f1"])
    best = out[best_name]
    macro_f1 = best["prf"]["macro"]["f1"]
    per_best = best["prf"]["per_class"]
    trade_cls = min(per_best, key=lambda a: per_best[a]["f1"])
    sweep = best["threshold_sweep"]
    n_total = sum(len(v["rows"]) for v in sweep.values())
    n_min = min((len(v["rows"]) for v in sweep.values()), default=0)
    bst_thr = sweep[trade_cls]["best"]["threshold"]
    # 口径来源实测：macro 评分调参随结果登记，OvR 扫描逐类带 pos_label
    caliber_ok = (bool(best["prf"].get("caliber"))
                  and len(sweep) == len(labels)
                  and all(sweep[a].get("pos_label") for a in labels))
    # ---------- D 档次条：代码规范率（工单 M2-1a D 档 ≥70%，必须是量出来的数）
    st = style.code_style_rate(
        style.project_root(os.path.abspath(__file__)),
        min_pass=float(gates["D"]["style_min"]))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "result.json")
    result = {"labels": labels, "n_train": len(Xtr), "n_test": len(Xte),
              "models": {k: {"acc": v["acc"], "missed": v["missed"],
                             "cm": v["cm"], "prf": v["prf"],
                             "threshold_sweep": v["threshold_sweep"]}
                         for k, v in out.items()},
              "acc_gap": round(gap, 4), "best_by_f1": best_name,
              "gates": gates, "code_style": st}
    io_utils.write_json(json_path, result)
    decision_ok = decision_record_ok(json_path, labels)
    v = tiers.verdict(
        gates, lr_acc=out["LR"]["acc"], dt_acc=out["DT"]["acc"],
        macro_f1=macro_f1, caliber_ok=caliber_ok,
        sweep_min_points=n_min, sweep_total_points=n_total,
        decision_ok=decision_ok, style_rate=st["rate"],
        summary="基线acc LR %.3f/DT %.3f｜最优 %s macro F1 实测 %.3f"
                "（口径 average=macro，OvR pos_label 逐类登记）｜阈值曲线每类最少"
                "%d 点（共 %d 点）｜决策记录%s｜实测规范率 %.2f%%" % (
                    out["LR"]["acc"], out["DT"]["acc"], best_name, macro_f1,
                    n_min, n_total, "完整（result.json 回读校验）" if decision_ok
                    else "缺失", st["rate"]))
    result["tier_verdict"] = v
    io_utils.write_json(json_path, result)
    cm_lines, prf_lines, rp_lines = [], [], []
    for nm, r in out.items():
        cm_lines.append("%s acc=%.3f 漏报=%s" % (nm, r["acc"], r["missed"]))
        for a in labels:
            cm_lines.append("    %s→%s" % (
                a, {b: r["cm"][a][b] for b in labels}))
        per = r["prf"]["per_class"]
        prf_lines.append("%s 逐类（precision/recall/F1/支撑数）：" % nm)
        for a in labels:
            prf_lines.append("    %s  P=%.3f R=%.3f F1=%.3f n=%d"
                             % (a, per[a]["precision"], per[a]["recall"],
                                per[a]["f1"], per[a]["support"]))
        mac = r["prf"]["macro"]
        prf_lines.append("    macro  P=%.3f R=%.3f F1=%.3f"
                         % (mac["precision"], mac["recall"], mac["f1"]))
        cal = r["prf"]["caliber"]
        prf_lines.append("    F1 口径来源（实测调用参数）：%s average=%s "
                         "labels=%d 类 zero_division=%d"
                         % (cal["scoring"], cal["average"],
                            cal["n_labels"], cal["zero_division"]))
        rp_lines.extend(["--- %s classification_report（sklearn 实测） ---" % nm])
        rp_lines.extend(r["report_lines"])
    # 阈值-权衡表：最优模型里 F1 最低（最吃亏）的类，展示精确率与召回率的此消彼长
    per = best["prf"]["per_class"]
    rows = sweep[trade_cls]["rows"][::3]
    tbl = io_utils.fmt_table(
        [[("%.2f" % r["threshold"]), ("%.3f" % r["precision"]),
          ("%.3f" % r["recall"]), ("%.3f" % r["f1"])] for r in rows],
        ["阈值", "精确率", "召回率", "F1"])
    bst = sweep[trade_cls]["best"]
    sweep_lines = ["选型依据：macro F1 最优模型 = %s（非 acc）" % best_name,
                   "权衡表：%s（%s 类 F1 最低=%s，漏报代价最大）"
                   % (trade_cls, best_name, per[trade_cls]["f1"])]
    sweep_lines.extend(tbl.splitlines())
    sweep_lines.append(
        "推荐阈值 %.2f：精确率 %.3f ↔ 召回率 %.3f（F1=%.3f 实测峰值）；"
        "阈值再升精确率↑召回率↓——漏报（故障→正常）代价高时宁留低阈值"
        % (bst["threshold"], bst["precision"], bst["recall"], bst["f1"]))
    for nm, r in out.items():
        sweep_lines.append("%s 各类最优阈值（F1 峰值，OvR pos_label 口径）：%s" % (
            nm, {a: (r["threshold_sweep"][a]["best"]["threshold"],
                     r["threshold_sweep"][a]["best"]["f1"],
                     r["threshold_sweep"][a]["pos_label"])
                 for a in r["threshold_sweep"]}))
    conclusion = ["双解 acc 差距 %.3f（目标 ≤0.10），但选型看 macro "
                  "F1：%s 最优（%.3f）；%s 类召回最低需在决策矩阵加权"
                  "（漏报代价）" % (gap, best_name, macro_f1, trade_cls)]
    sections = [("输入口径", ["样本 %d 条（三态）｜训练 %d / 测试 %d｜分层划分 stratify=y"
                 % (len(X), len(Xtr), len(Xte))]),
                ("双模型 + 混淆矩阵", cm_lines),
                ("Precision/Recall/F1（逐类+macro，良好线看 F1 不看 acc）",
                 prf_lines + rp_lines),
                ("决策阈值扫描（精确率↔召回率权衡，优秀B线证据）", sweep_lines),
                ("验收档位判定（实测）", tiers.render_lines(v)),
                ("人工答辩项（代码判不了，进档位判定不算达标）", tiers.MANUAL),
                ("代码规范率（D 档次条：≥%.0f%%）" % gates["D"]["style_min"],
                 _style_lines(st)),
                ("结论与建议", conclusion)]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
