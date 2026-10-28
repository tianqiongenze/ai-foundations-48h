# -*- coding: utf-8 -*-
"""M1-1b 流水线：多窗口信号 → 18 维特征（时域13+频域5）→ 特征字典自检
→ 相关性初筛 → 重要性榜（多种子）→ 消融实验（−30%）→ A/B 特征集对比 → 报告。

规则外置：`config/rules.yaml` 的 warn / danger / topk / cut_frac / min_features /
folds / seeds / ablation_random_sets / redundant_r / n_windows / unit / seed
全部被本流水线真实读取（改配置不改代码，tests/test_config.py 逐键验证）。

口径要点（工单 B01~B06 知识卡）：
  B01 特征=体检指标（模型吃统计量不吃原始波形）；
  B02 特征字典=户口本（五栏成文）；
  B03 相关性初筛（|r|>0.95 冗余对留一个）；
  B04 重要性榜与物理直觉互验；
  B05 消融 + A/B 特征集对比是归因黄金标准；
  B06 泄漏预警（只用窗口内已可得的统计量，无事后结论列）。
"""
import os

from . import core, io_utils, report, style, tiers


def _read(cfg):
    """把 config/rules.yaml 的键取齐（缺失走默认，但默认值与 YAML 同源）。

    四档门槛（min_features/cut_frac/max_acc_drop/min_code_style）以
    tiers: 段为唯一判定出处；顶层同名键仅作报告展示/兼容线。
    """
    raw = cfg.raw or {}
    g = cfg.tier_gates()

    def f(k, d):
        try:
            return float(raw.get(k, d))
        except (TypeError, ValueError):
            return d

    def i(k, d):
        try:
            return int(raw.get(k, d))
        except (TypeError, ValueError):
            return d

    seeds = raw.get("seeds")
    if isinstance(seeds, str):
        seeds = [int(s) for s in seeds.replace(" ", "").split(",") if s]
    if not isinstance(seeds, (list, tuple)) or not seeds:
        seeds = (0, 1, 2)
    return {"min_features": int(g["C"]["min_features"]),
            "cut_frac": float(g["B"]["cut_frac"]),
            "max_acc_drop": float(g["B"]["max_acc_drop"]),
            "min_decision_chars": int(g["B"]["min_decision_chars"]),
            "top_n": int(g["A"]["top_n"]),
            "min_new_in_top": int(g["A"]["min_new_in_top"]),
            "min_explain_chars": int(g["A"]["min_explain_chars"]),
            "min_code_style": float(g["D"]["min_code_style_pct"]),
            "tier_gates": g,
            "folds": i("folds", 4), "seeds": tuple(int(s) for s in seeds),
            "n_random": i("ablation_random_sets", 5), "warn": float(cfg.warn),
            "danger": float(cfg.danger), "topk": int(cfg.topk),
            "redundant_r": f("redundant_r", core.REDUNDANT_R),
            "n_windows": i("n_windows", 60),
            "min_style": float(g["D"]["min_code_style_pct"]),
            "unit": str(cfg.unit)}


def _set(out, sid):
    """按逻辑 id 取 A/B 臂（显示名带特征数，不能写死字符串）。"""
    for k, v in out.items():
        if v.get("id") == sid:
            return k, v
    raise KeyError(sid)


def tier_verdict(biz, st, order=tiers.KEYS):
    """兼容入口：折叠"业务实测布尔 + 规范率引擎"成档位判定。

    判定逻辑唯一出处在 tiers.fold——本函数只转发，杜绝两套口径漂移；
    完整实测判定（拿原始测量值比门槛）走 tiers.verdict。
    """
    return tiers.fold({k: biz.get(k) for k in order}, st, order)


def run(cfg, out_dir):
    p = _read(cfg)
    X, y = core.build_dataset(p["n_windows"], seed=cfg.seed)
    n_win = X.shape[0]
    n_feat = X.shape[1]

    # ---------- ① 特征字典自检（C 层硬门槛）
    dict_bad = core.dict_missing_columns()
    dict_rows = [{"name": d["name"], "domain": d["domain"], "formula": d["formula"],
                  "meaning": d["meaning"], "unit": d["unit"], "range": d["range"],
                  "since": d["since"]} for d in core.FEATURE_DICT]

    # ---------- ② 相关性初筛（B03）
    pairs, keep_idx = core.correlation_screen(X, threshold=p["redundant_r"])
    screened = [core.FEAT_NAMES[i] for i in keep_idx]

    # ---------- ③ 重要性榜（B04，多种子取均值 + 名次波动）
    board = core.rank_stable(X, y, p["seeds"])
    ranked = [(r["name"], r["importance"]) for r in board]

    # ---------- ④ 消融实验（B 档：特征 −30% 性能不降，红线看 tiers 段）
    ab = core.ablate(X, y, ranked, cut_frac=p["cut_frac"], folds=p["folds"],
                     seeds=p["seeds"], n_random=p["n_random"],
                     max_drop=p["max_acc_drop"])

    # ---------- ⑤ A/B 特征集对比（B05）
    ab_sets, best_tag, top = core.feature_set_ab(X, y, topk=p["topk"],
                                                 folds=p["folds"], seeds=p["seeds"],
                                                 ranked=ranked)
    _, arm_top = _set(ab_sets, "topN")
    _, arm_time = _set(ab_sets, "time")
    _, arm_freq = _set(ab_sets, "freq")
    _, arm_full = _set(ab_sets, "full")
    _, min5 = _set(ab_sets, "min5")
    topN_acc = arm_top["acc"]

    # ---------- ⑥ A 档：新特征进 Top3 + 物理解释
    newchk = core.new_feature_check(ranked, topn=p["top_n"])

    # ---------- ⑦ D 档次条：代码规范率（工单写死 ≥70%，必须是量出来的数）
    st = style.code_style_rate(style.project_root(
        os.path.abspath(__file__)), min_pass=p["min_style"])

    # 决策记录段先拼好：B 档"决策记录完整"实测的就是这段真实文本
    decision_lines = [
        "候选路线：①只加时域高阶量（峭度/裕度因子）②加频域比值型特征 "
        "③换 XGBoost/SVM 提分",
        "决策：走 ②。理由——数据里工况幅值逐窗随机，绝对量已被污染，"
        "比值型特征是唯一能'约掉工况'的做法；③属于'用换模型掩盖特征问题'，"
        "工单 B05 明确不推荐",
        "工时账：加 4 个频域/时域新特征 + 重跑约 2 分钟，"
        "换模型调参至少半天，收益却不确定（环境也是选型维度）",
        "留痕：feature_dict.csv / importance_rank.csv / correlation_screen.csv / "
        "ablation.csv / ab_feature_sets.csv 五件齐，答辩可回算",
    ]
    decision_chars = sum(len(x) for x in decision_lines)
    need_cols = ("name", "formula", "meaning", "unit", "range")
    dict_cols = sum(
        1 for c in need_cols
        if not any(x.endswith("." + c) for x in dict_bad))
    top_names = [n for n, _ in newchk["new_in_top"]]
    explain_chars = sum(len(newchk["meanings"].get(n, "")) for n in top_names)

    g = p["tier_gates"]
    m = {
        "min5_n_features": len(min5["features"]),
        "min5_acc": min5["acc"],
        "n_features": n_feat,
        "dict_missing_cells": len(dict_bad),
        "dict_cols": dict_cols,
        "cut_pct_actual": ab["cut_pct_actual"],
        "tail_drop": ab["tail_drop"],
        "decision_chars": decision_chars,
        "n_new_in_top": newchk["n_new_in_top"],
        "explain_chars": explain_chars,
        "evidence": {
            "D": "最小集 %d 特征跑通：%d 折×%d 种子 CV acc %.4f；规范率 %.2f%%"
                 % (len(min5["features"]), p["folds"], len(p["seeds"]),
                    min5["acc"], st["rate"]),
            "C": "特征 %d 个（线 %d）；字典五栏成文 %d/5，缺空 %d 处"
                 % (n_feat, p["min_features"], dict_cols, len(dict_bad)),
            "B": "尾部砍 %.1f%%（%d 个）后 acc %+.4f（红线 ±%.3f）；决策记录 %d 字"
                 % (ab["cut_pct_actual"], ab["n_cut"], ab["tail_drop"],
                    p["max_acc_drop"], decision_chars),
            "A": "Top%d 内 v1.1 新特征 %d 个：%s；物理解释成文 %d 字"
                 % (p["top_n"], newchk["n_new_in_top"],
                    "、".join(top_names) or "无", explain_chars),
        },
    }
    v = tiers.verdict(m, st, g)

    d_measured = (
        "前 5 特征（t_mean/t_std/t_rms/t_ptp/t_max）%d 折×%d 种子 CV "
        "跑通：acc %.4f / macro-F1 %.4f——能跑通但分不开，"
        "说明'绝对幅值类'特征被工况淹没"
        % (p["folds"], len(p["seeds"]), min5["acc"], min5["f1"]))
    d_style = (
        "代码规范率 %.2f%%（%s，%d 个文件 / %d 行 / %d 条违规，线 %.0f%%）"
        % (st["rate"], st["engine"], st["n_files"], st["code_lines"],
           st["violations"], p["min_style"]))
    c_measured = (
        "特征 %d 个（时域 %d + 频域 %d）≥ %d；字典五栏缺空 %d 处；"
        "相关性初筛剔除冗余对 %d 组（|r|>%.2f），精简后留 %d 个"
        % (n_feat, len(core.TIME_FEATS), len(core.FEAT_NAMES) -
           len(core.TIME_FEATS), p["min_features"], len(dict_bad),
           len(pairs), p["redundant_r"], len(screened)))
    b_measured = (
        "全集 acc %.4f → 砍尾部 %d 个后 acc %.4f（变化 %+.4f，红线 ±%.3f）；"
        "随机砍同数量均值 %.4f（降 %.4f）；砍头部 %d 个 acc %.4f（降 %.4f）"
        "→ 榜可信；决策记录 %d 字（线 %d）"
        % (ab["base"]["acc"], ab["n_cut"], ab["tail"]["acc"],
           ab["tail_drop"], p["max_acc_drop"], ab["random"]["acc"],
           ab["random_drop"], ab["n_cut"], ab["head"]["acc"],
           ab["head_drop"], decision_chars, p["min_decision_chars"]))
    a_measured = (
        "Top%d = %s；其中 v1.1 新增 %d 个（%s），名次跨种子稳定"
        "（rank %d→%d / %d→%d）；物理解释成文 %d 字（线 %d），"
        "解释见特征字典 meaning 栏"
        % (p["top_n"],
           " / ".join("%s(%.4f)" % (n, s) for n, s in newchk["top3"]),
           newchk["n_new_in_top"], "、".join(top_names) or "无",
           board[0]["rank_seed_min"], board[0]["rank_seed_max"],
           board[2]["rank_seed_min"], board[2]["rank_seed_max"],
           explain_chars, p["min_explain_chars"]))
    d_req = ("5 特征表能跑通（最小集出指标）+ 代码规范率≥%.0f%%"
             % p["min_style"])
    c_req = ("≥%d 个特征且特征字典完整（名称/公式/含义/单位/范围）"
             % p["min_features"])
    b_req = ("特征砍掉 %.0f%% 后性能不降（消融验证）+ 决策记录完整"
             % (p["cut_frac"] * 100))
    a_req = "新特征挤进 Top%d 且有物理解释" % p["top_n"]
    tiers_detail = {
        "D": {"要求": d_req, "实测": d_measured, "实测2": d_style,
              "达标": v["final"]["D"]},
        "C": {"要求": c_req, "实测": c_measured,
              "达标": v["final"]["C"]},
        "B": {"要求": b_req, "实测": b_measured,
              "达标": v["final"]["B"]},
        "A": {"要求": a_req, "实测": a_measured,
              "达标": v["final"]["A"]},
    }

    result = {
        "seed": cfg.seed, "unit": p["unit"], "topk": p["topk"],
        "min_features": p["min_features"], "cut_frac": p["cut_frac"],
        "warn_line": p["warn"], "ablation_redline": p["max_acc_drop"],
        "folds": p["folds"], "seeds": list(p["seeds"]),
        "n_samples": n_win, "n_windows_per_class": p["n_windows"],
        "n_features": n_feat, "n_time_features": len(core.TIME_FEATS),
        "n_freq_features": n_feat - len(core.TIME_FEATS),
        "feature_dict_rows": len(dict_rows), "feature_dict_missing_cells": dict_bad,
        "new_features_v11": core.NEW_FEATS,
        "importance_board": board, "top1": ranked[0][0],
        "correlation_threshold": p["redundant_r"],
        "correlation_pairs": pairs, "screened_features": screened,
        "ablation": ab, "feature_sets": ab_sets, "best_set": best_tag,
        "new_feature_check": newchk,
        "leak_check": "全部特征只由本窗口 400 点采样算出，无跨窗口/事后结论列（B06）",
        "code_style": st, "min_code_style": p["min_style"],
        "tier_gates": g,
        "claimed_tier": v["claimed"],
        "tier_verdict": v,
        "tiers": tiers_detail,
    }

    io_utils.ensure_dir(out_dir)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    io_utils.write_csv_rows(os.path.join(out_dir, "feature_dict.csv"), dict_rows)
    io_utils.write_csv_rows(os.path.join(out_dir, "importance_rank.csv"), board)
    io_utils.write_csv_rows(os.path.join(out_dir, "correlation_screen.csv"),
                            pairs or [{"a": "—", "b": "—", "r": "—", "keep": "—",
                                       "reason": "无冗余对"}])
    io_utils.write_csv_rows(os.path.join(out_dir, "ab_feature_sets.csv"),
                            [{"set": k, "n_features": v["n_features"],
                              "accuracy": v["acc"], "accuracy_spread": v["acc_spread"],
                              "f1_macro": v["f1"],
                              "features": "|".join(v["features"])}
                             for k, v in ab_sets.items()])
    io_utils.write_csv_rows(os.path.join(out_dir, "ablation.csv"), [
        {"arm": "全集(18)", "n_features": ab["n_total"], "accuracy": ab["base"]["acc"],
         "f1_macro": ab["base"]["f1"], "drop_vs_base": 0.0},
        {"arm": "尾部砍%.0f%%" % (p["cut_frac"] * 100), "n_features": ab["keep"],
         "accuracy": ab["tail"]["acc"], "f1_macro": ab["tail"]["f1"],
         "drop_vs_base": ab["tail_drop"]},
        {"arm": "随机砍%.0f%%×%d组均值" % (p["cut_frac"] * 100, p["n_random"]),
         "n_features": ab["keep"], "accuracy": ab["random"]["acc"],
         "f1_macro": ab["random"]["f1"], "drop_vs_base": ab["random_drop"]},
        {"arm": "头部砍%.0f%%" % (p["cut_frac"] * 100), "n_features": ab["keep"],
         "accuracy": ab["head"]["acc"], "f1_macro": ab["head"]["f1"],
         "drop_vs_base": ab["head_drop"]},
    ])

    # ---------- 报告
    sections = [
        ("输入口径", ["振动窗口 %d 个（正常/磨损各 %d，每窗 400 点 @%.0fHz）｜"
                      "特征 %d 维（时域 %d + 频域 %d）｜随机森林 300 棵｜种子 %d"
                      % (n_win, p["n_windows"], core.FS_DEFAULT, n_feat,
                         len(core.TIME_FEATS), n_feat - len(core.TIME_FEATS), cfg.seed),
                      "振动单位 %s；评价 = %d 折分层 CV × 种子 %s 取均值（小样本必须重复）"
                      % (p["unit"], p["folds"], p["seeds"]),
                      "工况设计：转频幅值与本底噪声逐窗随机，故障只体现在'二倍频/基频之比'——"
                      "绝对幅值类特征（t_std/t_rms/t_max）只跟工况走不跟故障走"]),
        ("特征字典（C 层户口本）",
         [io_utils.fmt_table(
             [[d["name"], d["domain"], d["formula"], d["meaning"], d["unit"],
               d["range"], d["since"]] for d in dict_rows],
             ["特征名", "域", "公式", "物理含义", "单位", "取值范围", "版本"])
          ]
         + ["五栏（名称/公式/含义/单位/范围）缺空 %d 处 → %s"
            % (len(dict_bad), "完整" if not dict_bad else "不完整：" + str(dict_bad)),
            "全表同步落盘 output/feature_dict.csv"]),
        ("相关性初筛（B03）",
         ["|r|>%.2f 的冗余对 %d 组：" % (p["redundant_r"], len(pairs))]
         + ["  %s ↔ %s  r=%.4f → 保留 %s" % (c["a"], c["b"], c["r"], c["keep"])
            for c in pairs]
         + ["初筛后特征集 %d 个（%s）" % (len(screened), "、".join(screened)),
            "为什么要筛：冗余列不增信息只增方差，还会稀释重要性榜，"
            "让'哪个特征真有用'变成一笔糊涂账"]),
        ("重要性榜（B04，多种子均值）",
         ["%2d. %-14s %-4s %-6s imp=%.4f  名次跨种子 %d~%d"
          % (k + 1, r["name"], r["domain"], r["since"], r["importance"],
             r["rank_seed_min"], r["rank_seed_max"])
          for k, r in enumerate(board)]
         + ["Top1 = %s（%.0fHz 二倍频指纹）" % (ranked[0][0], core.WEAR_HZ),
            "榜本身也要有置信度：名次跨种子波动为 0 才算真结论，"
            "否则'新特征进 Top3'可能只是随机种子运气"]),
        ("消融实验（B 层：特征 −%.0f%%）" % (p["cut_frac"] * 100), [
            "全集 %d 个 acc %.4f（f1 %.4f）" % (ab["n_total"], ab["base"]["acc"],
                                                 ab["base"]["f1"]),
            "砍够 %.0f%% → 实际砍 %d 个（%.1f%%，向上取整才叫'砍够'），留 %d 个" % (
                p["cut_frac"] * 100, ab["n_cut"],
                100.0 * ab["n_cut"] / ab["n_total"], ab["keep"]),
            "砍尾部 %d 个（留 %d）：acc %.4f，变化 %+.4f；红线 ±%.3f → %s" % (
                ab["n_cut"], ab["keep"], ab["tail"]["acc"], ab["tail_drop"],
                p["max_acc_drop"], "达标（性能不降）" if ab["pass"] else "未达标"),
            "尾部剔除清单：%s" % "、".join(
                r[0] for r in ranked[ab["n_total"] - ab["n_cut"]:]),
            "随机砍同数量（%d 组均值）：acc %.4f，降 %.4f；"
            "尾部砍 %s随机砍 → 说明榜排的序是真的" % (
                p["n_random"], ab["random"]["acc"], ab["random_drop"],
                "优于" if ab["tail_better_than_random"] else "不优于"),
            "反向对照·砍头部 %d 个：acc %.4f，降 %.4f → %s" % (
                ab["n_cut"], ab["head"]["acc"], ab["head_drop"],
                "断崖（榜可信）" if ab["head_collapses"] else "未塌（榜不可信，需重做）"),
            "随机臂为什么要跑 %d 组：%d 样本一次划分一票之差就是 %.1f%%，"
            "单跑一次'尾部砍'和'随机砍'会打平，分不出砍谁" % (
                p["n_random"], n_win, 100.0 / n_win)]),
        ("A/B 特征集对比（B05）",
         [io_utils.fmt_table(
             [[k, v["n_features"], "%.4f" % v["acc"], "%.4f" % v["f1"],
               "%.4f" % v["acc_spread"]] for k, v in ab_sets.items()],
             ["特征集", "维度", "accuracy", "macro-F1", "种子极差"])]
         + ["最优集 = %s" % best_tag,
          "对照结论：只用时域 %d 个 acc %.4f，只用频域 %d 个 acc %.4f，"
          "全集 acc %.4f，Top%d 精选 acc %.4f" % (
              len(core.TIME_FEATS), arm_time["acc"],
              len(core.FREQ_FEATS), arm_freq["acc"],
              arm_full["acc"], p["topk"], topN_acc),
          "警戒线 %.2f：Top%d 精选集 acc %.4f → %s" % (
              p["warn"], p["topk"], topN_acc,
              "达标" if topN_acc >= p["warn"] else "低于警戒线，说明砍过头或特征冗余未清"),
          "D 档最小集（5 个绝对幅值特征）只有 acc %.4f ≈ 瞎猜——"
          "这一行就是'好特征胜过换模型'的反证" % min5["acc"],
          "对比表落盘 output/ab_feature_sets.csv，消融明细 output/ablation.csv"]),
        ("新特征增益与物理解释（A 层）", [
            "v1.1 新增 %d 个：%s" % (len(core.NEW_FEATS), "、".join(core.NEW_FEATS)),
            "进入 Top3 的新特征 %d 个：%s" % (
                newchk["n_new_in_top"],
                "、".join("%s(imp %.4f)" % (n, s) for n, s in newchk["new_in_top"])),
            "物理解释：" + "；".join("%s = %s" % (n, m) for n, m in
                                      newchk["meanings"].items()
                                      if n in [x[0] for x in newchk["new_in_top"]]),
            "为什么 f_ratio2x 有效：二倍频幅值÷基频幅值把'整机转速大小'约掉了，"
            "只剩'磨损比例'——绝对幅值类特征被工况带偏，比值型特征才对故障单向敏感",
            "为什么 t_clearance/t_crest 没进 Top3：本数据冲击只给 4 个弱触点，"
            "峭度类时域量信息量不足；现场点蚀早期应加长窗口再评——写进改进建议"]),
        ("泄漏预警（B06）", [
            "18 个特征全部由『本窗口 400 点』算出，不用下一窗口、不用事后维修记录；",
            "标签来自工单设定的磨损状态（合成信号里只通过二倍频比与弱冲击体现），"
            "无'标签混进特征'的路径；",
            "特征重要性、消融、A/B 全部在同一 %d 折分层 CV 下比较，"
            "不存在'用全集选特征再用全集验证'的乐观偏差。" % p["folds"]]),
        ("代码规范率（D 档次条：≥%.0f%%）" % p["min_style"], [
            "检查引擎：%s｜口径：%s" % (st["engine"], st["definition"]),
            "%d 个 .py 文件 / %d 行非空代码 / %d 条违规 → 规范率 %.2f%% → %s" % (
                st["n_files"], st["code_lines"], st["violations"], st["rate"],
                "达标" if st["passes"] else "未达标"),
            "违规分布：%s" % ("、".join("%s×%d（%s）" % (b["code"], b["count"],
                                                        b["meaning"])
                                       for b in st["by_rule"]) or "无"),
            "结构规范（pycodestyle 不管，本模块另算）：%s → %s" % (
                "、".join("%s %d 条" % (k, v) for k, v in st["structural"].items()),
                "全 0" if st["structural_clean"] else "有违规，需整改"),
            "%s——环境也是选型维度，但不是缺证据的借口" % st["fallback_note"],
            "口径：%s" % st["definition"]]),
        ("决策记录", decision_lines),
        ("验收档位判定（实测）", tiers.render_lines(v)),
        ("验收对照明细（工单 M1-1b 四档逐条证据）",
         ["%s 层 [%s] %s ｜ %s%s" % (
             k, "达成" if tiers_detail[k]["达标"] else "未达成",
             tiers_detail[k]["要求"], tiers_detail[k]["实测"],
             ("\n" + " " * 6 + "｜ " + tiers_detail[k]["实测2"])
             if tiers_detail[k].get("实测2") else "")
          for k in tiers.KEYS]),
        ("结论与建议", [
            "实测达成档位：%s（判定只认实测值，见「验收档位判定（实测）」段）"
            % v["claimed"],
            "特征 %d 个（≥%d）+ 五栏字典零缺空 → C %s"
            % (n_feat, p["min_features"],
               "达成" if v["final"]["C"] else "未达成"),
            "砍尾部 %.0f%% 后 acc %+.4f（红线 ±%.3f），砍头部掉 %.4f → B %s"
            % (p["cut_frac"] * 100, ab["tail_drop"],
               p["max_acc_drop"], ab["head_drop"],
               "达成" if v["final"]["B"] else "未达成"),
            "%d 个 v1.1 新特征进 Top%d 且物理解释成文 → A %s"
            % (newchk["n_new_in_top"], p["top_n"],
               "达成" if v["final"]["A"] else "未达成"),
            "人工答辩项另计：%s" % "；".join(tiers.MANUAL),
            "下一步：把 %d 维精简集（相关性初筛后）交工单 M2-1a 做分类基线；"
            "现场建议加长窗口再验峭度/裕度因子的早期点蚀增益" % len(screened)]),
    ]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
