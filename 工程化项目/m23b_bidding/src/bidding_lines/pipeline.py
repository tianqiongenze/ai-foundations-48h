# -*- coding: utf-8 -*-
"""泄漏核查 → 干净窗口/图像数据集 → 六臂实测（RF/MLP/CNN/LSTM）→ 两轮对比
→ 选型决策矩阵（候选×权重×实测得分→结论）→ 四档验收判定（纯函数在 tiers.py）。

四档门槛唯一出处：config/rules.yaml 的 tiers: 段（逐字取自工单 M2-3b
"四、验收标准"），判定只吃本流水线**实测**值——程序不为"赢得竞标/评委加分"
这类人工环节写达标，也不为兜底臂（无真卷积/无真 LSTM）的数字背书。
"""
import os

import numpy as np

from . import core, io_utils, report, style, tiers


def tier_verdict(biz, st, order=tiers.KEYS):
    """兼容入口：折叠"业务实测布尔 + 规范率引擎"成档位判定。

    判定逻辑唯一出处在 tiers.fold——本函数只转发，杜绝两套口径漂移；
    完整实测判定（拿原始测量值比门槛）走 tiers.verdict。
    """
    return tiers.fold({k: biz.get(k) for k in order}, st, order)


def _style_lines(st):
    """规范率引擎结果 → 报告行（引擎/口径/文件行数违规数/达标判定，全实测）。"""
    return [
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
    ]


def _weights_from(gates_a):
    """从 tiers.A 门槛拼出矩阵权重（和必须为 1，判定与测试都盯这条）。"""
    return {"acc": float(gates_a["weight_acc"]),
            "speed": float(gates_a["weight_speed"]),
            "env": float(gates_a["weight_env"])}


def run(cfg, out_dir):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    g = cfg.tier_gates()
    cnn_target = float(g["C"]["cnn_min_acc"])
    lstm_err_target = float(g["C"]["lstm_max_err"])
    min_style = float(g["D"]["min_code_style_pct"])
    img_labels = ["正常", "气孔", "裂纹"]
    seq_labels = ["健康", "退化"]

    # ① 标签泄漏核查（审计留证）：旧构造 vs 修复构造，同一特征函数、同一量化口径
    aud_leak = core.leakage_audit(core.make_series_leaky, core.seq_feats)
    aud_fix = core.leakage_audit(core.make_series, core.seq_feats)

    # ② 拟合矩阵只用修复后数据——泄漏构造的数字只出现在核查段，不进任何模型
    Xi, yi = core.make_images(180, seed=cfg.seed)
    Xs, ys = core.make_series(180, seed=cfg.seed)
    Itr, Ite, yitr, yite = train_test_split(Xi, yi, test_size=0.3, stratify=yi,
                                            random_state=7)
    W, yw = core.make_windows(Xs, ys, win=16, stride=8)
    Wtr, Wte, ywtr, ywte = train_test_split(W, yw, test_size=0.3, stratify=yw,
                                            random_state=7)

    # ③ 六臂实测（选型矩阵：指标+训练秒全部真跑）
    Fimg_tr = np.array([core.img_feats(im) for im in Itr])
    Fimg_te = np.array([core.img_feats(im) for im in Ite])
    Fwin_tr = np.array([core.win_feats(w) for w in Wtr])
    Fwin_te = np.array([core.win_feats(w) for w in Wte])
    matrix = []
    r = core.fit_eval(RandomForestClassifier(random_state=0), Fimg_tr, yitr,
                      Fimg_te, yite)
    matrix.append({"line": "图像线", "arm": "传统ML(RF)", "engine": "sklearn", **r})
    from sklearn.preprocessing import StandardScaler
    isc = StandardScaler().fit(Itr.reshape(len(Itr), -1))
    r = core.fit_eval(MLPClassifier(hidden_layer_sizes=(64,), random_state=0, max_iter=200),
                      isc.transform(Itr.reshape(len(Itr), -1)), yitr,
                      isc.transform(Ite.reshape(len(Ite), -1)), yite)
    matrix.append({"line": "图像线", "arm": "MLP", "engine": "sklearn", **r})
    cnn = core.torch_cnn(Itr, yitr, Ite, yite, img_labels)
    matrix.append({"line": "图像线", "arm": "CNN(Conv2d)", "engine": cnn["engine"],
                   "acc": cnn["acc"], "sec": cnn["sec"]})
    r = core.fit_eval(RandomForestClassifier(random_state=0), Fwin_tr, ywtr,
                      Fwin_te, ywte)
    matrix.append({"line": "时序线", "arm": "传统ML(RF)", "engine": "sklearn", **r})
    r = core.fit_eval(MLPClassifier(hidden_layer_sizes=(32,), random_state=0, max_iter=200,
                                    early_stopping=True),
                      Wtr.reshape(len(Wtr), -1), ywtr, Wte.reshape(len(Wte), -1), ywte)
    matrix.append({"line": "时序线", "arm": "MLP", "engine": "sklearn", **r})
    lstm = core.torch_lstm(Wtr, ywtr, Wte, ywte, seq_labels)
    matrix.append({"line": "时序线", "arm": "LSTM(nn.LSTM)", "engine": lstm["engine"],
                   "acc": lstm["acc"], "sec": lstm["sec"]})
    cnn_row = next(m for m in matrix if m["arm"].startswith("CNN"))
    lstm_row = next(m for m in matrix if m["arm"].startswith("LSTM"))

    # ④ 及格D"指标可复现"：同种子重复拟合比对（真跑两遍）
    rp1 = core.fit_eval(RandomForestClassifier(random_state=0), Fwin_tr, ywtr,
                        Fwin_te, ywte)
    rp2 = core.fit_eval(RandomForestClassifier(random_state=0), Fwin_tr, ywtr,
                        Fwin_te, ywte)
    repro = rp1["acc"] == rp2["acc"]

    # 两路线基线 CV（保留原报告口径，数据为修复后）
    a = core.cv([core.img_feats(im) for im in Xi], yi)
    b = core.cv([core.seq_feats(x) for x in Xs], ys)
    pick = max(matrix, key=lambda m: (m["acc"], -m["sec"]))

    # ---------- D 档次条：代码规范率（工单 M2-3b D 档 ≥70%，必须是量出来的数）
    st = style.code_style_rate(style.project_root(
        os.path.abspath(__file__)), min_pass=min_style)

    # ---------- B 档可测半：CNN vs LSTM 两轮实测对比（同口径、同留出集切法）
    two_arm_real = bool(cnn_row["engine"] == "torch"
                        and lstm_row["engine"] == "torch")
    acc_gap = round(cnn_row["acc"] - lstm_row["acc"], 4)
    sec_gap = round(cnn_row["sec"] - lstm_row["sec"], 4)
    two_arm_lines = [
        "CNN（图像线，engine %s）：留出 acc %.4f，训练 %.3fs；口径：%s" % (
            cnn_row["engine"], cnn_row["acc"], cnn_row["sec"], cnn["note"]),
        "LSTM（时序线，engine %s）：留出 acc %.4f → 窗口错误率 %.4f（工单线 ≤%.2f），"
        "训练 %.3fs；口径：%s" % (
            lstm_row["engine"], lstm_row["acc"], 1.0 - lstm_row["acc"],
            lstm_err_target, lstm_row["sec"], lstm["note"]),
        "对比结论：acc 差 %+.4f 偏向 %s；训练秒差 %+.3fs 偏向 %s——"
        "'二选一'不凭单指标，最终按选型决策矩阵的加权实测分定" % (
            acc_gap, "CNN" if acc_gap >= 0 else "LSTM",
            sec_gap, "LSTM" if sec_gap > 0 else "CNN"),
        "两轮对比齐备性（程序实测）：%s" % (
            "两臂均为真引擎（torch），指标同口径可比" if two_arm_real
            else "存在兜底臂（无真卷积/无真 LSTM），对比不成立——"
                 "此情形下 B 档可测半不判达标")]

    # ---------- A 档：选型决策矩阵（候选×权重×实测得分→结论，core.decision_matrix）
    weights = _weights_from(g["A"])
    dm = core.decision_matrix(matrix, weights)

    # 决策记录段先拼好：B 档"决策记录完整"实测的就是这段真实文本
    decision_lines = [
        "候选：①图像线 CNN(nn.Conv2d) ②时序线 LSTM(nn.LSTM) ③传统ML(RF/MLP) 基线对照"
        "——工单要求'二选一'，③只作对照臂，深度臂不达标不缩窄就竞标等于抽签",
        "决策口径：先做泄漏核查（旧构造泄漏已实锤并从拟合矩阵剔除），再同口径比较 "
        "acc×训练秒×部署依赖，由选型决策矩阵按权重（rules.yaml tiers.A）定唯一推荐",
        "实测依据：六臂矩阵全部真跑产生（engine 字段如实标 torch/sklearn/兜底）；"
        "CNN 用留出 acc，LSTM 用窗口错误率（1-acc）直接对工单 C 档线——不互相换口径",
        "工时账：整矩阵单次运行 1 分钟内课堂可复跑；数据增强/预训练是下一档成本，"
        "本工单不花——竞标书只承诺实测达成部分，'基线≠上限'的风险如实写",
        "留痕：output/report.txt + result.json + decision_matrix.csv + "
        "docs/09-选型决策矩阵.md；竞标胜负/评委加分是现场人评，程序只备证据包",
    ]
    decision_chars = sum(len(x) for x in decision_lines)

    gA = g["A"]
    m = {
        "reproducible": bool(repro),
        "cnn_engine": cnn_row["engine"],
        "cnn_acc": cnn_row["acc"],
        "lstm_engine": lstm_row["engine"],
        "lstm_err": round(1.0 - lstm_row["acc"], 4),
        "two_arm_real": two_arm_real,
        "decision_chars": decision_chars,
        "matrix_rows": len(dm["rows"]),
        "weights_sum": dm["weights_sum"],
        "conclusion_chars": len(dm["conclusion"]),
        "evidence": {
            "D": "同种子两次 RF 拟合留出 acc %s（%.4f vs %.4f）；规范率 %.2f%%" % (
                "一致" if repro else "漂移", rp1["acc"], rp2["acc"], st["rate"]),
            "C": "CNN(%s) acc=%.4f（线≥%.2f）｜LSTM(%s) 窗口错误率=%.4f（线≤%.2f）"
                 "——兜底臂数字不认（require_real_engine=%s）" % (
                     cnn_row["engine"], cnn_row["acc"], cnn_target,
                     lstm_row["engine"], round(1.0 - lstm_row["acc"], 4),
                     lstm_err_target, g["C"]["require_real_engine"]),
            "B": "两轮对比：%s；决策记录 %d 字（线 %d）" % (
                "两臂真引擎同口径实测（acc 差 %+.4f、秒差 %+.3fs）" % (acc_gap, sec_gap)
                if two_arm_real else "存在兜底臂，对比不成立",
                decision_chars, g["B"]["min_decision_chars"]),
            "A": "矩阵 %d 行（线≥%d）；权重 acc/speed/env=%.1f/%.1f/%.1f 和=%.1f；"
                 "结论成文 %d 字（线≥%d）" % (
                     len(dm["rows"]), gA["min_matrix_rows"], weights["acc"],
                     weights["speed"], weights["env"], dm["weights_sum"],
                     len(dm["conclusion"]), gA["min_conclusion_chars"]),
        },
    }
    v = tiers.verdict(m, st, g)

    d_measured = (
        "同种子两次 RF 拟合留出 acc 完全一致=%s（%.4f vs %.4f）；代码规范率 "
        "%.2f%%（%s，%d 文件/%d 行/%d 违规，线 %.0f%%）" % (
            repro, rp1["acc"], rp2["acc"], st["rate"], st["engine"],
            st["n_files"], st["code_lines"], st["violations"], min_style))
    c_measured = (
        "CNN(%s) 实测 acc %.4f（线≥%.2f）｜LSTM(%s) 实测窗口错误率 %.4f"
        "（线≤%.2f）→ 至少一臂真引擎达标才算数" % (
            cnn_row["engine"], cnn_row["acc"], cnn_target,
            lstm_row["engine"], m["lstm_err"], lstm_err_target))
    b_measured = (
        "两轮对比 %s（acc 差 %+.4f、训练秒差 %+.3fs，同留出集口径）；"
        "决策记录成文 %d 字（线 %d）；竞标胜负另人工" % (
            "齐备" if two_arm_real else "不齐备（兜底臂不认）",
            acc_gap, sec_gap, decision_chars, g["B"]["min_decision_chars"]))
    a_measured = (
        "决策矩阵实测 %d 行×3 准则（性能/速度/环境全由实测换算），权重和 %.1f，"
        "结论：%s" % (len(dm["rows"]), dm["weights_sum"], dm["conclusion"]))
    a_req = ("给出选型决策矩阵（候选×权重×实测得分→结论，"
             "≥%d 行、权重和=1）" % gA["min_matrix_rows"])
    tiers_detail = {
        "D": {
            "要求": "指标可复现 + 代码规范率≥%.0f%%" % min_style,
            "实测": d_measured,
            "达标": v["final"]["D"],
        },
        "C": {
            "要求": "CNN≥%.0f%% 或 LSTM 误差≤%.0f%%（真卷积/真 LSTM 臂口径）" % (
                cnn_target * 100, lstm_err_target * 100),
            "实测": c_measured,
            "达标": v["final"]["C"],
        },
        "B": {
            "要求": "CNN vs LSTM 两轮实测对比 + 决策记录完整（竞标胜负/评委加分人工）",
            "实测": b_measured,
            "达标": v["final"]["B"],
        },
        "A": {"要求": a_req, "实测": a_measured,
              "达标": v["final"]["A"]},
    }

    result = {"leakage_audit_leaky": aud_leak, "leakage_audit_fixed": aud_fix,
              "image_line_cv": a, "series_line_cv": b,
              "selection_matrix": matrix, "cnn_arm": cnn, "lstm_arm": lstm,
              "reproducible": repro, "pick": pick,
              "n_windows": len(W), "window": {"win": 16, "stride": 8},
              "code_style": st, "min_code_style": min_style,
              "tier_gates": g,
              "decision_matrix": dm, "decision_record_chars": decision_chars,
              "two_arm_real": two_arm_real,
              "claimed_tier": v["claimed"],
              "tier_verdict": v,
              "tiers": tiers_detail}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    io_utils.write_csv_rows(
        os.path.join(out_dir, "decision_matrix.csv"), dm["rows"])
    mtx_tbl = io_utils.fmt_table(
        [[m["line"], m["arm"], m["engine"], "%.4f" % m["acc"], "%.3f" % m["sec"]]
         for m in matrix],
        ["路线", "方案", "引擎", "测试acc", "训练秒"])
    dm_tbl = io_utils.fmt_table(
        [[r["line"], r["arm"], r["engine"], "%.4f" % r["acc"],
          "%.3f" % r["sec"], "%.4f" % r["s_perf"], "%.4f" % r["s_speed"],
          "%.2f" % r["s_env"], "%.4f" % r["score"]] for r in dm["rows"]],
        ["路线", "方案", "引擎", "测试acc", "训练秒",
         "性能分", "速度分", "环境分", "加权总分"])
    leak_tbl = io_utils.fmt_table(
        [["旧构造(泄漏演示)", "%.4f" % aud_leak["overlap_frac"],
          "%.4f" % aud_leak["single_feat_cv"],
          "泄漏实锤" if aud_leak["leak_detected"] else "未见泄漏"],
         ["修复构造(进拟合矩阵)", "%.4f" % aud_fix["overlap_frac"],
          "%.4f" % aud_fix["single_feat_cv"],
          "泄漏实锤" if aud_fix["leak_detected"] else "未见泄漏"]],
        ["数据构造", "类间值域重叠度", "单特征CV", "判定"])

    # ---------- 决策矩阵同步成文进 docs（工单 A 档要求矩阵"给出"，报告+文档两处都要有）
    docs_path = os.path.join(style.project_root(os.path.abspath(__file__)),
                             "docs", "09-选型决策矩阵.md")
    docs_lines = [
        "# 选型决策矩阵（工单 M2-3b 优(A) 档证据）",
        "",
        "候选 × 权重 × 实测得分 → 结论。全部数字由 `python run.py` 真实跑出，",
        "得分口径：性能分=留出测试 acc；速度分=最快臂训练秒/本臂训练秒；",
        "环境分=实测引擎的部署依赖映射（sklearn 1.0 / torch 0.5 / 兜底 0.25）。",
        "权重外置于 config/rules.yaml 的 tiers.A（acc/speed/env=%.1f/%.1f/%.1f）。"
        % (weights["acc"], weights["speed"], weights["env"]),
        "",
        mtx_tbl,
        "",
        dm_tbl,
        "",
        "两轮对比（工单'二选一'的可测半）：",
        "  " + two_arm_lines[0],
        "  " + two_arm_lines[1],
        "  " + two_arm_lines[2],
        "",
        dm["conclusion"],
        "",
        "人工环节（程序不判）：%s" % "；".join(tiers.MANUAL),
        "",
    ]
    io_utils.write_text(docs_path, "\n".join(docs_lines))

    sections = [
        ("输入口径", ["图像线 180 张×3类｜时序线 180 条（泄漏修复版）→ 窗口 win=16/stride=8 "
                  "共 %d 窗×2类｜30%% 留出测试集，seed=%d" % (len(W), cfg.seed)]),
        ("标签泄漏核查（审计发现，如实入报）", [
            leak_tbl,
            "发现：旧 make_series 用样本号 i%%2 同时决定标签与幅值斜率——seq_feats 比值只是"
            "标签的另一种写法（类间重叠 %.4f、单特征 CV %.4f），旧报告 0.9944 属泄漏虚高"
            % (aud_leak["overlap_frac"], aud_leak["single_feat_cv"]),
            "处置：泄漏构造已从拟合矩阵剔除（仅存于本核查段）；修复构造引入连续隐退化强度+"
            "随机基线/相位，类间重叠升至 %.4f、单特征 CV 回落至 %.4f——后续所有模型指标"
            "只在修复数据上实测" % (aud_fix["overlap_frac"], aud_fix["single_feat_cv"]),
            "旧构造各区间：%s；修复后：%s" % (aud_leak["class_ranges"], aud_fix["class_ranges"])]),
        ("选型矩阵（传统ML vs MLP vs CNN vs LSTM，实测指标+训练秒）", [
            mtx_tbl,
            "CNN 口径：%s" % cnn["note"],
            "LSTM 口径：%s" % lstm["note"]]),
        ("两轮对比（CNN vs LSTM，工单'二选一'的实测根据）", two_arm_lines),
        ("选型决策矩阵（候选×权重×实测得分→结论）", [
            dm_tbl,
            "权重（rules.yaml tiers.A）：acc/speed/env = %.1f/%.1f/%.1f，和 %.1f；"
            "得分全部由实测换算，口径见表头" % (
                weights["acc"], weights["speed"], weights["env"],
                dm["weights_sum"]),
            dm["conclusion"],
            "同步落盘：output/decision_matrix.csv + docs/09-选型决策矩阵.md"]),
        ("两路线基线 CV（修复后数据）", [
            "图像线(RF@块统计) CV = %.4f｜时序线(RF@斜率指纹) CV = %.4f" % (a, b),
            "提示：图像线合成模式显著、CV 高企属数据设计使然；时序线 %.4f 才是去泄漏后的真实难度"
            % b]),
        ("竞标建议", [
            "按实测最优（单指标）：%s-%s（acc=%.4f，训练 %.3fs，%s）" % (
                pick["line"], pick["arm"], pick["acc"], pick["sec"],
                pick["engine"]),
            "按决策矩阵加权总分（多指标）：%s-%s（%.4f 分）——竞标书以此为推荐口径" % (
                dm["top"]["line"], dm["top"]["arm"], dm["top"]["score"]),
            "竞标书三要素齐备：方案对比=矩阵 %d 行；实验证据=全部实测+泄漏核查；"
            "风险承诺=基线≠上限，真增益需数据量/算力/标注成本复核" % len(matrix)]),
        ("决策记录", decision_lines),
        ("验收档位判定（实测）", tiers.render_lines(v)),
        ("验收对照明细（工单 M2-3b 四档逐条证据）",
         ["%s 层 [%s] %s ｜ %s" % (
             k, "达标" if tiers_detail[k]["达标"] else "未达标",
             tiers_detail[k]["要求"], tiers_detail[k]["实测"])
          for k in tiers.KEYS]),
        ("结论与建议", [
            "实测达成档位：%s（判定只认实测值，见「验收档位判定（实测）」段）"
            % v["claimed"],
            "指标可复现 %s + 规范率 %.2f%% → D %s" % (
                "（同种子两跑一致）" if repro else "（存在漂移）", st["rate"],
                "达成" if v["final"]["D"] else "未达成"),
            "真引擎臂实测过线（CNN %.4f / LSTM 误差 %.4f）→ C %s" % (
                cnn_row["acc"], m["lstm_err"],
                "达成" if v["final"]["C"] else "未达成"),
            "两轮对比 %s + 决策记录 %d 字 → B（可测半）%s" % (
                "齐备" if two_arm_real else "不齐备", decision_chars,
                "达成" if v["final"]["B"] else "未达成"),
            "选型决策矩阵 %d 行、权重和 %.1f、结论成文 → A %s" % (
                len(dm["rows"]), dm["weights_sum"],
                "达成" if v["final"]["A"] else "未达成"),
            "人工答辩项另计：%s" % "；".join(tiers.MANUAL),
            "先查泄漏再谈指标——去泄漏后 CNN/LSTM 的数字才配进决策矩阵；"
            "选型看 acc×训练秒×部署依赖三列，不只看分数"])]
    sections += [
        ("代码规范率（D 档次条：≥%.0f%%）" % min_style, _style_lines(st)),
    ]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
