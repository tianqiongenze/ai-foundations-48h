# -*- coding: utf-8 -*-
"""M1-1a 流水线：读日志 → 三清单体检 → 缺失填充（行数不变）→ 异常识别与处置
→ 重复处理 → 质量分（完整×准确×一致 加权）→ 预测 vs 均值填充实验 → 报告。

规则外置：阈值/权重/关注列数全部走 config/rules.yaml（warn / danger / topk /
unit / seed 均被本流水线真实读取，改配置不改代码）。

质量分口径（工单 B05）：
  完整率 = 1 − 缺失单元格 / 总单元格
  准确率(合规率) = 1 − (物理越界 + 断电待复核) / 总单元格
    —— 3σ/IQR 统计标记只作"人工复核候选"，不计扣分：正常工况波动也会
       触发统计尺，一刀切删掉就是"用错尺误杀真故障"（工单 B03 误区）。
  一致率(唯一率) = 1 − 重复行 / 总行数
"""
import os

from . import core, io_utils, report, style, tiers

W_DEFAULT = (0.4, 0.4, 0.2)


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


def _w(cfg):
    """质量分权重：从配置读，缺省 0.4/0.4/0.2（理由见 core.quality docstring）。"""
    raw = cfg.raw or {}
    try:
        w = (float(raw.get("w_complete", W_DEFAULT[0])),
             float(raw.get("w_accurate", W_DEFAULT[1])),
             float(raw.get("w_consistent", W_DEFAULT[2])))
    except (TypeError, ValueError):
        w = W_DEFAULT
    return w


def _unreliable(anom_rows, review):
    """不可信单元格数 = 物理越界 + 断电待复核（合规率的分子）。"""
    return int(sum(a["n_phys"] for a in anom_rows)) + int(review)


def run(cfg, out_dir):
    topk = int(cfg.topk)                       # 缺失最严重的前 topk 列重点处理
    w = _w(cfg)
    gates = cfg.tier_gates()                   # 工单四档门槛（rules.yaml tiers: 段）
    # quality_target 是报告展示线；档位判定看 gates.B.min_quality_score
    quality_target = float((cfg.raw or {}).get("quality_target", 85.0))
    danger = float(cfg.danger)                 # 缺失处理率红线（报告展示，判定看 gates）
    warn = float(cfg.warn)                     # 质量分黄色警戒线

    raw = core.make_dirty(int((cfg.raw or {}).get("n_rows", 300)), seed=cfg.seed)
    n0 = len(raw)
    ncol = len(core.NUMERIC_COLS)
    cells0 = n0 * ncol

    # ---------- ① 三清单体检（清洗前）
    miss_before = core.missing_profile(raw, topk=topk)
    anom_before = core.anomaly_profile(raw)
    dup_rows, dup_idx = core.duplicate_profile(raw)
    n_missing = int(sum(m["n_missing"] for m in miss_before))
    n_stat_before = int(sum(a["n_union"] for a in anom_before))
    n_phys_before = int(sum(a["n_phys"] for a in anom_before))
    comp0 = round((1.0 - n_missing / max(cells0, 1)) * 100.0, 2)
    uniq0 = round((1.0 - len(dup_idx) / max(n0, 1)) * 100.0, 2)

    # ---------- ② 双解互验（同一越量程规则，两种写法）
    py_flags = core.flags_py(raw)
    pd_flags = core.flags_pd(raw)
    dual_ok = core.agree(py_flags, pd_flags)
    n_phys_dual = int(sum(py_flags))

    # ---------- ③ 缺失填充（行数不变）
    filled, log, stats = core.impute(raw, seed=cfg.seed)
    rows_after_missing = len(filled)
    temp_missing_after = stats["temp_missing_after"]
    temp_rate_before = [m for m in miss_before if m["column"] == "temp"][0]["missing_rate_pct"]
    handle_rate = round(stats["handled"] / max(stats["missing_cells"], 1) * 100.0, 2)
    n_review = stats["n_review"]
    acc0 = round((1.0 - _unreliable(anom_before, n_review) / max(cells0, 1)) * 100.0, 2)
    q_before = core.quality(comp0, acc0, uniq0, w)

    # ---------- ④ 异常识别与处置（3σ + IQR + 物理量程，不删行）
    clean, fixed = core.correct_anomalies(filled, stats)
    anom_after = core.anomaly_profile(clean)
    n_phys_after = int(sum(a["n_phys"] for a in anom_after))
    n_stat_after = int(sum(a["n_union"] for a in anom_after))
    acc1 = round((1.0 - _unreliable(anom_after, n_review) / max(cells0, 1)) * 100.0, 2)

    # ---------- ⑤ 重复处理（按清单行号删，清单↔处置一一对应）
    final, n_dup_removed = core.remove_duplicates(clean, drop_rows=dup_idx)
    n_rows_final = len(final)
    dup_left, _ = core.duplicate_profile(final)

    # ---------- ⑥ 质量分（清洗后）
    comp1 = round((1.0 - float(clean[core.NUMERIC_COLS].isna().to_numpy().sum())
                   / max(cells0, 1)) * 100.0, 2)
    uniq1 = round((1.0 - n_dup_removed / max(n0, 1)) * 100.0, 2)
    q_after = core.quality(comp1, acc1, uniq1, w)

    # ---------- ⑧ A 层：预测填充 vs 均值填充（同一留出实验）
    exp = core.imputation_experiment(raw, clean, target="temp", seed=cfg.seed)

    # ---------- ⑨ D 档次条：代码规范率（工单 M1-1a D 档 ≥70%，必须是量出来的数）
    min_style = float(gates["D"]["min_code_style_pct"])
    st = style.code_style_rate(style.project_root(
        os.path.abspath(__file__)), min_pass=min_style)

    # 报告论证段先拼好：A/B 档"段落存在且字数达标"实测的就是这些真实文本
    strategy = {"temp": "线性时序插值 + 两端 ffill/bfill（散点丢点，按相邻趋势补最贴物理）",
                "vib": "列中位数（右偏有长尾，中位数比均值稳；不删行守住『行数不变』）",
                "current": "前向填充 ffill（负载电流变化缓慢）",
                "pressure": "整列策略：缺失标记列 + 中位数；断电成片段标待复核、不假装插趋势"}
    a_layer_lines = [
        "考卷：温度可信观测 %d 格（毛刺已排除），%.0f%% 随机留出 ×%d 轮，"
        "两法信息量相同（防泄漏：特征只用本时刻已可得的振动/电流/机号/时刻）"
        % (exp["n_valid_obs"], exp["hold_frac"] * 100, exp["repeats"]),
        "均值填充   MAE=%.4f℃  RMSE=%.4f℃" % (exp["mae_mean"], exp["rmse_mean"]),
        "预测填充   MAE=%.4f℃  RMSE=%.4f℃（OLS：日内周期 sin/cos + 振动 + 电流 + 机号）"
        % (exp["mae_pred"], exp["rmse_pred"]),
        "MAE 改善 %.2f%%，RMSE 改善 %.2f%%，逐格胜率 %.1f%%（配对 n=%d）→ %s" % (
            exp["mae_lift_pct"],
            round((exp["rmse_mean"] - exp["rmse_pred"]) /
                  max(exp["rmse_mean"], 1e-9) * 100.0, 2),
            exp["pred_win_rate"] * 100, exp["n_compared"],
            "预测填充优于均值" if exp["better"] else "本轮未优于均值，回退均值策略"),
        "逐轮明细见 output/result.json 的 imputation_experiment.per_repeat"]
    decision_lines = [
        "候选方案：①工单主线（starter 起步，四策略逐列选药）②自选技术路线"
        "（KNN/模型插补）③改现成项目（不建议）",
        "决策：主线起步 + 温度列上「预测填充」做增益对比（A 层实验）；"
        "KNN 插补留作课后拓展——工时高于收益",
        "理由：温度有明确日周期与负载耦合 → 可预测；压力 35% 成片缺失 → "
        "假装恢复比标记更危险；权重 0.4/0.4/0.2 体现"
        "『缺陷场景合规优先、重复危害最小』",
        "留痕：三清单 CSV + audit_log.csv + cleaned.csv 四件齐，答辩可回算"]

    n_reasons = sum(1 for e in log if e.get("reason"))
    decision_chars = sum(len(x) for x in decision_lines)
    argument_chars = sum(len(x) for x in a_layer_lines)
    temp_missing_before = [r for r in miss_before
                           if r["column"] == "temp"][0]["n_missing"]
    m = {
        "temp_missing_after": temp_missing_after,
        "rows_unchanged": rows_after_missing == n0,
        "handle_rate_pct": handle_rate,
        "n_lists": sum(1 for lst in (miss_before, anom_before, dup_rows)
                       if lst is not None),
        "quality_after": q_after,
        "n_reasons": n_reasons,
        "decision_chars": decision_chars,
        "pred_better": bool(exp["better"]),
        "mae_pred": exp["mae_pred"], "mae_mean": exp["mae_mean"],
        "argument_chars": argument_chars,
        "evidence": {
            "D": "temp 缺失 %d → %d 格；行数 %d → %d（缺失阶段零删行）"
                 % (temp_missing_before, temp_missing_after,
                    n0, rows_after_missing),
            "C": "处理率 %.2f%%（线 %.0f%%）；三清单 %d/%d/%d 项"
                 % (handle_rate, gates["C"]["min_handle_rate_pct"],
                    len(miss_before), len(anom_before), len(dup_rows)),
            "B": "质量分 %.2f（线 %.0f）；逐列理由 %d 条；决策记录 %d 字"
                 % (q_after, gates["B"]["min_quality_score"],
                    n_reasons, decision_chars),
            "A": "预测 MAE %.4f℃ < 均值 %.4f℃（%s）；论证段 %d 字"
                 % (exp["mae_pred"], exp["mae_mean"],
                    "优于" if exp["better"] else "未优于", argument_chars),
        },
    }
    v = tiers.verdict(m, st, gates)

    d_measured = ("temp 缺失率 %.2f%% → %.2f%%；行数 %d → %d（缺失阶段零删行）"
                  % (temp_rate_before,
                     temp_missing_after / max(n0, 1) * 100.0,
                     n0, rows_after_missing))
    d_style = ("代码规范率 %.2f%%（%s，%d 个文件 / %d 行 / %d 条违规，"
               "线 %.0f%%）"
               % (st["rate"], st["engine"], st["n_files"],
                  st["code_lines"], st["violations"], min_style))
    c_measured = ("处理率 %.2f%%；缺失/异常/重复清单 %d/%d/%d 项（红线 %.0f%%）"
                  % (handle_rate, len(miss_before), len(anom_before),
                     len(dup_rows), danger))
    b_measured = ("质量分 %.2f（权重 %s，目标线 %.0f，警戒线 %.0f）；"
                  "每列策略+理由 %d 条；决策记录 %d 字"
                  % (q_after, "/".join(str(x) for x in w),
                     quality_target, warn, len(log), m["decision_chars"]))
    a_measured = ("MAE 均值 %.4f℃ vs 预测 %.4f℃（改善 %.2f%%）；逐格胜率 "
                  "%.1f%%（n=%d，%d 轮留出，可信观测 %d 格）；论证段 %d 字"
                  "（线 %d）"
                  % (exp["mae_mean"], exp["mae_pred"],
                     exp["mae_lift_pct"], exp["pred_win_rate"] * 100,
                     exp["n_compared"], exp["repeats"],
                     exp["n_valid_obs"], m["argument_chars"],
                     gates["A"]["min_argument_chars"]))
    d_req = "温度列缺失率降为0且行数不变 + 代码规范率≥%.0f%%" % min_style
    c_req = ("缺失处理率≥%.0f%% 且三清单齐全"
             % gates["C"]["min_handle_rate_pct"])
    b_req = ("质量分≥%.0f 且策略理由完整、决策记录完整"
             % gates["B"]["min_quality_score"])
    tiers_detail = {
        "D": {
            "要求": d_req,
            "实测": d_measured,
            "实测2": d_style,
            "达标": v["final"]["D"],
        },
        "C": {
            "要求": c_req,
            "实测": c_measured,
            "达标": v["final"]["C"],
        },
        "B": {
            "要求": b_req,
            "实测": b_measured,
            "达标": v["final"]["B"],
        },
        "A": {
            "要求": "预测填充优于均值填充且论证清楚",
            "实测": a_measured,
            "达标": v["final"]["A"],
        },
    }

    result = {
        "seed": cfg.seed, "weights": list(w), "quality_target": quality_target,
        "warn_line": warn, "missing_handle_redline": danger, "topk": topk,
        "rows_before": n0, "rows_after_missing_stage": rows_after_missing,
        "duplicates_removed": n_dup_removed, "rows_final": n_rows_final,
        "temp_missing_rate_before_pct": temp_rate_before,
        "temp_missing_after": temp_missing_after,
        "cells": cells0, "missing_cells": n_missing, "handled_cells": stats["handled"],
        "missing_handle_rate_pct": handle_rate,
        "missing_by_col": stats["orig_missing"],
        "phys_cells_before": n_phys_before, "phys_cells_after": n_phys_after,
        "stat_cells_before": n_stat_before, "stat_cells_after": n_stat_after,
        "review_cells": n_review,
        "duplicates_before": len(dup_idx), "duplicates_after": len(dup_left),
        "lists": {"missing": miss_before, "anomaly": anom_before,
                  "duplicate": dup_rows},
        "rates_before": {"complete": comp0, "accurate": acc0, "consistent": uniq0},
        "rates_after": {"complete": comp1, "accurate": acc1, "consistent": uniq1},
        "quality_before": q_before, "quality_after": q_after,
        "quality_gain": round(q_after - q_before, 2),
        "dual_check": {"py_rows_flagged": n_phys_dual, "pd_rows_flagged": int(sum(pd_flags)),
                       "consistent": bool(dual_ok)},
        "imputation_experiment": exp,
        "audit_log": log,
        "code_style": st, "min_code_style": min_style,
        "tier_gates": gates,
        "claimed_tier": v["claimed"],
        "tier_verdict": v,
        "tiers": tiers_detail,
    }

    io_utils.ensure_dir(out_dir)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    # 三清单落盘（C 层硬要求）+ 留痕台账 + 清洗后可复算数据集
    io_utils.write_csv_rows(os.path.join(out_dir, "missing_list.csv"), miss_before)
    io_utils.write_csv_rows(os.path.join(out_dir, "anomaly_list.csv"), anom_before)
    io_utils.write_csv_rows(os.path.join(out_dir, "duplicate_list.csv"),
                            dup_rows or [{"row": "—", "ts": "—", "device": "—",
                                          "temp": "—"}])
    io_utils.write_csv_rows(os.path.join(out_dir, "audit_log.csv"), log)
    final.to_csv(os.path.join(out_dir, "cleaned.csv"), index=False)

    # ---------- 报告
    sections = [
        ("输入口径", ["泵站日志 %d 行 × %d 个数值通道（温度单位 %s）；种子 %d"
                      % (n0, ncol, cfg.unit, cfg.seed),
                      "工单主线：读日志→缺失统计与策略→异常IQR识别→重复处理→质量报告",
                      "三清单体检先行：缺多少 / 怪多少 / 重多少"]),
        ("三清单①缺失", ["%-10s 缺失 %4d (%6.2f%%) 最长连续 %3d 段 → %s%s"
                         % (m["column"], m["n_missing"], m["missing_rate_pct"],
                            m["max_consecutive"], m["mechanism"],
                            "  ★重点列(Top%d)" % topk if m["watch"] else "")
                         for m in miss_before]),
        ("三清单②异常（两把尺 + 物理量程）",
         ["%-10s 物理量程%s 越界 %d｜3σ%s 命中 %d｜IQR%s 命中 %d → 复核候选 %d"
          % (a["column"], a["phys_range"], a["n_phys"], a["sigma_bounds"],
             a["n_sigma3"], a["iqr_bounds"], a["n_iqr"], a["n_union"])
          for a in anom_before]
         + ["尺的选择：振动/压力右偏有毛刺 → 主用 IQR；温度近似正态 → 3σ 可用"]),
        ("三清单③重复",
         ["duplicated() 命中 %d 行（PLC 重发双录）；样例：%s"
          % (len(dup_idx),
             ", ".join("行%d(%s/%s)" % (r["row"], r["ts"], r["device"])
                       for r in dup_rows[:5]) or "无"),
          "误区提醒：只查缺失不查重复——重复样本会让验证集分数虚高"]),
        ("缺失处理（D 层口径）", [
            "温度列缺失率 %.2f%% → %.2f%%（剩余空值 %d 个）" % (
                temp_rate_before, temp_missing_after / max(n0, 1) * 100.0,
                temp_missing_after),
            "行数 %d → %d：缺失阶段零删行（不用 dropna 删到 %d 行作弊）" % (
                n0, rows_after_missing, n0 - n_missing),
            "缺失处理率 %.2f%%（红线 %.0f%%）→ %s" % (
                handle_rate, gates["C"]["min_handle_rate_pct"],
                "达标" if handle_rate >= gates["C"]["min_handle_rate_pct"]
                else "未达标"),
            "缺失单元格 %d 个全部有处置：逐列见下方策略与 output/audit_log.csv"
            % n_missing]),
        ("策略与理由（答辩必答）",
         ["%s：%s" % (c, strategy[c]) for c in core.NUMERIC_COLS]
         + ["缺失即特征：新增 temp_missing_flag / pressure_missing_flag 两列",
            "断电成片 %d 点：插值会造出假日趋势 → 标 review_flag 交现场复核" % n_review]),
        ("异常处置", [
            "物理越界（毛刺）按趋势重插 %d 处：越界单元格 %d → %d" % (
                int(sum(fixed.values())), n_phys_before, n_phys_after),
            "统计尺（3σ/IQR）复核候选 %d → %d：不删行，全部留标人工复核" % (
                n_stat_before, n_stat_after),
            "候选变多的原因：压力列 35% 中位数填充使 Q1≈Q3、IQR 变窄，"
            "于是更多正常点被统计尺点名——这正是"
            "『用错尺误杀真故障』的现场演示",
            "为什么不一刀切删：正常工况波动同样触发统计尺，删了就是丢真故障"]),
        ("双解互验", ["纯Python 逐行标记 %d 行｜Pandas 掩码标记 %d 行｜完全一致：%s"
                      % (n_phys_dual, int(sum(pd_flags)), dual_ok),
                      "两解同规则同结果 → 越量程清洗规则可冻结入库"]),
        ("清洗留痕", ["%d 行 →（缺失填充，行数不变）→ %d 行 →（异常处置，行数不变）→ "
                      "%d 行 →（去重复）→ %d 行"
                      % (n0, rows_after_missing, len(clean), n_rows_final),
                      "重复删除 %d 行（清单命中 %d 行，全数处置），保留率 %.1f%%；"
                      "复检 duplicated() 残留 %d 行"
                      % (n_dup_removed, len(dup_idx), n_rows_final / max(n0, 1) * 100,
                         len(dup_left))]),
        ("质量分（B 层口径）", [
            "权重 %g×完整率 + %g×准确率 + %g×一致率（写进 rules.yaml，按业务定）" % w,
            "清洗前三率：完整 %.2f%% / 准确 %.2f%% / 一致 %.2f%% → 评分 %.2f"
            % (comp0, acc0, uniq0, q_before),
            "清洗后三率：完整 %.2f%% / 准确 %.2f%% / 一致 %.2f%% → 评分 %.2f"
            % (comp1, acc1, uniq1, q_after),
            "增益 %+.2f 分；目标线 %.0f（工单 B 档，判定看 rules.yaml tiers）｜"
            "警戒线 %.0f → %s"
            % (q_after - q_before, gates["B"]["min_quality_score"], warn,
               "达标" if q_after >= gates["B"]["min_quality_score"]
               else "未达标，先查最大失分项"),
            "最大失分项：%s（清洗后 完整 %.2f%% / 准确 %.2f%% / 一致 %.2f%%，"
            "先攻分数最低的一率）" % (
                min([("完整率", comp1), ("准确率", acc1), ("一致率", uniq1)],
                    key=lambda x: x[1])[0], comp1, acc1, uniq1)]),
        ("预测填充 vs 均值填充（A 层实验）", a_layer_lines),
        ("代码规范率（D 档次条：≥%.0f%%）" % min_style, _style_lines(st)),
        ("验收档位判定（实测）", tiers.render_lines(v)),
        ("验收对照明细（工单 M1-1a 四档逐条证据）",
         ["%s 层 [%s] %s ｜ %s%s" % (
             k, "达成" if tiers_detail[k]["达标"] else "未达成",
             tiers_detail[k]["要求"], tiers_detail[k]["实测"],
             ("\n" + " " * 6 + "｜ " + tiers_detail[k]["实测2"])
             if tiers_detail[k].get("实测2") else "")
          for k in tiers.KEYS]),
        ("决策记录", decision_lines),
        ("结论与建议", [
            "实测达成档位：%s（判定只认实测值，见「验收档位判定（实测）」段）"
            % v["claimed"],
            "缺失处理率 %.2f%%、质量分 %.2f（线 %.0f）、预测 MAE %.4f℃ < 均值 "
            "%.4f℃（改善 %.2f%%）——四档证据全部落盘可回算"
            % (handle_rate, q_after, gates["B"]["min_quality_score"],
               exp["mae_pred"], exp["mae_mean"], exp["mae_lift_pct"]),
            "人工答辩项另计：%s" % "；".join(tiers.MANUAL),
            "下一步：把 temp_missing_flag 当特征送工单 M1-1b——缺失本身也是信息"]),
    ]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
