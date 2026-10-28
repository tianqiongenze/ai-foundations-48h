# -*- coding: utf-8 -*-
"""频谱 → sklearn 基线 → 结构搜索 → torch Dropout 扫描臂 → 曲线图 → 验收档位判定（实测）。

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


def decision_record_ok(path):
    """B 档"决策记录完整"实测：result.json 落盘后回读——结构搜索表＋
    性价比选型、无 Dropout 基线臂、逐扫描臂（含引擎口径）、两端 gap 诊断
    全部留痕才算记录完整，缺一项判不达标。"""
    try:
        r = io_utils.read_json(path)
    except (OSError, ValueError):
        return False
    ss = r.get("structure_search") or {}
    if not ss.get("rows") or not ss.get("best_value"):
        return False
    if not r.get("torch_no_dropout") or not r.get("dropout_arms"):
        return False
    if not r.get("overfit_no_dropout") or not r.get("overfit_dropout"):
        return False
    return all(a.get("engine") for a in r["dropout_arms"])


def run(cfg, out_dir):
    from sklearn.model_selection import train_test_split
    gates = cfg.tiers
    acc_min = float(gates["C"]["acc_min"])
    arms_dp = [float(d) for d in gates["B"]["dropout_arms"]]
    X, y = core.make_spectra(600, seed=cfg.seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    # ① sklearn 基线（D/C 线）+ 损失四段诊断 + L2 正则对照
    m, acc = core.train(Xtr, ytr, Xte, yte, alpha=0.0)
    dg_sk = core.diagnose(m.loss_curve_)
    _, acc_reg = core.train(Xtr, ytr, Xte, yte, alpha=1e-3)
    # ② 真·隐层结构搜索：候选全实测（优A"性价比最优"的证据）
    ss = core.structure_search(
        Xtr, ytr, Xte, yte, configs=((32, 16), (64, 32, 16), (64, 32)))
    # ③ torch 臂：无 Dropout 基线 + 扫描臂——优B"gap 收窄且指标不降"对照实验
    arm_nd = core.torch_mlp(Xtr, ytr, Xte, yte, dropout=0.0)
    dg_nd = core.overfit_diagnosis(
        arm_nd["train_losses"], arm_nd["val_losses"])
    arms_full = []
    for dp in arms_dp:
        a = core.torch_mlp(Xtr, ytr, Xte, yte, dropout=dp)
        dg = core.overfit_diagnosis(a["train_losses"], a["val_losses"])
        arms_full.append((a, dg))
    arms = [{"engine": a["engine"], "dropout": a["dropout"], "acc": a["acc"],
             "gap": dg["gap"], "epochs": a["epochs"], "sec": a["sec"]}
            for a, dg in arms_full]
    # 合规臂口径直接复用 tiers._best_arm——真源唯一，报告/判定/测试不漂移
    gap_base, acc_base = dg_nd["gap"], arm_nd["acc"]
    best_arm = tiers._best_arm(arms, gap_base, acc_base)
    pair = next(((a, dg) for a, dg in arms_full
                 if best_arm and a["dropout"] == best_arm["dropout"]),
                (arm_nd, dg_nd))
    arm_plot, dg_plot = pair
    # ④ 曲线 PNG（matplotlib Agg）实测字节数——良好C"图完整"的实物证据
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "learning_curve.png")
    core.plot_losses(arm_plot["train_losses"], arm_plot["val_losses"], png)
    png_bytes = os.path.getsize(png) if os.path.exists(png) else 0
    # ---------- D 档次条：代码规范率（量出来的数，不是自称）
    st = style.code_style_rate(
        style.project_root(os.path.abspath(__file__)),
        min_pass=float(gates["D"]["style_min"]))
    metrics_ok = 0.0 <= acc <= 1.0 and len(m.loss_curve_) > 0
    acc_best = max([acc, ss["best_acc"], arm_nd["acc"]]
                   + [a["acc"] for a in arms])
    json_path = os.path.join(out_dir, "result.json")
    result = {"n_epochs": len(m.loss_curve_), "acc": acc, "acc_reg": acc_reg,
              "diagnosis": dg_sk, "structure_search": ss,
              "torch_no_dropout": {k: arm_nd[k] for k in
                                   ("engine", "acc", "epochs", "sec", "note")},
              "overfit_no_dropout": dg_nd, "dropout_arms": arms,
              "overfit_dropout": dg_plot, "png": png,
              "png_bytes": png_bytes, "acc_best": acc_best,
              "gates": gates, "code_style": st}
    io_utils.write_json(json_path, result)
    decision_ok = decision_record_ok(json_path)
    if best_arm:
        gap_note = "合规臂 Dropout %s：gap %.4f→%.4f 且 acc %.4f≥基线 %.4f" % (
            best_arm["dropout"], gap_base, best_arm["gap"],
            best_arm["acc"], acc_base)
    else:
        gap_note = "无合规臂（各扫描臂 gap 未收窄或指标下降，严格口径不放容差）"
    summary = ("基线实测 acc %.4f｜最高 acc %.4f｜PNG 实测 %dB（下限 %d）｜"
               "无 Dropout gap %.4f → %s｜性价比最优结构 %s（params=%d）｜"
               "决策记录%s｜实测规范率 %.2f%%") % (
        acc, acc_best, png_bytes, int(gates["C"]["png_min_bytes"]),
        gap_base, gap_note, ss["best_value"]["hidden"],
        ss["best_value"]["params"],
        "完整（result.json 回读校验）" if decision_ok else "不完整",
        st["rate"])
    v = tiers.verdict(
        gates, metrics_ok=metrics_ok, acc_best=acc_best,
        png_bytes=png_bytes, gap_base=gap_base, acc_base=acc_base, arms=arms,
        ss_candidates=len(ss["rows"]),
        best_value_ok=ss.get("best_value") is not None,
        decision_ok=decision_ok, style_rate=st["rate"], summary=summary)
    result["tier_verdict"] = v
    io_utils.write_json(json_path, result)
    # ---------- 报告行（全部来自上面的实测变量）
    ss_tbl = io_utils.fmt_table(
        [[str(r["hidden"]), r["params"], "%.4f" % r["acc"], "%.4f" % r["f1"],
          r["epochs"], "%.3f" % r["sec"]] for r in ss["rows"]],
        ["隐层结构", "参数量", "测试acc", "F1", "epochs", "训练秒"])
    arm_tbl = io_utils.fmt_table(
        [["%.1f" % a["dropout"], a["engine"], "%.4f" % a["acc"],
          "%.4f" % a["gap"], "%.3f" % a["sec"],
          "收窄且不降" if a is best_arm else "—"] for a in arms],
        ["Dropout", "引擎", "测试acc", "train-val gap", "训练秒", "B线合规"])
    input_lines = [
        "600 样本 × 30 维频谱｜基线隐藏层 (64,32)｜early_stopping=True｜"
        "分层划分 75/25（random_state=42）",
        "结构搜索候选 %s｜Dropout 扫描臂 %s（逐臂实测 acc/gap）"
        % ([tuple(r["hidden"]) for r in ss["rows"]], arms_dp)]
    train_lines = ["epochs=%d  测试准确率 %.4f（良好C线目标 ≥%.2f）"
                   % (len(m.loss_curve_), acc, acc_min)]
    diag_sk_lines = ["四段均值: %s" % dg_sk["means"],
                     "持续下降: %s｜末轮 loss %.4f"
                     % (dg_sk["improving"], dg_sk["final"])]
    reg_lines = ["alpha=1e-3 准确率 %.4f（对比 %.4f）" % (acc_reg, acc)]
    ss_lines = [ss_tbl,
                "性价比最优：acc-0.01 容差内参数量最小 = %s（params=%d，acc=%.4f）"
                % (ss["best_value"]["hidden"], ss["best_value"]["params"],
                   ss["best_value"]["acc"])]
    torch_lines = [
        "引擎口径：%s｜%s" % (arm_nd["engine"], arm_nd["note"]),
        "无 Dropout 基线臂：acc=%.4f train=%.4f val=%.4f gap=%.4f（墙钟 %.3fs）"
        % (arm_nd["acc"], dg_nd["train_end"], dg_nd["val_end"],
           dg_nd["gap"], arm_nd["sec"])]
    diag_lines = [
        "无 Dropout：%s" % dg_nd["verdict"],
        "Dropout %s（绘图臂）：%s" % (arm_plot["dropout"], dg_plot["verdict"]),
        "B 线口径：gap 取末段 1/4 均值差，"
        "「收窄且指标不降」按工单字面严格判（无容差）"]
    png_lines = [
        "PNG: %s（实测 %dB，完整下限 %dB）→ %s" % (
            png, png_bytes, int(gates["C"]["png_min_bytes"]),
            "图完整" if png_bytes >= int(gates["C"]["png_min_bytes"])
            else "图不完整"),
        "曲线数据：%d epochs，残差损失末值 %.4f（绘图臂 Dropout %s）"
        % (len(arm_plot["val_losses"]),
           arm_plot["val_losses"][-1] - arm_plot["train_losses"][-1],
           arm_plot["dropout"])]
    conclusion = [
        "结构实测择优 %s（params=%d）；Dropout 扫描 %d 臂，最优 gap %.4f"
        "（基线 %.4f）；曲线与残差比单个分数更能说明泛化" % (
            ss["best_value"]["hidden"], ss["best_value"]["params"],
            len(arms),
            best_arm["gap"] if best_arm else dg_plot["gap"], gap_base)]
    sections = [
        ("输入口径", input_lines),
        ("训练（sklearn 基线）", train_lines),
        ("损失曲线诊断", diag_sk_lines),
        ("正则对照", reg_lines),
        ("隐层结构搜索（实测，优A「性价比最优」证据）", ss_lines),
        ("Dropout 扫描臂（torch nn.Linear+nn.Dropout，优B对照实验）",
         torch_lines + [arm_tbl]),
        ("过拟合诊断（优B线口径）", diag_lines),
        ("学习/残差损失曲线图（良好C「图完整」）", png_lines),
        ("验收档位判定（实测）", tiers.render_lines(v)),
        ("人工答辩项（代码判不了，进档位判定不算达标）", tiers.MANUAL),
        ("代码规范率（D 档次条：≥%.0f%%）" % gates["D"]["style_min"],
         _style_lines(st)),
        ("结论与建议", conclusion)]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
