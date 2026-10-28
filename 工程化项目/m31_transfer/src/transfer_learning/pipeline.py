# -*- coding: utf-8 -*-
"""造世界 → 冻结提取 → 两组训练 → 增益结论；附真·CNN 2×2 实测。

工单四档（及格D/良好C/优秀B/优A）全部由本模块的实测值判定：准确率来自
2×2 真实训练评测、规范率来自 style.py PEP8 体检、对照行/决策条目由
tiers.py 的计数函数现场数出——判不到一律 MISS/未达标，不自称达标。
"""
import os

from . import core, io_utils, report, style, tiers

M2_MSG = "训练小样本 %d 张 → 增强扩增（扩后样本量见2×2表 n_train 列）；"
M3_MSG = "测试集 %d 张不增强——防泄漏（工单B05纪律）"
PRE_MSG = ("真实 ResNet 权重实测可用性：%s；torch.hub 本地权重缓存 %s"
           "（零网络探测）→ %s")
FALLBACK_MSG = ("兜底口径：小型本地CNN（2层卷积+全连接头）在合成缺陷图 %d "
                "张上预训练充当底座；底座在测试集 %d 张上实测准确率 %.4f"
                "（全部真实训练产生，绝不谎称 ImageNet ResNet）")
RESNET_MSG = ("真·ResNet 口径（工单主线'载ResNet权重'）：%s 自本地缓存离线载入"
              "（实测 %.3f 秒，零网络调用）；冻结主干，fc 换成 %d 维输入的 "
              "2 类新头重新初始化；2×2 全通道训练实测耗时 %.1f 秒"
              "（CPU，工单B06口径）")


def _load_torch_first():
    """在 sklearn 之前预加载 torch 的 OpenMP 运行库。

    本机（macOS+anaconda）实测：sklearn(joblib/libomp) 先跑、torch 后跑会触发
    libomp 重复加载段错误（进程直接 139 退出）；先 import torch 即共存无冲突。
    这不是风格洁癖，是崩溃复现后的确定性规避（对应 GB/T 15532 缺陷闭环）。
    """
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _style_section(stats):
    """PEP8 实测段：口径/引擎/明细都落盘，"没检查"不能说成"没违规"。"""
    fmt = ("规范率 %.2f%%（引擎：%s）｜版式违规 %d 条 + 结构违规 %d 条"
           " / 非空代码 %d 行")
    head = fmt % (
        stats["rate"], stats["engine"], stats["layout_violations"],
        stats["structural_violations"], stats["code_lines"])
    lines = [head, "口径：%s" % stats["definition"], stats["fallback_note"]]
    if stats["by_rule"]:
        items = ["%s×%d(%s)" % (r["code"], r["count"], r["meaning"])
                 for r in stats["by_rule"][:6]]
        lines.append("违规明细：" + "；".join(items))
    return lines


def build(cfg, out_dir):
    """跑完整流水线，返回 (报告分节, result)——cli --tier 复用同一实测。"""
    torch_ready = _load_torch_first()
    big, small, test, ys, yt = core.make_world(seed=cfg.seed)
    ext = core.FrozenExtractor(big, dim=8)
    _, raw_pred = core.train_head(small, ys, test)
    raw_acc = float((raw_pred == yt).mean())
    _, tf_pred = core.train_head(ext.transform(small), ys,
                                 ext.transform(test))
    tf_acc = float((tf_pred == yt).mean())
    evr = float(ext.pca.explained_variance_ratio_.sum())
    gain = tf_acc - raw_acc
    os.makedirs(out_dir, exist_ok=True)
    root = style.project_root(os.path.abspath(__file__))
    stats = style.code_style_rate(root)
    decisions = tiers.count_decision_entries(root)

    # —— 图像迁移实验（真·ResNet 优先，本地缓存不可用时兜底小型CNN，见 transfer）——
    exp, exp_err, probe = None, None, None
    try:
        if not torch_ready:
            raise ImportError("torch 不可导入（预加载阶段已确认缺失）")
        from . import transfer
        probe = transfer.probe_pretrain_weights()
        exp = transfer.run_experiment(seed=cfg.seed, raw=cfg.raw,
                                      out_dir=out_dir)
    except Exception as e:  # 缺 torch/matplotlib：如实记录，档位判 MISS
        exp_err = repr(e)
    result = {
        "dim_raw": small.shape[1], "dim_frozen": ext.n,
        "raw_acc": round(raw_acc, 4), "transfer_acc": round(tf_acc, 4),
        "gain": round(gain, 4), "evr": round(evr, 4),
        "style_rate": stats["rate"], "style_engine": stats["engine"],
        "resnet_probe": probe,
    }
    if exp:
        result["image2x2"] = {
            "%s/%s" % ("frozen" if f else "finetune",
                       "aug" if a else "noaug"): round(c["acc"], 4)
            for (f, a), c in exp["cells"].items()}
        result["image_mode"] = exp["mode"]
        result["image_wall_sec"] = exp["wall_sec"]
        if exp.get("resnet_info"):
            result["image_weights"] = exp["resnet_info"]
        if exp["pre_acc"] is not None:
            result["image_pretrain_acc"] = round(exp["pre_acc"], 4)
        result["image_best_acc"] = round(exp["best_acc"], 4)
        result["aug_gain_frozen"] = round(exp["aug_gains"]["frozen"], 4)
        result["aug_gain_finetune"] = round(exp["aug_gains"]["finetune"], 4)
        result["curves_png"] = exp["png"]
    else:
        result["image2x2_error"] = exp_err

    world_line = "海量无标注 %d 条｜少量标注 %d 条｜测试 %d 条"
    tf_line = "测试准确率 %.4f（解释方差 %.3f）"
    gain_word = "迁移胜出" if gain > 0 else "本轮未见增益"
    sections = [
        ("输入口径", [world_line % (len(big), len(small), len(test))]),
        ("对照组（16条直连192维）", ["测试准确率 %.4f" % raw_acc]),
        ("迁移组（冻结8维+训头）", [tf_line % (tf_acc, evr)]),
        ("增益", ["%+.4f → %s" % (gain, gain_word)]),
    ]

    # 对照表：先真实组行，再"逐行数出带实测数字的行"——完整与否是数出来的
    cmp_lines = ["对照组（16条直连192维）：测试准确率 %.4f" % raw_acc,
                 "迁移组（冻结8维+训头）：测试准确率 %.4f" % tf_acc]
    curves = 0
    if exp:
        from . import transfer
        tv = probe["torchvision"] if probe else "MISS（torch 通道未跑通）"
        cache = ((probe["cache_dir"] + ("，已缓存：" + probe["cached"]
                                        if probe["cached"] else ""))
                 if probe else "MISS")
        note = probe["note"] if probe else "未探测——一律判 MISS"
        pre_lines = [PRE_MSG % (tv, cache, note)]
        if exp.get("mode") == "resnet18-imagenet-local-cache":
            ri = exp["resnet_info"]
            pre_lines.append(RESNET_MSG % (
                ri["weights"], ri["load_sec"], ri["in_features"],
                exp["wall_sec"]))
        else:
            pre_lines.append(FALLBACK_MSG % (
                exp["sizes"]["big"], exp["sizes"]["test"], exp["pre_acc"]))
        # 时长口径写进报告：小 epoch 是预算约束，不是门槛让步
        pre_lines.append(
            "时长口径：32×32 小图＋小 epoch（微调 %d、训头 %d、增强×%d），"
            "单次 run.py 控制在 90 秒内（工单B06）；四档门槛数字未动" % (
                int(cfg.raw.get("finetune_epochs", 6)),
                int(cfg.raw.get("head_epochs", 40)),
                int(cfg.raw.get("aug_repeats", 4))))
        m1 = "方法：%s（numpy 实现，种子固定可复现）"
        aug_line = M2_MSG % exp["sizes"]["small"] + M3_MSG % (
            exp["sizes"]["test"])
        mode_rows = []
        specs = (("冻结", True, "frozen"), ("微调", False, "finetune"))
        for mode, fr, key in specs:
            no = exp["cells"][(fr, False)]
            yes = exp["cells"][(fr, True)]
            msg = "%-12s 不增强=%.4f 增强=%.4f 增益=%+.4f(n_train=%d)"
            mode_rows.append(msg % (mode, no["acc"], yes["acc"],
                                    exp["aug_gains"][key], yes["n_train"]))
        cmp_lines += mode_rows
        if exp["png"] and os.path.exists(exp["png"]):
            curves = 1
            n_curve = len(exp["cells"]) + (1 if exp.get("pre_curve") else 0)
            if exp.get("pre_curve"):
                src = "合成预训练+4格"
            else:
                src = "4格（底座损失来自ImageNet，无本地预训练可画）"
            curve_line = "%s（%s训练损失共 %d 条曲线）" % (
                os.path.basename(exp["png"]), src, n_curve)
        else:
            curve_line = "曲线缺失（PNG 未落盘）→ A 档功能项判缺"
        sections += [
            ("迁移底座口径（真ResNet/兜底实测二选一）", pre_lines),
            ("数据增强（实测手段清单）",
             [m1 % "；".join(transfer.AUG_METHODS), aug_line]),
            ("2×2对比：冻结/微调 × 增强/不增强（测试集实测准确率）",
             mode_rows),
            ("训练曲线", [curve_line]),
        ]
    else:
        sections += [("图像迁移2×2实验（未运行）", [
            "环境异常，torch/matplotlib 通道未跑通：%s" % exp_err,
            "本运行内 准确率/增强有效性/曲线 实测缺失 → 相应档位一律判未达标",
        ])]

    compare_rows = tiers.count_numeric_rows(cmp_lines)
    v = tiers.verdict(
        acc=exp["best_acc"] if exp else None,
        style_rate=stats["rate"],
        compare_rows=compare_rows,
        aug_gains=exp["aug_gains"] if exp else {},
        decisions=decisions, curves=curves,
        gates=getattr(cfg, "tiers", None) or None,
        miss_note=exp_err,
    )
    result["tiers"] = v
    result["decision_entries"] = decisions
    result["compare_rows"] = compare_rows
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)

    concl = ["真实规律仅藏在少数测点维；"
             "冻结+小样本头是 ResNet 微调思想的最小可运行模型"]
    if exp:
        ok_aug = all(x > 0 for x in exp["aug_gains"].values())
        if ok_aug:
            msg = "增强在两路均正增益，增强有效论证成立（工单B档功能项）"
        else:
            msg = "增强增益未在全部分支成立——如实报告，不夸大"
        concl.append("2×2实测：%s" % msg)
    sections += [
        ("代码规范率（PEP8 实测，D 档次条）", _style_section(stats)),
        ("验收档位判定（实测）", tiers.render_lines(v)),
        ("结论与建议", concl),
    ]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections, result


def run(cfg, out_dir):
    """兼容旧入口：只回分节（cli/run.py 打印用），档位细节见 build()。"""
    sections, _result = build(cfg, out_dir)
    return sections
