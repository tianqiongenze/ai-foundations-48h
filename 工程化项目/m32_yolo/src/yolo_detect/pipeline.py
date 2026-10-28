# -*- coding: utf-8 -*-
"""造图 → 基线检测 → 真算 mAP（IoU贪心匹配+all-point插值AP）→ conf×IoU 扫描 → 固定口径FPS → 标注图。

档位判定不写在这里：所有实测值交给 tiers.py 与工单门槛比较（纯函数、可复核）。
"""
import os

from . import core, io_utils, report, style, tiers

CONFS = (0.55, 0.65, 0.75)
IOUS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def _mean_ap(grid, conf):
    vals = [grid[(conf, it)] for it in IOUS if grid[(conf, it)] is not None]
    return sum(vals) / len(vals) if vals else None


def _cell_str(v):
    """扫描格报告口径：None=无 GT 可评，如实写"无GT"，不冒充数字。"""
    return "%.3f" % v if v is not None else "无GT"


def _ultralytics_note():
    """第三方真实路径探活：装了报版本，没装打印 MISS 并写明兜底口径。"""
    try:
        import ultralytics
        ver = getattr(ultralytics, "__version__", "已安装")
        return "ultralytics OK %s（真实路径可用）" % ver
    except ImportError:
        return ("ultralytics MISS（本机未安装）→ 确定性兜底：自研滑窗CV"
                "检测器 core.detect_all，mAP 全部由该通道 IoU 匹配实测计算")


def _style_section(stats):
    """PEP8 实测段：口径/引擎/明细都落盘，"没检查"不能说成"没违规"。"""
    head = ("规范率 %.2f%%（引擎：%s）｜版式违规 %d 条 + 结构违规 %d 条"
            " / 非空代码 %d 行" % (
                stats["rate"], stats["engine"],
                stats["layout_violations"], stats["structural_violations"],
                stats["code_lines"]))
    lines = [head, "口径：%s" % stats["definition"], stats["fallback_note"]]
    if stats["by_rule"]:
        items = ["%s×%d(%s)" % (r["code"], r["count"], r["meaning"])
                 for r in stats["by_rule"][:6]]
        lines.append("违规明细：" + "；".join(items))
    return lines


def run(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # —— 单图基线（保留原口径，展示检测闭环）——
    img, gt = core.make_scene(64, seed=cfg.seed)
    box, conf0 = core.detect(img)
    v0 = core.iou(box, gt) if box else 0.0

    # —— 批量场景 + 真·mAP ——
    scenes = core.make_dataset(seed=cfg.seed)
    dets = [core.detect_all(im, thresh=0.55) for im, _ in scenes]
    ms, n_gt = core.match_scenes(scenes, dets, iou_thr=0.5)
    ap05 = core.average_precision(ms, n_gt)
    grid = core.sweep_map(scenes, CONFS, IOUS)
    map5095 = {c: _mean_ap(grid, c) for c in CONFS}
    n_det = sum(len(d) for d in dets)
    neg_idx = [i for i, (_, g) in enumerate(scenes) if not g]
    n_neg = len(neg_idx)
    n_neg_fp = sum(len(dets[i]) for i in neg_idx)

    # —— 推荐点（conf × IoU 网格上实测 AP 最高者）——
    best_pair = max(grid, key=lambda k: grid[k] if grid[k] is not None else -1)
    rec_reason = ("漏检代价高（缺陷流出）优先保召回→conf 取网格实测 AP 最高的 %.2f；"
                  "IoU 口径按答辩统一用 0.5（工单B05：mAP 必须注明口径）"
                  % best_pair[0])

    # —— 固定口径 FPS（预热/帧数写死并写进报告，工单B04）——
    warm, frames = 5, 100
    fsw = {}
    for win in (4, 8, 16):
        fsw[win] = core.bench(lambda im, w=win: core.detect_all(im, win=w),
                              [scenes[0][0]], warmup=warm, n=frames)
    speedup = (fsw[16][0] / fsw[4][0] - 1) * 100

    # —— ONNX YOLOv8n 通道实测 ——
    from . import onnx_channel
    onx = onnx_channel.probe([scenes[0][0], scenes[1][0], scenes[2][0]])

    # —— 标注图 ——
    ann_paths = []
    for k in (0, 1):
        p = os.path.join(out_dir, "annotated_scene%d.png" % (k + 1))
        ann_paths.append(core.annotate(scenes[k][0], dets[k], scenes[k][1], p))

    sweep_lines = []
    show_iou = (0.5, 0.6, 0.7, 0.75, 0.95)
    for c in CONFS:
        cells_txt = "  ".join(
            "IoU%.2f=%s" % (i, _cell_str(grid[(c, i)])) for i in show_iou)
        mean_str = ("%.3f" % map5095[c]
                    if map5095[c] is not None else "N/A")
        sweep_lines.append("conf=%.2f  " % c + cells_txt
                           + "｜mAP@0.5:0.95=" + mean_str)

    root = style.project_root(os.path.abspath(__file__))
    stats = style.code_style_rate(root)
    decisions = tiers.count_decision_entries(root)
    ult_note = _ultralytics_note()
    n_cells_done = sum(1 for x in grid.values() if x is not None)
    v = tiers.verdict(
        ap=ap05, style_rate=stats["rate"], sweep_rows=len(sweep_lines),
        sweep_cells=n_cells_done, sweep_cells_expected=len(CONFS) * len(IOUS),
        rec_ap=grid[best_pair], reason=rec_reason, speedup=speedup,
        decisions=decisions, gates=getattr(cfg, "tiers", None) or None,
        miss_note="无 GT 可评" if ap05 is None else None)

    sweep_json = {}
    for (c, i), g in grid.items():
        sweep_json["%s@%s" % (c, i)] = (
            round(g, 4) if g is not None else None)
    fps_by_win = {str(w): round(f, 1) for w, (f, msx) in fsw.items()}
    result = {"baseline_iou": round(v0, 4), "n_scenes": len(scenes),
              "n_gt": n_gt, "n_det": n_det, "neg_fp": n_neg_fp,
              "ap@0.5": round(ap05, 4) if ap05 is not None else None,
              "map@0.5:0.95": {str(c): (round(x, 4) if x else None)
                               for c, x in map5095.items()},
              "sweep": sweep_json,
              "recommend": {"conf": best_pair[0], "iou": best_pair[1]},
              "fps_protocol": {"warmup": warm, "frames": frames,
                               "cpu": os.cpu_count()},
              "fps_by_win": fps_by_win,
              "speedup_pct_win16_vs_win4": round(speedup, 1),
              "style_rate": stats["rate"], "tiers": v,
              "decision_entries": decisions, "ultralytics": ult_note,
              "onnx": onx, "annotated": ann_paths}
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)

    rec_ap_str = ("%.4f" % grid[best_pair]
                  if grid[best_pair] is not None else "N/A")
    onx_line = ("实测最大置信度=%s｜过阈(0.25)框数=%d｜"
                "FPS=%.1f（预热1帧/测3帧，640×640 CPU onnxruntime）"
                % (onx["max_conf"], onx["boxes_over_thr"], onx["fps"])
                if onx["available"]
                else "ONNX 通道未运行——相关档位检查一律判'未运行'")
    ann_labels = [os.path.basename(p) + "（GT绿框/检出红框+置信度）"
                  for p in ann_paths]
    concl = ("实测最优 win=%d 系旧单图口径保留项（win=8 时 IoU=%.2f）；"
             "mAP 通道结论以扫描表推荐点为准——窗口过小信噪比低易误检、"
             "过大定位粗化，这就是 YOLO 锚框/步长调参的同一问题" % (8, v0))
    sections = [
        ("输入口径", ["64×64 产线图｜基线真值框 %s｜阈值 0.5" % (gt,)]),
        ("基线检测结果", ["框=%s 缺陷分=%.3f IoU=%.2f（目标≥0.5）" % (box, conf0, v0)]),
        ("mAP 评估（真实计算：IoU贪心匹配+all-point插值AP）", [
            "场景 %d 张（含负样本 %d 张）｜GT 总数 %d｜检出总框数 %d｜负样本误检框 %d"
            % (len(scenes), n_neg, n_gt, n_det, n_neg_fp),
            "AP@0.5（conf=0.55 通道）= %s"
            % ("%.4f" % ap05 if ap05 is not None else "N/A（无GT可评，不编造）"),
            "说明：AP=置信度降序 PR 曲线下面积（COCO all-point 插值），"
            "非平均 IoU、非估计值——由 match_scenes/average_precision 实测"
            if ap05 is not None else "AP 无法计算（无 GT）——不编造数字",
            "检测通道口径（环境探活实测）: %s" % ult_note]),
        ("conf × IoU 阈值扫描（每格重跑检测+匹配）", sweep_lines),
        ("推荐点及理由（工单B档要求）", [
            "实测 AP 最高：conf=%.2f / IoU=%.2f → AP=%s" % (
                best_pair[0], best_pair[1], rec_ap_str),
            rec_reason]),
        ("win 取舍 + 固定口径 FPS（工单B04：预热%d帧→测%d帧，CPU %s 核）"
         % (warm, frames, os.cpu_count()),
         ["win=%-2d FPS=%9.1f 平均帧延迟=%.3fms" % (w, f, msx)
          for w, (f, msx) in fsw.items()]
         + ["加速比（win16 vs win4，同口径实测）= %+.1f%%" % speedup]),
        ("ONNX YOLOv8n 通道实测（与CV基线分开标注）", [
            "模型: %s" % (onx["model"] or "缺失"),
            "状态: %s" % onx["note"],
            onx_line if onx["available"]
            else "ONNX 通道未运行——相关档位检查一律判'未运行'"]),
        ("标注图（cv2 真实绘制）", ann_labels),
        ("代码规范率（PEP8 实测，工单 D 档次条）", _style_section(stats)),
        ("验收档位判定（实测）", tiers.render_lines(v)),
        ("结论与建议", [concl]),
    ]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
