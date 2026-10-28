# -*- coding: utf-8 -*-
"""零依赖检测器：滑窗 + 阈值 + IoU + FPS（纯函数）。"""
import numpy as np


def make_scene(size=64, seed=7):
    """64×64 产线图 + 已知真值框（一个亮点缺陷）。"""
    rng = np.random.RandomState(seed)
    img = rng.normal(0.4, 0.05, (size, size))
    r0, c0, w = 40, 30, 8
    img[r0:r0 + w, c0:c0 + w] += 0.35
    return img, (c0, r0, w, w)


def score_map(img, win=8):
    """滑窗分数图：每个 win×win 窗的均值。"""
    H, W = img.shape[0] // win, img.shape[1] // win
    smap = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            rs, re_ = r * win, (r + 1) * win
            cs, ce = c * win, (c + 1) * win
            smap[r, c] = img[rs:re_, cs:ce].mean()
    return smap


def detect(img, win=8, thresh=0.5):
    smap = score_map(img, win)
    r, c = np.unravel_index(np.argmax(smap), smap.shape)
    if smap[r, c] < thresh:
        return None, float(smap[r, c])
    return (int(c) * win, int(r) * win, win, win), float(smap[r, c])


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    un = a[2] * a[3] + b[2] * b[3] - inter
    return inter / un if un else 0.0


def fps(detect_fn, img, n=100):
    import time
    t0 = time.time()
    for _ in range(n):
        detect_fn(img)
    return round(n / (time.time() - t0), 1)


# ------------------------------------------------------------------ 批量场景
def make_dataset(n_scenes=12, n_neg=3, size=64, seed=7):
    """批量产线场景：每图 1-2 个亮斑缺陷 + 纯背景负样本（负样本用于实测误检）。

    mAP 必须有'检测器输出的置信度分布'与'漏检/误检'两类失败样本才有意义，
    单图单框只能得一个 IoU 点——所以这里扩成场景集。返回 [(img, [gt,...]), ...]。
    """
    rng = np.random.RandomState(seed)
    scenes = []
    for i in range(n_scenes + n_neg):
        img = rng.normal(0.4, 0.05, (size, size))
        gts = []
        if i < n_scenes:
            for _ in range(rng.randint(1, 3)):
                w = rng.randint(6, 11)
                x = rng.randint(0, size - w)
                y = rng.randint(0, size - w)
                img[y:y + w, x:x + w] += rng.uniform(0.30, 0.45)
                gts.append((int(x), int(y), int(w), int(w)))
        scenes.append((img, gts))
    return scenes


def score_map_vec(img, win):
    """矢量化无重叠滑窗对比分：窗内最强 25% 像素的均值。

    为何不用窗均值：缺陷常只覆盖窗口一角（部分覆盖），窗均值被背景稀释→漏检；
    top-k 均值等价于'局部最强响应'，与 YOLO anchor 命中思想一致（实测可复现）。
    """
    h = img.shape[0] // win
    w = img.shape[1] // win
    blocks = (img[:h * win, :w * win].reshape(h, win, w, win)
              .transpose(0, 2, 1, 3).reshape(h, w, win * win))
    k = max(1, win * win // 4)
    return np.sort(blocks, axis=2)[:, :, -k:].mean(axis=2)


def candidates(img, win=8, thresh=0.55):
    """所有过阈滑窗候选：[((x,y,w,h), conf), ...]，conf=窗内最强响应均值。

    阈值 0.55 由实测分布定：背景窗 top-25% 均值≈0.49±0.01、缺陷窗≈0.72+ ——两簇分开。"""
    smap = score_map_vec(img, win)
    return [((int(c) * win, int(r) * win, win, win), float(smap[r, c]))
            for r, c in np.argwhere(smap >= thresh)]


def refine_thr(img):
    """精修阈值=中位数+7×MAD：亮斑对比≥0.30、噪声σ=0.05 → 前景全过线、背景几乎零散点。"""
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med))) + 1e-9
    return med + 7.0 * mad


def refine_box(img, box, thr):
    """候选框精修：搜索域内阈值化 → 加权质心定位 + 矩法估计边长（std×3.46 与 √面积取大，再+1）。

    外接矩形会被零星噪声拉爆、纯 √面积 又欠估细长缺陷（均为实测教训）；
    均匀亮斑理论 std=side/√12，反解边长并把质心/尺寸 +1 补偿阈值边界损失——
    把'分类窗'升级为'定位框'，YOLO 回归偏移量解决的是同一个问题。
    """
    x, y, w, h = box
    pad = w
    y0, y1 = max(0, y - pad), min(img.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(img.shape[1], x + w + pad)
    sub = img[y0:y1, x0:x1]
    mask = sub > thr
    if not mask.any():
        return box
    rs, cs = np.nonzero(mask)
    wts = np.clip(sub[mask] - thr, 0.0, None) + 1e-6
    cy = float(np.average(rs, weights=wts))
    cx = float(np.average(cs, weights=wts))
    sy = float(np.sqrt(np.average((rs - cy) ** 2, weights=wts)))
    sx = float(np.sqrt(np.average((cs - cx) ** 2, weights=wts)))
    side_a = np.sqrt(float(mask.sum()))
    bw = max(2, int(round(max(3.464 * sx, side_a))) + 1)
    bh = max(2, int(round(max(3.464 * sy, side_a))) + 1)
    bx = int(round(x0 + cx - bw / 2.0))
    by = int(round(y0 + cy - bh / 2.0))
    return (max(0, min(bx, img.shape[1] - bw)),
            max(0, min(by, img.shape[0] - bh)), bw, bh)


def nms(dets, iou_thr=0.5):
    """贪心非极大值抑制：置信度降序，与已保留框 IoU>iou_thr 的候选丢弃。"""
    out = []
    for box, sc in sorted(dets, key=lambda d: -d[1]):
        if all(iou(box, b) <= iou_thr for b, _ in out):
            out.append((box, sc))
    return out


def detect_all(img, win=8, thresh=0.55):
    """完整检测通道：滑窗候选 → 精修 → NMS。返回 [((x,y,w,h), conf), ...]。"""
    thr = refine_thr(img)
    refined = [(refine_box(img, b, thr), s)
               for b, s in candidates(img, win, thresh)]
    return nms(refined, 0.5)


# ------------------------------------------------------------------ mAP 评估
def match_scenes(scenes, dets_per_scene, iou_thr=0.5):
    """跨场景全局贪心匹配（置信度降序）：返回 [(conf, is_tp), ...] 与 GT 总数。"""
    flat = []
    for k, dets in enumerate(dets_per_scene):
        for box, sc in dets:
            flat.append((sc, k, box))
    flat.sort(key=lambda t: -t[0])
    taken = [set() for _ in scenes]
    matches = []
    for sc, k, box in flat:
        cands = [(iou(box, gt), gi) for gi, gt in enumerate(scenes[k][1])
                 if gi not in taken[k]]
        if cands:
            v, gi = max(cands)
            if v >= iou_thr:
                taken[k].add(gi)
                matches.append((sc, 1))
                continue
        matches.append((sc, 0))
    n_gt = sum(len(g) for _, g in scenes)
    return matches, n_gt


def average_precision(matches, n_gt):
    """all-point 插值 AP（COCO 口径）：置信度降序累计 PR 曲线下面积。

    matches 需已按 conf 降序；n_gt=0 时无定义，返回 None（不编造数字）。
    """
    if n_gt == 0 or not matches:
        return None
    tp = np.array([m[1] for m in matches])
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    prec = tp_cum / (tp_cum + np.cumsum(fp))
    rec = tp_cum / n_gt
    ap, prev_r = 0.0, 0.0
    for p, r in zip(prec, rec):
        p_max = float(prec[np.searchsorted(rec, r, side="left"):].max())
        ap += (r - prev_r) * p_max
        prev_r = r
    return float(ap)


def sweep_map(scenes, confs, ious, win=8):
    """conf × IoU 双阈值扫描：返回 {(conf, iou): AP}（每格都真跑一遍检测+匹配）。"""
    grid = {}
    for conf in confs:
        dets = [detect_all(img, win=win, thresh=conf) for img, _ in scenes]
        for it in ious:
            ms, n_gt = match_scenes(scenes, dets, iou_thr=it)
            ms.sort(key=lambda t: -t[0])
            grid[(conf, it)] = average_precision(ms, n_gt)
    return grid


# ------------------------------------------------------------------ 可视化/测速
def annotate(img, dets, gts, path):
    """GT 绿框、检出红框（带置信度）画到 8× 放大灰度图并落盘（cv2 真实绘制）。"""
    import cv2
    up = cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8),
                    (img.shape[0] * 8, img.shape[1] * 8),
                    interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in gts:
        cv2.rectangle(canvas, (x * 8, y * 8), ((x + w) * 8, (y + h) * 8),
                      (0, 200, 0), 2)
    for (x, y, w, h), sc in dets:
        cv2.rectangle(canvas, (x * 8, y * 8), ((x + w) * 8, (y + h) * 8),
                      (0, 0, 220), 1)
        cv2.putText(canvas, "%.2f" % sc, (x * 8, max(y * 8 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
    cv2.imwrite(path, canvas)
    return path


def bench(fn, imgs, warmup=5, n=100):
    """固定口径测速：预热 warmup 帧 → 连续测 n 帧 → (FPS, 平均帧耗时ms)。

    工单B04：首帧冷启动必慢，口径不写清数字无意义——口径原文写进报告。
    """
    import time
    for i in range(warmup):
        fn(imgs[i % len(imgs)])
    t0 = time.perf_counter()
    for i in range(n):
        fn(imgs[i % len(imgs)])
    dt = time.perf_counter() - t0
    return n / dt, dt / n * 1000.0
