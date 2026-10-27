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
            smap[r, c] = img[r * win:(r + 1) * win, c * win:(c + 1) * win].mean()
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
