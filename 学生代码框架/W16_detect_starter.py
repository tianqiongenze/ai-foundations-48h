# -*- coding: utf-8 -*-
"""M3-2 实时检测 starter（教学周8）：滑窗+阈值=检测器雏形；FPS与mAP的取舍亲手量

零依赖版检测原理（YOLO思想的极简骨架）：
  图像 → 滑窗扫描 → 每窗"缺陷分" → 阈值 → 定位框 → IoU评估 + FPS计时
真机版(ultralytics YOLOv8n离线权重)见工单——本文件先让你"手搓一个检测器"再谈调参。
"""
import time

import numpy as np


def make_scene(seed=7):
    """合成64×64产线图：1个亮点缺陷(已知真值框)"""
    rng = np.random.RandomState(seed)
    img = rng.normal(0.4, 0.05, (64, 64))
    r0, c0, w = 40, 30, 8
    img[r0:r0 + w, c0:c0 + w] += 0.35
    return img, (c0, r0, w, w)                     # 真值框 x,y,w,h


def score_map(img, win=8):
    """滑窗缺陷分：每个win×win窗的平均亮度（检测器的"卷积"雏形）"""
    # TODO-W8-1: 双重循环步长=win，对每个窗取 img[r:r+win, c:c+win].mean() 存入 H×W 分数图
    return None


def detect(img, win=8, thresh=0.5):
    """阈值→定位框：分数图最大点(超过阈值)映射回原图坐标"""
    smap = score_map(img, win)
    if smap is None:
        return None
    r, c = np.unravel_index(np.argmax(smap), smap.shape)
    if smap[r, c] < thresh:
        return None
    return (c * win, r * win, win, win)


def iou(a, b):
    """交并比：评估检测框与真值框的重合度（mAP的原子）"""
    # TODO-W8-2: 计算交集面积与并集面积之比（4行以内可完成）
    return 0.0


def fps_bench(detect_fn, img, n=200):
    t0 = time.time()
    for _ in range(n):
        detect_fn(img)
    return n / (time.time() - t0)


if __name__ == "__main__":
    img, gt = make_scene()
    box = detect(img)
    if box:
        v = iou(box, gt)
        f = fps_bench(detect, img)
        print(f"检测框={box} 真值={gt} IoU={v:.2f} FPS≈{f:.0f}")
        print("目标：IoU≥0.5（TODO-W8-2完成后）；再改win=4/16看FPS-IoU怎么换——那就是YOLO调参")
    else:
        print("先完成TODO-W8-1（滑窗分数图），检测器就活了")
