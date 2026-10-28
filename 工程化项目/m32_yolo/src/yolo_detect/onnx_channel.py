# -*- coding: utf-8 -*-
"""YOLOv8n ONNX 通道实测：能跑就报真数字，跑不了/不达标就如实标注。

课程 data/weights/yolov8n.onnx 是静态输入(1,3,640,640)导出，onnxruntime 可离线加载；
但它是 COCO 80 类预训练权重——对本课合成工业缺陷**没有对应类别**，本模块把这一点
变成实测证据（报告最大置信度与过阈框数），而不是假装 YOLO 通道可用。
ultralytics 未安装（离线机），notebook 通道不可用——传统CV基线承担档位指标，两通道分开标注。
"""
import os

import numpy as np

# 项目根向上找课程数据包：m32_yolo → 工程化项目 → 48课时版 → data/weights
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
ONNX_CANDIDATES = [
    os.path.join(_ROOT, os.pardir, "data", "weights", "yolov8n.onnx"),
    os.path.join(os.getcwd(), "data", "weights", "yolov8n.onnx"),
]


def find_model():
    for p in ONNX_CANDIDATES:
        if os.path.exists(p):
            return os.path.normpath(p)
    return None


def preprocess(img, size=640):
    """灰度 64×64 → 3×640×640 的 float32（/255，与 ultralytics 导出前处理一致）。"""
    scaled = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    big = np.kron(scaled, np.ones((size // img.shape[0], size // img.shape[1]))
                  ).astype(np.float32) / 255.0
    return np.stack([big] * 3, axis=0)[None]


def probe(imgs, warmup=1, n=3, conf_thr=0.25):
    """实测 ONNX 通道：最大置信度、过阈框数、固定口径 FPS。任何异常都如实返回原因。"""
    out = {"available": False, "model": None, "max_conf": None,
           "boxes_over_thr": None, "fps": None, "frame_ms": None,
           "note": "未运行"}
    path = find_model()
    if path is None:
        out["note"] = "离线兜底：课程 data/weights/yolov8n.onnx 不存在，ONNX 通道未运行"
        return out
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
    except Exception as e:
        out["note"] = "离线兜底：onnxruntime 加载失败（%r），ONNX 通道未运行" % e
        return out

    def run_one(im):
        return sess.run(None, {iname: preprocess(im)})[0]

    x = run_one(imgs[0])
    max_conf = None
    boxes = 0
    for im in imgs:
        o = run_one(im)              # (1, 84, 8400)：前4行框、后84行类别分
        scores = np.asarray(o[0][4:])
        mc = float(scores.max())
        max_conf = mc if max_conf is None else max(max_conf, mc)
        boxes += int((scores > conf_thr).sum())
    import time
    for _ in range(warmup):
        run_one(imgs[0])
    t0 = time.perf_counter()
    for i in range(n):
        run_one(imgs[i % len(imgs)])
    dt = (time.perf_counter() - t0) / n
    out.update({"available": True, "model": path,
                "max_conf": round(max_conf, 4), "boxes_over_thr": boxes,
                "fps": round(1.0 / dt, 2), "frame_ms": round(dt * 1000, 1),
                "note": "COCO 80 类无'工业缺陷'类：合成缺陷上实测最大置信度见本行，"
                        "过阈(%.2f)框数=%d——YOLO 通道对合成缺陷不可用是实测结论，非猜测"
                        % (conf_thr, boxes)})
    return out
