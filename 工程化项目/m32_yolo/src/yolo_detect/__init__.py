# -*- coding: utf-8 -*-
"""YOLO 实时缺陷检测 —— 工程化实现包。

分层：config（配置）/ core（纯逻辑）/ io_utils（IO）/ report（报告）/ style（规范率体检）/
tiers（工单四档实测判定，纯函数）/ pipeline（编排）/ cli（命令入口）。
"""
__version__ = "1.0.0"
__all__ = ["config", "core", "io_utils", "report", "style", "tiers",
           "pipeline", "cli", "onnx_channel"]
