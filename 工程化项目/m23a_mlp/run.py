#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""项目唯一入口（免安装态）。

用法:
  python run.py                 # 按默认配置跑完整流水线并输出报告
  python run.py --config x.yaml # 指定配置文件
  python run.py --outdir out/   # 指定输出目录
  python run.py --quiet         # 只输出到文件，不打印到控制台

实现说明：真正的参数解析在 `mlp_clf.cli:main`（与 pip 安装后的 `%(prog)s` 命令同一入口），
本文件只负责把 src/ 挂进 sys.path，保证"克隆即跑"与"安装即用"两条路零口径差。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from mlp_clf.cli import main  # noqa: E402  （sys.path 注入后方可导入，属 src 布局必需）

if __name__ == "__main__":
    sys.exit(main())
