# -*- coding: utf-8 -*-
"""综合答辩：档案袋自检 + 提纲生成（纯函数）。"""
import os

LAYERS = [
    ("数据层", "W07_clean_starter.py", "脏数据→可信数据集：清洗规则与质量分"),
    ("特征层", "W18_feature_starter.py", "特征字典：时域/频域双解与重要性证据"),
    ("模型层", "W19_classify_starter.py", "双模型对比与混淆矩阵里的漏报账"),
    ("服务层", "W17_prompt_starter.py", "三版 Prompt 对比：LLM 回答质量 +50%"),
    ("智能体层", "W19_multiagent_starter.py", "调度策略：随机→安全优先，协调正确率 100%"),
]

THREE_Q = [
    ("① 需求（30秒）", "这个项目解决什么工业问题？给谁用？"),
    ("② 方案（60秒）", "数据从哪来→怎么处理→输出什么（三段式）"),
    ("③ 评估（60秒）", "好在哪？用哪个指标证明？踩过什么坑？"),
]


def check(fw_dir):
    """档案袋自检：返回 [(层, 是否存在, 讲什么), ...]。"""
    out = []
    for name, fn, talk in LAYERS:
        out.append((name, os.path.exists(os.path.join(fw_dir, fn)), talk))
    return out


def outline(rows):
    n_ok = sum(1 for _, ok, _ in rows if ok)
    lines = ["=== 答辩提纲（一图贯通 · 自动生成） ==="]
    for i, (layer, ok, talk) in enumerate(rows, 1):
        lines.append("  %d. %s [%s] —— %s" % (i, layer, "OK" if ok else "--", talk))
    lines.append("")
    lines.append("证据链 %d/%d 层就绪%s" % (n_ok, len(rows),
                 "——可以上台了" if n_ok == len(rows) else "——缺的层答辩前补齐"))
    return lines, n_ok
