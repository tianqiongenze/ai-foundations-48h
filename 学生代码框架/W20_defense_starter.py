# -*- coding: utf-8 -*-
# [教学脚手架] 下列导入/变量供 TODO 完成后使用，故标注 noqa 而非删除
"""Z 综合项目答辩 starter（教学周12）：档案袋自检 + 一图贯通提纲生成

跑法: /opt/anaconda3/bin/python W20_defense_starter.py
产出: 答辩提纲自动生成（数据→特征→模型→服务→智能体 五层各讲什么）
"""
import glob  # noqa: F401
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [
    ("数据层", "W07_clean_starter.py", "脏数据→可信数据集：清洗规则与质量分"),
    ("特征层", "W10_feature_starter.py", "特征字典：时域/频域双解与重要性证据"),
    ("模型层", "W11_classify_starter.py", "双模型对比与混淆矩阵里的漏报账"),
    ("服务层", "W17_prompt_starter.py", "三版Prompt对比：LLM回答质量+50%"),
    ("智能体层", "W19_multiagent_starter.py", "调度策略：FIFO→WSPT省20%加权流时间"),
]


def check_files():
    """档案袋自检：五层starter在不在（在=你的证据链齐了）"""
    # TODO-Z-1: 对每层检查 os.path.exists(os.path.join(HERE, fn))，
    #   返回 [(层名, 是否存在, 讲什么), ...]
    return []


def outline():
    rows = check_files()
    if not rows:
        print("先完成TODO-Z-1（档案袋自检）")
        return
    print("=== 答辩提纲（一图贯通·自动生成） ===")
    for i, (layer, ok, talk) in enumerate(rows, 1):
        mark = "✓" if ok else "✗"
        print(f"  {i}. {layer} [{mark}] —— {talk}")
    n_ok = sum(r[1] for r in rows)
    print(f"\n证据链 {n_ok}/5 层就绪" + ("——可以上台了（记得带campus_skills当彩蛋）" if n_ok == 5
                                   else "——缺的层答辩前补齐"))


if __name__ == "__main__":
    outline()
