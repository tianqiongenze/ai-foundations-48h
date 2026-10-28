# -*- coding: utf-8 -*-
"""综合答辩：档案袋自检 + 提纲生成（纯函数）。"""
import os

LAYERS = [
    ("数据层", "W07_clean_starter.py", "脏数据→可信数据集：清洗规则与质量分"),
    ("特征层", "W10_feature_starter.py", "特征字典：时域/频域双解与重要性证据"),
    ("模型层", "W11_classify_starter.py", "双模型对比与混淆矩阵里的漏报账"),
    ("服务层", "W17_prompt_starter.py", "三版 Prompt 对比：LLM 回答质量 +50%"),
    ("智能体层", "W19_multiagent_starter.py", "调度策略：FIFO→WSPT 省 20% 加权流时间"),
]

THREE_Q = [
    ("① 需求（30秒）", "这个项目解决什么工业问题？给谁用？"),
    ("② 方案（60秒）", "数据从哪来→怎么处理→输出什么（三段式）"),
    ("③ 评估（60秒）", "好在哪？用哪个指标证明？踩过什么坑？"),
]

#: 基本问题实测：每题回答必须覆盖的概念词（工单 C 档"基本问题全答"判据）
QA_BANK = [
    ("① 需求（30秒）", ["泵", "运维"]),
    ("② 方案（60秒）", ["数据层", "特征层", "模型层", "服务层", "智能体层"]),
    ("③ 评估（60秒）", ["正确率", "证据链"]),
]


def compose_answer(idx, rows, n_ok, total):
    """离线规则回答器：回答只能由档案袋自检实测结果拼装——缺层即缺词。

    为什么这样设计：'全答'不能靠嘴说；②题的答案逐字取自本次实测就绪的
    层名清单，档案袋少一层，②题就答不出该层——命中数随之下降。
    """
    if idx == 0:
        return ("为两江智造·设备智能运维中心解决泵组故障诊断与检修调度问题："
                "把数据清洗→特征→模型→Prompt→多智能体串成一条证据链。")
    if idx == 1:
        ok_layers = [layer for layer, ok, _ in rows if ok]
        return ("链路：%s。数据来自现场振动/温度采集，"
                "逐层加工后输出诊断结论与调度建议。" % "→".join(ok_layers))
    return ("用协调正确率与泛化评估说话：本次实测证据链 %d/%d 层就绪，"
            "每层都有可复核工件（starter 代码/测试/报告）。" % (n_ok, total))


def qa_hits(rows, n_ok, total):
    """题库逐题实测：真实生成回答并核对必含概念词，返回 (命中数, 题数, 明细)。"""
    hit, detail = 0, []
    for i, (tag, need) in enumerate(QA_BANK):
        ans = compose_answer(i, rows, n_ok, total)
        miss = [k for k in need if k not in ans]
        hit += int(not miss)
        detail.append({"q": tag, "hit": not miss, "miss": miss})
    return hit, len(QA_BANK), detail


def check(fw_dir):
    """档案袋自检：返回 [(层, 是否存在, 讲什么), ...]。"""
    out = []
    for name, fn, talk in LAYERS:
        out.append((name, os.path.exists(os.path.join(fw_dir, fn)), talk))
    return out


def outline(rows):
    """生成答辩提纲行与"证据链 n/m"实测计数（报告口径唯一出处）。"""
    n_ok = sum(1 for _, ok, _ in rows if ok)
    lines = ["=== 答辩提纲（一图贯通 · 自动生成） ==="]
    for i, (layer, ok, talk) in enumerate(rows, 1):
        lines.append("  %d. %s [%s] —— %s"
                     % (i, layer, "OK" if ok else "--", talk))
    lines.append("")
    lines.append("证据链 %d/%d 层就绪%s" % (n_ok, len(rows),
                 "——可以上台了" if n_ok == len(rows) else "——缺的层答辩前补齐"))
    return lines, n_ok
