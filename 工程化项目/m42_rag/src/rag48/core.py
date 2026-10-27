# -*- coding: utf-8 -*-
"""RAG：二元词重合检索 + 引用生成 + 防幻觉拒答（纯函数）。"""
import json
import os
import re

KB_CANDIDATES = [
    os.path.join("config", "knowledge_base.json"),      # ① 项目自带副本（首选，零外部依赖）
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "knowledge_base.json"),
    os.path.join("data", "knowledge_base.json"),        # ② 包级数据目录
    os.path.join("..", "data", "knowledge_base.json"),
    os.path.join("..", "..", "data", "knowledge_base.json"),
]


def load_kb(path=None):
    """加载知识库，返回 (kb, source)。

    探测顺序：项目自带 config/ 副本 → 项目 data/ → 上级 data/ → 内置兜底 4 条。
    任何一级命中都返回来源标记，便于排错（避免"悄悄降级"却不知道）。
    """
    cands = [path] if path else KB_CANDIDATES
    items, source = None, "内置兜底(4条)"
    for c in cands:
        if c and os.path.exists(c):
            try:
                raw = json.load(open(c, encoding="utf-8"))
                got = [(cat, it) for cat, lst in raw.items() for it in lst]
                if got:
                    items, source = got, c
                    break
            except Exception:
                continue
    if items is None:
        items = [
            ("设备运维规范", "泵类设备振动速度有效值超过4.5mm/s应列入重点关注，超过7.1mm/s必须停机检查。"),
            ("设备运维规范", "轴承温度报警阈值设定为75℃，达到85℃应立即停机并检查润滑与冷却回路。"),
            ("故障排查指南", "泵体异响且压力波动：优先排查叶轮堵塞与汽蚀，检查进口滤网和液位。"),
            ("故障排查指南", "电机电流偏高而转速正常：排查机械过载、轴承磨损与三相不平衡。"),
        ]
    kb = []
    for i, (cat, txt) in enumerate(items, 1):
        kb.append({"id": "R%d" % i, "category": cat, "text": txt,
                   "tokens": bigrams(txt)})
    if source == "内置兜底(4条)":
        import warnings
        warnings.warn("未找到 knowledge_base.json，已使用内置兜底 4 条（检索覆盖率会下降）")
    return kb, source


def bigrams(text):
    out = set()
    for seg in re.findall(r"[\u4e00-\u9fff]+", str(text)):
        if len(seg) >= 2:
            out |= {seg[i:i + 2] for i in range(len(seg) - 1)}
            out.add(seg)
    return out


def retrieve(query, kb, topk=3, min_score=0.05):
    q = bigrams(query)
    scored = [(len(q & k["tokens"]) / max(len(q), 1), k) for k in kb]
    cand = [s for s in scored if s[0] >= min_score]
    cand.sort(key=lambda x: (-x[0], x[1]["id"]))
    return [{"id": k["id"], "category": k["category"], "text": k["text"],
             "score": round(s, 4)} for s, k in cand[:topk]]


def answer(query, kb, topk=3, min_score=0.05):
    hits = retrieve(query, kb, topk, min_score)
    if not hits:
        return {"grounded": False,
                "text": "查无依据：请补充设备编号与测点数据后再检索（宁说无据，不硬编）。",
                "citations": []}
    refs = chr(10).join("  [%s] (%s) %s" % (h["id"], h["category"], h["text"]) for h in hits)
    return {"grounded": True,
            "text": "依据知识库:" + chr(10) + refs + chr(10) + "  → 建议：按上述条款执行并回填工单。",
            "citations": hits}
