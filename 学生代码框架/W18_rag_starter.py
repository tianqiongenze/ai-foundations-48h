# -*- coding: utf-8 -*-
"""M4-2 RAG知识库问答 starter（教学周10）：切分→检索→引用生成三步，回答必须带出处

解A（本文件）：bigram重合度检索——教学直观
解B（升级）：ChromaDB向量库——见 本包教师参考/rag_chroma_demo.py（两解互验版）
"""
import re


def make_kb():
    return [
        ("运维规范", "泵类设备振动速度有效值超过4.5mm/s应列入重点监测"),
        ("运维规范", "轴承温度报警阈值设定为75℃，达到85℃必须停机检查"),
        ("排查指南", "泵体异响且压力波动：优先检查叶轮是否堵塞"),
        ("排查指南", "电机电流偏高而转速正常：疑似轴承磨损，测特征频率确认"),
    ]


def tokens(text):
    toks = []
    for r in re.findall(r"[\u4e00-\u9fa5]+", text):
        toks += [r[i:i + 2] for i in range(len(r) - 1)] + [r]
    return set(toks)


def retrieve(query, docs, k=2):
    """解A检索：重合度打分取Top-k"""
    # TODO-W10-1: q=tokens(query)，对docs每条算 len(q&tokens(d))/(len(q)+1e-6)，
    #   降序取前k条（得分>0）
    return []


def rag_answer(query):
    docs = make_kb()
    hits = retrieve(query, docs)
    if not hits:
        return "查无依据：请补充设备编号与测点数据后再检索（宁说无据，不硬编）"
    refs = "\n".join(f"- [{c}] {d}" for c, d in hits)
    return f"依据知识库：\n{refs}\n→ 建议：按上述条款执行并回填工单。"


if __name__ == "__main__":
    for q in ["泵振动超标要看哪个标准", "电机电流高但转速正常", "工装夹具的存放位置"]:
        print("Q:", q)
        print("A:", rag_answer(q))
        print("目标：前两问带出处、第三问敢说'查无依据' —— 这就是RAG的'有据可查'")
