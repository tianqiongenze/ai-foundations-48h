# -*- coding: utf-8 -*-
"""批量问答 → 引用命中统计 → 防幻觉比例。"""
import os

from . import core, io_utils, report


def run(cfg, out_dir):
    kb, source = core.load_kb()
    queries = ["泵振动超标要看哪个标准", "电机电流高但转速正常", "工装夹具的存放位置"]
    rows, grounded = [], 0
    for q in queries:
        a = core.answer(q, kb, topk=int(cfg.topk), min_score=float(cfg.raw.get("min_score", 0.05)))
        grounded += int(a["grounded"])
        rows.append({"query": q, "grounded": a["grounded"], "citations": a["citations"]})
    result = {"kb_size": len(kb), "kb_source": source, "n_queries": len(queries), "grounded": grounded,
              "hallucination_rate": round(1 - grounded / max(len(queries), 1), 4),
              "rows": rows}
    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    lines = []
    for r in rows:
        if r["grounded"]:
            lines.append("Q: %s" % r["query"])
            lines.append("   → 命中 %d 条 %s（最佳分数 %.3f）"
                         % (len(r["citations"]),
                            ",".join(c["id"] for c in r["citations"]),
                            r["citations"][0]["score"]))
        else:
            lines.append("Q: %s" % r["query"])
            lines.append("   → 查无依据（拒答，未编造）")
    sections = [("输入口径", ["知识库 %d 条（来源: %s）｜二元词重合检索｜Top-%d｜阈值 %.2f"
                % (len(kb), source, int(cfg.topk), float(cfg.raw.get("min_score", 0.05)))]),
                ("问答明细", lines),
                ("命中率", ["有据可查 %d / %d（幻觉率 %.0f%%）"
                 % (grounded, len(queries), result["hallucination_rate"] * 100)]),
                ("结论与建议", ["RAG 底线='有据可查+诚实拒答'；查无依据的查询用于发现知识库盲区"])]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    return sections
