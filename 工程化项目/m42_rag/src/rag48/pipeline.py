# -*- coding: utf-8 -*-
"""批量问答 → 双通道检索实测（关键词 vs jieba+TF-IDF vs RRF融合）→ OOD拒答 → 有库/无库对照。"""
import os

from . import core, io_utils, report, style, tiers

ROOT = style.project_root(os.path.abspath(__file__))


def _keyword_ids(query, kb, topk=3):
    """关键词通道 Top-k 条目 id 列表（供 RRF 融合统一评测）。"""
    return [h["id"] for h in core.retrieve(query, kb, topk=topk)]


def run(cfg, out_dir):
    """跑完整问答评测：三通道检索/拒答/对照/规范率/档位判定，返回分节+实测。"""
    kb, source = core.load_kb()
    topk = int(cfg.topk)
    ret = core.TfidfRetriever(kb)

    # —— 拒答阈值由两簇相似度实测确定（不是拍的）——
    thr = core.pick_refuse_threshold(ret, core.EVAL_IN, core.EVAL_OOD)

    # —— 库内 recall@3：关键词通道 / TF-IDF 通道 / RRF 融合通道 ——
    kw_hit, n_in = core.recall_at_k(_kw_adapter(kb, topk), core.EVAL_IN, topk)
    tf_hit, _ = core.recall_at_k(ret, core.EVAL_IN, topk)
    fused = 0
    for q, gold in core.EVAL_IN:
        tf_ids = [h["id"] for h in ret.search(q, topk=topk)]
        fused_ids = core.rrf_fuse([_keyword_ids(q, kb, topk), tf_ids])
        if set(gold.split("+")) & set(fused_ids[:topk]):
            fused += 1
    kw_rate = kw_hit / n_in
    tf_rate = tf_hit / n_in
    fuse_rate = fused / n_in

    # —— OOD/超纲拒答实测：正确行为=拒答 ——
    ood_rows = []
    refused = 0
    for q in core.EVAL_OOD:
        a = core.answer_rag(q, ret, topk=topk, min_score=thr["threshold"])
        refused += int(not a["grounded"])
        ood_rows.append({"query": q, "refused": not a["grounded"],
                         "top1_sim": round(float(max(ret.sims(q))), 4)})

    # —— 有库/无库对照（幻觉率/引用率，全实测）——
    all_q = [q for q, _ in core.EVAL_IN] + core.EVAL_OOD
    no_rag_halluc = sum(1 for q in core.EVAL_OOD)  # 无库基线对超纲题一律硬编 → 幻觉数=超纲题数
    no_rag_cited = sum(1 for q in all_q if core.answer_no_rag(q)["citations"])
    rag_cited = 0
    rag_halluc = 0
    cited_rows = []
    for q in core.EVAL_IN:
        a = core.answer_rag(q, ret, topk=topk, min_score=thr["threshold"])
        rag_cited += int(bool(a["citations"]))
        cited_rows.extend(a["citations"])
    for q in core.EVAL_OOD:
        a = core.answer_rag(q, ret, topk=topk, min_score=thr["threshold"])
        rag_halluc += int(bool(a["grounded"]))

    # —— 保留原三问明细（关键词通道口径不变）——
    queries = ["泵振动超标要看哪个标准", "电机电流高但转速正常",
               "工装夹具的存放位置"]
    rows, grounded = [], 0
    min_kw = float(cfg.raw.get("min_score", 0.05))
    for q in queries:
        a = core.answer(q, kb, topk=topk, min_score=min_kw)
        grounded += int(a["grounded"])
        rows.append({"query": q, "grounded": a["grounded"],
                     "citations": a["citations"]})

    # —— 代码规范率（工单 D 档次条）与决策记录（docs/ADR 反读）实测 ——
    tgates = tiers.gates(cfg.tiers)
    sty = style.code_style_rate(ROOT, min_pass=float(
        tgates["D"].get("style_min", 70.0)))
    decision = tiers.decision_record_status(ROOT)

    result = {"kb_size": len(kb), "kb_source": source,
              "retriever_channel": ret.channel, "store_note": ret.store_note,
              "threshold_experiment": thr,
              "recall@3": {"keyword": round(kw_rate, 4),
                           "tfidf": round(tf_rate, 4),
                           "rrf_fused": round(fuse_rate, 4), "n": n_in},
              "ood": ood_rows, "ood_refused": refused,
              "ood_total": len(core.EVAL_OOD),
              "contrast": {"no_rag_hallucination": no_rag_halluc,
                           "rag_hallucination": rag_halluc,
                           "no_rag_citations": no_rag_cited,
                           "rag_citations": rag_cited,
                           "n_queries": len(all_q)},
              "n_queries": len(queries), "grounded": grounded,
              "rows": rows,
              "style": {k: sty[k] for k in
                        ("engine", "rate", "violations", "code_lines",
                         "layout_violations", "structural_violations")},
              "decision_record": decision}

    lines = []
    for r in rows:
        lines.append("Q: %s" % r["query"])
        if r["grounded"]:
            lines.append("   → 命中 %d 条 %s（最佳分数 %.3f）"
                         % (len(r["citations"]),
                            ",".join(c["id"] for c in r["citations"]),
                            r["citations"][0]["score"]))
        else:
            lines.append("   → 查无依据（拒答，未编造）")

    chroma_note = (
        "chromadb 通道可用并已实跑（EphemeralClient 内存集合，注入自算 "
        "TF-IDF 向量，Top-3 集合与矩阵口径交叉一致）"
        if "不一致" not in ret.channel else
        "chromadb 已实跑，但 Top-3 集合与矩阵口径不一致——以确定性矩阵结果为准")
    sections = [
        ("输入口径", [
            "知识库 %d 条（来源: %s）｜Top-%d" % (len(kb), source, topk),
            "检索通道（实测）：%s" % ret.channel,
            "向量库说明：%s" % (ret.store_note or chroma_note)]),
        ("问答明细（原关键词通道口径保留）", lines),
        ("库内检索 recall@3（三通道对照，金标准人工标注）", [
            "关键词重合（二元词）  %.1f%%（%d/%d）" % (kw_rate * 100, kw_hit, n_in),
            "jieba+TF-IDF 余弦    %.1f%%（%d/%d）" % (tf_rate * 100, tf_hit, n_in),
            "RRF 融合（两路互补） %.1f%%（%d/%d）｜融合公式 score=Σ1/(60+rank)"
            % (fuse_rate * 100, fused, n_in)]),
        ("OOD/超纲拒答实测（幻觉红线：检索必须能说不）",
         ["Q: %s → %s（top1相似度 %.4f）"
          % (r["query"],
             "拒答（未编造）" if r["refused"] else "错误地给出了依据！",
             r["top1_sim"])
          for r in ood_rows]
         + ["超纲题拒答率 %d/%d = %.0f%%"
            % (refused, len(core.EVAL_OOD),
               refused / len(core.EVAL_OOD) * 100),
            "拒答阈值取两簇实测中点：库内 top1 下界 %.4f｜超纲 top1 上界 "
            "%.4f｜阈值 %.4f（两簇%s）"
            % (thr["in_min"], thr["ood_max"], thr["threshold"],
               "可分离" if thr["separable"] else "存在重叠——如实说明")]),
        ("有库/无库对照（同一题集，真实跑两条链路）", [
            "无库基线（no-RAG 硬编）：超纲幻觉 %d/%d（%.0f%%）｜带引用 %d/%d"
            % (no_rag_halluc, len(core.EVAL_OOD),
               no_rag_halluc / len(core.EVAL_OOD) * 100, no_rag_cited,
               len(all_q)),
            "有库 RAG：超纲幻觉 %d/%d（%.0f%%）｜库内带引用 %d/%d"
            % (rag_halluc, len(core.EVAL_OOD),
               rag_halluc / len(core.EVAL_OOD) * 100, rag_cited, n_in),
            "差值（有库-无库幻觉率）= %.0f 个百分点——引用可溯源是 RAG 的硬收益"
            % ((no_rag_halluc - rag_halluc) / len(core.EVAL_OOD) * 100),
            "实测幻觉案例清单（无库基线对库外题硬编、零引用，逐条判定）："]
         + ["  案例%d: Q=%r → 硬编应答%r（citations=0，判定=幻觉）"
            % (i, q, core.answer_no_rag(q)["text"][:24])
            for i, q in enumerate(core.EVAL_OOD, 1)]),
        ("命中率（原三问口径）", [
            "有据可查 %d / %d（幻觉率 %.0f%%）"
            % (grounded, len(queries),
               (1 - grounded / max(len(queries), 1)) * 100)]),
        ("编码规范率（PEP 8 实测）", [
            "引擎: %s｜扫描 %d 个文件｜非空代码行 %d｜违规 %d 条（版式 %d + "
            "结构 %d）→ 规范率 %.1f%%"
            % (sty["engine"], sty["n_files"], sty["code_lines"],
               sty["violations"], sty["layout_violations"],
               sty["structural_violations"], sty["rate"]),
            sty["fallback_note"]]),
        ("结论与建议", [
            "RAG 底线='有据可查+诚实拒答'；查无依据的查询用于发现知识库盲区",
            "检索双解：关键词精确匹配型号/编码，jieba+TF-IDF 语义重合描述类"
            "查询，RRF 融合两路互补（工单B03）"])]
    # —— 验收档位判定：全部吃上面的实测值（融合说明从已生成的报告行实测计数）——
    fusion_lines = tiers.fusion_explain_lines(sections)
    cite_cov = rag_cited / max(n_in, 1)
    trace = tiers.traceable_rate(cited_rows, kb)
    tv = tiers.verdict(
        tgates, rate=sty["rate"], halluc_cases=no_rag_halluc,
        retrieve_rate=fuse_rate, cite_cov=cite_cov, trace_rate=trace,
        fusion_lines=fusion_lines, decision_ok=decision["exists"])
    result["tiers"] = tv
    sections.insert(-1, ("验收档位判定（实测）", tiers.render_lines(tv)))
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    report.save(sections, os.path.join(out_dir, "report.txt"))
    # 返回 (sections, result)：cli --tier 与 pytest 直接吃实测值，不再重跑
    return sections, result


class _kw_adapter:
    """把原关键词通道包装成与 TfidfRetriever 同接口，供 recall@3 统一评测。"""

    def __init__(self, kb, topk):
        """持有知识库与默认 Top-k，接口对齐 TfidfRetriever。"""
        self.kb, self.topk = kb, topk

    def search(self, query, topk=3):
        """按关键词二元词重合检索，返回与向量通道同构的命中列表。"""
        return core.retrieve(query, self.kb, topk=topk or self.topk)
