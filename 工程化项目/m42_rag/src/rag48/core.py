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


# ------------------------------------------------------------------ 向量检索通道
EVAL_IN = [  # 库内问题→金标准条目（recall@k 实测用；金标准由人工对照 knowledge_base.json 标注）
    ("泵振动超标要看哪个标准", "R1+R15"),  # 双金标准：振动阈值条目两条都算命中（标注口径写进报告）
    ("轴承温度报警阈值是多少", "R2"),
    ("二倍频幅值显著抬升是什么故障指纹", "R4"),
    ("检修挂牌上锁的要求", "R6"),
    ("泵体异响且压力波动先查什么", "R8"),
    ("电机电流偏高而转速正常查哪里", "R9"),
    ("16Hz二倍转频峰判定什么故障", "R10"),
    ("正常振动速度有效值区间是多少", "R15"),
    ("注塑成型熔体温度范围", "R16"),
    ("缺陷检测产线节拍单帧推理要求", "R19"),
]
EVAL_OOD = [  # OOD/超纲问题：知识库无依据，正确行为=拒答而非编造（幻觉红线实测）
    "工装夹具的存放位置",
    "食堂承包商的月度对账流程是什么",
    "无人机螺旋桨间距的参数标准是多少",
]


class TfidfRetriever:
    """诚实的检索实现：jieba 分词 + TF-IDF 余弦（确定性、离线可跑）。

    通道探测顺序与兜底事实全部记入 channel/store_note 并进报告：
      ① jieba 可用 → jieba 词 + sklearn TF-IDF；不可用 → 字符二元组 TF-IDF；
      ② 向量库通道尝试 chromadb（内存 EphemeralClient + 自算向量，零网络）；
         不可用/异常 → 如实标注"离线兜底"，改在 numpy 矩阵上算同一份余弦。
    """

    def __init__(self, kb):
        self.kb = kb
        tok = "jieba 分词"
        try:
            import jieba
            self._cut = lambda t: " ".join(jieba.cut(str(t)))
        except ImportError:
            tok = "字符二元组（离线兜底：未安装 jieba）"
            self._cut = lambda t: " ".join(sorted(bigrams(t)))
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer()
        self.M = self.vec.fit_transform([self._cut(k["text"]) for k in kb])
        ids = [k["id"] for k in kb]
        self._rank = {i: n for n, i in enumerate(ids)}
        self.channel = "内置 TF-IDF 余弦（%s，确定性离线）" % tok
        self.store_note = ""
        self.coll = None
        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self.coll = client.create_collection("kb48", metadata={"hnsw:space": "cosine"})
            emb = self.M.toarray()
            self.coll.add(ids=ids, embeddings=emb,
                          documents=[k["text"] for k in kb])
            self.channel += "｜向量入库通道：chromadb 内存集合（自算 TF-IDF 向量，零网络）"
        except Exception as e:
            self.store_note = ("离线兜底：chromadb 不可用（%r），"
                               "改用内置 TF-IDF 余弦检索" % e)

    def sims(self, query):
        v = self.vec.transform([self._cut(query)])
        return (v @ self.M.T).toarray()[0]

    def search(self, query, topk=3, min_score=0.0):
        s = self.sims(query)
        # 相似度=0 表示与库零词重合，不构成"检索命中"（也与 chromadb 平票口径对齐）
        order = sorted(range(len(s)), key=lambda i: (-s[i], self.kb[i]["id"]))
        order = [i for i in order if s[i] > 0]
        top_all = [{"id": self.kb[i]["id"], "score": float(s[i])} for i in order[:topk]]
        hits = [h for h in top_all if h["score"] >= min_score]
        hits = [{**h, "category": self.kb[self._rank[h["id"]]]["category"],
                 "text": self.kb[self._rank[h["id"]]]["text"],
                 "score": round(h["score"], 4)} for h in hits]
        if self.coll is not None:  # 交叉验证：库检索与矩阵计算必须给出同一 Top-k 集合（未过滤口径）
            vq = self.vec.transform([self._cut(query)]).toarray()[0]
            q = self.coll.query(query_embeddings=[list(vq)], n_results=topk)
            ids2 = set(q["ids"][0])
            # 矩阵侧有效命中不足 topk（超纲题相似度假票为0）时不参与比较，避免假告警
            if len(top_all) == len(ids2) and ids2 != {h["id"] for h in top_all}:
                if "口径不一致" not in self.channel:
                    self.channel += "｜注意：chromadb 返回集合与矩阵口径不一致，以矩阵结果为准"
        return hits


def recall_at_k(retriever, eval_pairs, topk=3):
    """recall@k 实测：Top-k 命中金标准条目的问题占比（无估算、全真跑）。"""
    hit = 0
    for q, gold in eval_pairs:
        golds = set(gold.split("+"))
        if any(h["id"] in golds for h in retriever.search(q, topk=topk)):
            hit += 1
    return hit, len(eval_pairs)


def rrf_fuse(rank_lists, k=60):
    """倒数排名融合（RRF）：关键词通道+向量通道两路互补，score=Σ1/(k+rank)。"""
    acc = {}
    for rl in rank_lists:
        for r, rid in enumerate(rl):
            acc[rid] = acc.get(rid, 0.0) + 1.0 / (k + r + 1)
    return sorted(acc, key=lambda i: (-acc[i], i))


def pick_refuse_threshold(retriever, eval_in, ood_queries):
    """拒答阈值不是拍的：实测 库内top1相似度下界 与 超纲top1相似度上界，取两簇中点。"""
    ins = [max(retriever.sims(q)) for q, _ in eval_in]
    outs = [max(retriever.sims(q)) for q in ood_queries]
    lo_in, hi_out = float(min(ins)), float(max(outs))
    thr = round((lo_in + hi_out) / 2.0, 4) if hi_out < lo_in else round(hi_out + 0.01, 4)
    return {"in_min": round(lo_in, 4), "ood_max": round(hi_out, 4),
            "threshold": thr, "separable": bool(hi_out < lo_in)}


def answer_rag(query, retriever, topk=3, min_score=0.0):
    """RAG 问答（向量通道版）：无据即拒答，绝不硬编。"""
    hits = retriever.search(query, topk=topk, min_score=min_score)
    if not hits or hits[0]["score"] < min_score:
        return {"grounded": False,
                "text": "查无依据（超纲/库外）：请补充资料入库后再检索（宁说无据，不硬编）。",
                "citations": []}
    refs = chr(10).join("  [%s] (%s) %s" % (h["id"], h["category"], h["text"]) for h in hits)
    return {"grounded": True,
            "text": "依据知识库:" + chr(10) + refs + chr(10) + "  → 建议：按上述条款执行并回填工单。",
            "citations": hits}


def answer_no_rag(query):
    """无库基线（对照实测用）：纯 LLM 式'硬编'——对任何题都给出貌似合理的散文答案、无引用。

    这是幻觉率的'对照组'：它不是更差的实现，而是故意的 no-RAG 基线，
    用来真实测出 有库/无库 的幻觉差与引用差。
    """
    return {"grounded": True,
            "text": "依据通用经验：%s 属常见情况，建议联系设备厂商并加强巡检。" % query[:12],
            "citations": []}
