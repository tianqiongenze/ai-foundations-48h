# -*- coding: utf-8 -*-
"""
本地模拟LLM服务（离线可用）—— M4.2《工业智能决策助手》课内必做环境
v2.0（33号轮）：Web层升级 Litestar(全异步) + Granian(Rust运行时)，
替代初版 http.server —— 接口行为100%兼容，01/AI48两包工单代码零改动。

设计要点（对应v2.0可行性红线）：
1. OpenAI兼容接口 POST /v1/chat/completions，学生可用官方openai SDK或requests调用；
2. 内置"知识检索(RAG) + 规则生成"双通道：
   - 请求带 context 字段(RAG检索结果)时优先引用知识库条目；
   - 无 context 时按bigram重合度匹配本地 knowledge_base.json（教学直观解）；
     教师升级演示：Chroma向量检索版见 教师参考/rag_chroma_demo.py（解B）；
3. HALLUCINATE=1 模拟真实LLM的输出波动，供"容错（重试/校验/降级）"教学。

启动:  python mock_llm_server.py [port]        # 默认8000（内部转由granian承载）
      python -m granian --interface asgi "mock_llm_server:app" --port 8000 --workers 2
自测:  curl http://127.0.0.1:8000/health
"""
import json
import os
import re
import sys

from litestar import Litestar, Request, get, post

KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "data", "knowledge_base.json")
if not os.path.exists(KB_PATH):
    KB_PATH = os.path.join("data", "knowledge_base.json")

HALLUCINATE = os.environ.get("HALLUCINATE", "0") == "1"


def load_kb():
    try:
        with open(KB_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _tokens(text):
    """中文按字符bigram切分(无分词器也可用), 英数按词切分。"""
    toks = []
    runs = re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z0-9.]+", text)
    for r in runs:
        if re.match(r"[\u4e00-\u9fa5]", r):
            toks += [r[i:i + 2] for i in range(len(r) - 1)]
            toks.append(r)                     # 整词也保留(长词命中权重高)
        else:
            toks.append(r.lower())
    return set(toks)


def retrieve(query, topk=2):
    """极简检索（教学解A）：字符bigram+整词重合度打分。
    工程解B（Chroma向量库）见 教师参考/rag_chroma_demo.py，两解可同题互验。"""
    kb = load_kb()
    q = _tokens(query)
    scored = []
    for cat, items in kb.items():
        for it in items:
            score = len(q & _tokens(it)) / (len(q) + 1e-6)
            scored.append((score, cat, it))
    scored.sort(reverse=True)
    return [s for s in scored[:topk] if s[0] > 0]


def compose_reply(messages):
    """规则式回答生成：优先RAG上下文，其次本地检索，最后兜底。"""
    user = ""
    for m in messages:
        if m.get("role") == "user":
            user = m.get("content", "")
    ctx = ""
    for m in messages:
        if m.get("role") == "system" and m.get("content", "").startswith("CONTEXT:"):
            ctx = m["content"][len("CONTEXT:"):]

    if ctx:  # 学生RAG检索到的知识 → 引用生成（RAG通道）
        return (f"根据知识库检索到以下依据：\n{ctx.strip()}\n"
                f"综合建议：请按上述规范执行，先做安全隔离再逐项排查，"
                f"处理完毕后在工单系统记录本次事件与更换件清单。"), True

    hits = retrieve(user)
    if hits:
        refs = "\n".join(f"- [{c}] {t}" for _, c, t in hits)
        return (f"针对「{user[:40]}」，本地知识库给出以下参考：\n{refs}\n"
                f"请结合现场实测数据确认后执行；如仍异常请升级至设备工程师。"), True

    if HALLUCINATE:  # 演示"幻觉+不稳定", 用于容错教学
        return "建议直接更换整套设备并联系厂家。（模拟幻觉输出，用于容错演示）", False
    return ("知识库暂无直接匹配条目。建议：1)核对设备编号与测点；"
            "2)提供振动/温度/电流实测值后再检索；3)按《设备运维规范》通用条款先行安全隔离。",
            False)


@get("/health")
async def health() -> dict:
    return {"status": "ok", "kb": os.path.exists(KB_PATH),
            "hallucinate": HALLUCINATE, "stack": "litestar+granian"}


@post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict:
    try:
        data = await request.json()
        answer, grounded = compose_reply(data.get("messages", []))
        return {
            "id": "mock-001",
            "object": "chat.completion",
            "model": data.get("model", "mock-industrial-llm"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
            "grounded": grounded,          # 教学用: 是否基于知识库
        }
    except Exception:                     # 模拟偶发服务异常→学生做重试
        raise


app = Litestar([health, chat_completions], logging_config=None)


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "8000"
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"本地模拟LLM启动: http://0.0.0.0:{port}  (知识库: {KB_PATH})")
    print("OpenAI兼容端点: POST /v1/chat/completions ；健康检查: GET /health")
    print("安全提示: 监听0.0.0.0仅供机房局域网教学使用, 切勿暴露到公网")
    os.chdir(here)
    os.execv(sys.executable, [sys.executable, "-m", "granian", "--interface", "asgi",
                              "mock_llm_server:app", "--host", "0.0.0.0",
                              "--port", str(port), "--workers", "1"])
