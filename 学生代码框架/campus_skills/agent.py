# -*- coding: utf-8 -*-
"""校园生活智能体（Skills版）——AI48 M4拓展实训：大学时间管理等3个技能

定位（呼应工单M4-1/M4-2与M5-1）：
- 用"自己的大学生活"当第一个智能体场景（先学会，再迁移到工业运维——同一套Skill架构）
- Skill=OpenAI tools协议规格 → 不限平台：本地agent / mock LLM / 扣子(Coze) / Dify
  导入同一份JSON规格即可（见 SKILL_EXPORT.md）

三个技能：
  skill_schedule  课表助手：今天有什么课/下一节/空闲时段
  skill_deadline  DDL管家：作业清单→WSPT价值密度排序→今日最急Top3
  skill_focus     专注计划：空闲段×DDL→番茄钟时间块计划

运行: /opt/anaconda3/bin/python agent.py "我今天下午有什么安排？最急的DDL是啥？"
     /opt/anaconda3/bin/python verify_skills.py   # 断言自检
"""
import json
import sys

from skill_schedule import today_courses, next_course, free_slots
from skill_deadline import top_urgent
from skill_focus import make_plan

SKILLS_SPEC = [
    # TODO-CS-3: 按 OpenAI tools 协议补全三个Skill的JSON规格（name/description/parameters）
    #   格式参考：{"type":"function","function":{"name":..., "description":..., "parameters":{...}}}
    #   完整实现见 工具/campus_skills/agent.py；规格写好后可导入扣子/Dify（见SKILL_EXPORT.md）
]

REGISTRY = {
    "skill_schedule": lambda a: {"today": today_courses(), "next": next_course(),
                                 "free": free_slots()}[a.get("query", "today")],
    "skill_deadline": lambda a: top_urgent(),
    "skill_focus": lambda a: make_plan(),
}


def plan_tools(question: str) -> list:
    """本地规划器（离线；设OPENAI_BASE_URL可换真实LLM/mock_llm_server做function calling）"""
    import os
    if os.environ.get("OPENAI_BASE_URL"):
        from urllib.request import Request, urlopen
        body = json.dumps({"model": "campus-agent",
                           "messages": [{"role": "user", "content": question}],
                           "tools": SKILLS_SPEC, "tool_choice": "auto"}).encode()
        r = urlopen(Request(os.environ["OPENAI_BASE_URL"].rstrip("/") + "/v1/chat/completions",
                            data=body, headers={"Content-Type": "application/json"}), timeout=20)
        msg = json.loads(r.read())["choices"][0]["message"]
        return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in msg.get("tool_calls", [])]
    order = []
    if any(k in question for k in ("课", "安排", "空闲", "下午", "上午")):
        order.append(("skill_schedule", {"query": "today"}))
    if any(k in question for k in ("DDL", "作业", "急", "deadline", "最急")):
        order.append(("skill_deadline", {}))
    if any(k in question for k in ("计划", "专注", "番茄", "怎么学", "时间")):
        order.append(("skill_focus", {}))
    return order or [("skill_schedule", {"query": "today"})]


def ask(question: str) -> dict:
    trace = []
    for name, args in plan_tools(question):
        obs = REGISTRY[name](args)
        trace.append({"skill": name, "obs": obs})
    return {"question": question, "trace": trace, "answer": _compose(trace)}


def _compose(trace):
    lines = []
    for t in trace:
        o = t["obs"]
        if t["skill"] == "skill_schedule":
            lines.append(f"📚 今天课程：{o['summary']}")
            if o.get("next"):
                lines.append(f"   下一节：{o['next']}")
            lines.append(f"   空闲时段：{o['free_summary']}")
        elif t["skill"] == "skill_deadline":
            lines.append("⏰ 今日最急DDL Top3：")
            lines += [f"   {i+1}. {d['title']}（{d['why']}）" for i, d in enumerate(o)]
        elif t["skill"] == "skill_focus":
            lines.append("🍅 专注计划：")
            lines += [f"   {b['slot']} → {b['do']}（{b['mode']}）" for b in o]
    lines.append("（由校园智能体生成：先课后急、劳逸结合——WSPT价值密度调度）")
    return "\n".join(lines)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "我今天有什么安排？最急的DDL是啥？给我个专注计划"
    print(ask(q)["answer"])
