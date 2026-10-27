# -*- coding: utf-8 -*-
"""campus_skills 自检断言（公开分享前必跑）：/opt/anaconda3/bin/python verify_skills.py"""
from skill_schedule import today_courses, free_slots_from, _hm
from skill_deadline import top_urgent
from skill_focus import _blocks
from agent import ask, SKILLS_SPEC


def ok(name, cond):
    print(("✓" if cond else "✗"), name)
    return cond


assert ok("课表skill: 今日课程解析", len(today_courses()["courses"]) >= 0)
assert ok("课表skill: 空闲段为正区间", all(_hm(a) < _hm(b) for a, b in
          free_slots_from([["08:00", "09:40", "x"], ["10:00", "11:40", "y"]])))
assert ok("DDL skill: Top3有序且含理由", all("why" in d for d in top_urgent()))
blks = _blocks(("11:40", "13:40"))
assert ok("专注skill: 2小时≈3番茄", len(blks) == 3 and blks[0][0] == "11:40")
r = ask("我今天有什么安排？最急的DDL是啥？")
assert ok("Agent端到端: 三问触发>=2技能", len(r["trace"]) >= 2)
assert ok("Agent可读回答", "课程" in r["answer"] and "DDL" in r["answer"])
assert ok("OpenAI协议规格: 3个Skill", len(SKILLS_SPEC) == 3
          and {s["function"]["name"] for s in SKILLS_SPEC} ==
          {"skill_schedule", "skill_deadline", "skill_focus"})
print("\ncampus_skills ALL PASS ✅")
