# -*- coding: utf-8 -*-
"""Skill③ 专注计划——空闲时段 × 最急DDL → 番茄钟时间块（今日可执行计划）

规则：最急DDL优先占用最近的空闲段；25分钟番茄钟+5分钟休息；
     连续2个番茄后安排15分钟长休息（劳逸结合，拒绝疲劳战术）。
"""
from skill_deadline import top_urgent
from skill_schedule import _load, _today, _hm, free_slots_from


def _blocks(slot) -> list:
    """把 ('11:40','18:00') 空闲段切成番茄块：25'+5'，第3块起长休15'"""
    a, b = _hm(slot[0]), _hm(slot[1])
    out, used, nth = [], 0, 0
    while b - (a + used) >= 25:
        start = a + used
        mode = "🍅番茄25'+5'休息" if nth % 3 != 2 else "🍅番茄25'+长休15'"
        out.append((f"{start//60:02d}:{start % 60:02d}", 25, mode))
        used += 30 if nth % 3 != 2 else 40         # 2番茄后长休15'
        nth += 1
    return out


def make_plan() -> list:
    slots = free_slots_from(_load().get(_today(), []))
    tasks = [d["title"] for d in top_urgent()] + ["自由复习/预习"]
    plan, k = [], 0
    for s in slots:
        for start, mins, mode in _blocks(s):
            plan.append({"slot": f"{s[0]}-{s[1]}段 {start}起({mins}分钟)",
                         "do": tasks[min(k, len(tasks) - 1)], "mode": mode})
            k += 1
    return plan[:6]                                # 一天最多6块，留白也是时间管理


if __name__ == "__main__":
    import json
    print(json.dumps(make_plan(), ensure_ascii=False, indent=1))
