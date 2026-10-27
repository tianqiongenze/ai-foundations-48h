# -*- coding: utf-8 -*-
"""Skill① 课表助手——解析课表JSON，回答今天/下一节/空闲

数据文件 my_schedule.json 学生自己维护（也练了JSON读写）。
"""
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SCHED = os.path.join(HERE, "my_schedule.json")

DEFAULT = {
    "周一": [["08:00", "09:40", "Python编程基础"],
           ["10:00", "11:40", "工业数据采集"]],
    "周二": [["14:00", "15:40", "AI应用基础(48课时)"],
           ["16:00", "17:40", "体育"]],
    "周三": [["08:00", "09:40", "高等数学"],
           ["10:00", "11:40", "思政课"]],
    "周四": [["14:00", "17:40", "AI实训(工单日)"]],
    "周五": [["08:00", "09:40", "英语"],
           ["10:00", "11:40", "Python编程基础"]],

}

WEEK_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _load():
    if not os.path.exists(SCHED):
        with open(SCHED, "w", encoding="utf8") as f:
            json.dump(DEFAULT, f, ensure_ascii=False, indent=1)
    return json.load(open(SCHED, encoding="utf8"))


def _today() -> str:
    return WEEK_CN[datetime.now().weekday()]


def _hm(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _now_min():
    d = datetime.now()
    return d.hour * 60 + d.minute


def today_courses(day: str = None) -> dict:
    day = day or _today()
    table = _load().get(day, [])
    now = _now_min()
    for s, e, name in table:
        if _hm(s) <= now < _hm(e):
            status = "在上"
            break
    else:
        status = "课间/无课"
    return {"day": day, "courses": [{"start": s, "end": e, "name": n} for s, e, n in table],
            "summary": f"{day}共{len(table)}节：" +
                       "、".join(f"{s} {n}" for s, e, n in table) or "无课",
            "status": status,
            "free_summary": _free_text(table)}


def next_course(day: str = None):
    day = day or _today()
    now = _now_min()
    for s, e, name in _load().get(day, []):
        if _hm(s) > now:
            return {"start": s, "name": name, "in_min": _hm(s) - now}
    return None


def _free_text(table):
    slots = free_slots_from(table)
    return "、".join(f"{a}-{b}" for a, b in slots) or "无"


def free_slots_from(table, day=None):
    """课外空闲段：默认 08:00-18:00 减去课程区间"""
    day = day or _today()
    busy = [(_hm(s), _hm(e)) for s, e, n in table]
    free, cur = [], 8 * 60
    for bs, be in sorted(busy):
        if bs > cur:
            free.append((cur, bs))
        cur = max(cur, be)
    if cur < 18 * 60:
        free.append((cur, 18 * 60))
    return [(f"{a//60:02d}:{a % 60:02d}", f"{b//60:02d}:{b % 60:02d}") for a, b in free]


def free_slots(day: str = None):
    return {"day": day or _today(), "slots": free_slots_from(_load().get(day or _today(), []))}


if __name__ == "__main__":
    print(json.dumps(today_courses(), ensure_ascii=False, indent=1))
    print("下一节:", next_course())
    print("空闲:", free_slots())
