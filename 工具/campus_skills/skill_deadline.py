# -*- coding: utf-8 -*-
"""Skill② DDL管家——作业/任务清单 → WSPT价值密度排序 → 今日最急Top3

排序规则（与工单M5-1同源，学以致用）：
  价值密度 = 重要度 / 剩余工作量(小时)，高者优先；剩余<24h直接置顶（紧急穿透）。
数据文件 my_ddl.json 学生自己维护。
"""
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DDL = os.path.join(HERE, "my_ddl.json")

DEFAULT = [
    {"title": "AI48工单M4-1（LLM诊断）", "due": "明天 18:00", "hours": 3, "weight": 9},
    {"title": "高数作业第3章", "due": "后天 08:00", "hours": 1.5, "weight": 6},
    {"title": "英语口语准备", "due": "周五 14:00", "hours": 2, "weight": 4},
    {"title": "社团活动策划", "due": "下周三", "hours": 4, "weight": 3},
]


def _load():
    if not os.path.exists(DDL):
        with open(DDL, "w", encoding="utf8") as f:
            json.dump(DEFAULT, f, ensure_ascii=False, indent=1)
    return json.load(open(DDL, encoding="utf8"))


def _hours_left(due: str) -> float:
    """解析'明天 18:00'/'后天 08:00'/'周五 14:00'/'下周三'风格→剩余小时（粗粒度教学版）"""
    now = datetime.now()
    table = {"今天": 0, "明天": 24, "后天": 48, "大后天": 72}
    for k, h in table.items():
        if due.startswith(k):
            hh, mm = map(int, (due.split()[1] if len(due.split()) > 1 else "23:59").split(":"))
            return h + (hh - now.hour) + (0 if now.minute < 60 else 0)
    if due.startswith("周五"):
        return 24 * ((4 - now.weekday()) % 7 or 7)
    if due.startswith("下周"):
        return 24 * 7
    return 48.0


def top_urgent(n: int = 3) -> list:
    items = _load()
    for it in items:
        hl = _hours_left(it["due"])
        density = it["weight"] / max(it["hours"], 0.5)     # WSPT价值密度
        it["_hl"], it["_d"] = hl, density
        it["why"] = (f"紧急！仅剩~{hl:.0f}h" if hl < 24
                     else f"价值密度{density:.1f}最高（重要{it['weight']}/工时{it['hours']}h）")
    items.sort(key=lambda x: (-1 if x["_hl"] < 24 else 0, -x["_d"]))
    return [{"title": it["title"], "due": it["due"], "why": it["why"]}
            for it in items[:n]]


if __name__ == "__main__":
    print(json.dumps(top_urgent(), ensure_ascii=False, indent=1))
