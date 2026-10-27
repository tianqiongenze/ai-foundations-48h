# -*- coding: utf-8 -*-
"""三版 Prompt 对比：通道可注入 + 质量评分（纯函数）。"""

PROMPTS = {
    "朴素": "帮我看看设备怎么了。",
    "角色": "你是设备运维工程师。泵振动特征频率异常，轴承磨损可能性大。请给出处置建议。",
    "示例": "示例：输入'电机异响发热'→输出'建议：检查轴承润滑'。"
          "现在：泵振动8Hz/16Hz特征频率抬升，请按示例格式给处置建议。",
}
QUESTION = "3号泵振动在爬升，运转声音发闷，帮我看看。"
KEYWORDS = ["轴承", "磨损", "特征频率", "16Hz", "8Hz", "润滑"]


class MockChannel:
    """离线规则通道：按 Prompt+Question 命中关键词给答案（课堂可复现）。"""

    def __call__(self, system, user):
        text = system + user
        if any(k in text for k in KEYWORDS):
            return "建议：更换轴承（依据：%s…" % user[:12]
        return "建议：检查电源。"


def call(channel, system, user):
    return channel(system, user)


def score(answer):
    """质量分：命中正确处置=1.0，给出建议=0.5，否则 0。"""
    if "轴承" in answer and "更换" in answer:
        return 1.0
    if "建议" in answer:
        return 0.5
    return 0.0


def compare(channel):
    base = None
    out = {}
    for tag, p in PROMPTS.items():
        ans = call(channel, p, QUESTION)
        s = score(ans)
        if base is None:
            base = s
        out[tag] = {"score": s, "delta": round(s - base, 3), "answer": ans}
    return out, base
