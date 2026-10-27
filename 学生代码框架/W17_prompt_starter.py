# -*- coding: utf-8 -*-
"""M4-1 LLM诊断·三版Prompt starter（教学周9）：同一个问题三种问法，量化Prompt价值

离线可跑：内置"规则LLM"（关键词命中评分模拟回答质量）；
真机版：把 call_llm 换成 POST {OPENAI_BASE_URL}/v1/chat/completions 或
        工具/../campus_skills 的同款协议（Prompt工程不分平台）。
"""
import json
import os
import urllib.request

KB = {"更换轴承": ["轴承", "磨损", "特征频率"], "清洗叶轮": ["叶轮", "堵塞", "压力波动"],
      "紧固螺栓": ["螺栓", "松动", "地脚"]}


def call_llm(prompt: str, question: str) -> str:
    """三通道：真实LLM(设了BASE_URL) → 否则规则模拟（教学离线兜底）"""
    base = os.environ.get("OPENAI_BASE_URL")
    if base:
        body = json.dumps({"model": "starter", "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}]}).encode()
        r = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                   data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=20).read())["choices"][0]["message"]["content"]
    for action, keys in KB.items():                # 规则模拟：命中关键词→给对应处置
        if any(k in prompt + question for k in keys):
            return f"建议：{action}（依据：{question[:12]}…）"
    return "建议：检查电源。"


def score(answer: str) -> float:
    """回答质量分：正确处置命中=1.0，错误处置=0.0（工单A/B对比的量化口径）"""
    return 1.0 if ("轴承" in answer and "更换" in answer) else 0.5 if "建议" in answer else 0.0


PROMPTS = {
    "朴素": "帮我看看设备怎么了。",
    "角色": "你是设备运维工程师。泵振动特征频率异常，轴承磨损可能性大。请给出处置建议。",
    "示例": "示例：输入'电机异响发热'→输出'建议：检查轴承润滑'。"
    "现在：泵振动8Hz/16Hz特征频率抬升，请按示例格式给处置建议。",
}
# 避坑：问题本身**不能**含"轴承/特征频率"等诊断关键词——否则朴素Prompt靠题目泄漏也能答对，
#       三版对比就失去了意义。这里只给"症状"（现场工况），诊断先验必须由 Prompt 提供。
QUESTION = "3号泵振动在爬升，运转声音发闷，帮我看看。"


def main():
    results = {}
    for tag, p in PROMPTS.items():
        # TODO-W9-1: ans = call_llm(p, QUESTION)，results[tag] = score(ans)
        pass
    if results:
        base = results["朴素"]
        for tag, s in results.items():
            print(f"{tag}: 质量分={s:.1f}（较朴素{'+' if s >= base else ''}{(s - base) * 100:.0f}%）")
        print("目标：角色/示例版较朴素提升≥50% —— 三版对比写进Prompt实验记录表")
    else:
        print("完成TODO-W9-1（一行调用+一行评分）——三版Prompt立见高下")


if __name__ == "__main__":
    main()
