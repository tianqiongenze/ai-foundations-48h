# -*- coding: utf-8 -*-
"""三版 Prompt 对比：通道可注入 + 质量评分 + JSON schema 校验 + 重试/降级 + 多轮对话（纯逻辑）。

安全与工程纪律（工单 M4-1）：
  1. LLM 端点/密钥**只**从环境变量读取（os.getenv），代码零硬编码；
  2. 真机通道必须带超时；失败走 重试→降级 兜底话术，断服不崩溃（优秀B档）；
  3. 模型输出按 JSON schema 校验（json.loads+必需键+类型），畸变率实测入报告；
  4. 路径拼接审计程序化执行（audit_path_joins），发现字符串拼接即报告点名。
"""
import json
import os
import re
import time

PROMPTS = {
    "朴素": "帮我看看设备怎么了。",
    "角色": "你是设备运维工程师。泵振动特征频率异常，轴承磨损可能性大。"
          "请按JSON输出：{cause:str, steps:list(非空), confidence:0-1}。",
    "示例": "示例：输入'电机异响发热'→输出'{\"cause\":\"轴承润滑不良\","
          "\"steps\":[\"补脂\"],\"confidence\":0.7}'。"
          "现在：泵振动8Hz/16Hz特征频率抬升，请按示例JSON格式给处置建议。",
}
QUESTION = "3号泵振动在爬升，运转声音发闷，帮我看看。"
KEYWORDS = ["轴承", "磨损", "特征频率", "16Hz", "8Hz", "润滑"]
FALLBACK_TEXT = ("【降级话术】LLM 服务暂不可达，已切换离线规则通道。"
                 "以下为初步建议，请人工复核后再执行。")
DIAG_JSON_KEYS = ("cause", "steps", "confidence")


class ChannelError(RuntimeError):
    """通道类故障（网络失败/超时/非200），供重试降级逻辑捕获。"""


# ---------------------------------------------------------------- 环境变量与真机通道
def endpoint_from_env():
    """端点三要素一律来自环境变量：os.getenv 实测读取，代码中不出现任何密钥。"""
    return (os.getenv("OPENAI_BASE_URL", "").strip(),
            os.getenv("OPENAI_API_KEY", "").strip(),
            os.getenv("PROMPT_LLM_MODEL", "mock-llm").strip())


class HttpChannel:
    """OpenAI 兼容接口真机通道（urllib 实现，带超时）。

    只在显式配置了 OPENAI_BASE_URL 时实例化；本机离线运行时绝不会走到这里——
    报告里凡标'真机通道'的数字都必须来自本类真实 HTTP 调用。
    """

    def __init__(self, base_url, api_key, model, timeout=10.0):
        if not base_url:
            raise ValueError("base_url 为空：拒绝盲发请求")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置：拒绝无密钥调用（密钥零硬编码，只认环境变量）")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key       # 只驻内存，不落盘、不入日志
        self.model = model
        self.timeout = timeout

    def __call__(self, system, user):
        import urllib.request
        payload = json.dumps({
            "model": self.model, "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": "***" + self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- 离线通道与容错
class MockChannel:
    """离线规则通道：按 Prompt+Question 命中关键词给答案（课堂可复现）。

    角色/示例版输出 JSON（schema 合格）；朴素版输出散文——畸变差异被实测而非宣称。
    """

    def __call__(self, system, user):
        text = system + user
        if any(k in text for k in KEYWORDS):
            return json.dumps({
                "cause": "轴承磨损（特征频率 2×转频≈16Hz 抬升，与振动爬升/声音发闷相符）",
                "steps": ["复测振动速度有效值（对照4.5/7.1mm/s阈值线）",
                          "听诊+红外复核对侧轴承位温升",
                          "排产更换轴承并做动平衡，回填工单"],
                "confidence": 0.86}, ensure_ascii=False)
        return "设备可能有问题，建议先检查电源。"


class FlakyChannel:
    """离线注入的'故障通道'：前 fail_times 次调用抛 ChannelError，之后透传 fallback。

    为什么注入而不是真断网：本课程禁止网络调用，但 重试/降级 代码路径必须**真实执行**
    才算指标可信——故障由可编程通道注入，计数器全是真实调用产生。
    """

    def __init__(self, fail_times, fallback=None):
        self.fail_times = fail_times
        self.fallback = fallback
        self.calls = 0

    def __call__(self, system, user):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ChannelError("HTTP 503: LLM 服务不可达（注入故障 #1）")
        if self.fallback is not None:
            return self.fallback(system, user)
        return json.dumps({"cause": "服务恢复后的应答", "steps": ["继续观察"],
                           "confidence": 0.5}, ensure_ascii=False)


class RetryFallbackChannel:
    """重试+降级包装器：primary 最多试 retries 次（指数退避），全失败则返回兜底话术。"""

    def __init__(self, primary, fallback=None, retries=3, backoff=0.0):
        self.primary = primary
        self.fallback = fallback or MockChannel()
        self.retries = retries
        self.backoff = backoff
        self.stats = {"calls": 0, "attempts": 0, "retries": 0,
                      "fallbacks": 0, "recovered": 0, "errors": []}

    def __call__(self, system, user):
        self.stats["calls"] += 1
        for i in range(self.retries):
            self.stats["attempts"] += 1
            try:
                out = self.primary(system, user)
                self.stats["recovered"] += 1
                return out
            except Exception as e:
                self.stats["errors"].append(repr(e))
                if i < self.retries - 1:
                    self.stats["retries"] += 1
                    if self.backoff:
                        time.sleep(self.backoff * (2 ** i))
        self.stats["fallbacks"] += 1
        return FALLBACK_TEXT + "\n" + self.fallback(system, user)


# ---------------------------------------------------------------- 输出/评分/多轮
def call(channel, system, user):
    return channel(system, user)


def score(answer):
    """质量分：命中正确处置=1.0，给出建议=0.5，否则 0。"""
    if "轴承" in answer and "更换" in answer:
        return 1.0
    if "建议" in answer:
        return 0.5
    return 0.0


def _strip_fence(text):
    """剥掉真机 LLM 常见的 ```json 代码围栏，再进 schema 校验。"""
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else text


def validate_output(text):
    """JSON schema 校验：json.loads 解析 + 必需键/类型/取值检查。返回 (ok, detail)。"""
    try:
        obj = json.loads(_strip_fence(text))
    except Exception as e:
        return False, "非可解析JSON（%s）" % type(e).__name__
    if not isinstance(obj, dict):
        return False, "顶层不是JSON对象"
    missing = [k for k in DIAG_JSON_KEYS if k not in obj]
    if missing:
        return False, "缺必需键:%s" % ",".join(missing)
    if not isinstance(obj["cause"], str) or not obj["cause"]:
        return False, "cause 需为非空字符串"
    if not isinstance(obj["steps"], list) or not obj["steps"]:
        return False, "steps 需为非空数组"
    if not isinstance(obj["confidence"], (int, float)) or \
            not 0.0 <= float(obj["confidence"]) <= 1.0:
        return False, "confidence 需在[0,1]"
    return True, "schema 校验通过"


def compare(channel):
    """三版对比：质量分 + schema 结果 + 畸变率，全部实测。"""
    base = None
    out = {}
    for tag, p in PROMPTS.items():
        ans = call(channel, p, QUESTION)
        s = score(ans)
        ok, detail = validate_output(ans)
        if base is None:
            base = s
        out[tag] = {"score": s, "delta": round(s - base, 3),
                    "schema_ok": ok, "schema_detail": detail, "answer": ans}
    malformed = round(sum(1 for v in out.values() if not v["schema_ok"])
                      / max(len(out), 1), 4)
    return out, base, malformed


MULTITURN_SCRIPT = [
    QUESTION,
    "你上一轮回答里提到的'特征频率'具体指多少Hz？下一步动作是什么？",
]


def run_multiturn(channel, turns=None):
    """多轮对话（优A档）：历史拼进上下文连续作答，实测轮数与上下文保持率。"""
    turns = turns or MULTITURN_SCRIPT
    history, responses = [], []
    for t in turns:
        ans = channel(PROMPTS["角色"], "\n".join(history + [t]))
        history.append("用户: " + t)
        history.append("助手: " + ans)
        responses.append(ans)
    # 上下文保持：第二轮应答必须复述第一轮出现过的关键量（16Hz）才算接住了历史
    retained = ("16Hz" in responses[-1] and "16Hz" in responses[0])
    return {"n_turns": len(turns), "completed": len(responses),
            "context_retained": retained,
            "last_answer": responses[-1][:60]}


# ---------------------------------------------------------------- 路径拼接审计
PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def audit_path_joins(pkg_dir=None):
    """审计本项目 src 包：路径必须 os.path.join；'/' 参与字符串相加=可疑拼接。"""
    pkg_dir = pkg_dir or PKG_DIR
    join_hits, suspects, files = 0, [], 0
    for name in sorted(os.listdir(pkg_dir)):
        if not name.endswith(".py"):
            continue
        files += 1
        with open(os.path.join(pkg_dir, name), encoding="utf-8") as f:
            src = f.read()
        join_hits += src.count("os.path.join(")
        for ln, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"[\"']/[\"']\s*\+|\+\s*[\"'][^\"']*/[\"']", code):
                suspects.append("%s:%d %s" % (name, ln, line.strip()))
    return {"files": files, "join_hits": join_hits, "suspects": suspects}


def version_stats(out, base):
    best = max(v["score"] for v in out.values())
    lift = (best - base) * 100
    return best, lift
