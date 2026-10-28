# -*- coding: utf-8 -*-
"""代码规范率自检（工单 M2-3a D 档：代码规范率 ≥ 70%）。

为什么要自带一个检查器：工单要一个**可复算的数字**，而不是"我感觉挺规范"。

两条路线（本课主线原则：环境也是选型维度）：
  真实路径 —— 环境里有 `pycodestyle`（PEP8 官方参考实现）就用它统计版式违规；
  兜底路径 —— 没装就退回本模块的极简检查（行宽 / 制表符），照样出数。
版式之外的**结构规范**（模块与函数 docstring、snake_case 命名、制表符、
`import *`）pycodestyle 并不管，所以本模块**无论有没有 pycodestyle 都另算一份**，
两类都以"条"为单位一起计入规范率（同单位才敢相加），再按类别分开报明细——
不把"没检查"说成"没违规"。
不允许的做法：把规范率写成常量糊弄验收。
"""
import ast
import os
import re

MAX_LINE = 79
_SNAKE = re.compile(r"^[a-z_][a-z0-9_]*$")
_IMPORT_STAR = re.compile(r"^\s*(from\s+\S+\s+)?import\s+\*")

#: 报告里只写规则码学生看不懂，故给每条配一句人话
PEP8_READABLE = {
    "E501": "行宽超 %d 字符" % MAX_LINE,
    "E127": "续行多缩进（视觉对齐歧义）",
    "E128": "续行少缩进（视觉对齐歧义）",
    "E126": "续行悬挂缩进对齐歧义",
    "E131": "注释续行对齐歧义",
    "E306": "嵌套定义前缺空行",
    "E302": "顶层定义前缺两行空行",
    "E303": "空行过多",
    "E305": "函数后缺两行空行",
    "E226": "缺算术运算符空格",
    "E231": "缺逗号/冒号后空格",
    "E203": "冒号前多余空格",
    "W191": "缩进含制表符",
    "W291": "行尾多余空格",
    "W293": "空行含空格",
    "W391": "文件末尾多余空行",
}


def py_files(root):
    """项目内全部 .py（跳过缓存与产物目录）。"""
    skip = {"__pycache__", ".git", ".venv", "output", "build", "dist",
            ".pytest_cache"}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def code_lines(src):
    """非空代码行数（规范率分母口径，写进 definition 便于别人复算）。"""
    return max(len([x for x in src.splitlines() if x.strip()]), 1)


def _first_is_doc(body):
    return bool(body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str))


def structural_problems(src):
    """pycodestyle 不管的四类硬规范，逐条数出来（返回 {类别: 条数}）。"""
    out = {"缺docstring": 0, "命名非snake_case": 0, "含制表符": 0,
           "import星号": 0}
    for line in src.splitlines():
        if "\t" in line:
            out["含制表符"] += 1
        if _IMPORT_STAR.search(line):
            out["import星号"] += 1
    try:
        tree = ast.parse(src)
    except SyntaxError:
        out["缺docstring"] += 1
        return out
    if not _first_is_doc(tree.body):
        out["缺docstring"] += 1                      # 模块缺 docstring
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not _first_is_doc(node.body):
                out["缺docstring"] += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not _is_named_ok(node.name):
                out["命名非snake_case"] += 1          # 驼峰/大写函数名
    return out


def _is_named_ok(name):
    """PEP8 要求函数小写；pytest 惯例的 `test_D_xxx`（档位大写字母）不算违规。"""
    return _SNAKE.match(name) or name.startswith("test_")


def _layout_pycodestyle(path):
    """真实路径：返回 (版式违规数, {规则码: (条数, 人话说明)})。

    用 StyleGuide 而不是直接 new Checker——Checker 的构造签名在不同
    pycodestyle 版本里不一致，StyleGuide 才是官方稳定入口。
    """
    import pycodestyle
    report = pycodestyle.StyleGuide(quiet=True).check_files([path])
    counters = getattr(report, "counters", {}) or {}
    cnt = {}
    for code, n in counters.items():
        if n and code[:1] in ("E", "W", "C"):
            cnt[code] = cnt.get(code, 0) + int(n)
    return sum(cnt.values()), {k: (c, PEP8_READABLE.get(k, k))
                               for k, c in cnt.items()}


def _layout_fallback(path, src):
    """兜底路径：环境没有 pycodestyle 时也能出数（行宽 + 制表符）。"""
    over = sum(1 for line in src.splitlines() if len(line) > MAX_LINE)
    tabs = sum(1 for line in src.splitlines() if "\t" in line)
    by = {}
    if over:
        by["E501"] = (over, PEP8_READABLE["E501"] + "（内置兜底）")
    if tabs:
        by["W191"] = (tabs, PEP8_READABLE["W191"] + "（内置兜底）")
    return over + tabs, by


def _pick_engine():
    """环境探测：有 pycodestyle 走权威实现，没有就走兜底——两条路都能出数。"""
    try:
        import pycodestyle  # noqa: F401
        return "pycodestyle（PEP8 官方参考实现）", True
    except ImportError:
        return "内置兜底（行宽/制表符）——本机未安装 pycodestyle", False


def code_style_rate(root, min_pass=70.0):
    """统计 root 下全部 .py 的代码规范率（工单 M2-3a D 档次条）。

    口径（写进报告，便于别人复算）：
        规范率 = 100 × (1 − (版式违规条数 + 结构违规条数) / 非空代码行数)
    版式项交给 pycodestyle，结构项（docstring/命名/制表符/import*）本模块自算——
    两类都是"条"，同单位相加才不会把"没检查"说成"没违规"。
    达标 = 规范率 ≥ min_pass（工单只写这一条线，不加自创的"结构项须全 0"）。
    """
    files = py_files(root)
    engine, has_pycodestyle = _pick_engine()
    tot_layout = tot_struct = tot_l = 0
    per = []
    by_rule = {}
    struct = {"缺docstring": 0, "命名非snake_case": 0, "含制表符": 0,
              "import星号": 0}
    for path in files:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        v, by = (_layout_pycodestyle(path) if has_pycodestyle
                 else _layout_fallback(path, src))
        struct_here = structural_problems(src)
        sv = sum(struct_here.values())
        n_lines = code_lines(src)
        tot_layout += v
        tot_struct += sv
        tot_l += n_lines
        for code, (c, name) in by.items():
            got = by_rule.get(code)
            by_rule[code] = (got[0] + c, name) if got else (c, name)
        for key, c in struct_here.items():
            struct[key] = struct.get(key, 0) + c
        per.append({"file": os.path.relpath(path, root), "code_lines": n_lines,
                    "layout_violations": v, "structural_violations": sv,
                    "file_rate": round(100.0 * (1.0 - (v + sv) / n_lines), 2)})
    rate = round(100.0 * (1.0 - (tot_layout + tot_struct) / max(tot_l, 1)), 2)
    struct_clean = all(c == 0 for c in struct.values())
    return {"engine": engine, "engine_real": bool(has_pycodestyle),
            "n_files": len(files), "violations": tot_layout + tot_struct,
            "layout_violations": tot_layout,
            "structural_violations": tot_struct,
            "code_lines": tot_l, "rate": rate, "min_pass": float(min_pass),
            "by_rule": [{"code": k, "count": c, "meaning": m}
                        for k, (c, m) in sorted(by_rule.items(),
                                                key=lambda kv: -kv[1][0])],
            "structural": struct, "structural_clean": struct_clean,
            "passes": bool(rate >= min_pass),
            "fallback_note": ("真实路径：pycodestyle 已安装并运行"
                             if has_pycodestyle else
                             "兜底路径：未装 pycodestyle，改用内置行宽/制表符检查"),
            "per_file": per,
            "definition": "规范率 = 100 x (1 - (版式违规条数 + 结构违规条数) / "
                          "非空代码行数)；结构项 = 缺docstring/命名非snake_case/"
                          "制表符/import* 四类计数之和"}


def project_root(here):
    """<root>/src/<pkg>/style.py -> <root>"""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(here))))
