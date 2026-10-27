# -*- coding: utf-8 -*-
"""《AI应用基础（48课时版）》 环境一键自检（python check_env.py [--deep]）"""
import sys

# ===== 解释器身份与 numpy 版本门禁（69号轮：防 PyCharm 误选旧环境）=====


def _env_guard(min_numpy=2, strict=True):
    """打印当前解释器身份；numpy 低于要求时给出 PyCharm 排障三步并（严格模式）退出"""
    exe = sys.executable.replace("\\", "/")
    env = exe.split("/envs/")[-1].split("/")[0] if "/envs/" in exe else "anaconda-base"
    print(f"[解释器] {env}  ({exe})")
    try:
        import numpy
        major = int(numpy.__version__.split(".")[0])
        if major >= min_numpy:
            print(f"[numpy  ] {numpy.__version__}  OK")
            return True
        print(f"[numpy  ] {numpy.__version__}  ✗ 需要 numpy>={min_numpy}.0")
    except ImportError:
        print(f"[numpy  ] 未安装（需要 numpy>={min_numpy}.0）")
    print("=" * 58)
    print("❗ numpy 版本异常：大概率是 PyCharm 选错了旧解释器环境。")
    print("   排障三步：")
    print("   1) PyCharm 右下角解释器 → Add New Interpreter → Conda Environment")
    print("      → 选择本课程环境（02包=py02 / 03包=pp03 / 05包=ai48）")
    print("   2) 终端执行: conda activate 课程环境名")
    print("   3) 重装依赖: pip install -r requirements.txt")
    print("   详见《软件操作指南/05-PyCharm使用指南.md》'numpy版本异常'排障一节。")
    print("=" * 58)
    if strict:
        sys.exit(1)
    return False


_env_guard(min_numpy=2, strict=True)


DEEP = "--deep" in sys.argv


def imp(mod):
    def f():
        import importlib
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    return f


def font(path):
    def f():
        import os
        assert os.path.exists(path), "缺少字体: " + path
        return "OK"
    return f


def fileexists(path):
    def f():
        import os
        assert os.path.exists(path), "缺少: " + path
        return ""
    return f


def port(host, p):
    def f():
        import socket
        s = socket.create_connection((host, p), timeout=2)
        s.close()
        return f"port {p} ✓"
    return f


ok = warn = fail = 0


def check(name, need, fn):
    global ok, warn, fail
    try:
        extra = fn() or ""
        print(f"[PASS] {name} {extra}")
        ok += 1
    except Exception as e:
        if need:
            print(f"[FAIL] {name} —— {e}")
            fail += 1
        else:
            print(f"[WARN] {name}（按周可选）—— {e}")
            warn += 1


def _py_ver():
    if sys.version_info >= (3, 12):
        return "v%d.%d" % sys.version_info[:2]
    raise RuntimeError(sys.version.split()[0])


check("Python≥3.12", True, _py_ver)
check("numpy", True, imp("numpy"))
check("pandas", True, imp("pandas"))
check("scikit-learn", True, imp("sklearn"))
check("xgboost(W4)", False, imp("xgboost"))
check("torch CPU(W5)", False, imp("torch"))
check("torchvision(W7)", False, imp("torchvision"))
check("onnxruntime(W8)", False, imp("onnxruntime"))
check("requests(W9)", True, imp("requests"))
check("仿真环境(W11零依赖)", True, fileexists("工具/mini_sched_env.py"))
check("思源字体", True, font("../fonts/SourceHanSansCN-Regular.otf"))
if DEEP:
    def _sim():
        import sys as s
        s.path.insert(0, "工具")
        from mini_sched_env import SchedEnv
        e = SchedEnv(seed=1)
        e.reset()
        e.step(0)
        return "reset/step ✓"
    check("仿真自跑", True, _sim)
    check("mock服务(本包工具/):8000", False, port("127.0.0.1", 8000))
print("-" * 46)
print(f"结果: PASS {ok} | WARN {warn} | FAIL {fail}")
print("开课口径: 必需项FAIL=0 即可开课" + ("；--deep已含服务/数据连通" if DEEP else "（--deep 检查服务与数据）"))
# ===== M4系新栈自检（33号轮：mock LLM v2与RAG双解依赖） =====
print("[M4系] LLM服务与向量检索新栈（选做，不装不影响M1-M3）")
for _m in ("litestar", "chromadb"):
    try:
        __import__(_m)
        print(f"[PASS] {_m} 可用（mock_llm_server v2 / RAG双解）")
    except ImportError:
        print(f"[WARN] {_m} 缺失 → pip install {_m}（M4可用旧版http.server方案替代）")

sys.exit(1 if fail else 0)
