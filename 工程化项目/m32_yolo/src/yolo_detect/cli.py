# -*- coding: utf-8 -*-
"""命令行入口：`m32_yolo` 安装后的 console_script 真实落点。

pyproject.toml 的 [project.scripts] 指向本模块（`yolo_detect.cli:main`），
仓库内的 `run.py` 也委托到这里——两条入口共用同一套 config/pipeline，
杜绝"两套入口两个口径"。

用法（等价）:
  pip install -e . && m32_yolo                 # 安装态
  python run.py                                # 免安装态（克隆即跑）
  m32_yolo --config config/rules.yaml --outdir output --quiet
  m32_yolo --tier   # 打印工单 D/C/B/A 门槛与实测判定（跑一次流水线）
  m32_yolo --check  # 环境自检：第三方能力 OK/MISS 如实打印
"""
import argparse
import json
import os
import sys

from . import config as C
from . import pipeline as P
from . import tiers as T

# src/yolo_detect/cli.py → src/yolo_detect → src → 项目根
_HERE3 = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE3))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "rules.yaml")
DEFAULT_OUTDIR = os.path.join(ROOT, "output")


def _default_config():
    """安装态下包可能被装到 site-packages：按候选顺序找配置文件。"""
    candidates = (DEFAULT_CONFIG,
                  os.path.join(os.getcwd(), "config", "rules.yaml"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return DEFAULT_CONFIG


def env_check():
    """环境也是选型维度：ultralytics/onnxruntime 等装没装现场探活。"""
    mods = ["cv2", "onnxruntime", "numpy", "yaml", "pycodestyle"]
    print("[m32_yolo] 环境自检 · Python %s" % sys.version.split()[0])
    for m in mods:
        try:
            mod = __import__(m, fromlist=["_"])
            print("  OK   %-14s %s"
                  % (m, getattr(mod, "__version__", "已安装")))
        except ImportError:
            print("  MISS %-14s 将走确定性兜底（见报告口径段）" % m)
    try:
        import ultralytics  # noqa: F401
        print("  OK   ultralytics    已安装（真实路径可用）")
    except ImportError:
        print("  MISS ultralytics    检测基线走自研滑窗CV（core.detect_all）")
    return 0


def _print_tiers(outdir):
    """从刚跑完的 result.json 里读实测判定并打印——数字来自流水线实测。"""
    rj = os.path.join(outdir, "result.json")
    with open(rj, encoding="utf-8") as f:
        result = json.load(f)
    v = result["tiers"]
    print("工单 M3-2 验收档位（门槛取自 config/rules.yaml tiers: 段）:")
    for (name, _need, key), _ in zip(T.TIERS, T.KEYS):
        print("  %s 门槛 %s" % (name, v["gates"][key]))
    for line in T.render_lines(v):
        print(line)
    return 0


def main(argv=None):
    """命令分发：--check 探活 / --tier 档位实测 / 默认跑完整流水线。"""
    ap = argparse.ArgumentParser(
        prog="m32_yolo", description="YOLO 实时缺陷检测（工单 M3-2）")
    ap.add_argument("--config", default=_default_config(),
                    help="规则配置文件（YAML）")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="结果输出目录")
    ap.add_argument("--quiet", action="store_true", help="只落盘，不打印报告")
    ap.add_argument("--tier", action="store_true",
                    help="打印工单四档门槛与实测判定（会跑一次完整流水线）")
    ap.add_argument("--check", action="store_true", help="环境自检")
    a = ap.parse_args(argv)

    if a.check:
        return env_check()
    cfg = C.load_config(a.config)
    sections = P.run(cfg, out_dir=a.outdir)
    if a.tier:
        return _print_tiers(a.outdir)
    if not a.quiet:
        print(render(sections))
    return 0


def render(sections):
    """把报告分节渲染成文本（与 report.py 同口径，避免两份实现漂移）。"""
    from . import report
    return report.render([(t, l) for t, l in sections])


if __name__ == "__main__":
    sys.exit(main())
