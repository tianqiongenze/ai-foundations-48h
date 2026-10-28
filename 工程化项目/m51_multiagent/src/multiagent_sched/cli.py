# -*- coding: utf-8 -*-
"""命令行入口：`m51_multiagent` 安装后的 console_script 真实落点。

pyproject.toml 的 [project.scripts] 指向本模块（`multiagent_sched.cli:main`），
仓库内的 `run.py` 也委托到这里——两条入口共用同一套 config/pipeline，
杜绝"两套入口两个口径"。

用法（两种等价）:
  pip install -e . && m51_multiagent                 # 安装态
  python run.py                              # 免安装态（克隆即跑）
  m51_multiagent --config config/rules.yaml --outdir output --quiet
"""
import argparse
import os
import sys

from . import config as C
from . import pipeline as P
from . import tiers as T

# src/multiagent_sched/cli.py → src/multiagent_sched → src → 项目根
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "rules.yaml")
DEFAULT_OUTDIR = os.path.join(ROOT, "output")


def _default_config():
    """安装态下包可能被装到 site-packages：按候选顺序找配置文件。"""
    cwd_cfg = os.path.join(os.getcwd(), "config", "rules.yaml")
    for p in (DEFAULT_CONFIG, cwd_cfg):
        if os.path.exists(p):
            return p
    return DEFAULT_CONFIG


def main(argv=None):
    """命令行主入口：默认跑流水线落盘；--tier 打印四档实测判定。"""
    ap = argparse.ArgumentParser(
        prog="m51_multiagent", description="多智能体检修调度（工单 M5-1）")
    ap.add_argument("--config", default=_default_config(), help="规则配置文件（YAML）")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="结果输出目录")
    ap.add_argument("--quiet", action="store_true", help="只落盘，不打印报告")
    ap.add_argument("--tier", action="store_true",
                    help="跑实测并打印工单 D/C/B/A 档位判定（门槛见 rules.yaml）")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    sections, result = P.run(cfg, out_dir=a.outdir)
    if a.tier:
        print("工单 M5-1 验收档位（门槛来自 config/rules.yaml tiers: 段）:")
        for line in T.render_lines(result["tiers"]):
            print("  " + line)
        return 0
    if not a.quiet:
        print(render(sections))
    return 0


def render(sections):
    """把报告分节渲染成文本（与 report.py 同口径，避免两份实现漂移）。"""
    from . import report
    return report.render([(t, l) for t, l in sections])


if __name__ == "__main__":
    sys.exit(main())
