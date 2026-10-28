# -*- coding: utf-8 -*-
"""命令行入口：`m21a_classic_clf` 安装后的 console_script 真实落点。

pyproject.toml 的 [project.scripts] 指向本模块（`classic_clf.cli:main`），
仓库内的 `run.py` 也委托到这里——两条入口共用同一套 config/pipeline，
杜绝"两套入口两个口径"。

用法（两种等价）:
  pip install -e . && m21a_classic_clf                 # 安装态
  python run.py                              # 免安装态（克隆即跑）
  m21a_classic_clf --config config/rules.yaml --outdir output --quiet
  m21a_classic_clf --tier                    # 工单 D/C/B/A 档实测判定
"""
import argparse
import json
import os
import sys

from . import config as C
from . import pipeline as P
from . import report as R
from . import tiers as T

# src/classic_clf/cli.py → src/classic_clf → src → 项目根
ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "rules.yaml")
DEFAULT_OUTDIR = os.path.join(ROOT, "output")


def _default_config():
    """安装态下包可能被装到 site-packages：按候选顺序找配置文件。"""
    for p in (DEFAULT_CONFIG,
              os.path.join(os.getcwd(), "config", "rules.yaml")):
        if os.path.exists(p):
            return p
    return DEFAULT_CONFIG


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="m21a_classic_clf",
        description="传统分类：逻辑回归 + 决策树（工单 M2-1a）")
    ap.add_argument("--config", default=_default_config(),
                    help="规则配置文件（YAML）")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="结果输出目录")
    ap.add_argument("--quiet", action="store_true", help="只落盘，不打印报告")
    ap.add_argument("--tier", action="store_true",
                    help="打印工单 D/C/B/A 档位实测判定（流水线真跑一遍）")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    sections = P.run(cfg, out_dir=a.outdir)
    if a.tier:
        print(R.render(tier_sections(a.outdir)))
        return 0
    if not a.quiet:
        print(R.render(sections))
    return 0


def tier_sections(outdir):
    """从落盘的 result.json 取**实测**档位判定，渲染成 --tier 输出段。

    档位由流水线真跑一遍产生——这里只读不算，杜绝 CLI 与流水线两套口径。
    """
    with open(os.path.join(outdir, "result.json"), encoding="utf-8") as f:
        r = json.load(f)
    v = r["tier_verdict"]
    st = r["code_style"]
    head = ["工单 M2-1a 验收档位（实测）｜承诺档位: %s" % v["claimed"]]
    body = T.render_lines(v)
    style_lines = ["规范率实测 %.2f%%（%d 文件/%d 行/%d 违规，引擎 %s）→ %s" % (
        st["rate"], st["n_files"], st["code_lines"], st["violations"],
        st["engine"], "达标" if st["passes"] else "未达标")]
    manual = ["  人工答辩项: " + m for m in T.MANUAL]
    return [("档位实测判定", head + body + manual),
            ("代码规范率（D 档次条）", style_lines)]


if __name__ == "__main__":
    sys.exit(main())
