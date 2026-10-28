# -*- coding: utf-8 -*-
"""命令行入口：`m11b_feature_design` 安装后的 console_script 真实落点。

pyproject.toml 的 [project.scripts] 指向本模块（`feature_design48.cli:main`），
仓库内的 `run.py` 也委托到这里——两条入口共用同一套 config/pipeline，
杜绝"两套入口两个口径"。

用法（两种等价）:
  pip install -e . && m11b_feature_design                 # 安装态
  python run.py                              # 免安装态（克隆即跑）
  m11b_feature_design --config config/rules.yaml --outdir output --quiet
"""
import argparse
import json
import os
import sys

from . import config as C
from . import pipeline as P

# src/feature_design48/cli.py → src/feature_design48 → src → 项目根
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "rules.yaml")
DEFAULT_OUTDIR = os.path.join(ROOT, "output")


def _default_config():
    """安装态下包可能被装到 site-packages：按候选顺序找配置文件。"""
    for p in (DEFAULT_CONFIG, os.path.join(os.getcwd(), "config", "rules.yaml")):
        if os.path.exists(p):
            return p
    return DEFAULT_CONFIG


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m11b_feature_design", description="特征设计与优化（工单 M1-1b）")
    ap.add_argument("--config", default=_default_config(), help="规则配置文件（YAML）")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="结果输出目录")
    ap.add_argument("--quiet", action="store_true", help="只落盘，不打印报告")
    ap.add_argument("--tier", action="store_true",
                    help="打印工单 D/C/B/A 档位实测判定 + 代码规范率（非静态表）")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    sections = P.run(cfg, out_dir=a.outdir)
    if a.tier:
        print(render(tier_sections(a.outdir)))
        return 0
    if not a.quiet:
        print(render(sections))
    return 0


def tier_sections(outdir):
    """从落盘的 result.json 取**实测**档位判定，渲染成 --tier 输出段。

    档位与规范率都由流水线真跑一遍产生——这里只读不算，杜绝 CLI 与流水线两套口径。
    """
    from . import tiers as T
    with open(os.path.join(outdir, "result.json"), encoding="utf-8") as f:
        r = json.load(f)
    st = r["code_style"]
    tier_detail = r["tiers"]
    head = T.render_lines(r["tier_verdict"])
    body = []
    for k in T.KEYS:
        t = tier_detail[k]
        line = "%s 层 [%s] %s ｜ %s" % (
            k, "达标" if t["达标"] else "未达标", t["要求"], t["实测"])
        if t.get("实测2"):
            line += " ｜ " + t["实测2"]
        body.append(line)
    style_lines = [
        "检查引擎：%s｜口径：%s" % (st["engine"], st["definition"]),
        "%d 个 .py 文件 / %d 行非空代码 / %d 条违规 → 规范率 %.2f%% → %s" % (
            st["n_files"], st["code_lines"], st["violations"], st["rate"],
            "达标" if st["passes"] else "未达标"),
    ]
    return [("档位实测判定", head),
            ("档位逐条证据", body),
            ("代码规范率（D 档次条：≥%.0f%%）" % r["min_code_style"], style_lines)]


def render(sections):
    """把报告分节渲染成文本（与 report.py 同口径，避免两份实现漂移）。"""
    from . import report
    return report.render([(t, l) for t, l in sections])


if __name__ == "__main__":
    sys.exit(main())
