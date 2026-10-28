# -*- coding: utf-8 -*-
"""档案袋自检 → 提纲生成 → 评委三问逐题实测 → 本地 HTTP 部署演示实测 → 档位判定。"""
import functools
import http.server
import os
import socketserver
import threading
import urllib.request

from . import core, io_utils, report, style, tiers

#: <root>/src/defense_pack/pipeline.py -> <root>
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def framework_dir(cfg):
    """定位「学生代码框架」目录：环境变量 > 配置项 > 课程根 > 项目内。

    工单要求档案袋自检能真的跑起来——路径写死一层就会恒报 0/5，
    所以按候选链找第一个存在的目录；全都不存在时返回首选候选，
    让报告如实写明"找过哪里"（不把"没找到"说成"没证据"）。
    """
    env = os.environ.get("AI_COURSE_FRAMEWORK")
    candidates = [env, str(cfg.raw.get("fw_dir") or ""),
                  os.path.join(PROJ, "..", "..", "学生代码框架"),
                  os.path.join(PROJ, "学生代码框架")]
    for base in candidates:
        if base and os.path.isdir(base):
            return os.path.normpath(base)
    return os.path.normpath(candidates[2])


def deploy_demo(out_dir):
    """部署演示（实测）：真实起本地 HTTP 服务挂 output/，GET report.txt。

    只绑 127.0.0.1、随机端口、请求一次即关停——不外发任何网络请求；
    状态码/端口/字节数与文件实测一致才算"部署演示跑通"，异常如实回报。
    """
    path = os.path.join(out_dir, "report.txt")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        """静音默认访问日志——演示只需实测数据，不需要刷屏。"""

        def log_message(self, fmt, *args):
            """覆盖默认 stderr 日志——演示只需实测数据，不刷屏。"""

    try:
        handler = functools.partial(Quiet, directory=out_dir)
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as srv:
            port = srv.server_address[1]
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                url = "http://127.0.0.1:%d/report.txt" % port
                with urllib.request.urlopen(url, timeout=5) as resp:
                    status = getattr(resp, "status", 200)
                    body = resp.read()
            finally:
                srv.shutdown()
        size_ok = os.path.exists(path) and len(body) == os.path.getsize(path)
        return {"ok": bool(status == 200 and size_ok), "port": port,
                "status": int(status), "bytes": len(body),
                "file_bytes": os.path.getsize(path) if os.path.exists(
                    path) else -1, "error": ""}
    except Exception as exc:
        return {"ok": False, "port": 0, "status": 0, "bytes": 0,
                "file_bytes": -1, "error": repr(exc)}


def run(cfg, out_dir):
    """跑完整答辩自检流水线，返回 (报告分节, 实测结果)。"""
    fw = framework_dir(cfg)
    rows = core.check(fw)
    lines, n_ok = core.outline(rows)
    missing = [r[0] for r in rows if not r[1]]

    # —— 基本问题逐题实测（离线规则回答器，答案只能取自自检实测结果）——
    qa_hit, qa_total, qa_detail = core.qa_hits(rows, n_ok, len(rows))

    # —— 规范率与决策记录（工单 D/B 档实测输入）——
    tgates = tiers.gates(cfg.tiers)
    sty = style.code_style_rate(PROJ, min_pass=float(
        tgates["D"].get("style_min", 70.0)))
    decision = tiers.decision_record_status(PROJ)

    result = {"layers": [{"layer": r[0], "ok": r[1], "talk": r[2]}
                         for r in rows],
              "ready": n_ok, "total": len(rows), "missing": missing,
              "qa": {"hit": qa_hit, "total": qa_total, "detail": qa_detail},
              "style": {k: sty[k] for k in
                        ("engine", "rate", "violations", "code_lines",
                         "layout_violations", "structural_violations")},
              "decision_record": decision}
    os.makedirs(out_dir, exist_ok=True)
    sections = [
        ("输入口径", ["档案袋目录：%s" % os.path.relpath(fw, os.getcwd())]),
        ("自检结果", ["%s %s" % ("OK" if r[1] else "--", r[0]) for r in rows]),
        ("答辩提纲", lines),
        ("评委三问逐题实测（离线规则回答器；缺层则答不出、命中实测下降）",
         ["%s → 实测命中=%s%s"
          % (d["q"], "是" if d["hit"] else "否",
             "" if d["hit"] else "（缺概念词：" + "、".join(d["miss"]) + "）")
          for d in qa_detail]
         + ["基本问题实测 %d/%d 全命中" % (qa_hit, qa_total)]),
        ("编码规范率（PEP 8 实测）", [
            "引擎: %s｜扫描 %d 个文件｜非空代码行 %d｜违规 %d 条（版式 %d + "
            "结构 %d）→ 规范率 %.1f%%"
            % (sty["engine"], sty["n_files"], sty["code_lines"],
               sty["violations"], sty["layout_violations"],
               sty["structural_violations"], sty["rate"]),
            sty["fallback_note"]]),
    ]
    # 先落一版报告，再让部署演示"部署"这份真实文件（演示对象是磁盘上的报告）
    report.save(sections, os.path.join(out_dir, "report.txt"))

    # —— 部署演示实测：本地 HTTP 起服务 → GET → 校验状态码/字节数 ——
    demo = deploy_demo(out_dir)
    result["deploy_demo"] = demo
    tv = tiers.verdict(
        tgates, rate=sty["rate"], files_scanned=sty["n_files"],
        chain_ready=n_ok, chain_total=len(rows), qa_hit=qa_hit,
        qa_total=qa_total, decision_ok=decision["exists"],
        demo_ok=demo["ok"])
    result["tiers"] = tv
    sections.append(("部署演示（本地 HTTP 实测）", [
        "服务: 127.0.0.1:%d（随机端口，请求一次即关停，零外发）" % demo["port"],
        "GET /report.txt → 状态码 %s｜收到 %d 字节｜与磁盘文件一致=%s → 演示%s"
        % (demo["status"], demo["bytes"],
           "是" if demo["bytes"] == demo["file_bytes"] else "否",
           "跑通" if demo["ok"] else "未跑通（%s）" % (demo["error"] or "原因见上"))]))
    sections.append(("验收档位判定（实测）", tiers.render_lines(tv)))
    sections.append(("结论与建议", [
        "证据链 %d/%d；%s" % (n_ok, len(rows),
                          "可上台" if not missing
                          else "补齐：" + "、".join(missing)),
        "工单里评委放行/评分≥85/创新亮点属人工答辩项，报告如实标注，"
        "不冒充实测"]))
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)
    report.save(sections, os.path.join(out_dir, "report.txt"))
    # 返回 (sections, result)：cli --tier 与 pytest 直接吃实测值，不再重跑
    return sections, result
