# -*- coding: utf-8 -*-
"""三版 Prompt → 打分/schema校验 → 提升幅度；附 环境变量端点、重试降级、多轮对话、路径审计 实测。"""
import os

from . import core, io_utils, report, style, tiers

ROOT = style.project_root(os.path.abspath(__file__))


def run(cfg, out_dir):
    """跑完整实验流水线：三版对比/降级/多轮/规范率/档位判定，返回分节+实测。"""
    os.makedirs(out_dir, exist_ok=True)
    # —— ① 端点解析：只认环境变量，离线即兜底 MockChannel（兜底事实写进报告）——
    base_url, api_key, model = core.endpoint_from_env()
    if base_url and api_key:
        channel = core.HttpChannel(
            base_url, api_key, model,
            timeout=float(cfg.raw.get("llm_timeout", 10)))
        mode = ("真机通道：OPENAI_BASE_URL=%s（来自 os.getenv，模型=%s，"
                "密钥仅驻内存不硬编码）" % (base_url, model))
    else:
        channel = core.MockChannel()
        mode = ("离线兜底：环境变量 OPENAI_BASE_URL/OPENAI_API_KEY 未配置（本机无网络），"
                "按课程规则改用离线规则通道 MockChannel——端点只认 os.getenv，代码零硬编码密钥")

    out, base, malformed = core.compare(channel)
    best, lift = core.version_stats(out, base)

    # —— ② 重试→降级：注入故障通道真实执行代码路径（无网络请求）——
    recov = core.RetryFallbackChannel(
        core.FlakyChannel(2, fallback=core.MockChannel()), retries=3)
    ans_recover = recov(core.PROMPTS["角色"], core.QUESTION)
    dead = core.RetryFallbackChannel(core.FlakyChannel(10 ** 6), retries=3)
    ans_fallback = dead(core.PROMPTS["角色"], core.QUESTION)
    fb_ok, fb_detail = core.validate_output(ans_fallback)

    # —— ③ 多轮对话（优A档能力）——
    mt = core.run_multiturn(core.MockChannel())

    # —— ④ 路径拼接审计（程序化实测）——
    audit = core.audit_path_joins()

    # —— ⑤ 代码规范率（工单 D 档次条：pycodestyle 实测，不是自称）——
    sty = style.code_style_rate(ROOT, min_pass=float(
        tiers.gates(cfg.tiers)["D"].get("style_min", 70.0)))

    # —— ⑥ 决策记录完整性（docs/ADR 反读实测，四节齐全才算）——
    decision = tiers.decision_record_status(ROOT)
    fallback_hit = (ans_fallback.startswith(core.FALLBACK_TEXT)
                    and dead.stats["fallbacks"] == 1)

    tgates = tiers.gates(cfg.tiers)
    tv = tiers.verdict(
        tgates, rate=sty["rate"],
        answers_ok=all(v["answer"] for v in out.values()),
        table_rows=len(out), lift_pct=round(lift, 1),
        fallback_hit=fallback_hit, decision_ok=decision["exists"],
        n_turns=mt["completed"], context_retained=mt["context_retained"])

    result = {"mode": mode, "question": core.QUESTION, "base": base,
              "results": out, "best_score": best, "lift_pct": round(lift, 1),
              "malformed_rate": malformed,
              "retry_recover_stats": recov.stats, "dead_stats": dead.stats,
              "fallback_schema_ok": fb_ok, "multiturn": mt,
              "path_audit": audit,
              "style": {k: sty[k] for k in
                        ("engine", "rate", "violations", "code_lines",
                         "layout_violations", "structural_violations")},
              "decision_record": decision, "tiers": tv}
    io_utils.write_json(os.path.join(out_dir, "result.json"), result)

    lines = ["%-4s 质量分=%.1f（较朴素 %+.0f%%）｜schema=%s → %s"
             % (tag, v["score"], v["delta"] * 100,
                "合格" if v["schema_ok"] else "畸变(" + v["schema_detail"] + ")",
                v["answer"][:30].replace("\n", " "))
             for tag, v in out.items()]

    sections = [
        ("输入口径", ["问题（仅给症状，不含诊断关键词）: %s" % core.QUESTION,
                  "LLM 通道：%s" % mode]),
        ("三版对比", lines),
        ("JSON schema 校验（实测：json.loads + 必需键 cause/steps/confidence + 类型）", [
            "三版输出中畸变 %d/3 → 畸变率 %.1f%%（朴素版散文输出即畸变来源——由校验代码判定）"
            % (round(malformed * 3), malformed * 100),
            "降级话术兜底应答 schema 校验：%s（%s）"
            % ("合格" if fb_ok else "不合格（话术非JSON，属预期兜底）",
               fb_detail)]),
        ("重试与降级（故障注入真实执行——不产生网络请求，断服不崩溃=零异常逃逸）", [
            ("场景A 断服2次后恢复：attempts=%d retries=%d recovered=%d "
             "fallbacks=%d → 应答=%s…"
             % (recov.stats["attempts"], recov.stats["retries"],
                recov.stats["recovered"], recov.stats["fallbacks"],
                ans_recover[:24].replace("\n", " "))),
            ("场景B 完全断服：attempts=%d retries=%d fallbacks=%d，"
             "程序未崩溃，输出降级话术："
             % (dead.stats["attempts"], dead.stats["retries"],
                dead.stats["fallbacks"])),
            "  " + core.FALLBACK_TEXT]),
        ("多轮对话（优A档能力实测）", [
            "轮数 %d/%d｜上下文保持（第二轮复述首轮关键量16Hz）=%s｜末轮摘要：%s"
            % (mt["completed"], mt["n_turns"], mt["context_retained"],
               mt["last_answer"].replace("\n", " "))]),
        ("路径拼接审计（代码级实测）", [
            "扫描 %d 个模块｜os.path.join 调用 %d 处｜'/'字符串相加可疑点 %d 处%s"
            % (audit["files"], audit["join_hits"], len(audit["suspects"]),
               "" if not audit["suspects"]
               else "（违规行：" + "; ".join(audit["suspects"]) + "）")]),
        ("结论", ["最优 %.1f，较朴素 %+.0f%% → %s"
                % (best, lift, "达标（≥+50%）" if lift >= 50 else "未达标")]),
        ("编码规范率（PEP 8 实测）", [
            "引擎: %s｜扫描 %d 个文件｜非空代码行 %d｜违规 %d 条（版式 %d + "
            "结构 %d）→ 规范率 %.1f%%"
            % (sty["engine"], sty["n_files"], sty["code_lines"],
               sty["violations"], sty["layout_violations"],
               sty["structural_violations"], sty["rate"]),
            sty["fallback_note"]]),
        ("验收档位判定（实测）", tiers.render_lines(tv)),
        ("说明", [
            "本演示走离线规则通道；设 OPENAI_BASE_URL 与 OPENAI_API_KEY "
            "环境变量即切真机 HTTP 通道"
            "（urllib+超时，重试耗尽自动降级，不硬编码任何密钥）"]),
        ("结论与建议", [
            "角色给领域先验、示例给输出格式——两者叠加才让 LLM 照单办事；"
            "schema 校验把'格式乱没法接程序'变成可统计的畸变率指标"]),
    ]
    report.save(sections, os.path.join(out_dir, "report.txt"))
    # 返回 (sections, result)：cli --tier 与 pytest 直接吃实测值，不再重跑
    return sections, result
