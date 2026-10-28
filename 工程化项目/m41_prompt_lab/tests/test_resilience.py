# -*- coding: utf-8 -*-
"""容错/schema/多轮/审计 测试——删除对应能力（os.environ 读取、重试降级、JSON校验）时必失败。"""
import inspect

from prompt_lab import core, pipeline


def test_endpoint_comes_from_os_environ_not_hardcoded():
    src = inspect.getsource(core)
    assert "os.getenv" in src or "os.environ" in src          # 端点必须走环境变量
    assert "sk-" not in src                                    # 代码中零硬编码密钥
    assert core.endpoint_from_env.__doc__ is not None
    assert len(core.endpoint_from_env()) == 3                   # (base, key, model) 三元组


def test_endpoint_from_env_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-only-in-memory")
    base, key, model = core.endpoint_from_env()
    assert base == "http://127.0.0.1:9999/v1" and key == "test-key-only-in-memory"


def test_http_channel_refuses_missing_key():
    raised = False
    try:
        core.HttpChannel("http://x/v1", "", "m")
    except ValueError:
        raised = True
    assert raised


def test_validate_output_cases():
    ok, _ = core.validate_output('{"cause":"c","steps":["a"],"confidence":0.5}')
    assert ok
    assert not core.validate_output("建议：检查电源。")[0]           # 散文=畸变
    assert not core.validate_output('{"cause":"c"}')[0]              # 缺键
    assert not core.validate_output('{"cause":"","steps":["a"],"confidence":0.5}')[0]
    assert not core.validate_output('{"cause":"c","steps":[],"confidence":0.5}')[0]
    assert not core.validate_output('{"cause":"c","steps":["a"],"confidence":1.7}')[0]
    ok2, _ = core.validate_output('```json\n{"cause":"c","steps":["a"],"confidence":0.9}\n```')
    assert ok2                                                        # 代码围栏可剥


def test_mock_channel_schema_split():
    out, base, malformed = core.compare(core.MockChannel())
    assert out["朴素"]["schema_ok"] is False
    assert out["角色"]["schema_ok"] and out["示例"]["schema_ok"]
    assert malformed == round(1 / 3, 4)


def test_retry_recovers_after_two_failures():
    flaky = core.FlakyChannel(2, fallback=core.MockChannel())
    ch = core.RetryFallbackChannel(flaky, retries=3)
    ans = ch(core.PROMPTS["角色"], core.QUESTION)
    assert ch.stats["attempts"] == 3 and ch.stats["retries"] == 2
    assert ch.stats["recovered"] == 1 and ch.stats["fallbacks"] == 0
    assert len(ch.stats["errors"]) == 2
    assert "轴承" in ans


def test_total_outage_degrades_without_crash():
    ch = core.RetryFallbackChannel(core.FlakyChannel(10 ** 6), retries=3)
    ans = ch(core.PROMPTS["角色"], core.QUESTION)      # 关键：不抛异常
    assert ans.startswith(core.FALLBACK_TEXT)
    assert ch.stats["attempts"] == 3 and ch.stats["fallbacks"] == 1
    assert "轴承" in ans                               # 兜底话术后仍给出可用建议


def test_multiturn_keeps_context():
    mt = core.run_multiturn(core.MockChannel())
    assert mt["completed"] == mt["n_turns"] == 2
    assert mt["context_retained"] is True


def test_audit_path_joins_clean():
    audit = core.audit_path_joins()
    assert audit["files"] >= 6 and audit["join_hits"] > 0
    assert audit["suspects"] == []


def test_pipeline_report_new_sections(tmp_path):
    from prompt_lab.config import Settings
    pipeline.run(Settings(seed=7), str(tmp_path))
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    for key in ("JSON schema 校验", "重试与降级", "降级话术", "多轮对话",
                "路径拼接审计", "验收档位判定", "编码规范率"):
        assert key in text
    import json
    data = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert data["malformed_rate"] > 0
    assert data["retry_recover_stats"]["retries"] == 2
