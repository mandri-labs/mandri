import asyncio
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.hub import Hub
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.usage import UsageFilters, UsageObservation
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage
from mandri.runtime.service import RuntimeService
from mandri.runtime.usage import NativeUsageCollector, observe_usage
from mandri.runtime.usage_accounts import NativeAccountReader
from mandri.runtime.usage_profiles import usage_profile_id
from mandri.sessions.usage.types import NativeUsageContext


@pytest.fixture
async def repository(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "synthetic.sqlite")
    await db.migrate()
    await migrate_usage(db._require_connection())
    yield UsageRepository(db)
    await db.close()


def context(harness="agy", **fields):
    return NativeUsageContext(
        "s",
        harness,
        "process",
        native_id="native",
        routing="native",
        inherited_history="none",
        process_started_at=1000,
        **fields,
    )


def agy_event(value=100):
    return {
        "event": "result",
        "result": {
            "conversation_id": "native",
            "usage": {
                "input_tokens": value,
                "output_tokens": 10,
                "cache_read_tokens": 2,
                "thinking_tokens": 0,
                "total_tokens": value + 10,
            },
        },
    }


async def test_fresh_native_first_result_is_dated_and_replay_does_not_add(repository):
    collector = NativeUsageCollector(repository.record, clock=lambda: 2000)
    await collector(context(), agy_event())
    summary = (await repository.overview(UsageFilters(from_ms=999, to_ms=3000)))["summary"]
    assert summary["input_tokens"] == 100
    await collector(context(), agy_event())
    assert (await repository.overview())["summary"]["input_tokens"] == 100
    await collector(context(), agy_event(150))
    assert (await repository.overview())["summary"]["input_tokens"] == 150


async def test_resumed_agy_baseline_avoids_lifetime_duplication(repository):
    collector = NativeUsageCollector(repository.record, clock=lambda: 2000)
    await collector(context(), agy_event())
    resumed = replace(context(), process_epoch="resumed", resumed=True, process_started_at=3000)
    next_collector = NativeUsageCollector(repository.record, clock=lambda: 4000)
    await next_collector(resumed, agy_event(100))
    await next_collector(resumed, agy_event(125))
    assert (await repository.overview())["summary"]["input_tokens"] == 125


async def test_decreasing_crash_total_preserves_valid_counters(repository):
    collector = NativeUsageCollector(repository.record, clock=lambda: 2000)
    await collector(context(), agy_event())
    assert not await observe_usage(collector, context(), agy_event(0))
    assert (await repository.overview())["summary"]["input_tokens"] == 100


async def test_claude_fallback_series_does_not_add_later_model_aggregate(repository):
    collector = NativeUsageCollector(repository.record, clock=lambda: 2000)
    await collector(
        context("claude"),
        {
            "type": "result",
            "uuid": "turn1",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
            },
        },
    )
    await collector(
        context("claude"),
        {
            "type": "result",
            "uuid": "turn2",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 4,
            },
            "modelUsage": {"model": {"inputTokens": 25, "outputTokens": 7}},
        },
    )
    assert (await repository.overview())["summary"]["input_tokens"] == 25


async def test_claude_model_series_does_not_add_mainloop_or_children(repository):
    collector = NativeUsageCollector(repository.record, clock=lambda: 2000)
    event = {
        "type": "result",
        "uuid": "turn1",
        "usage": {"input_tokens": 8},
        "modelUsage": {"model": {"inputTokens": 10, "outputTokens": 3}},
    }
    await collector(context("claude"), event)
    await collector(replace(context("claude"), session_id="child", parent_session_id="s"), event)
    assert (await repository.overview())["summary"]["input_tokens"] == 10


async def test_feed_collects_without_browser_and_preserves_ui_when_sink_fails(monkeypatch):
    hub = Hub()
    published = Mock(wraps=hub.publish)
    monkeypatch.setattr(hub, "publish", published)
    runtime = RuntimeService({}, hub=hub)
    state = runtime._session_state("s")
    state.launched_model = (ModelSource.NATIVE, "default", None)
    state.native_id = "native"
    callback = AsyncMock(side_effect=RuntimeError("sensitive exception text"))
    runtime.set_usage_observer(callback)
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    stdout.feed_data((json.dumps(agy_event()) + "\n").encode())
    stdout.feed_eof()
    stderr.feed_eof()
    process = SimpleNamespace(process=SimpleNamespace(stdout=stdout, stderr=stderr))
    monkeypatch.setattr(runtime, "_attach_approvals", lambda *args: None)
    monkeypatch.setattr(runtime, "_watch_feed_eof", lambda *args: None)
    runtime._attach_feed("s", "agy", process, inherited_history="none")
    await asyncio.gather(*state.feed.tasks)
    callback.assert_awaited_once()
    captured = callback.call_args.args[0]
    assert captured.routing == "native" and captured.process_started_at is not None
    assert captured.native_id == "native"
    frames = [call.args[1] for call in published.call_args_list]
    assert any(frame.get("raw") == agy_event() for frame in frames)
    assert not any(
        frame.get("raw", {}).get("error") == "usage_collection_failed" for frame in frames
    )
    await state.feed.stop()


async def test_account_reads_coalesce_and_unsupported_never_invokes_command(tmp_path):
    read = AsyncMock(return_value={"rateLimits": {"primary": {"usedPercent": 50}}})
    reader = NativeAccountReader("profile", "codex", clock=lambda: 1000)
    first, second = await asyncio.gather(
        reader.read_codex(read, qualified=True),
        reader.read_codex(read, qualified=True),
    )
    assert first == second and first.status == "available"
    read.assert_awaited_once()
    agy = NativeAccountReader("profile", "agy")
    result = await agy.read_agy(
        tmp_path / "not-executed", cwd=tmp_path, env={}, qualified_version=None
    )
    assert result.status == "unsupported"


def test_profile_identity_shared_for_shared_launch_home_and_isolated_for_docker():
    first = usage_profile_id("codex", {"CODEX_HOME": "/synthetic/profile"})
    assert first == usage_profile_id("codex", {"CODEX_HOME": "/synthetic/profile"})
    assert first != usage_profile_id("codex", {"CODEX_HOME": "/synthetic/other"})
    assert first != usage_profile_id("codex", {}, isolated_scope="isolated")
    assert "/synthetic" not in first


async def test_account_push_uses_profile_and_keeps_account_identity_unverified():
    sink, accounts = AsyncMock(), AsyncMock()
    collector = NativeUsageCollector(sink, account_sink=accounts, clock=lambda: 1000)
    await collector(
        context("codex", profile_id="shared-profile"),
        {
            "method": "account/rateLimits/updated",
            "params": {
                "rateLimits": {"primary": {"usedPercent": 0}},
            },
        },
    )
    value = accounts.call_args.args[0]
    assert value.account_id == "shared-profile" and not value.verified
    sink.assert_not_called()


async def test_account_read_failure_does_not_replace_successful_push():
    accounts = AsyncMock()
    collector = NativeUsageCollector(AsyncMock(), account_sink=accounts)
    scope = context("codex", profile_id="shared")
    await collector(
        scope,
        {
            "method": "account/rateLimits/updated",
            "params": {
                "rateLimits": {"primary": {"usedPercent": 25}},
            },
        },
    )
    await collector(scope, {"method": "account/rateLimits/unavailable"})
    await collector(scope, {"method": "account/rateLimits/updated", "params": {}})
    accounts.assert_awaited_once()
    assert accounts.call_args.args[0].status == "available"


async def test_codex_fresh_request_and_resume_excludes_lifetime_snapshot(repository):
    event = codex_event(100, 100)
    collector = NativeUsageCollector(repository.record, clock=lambda: 2000)
    scope = context("codex", observed_model="model-a")
    await collector(scope, event)
    assert (await repository.overview(UsageFilters(from_ms=999, to_ms=3000)))["summary"][
        "input_tokens"
    ] == 100
    resumed = replace(scope, resumed=True, process_epoch="resumed", process_started_at=3000)
    next_collector = NativeUsageCollector(repository.record, clock=lambda: 4000)
    await next_collector(resumed, codex_event(140, 40))
    assert (await repository.overview())["summary"]["input_tokens"] == 100
    await next_collector(resumed, codex_event(160, 20))
    assert (await repository.overview())["summary"]["input_tokens"] == 120


async def test_codex_resume_excludes_gateway_overlap_then_collects_new_native_interval(repository):
    scope = context("codex", observed_model="model-a")
    await NativeUsageCollector(repository.record, clock=lambda: 2000)(scope, codex_event(100, 100))
    await repository.record(
        UsageObservation(
            source="gateway",
            source_key="gateway-request",
            fact_key="gateway-request",
            session_id="s",
            occurred_at=3000,
            observed_at=3000,
            input_tokens=30,
            output_tokens=5,
        )
    )
    resumed = replace(scope, resumed=True, process_epoch="resumed", process_started_at=3500)
    clock = [4000]
    collector = NativeUsageCollector(repository.record, clock=lambda: clock[0])
    await collector(resumed, codex_event(180, 80))
    assert (await repository.overview())["summary"]["input_tokens"] == 130
    clock[0] = 5000
    await collector(resumed, codex_event(200, 20))
    assert (await repository.overview())["summary"]["input_tokens"] == 150


def codex_event(total, last, turn="turn"):
    def counts(value):
        return {
            "inputTokens": value,
            "cachedInputTokens": value // 2,
            "outputTokens": value // 10,
            "reasoningOutputTokens": value // 20,
            "totalTokens": value + value // 10,
        }

    return {
        "method": "thread/tokenUsage/updated",
        "params": {
            "threadId": "native",
            "turnId": turn,
            "tokenUsage": {"total": counts(total), "last": counts(last)},
        },
    }


def turn_started(model, turn="turn"):
    return {
        "method": "turn/started",
        "params": {"threadId": "native", "turn": {"id": turn, "model": model}},
    }


async def test_codex_live_models_follow_request_not_lifetime_and_ignore_repeats():
    sink = AsyncMock()
    collector = NativeUsageCollector(sink, clock=lambda: 2000)
    scope = context("codex", observed_model="launched-model")
    await collector(scope, turn_started("model-a"))
    await collector(scope, codex_event(100, 100))
    await collector(scope, codex_event(100, 100))
    await collector(scope, codex_event(200, 100))
    await collector(scope, turn_started("model-b", "second"))
    await collector(scope, codex_event(240, 40, "second"))
    values = [call.args[0] for call in sink.call_args_list]
    assert [value.model for value in values] == ["model-a", "model-a", "model-b"]
    assert [value.input_tokens for value in values] == [100, 100, 40]
    assert all(value.request_count == 1 and value.kind == "delta" for value in values)
    assert all(value.cache_write_tokens == 0 and value.complete for value in values)
    assert len({value.source_key for value in values}) == 3
    assert all(value.pricing_context.get("evidence") != "history_request" for value in values)


async def test_codex_live_fallback_never_reuses_previous_turn_model():
    sink = AsyncMock()
    collector = NativeUsageCollector(sink)
    scope = context("codex", observed_model="launched-model")
    await collector(scope, turn_started("old-model"))
    await collector(scope, codex_event(100, 100))
    await collector(scope, turn_started(None, "next"))
    await collector(scope, codex_event(140, 40, "next"))
    await collector(replace(scope, observed_model="new-launch"), codex_event(160, 20, "unseen"))
    assert [call.args[0].model for call in sink.call_args_list] == [
        "old-model",
        "launched-model",
        "new-launch",
    ]


async def test_codex_live_resumed_active_turn_uses_last_and_reset_is_not_lifetime():
    sink = AsyncMock()
    collector = NativeUsageCollector(sink)
    scope = context("codex", observed_model="launched-model", resumed=True)
    await collector(scope, turn_started("model-a"))
    await collector(scope, codex_event(1000, 40))
    await collector(scope, codex_event(1020, 20))
    await collector(scope, codex_event(0, 20))
    await collector(scope, codex_event(20, 20))
    await collector(scope, codex_event(20, 20))
    values = [call.args[0] for call in sink.call_args_list]
    assert [value.input_tokens for value in values] == [40, 20, 20]
    assert len({value.source_key for value in values}) == 3


async def test_codex_live_failed_sink_does_not_advance_request_baseline():
    sink = AsyncMock(side_effect=[RuntimeError("unavailable"), None, None])
    collector = NativeUsageCollector(sink)
    scope = context("codex", observed_model="model-a")
    event = codex_event(100, 100)
    assert not await observe_usage(collector, scope, event)
    assert await observe_usage(collector, scope, event)
    await collector(scope, codex_event(120, 20))
    assert [call.args[0].input_tokens for call in sink.call_args_list] == [100, 100, 20]
    assert sink.call_args_list[0].args[0].source_key == sink.call_args_list[1].args[0].source_key


async def test_usage_waits_for_database_contention_without_cancelling_collection(
    repository, tmp_path, caplog
):
    collector = NativeUsageCollector(repository.record, clock=lambda: 2000)
    scope = context("codex", observed_model="model-a")
    writer = sqlite3.connect(tmp_path / "synthetic.sqlite")
    writer.execute("BEGIN IMMEDIATE")
    task = asyncio.create_task(observe_usage(collector, scope, codex_event(100, 100)))
    try:
        await asyncio.sleep(0.65)
        assert not task.done()
    finally:
        writer.rollback()
        writer.close()
        result = await task
    assert result
    assert "collection failed" not in caplog.text
    await collector(scope, codex_event(120, 20))
    summary = (await repository.overview())["summary"]
    assert summary["input_tokens"] == 120
    assert summary["request_count"] == 2


async def test_usage_failure_identifies_counters_in_logs_without_private_output(caplog):
    collector = NativeUsageCollector(AsyncMock(side_effect=RuntimeError("private output")))
    assert not await observe_usage(
        collector, context("codex", observed_model="model-a"), codex_event(100, 100)
    )
    assert "harness=codex session=s" in caplog.text
    assert "fields=input_tokens,output_tokens,cache_read_tokens" in caplog.text
    assert "error=RuntimeError" in caplog.text
    assert "private output" not in caplog.text


async def test_codex_live_without_model_is_provisional_and_other_threads_do_not_advance():
    sink = AsyncMock()
    collector = NativeUsageCollector(sink)
    scope = context("codex")
    foreign = codex_event(100, 100)
    foreign["params"]["threadId"] = "different"
    await collector(scope, foreign)
    sink.assert_not_called()
    await collector(scope, codex_event(120, 120))
    assert not sink.call_args.args[0].authoritative
    assert sink.call_args.args[0].model is None


async def test_codex_live_model_switch_does_not_assign_aggregate_to_latest_model():
    sink = AsyncMock()
    collector = NativeUsageCollector(sink)
    scope = context("codex", observed_model="model-a")
    await collector(scope, codex_event(100, 100))
    await collector(replace(scope, observed_model="model-b"), codex_event(200, 40, "next"))
    assert [call.args[0].input_tokens for call in sink.call_args_list] == [100, 40]
    assert sink.call_args.args[0].request_count == 1
