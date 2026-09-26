import json
from dataclasses import asdict, replace
from decimal import Decimal

import pytest
from mandri.sessions.usage.accounts import (
    agy_statusline_snapshot,
    claude_account_snapshot,
    codex_account_snapshot,
)
from mandri.sessions.usage.adapter import to_usage_observation
from mandri.sessions.usage.history import read_usage_batch
from mandri.sessions.usage.normalize import normalize_native_usage
from mandri.sessions.usage.types import NativeUsageContext


def context(harness="codex", **fields):
    return NativeUsageContext(
        "session",
        harness,
        "process",
        native_id="native",
        routing="native",
        inherited_history="none",
        **fields,
    )


def codex_event(input_tokens=100, cached=20):
    return {
        "method": "thread/tokenUsage/updated",
        "params": {
            "threadId": "native",
            "turnId": "turn",
            "tokenUsage": {
                "total": {
                    "inputTokens": input_tokens,
                    "cachedInputTokens": cached,
                    "outputTokens": 10,
                    "reasoningOutputTokens": 3,
                    "totalTokens": 110,
                },
                "last": {"inputTokens": 999999},
                "modelContextWindow": 200000,
            },
        },
    }


def test_codex_normalizes_subsets_and_keeps_allowlisted_evidence():
    event = codex_event()
    event["secret"] = "DO_NOT_RETAIN"
    event["params"]["model"] = "synthetic-model"
    (item,) = normalize_native_usage(context(), event, observed_at_ms=1)
    assert item.counters["uncached_input_tokens"] == 80
    assert item.counters["visible_output_tokens"] == 7
    assert item.model is None
    assert item.observed_model == "synthetic-model"
    value = to_usage_observation(item, sequence=1)
    assert value.authoritative and value.model is None
    assert value.turn_id == "turn" and value.observed_model == "synthetic-model"
    assert "DO_NOT_RETAIN" not in str(asdict(value))
    assert "200000" not in str(asdict(value))


@pytest.mark.parametrize("routing", ["gateway", "unknown"])
def test_native_is_not_authoritative_for_unproven_overlap(routing):
    (item,) = normalize_native_usage(
        replace(context(), routing=routing), codex_event(), observed_at_ms=1
    )
    assert not to_usage_observation(item, sequence=1).authoritative


@pytest.mark.parametrize(
    "fields",
    [
        {"parent_session_id": "parent"},
        {"inherited_history": "unknown"},
        {"native_id": "another"},
    ],
)
def test_fork_children_and_other_native_thread_excluded(fields):
    (item,) = normalize_native_usage(replace(context(), **fields), codex_event(), observed_at_ms=1)
    assert not to_usage_observation(item, sequence=1).authoritative


def test_invalid_counts_and_missing_zero_are_distinct():
    event = codex_event(0, 2)
    (item,) = normalize_native_usage(context(), event, observed_at_ms=1)
    assert item.counters["input_tokens"] == 0
    assert item.counters["cache_write_tokens"] is None
    assert item.counters["uncached_input_tokens"] is None
    assert not to_usage_observation(item, sequence=1).authoritative
    event["params"]["tokenUsage"]["total"]["inputTokens"] = True
    (item,) = normalize_native_usage(context(), event, observed_at_ms=1)
    assert item.counters["input_tokens"] is None


def test_codex_history_matches_live_identity_and_process_resumes_share_epoch():
    (live,) = normalize_native_usage(context(), codex_event(), observed_at_ms=1)
    history = {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 10,
                    "reasoning_output_tokens": 3,
                    "total_tokens": 110,
                },
            },
        },
    }
    (old,) = normalize_native_usage(
        replace(context(), process_epoch="resumed"),
        history,
        observed_at_ms=2,
        origin="history",
    )
    assert old.series_key == live.series_key
    assert old.source_key == live.source_key


def test_claude_model_totals_and_mainloop_have_separate_scopes():
    event = {
        "type": "result",
        "uuid": "turn",
        "usage": {"input_tokens": 4},
        "modelUsage": {
            "model": {
                "inputTokens": 20,
                "outputTokens": 2,
                "cacheReadInputTokens": 3,
                "costUSD": "0.01",
            }
        },
    }
    items = normalize_native_usage(context("claude"), event, observed_at_ms=1)
    assert [item.scope for item in items] == ["call_model", "turn_main_loop"]
    assert items[0].includes_descendants is True
    assert items[0].reported_cost_usd == Decimal("0.01")
    assert items[0].counters["uncached_input_tokens"] == 20


def test_claude_canonical_total_requires_all_disjoint_counters():
    event = {
        "type": "result",
        "modelUsage": {
            "model": {
                "inputTokens": 100,
                "outputTokens": 20,
                "cacheReadInputTokens": 30,
                "cacheCreationInputTokens": 40,
            }
        },
    }
    (item,) = normalize_native_usage(context("claude"), event, observed_at_ms=1)
    assert item.counters["total_tokens"] == 190
    del event["modelUsage"]["model"]["cacheCreationInputTokens"]
    (item,) = normalize_native_usage(context("claude"), event, observed_at_ms=1)
    assert item.counters["total_tokens"] is None


def test_claude_repeated_assistant_message_replaces_detail_identity():
    event = {
        "type": "assistant",
        "message": {"id": "message", "model": "model", "usage": {"input_tokens": 4}},
    }
    (first,) = normalize_native_usage(context("claude"), event, observed_at_ms=1)
    event["message"]["usage"]["output_tokens"] = 6
    (second,) = normalize_native_usage(context("claude"), event, observed_at_ms=2)
    assert first.source_key == second.source_key
    assert (
        to_usage_observation(second, sequence=2).pricing_context["upstream_request_id"] == "message"
    )
    assert not second.authoritative


def test_agy_cache_not_subtracted_from_cli_input_and_step_not_added():
    usage = {
        "input_tokens": 278,
        "cache_read_tokens": 30214,
        "output_tokens": 4,
        "thinking_tokens": 0,
        "total_tokens": 282,
    }
    (item,) = normalize_native_usage(
        context("agy"),
        {"event": "result", "result": {"usage": usage, "model": "gemini-example"}},
        observed_at_ms=1,
    )
    assert item.counters["uncached_input_tokens"] == 278
    assert item.counters["total_tokens"] == 30496
    assert item.counters["reported_total_tokens"] == 282
    assert item.observed_model == "gemini-example"
    assert to_usage_observation(item, sequence=1).authoritative
    (step,) = normalize_native_usage(
        context("agy"),
        {"event": "step_update", "step_update": {"step_index": 2, "usage": usage}},
        observed_at_ms=1,
    )
    assert not to_usage_observation(step, sequence=1).authoritative


def test_history_bounds_torn_tail_append_and_rotation(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    line = (
        json.dumps(
            {
                "type": "assistant",
                "message": {"id": "message", "model": "model", "usage": {"input_tokens": 4}},
            }
        ).encode()
        + b"\n"
    )
    path.write_bytes(line + b'{"unfinished":')
    first = read_usage_batch(path, context("claude"), observed_at_ms=1, max_records=1)
    assert first.records_read == 1 and first.has_more
    partial = read_usage_batch(path, context("claude"), first.cursor, observed_at_ms=2)
    assert partial.status == "partial_record" and partial.cursor.offset == len(line)
    with path.open("ab") as stream:
        stream.write(b"0}\n" + line)
    appended = read_usage_batch(path, context("claude"), partial.cursor, observed_at_ms=3)
    assert appended.records_read == 2 and len(appended.observations) == 1
    assert not appended.has_more
    path.write_bytes(b"x" * len(line) + line)
    assert (
        read_usage_batch(path, context(), first.cursor, observed_at_ms=4).status == "source_changed"
    )


def test_history_oversize_and_malformed_are_explicit(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    path.write_bytes(b"bad-json\n" + b"x" * 200)
    batch = read_usage_batch(path, context(), observed_at_ms=1, max_record_bytes=100, max_bytes=500)
    assert batch.status == "oversize_record"
    assert batch.records_read == 1 and batch.cursor.offset == 209


def test_oversized_history_record_does_not_block_later_usage(tmp_path):
    path = tmp_path / "large.jsonl"
    line = (
        json.dumps(
            {
                "type": "assistant",
                "message": {"id": "message", "model": "model", "usage": {"input_tokens": 4}},
            }
        ).encode()
        + b"\n"
    )
    path.write_bytes(b"x" * 4096 + b"\n" + line)
    cursor = None
    observations = []
    for _ in range(10):
        batch = read_usage_batch(
            path,
            context("claude"),
            cursor,
            observed_at_ms=1,
            max_record_bytes=512,
            max_bytes=1024,
        )
        assert batch.cursor is not None
        assert batch.cursor.offset > (cursor.offset if cursor else 0)
        assert batch.cursor.offset - (cursor.offset if cursor else 0) <= 1024
        observations.extend(batch.observations)
        cursor = batch.cursor
        if not batch.has_more:
            break
    assert len(observations) == 1
    assert cursor.offset == path.stat().st_size


def test_account_windows_keep_scopes_nullable_resets_and_zero():
    value = codex_account_snapshot(
        "profile",
        {
            "rateLimitsByLimitId": {
                "first": {"primary": {"usedPercent": 0, "resetsAt": None}},
                "second": {"secondary": {"usedPercent": 100, "resetsAt": 20}},
            },
            "email": "DO_NOT_RETAIN",
        },
        observed_at_ms=1,
    )
    assert len(value.windows) == 2 and value.windows[0]["used_percent"] == 0
    assert "resets_at" not in value.windows[0]
    assert value.windows[1]["resets_at"] == 20000
    assert not value.verified and "DO_NOT_RETAIN" not in str(asdict(value))
    claude = claude_account_snapshot(
        "profile", {"rate_limit_info": {"status": "rejected"}}, observed_at_ms=1
    )
    assert claude.windows == ({"status": "rejected"},)
    assert (
        agy_statusline_snapshot(
            "profile", {"quota": {"x": {"remaining_fraction": 0}}}, observed_at_ms=1
        ).status
        == "available"
    )
