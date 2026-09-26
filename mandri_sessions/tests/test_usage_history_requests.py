import json
from dataclasses import asdict, replace
from decimal import Decimal

import pytest
from mandri.core.types.usage import UsageFilters, UsagePrice
from mandri.core.usage_pricing import value_usage
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.sessions.usage.adapter import to_usage_observation
from mandri.sessions.usage.history import UsageCursor, read_usage_batch
from mandri.sessions.usage.history_headers import irrelevant_codex_record
from mandri.sessions.usage.types import NativeUsageContext


def context(harness="codex", **fields):
    return NativeUsageContext(
        "session",
        harness,
        "history",
        native_id="native",
        routing="native",
        inherited_history="none",
        provider_kind="openai" if harness == "codex" else "anthropic",
        **fields,
    )


def totals(value):
    return {
        "input_tokens": value,
        "cached_input_tokens": value // 2,
        "output_tokens": value // 10,
        "reasoning_output_tokens": value // 20,
        "total_tokens": value + value // 10,
    }


def token_event(value, last=None):
    info = {"total_token_usage": totals(value)}
    if last is not None:
        info["last_token_usage"] = totals(last)
    return {
        "timestamp": "2026-09-20T00:00:00Z",
        "type": "event_msg",
        "payload": {"type": "token_count", "info": info},
    }


def model_event(model="model-a", turn="turn-a"):
    return {"type": "turn_context", "payload": {"model": model, "turn_id": turn}}


def write_events(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def scan(path, scope=None, max_records=200):
    cursor = None
    observations = []
    for _ in range(100):
        batch = read_usage_batch(
            path, scope or context(), cursor, observed_at_ms=10, max_records=max_records
        )
        observations.extend(batch.observations)
        cursor = UsageCursor(**json.loads(json.dumps(asdict(batch.cursor))))
        if not batch.has_more:
            return observations, batch
    pytest.fail("History did not reach EOF")


@pytest.mark.parametrize("page_size", [1, 2, 3, 200])
def test_codex_pagination_models_repeated_last_and_stable_replay(tmp_path, page_size):
    path = tmp_path / "history.jsonl"
    write_events(
        path,
        [
            model_event(),
            token_event(100, 100),
            token_event(100, 100),
            token_event(200, 100),
            model_event("model-b", "turn-b"),
            token_event(240, 40),
            token_event(240, 40),
        ],
    )
    items, batch = scan(path, max_records=page_size)
    assert batch.status == "ready"
    assert [item.model for item in items] == ["model-a", "model-a", "model-b"]
    assert [item.counters["input_tokens"] for item in items] == [100, 100, 40]
    assert [item.turn_id for item in items] == ["turn-a", "turn-a", "turn-b"]
    assert all(item.counters["request_count"] == 1 for item in items)
    replay, _ = scan(path, replace(context(), process_epoch="another"))
    assert [item.source_key for item in items] == [item.source_key for item in replay]
    assert len({item.source_key for item in items}) == 3
    assert batch.cursor.parser_version == 2
    state = json.loads(batch.cursor.state_json)
    assert state["model"] == "model-b" and state["previous"]["input_tokens"] == 240


def test_codex_counter_subsets_schema_zero_and_dollar_value(tmp_path):
    path = tmp_path / "history.jsonl"
    write_events(path, [model_event(), token_event(100, 100)])
    (item,), _ = scan(path)
    value = to_usage_observation(item, sequence=0)
    assert value.input_tokens == 100 and value.cache_read_tokens == 50
    assert value.output_tokens == 10 and value.reasoning_tokens == 5
    assert value.cache_write_tokens == 0 and value.total_tokens == 110
    assert value.request_count == 1 and value.complete and value.authoritative
    assert value.input_includes_cache and value.output_includes_reasoning
    assert value.pricing_context["context_tokens"] == 100
    assert value.pricing_context["evidence"] == "history_request"
    price = UsagePrice(
        price_id="price",
        provider="openai",
        model="model-a",
        effective_from=0,
        rates={
            "input_tokens": Decimal("2"),
            "cache_read_tokens": Decimal("0.2"),
            "output_tokens": Decimal("10"),
        },
    )
    assert value_usage(value, [price]) == (Decimal("0.00021"), "price")


def test_codex_explicit_cache_write_is_a_subset(tmp_path):
    path = tmp_path / "history.jsonl"
    event = token_event(100)
    event["payload"]["info"]["total_token_usage"]["cache_write_input_tokens"] = 10
    write_events(path, [model_event(), event])
    (item,), _ = scan(path)
    assert item.counters["cache_write_tokens"] == 10
    assert item.counters["uncached_input_tokens"] == 40


def test_codex_resets_repeated_last_and_equal_sized_requests(tmp_path):
    path = tmp_path / "history.jsonl"
    write_events(
        path,
        [
            model_event(),
            token_event(100, 100),
            token_event(200, 100),
            token_event(0, 100),
            token_event(100, 100),
            token_event(100, 100),
            token_event(140, 40),
            token_event(20, 20),
            token_event(40, 20),
        ],
    )
    items, batch = scan(path, max_records=1)
    assert batch.status == "ready"
    assert [item.counters["input_tokens"] for item in items] == [100, 100, 100, 40, 20, 20]
    assert len({item.source_key for item in items}) == 6


def test_codex_unproven_reset_does_not_replay_stale_last(tmp_path):
    path = tmp_path / "history.jsonl"
    write_events(
        path,
        [
            model_event(),
            token_event(100, 100),
            token_event(200, 100),
            token_event(140, 100),
            token_event(160, 20),
        ],
    )
    items, batch = scan(path)
    assert [item.counters["input_tokens"] for item in items] == [100, 100, 20]
    assert batch.status == "partial_usage"
    assert json.loads(batch.cursor.state_json)["discard_reasons"] == {"unproven_reset": 1}


def test_codex_first_snapshot_uses_validated_last_not_lifetime_model(tmp_path):
    path = tmp_path / "history.jsonl"
    write_events(path, [model_event("model-b"), token_event(1000, 40), token_event(1020, 20)])
    items, _ = scan(path, max_records=1)
    assert [item.counters["input_tokens"] for item in items] == [40, 20]


def test_codex_aggregate_has_no_request_context_tier_or_complete_claim(tmp_path):
    path = tmp_path / "history.jsonl"
    write_events(path, [model_event(), token_event(100, 100), token_event(200, 40)])
    items, _ = scan(path)
    value = to_usage_observation(items[-1], sequence=1)
    assert value.input_tokens == 100 and value.request_count is None
    assert "context_tokens" not in value.pricing_context and not value.complete


@pytest.mark.parametrize("mutation", ["negative", "boolean", "cache_exceeds_input", "total"])
def test_codex_invalid_counters_expose_diagnostics(tmp_path, mutation):
    path = tmp_path / "history.jsonl"
    event = token_event(100, 100)
    raw = event["payload"]["info"]["total_token_usage"]
    if mutation == "negative":
        raw["input_tokens"] = -1
    elif mutation == "boolean":
        raw["input_tokens"] = True
    elif mutation == "cache_exceeds_input":
        raw["cached_input_tokens"] = 101
    else:
        raw["total_tokens"] = 111
    write_events(path, [model_event(), event])
    items, batch = scan(path)
    assert not items and batch.status == "partial_usage"
    assert json.loads(batch.cursor.state_json)["discard_reasons"] == {"invalid_counters": 1}


def test_codex_missing_model_keeps_counter_baseline_and_reports_gap(tmp_path):
    path = tmp_path / "history.jsonl"
    write_events(path, [token_event(100, 100), model_event(), token_event(120, 20)])
    (item,), batch = scan(path, max_records=1)
    assert item.counters["input_tokens"] == 20
    state = json.loads(batch.cursor.state_json)
    assert state["discarded_events"] == 1 and state["discard_reasons"] == {"missing_model": 1}


def test_codex_fork_boundary_excludes_inherited_models_and_totals(tmp_path):
    path = tmp_path / "history.jsonl"
    write_events(
        path,
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "native",
                    "forked_from_id": "parent",
                    "subagent_history_start_ordinal": 4,
                },
            },
            {"type": "session_meta", "payload": {"id": "parent"}},
            model_event("parent-model"),
            token_event(1000, 100),
            model_event("child-model"),
            token_event(1040, 40),
            token_event(1060, 20),
        ],
    )
    scope = replace(context(parent_session_id="parent"), inherited_history="unknown")
    items, batch = scan(path, scope, max_records=1)
    assert batch.status == "ready"
    assert [item.counters["input_tokens"] for item in items] == [40, 20]
    assert all(item.model == "child-model" and item.own_usage_proven for item in items)
    assert all(to_usage_observation(item, sequence=1).authoritative for item in items)
    assert all(item.context.parent_session_id == "parent" for item in items)


@pytest.mark.parametrize(
    "metadata",
    [
        {"id": "native", "forked_from_id": "parent"},
        {"id": "different"},
    ],
)
def test_codex_unproven_ownership_never_adds_child_or_foreign_totals(tmp_path, metadata):
    path = tmp_path / "history.jsonl"
    write_events(
        path, [{"type": "session_meta", "payload": metadata}, model_event(), token_event(100, 100)]
    )
    items, batch = scan(path, context(parent_session_id="parent"))
    assert not items and batch.status == "partial_usage"
    assert json.loads(batch.cursor.state_json)["discard_reasons"] == {"unproven_ownership": 1}


def test_codex_fresh_child_metadata_proves_own_requests(tmp_path):
    path = tmp_path / "history.jsonl"
    write_events(
        path,
        [
            {"type": "session_meta", "payload": {"id": "native", "parent_thread_id": "parent"}},
            model_event(),
            token_event(100, 100),
        ],
    )
    (item,), _ = scan(path, context(parent_session_id="parent"))
    assert to_usage_observation(item, sequence=1).authoritative


def test_cursor_checkpoint_has_no_transcript_or_model_fallback_leak(tmp_path):
    path = tmp_path / "history.jsonl"
    event = model_event()
    event["payload"]["instructions"] = "PRIVATE_TEXT"
    write_events(
        path,
        [
            event,
            {"type": "response_item", "payload": {"text": "PRIVATE_TEXT"}},
            token_event(100, 100),
        ],
    )
    first = read_usage_batch(path, context(), observed_at_ms=1, max_records=1)
    assert not first.observations and "PRIVATE_TEXT" not in first.cursor.state_json
    second = read_usage_batch(path, context(), first.cursor, observed_at_ms=2)
    assert second.observations[0].model == "model-a"
    assert "PRIVATE_TEXT" not in str(asdict(second))
    stale = replace(first.cursor, parser_version=1)
    assert read_usage_batch(path, context(), stale, observed_at_ms=2).status == "source_changed"


def claude_message(output, model="claude-model"):
    return {
        "type": "assistant",
        "sessionId": "native",
        "timestamp": "2026-09-20T00:00:00Z",
        "message": {
            "id": "message",
            "model": model,
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 20,
                "output_tokens": output,
            },
        },
    }


async def test_claude_history_message_upserts_and_result_totals_never_overlap(tmp_path):
    path = tmp_path / "claude.jsonl"
    write_events(
        path,
        [
            claude_message(1),
            claude_message(7),
            {
                "type": "result",
                "modelUsage": {"claude-model": {"inputTokens": 99999}},
                "usage": {"input_tokens": 99999},
            },
        ],
    )
    items, _ = scan(path, context("claude"), max_records=1)
    assert len(items) == 2 and items[0].source_key == items[1].source_key
    values = [to_usage_observation(item, sequence=0) for item in items]
    assert all(value.authoritative and value.complete for value in values)
    assert values[-1].total_tokens == 67 and values[-1].request_count == 1
    assert values[-1].pricing_context["context_tokens"] == 60
    assert values[-1].input_includes_cache is False
    assert values[1].sequence > values[0].sequence
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "usage.sqlite")
    await database.migrate()
    try:
        repository = UsageRepository(database)
        for value in values:
            await repository.record(value)
        overview = await repository.overview(UsageFilters())
        assert overview["summary"]["input_tokens"] == 10
        assert overview["summary"]["output_tokens"] == 7
        assert overview["summary"]["request_count"] == 1
    finally:
        await database.close()


def test_claude_message_id_survives_restart_model_enrichment_and_pagination(tmp_path):
    path = tmp_path / "claude.jsonl"
    write_events(path, [claude_message(0, model=None), claude_message(7)])
    items, _ = scan(path, context("claude"), max_records=1)
    replay, _ = scan(path, replace(context("claude"), process_epoch="restart"))
    assert len({item.source_key for item in [*items, *replay]}) == 1
    assert not to_usage_observation(items[0], sequence=0).authoritative
    assert to_usage_observation(items[1], sequence=0).authoritative


@pytest.mark.parametrize("kind", ["response_item", "compacted", "event_msg"])
def test_oversized_non_usage_records_preserve_model_and_counters_across_pages(tmp_path, kind):
    path = tmp_path / "large.jsonl"
    payload = {"type": "item_completed", "text": "PRIVATE_TEXT" * 1000}
    write_events(
        path,
        [
            model_event(),
            token_event(100, 100),
            {"type": kind, "payload": payload},
            token_event(120, 20),
        ],
    )
    cursor = None
    items = []
    for _ in range(100):
        batch = read_usage_batch(
            path,
            context(),
            cursor,
            observed_at_ms=1,
            max_record_bytes=512,
            max_bytes=1024,
        )
        items.extend(batch.observations)
        cursor = UsageCursor(**asdict(batch.cursor))
        assert "PRIVATE_TEXT" not in str(asdict(batch))
        if not batch.has_more:
            break
    assert batch.status == "ready" and not batch.has_more
    assert [item.counters["input_tokens"] for item in items] == [100, 20]
    assert json.loads(cursor.state_json)["discarded_events"] == 0


@pytest.mark.parametrize(
    "prefix",
    [
        b'{"type":"turn_context","payload":{"text":"',
        b'{"type":"event_msg","payload":{"type":"token_count","text":"',
        b'{"nested":{"type":"response_item"},"type":"turn_context","payload":',
        b'{"payload":{"text":"response_item',
    ],
)
def test_oversized_relevant_or_unknown_header_never_preserves_context(prefix):
    assert not irrelevant_codex_record(prefix)


def test_history_provider_metadata_survives_model_changes_without_openai_guess(tmp_path):
    path = tmp_path / "providers.jsonl"
    write_events(
        path,
        [
            {"type": "session_meta", "payload": {"id": "native", "model_provider": "custom"}},
            model_event(),
            token_event(100, 100),
            {"type": "turn_context", "payload": {"model": "model-b", "model_provider": "openai"}},
            token_event(120, 20),
        ],
    )
    items, _ = scan(path, max_records=1)
    first, second = [to_usage_observation(item, sequence=0) for item in items]
    assert first.pricing_context["provider_kind"] is None
    assert first.pricing_context["observed_provider"] == "custom"
    assert second.pricing_context["provider_kind"] == "openai"
    assert first.authoritative and second.authoritative
    mapped, _ = scan(path, replace(context(), observed_model="model-a", provider_kind="anthropic"))
    assert (
        to_usage_observation(mapped[0], sequence=0).pricing_context["provider_kind"] == "anthropic"
    )
    assert to_usage_observation(mapped[1], sequence=0).pricing_context["provider_kind"] == "openai"


def test_missing_last_never_fabricates_request_count_or_context_size(tmp_path):
    path = tmp_path / "aggregate.jsonl"
    write_events(path, [model_event(), token_event(100), token_event(200)])
    items, _ = scan(path)
    for item in items:
        value = to_usage_observation(item, sequence=0)
        assert value.request_count is None and not value.complete
        assert "context_tokens" not in value.pricing_context


def test_codex_model_switch_aggregate_only_assigns_proven_last_request(tmp_path):
    path = tmp_path / "aggregate.jsonl"
    write_events(
        path, [model_event(), token_event(100, 100), model_event("model-b"), token_event(200, 40)]
    )
    items, batch = scan(path, max_records=1)
    assert [item.counters["input_tokens"] for item in items] == [100, 40]
    assert items[1].model == "model-b" and items[1].counters["request_count"] == 1
    assert json.loads(batch.cursor.state_json)["discard_reasons"] == {"unattributed_aggregate": 1}
