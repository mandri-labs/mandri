import json
import sqlite3
from dataclasses import asdict, replace
from decimal import Decimal

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, ProjectPath
from mandri.core.ports.transcripts import SessionRef
from mandri.core.types.usage import UsageFilters
from mandri.core.usage_pricing import REVIEWED_AT, bundled_prices, value_usage
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.sessions.usage_opencode import OpencodeUsageReader, opencode_usage_observation


def message(identity="message-1", **changes):
    return {
        "id": identity,
        "role": "assistant",
        "sessionID": "native",
        "providerID": "openai",
        "modelID": "gpt-4.1-mini",
        "time": {"created": REVIEWED_AT + 10, "completed": REVIEWED_AT + 20},
        "tokens": {
            "input": 400,
            "output": 200,
            "reasoning": 50,
            "cache": {"read": 600, "write": 0},
            "total": 1200,
        },
        "cost": "0.00062",
        "path": {"cwd": "private-directory"},
        "text": "private-content",
        **changes,
    }


def observation(info, **kwargs):
    return opencode_usage_observation(
        info,
        session_id="managed",
        native_id="native",
        session_created_at_ms=REVIEWED_AT,
        observed_at_ms=REVIEWED_AT + 100,
        **kwargs,
    )


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "opencode.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE session (id TEXT PRIMARY KEY, time_created INTEGER)")
        db.execute("CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT)")
        db.executemany(
            "INSERT INTO session VALUES (?, ?)", [("native", REVIEWED_AT), ("child", REVIEWED_AT)]
        )
    return path


def insert(path, *messages):
    with sqlite3.connect(path) as db:
        db.executemany(
            "INSERT INTO message VALUES (?, ?, ?)",
            [(item["id"], item.get("sessionID", "native"), json.dumps(item)) for item in messages],
        )


def batch(path, native="native", **kwargs):
    return OpencodeUsageReader(path).read_usage(
        SessionRef(HarnessKind.OPENCODE, HarnessSessionId(native), ProjectPath("project")),
        session_id=f"managed-{native}",
        observed_at_ms=REVIEWED_AT + 100,
        **kwargs,
    )


def test_metadata_provenance_and_price():
    item = observation(message(), billing_mode="api")
    assert item is not None
    assert (item.provider, item.model, item.billing_mode) == ("openai", "gpt-4.1-mini", "api")
    assert item.input_includes_cache is False
    assert item.output_includes_reasoning is True
    assert item.pricing_context["context_tokens"] == 1000
    assert item.reported_cost_usd == Decimal("0.00062")
    assert item.reported_cost_basis == "harness_estimate"
    assert item.complete and item.authoritative
    assert item.request_count == 1
    assert "private" not in json.dumps(asdict(item), default=str)
    assert value_usage(item, bundled_prices())[0] == Decimal("0.00054")


def test_pagination_exact_session_and_stable_message_keys(store):
    insert(store, message("first"), message("second"), message("child-message", sessionID="child"))
    before = store.read_bytes()
    first = batch(store, max_records=1)
    second = batch(store, max_records=1, cursor=first.cursor)
    again = batch(store)
    assert first.has_more and not second.has_more
    assert first.records_read == second.records_read == 1
    assert {item.source_key for item in again.observations} == {
        first.observations[0].source_key,
        second.observations[0].source_key,
    }
    assert len(batch(store, "child").observations) == 1
    assert store.read_bytes() == before


def test_excludes_fork_copies_and_user_metadata(store):
    insert(
        store,
        message("copied", time={"created": REVIEWED_AT - 10, "completed": REVIEWED_AT - 5}),
        message("user", role="user"),
        message("own"),
    )
    result = batch(store)
    assert result.status == "ready"
    assert len(result.observations) == 1
    assert result.observations[0].pricing_context["message_id"] == "own"


def test_malformed_and_oversize_records_are_bounded_and_skipped(store):
    insert(store, message("valid"), message("oversize", text="x" * 65536))
    with sqlite3.connect(store) as db:
        db.execute("INSERT INTO message VALUES ('broken', 'native', '{')")
    result = batch(store, max_records=2)
    assert result.status == "partial" and result.has_more
    assert result.records_read == 2 and result.observations == ()
    next_batch = batch(store, cursor=result.cursor)
    assert len(next_batch.observations) == 1


def test_unavailable_does_not_create_store(tmp_path):
    path = tmp_path / "missing.db"
    assert batch(path).status == "unavailable"
    assert not path.exists()


@pytest.mark.parametrize("limit", [0, 1001, True])
def test_rejects_unbounded_reads(store, limit):
    with pytest.raises(ValueError):
        batch(store, max_records=limit)


def test_partial_and_zero_counters_are_preserved_without_invention():
    item = observation(message(tokens={"input": 0, "output": True}, cost="NaN"))
    assert item is not None
    assert item.input_tokens == 0 and item.output_tokens is None
    assert item.cache_read_tokens is None and item.reported_cost_usd is None
    assert not item.complete
    zero = observation(message(cost=0))
    assert zero.reported_cost_usd == Decimal(0)


@pytest.mark.parametrize(
    "provider,mode,kind",
    [
        ("opencode", "api", "opencode"),
        ("opencode-go", "subscription", "opencode_go"),
        ("ollama", "local", "ollama"),
        ("google", "api", "gemini"),
    ],
)
def test_native_provider_classification(provider, mode, kind):
    item = observation(message(providerID=provider))
    assert item.billing_mode == mode
    assert item.pricing_context["provider_kind"] == kind
    assert item.provider == provider


def test_gateway_routing_is_not_authoritative_native_spend():
    item = observation(message(), routing="gateway")
    assert not item.authoritative and not item.non_overlapping
    assert item.epoch is not None


def test_history_authority_does_not_assert_disjoint_gateway_coverage(store):
    insert(store, message())
    item = batch(store).observations[0]
    assert item.authoritative and not item.non_overlapping
    assert item.pricing_context["history_request"] is True
    assert item.pricing_context["message_completed_at"] == REVIEWED_AT + 20


async def test_completion_update_keeps_identity_and_replaces_incomplete_usage(store, tmp_path):
    incomplete = message(time={"created": REVIEWED_AT + 10}, tokens={"input": 400}, cost=0)
    insert(store, incomplete)
    before = batch(store).observations[0]
    assert not before.complete
    assert before.pricing_context["message_completed_at"] is None
    with sqlite3.connect(store) as connection:
        connection.execute("UPDATE message SET data = ?", (json.dumps(message()),))
    after = (
        OpencodeUsageReader(store)
        .read_usage(
            SessionRef(HarnessKind.OPENCODE, HarnessSessionId("native"), ProjectPath("project")),
            session_id="managed-native",
            observed_at_ms=before.sequence + 1,
        )
        .observations[0]
    )
    assert after.complete and before.fact_key == after.fact_key
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "usage.sqlite")
    await db.migrate()
    try:
        repository = UsageRepository(db)
        await repository.record(before)
        await repository.record(after)
        summary = (await repository.overview())["summary"]
        assert summary["fact_count"] == 1
        assert summary["incomplete_fact_count"] == 0
        assert summary["reported_cost_usd"] == "0.00062"
        assert summary["output_tokens"] == 200
    finally:
        await db.close()


@pytest.mark.parametrize(
    "total,inclusive,expected",
    [(1250, False, "0.00062"), (1200, True, "0.00054"), (1300, None, None), (None, None, None)],
)
@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "custom-provider"])
def test_reasoning_overlap_uses_counter_evidence_across_versions(
    total, inclusive, expected, provider
):
    info = message(providerID=provider)
    info["tokens"]["total"] = total
    item = observation(info)
    assert item.output_includes_reasoning is inclusive
    priced = replace(
        item, provider="openai", pricing_context={**item.pricing_context, "provider_kind": "openai"}
    )
    assert value_usage(priced, bundled_prices())[0] == (
        Decimal(expected) if expected is not None else None
    )


async def test_replaying_and_updating_history_replaces_fact(store, tmp_path):
    insert(store, message())
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "mandri.sqlite")
    await db.migrate()
    repository = UsageRepository(db)
    try:
        for price in bundled_prices():
            await repository.add_price(price)
        item = batch(store).observations[0]
        await repository.record(item)
        await repository.record(item)
        await repository.record(replace(item, sequence=item.sequence + 1, output_tokens=300))
        result = await repository.overview(UsageFilters())
        assert result["summary"]["fact_count"] == 1
        assert result["summary"]["output_tokens"] == 300
        assert Decimal(result["summary"]["usd_equivalent"]) == Decimal("0.00070")
    finally:
        await db.close()


async def test_failed_native_request_without_output_is_absent_from_usage(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "usage.db")
    await db.migrate()
    repo = UsageRepository(db)
    value = observation(
        message(
            error={"name": "APIError"},
            cost=0,
            tokens={"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}},
        )
    )
    assert value is not None
    try:
        await repo.record(value)
        result = await repo.overview()
        assert result["summary"]["fact_count"] == 0
        assert result["summary"]["incomplete_fact_count"] == 0
        assert result["breakdown"] == []
    finally:
        await db.close()
