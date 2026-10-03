from dataclasses import replace
from decimal import Decimal

import pytest
from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.database import usage_coverage
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage


@pytest.fixture
async def repository(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "usage.db")
    await db.migrate()
    await migrate_usage(db._require_connection())
    yield UsageRepository(db)
    await db.close()


def request(key: str, count: int, **fields):
    return UsageObservation(
        source="native:codex",
        source_key=key,
        fact_key=key,
        session_id="s",
        input_tokens=count,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        request_count=1,
        occurred_at=1000,
        sequence=1,
        input_includes_cache=True,
        output_includes_reasoning=True,
        pricing_context={"evidence": "history_request"},
        **fields,
    )


async def test_rebuild_retains_old_totals_until_atomic_swap_and_suppresses_live_duplicate(
    repository,
):
    old = request("old", 500, model=None)
    await repository.record(old)
    first = request("r1", 20, model="m")
    second = request("r2", 30, model="m")
    cursor = {"session_id": "s", "status": "backfill", "parser_version": 2}
    await repository.stage_history([first], "history", cursor, reset=True)
    assert (await repository.overview())["summary"]["input_tokens"] == 500
    await repository.stage_history(
        [second], "history", {**cursor, "status": "ready"}, complete=True
    )
    summary = (await repository.overview())["summary"]
    assert summary["input_tokens"] == 50
    assert summary["fact_count"] == 2
    assert summary["unclassified_fact_count"] == 0
    await repository.record(replace(first, source_key="live", fact_key="live", pricing_context={}))
    assert (await repository.overview())["summary"]["input_tokens"] == 50


async def test_rebuild_rollback_keeps_old_metrics_and_stage(repository):
    await repository.record(request("old", 500))
    cursor = {"session_id": "s", "status": "backfill"}
    await repository.stage_history([request("r", 20)], "history", cursor, reset=True)
    with pytest.raises(ValueError):
        await repository.stage_history([request("bad", -1)], "history", cursor, complete=True)
    assert (await repository.overview())["summary"]["input_tokens"] == 500
    await repository.stage_history([], "history", {**cursor, "status": "ready"}, complete=True)
    assert (await repository.overview())["summary"]["input_tokens"] == 20


async def test_erasure_removes_staged_history_and_prevents_rebuild(repository):
    cursor = {"session_id": "s", "status": "backfill"}
    await repository.stage_history([request("r", 20)], "history", cursor, reset=True)
    await repository.erase_session("s")
    await repository.stage_history([], "history", cursor, complete=True)
    assert (await repository.overview())["summary"]["fact_count"] == 0


async def test_gateway_interval_preserves_earlier_later_history_and_other_models(repository):
    early = replace(request("early", 20, model="m"), occurred_at=100)
    late = replace(request("late", 30, model="m"), occurred_at=300)
    other = replace(request("other", 40, model="native-model"), occurred_at=300)
    overlap = replace(request("overlap", 35, model="m"), occurred_at=220)
    await repository.record(early)
    await repository.record(late)
    await repository.record(other)
    await repository.record(overlap)
    gateway = replace(
        request("gateway", 30, model="m"),
        source="gateway",
        occurred_at=200,
        observed_at=250,
        complete=True,
        pricing_context={},
    )
    await repository.record(gateway)
    result = (await repository.overview())["summary"]
    assert result["input_tokens"] == 120
    assert result["fact_count"] == 4
    await repository.record(overlap)
    assert (await repository.overview())["summary"]["input_tokens"] == 120
    assert (await repository.overview())["sync_state"]["status"] == "partial"


async def test_cursor_only_progress_does_not_invalidate_usage_snapshot(repository):
    checkpoint = {"session_id": "s", "phase": "active", "status": "ready", "offset": 100}
    first = await repository.record_batch([], "history", checkpoint)
    second = await repository.record_batch([], "history", {**checkpoint, "offset": 200})
    assert second == first
    assert (await repository.read_cursor("history"))["offset"] == 200
    third = await repository.record_batch([], "history", {**checkpoint, "status": "partial"})
    assert third > second


async def test_replayed_final_checkpoint_keeps_committed_facts(repository):
    cursor = {"session_id": "s", "status": "ready"}
    first = await repository.stage_history(
        [request("r", 20)], "history", cursor, reset=True, complete=True
    )
    replay = await repository.stage_history([], "history", cursor, complete=True)
    assert replay == first
    assert (await repository.overview())["summary"]["input_tokens"] == 20
    replay_with_records = await repository.stage_history(
        [request("r", 20)], "history", cursor, complete=True
    )
    assert replay_with_records == first
    assert (await repository.overview())["summary"]["input_tokens"] == 20


async def test_public_current_price_can_revalue_old_observation_without_rescaling_tokens(
    repository,
):
    await repository.record(request("r", 1_000_000, model="m", provider="p"))
    await repository.add_prices(
        [
            UsagePrice(
                price_id="p:m:today",
                provider="p",
                model="m",
                effective_from=2000,
                reviewed_at=2000,
                valuation_basis="current_price_comparison",
                rates={
                    "input_tokens": Decimal("10"),
                    "output_tokens": Decimal("50"),
                    "cache_read_tokens": Decimal("1"),
                    "cache_write_tokens": Decimal("12.5"),
                },
            )
        ]
    )
    result = await repository.revalue_unpriced()
    assert result["updated"] == 1
    summary = (await repository.overview())["summary"]
    assert Decimal(summary["usd_equivalent"]) == Decimal("10.0005")
    assert summary["valuation_bases"] == {"current_price_comparison": 1}


@pytest.mark.parametrize("gateway_first", [False, True])
async def test_mandri_harness_alias_is_never_counted(repository, gateway_first):
    harness = replace(
        request("harness", 999, model="mandri_gateway", provider="mandri"), source="native:opencode"
    )
    gateway = replace(
        request("gateway", 100, model="glm-5.3", provider="opencode_go"), source="gateway"
    )
    for value in (gateway, harness) if gateway_first else (harness, gateway):
        await repository.record(value)
    result = await repository.overview()
    assert result["summary"]["fact_count"] == 1
    assert result["summary"]["input_tokens"] == 100
    assert result["breakdown"][0]["key"] == "glm-5.3"


async def test_ingestion_normalizes_route_and_harness_facts_before_valuation(repository):
    harness = replace(
        request("harness", 999, model="mandri_gateway", provider="mandri"), source="native:opencode"
    )
    gateway = replace(
        request("gateway", 100, model="wrong-model", provider="opencode_go"),
        source="gateway",
        observed_model="wrong-model",
        cache_write_tokens=None,
        pricing_context={
            "selected_model": "custom_openai/glm-5.3-flash",
            "provider_kind": "opencode_go",
            "usage_protocol": "openai",
            "raw_usage": {"prompt_tokens": 100, "completion_tokens": 10},
        },
    )
    for value in (harness, gateway):
        await repository.record(value)
    await repository.add_price(
        UsagePrice(
            price_id="glm",
            provider="opencode_go",
            model="glm-5.3-flash",
            effective_from=0,
            rates={"input_tokens": Decimal("0.15"), "output_tokens": Decimal("0.5")},
        )
    )
    result = await repository.revalue_unpriced(all_facts=True)
    assert result["updated"] == 1
    overview = await repository.overview()
    assert overview["summary"]["fact_count"] == 1
    assert overview["summary"]["usd_equivalent"] == "0.00002"
    assert overview["breakdown"][0]["key"] == "glm-5.3-flash"
    assert (await repository.revalue_unpriced(all_facts=True))["updated"] == 0
    evidence = await repository._database.fetch_one(
        "SELECT payload FROM usage_observation WHERE source_key='harness'"
    )
    assert evidence is not None


async def test_ingestion_excludes_total_failures_from_valuation(
    repository,
):
    for key in ("luna", "free"):
        value = replace(
            request(key, 0, model=key),
            source="gateway",
            input_tokens=None,
            output_tokens=None,
            complete=False,
            pricing_context={"status": "failed"},
        )
        await repository.record(value)
    assert (await repository.overview())["summary"]["fact_count"] == 0
    assert (await repository.revalue_unpriced(all_facts=True))["updated"] == 0
    result = await repository.overview()
    assert result["summary"]["fact_count"] == 0
    assert result["summary"]["unpriced_fact_count"] == 0
    assert result["summary"]["incomplete_fact_count"] == 0
    assert result["breakdown"] == []
    assert result["timeseries"] == []
    assert (await repository.revalue_unpriced(all_facts=True))["updated"] == 0


async def test_staging_progress_only_invalidates_visible_changes(repository):
    cursor = {"session_id": "s", "status": "backfill", "offset": 100}
    first = await repository.stage_history([request("a", 20)], "history", cursor, reset=True)
    second = await repository.stage_history(
        [request("b", 30)], "history", {**cursor, "offset": 200}
    )
    assert second == first
    assert (await repository.read_cursor("history"))["offset"] == 200
    partial = await repository.stage_history([], "history", {**cursor, "status": "partial"})
    assert partial > second
    completed = await repository.stage_history(
        [], "history", {**cursor, "status": "ready"}, complete=True
    )
    assert completed > partial
    assert (await repository.overview())["summary"]["input_tokens"] == 50


async def test_rebuild_loads_gateway_coverage_once_and_refreshes_it_next_transaction(
    repository, monkeypatch
):
    calls = []
    original = usage_coverage.gateway_facts

    def track(db, value):
        calls.append(value.session_id)
        return original(db, value)

    monkeypatch.setattr(usage_coverage, "gateway_facts", track)
    values = [request(f"r{i}", 20, model="m") for i in range(100)]
    cursor = {"session_id": "s", "status": "ready"}
    await repository.stage_history(values, "history", cursor, reset=True, complete=True)
    assert calls == ["s"]
    gateway = replace(
        request("gateway", 30, model="m"),
        source="gateway",
        occurred_at=900,
        observed_at=1100,
        complete=True,
        pricing_context={},
    )
    await repository.record(gateway)
    calls.clear()
    await repository.stage_history(values, "history", cursor, reset=True, complete=True)
    assert calls == ["s"]
    summary = (await repository.overview())["summary"]
    assert summary["fact_count"] == 1
    assert summary["input_tokens"] == 30
