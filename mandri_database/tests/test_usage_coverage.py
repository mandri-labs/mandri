from dataclasses import replace
from itertools import permutations

import pytest
from mandri.core.types.usage import UsageObservation
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository


@pytest.fixture
async def repository(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "coverage.db")
    await db.migrate()
    yield UsageRepository(db)
    await db.close()


def native(**fields):
    return UsageObservation(
        **{
            "source": "native:codex",
            "source_key": "native",
            "fact_key": "native",
            "session_id": "child",
            "root_session_id": "root",
            "model": "model",
            "input_tokens": 100,
            "request_count": 1,
            "occurred_at": 150,
            "observed_at": 9000,
            "pricing_context": {"provider_kind": "openai", "evidence": "history_request"},
            **fields,
        }
    )


def gateway(**fields):
    return UsageObservation(
        **{
            "source": "gateway",
            "source_key": "gateway",
            "fact_key": "gateway",
            "session_id": "child",
            "root_session_id": "root",
            "model": "model",
            "input_tokens": 20,
            "request_count": 1,
            "occurred_at": 100,
            "observed_at": 200,
            "complete": True,
            "pricing_context": {"provider_kind": "openai", "status": "completed"},
            **fields,
        }
    )


def identified(value, identity):
    return replace(
        value, pricing_context={**value.pricing_context, "upstream_request_id": identity}
    )


async def check_pair(repository, first, second, reverse, tokens, facts, partial):
    for item in (second, first) if reverse else (first, second):
        await repository.record(item)
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == tokens
    assert result["summary"]["fact_count"] == facts
    assert (result["sync_state"]["status"] == "partial") is partial
    revision = result["revision"]
    await repository.record(first)
    await repository.record(second)
    assert (await repository.overview())["revision"] == revision


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("when", [99, 201, 10000])
async def test_native_outside_bounded_gateway_interval_survives(repository, reverse, when):
    await check_pair(repository, native(occurred_at=when), gateway(), reverse, 120, 2, False)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("when", [100, 150, 200])
async def test_bounded_ambiguous_overlap_uses_gateway_with_gap(repository, reverse, when):
    await check_pair(repository, native(occurred_at=when), gateway(), reverse, 20, 1, True)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("owner", ["sibling", "root"])
async def test_explicit_distinct_sessions_never_collapse_under_root(repository, reverse, owner):
    await check_pair(repository, native(session_id=owner), gateway(), reverse, 120, 2, False)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("owner", ["child", "root", None])
async def test_root_only_gateway_quarantines_bounded_overlap_with_gap(repository, reverse, owner):
    await check_pair(
        repository, native(session_id=owner), gateway(session_id=None), reverse, 20, 1, True
    )


@pytest.mark.parametrize("reverse", [False, True])
async def test_unrelated_roots_do_not_share_coverage(repository, reverse):
    await check_pair(
        repository,
        native(),
        gateway(session_id=None, root_session_id="other"),
        reverse,
        120,
        2,
        False,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "fields",
    [
        {"observed_at": None},
        {"observed_at": 50},
        {"occurred_at": None},
        {"complete": False, "pricing_context": {"provider_kind": "openai", "status": "pending"}},
    ],
)
async def test_unproven_gateway_extent_retains_native_but_reports_gap(repository, reverse, fields):
    await check_pair(repository, native(), gateway(**fields), reverse, 120, 2, True)


@pytest.mark.parametrize("reverse", [False, True])
async def test_missing_native_time_cannot_establish_overlap(repository, reverse):
    await check_pair(repository, native(occurred_at=None), gateway(), reverse, 120, 2, True)


@pytest.mark.parametrize("reverse", [False, True])
async def test_scan_time_is_never_native_request_end(repository, reverse):
    await check_pair(repository, native(occurred_at=50), gateway(), reverse, 120, 2, False)


@pytest.mark.parametrize("reverse", [False, True])
async def test_real_native_request_interval_can_overlap_despite_late_completion(
    repository, reverse
):
    await check_pair(
        repository, native(interval_start=150, occurred_at=250), gateway(), reverse, 20, 1, True
    )


@pytest.mark.parametrize("reverse", [False, True])
async def test_shared_upstream_identity_precedes_timestamps_and_model_aliases(repository, reverse):
    await check_pair(
        repository,
        identified(native(occurred_at=900, model="alias"), "response-1"),
        identified(gateway(), "response-1"),
        reverse,
        20,
        1,
        False,
    )


@pytest.mark.parametrize("reverse", [False, True])
async def test_distinct_upstream_requests_survive_equal_times_and_counters(repository, reverse):
    await check_pair(
        repository,
        identified(native(input_tokens=20), "response-1"),
        identified(gateway(), "response-2"),
        reverse,
        40,
        2,
        False,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("field", ["model", "provider_kind"])
async def test_known_distinct_models_or_providers_are_independent(repository, reverse, field):
    value = (
        native(model="other")
        if field == "model"
        else native(pricing_context={"evidence": "history_request", "provider_kind": "anthropic"})
    )
    await check_pair(repository, value, gateway(), reverse, 120, 2, False)


@pytest.mark.parametrize("reverse", [False, True])
async def test_local_message_identity_is_not_upstream_identity(repository, reverse):
    value = native(pricing_context={"provider_kind": "openai", "message_id": "response-1"})
    await check_pair(repository, value, identified(gateway(), "response-1"), reverse, 20, 1, True)


@pytest.mark.parametrize("order", list(permutations(range(3))))
async def test_multiple_gateway_arrival_orders_preserve_gap_for_quarantined_native(
    repository, order
):
    values = (
        identified(native(), "response-1"),
        identified(gateway(), "response-1"),
        gateway(source_key="second", fact_key="second"),
    )
    for index in order:
        await repository.record(values[index])
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 40
    assert result["summary"]["fact_count"] == 2
    assert result["sync_state"]["status"] == "partial"


async def test_pending_gateway_completion_reconciles_existing_native(repository):
    await repository.record(native())
    pending = gateway(
        complete=False, pricing_context={"provider_kind": "openai", "status": "pending"}
    )
    await repository.record(pending)
    assert (await repository.overview())["summary"]["input_tokens"] == 120
    await repository.record(replace(gateway(), sequence=1))
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 20
    assert result["sync_state"]["status"] == "partial"


@pytest.mark.parametrize("reverse", [False, True])
async def test_cumulative_interval_quarantine_does_not_invent_partial_counter_allocation(
    repository, reverse
):
    value = native(
        interval_start=50, occurred_at=250, epoch="epoch", request_count=3, pricing_context={}
    )
    await check_pair(repository, value, gateway(), reverse, 20, 1, True)


@pytest.mark.parametrize("reverse", [False, True])
async def test_explicit_non_overlapping_native_evidence_is_preserved(repository, reverse):
    await check_pair(repository, native(non_overlapping=True), gateway(), reverse, 120, 2, False)


async def test_pending_gap_resolves_when_finished_interval_proves_disjoint(repository):
    await repository.record(native(occurred_at=300))
    await repository.record(
        gateway(complete=False, pricing_context={"provider_kind": "openai", "status": "pending"})
    )
    assert (await repository.overview())["sync_state"]["gap_count"] == 1
    await repository.record(replace(gateway(), sequence=1))
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 120
    assert result["sync_state"]["gap_count"] == 0


async def test_pending_gap_resolves_with_genuine_final_request_identity(repository):
    await repository.record(identified(native(), "response-1"))
    await repository.record(
        gateway(complete=False, pricing_context={"provider_kind": "openai", "status": "pending"})
    )
    await repository.record(identified(replace(gateway(), sequence=1), "response-1"))
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 20
    assert result["sync_state"]["gap_count"] == 0


async def test_rebuild_replays_same_bounded_coverage_without_changing_totals(repository):
    early, overlap, late = (
        native(source_key="early", fact_key="early", occurred_at=50),
        native(),
        native(source_key="late", fact_key="late", occurred_at=300),
    )
    await repository.record(gateway())
    cursor = {"session_id": "child", "status": "ready"}
    values = [early, overlap, late]
    await repository.stage_history(values, "history", cursor, reset=True, complete=True)
    first = await repository.overview()
    await repository.stage_history(values, "history", cursor, reset=True, complete=True)
    second = await repository.overview()
    assert first["summary"] == second["summary"]
    assert second["summary"]["input_tokens"] == 220
    assert first["sync_state"]["gap_count"] == second["sync_state"]["gap_count"] == 1


async def test_restart_and_replay_preserve_quarantine_and_revision(tmp_path):
    path = tmp_path / "restart.db"
    db = AiosqliteDatabase()
    await db.connect(path)
    await db.migrate()
    repository = UsageRepository(db)
    await repository.record(native())
    await repository.record(gateway())
    first = await repository.overview()
    await db.close()
    await db.connect(path)
    try:
        repository = UsageRepository(db)
        await repository.record(gateway())
        await repository.record(native())
        second = await repository.overview()
        assert first["summary"] == second["summary"]
        assert first["revision"] == second["revision"]
        assert second["sync_state"]["gap_count"] == 1
    finally:
        await db.close()
