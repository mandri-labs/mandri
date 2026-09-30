from dataclasses import replace
from itertools import permutations

import pytest
from mandri.core.types.usage import UsageObservation
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage


@pytest.fixture
async def repository(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "history.db")
    await db.migrate()
    await migrate_usage(db._require_connection())
    yield UsageRepository(db)
    await db.close()


def pair():
    native = UsageObservation(
        source="native:claude",
        source_key="native",
        fact_key="native",
        session_id="session",
        harness="claude",
        provider="opencode_go",
        model="mimo",
        request_count=1,
        input_tokens=10,
        cache_read_tokens=90,
        cache_write_tokens=0,
        output_tokens=7,
        input_includes_cache=False,
        occurred_at=195,
        observed_at=1000,
        complete=True,
        pricing_context={
            "route_id": "route",
            "provider_kind": "opencode_go",
            "evidence": "history_request",
            "attributed_source": "gateway",
            "model_basis": "historical_gateway_route",
            "upstream_request_id": "converted",
        },
    )
    gateway = replace(
        native,
        source="gateway",
        source_key="gateway",
        fact_key="gateway",
        session_id=None,
        input_tokens=100,
        input_includes_cache=True,
        occurred_at=100,
        observed_at=200,
        pricing_context={
            "route_id": "route",
            "provider_kind": "opencode_go",
            "usage_protocol": "openai",
            "status": "completed",
            "upstream_request_id": "upstream",
        },
    )
    return native, gateway


@pytest.mark.parametrize("order", list(permutations(range(2))))
@pytest.mark.parametrize("root", [None, "session"])
async def test_unique_converted_historical_request_counted_once(repository, order, root):
    native, gateway = pair()
    values = native, replace(gateway, root_session_id=root)
    for index in order:
        await repository.record(values[index])
    assert (await repository.overview())["summary"]["fact_count"] == 2
    revision = await repository.reconcile_gateway_history()
    result = await repository.overview()
    assert result["summary"]["fact_count"] == 1
    assert result["summary"]["input_tokens"] == 100
    assert result["source_breakdown"]["gateway"]["fact_count"] == 1
    assert await repository.reconcile_gateway_history() == revision
    assert len(await repository._database.fetch_all("SELECT * FROM usage_observation")) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_tokens", 8),
        ("cache_read_tokens", 91),
        ("cache_write_tokens", 1),
        ("input_tokens", 11),
        ("occurred_at", 201),
        ("model", "other"),
        ("input_includes_cache", None),
        ("output_tokens", 0),
        ("non_overlapping", True),
        ("session_id", "other-session"),
    ],
)
async def test_incomplete_or_different_request_preserved(repository, field, value):
    native, gateway = pair()
    gateway = replace(gateway, root_session_id="session")
    await repository.record(replace(native, **{field: value}))
    await repository.record(gateway)
    await repository.reconcile_gateway_history()
    assert (await repository.overview())["summary"]["fact_count"] == 2


@pytest.mark.parametrize("source", ["native", "gateway"])
async def test_multiple_candidates_never_collapsed_and_previous_link_is_restored(
    repository, source
):
    native, gateway = pair()
    await repository.record(native)
    await repository.record(gateway)
    await repository.reconcile_gateway_history()
    duplicate = native if source == "native" else gateway
    await repository.record(replace(duplicate, source_key="other", fact_key="other"))
    await repository.reconcile_gateway_history()
    assert (await repository.overview())["summary"]["fact_count"] == 3


async def test_rebuilt_history_preserves_gateway_priority(repository):
    native, gateway = pair()
    await repository.record(native)
    await repository.record(gateway)
    await repository.reconcile_gateway_history()
    await repository.stage_history(
        [native],
        "history",
        {"session_id": "session", "status": "ready"},
        reset=True,
        complete=True,
    )
    await repository.reconcile_gateway_history()
    assert (await repository.overview())["summary"]["fact_count"] == 1
