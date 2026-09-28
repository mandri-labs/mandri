import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest
from mandri.api.app import create_app
from mandri.api.deps import LifespanState, database, usage_repository
from mandri.core.types.usage import UsageAccount, UsageObservation, UsagePrice
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage


@pytest.fixture
async def usage_client() -> AsyncIterator[
    tuple[httpx.AsyncClient, UsageRepository, AiosqliteDatabase]
]:
    db = AiosqliteDatabase()
    await db.connect(":memory:")
    await db.migrate()
    await migrate_usage(db._require_connection())
    repo = UsageRepository(db)
    app = create_app()
    app.dependency_overrides[usage_repository] = lambda: repo
    app.dependency_overrides[database] = lambda: db
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, repo, db
    await db.close()


def _observation() -> UsageObservation:
    return UsageObservation(
        source="gateway",
        source_key="request-1",
        fact_key="request-1",
        session_id="session-1",
        project_path="/synthetic/project",
        harness="codex",
        model="synthetic-model",
        provider="synthetic-provider",
        occurred_at=1750000000000,
        input_tokens=1000,
        cache_read_tokens=600,
        cache_write_tokens=0,
        output_tokens=200,
        reasoning_tokens=50,
        total_tokens=1200,
        input_includes_cache=True,
        output_includes_reasoning=True,
        request_count=1,
        complete=True,
    )


async def test_old_account_readings_become_stale_without_changing_quota(usage_client):
    client, repo, _ = usage_client
    await repo.upsert_account(
        UsageAccount(
            account_id="profile",
            harness="codex",
            observed_at=1,
            status="available",
            windows=({"used_percent": 42},),
        )
    )
    response = await client.get("/v1/usage/accounts")
    assert response.status_code == 200
    account = response.json()["accounts"][0]
    assert account["status"] == "stale"
    assert account["windows"][0]["used_percent"] == 42
    assert (await repo.accounts())[0]["status"] == "available"


async def test_overview_retains_deleted_session_and_marks_unknown_price(usage_client):
    client, repo, db = usage_client
    await db.execute(
        "INSERT INTO session(id,harness,project_path,created_at,updated_at,state,last_synced_at)"
        " VALUES('session-1','codex','/synthetic/project',1,1,'stopped',1)"
    )
    await repo.record(_observation())
    response = await client.get("/v1/usage/overview")
    assert response.status_code == 200
    assert response.json()["summary"]["total_tokens"] == 1200
    assert response.json()["summary"]["usd_equivalent"] is None
    await db.execute("UPDATE session SET deleted=1 WHERE id='session-1'")
    retained = (await client.get("/v1/usage/overview")).json()
    assert retained["summary"]["total_tokens"] == 1200
    hidden = (await client.get("/v1/usage/overview?include_deleted=false")).json()
    assert hidden["summary"]["fact_count"] == 0


async def test_unknown_zero_and_model_segments_remain_distinct(usage_client):
    client, repo, _ = usage_client
    await repo.record(_observation())
    await repo.record(
        replace(
            _observation(),
            source_key="request-2",
            fact_key="request-2",
            model="other-model",
            input_tokens=None,
            output_tokens=0,
            total_tokens=None,
        )
    )
    result = (await client.get("/v1/usage/overview?group_by=model")).json()
    groups = {row["key"]: row for row in result["breakdown"]}
    assert groups["other-model"]["input_tokens"] is None
    assert groups["other-model"]["output_tokens"] == 0
    assert groups["synthetic-model"]["input_tokens"] == 1000


@pytest.mark.parametrize(
    "query",
    [
        "from_ms=20&to_ms=10",
        "timezone=No/SuchZone",
        "limit=501",
        "offset=-1",
        "group_by=unsafe",
        "from_ms=-1",
    ],
)
async def test_invalid_usage_filters_are_rejected(usage_client, query):
    client, _, _ = usage_client
    assert (await client.get(f"/v1/usage/overview?{query}")).status_code == 422


async def test_explicit_erase_requires_confirmation_and_suppresses_replay(usage_client):
    client, repo, _ = usage_client
    await repo.record(_observation())
    path = "/v1/usage/sessions/session-1/erase"
    assert (await client.post(path, json={})).status_code == 422
    assert (await client.post(path, json={"confirmed": True})).status_code == 200
    await repo.record(_observation())
    result = (await client.get("/v1/usage/overview")).json()
    assert result["summary"]["fact_count"] == 0


async def test_live_session_usage_cannot_be_erased(usage_client):
    client, _, db = usage_client
    await db.execute(
        "INSERT INTO session(id,harness,project_path,created_at,updated_at,state,last_synced_at)"
        " VALUES('session-1','codex','/synthetic/project',1,1,'live',1)"
    )
    response = await client.post("/v1/usage/sessions/session-1/erase", json={"confirmed": True})
    assert response.status_code == 409


async def test_price_override_values_disjoint_tokens(usage_client):
    client, repo, _ = usage_client
    response = await client.post(
        "/v1/usage/prices",
        json={
            "price_id": "synthetic-v1",
            "provider": "synthetic-provider",
            "model": "synthetic-model",
            "effective_from": 0,
            "rates": {
                "input_tokens": "2",
                "cache_read_tokens": "0.2",
                "output_tokens": "10",
                "reasoning_tokens": "10",
                "cache_write_tokens": "0",
            },
        },
    )
    assert response.status_code == 200
    await repo.record(_observation())
    result = (await client.get("/v1/usage/overview")).json()
    assert float(result["summary"]["usd_equivalent"]) == 0.00292


@pytest.mark.parametrize("rate", ["NaN", "Infinity", "-1", "not-a-number"])
async def test_invalid_price_rates(usage_client, rate):
    client, _, _ = usage_client
    response = await client.post(
        "/v1/usage/prices",
        json={
            "price_id": "bad",
            "provider": "synthetic",
            "model": "synthetic",
            "effective_from": 0,
            "rates": {"input_tokens": rate},
        },
    )
    assert response.status_code == 422


async def test_source_breakdown_uses_all_filtered_facts_before_pagination(usage_client):
    client, repo, _ = usage_client
    await repo.add_price(
        UsagePrice(
            price_id="source-prices",
            provider="synthetic-provider",
            model="synthetic-model",
            effective_from=0,
            rates={"input_tokens": Decimal(1)},
        )
    )
    for index, source in enumerate(
        ("gateway", "native:codex", "native:claude", "native:agy", "native:opencode", "gateway"), 1
    ):
        await repo.record(
            replace(
                _observation(),
                source=source,
                source_key=str(index),
                fact_key=str(index),
                session_id=str(index),
                input_tokens=index * 1000000,
                output_tokens=0,
                reasoning_tokens=0,
                cache_read_tokens=0,
                total_tokens=index * 1000000,
            )
        )
    result = (await client.get("/v1/usage/overview?group_by=session&limit=1")).json()
    assert len(result["breakdown"]) == 1
    assert result["breakdown_total"] == 6
    sources = result["source_breakdown"]
    assert {key: value["usd_equivalent"] for key, value in sources.items()} == {
        "gateway": "7",
        "codex": "2",
        "claude": "3",
        "agy": "4",
        "other": "5",
    }
    assert sum(value["fact_count"] for value in sources.values()) == result["summary"]["fact_count"]
    assert sum(Decimal(value["usd_equivalent"]) for value in sources.values()) == Decimal(
        result["summary"]["usd_equivalent"]
    )
    scoped = (await client.get("/v1/usage/overview?session_id=2&group_by=model")).json()
    assert scoped["source_breakdown"]["codex"]["usd_equivalent"] == "2"
    assert scoped["source_breakdown"]["gateway"]["fact_count"] == 0
    outside = (await client.get("/v1/usage/overview?from_ms=1&to_ms=2")).json()
    assert all(value["fact_count"] == 0 for value in outside["source_breakdown"].values())


async def test_manual_refresh_waits_for_collection_and_allows_immediate_retries():
    app = create_app()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def refresh():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return True

    app.state.lifespan = LifespanState(usage_refresh=refresh)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        pending = asyncio.create_task(client.post("/v1/usage/refresh"))
        await entered.wait()
        assert not pending.done()
        release.set()
        response = await pending
        assert response.json() == {"status": "completed", "retry_after_ms": 0}
        response = await client.post("/v1/usage/refresh")
        assert response.status_code == 200
        assert calls == 2


async def test_refresh_failure_is_reported_instead_of_claiming_completion():
    app = create_app()

    async def refresh():
        raise RuntimeError("private native error")

    app.state.lifespan = LifespanState(usage_refresh=refresh)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/usage/refresh")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "usage_refresh_failed"
        assert "private native error" not in response.text
