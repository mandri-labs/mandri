import asyncio
import threading
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest
from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage
from mandri.database.usage_valuation import _compute


@pytest.fixture
async def repository(tmp_path):
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "usage.db")
    await database.migrate()
    await migrate_usage(database._require_connection())
    try:
        yield UsageRepository(database)
    finally:
        await database.close()


def observation(key: str = "request") -> UsageObservation:
    return UsageObservation(
        source="gateway",
        source_key=key,
        fact_key=key,
        session_id="session",
        provider="provider",
        model="model",
        input_tokens=100,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        request_count=1,
        occurred_at=1000,
        input_includes_cache=False,
        output_includes_reasoning=False,
    )


def tariff() -> UsagePrice:
    return UsagePrice(
        price_id="price",
        provider="provider",
        model="model",
        effective_from=0,
        rates={"input_tokens": Decimal("1")},
    )


async def test_valuation_does_not_overwrite_a_concurrent_observation(repository, monkeypatch):
    await repository.record(observation())
    await repository.add_price(tariff())
    started, release = threading.Event(), threading.Event()

    def compute(rows):
        started.set()
        assert release.wait(3)
        return _compute(rows)

    monkeypatch.setattr("mandri.database.usage_valuation._compute", compute)
    task = asyncio.create_task(repository.revalue_unpriced())
    try:
        async with asyncio.timeout(1):
            while not started.is_set():
                await asyncio.sleep(0.001)
        async with asyncio.timeout(1):
            await repository.record(replace(observation(), sequence=1, input_tokens=200))
    finally:
        release.set()
    assert (await task)["updated"] == 0
    summary = (await repository.overview())["summary"]
    assert summary["input_tokens"] == 200
    assert summary["usd_equivalent"] == "0.0002"


async def test_completed_valuation_does_not_restart_and_new_tariffs_invalidate_it(repository):
    for key in ("a", "b"):
        await repository.record(observation(key))
    await repository.add_price(tariff())
    assert (await repository.revalue_pending(limit=1))["next_key"] == "a"
    restarted = UsageRepository(repository._database)
    assert (await restarted.revalue_pending(limit=1))["next_key"] == "b"
    assert (await restarted.revalue_pending(limit=1))["next_key"] is None
    assert (await restarted.revalue_pending())["updated"] == 0
    await restarted.add_price(
        replace(tariff(), price_id="next", effective_from=500, rates={"input_tokens": Decimal("2")})
    )
    assert (await restarted.revalue_pending())["updated"] == 2
    assert (await restarted.overview())["summary"]["usd_equivalent"] == "0.0004"


async def test_valuation_does_not_repeat_coverage_or_load_unrelated_tariffs(
    repository, monkeypatch
):
    await repository.add_prices(
        [replace(tariff(), price_id=str(i), model=f"unrelated-{i}") for i in range(200)]
    )
    for index in range(100):
        await repository.record(observation(str(index)))
    await repository.add_price(tariff())

    def unexpected(*args, **kwargs):
        raise AssertionError("Coverage belongs to ingestion, not price valuation")

    monkeypatch.setattr("mandri.database.usage.select_coverage", unexpected)
    result = await repository.revalue_unpriced(limit=20)
    assert result["updated"] == 20
    assert result["next_key"] is not None
    plan = await repository._database.fetch_all(
        "EXPLAIN QUERY PLAN SELECT payload FROM usage_price WHERE model=? AND provider IN (?,?)",
        ("model", "provider", "provider"),
    )
    assert any("ix_usage_price_model" in row["detail"] for row in plan)
    plan = await repository._database.fetch_all(
        "EXPLAIN QUERY PLAN SELECT fact_key FROM usage_fact"
        " WHERE price_id IS NULL AND fact_key>? ORDER BY fact_key LIMIT ?",
        ("", 20),
    )
    assert any("ix_usage_unpriced" in row["detail"] for row in plan)


async def test_aggregation_keeps_large_counters_and_decimal_fractions_exact(repository):
    amounts = [Decimal("1e60"), Decimal("0.000000000000000001"), Decimal("2.5")]
    for index, amount in enumerate(amounts):
        await repository.record(
            replace(
                observation(str(index)),
                model=f"model-{index}",
                input_tokens=10**40,
                occurred_at=1000 + index * 86400000,
                reported_cost_usd=amount,
            )
        )
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 3 * 10**40
    with localcontext() as context:
        context.prec = 100
        assert Decimal(result["summary"]["reported_cost_usd"]) == sum(amounts)
    assert result["summary"]["fact_count"] == 3
    assert len(result["breakdown"]) == len(result["timeseries"]) == 3
