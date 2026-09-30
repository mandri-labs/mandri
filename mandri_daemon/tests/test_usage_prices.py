import asyncio
from collections.abc import Sequence
from copy import deepcopy
from decimal import Decimal

import httpx
import pytest
from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.core.usage_pricing import value_usage
from mandri.daemon import usage_prices
from mandri.daemon.usage_catalog_sources import MODELS_DEV_URL, OPENROUTER_URL
from mandri.daemon.usage_prices import CACHE_SECONDS, RETRY_SECONDS, PriceCatalogSync
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage

REVIEWED_AT = 1000

PAYLOADS = {
    MODELS_DEV_URL: {
        "openai": {
            "models": {
                "model": {
                    "id": "model",
                    "modalities": {"output": ["text"]},
                    "cost": {"input": "1.25", "output": 10},
                }
            }
        }
    },
    OPENROUTER_URL: {
        "data": [
            {
                "id": "vendor/model:free",
                "architecture": {"output_modalities": ["text"]},
                "pricing": {"prompt": "0", "completion": "0"},
            }
        ]
    },
}


class Repository:
    def __init__(self):
        self.prices = {}
        self.state = None
        self.writes = 0

    async def sync_catalog_prices(self, prices: Sequence[UsagePrice]) -> int:
        self.writes += 1
        old = self.prices
        self.prices = {}
        for price in prices:
            existing = old.get(price.price_id)
            assert existing is None or existing == price
            self.prices[price.price_id] = price
        return self.writes

    async def catalog_prices(self, source):
        return tuple(price for price in self.prices.values() if price.source == source)

    async def remove_bundled_prices(self):
        return self.writes

    async def catalog_state(self, source="public"):
        return deepcopy(self.state)

    async def save_catalog_state(self, value, source="public"):
        self.state = deepcopy(value)
        return self.writes


class Chunks(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b" " * 150
        yield b" " * 150


async def test_sync_uses_only_public_get_without_inherited_credentials_and_caches():
    seen = []

    def respond(request):
        seen.append(request)
        assert request.method == "GET"
        assert not request.content
        assert not request.url.query
        assert not {"authorization", "cookie", "x-api-key", "x-private"} & set(request.headers)
        return httpx.Response(200, json=PAYLOADS[str(request.url)])

    repository = Repository()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        auth=("private", "secret"),
        headers={"x-api-key": "secret", "x-private": "private"},
        cookies={"session": "secret"},
    ) as http:
        sync = PriceCatalogSync(repository, http, now=lambda: REVIEWED_AT / 1000 + 1)
        first, second = await asyncio.gather(sync.refresh(), sync.refresh())
        assert not first["cached"] and second["cached"]
        assert not first["errors"]
        assert first["price_count"] == 4
        assert {str(request.url) for request in seen} == set(PAYLOADS)
        assert repository.writes == 1
        restored = PriceCatalogSync(repository, http, now=lambda: REVIEWED_AT / 1000 + 2)
        after_restart = await restored.refresh()
        assert after_restart["cached"]
        assert after_restart["price_count"] == first["price_count"]
        assert len(seen) == 2
    assert all(
        price.valuation_basis == "current_price_comparison" for price in repository.prices.values()
    )


async def test_refresh_failures_keep_prior_prices_and_back_off_per_source():
    now = [REVIEWED_AT / 1000 + 1]
    fail = [False]
    seen = []

    def respond(request):
        seen.append(str(request.url))
        if fail[0] and str(request.url) == MODELS_DEV_URL:
            return httpx.Response(503)
        return httpx.Response(200, json=PAYLOADS[str(request.url)])

    repository = Repository()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        sync = PriceCatalogSync(repository, http, clock=lambda: now[0], now=lambda: now[0])
        first = await sync.refresh()
        now[0] += CACHE_SECONDS + 1
        fail[0] = True
        failed = await sync.refresh()
        assert failed["errors"] == {MODELS_DEV_URL: "HTTPStatusError"}
        assert (
            failed["sources"][MODELS_DEV_URL]["reviewed_at"]
            == first["sources"][MODELS_DEV_URL]["reviewed_at"]
        )
        assert failed["sources"][MODELS_DEV_URL]["stale"]
        assert (await sync.refresh())["cached"]
        assert len(seen) == 4
        now[0] += RETRY_SECONDS + 1
        fail[0] = False
        recovered = await sync.refresh()
        assert not recovered["errors"]
        assert len(seen) == 5


@pytest.mark.parametrize("mode", ["redirect", "invalid", "large_header", "large_body", "timeout"])
async def test_bounded_failures_never_invent_fallback_prices(monkeypatch, mode):
    monkeypatch.setattr(usage_prices, "MAX_RESPONSE_BYTES", 250)
    seen = []

    def respond(request):
        seen.append(str(request.url))
        if mode == "redirect":
            return httpx.Response(302, headers={"location": "https://untrusted.invalid/secret"})
        if mode == "invalid":
            return httpx.Response(200, content="not JSON")
        if mode == "large_header":
            return httpx.Response(200, headers={"content-length": "999"}, content="{}")
        if mode == "large_body":
            return httpx.Response(200, stream=Chunks())
        raise httpx.ReadTimeout("synthetic timeout")

    repository = Repository()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        result = await PriceCatalogSync(repository, http).refresh()
    assert len(result["errors"]) == 2
    assert result["price_count"] == 0
    assert set(seen) == set(PAYLOADS)
    assert not repository.prices


async def test_sync_persists_prices_and_freshness_in_real_repository(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "catalog.db")
    await db.migrate()
    await migrate_usage(db._require_connection())
    repository = UsageRepository(db)
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(200, json=PAYLOADS[str(request.url)])

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        first = await PriceCatalogSync(repository, http).refresh()
        second = await PriceCatalogSync(repository, http).refresh()
    assert first["price_count"] == second["price_count"]
    assert len(requests) == 2
    rows = await db.fetch_all("SELECT payload FROM usage_price")
    assert len(rows) == 4
    assert (await repository.catalog_state())["version"] == 2
    assert first["revision"] == second["revision"]
    await db.close()


async def test_json_decimal_rates_remain_exact():
    body = (
        b'{"openai":{"models":{"m":{"modalities":{"output":["text"]},'
        b'"cost":{"input":0.123456789123456789,"output":1}}}}}'
    )

    def respond(request):
        if str(request.url) == MODELS_DEV_URL:
            return httpx.Response(200, content=body)
        return httpx.Response(200, json=PAYLOADS[OPENROUTER_URL])

    repository = Repository()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        await PriceCatalogSync(repository, http).refresh()
    price = next(price for price in repository.prices.values() if price.model == "m")
    assert price.rates["input_tokens"] == Decimal("0.123456789123456789")


async def test_unchanged_prices_keep_immutable_versions_across_refresh_and_restart():
    now = [REVIEWED_AT / 1000 + 1]
    payloads = deepcopy(PAYLOADS)

    def respond(request):
        return httpx.Response(200, json=payloads[str(request.url)])

    repository = Repository()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        sync = PriceCatalogSync(repository, http, now=lambda: now[0], clock=lambda: now[0])
        first = await sync.refresh()
        original_ids = set(repository.prices)
        now[0] += CACHE_SECONDS + 1
        second = await sync.refresh()
        assert set(repository.prices) == original_ids
        assert repository.writes == 1
        assert (
            second["sources"][MODELS_DEV_URL]["reviewed_at"]
            == first["sources"][MODELS_DEV_URL]["reviewed_at"]
        )
        assert (
            second["sources"][MODELS_DEV_URL]["checked_at"]
            > first["sources"][MODELS_DEV_URL]["checked_at"]
        )
        now[0] += CACHE_SECONDS + 1
        restarted = PriceCatalogSync(repository, http, now=lambda: now[0], clock=lambda: now[0])
        await restarted.refresh()
        assert set(repository.prices) == original_ids
        now[0] += CACHE_SECONDS + 1
        payloads[MODELS_DEV_URL]["openai"]["models"]["model"]["cost"]["input"] = "2.5"
        changed = await restarted.refresh()
        assert len(repository.prices) == len(original_ids)
        assert changed["sources"][MODELS_DEV_URL]["reviewed_at"] == int(now[0] * 1000)


async def test_total_request_deadline_is_bounded(monkeypatch):
    monkeypatch.setattr(usage_prices, "FETCH_TIMEOUT_SECONDS", 0.01)

    async def respond(request):
        await asyncio.Event().wait()
        raise AssertionError("Response must time out")

    repository = Repository()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        result = await PriceCatalogSync(repository, http).refresh()
    assert set(result["errors"].values()) == {"TimeoutError"}
    assert result["price_count"] == 0


async def test_restart_offline_retains_only_downloaded_prices(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "catalog.db")
    await db.migrate()
    await migrate_usage(db._require_connection())
    repository = UsageRepository(db)
    now = [10.0]

    def respond(request):
        return httpx.Response(200, json=PAYLOADS[str(request.url)])

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            await PriceCatalogSync(repository, http, now=lambda: now[0]).refresh()
        initial = await repository.catalog_prices(MODELS_DEV_URL)
        now[0] += CACHE_SECONDS + 1

        def offline(request):
            raise httpx.ConnectError("offline")

        async with httpx.AsyncClient(transport=httpx.MockTransport(offline)) as http:
            result = await PriceCatalogSync(repository, http, now=lambda: now[0]).refresh()
        assert len(result["errors"]) == 2
        assert await repository.catalog_prices(MODELS_DEV_URL) == initial
        value = UsageObservation(
            source="native:codex",
            source_key="a",
            fact_key="a",
            model="model",
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            input_includes_cache=False,
            output_includes_reasoning=True,
            pricing_context={"comparison_basis": "standard_text_api"},
        )
        assert value_usage(value, initial)[0] == Decimal("0.000225")
    finally:
        await db.close()


async def test_legacy_embedded_prices_are_removed_without_erasing_user_or_public_prices(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "catalog.db")
    await db.migrate()
    await migrate_usage(db._require_connection())
    repository = UsageRepository(db)
    try:
        for identity, source, basis in [
            ("2026-01-01.v1:openai:model", "https://provider.invalid", "historical_tariff"),
            ("2026-01-01.v2:openai:model", "https://provider.invalid", "current_price_comparison"),
            ("current:go:100:model:peak", "https://provider.invalid", "current_price_comparison"),
            ("2026-01-01.v3:openai:model", "user_override", "current_price_comparison"),
            ("downloaded", MODELS_DEV_URL, "current_price_comparison"),
        ]:
            await repository.add_price(
                UsagePrice(
                    price_id=identity,
                    provider="openai",
                    model="model",
                    effective_from=0,
                    source=source,
                    valuation_basis=basis,
                    reviewed_at=1,
                    rates={"input_tokens": Decimal(7), "output_tokens": Decimal(11)},
                )
            )
        await repository.remove_bundled_prices()
        rows = await db.fetch_all("SELECT price_id FROM usage_price ORDER BY price_id")
        assert [row["price_id"] for row in rows] == ["2026-01-01.v3:openai:model", "downloaded"]
    finally:
        await db.close()


async def test_forced_refresh_discovers_new_models_within_cache_period():
    payloads = deepcopy(PAYLOADS)
    seen = []

    def respond(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=payloads[str(request.url)])

    repository = Repository()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        sync = PriceCatalogSync(repository, http)
        await sync.refresh()
        payloads[MODELS_DEV_URL]["openai"]["models"]["new-model"] = {
            "modalities": {"output": ["text"]},
            "cost": {"input": 7, "output": 11},
        }
        assert (await sync.refresh())["cached"]
        assert len(seen) == 2
        assert not (await sync.refresh(force=True))["cached"]
        assert len(seen) == 4
        assert any(price.model == "new-model" for price in repository.prices.values())


async def test_catalog_snapshot_retires_removed_prices_and_keeps_user_overrides(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "catalog.db")
    await db.migrate()
    await migrate_usage(db._require_connection())
    repo = UsageRepository(db)
    downloaded = UsagePrice(
        price_id="public-old",
        provider="openai",
        model="retired-model",
        effective_from=0,
        reviewed_at=1,
        valuation_basis="current_price_comparison",
        source=MODELS_DEV_URL,
        rates={"input_tokens": Decimal(7), "output_tokens": Decimal(11)},
    )
    try:
        await repo.add_price(downloaded)
        await repo.add_price(
            UsagePrice(
                price_id="custom",
                provider="openai",
                model="retired-model",
                effective_from=0,
                source="user_override",
                rates={"input_tokens": Decimal(17), "output_tokens": Decimal(21)},
            )
        )
        await repo.sync_catalog_prices([])
        assert not await repo.catalog_prices(MODELS_DEV_URL)
        rows = await db.fetch_all("SELECT price_id FROM usage_price")
        assert [row["price_id"] for row in rows] == ["custom"]
        with pytest.raises(ValueError, match="downloaded current prices"):
            await repo.sync_catalog_prices(
                [
                    UsagePrice(
                        price_id="bad",
                        provider="openai",
                        model="model",
                        effective_from=0,
                        rates={"input_tokens": Decimal(7)},
                    )
                ]
            )
        assert [
            row["price_id"] for row in await db.fetch_all("SELECT price_id FROM usage_price")
        ] == ["custom"]
    finally:
        await db.close()
