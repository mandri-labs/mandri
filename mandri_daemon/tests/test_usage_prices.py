import asyncio
from collections.abc import Sequence
from copy import deepcopy
from decimal import Decimal

import httpx
import pytest
from mandri.core.types.usage import UsagePrice
from mandri.core.usage_pricing import REVIEWED_AT, bundled_prices
from mandri.daemon import usage_prices
from mandri.daemon.usage_price_catalog import MODELS_DEV_URL, OPENROUTER_URL
from mandri.daemon.usage_prices import CACHE_SECONDS, RETRY_SECONDS, PriceCatalogSync
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage

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

    async def add_prices(self, prices: Sequence[UsagePrice]) -> int:
        self.writes += 1
        for price in prices:
            existing = self.prices.get(price.price_id)
            assert existing is None or existing == price
            self.prices[price.price_id] = price
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
        assert first["price_count"] == len(bundled_prices()) + 2
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
async def test_bounded_failures_still_persist_official_fallback(monkeypatch, mode):
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
    assert result["price_count"] == len(bundled_prices())
    assert set(seen) == set(PAYLOADS)
    assert any(price.model == "gpt-6-astra" for price in repository.prices.values())


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
    assert len(rows) == len(bundled_prices()) + 2
    assert (await repository.catalog_state())["version"] == 1
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
        assert len(repository.prices) == len(original_ids) + 1
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
    assert result["price_count"] == len(bundled_prices())
