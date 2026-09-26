import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Protocol

import httpx
from mandri.core.types.usage import UsagePrice
from mandri.core.usage_pricing import bundled_prices
from mandri.daemon.usage_price_catalog import (
    MODELS_DEV_URL,
    OPENROUTER_URL,
    parse_models_dev,
    parse_openrouter,
)

CACHE_SECONDS = 6 * 60 * 60
RETRY_SECONDS = 5 * 60
FETCH_TIMEOUT_SECONDS = 15
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class PriceRepository(Protocol):
    async def add_prices(self, prices: Sequence[UsagePrice]) -> int: ...

    async def catalog_state(self, source: str = "public") -> dict[str, Any] | None: ...

    async def save_catalog_state(self, value: dict[str, Any], source: str = "public") -> int: ...


@dataclass
class _Snapshot:
    prices: tuple[UsagePrice, ...] = ()
    retry_at: float = 0
    error: str | None = None
    price_count: int = 0
    reviewed_at: int | None = None
    checked_at: int | None = None
    fingerprint: str | None = None


class PriceCatalogSync:
    def __init__(
        self,
        repository: PriceRepository,
        http: httpx.AsyncClient | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._repository = repository
        self._http = http
        self._clock = clock
        self._now = now
        self._lock = asyncio.Lock()
        self._sources = {MODELS_DEV_URL: _Snapshot(), OPENROUTER_URL: _Snapshot()}
        self._persisted: tuple[UsagePrice, ...] = ()
        self._revision = 0
        self._loaded = False

    async def refresh(self) -> dict[str, object]:
        async with self._lock:
            if not self._loaded:
                await self._restore()
                self._loaded = True
            pending = [
                url for url, state in self._sources.items() if self._clock() >= state.retry_at
            ]
            if pending:
                if self._http is None:
                    async with httpx.AsyncClient(trust_env=False) as http:
                        await asyncio.gather(*(self._fetch(http, url) for url in pending))
                else:
                    await asyncio.gather(*(self._fetch(self._http, url) for url in pending))
            prices = (
                *bundled_prices(),
                *(price for state in self._sources.values() for price in state.prices),
            )
            if prices != self._persisted:
                self._revision = await self._repository.add_prices(prices)
                self._persisted = prices
            if pending:
                self._revision = await self._repository.save_catalog_state(self._metadata())
            return {
                "revision": self._revision,
                "price_count": len(bundled_prices())
                + sum(state.price_count for state in self._sources.values()),
                "cached": not pending,
                "valuation_basis": "current_price_comparison",
                "sources": {
                    url: {
                        "price_count": state.price_count,
                        "reviewed_at": state.reviewed_at,
                        "checked_at": state.checked_at,
                        "cached": url not in pending,
                        "stale": state.error is not None,
                    }
                    for url, state in self._sources.items()
                },
                "errors": {url: state.error for url, state in self._sources.items() if state.error},
            }

    async def _restore(self) -> None:
        saved = await self._repository.catalog_state()
        if not saved or saved.get("version") != 1:
            return
        sources = saved.get("sources")
        if not isinstance(sources, dict):
            return
        for url, state in self._sources.items():
            metadata = sources.get(url)
            if not isinstance(metadata, dict):
                continue
            retry_at = metadata.get("retry_at")
            count = metadata.get("price_count")
            reviewed_at = metadata.get("reviewed_at")
            if type(retry_at) is not int or type(count) is not int or not 0 <= count <= 5000:
                continue
            if reviewed_at is not None and (type(reviewed_at) is not int or reviewed_at < 0):
                continue
            remaining = min(CACHE_SECONDS, max(0, retry_at / 1000 - self._now()))
            state.retry_at = self._clock() + remaining
            state.price_count = count
            state.reviewed_at = reviewed_at
            checked_at = metadata.get("checked_at")
            state.checked_at = checked_at if type(checked_at) is int else None
            fingerprint = metadata.get("fingerprint")
            state.fingerprint = fingerprint if isinstance(fingerprint, str) else None
            error = metadata.get("error")
            state.error = error if isinstance(error, str) else None

    def _metadata(self) -> dict[str, Any]:
        return {
            "version": 1,
            "sources": {
                url: {
                    "price_count": state.price_count,
                    "reviewed_at": state.reviewed_at,
                    "checked_at": state.checked_at,
                    "fingerprint": state.fingerprint,
                    "error": state.error,
                    "retry_at": int((self._now() + state.retry_at - self._clock()) * 1000),
                }
                for url, state in self._sources.items()
            },
        }

    async def _fetch(self, http: httpx.AsyncClient, url: str) -> None:
        state = self._sources[url]
        try:
            async with asyncio.timeout(FETCH_TIMEOUT_SECONDS):
                payload = await _public_json(http, url)
            parse = parse_models_dev if url == MODELS_DEV_URL else parse_openrouter
            prices = parse(payload, int(self._now() * 1000))
        except (httpx.HTTPError, TimeoutError, ValueError, RecursionError) as error:
            state.error = type(error).__name__
            state.retry_at = self._clock() + RETRY_SECONDS
            return
        fingerprint = _fingerprint(prices)
        if fingerprint != state.fingerprint:
            state.prices = prices
            state.price_count = len(prices)
            state.reviewed_at = prices[0].reviewed_at
            state.fingerprint = fingerprint
        state.checked_at = int(self._now() * 1000)
        state.error = None
        state.retry_at = self._clock() + CACHE_SECONDS


def _fingerprint(prices: Sequence[UsagePrice]) -> str:
    rows = [
        json.dumps(
            {
                key: value
                for key, value in asdict(price).items()
                if key not in {"price_id", "reviewed_at"}
            },
            sort_keys=True,
            default=str,
        )
        for price in prices
    ]
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


async def _public_json(http: httpx.AsyncClient, url: str) -> object:
    request = httpx.Request(
        "GET",
        url,
        headers={"Accept": "application/json", "Accept-Encoding": "identity"},
        extensions={"timeout": httpx.Timeout(FETCH_TIMEOUT_SECONDS).as_dict()},
    )
    response = await http.send(request, auth=None, follow_redirects=False, stream=True)
    try:
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length is not None and int(length) > MAX_RESPONSE_BYTES:
            raise ValueError("Catalog response exceeds byte limit")
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                raise ValueError("Catalog response exceeds byte limit")
            body.extend(chunk)
        return json.loads(body, parse_float=Decimal)
    finally:
        await response.aclose()
