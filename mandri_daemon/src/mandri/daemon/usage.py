import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from mandri.core.hub import Hub, Topic
from mandri.core.types.usage import UsageAccount, UsageObservation
from mandri.database.usage import UsageRepository
from mandri.gateway.usage import GatewayUsageRecord
from mandri.gateway.usage_observation import to_observation

logger = logging.getLogger(__name__)


class UsageCoordinator:
    def __init__(self, repository: UsageRepository, hub: Hub) -> None:
        self.repository = repository
        self._hub = hub
        self._wake = asyncio.Event()
        self._next_accounts = 0.0
        self._accounts_task: asyncio.Future[None] | None = None
        self._revision = 0
        self._coverage_revision = -1
        self._published = 0
        self._notification: asyncio.TimerHandle | None = None
        self.reconcile: Callable[[], Awaitable[bool | None]] | None = None
        self.refresh_accounts: Callable[[], Awaitable[None]] | None = None
        self.refresh_prices: Callable[[], Awaitable[dict[str, Any]]] | None = None
        self.force_refresh_prices: Callable[[], Awaitable[dict[str, Any]]] | None = None

    async def record(self, observation: UsageObservation) -> None:
        self._changed(await self.repository.record(observation))

    async def account(self, account: UsageAccount) -> None:
        self._changed(await self.repository.upsert_account(account))

    async def gateway(self, record: GatewayUsageRecord) -> None:
        await self.record(to_observation(record))

    def _changed(self, revision: int) -> None:
        self._revision = max(self._revision, revision)
        if self._revision > self._published and self._notification is None:
            self._notification = asyncio.get_running_loop().call_later(0.25, self._publish)

    def _publish(self) -> None:
        self._notification = None
        if self._revision > self._published:
            self._hub.publish(Topic("usage.changed"), {"revision": self._revision})
            self._published = self._revision

    async def refresh(self) -> bool:
        if self.force_refresh_prices is not None:
            await self.force_refresh_prices()
        self._wake.set()
        await self._collect_accounts()
        return True

    async def _collect_accounts(self) -> None:
        if self.refresh_accounts is None:
            return
        if self._accounts_task is None or self._accounts_task.done():
            self._next_accounts = time.monotonic() + 300
            self._accounts_task = asyncio.ensure_future(self.refresh_accounts())
        await asyncio.shield(self._accounts_task)

    async def run(self) -> None:
        while True:
            self._wake.clear()
            busy = False
            if self.refresh_prices is not None:
                try:
                    await self.refresh_prices()
                except Exception:
                    logger.exception("Usage price refresh failed; stored tariffs retained")
            if self.refresh_accounts is not None and time.monotonic() >= self._next_accounts:
                try:
                    await self._collect_accounts()
                except Exception:
                    logger.warning("Native quota refresh failed; previous readings retained")
            if self.reconcile is not None:
                try:
                    busy = bool(await self.reconcile())
                except Exception:
                    logger.exception(
                        "Usage reconciliation failed; retained metrics remain available"
                    )
            try:
                if await self.repository.revision() != self._coverage_revision:
                    self._coverage_revision = await self.repository.reconcile_gateway_history()
                valuation = await self.repository.revalue_pending()
                busy = busy or valuation["next_key"] is not None
                self._changed(await self.repository.revision())
            except Exception:
                logger.warning("Usage valuation failed; unpriced quantities remain available")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=0.25 if busy else 5)

    def close(self) -> None:
        if self._accounts_task is not None:
            self._accounts_task.cancel()
        if self._notification is not None:
            self._notification.cancel()
            self._notification = None
        self._publish()
