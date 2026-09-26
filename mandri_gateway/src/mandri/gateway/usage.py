import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Literal, Protocol

import anyio
from mandri.core.clock import system_now_ms
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.usage_attribution import UsageAttribution
from mandri.gateway.usage_identity import billing_mode

logger = logging.getLogger(__name__)
PERSISTENCE_TIMEOUT_SECONDS = 0.5
UsageStatus = Literal["pending", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class GatewayUsageRecord:
    request_id: str
    route_id: str
    provider_name: str
    provider_kind: str
    selected_model: str
    protocol: str
    started_at: int
    updated_at: int
    session_id: str | None = None
    root_session_id: str | None = None
    project_path: str | None = None
    harness: str | None = None
    billing_mode: str = "unknown"
    modality: str | None = None
    service_tier: str | None = None
    observed_model: str | None = None
    observed_provider: str | None = None
    requested_model: str | None = None
    upstream_protocol: str | None = None
    upstream_request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    provider_cost: Decimal | None = None
    provider_cost_currency: str | None = None
    raw_usage_json: str = "{}"
    usage_protocol: str | None = None
    status: UsageStatus = "pending"
    revision: int = 0
    transport_attempts: int = 0
    incomplete: bool = True
    output_observed: bool = False


class GatewayUsageSink(Protocol):
    async def __call__(self, record: GatewayUsageRecord) -> None: ...


class UsageCollector:
    def __init__(
        self,
        route: ResolvedRoute,
        protocol: str,
        sink: GatewayUsageSink,
        attribution: UsageAttribution | None = None,
        *,
        modality: str | None = None,
        service_tier: str | None = None,
    ):
        attribution = attribution or UsageAttribution()
        now = int(system_now_ms())
        self.record = GatewayUsageRecord(
            request_id=uuid.uuid4().hex,
            route_id=str(route.route_id),
            provider_name=route.provider.name,
            provider_kind=route.provider.kind.value,
            selected_model=str(route.model.model_ref),
            protocol=protocol,
            started_at=now,
            updated_at=now,
            root_session_id=attribution.root_session_id,
            project_path=attribution.project_path,
            harness=attribution.harness,
            modality=modality,
            service_tier=service_tier,
            billing_mode=billing_mode(route.provider.kind),
        )
        self.sink = sink
        self.finished = False
        self.raw_usage: dict[str, Any] = {}
        self.observation_incomplete = False
        self.upstream_failed = False
        self.persistence_disabled = False

    async def publish(self, **changes: Any) -> None:
        self.record = replace(
            self.record,
            **changes,
            revision=self.record.revision + 1,
            updated_at=int(system_now_ms()),
        )
        if self.persistence_disabled:
            return
        try:
            with anyio.move_on_after(PERSISTENCE_TIMEOUT_SECONDS, shield=True) as timeout:
                await self.sink(self.record)
            if timeout.cancelled_caught:
                self.persistence_disabled = True
                self.observation_incomplete = True
                logger.warning("Gateway usage persistence timed out for %s", self.record.request_id)
        except Exception:
            logger.warning(
                "Gateway usage persistence failed for request %s", self.record.request_id
            )

    async def finish(self, status: UsageStatus) -> None:
        if self.finished:
            return
        self.finished = True
        if status == "completed" and self.upstream_failed:
            status = "failed"
        await self.publish(
            status=status,
            incomplete=(
                status != "completed"
                or self.observation_incomplete
                or self.record.input_tokens is None
                or self.record.output_tokens is None
                or self.record.transport_attempts != 1
            ),
        )


CURRENT_USAGE: ContextVar[UsageCollector | None] = ContextVar("gateway_usage", default=None)


@contextmanager
def usage_scope(collector: UsageCollector) -> Any:
    token = CURRENT_USAGE.set(collector)
    try:
        yield
    finally:
        CURRENT_USAGE.reset(token)


class UsageStream(AsyncIterator[Any]):
    def __init__(self, source: AsyncIterator[Any], collector: UsageCollector):
        self.source = source
        self.collector = collector
        self.closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    async def __anext__(self) -> Any:
        try:
            with usage_scope(self.collector):
                return await anext(self.source)
        except StopAsyncIteration:
            await self.collector.finish("completed")
            await self.aclose()
            raise
        except BaseException as error:
            await self.collector.finish("failed" if isinstance(error, Exception) else "cancelled")
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        with anyio.CancelScope(shield=True):
            try:
                close = getattr(self.source, "aclose", None)
                if close is not None:
                    await close()
            finally:
                await self.collector.finish("cancelled")


async def collect_call(collector: UsageCollector, call: Callable[[], Awaitable[Any]]) -> Any:
    await collector.publish()
    try:
        with usage_scope(collector):
            result = await call()
    except BaseException as error:
        await collector.finish("failed" if isinstance(error, Exception) else "cancelled")
        raise
    if isinstance(result, AsyncIterator):
        return UsageStream(result, collector)
    await collector.finish("completed")
    return result
