from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal


@dataclass(frozen=True)
class NativeUsageContext:
    session_id: str
    harness: str
    process_epoch: str
    native_id: str | None = None
    project_path: str | None = None
    routing: Literal["native", "gateway", "unknown"] = "unknown"
    profile_id: str | None = None
    source_version: str | None = None
    parent_session_id: str | None = None
    inherited_history: Literal["none", "excluded", "unknown"] = "unknown"
    resumed: bool = False
    observed_model: str | None = None
    model_scope_proven: bool = False
    turn_id: str | None = None
    process_started_at: int | None = None
    billing_mode: str = "unknown"
    provider_kind: str | None = None


@dataclass(frozen=True)
class NativeUsageObservation:
    source_key: str
    series_key: str
    context: NativeUsageContext
    scope: str
    counters: Mapping[str, int | None]
    observed_at_ms: int
    occurred_at_ms: int | None = None
    native_id: str | None = None
    turn_id: str | None = None
    model: str | None = None
    cumulative: bool = True
    includes_descendants: bool | None = None
    authoritative: bool = False
    additive: bool = False
    completeness: str = "partial"
    anomalies: tuple[str, ...] = ()
    reported_cost_usd: Decimal | None = None
    origin: str = "live"
    observed_model: str | None = None
    source_position: int | None = None
    own_usage_proven: bool = False
    provider_kind: str | None = None
    observed_provider: str | None = None
    upstream_request_id: str | None = None


NativeUsageObserver = Callable[[NativeUsageContext, Mapping[str, Any]], Awaitable[None]]
NativeUsageSink = Callable[[NativeUsageObservation], Awaitable[None]]
