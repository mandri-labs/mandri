from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class UsageObservation:
    source: str
    source_key: str
    fact_key: str
    session_id: str | None = None
    root_session_id: str | None = None
    project_path: str | None = None
    harness: str | None = None
    provider: str | None = None
    model: str | None = None
    billing_mode: str = "unknown"
    account_id: str | None = None
    native_session_id: str | None = None
    turn_id: str | None = None
    observed_model: str | None = None
    occurred_at: int | None = None
    observed_at: int | None = None
    interval_start: int | None = None
    coverage_end: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    native_total_tokens: int | None = None
    request_count: int | None = None
    kind: Literal["delta", "cumulative"] = "delta"
    epoch: str | None = None
    sequence: int = 0
    baseline: bool = False
    authoritative: bool = True
    non_overlapping: bool = False
    complete: bool = False
    input_includes_cache: bool | None = None
    output_includes_reasoning: bool | None = None
    reported_cost_usd: Decimal | None = None
    reported_cost_basis: str | None = None
    included_session_ids: tuple[str, ...] = ()
    pricing_context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class UsageFilters:
    session_id: str | None = None
    root_session_id: str | None = None
    project_path: str | None = None
    harness: str | None = None
    provider: str | None = None
    model: str | None = None
    billing_mode: str | None = None
    from_ms: int | None = None
    to_ms: int | None = None
    timezone: str = "UTC"
    include_deleted: bool = True
    include_descendants: bool = True
    group_by: Literal["session", "project", "model", "harness", "billing_mode"] = "model"
    limit: int = 100
    offset: int = 0
    expected_revision: int | None = None


@dataclass(frozen=True, kw_only=True)
class UsageAccount:
    account_id: str
    harness: str
    observed_at: int
    status: str = "unavailable"
    plan: str | None = None
    auth_mode: str | None = None
    verified: bool = False
    windows: tuple[dict[str, object], ...] = ()
    credits: Decimal | None = None
    monthly_fee_usd: Decimal | None = None

    @property
    def has_account_data(self) -> bool:
        return bool(
            self.plan
            or self.auth_mode
            or self.verified
            or self.windows
            or self.credits is not None
            or self.monthly_fee_usd is not None
        )


@dataclass(frozen=True, kw_only=True)
class UsagePrice:
    price_id: str
    provider: str
    model: str
    effective_from: int
    rates: dict[str, Decimal] = field(default_factory=dict)
    effective_to: int | None = None
    source: str = "explicit"
    calculation_version: str = "1"
    constraints: dict[str, object] = field(default_factory=dict)
    reviewed_at: int | None = None
    currency: str = "USD"
    unit: str = "per_million_tokens"
    valuation_basis: Literal["historical_tariff", "current_price_comparison"] = "historical_tariff"
    source_provider: str | None = None
    source_model: str | None = None
