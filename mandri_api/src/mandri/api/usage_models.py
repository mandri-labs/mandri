from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UsageMetricsOut(BaseModel):
    fact_count: int
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    request_count: int | None
    usd_equivalent: str | None
    reported_cost_usd: str | None
    missing_fields: dict[str, int]
    unpriced_fact_count: int
    incomplete_fact_count: int
    unclassified_fact_count: int = 0
    valuation_bases: dict[str, int] = Field(default_factory=dict)
    unpriced_reasons: dict[str, int] = Field(default_factory=dict)


class UsageBucketOut(UsageMetricsOut):
    date: str


class UsageGroupOut(UsageMetricsOut):
    key: str | None
    deleted: bool = False


class UsageSyncStateOut(BaseModel):
    scope: str = "daemon"
    status: str = "unavailable"
    source_count: int = 0
    gap_count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    discarded_event_count: int = 0
    discard_reasons: dict[str, int] = Field(default_factory=dict)


class UsageOverviewOut(BaseModel):
    revision: int
    as_of: int
    last_observed_at: int | None = None
    history_status: dict[str, int] = Field(default_factory=dict)
    sync_state: UsageSyncStateOut = Field(default_factory=UsageSyncStateOut)
    timezone: str
    summary: UsageMetricsOut
    source_breakdown: dict[str, UsageMetricsOut] = Field(default_factory=dict)
    timeseries: list[UsageBucketOut]
    breakdown: list[UsageGroupOut]
    breakdown_total: int
    undated: UsageMetricsOut
    unallocated: UsageMetricsOut
    catalog: dict[str, dict[str, object]] = Field(default_factory=dict)


class UsageAccountOut(BaseModel):
    account_id: str
    harness: str
    observed_at: int
    status: str
    plan: str | None = None
    auth_mode: str | None = None
    verified: bool = False
    windows: list[dict[str, str | int | float | bool | None]] = Field(default_factory=list)
    credits: str | None = None
    monthly_fee_usd: str | None = None


class UsageAccountsOut(BaseModel):
    accounts: list[UsageAccountOut]
    as_of: int
    revision: int


class UsageCapabilityOut(BaseModel):
    harness: str
    live: str
    history: str
    quotas: str
    detail: str


class UsageCapabilitiesOut(BaseModel):
    capabilities: list[UsageCapabilityOut]
    as_of: int


class UsageRefreshOut(BaseModel):
    status: Literal["queued", "completed", "unavailable"]
    retry_after_ms: int


class UsageEraseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed: bool = False


class UsageEraseOut(BaseModel):
    revision: int


class UsagePriceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price_id: str = Field(min_length=1, max_length=200)
    provider: str = Field(min_length=1, max_length=200)
    model: str = Field(min_length=1, max_length=200)
    effective_from: int = Field(ge=0)
    effective_to: int | None = Field(default=None, ge=0)
    rates: dict[str, str]


class UsagePriceOut(BaseModel):
    revision: int
