from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Query, Request
from mandri.api.deps import Database, Usage, app_state
from mandri.api.errors import ApiError
from mandri.api.usage_models import (
    UsageAccountsOut,
    UsageCapabilitiesOut,
    UsageCapabilityOut,
    UsageEraseIn,
    UsageEraseOut,
    UsageOverviewOut,
    UsagePriceIn,
    UsagePriceOut,
    UsageRefreshOut,
)
from mandri.core.clock import system_now_ms
from mandri.core.hub import Topic
from mandri.core.types.usage import UsageFilters, UsagePrice

router = APIRouter(prefix="/usage", tags=["usage"])


def usage_filters(
    session_id: str | None = None,
    root_session_id: str | None = None,
    project_path: str | None = None,
    harness: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    billing_mode: str | None = None,
    from_ms: int | None = Query(default=None, ge=0),
    to_ms: int | None = Query(default=None, ge=0),
    timezone: str = "UTC",
    include_deleted: bool = True,
    include_descendants: bool = True,
    group_by: Literal["session", "project", "model", "harness", "billing_mode"] = "model",
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    expected_revision: int | None = Query(default=None, ge=0),
) -> UsageFilters:
    if from_ms is not None and to_ms is not None and from_ms >= to_ms:
        raise ApiError("validation_error", "The time range must have a positive duration", 422)
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ApiError("validation_error", "Unknown timezone", 422) from None
    return UsageFilters(
        session_id=session_id,
        root_session_id=root_session_id,
        project_path=project_path,
        harness=harness,
        provider=provider,
        model=model,
        billing_mode=billing_mode,
        from_ms=from_ms,
        to_ms=to_ms,
        timezone=timezone,
        include_deleted=include_deleted,
        include_descendants=include_descendants,
        group_by=group_by,
        limit=limit,
        offset=offset,
        expected_revision=expected_revision,
    )


@router.get("/overview", operation_id="usage_overview")
async def overview(
    repository: Usage, filters: Annotated[UsageFilters, Depends(usage_filters)]
) -> UsageOverviewOut:
    try:
        result = await repository.overview(filters)
    except ValueError as error:
        raise ApiError("usage_query_invalid", str(error), 409) from None
    return UsageOverviewOut.model_validate(result)


@router.get("/accounts", operation_id="usage_accounts")
async def accounts(repository: Usage) -> UsageAccountsOut:
    snapshot = await repository.account_snapshot()
    for account in snapshot["accounts"]:
        if account["status"] == "available" and snapshot["as_of"] - account["observed_at"] > 300000:
            account["status"] = "stale"
    return UsageAccountsOut.model_validate(snapshot)


@router.get("/capabilities", operation_id="usage_capabilities")
async def capabilities() -> UsageCapabilitiesOut:
    return UsageCapabilitiesOut(
        as_of=system_now_ms(),
        capabilities=[
            UsageCapabilityOut(
                harness="gateway",
                live="available",
                history="live_from_collection",
                quotas="not_applicable",
                detail="Provider-reported request usage; missing metadata remains unknown.",
            ),
            UsageCapabilityOut(
                harness="codex",
                live="native_events",
                history="partial",
                quotas="native_query",
                detail="Independent native quota queries and per-turn host request history.",
            ),
            UsageCapabilityOut(
                harness="claude",
                live="native_events",
                history="partial",
                quotas="native_query",
                detail="Independent native quota queries and native quota events.",
            ),
            UsageCapabilityOut(
                harness="agy",
                live="native_events",
                history="unsupported",
                quotas="native_query",
                detail="Independent native /usage quota queries and native process counters.",
            ),
            UsageCapabilityOut(
                harness="pi",
                live="native_events",
                history="partial",
                quotas="provider_dependent",
                detail="Independent OAuth quota queries for supported native providers.",
            ),
            UsageCapabilityOut(
                harness="opencode",
                live="gateway_only",
                history="partial",
                quotas="unsupported",
                detail="Host SQLite assistant usage; gateway coverage takes priority.",
            ),
        ],
    )


@router.post("/refresh", operation_id="refresh_usage")
async def refresh(request: Request) -> UsageRefreshOut:
    state = app_state(request.app)
    if state is None or state.usage_refresh is None:
        return UsageRefreshOut(status="unavailable", retry_after_ms=0)
    try:
        await state.usage_refresh()
    except Exception:
        raise ApiError(
            "usage_refresh_failed", "Native quotas could not be refreshed.", 503
        ) from None
    return UsageRefreshOut(status="completed", retry_after_ms=0)


def _publish(request: Request, revision: int, session_id: str | None = None) -> None:
    state = app_state(request.app)
    if state is not None and state.hub is not None:
        state.hub.publish(Topic("usage.changed"), {"revision": revision, "session_id": session_id})


@router.post("/sessions/{session_id}/erase", operation_id="erase_session_usage")
async def erase(
    session_id: str,
    body: UsageEraseIn,
    repository: Usage,
    database: Database,
    request: Request,
) -> UsageEraseOut:
    if not body.confirmed:
        raise ApiError("confirmation_required", "Explicit usage erasure confirmation required", 422)
    session = await database.fetch_one("SELECT state FROM session WHERE id=?", (session_id,))
    if session is not None and session["state"] == "live":
        raise ApiError("session_running", "Stop the session before erasing its usage", 409)
    try:
        revision = await repository.erase_session(session_id)
    except ValueError as error:
        raise ApiError("usage_erasure_scope", str(error), 409) from None
    _publish(request, revision, session_id)
    return UsageEraseOut(revision=revision)


@router.post("/prices", operation_id="add_usage_price")
async def add_price(body: UsagePriceIn, repository: Usage, request: Request) -> UsagePriceOut:
    allowed = {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    }
    try:
        rates = {key: Decimal(value) for key, value in body.rates.items()}
        if not rates or rates.keys() - allowed:
            raise ValueError("Rates must use supported disjoint token dimensions")
        if any(not value.is_finite() or value < 0 for value in rates.values()):
            raise ValueError("Rates must be finite nonnegative USD per million tokens")
        if body.effective_to is not None and body.effective_to <= body.effective_from:
            raise ValueError("The price effective interval must have a positive duration")
        revision = await repository.add_price(
            UsagePrice(
                price_id=body.price_id,
                provider=body.provider,
                model=body.model,
                effective_from=body.effective_from,
                effective_to=body.effective_to,
                rates=rates,
                source="user_override",
            )
        )
    except (ValueError, InvalidOperation) as error:
        raise ApiError("invalid_usage_price", str(error), 422) from None
    _publish(request, revision)
    return UsagePriceOut(revision=revision)
