import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from mandri.core.ids import HarnessKind
from mandri.core.ports.transcripts import SessionRef
from mandri.core.types.usage import UsageObservation
from mandri.sessions.transcripts.errors import TranscriptStoreError
from mandri.sessions.transcripts.opencode_transcripts import (
    SQLITE_TIMEOUT_S,
    OpencodeTranscriptReader,
)


@dataclass(frozen=True)
class OpencodeUsageBatch:
    observations: tuple[UsageObservation, ...]
    cursor: int | None
    status: str
    has_more: bool = False
    records_read: int = 0


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


def _identity(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() and len(value) <= 512 else None


def _cost(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def opencode_usage_observation(
    info: dict[str, Any],
    *,
    session_id: str,
    native_id: str,
    observed_at_ms: int,
    session_created_at_ms: int,
    root_session_id: str | None = None,
    project_path: str | None = None,
    billing_mode: str = "unknown",
    routing: str = "native",
    history_request: bool = False,
) -> UsageObservation | None:
    if info.get("role") != "assistant" or info.get("sessionID", native_id) != native_id:
        return None
    message_id = _identity(info.get("id"))
    provider = _identity(info.get("providerID"))
    model = _identity(info.get("modelID"))
    tokens = info.get("tokens")
    times = info.get("time")
    if not message_id or not provider or not model or not isinstance(tokens, dict):
        return None
    if not isinstance(times, dict):
        return None
    created = _count(times.get("created"))
    completed = _count(times.get("completed"))
    if created is None or created < session_created_at_ms:
        return None
    if completed is not None and completed < created:
        return None
    cache = tokens.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    counters = {
        "input_tokens": _count(tokens.get("input")),
        "output_tokens": _count(tokens.get("output")),
        "reasoning_tokens": _count(tokens.get("reasoning")),
        "cache_read_tokens": _count(cache.get("read")),
        "cache_write_tokens": _count(cache.get("write")),
    }
    cost = _cost(info.get("cost"))
    if not any(value is not None for value in counters.values()) and cost is None:
        return None
    provider_kind = {"google": "gemini", "opencode-go": "opencode_go"}.get(provider, provider)
    mode = billing_mode
    if mode == "unknown":
        if provider_kind == "opencode_go":
            mode = "subscription"
        elif provider_kind in {"ollama", "lmstudio", "lm_studio"}:
            mode = "local"
        elif provider_kind in {"openrouter", "opencode", "google", "gemini"}:
            mode = "api"
    context_tokens = (
        sum(
            counters[key] or 0
            for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens")
        )
        if all(
            counters[key] is not None
            for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens")
        )
        else None
    )
    reasoning = counters["reasoning_tokens"]
    output = counters["output_tokens"]
    total = _count(tokens.get("total"))
    output_includes_reasoning = False if reasoning == 0 else None
    if context_tokens is not None and output is not None and total is not None:
        if reasoning is not None and total == context_tokens + output + reasoning:
            output_includes_reasoning = False
        elif reasoning is not None and output >= reasoning and total == context_tokens + output:
            output_includes_reasoning = True
    identity = hashlib.sha256(json.dumps([native_id, message_id]).encode()).hexdigest()
    return UsageObservation(
        source="native:opencode",
        source_key=identity,
        fact_key=f"native:opencode:{identity}",
        epoch=f"opencode:message:{identity}",
        session_id=session_id,
        root_session_id=root_session_id,
        native_session_id=native_id,
        project_path=project_path,
        harness="opencode",
        provider=provider,
        model=model,
        observed_model=model,
        billing_mode=mode,
        occurred_at=completed if completed is not None else created,
        interval_start=created,
        observed_at=observed_at_ms,
        sequence=observed_at_ms,
        request_count=1,
        authoritative=routing == "native",
        non_overlapping=False,
        complete=(
            completed is not None
            and info.get("error") is None
            and all(value is not None for value in counters.values())
        ),
        input_includes_cache=False,
        output_includes_reasoning=output_includes_reasoning,
        total_tokens=total,
        native_total_tokens=total,
        reported_cost_usd=cost,
        reported_cost_basis="harness_estimate" if cost is not None else None,
        pricing_context={
            "comparison_basis": "standard_text_api",
            "provider_kind": provider_kind,
            "model_basis": "native_message",
            "counter_basis": "opencode_assistant_message",
            "reasoning_overlap_basis": (
                "zero_reasoning"
                if reasoning == 0
                else "reconciled_total"
                if output_includes_reasoning is not None
                else "unknown"
            ),
            "cost_basis": "opencode_model_catalog",
            "message_id": message_id,
            "message_created_at": created,
            "message_completed_at": completed,
            "routing": routing,
            "history_request": history_request,
            "status": "failed"
            if info.get("error") is not None
            else ("completed" if completed is not None else "pending"),
            **({"context_tokens": context_tokens} if context_tokens is not None else {}),
        },
        input_tokens=counters["input_tokens"],
        output_tokens=counters["output_tokens"],
        reasoning_tokens=counters["reasoning_tokens"],
        cache_read_tokens=counters["cache_read_tokens"],
        cache_write_tokens=counters["cache_write_tokens"],
    )


class OpencodeUsageReader(OpencodeTranscriptReader):
    def read_usage(
        self,
        session: SessionRef,
        *,
        session_id: str,
        observed_at_ms: int,
        cursor: int | None = None,
        max_records: int = 200,
        root_session_id: str | None = None,
        billing_mode: str = "unknown",
        routing: str = "native",
    ) -> OpencodeUsageBatch:
        if type(max_records) is not int or not 1 <= max_records <= 1000:
            raise ValueError("Invalid OpenCode usage scan bound")
        if cursor is not None and (type(cursor) is not int or cursor <= 0):
            raise ValueError("Invalid OpenCode usage cursor")
        if _count(observed_at_ms) is None:
            raise ValueError("Invalid OpenCode observation time")
        if session.harness is not HarnessKind.OPENCODE:
            return OpencodeUsageBatch((), cursor, "unsupported")
        try:
            created = self._session_created(str(session.native_id))
            if created is None:
                return OpencodeUsageBatch((), cursor, "unavailable")
            rows = self._read_rows(
                str(session.native_id), cursor, max_records + 1, metadata_only=True
            )
        except (OSError, sqlite3.Error, TranscriptStoreError):
            return OpencodeUsageBatch((), cursor, "unavailable")
        has_more = len(rows) > max_records
        rows = rows[:max_records]
        observations = []
        status = "ready"
        for _, raw, message_id in rows:
            if raw is None:
                status = "partial"
                continue
            try:
                info = json.loads(str(raw), parse_float=Decimal)
            except (ValueError, RecursionError):
                status = "partial"
                continue
            if not isinstance(info, dict):
                status = "partial"
                continue
            item = opencode_usage_observation(
                {**info, "id": message_id},
                session_id=session_id,
                native_id=str(session.native_id),
                observed_at_ms=observed_at_ms,
                session_created_at_ms=created,
                root_session_id=root_session_id,
                project_path=str(session.project_path)
                if session.project_path is not None
                else None,
                billing_mode=billing_mode,
                routing=routing,
                history_request=True,
            )
            if item is not None:
                observations.append(item)
            elif info.get("role") == "assistant":
                times = info.get("time")
                original = _count(times.get("created")) if isinstance(times, dict) else None
                if original is None or original >= created:
                    status = "partial"
        return OpencodeUsageBatch(
            tuple(observations),
            int(rows[-1][0]) if has_more else None,
            status,
            has_more,
            len(rows),
        )

    def _session_created(self, native_id: str) -> int | None:
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_S)) as connection:
            row = connection.execute(
                "SELECT time_created FROM session WHERE id = ?", (native_id,)
            ).fetchone()
        return _count(row[0]) if row else None
