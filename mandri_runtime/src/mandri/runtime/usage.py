import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

from mandri.core.clock import system_now_ms
from mandri.core.types.usage import UsageAccount, UsageObservation
from mandri.sessions.usage.accounts import claude_account_snapshot, codex_account_snapshot
from mandri.sessions.usage.adapter import to_usage_observation
from mandri.sessions.usage.codex_live import CodexLiveState, normalize_codex_live
from mandri.sessions.usage.counters import record
from mandri.sessions.usage.normalize import normalize_native_usage
from mandri.sessions.usage.types import NativeUsageContext, NativeUsageObserver

logger = logging.getLogger(__name__)
UsageSink = Callable[[UsageObservation], Awaitable[object]]
UsageAccountSink = Callable[[UsageAccount], Awaitable[object]]


class NativeUsageCollector:
    def __init__(
        self,
        sink: UsageSink,
        *,
        account_sink: UsageAccountSink | None = None,
        clock: Callable[[], int] = system_now_ms,
    ) -> None:
        self._sink = sink
        self._account_sink = account_sink
        self._clock = clock
        self._sequences: OrderedDict[str, int] = OrderedDict()
        self._codex: OrderedDict[tuple[str, str | None, str], CodexLiveState] = OrderedDict()
        self._claude_modes: OrderedDict[str, str] = OrderedDict()
        self._accounts: OrderedDict[str, UsageAccount] = OrderedDict()

    async def __call__(self, context: NativeUsageContext, event: Mapping[str, Any]) -> None:
        await self.observe(context, event)

    async def observe(self, context: NativeUsageContext, event: Mapping[str, Any]) -> None:
        now = self._clock()
        codex_key = (context.session_id, context.native_id, context.process_epoch)
        codex_state = self._codex.get(codex_key, CodexLiveState())
        if context.harness == "codex":
            observations, codex_state = normalize_codex_live(
                context, event, codex_state, observed_at_ms=now
            )
        else:
            observations = normalize_native_usage(context, event, observed_at_ms=now)
        if context.harness == "claude" and event.get("type") == "result":
            preferred = (
                "call_model"
                if any(o.scope == "call_model" for o in observations)
                else "turn_main_loop"
            )
            mode = self._claude_modes.setdefault(context.process_epoch, preferred)
            observations = tuple(
                replace(o, authoritative=o.authoritative and o.scope == mode) for o in observations
            )
            if len(self._claude_modes) > 4096:
                self._claude_modes.popitem(last=False)
        for observation in observations:
            observation = replace(observation, occurred_at_ms=observation.occurred_at_ms or now)
            prior = self._sequences.get(observation.series_key)
            sequence = max(now, (prior or 0) + 1)
            value = to_usage_observation(observation, sequence=sequence)
            reset_proven = observation.scope == "call_model" or (
                not context.resumed and context.inherited_history == "none"
            )
            if (
                value.authoritative
                and value.kind == "cumulative"
                and prior is None
                and reset_proven
                and context.process_started_at is not None
            ):
                zero = replace(
                    observation,
                    source_key=f"{observation.series_key}:start",
                    counters={
                        key: 0 if count is not None else None
                        for key, count in observation.counters.items()
                    },
                    occurred_at_ms=context.process_started_at,
                    observed_at_ms=context.process_started_at,
                    reported_cost_usd=0 * observation.reported_cost_usd
                    if observation.reported_cost_usd is not None
                    else None,
                )
                baseline = replace(
                    to_usage_observation(
                        zero,
                        sequence=context.process_started_at,
                    ),
                    baseline=True,
                )
                await self._record(baseline)
                self._sequences[observation.series_key] = context.process_started_at
                sequence = max(sequence, context.process_started_at + 1)
                value = replace(value, sequence=sequence)
            if (
                context.harness == "agy"
                and context.resumed
                and prior is None
                and value.kind == "cumulative"
            ):
                value = replace(value, baseline=True)
            await self._record(value)
            self._sequences[observation.series_key] = sequence
            self._sequences.move_to_end(observation.series_key)
            if len(self._sequences) > 4096:
                self._sequences.popitem(last=False)
        if context.harness == "codex":
            self._codex[codex_key] = codex_state
            self._codex.move_to_end(codex_key)
            if len(self._codex) > 4096:
                self._codex.popitem(last=False)
        if self._account_sink is None or context.profile_id is None:
            return
        account = None
        if context.harness == "codex" and event.get("method") == "account/rateLimits/updated":
            account = codex_account_snapshot(
                context.profile_id,
                record(event.get("params")),
                observed_at_ms=now,
            )
        elif event.get("method") == "account/rateLimits/unavailable":
            account = UsageAccount(
                account_id=context.profile_id,
                harness=context.harness,
                observed_at=now,
                status="unsupported" if context.harness in {"agy", "claude"} else "unavailable",
            )
        elif context.harness == "claude" and event.get("type") == "rate_limit_event":
            account = claude_account_snapshot(context.profile_id, event, observed_at_ms=now)
        if account is not None:
            previous = self._accounts.get(account.account_id)
            if previous is not None and account.status != "available":
                return
            await self._account_sink(account)
            self._accounts[account.account_id] = account
            self._accounts.move_to_end(account.account_id)
            if len(self._accounts) > 4096:
                self._accounts.popitem(last=False)

    async def _record(self, value: UsageObservation) -> None:
        try:
            await self._sink(value)
        except Exception as error:
            fields = ",".join(
                name
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                    "request_count",
                    "reported_cost_usd",
                )
                if getattr(value, name) is not None
            )
            logger.warning(
                "Usage counters could not be saved: harness=%s session=%s fields=%s error=%s",
                value.harness,
                value.session_id,
                fields,
                type(error).__name__,
            )
            raise


async def observe_usage(
    observer: NativeUsageObserver | None, context: NativeUsageContext, event: Mapping[str, Any]
) -> bool:
    if observer is None:
        return True
    try:
        await observer(context, event)
        return True
    except Exception as error:
        statistics = (
            "account_quotas"
            if event.get("method")
            in ("account/rateLimits/updated", "account/rateLimits/unavailable")
            or event.get("type") == "rate_limit_event"
            else "token_counters"
        )
        logger.warning(
            "Native usage collection failed: harness=%s session=%s statistics=%s error=%s; "
            "historical reconciliation required",
            context.harness,
            context.session_id,
            statistics,
            type(error).__name__,
        )
        return False
