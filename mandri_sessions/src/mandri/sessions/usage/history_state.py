import json
from dataclasses import asdict, dataclass, field
from typing import Self

from mandri.sessions.usage.codex_counters import valid_totals
from mandri.sessions.usage.counters import text


@dataclass
class HistoryState:
    model: str | None = None
    turn_id: str | None = None
    provider_kind: str | None = None
    observed_provider: str | None = None
    previous: dict[str, int] | None = None
    previous_model: str | None = None
    ordinal: int = 0
    own_start: int | None = None
    own_usage_proven: bool = False
    metadata_seen: bool = False
    ownership_blocked: bool = False
    gap: bool = False
    skip_irrelevant: bool = False
    discarded_events: int = 0
    discard_reasons: dict[str, int] = field(default_factory=dict)

    def discard(self, reason: str) -> None:
        self.gap = True
        self.discarded_events += 1
        self.discard_reasons[reason] = self.discard_reasons.get(reason, 0) + 1

    def serialize(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def restore(cls, value: str) -> Self:
        raw = json.loads(value)
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("Invalid usage parser state")
        state = cls(**raw)
        if any(
            value is not None and text(value) is None
            for value in (
                state.model,
                state.turn_id,
                state.provider_kind,
                state.observed_provider,
                state.previous_model,
            )
        ):
            raise ValueError("Invalid usage model context")
        if state.previous is not None and (
            not isinstance(state.previous, dict) or not valid_totals(state.previous)
        ):
            raise ValueError("Invalid usage checkpoint counters")
        if type(state.ordinal) is not int or state.ordinal < 0:
            raise ValueError("Invalid usage record ordinal")
        if state.own_start is not None and (
            type(state.own_start) is not int or state.own_start < 0
        ):
            raise ValueError("Invalid usage ownership boundary")
        if (
            type(state.discarded_events) is not int
            or state.discarded_events < 0
            or not isinstance(state.discard_reasons, dict)
            or set(state.discard_reasons)
            - {
                "invalid_counters",
                "unproven_reset",
                "missing_model",
                "unproven_ownership",
                "malformed_record",
                "oversize_record",
                "missing_message_identity",
                "unattributed_prefix",
                "unattributed_aggregate",
            }
            or any(type(value) is not int or value < 0 for value in state.discard_reasons.values())
            or sum(state.discard_reasons.values()) != state.discarded_events
        ):
            raise ValueError("Invalid usage coverage diagnostics")
        if any(
            type(value) is not bool
            for value in (
                state.own_usage_proven,
                state.metadata_seen,
                state.ownership_blocked,
                state.gap,
                state.skip_irrelevant,
            )
        ):
            raise ValueError("Invalid usage ownership state")
        return state
