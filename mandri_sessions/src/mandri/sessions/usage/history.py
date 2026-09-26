import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

from mandri.sessions.usage.codex_history import codex_history_usage
from mandri.sessions.usage.history_headers import irrelevant_codex_record
from mandri.sessions.usage.history_state import HistoryState
from mandri.sessions.usage.normalize import normalize_native_usage
from mandri.sessions.usage.types import NativeUsageContext, NativeUsageObservation


@dataclass(frozen=True)
class UsageCursor:
    identity: str
    offset: int
    anchor: str
    parser_version: int = 2
    state_json: str = "{}"


@dataclass(frozen=True)
class UsageBatch:
    observations: tuple[NativeUsageObservation, ...]
    cursor: UsageCursor | None
    status: str
    has_more: bool = False
    records_read: int = 0


def read_usage_batch(
    path: Path,
    context: NativeUsageContext,
    cursor: UsageCursor | None = None,
    *,
    observed_at_ms: int,
    max_records: int = 200,
    max_bytes: int = 4 * 1024 * 1024,
    max_record_bytes: int = 1024 * 1024,
) -> UsageBatch:
    if not 1 <= max_records <= 10000 or not 1 <= max_record_bytes <= max_bytes <= 64 * 1024 * 1024:
        raise ValueError("Invalid usage scan bounds")
    if context.harness not in {"codex", "claude", "agy", "pi"}:
        return UsageBatch((), cursor, "unsupported")
    observations: list[NativeUsageObservation] = []
    try:
        state = HistoryState.restore(cursor.state_json) if cursor else HistoryState()
    except (TypeError, ValueError):
        return UsageBatch((), cursor, "source_changed")
    try:
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            identity = hashlib.sha256(f"{stat.st_dev}:{stat.st_ino}".encode()).hexdigest()
            offset = cursor.offset if cursor else 0
            if cursor and (
                cursor.parser_version != 2
                or cursor.identity != identity
                or offset > stat.st_size
                or offset < 0
                or _anchor(stream, offset) != cursor.anchor
            ):
                return UsageBatch((), cursor, "source_changed")
            stream.seek(offset)
            skipping = False
            if offset:
                stream.seek(offset - 1)
                skipping = stream.read(1) != b"\n"
                stream.seek(offset)
            read = 0
            status = "ready"
            while read < max_records and stream.tell() - offset < max_bytes:
                start = stream.tell()
                remaining = max_bytes - (start - offset)
                line = stream.readline(min(max_record_bytes + 1, remaining))
                if not line:
                    break
                if skipping:
                    if not state.skip_irrelevant:
                        status = "oversize_record"
                    if line.endswith(b"\n"):
                        skipping = False
                        state.skip_irrelevant = False
                        read += 1
                        state.ordinal += 1
                    continue
                if not line.endswith(b"\n"):
                    if len(line) > max_record_bytes:
                        state.skip_irrelevant = (
                            context.harness == "codex" and irrelevant_codex_record(line)
                        )
                        if not state.skip_irrelevant:
                            status = "oversize_record"
                            state.discard("oversize_record")
                            state.model = None
                            state.previous = None
                        skipping = True
                        continue
                    elif start + len(line) < stat.st_size:
                        status = "batch_limit"
                    else:
                        status = "partial_record"
                    stream.seek(start)
                    break
                read += 1
                if len(line) > max_record_bytes:
                    if context.harness != "codex" or not irrelevant_codex_record(line):
                        status = "oversize_record"
                        state.discard("oversize_record")
                        state.model = None
                        state.previous = None
                    state.ordinal += 1
                    continue
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    status = "malformed_record"
                    state.discard("malformed_record")
                    state.model = None
                    state.previous = None
                    state.ordinal += 1
                    continue
                if isinstance(event, dict):
                    if context.harness == "codex":
                        observations.extend(
                            codex_history_usage(
                                context,
                                event,
                                state,
                                position=start,
                                observed_at_ms=observed_at_ms,
                            )
                        )
                    else:
                        items = normalize_native_usage(
                            context,
                            event,
                            observed_at_ms=observed_at_ms,
                            origin="history",
                        )
                        observations.extend(replace(item, source_position=start) for item in items)
                state.ordinal += 1
            end = stream.tell()
            return UsageBatch(
                tuple(observations),
                UsageCursor(identity, end, _anchor(stream, end), state_json=state.serialize()),
                "partial_usage" if state.gap and status == "ready" else status,
                end < stat.st_size,
                read,
            )
    except OSError:
        return UsageBatch((), cursor, "unavailable")


def _anchor(stream: BinaryIO, offset: int) -> str:
    stream.seek(max(0, offset - 256))
    return hashlib.sha256(stream.read(min(offset, 256))).hexdigest()
