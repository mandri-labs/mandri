import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mandri.core.ports.transcripts import SessionRef
from mandri.core.types.conversation_status import WorkDelta, WorkObservation
from mandri.sessions.native_work_context import NativeWorkContext

MAX_WORK_SCAN_BYTES = 1024 * 1024
MAX_WORK_RECORD_BYTES = 65536


def jsonl_work_delta(
    path: Path,
    session: SessionRef,
    checkpoint: dict[str, Any] | None,
    include: Callable[[int, int], bool] | None = None,
    context: NativeWorkContext | None = None,
    require_owner: bool = False,
    default_owner: str | None = None,
) -> WorkDelta:
    stat = path.stat()
    identity = f"{stat.st_dev}:{stat.st_ino}"
    offset = int(checkpoint.get("offset", 0)) if checkpoint else 0
    prefix_size = int(checkpoint.get("prefix_size", 0)) if checkpoint else 0
    with path.open("rb") as handle:
        previous_prefix = hashlib.sha256(handle.read(prefix_size)).hexdigest()
        baseline = (
            checkpoint is None
            or checkpoint.get("identity") != identity
            or offset > stat.st_size
            or checkpoint.get("prefix") != previous_prefix
        )
        if baseline:
            handle.seek(max(0, stat.st_size - MAX_WORK_RECORD_BYTES))
            tail_start = handle.tell()
            tail = handle.read()
            newline = tail.rfind(b"\n")
            offset = tail_start + newline + 1 if newline >= 0 else 0
        handle.seek(offset)
        records = b"" if baseline else handle.read(MAX_WORK_SCAN_BYTES)
        context = context or NativeWorkContext(
            session.harness,
            str(session.native_id),
            checkpoint.get("context", {}) if checkpoint and not baseline else {},
        )
        if baseline:
            begin = max(0, offset - MAX_WORK_SCAN_BYTES)
            handle.seek(begin)
            history = handle.read(offset - begin)
            if begin:
                history = history.partition(b"\n")[2]
                context.mark_uncertain()
            for line in history.splitlines():
                if len(line) > MAX_WORK_RECORD_BYTES:
                    context.mark_uncertain()
                    continue
                try:
                    raw = json.loads(line)
                    if isinstance(raw, dict):
                        context.observe(raw)
                except (ValueError, TypeError, RecursionError):
                    context.mark_uncertain()
        observations: list[WorkObservation] = []
        discarded = bool(checkpoint and checkpoint.get("discarded")) and not baseline
        for line in records.splitlines(keepends=True):
            complete = line.endswith(b"\n")
            if not complete and len(line) <= MAX_WORK_RECORD_BYTES and not discarded:
                break
            start = offset
            offset += len(line)
            if include is not None and not include(start, offset):
                continue
            if discarded or len(line) > MAX_WORK_RECORD_BYTES or not complete:
                discarded = not complete
                context.mark_uncertain()
                observations.append(WorkObservation(state="unknown"))
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("Invalid work record")
                if require_owner:
                    details = raw.get("data", raw.get("step_update", raw.get("result", raw)))
                    owner = (
                        details.get("conversationId", details.get("conversation_id"))
                        if isinstance(details, dict)
                        else None
                    )
                    if owner is None:
                        if default_owner != str(session.native_id):
                            continue
                        raw = {**raw, "conversation_id": default_owner}
                observations.extend(context.observe(raw))
            except (ValueError, TypeError, RecursionError):
                context.mark_uncertain()
                observations.append(WorkObservation(state="unknown"))
        prefix_size = min(64, stat.st_size)
        handle.seek(0)
        prefix = hashlib.sha256(handle.read(prefix_size)).hexdigest()
    return WorkDelta(
        {
            "identity": identity,
            "offset": offset,
            "prefix_size": prefix_size,
            "prefix": prefix,
            "discarded": discarded,
            "context": context.checkpoint(),
        },
        tuple(observations),
        baseline,
    )
