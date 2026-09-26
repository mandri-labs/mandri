"""Bounded native status inspection without loading display history."""

import json
import os
from pathlib import Path

from mandri.core.ids import HarnessKind
from mandri.sessions.native_activity import native_model, native_turn_busy
from mandri.sessions.transcripts.errors import TranscriptStoreError
from mandri.sessions.transcripts.records import ReverseRecords

MAX_STATUS_SCAN_BYTES = 8 * 1024 * 1024
MAX_STATUS_RECORD_BYTES = 65536
MAX_STATUS_RECORDS = 2000


def jsonl_status(path: Path, harness: HarnessKind) -> tuple[bool | None, str | None]:
    busy, model = None, None
    try:
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            if size == 0:
                return False, None
            scan = ReverseRecords(handle, size, MAX_STATUS_SCAN_BYTES)
            for count, record in enumerate(scan, 1):
                if record.size > MAX_STATUS_RECORD_BYTES:
                    break
                handle.seek(record.start)
                try:
                    line = handle.read(record.size).decode("utf-8")
                    if not isinstance(json.loads(line), dict):
                        break
                except (ValueError, RecursionError):
                    break
                if busy is None:
                    busy = native_turn_busy(harness, [line])
                if model is None:
                    model = native_model([line])
                if (busy is not None and model is not None) or count >= MAX_STATUS_RECORDS:
                    break
    except OSError as error:
        raise TranscriptStoreError(f"cannot inspect transcript status: {error}") from error
    return busy, model
