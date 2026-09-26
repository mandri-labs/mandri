"""Reverse byte pagination over complete JSONL records."""

import json
import os
from pathlib import Path

from mandri.core.ids import PageToken, RawEvent
from mandri.core.ports.transcripts import TranscriptPage
from mandri.sessions.transcripts.errors import PageTokenStaleError, TranscriptStoreError
from mandri.sessions.transcripts.record_view import MAX_INLINE_BYTES, large_record
from mandri.sessions.transcripts.records import ReverseRecords, file_identity
from mandri.sessions.transcripts.tokens import (
    PageTokenData,
    decode_page_token,
    encode_page_token,
    now_ms,
)

RECENT_KIND = "jsonl-recent"
MAX_PAGE_BYTES = 8 * 1024 * 1024
MAX_SCAN_BYTES = 8 * 1024 * 1024


def recent_jsonl(path: Path, cursor: PageToken | None, limit: int) -> TranscriptPage:
    try:
        return _recent(path, cursor, limit)
    except OSError as error:
        raise TranscriptStoreError(f"cannot read transcript: {error}") from error


def _recent(path: Path, cursor: PageToken | None, limit: int) -> TranscriptPage:
    entries: list[RawEvent] = []
    page_bytes = 0
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        identity = file_identity(path, stat)
        end, record_end = stat.st_size, None
        if cursor is not None:
            token = decode_page_token(cursor, RECENT_KIND)
            if token.file_id != identity or (token.file_size or 0) > stat.st_size:
                raise PageTokenStaleError("transcript replaced or truncated")
            end, record_end = token.offset or 0, token.record_end
        scan = ReverseRecords(handle, end, MAX_SCAN_BYTES, record_end)
        for record in scan:
            if record.size > MAX_INLINE_BYTES:
                entry = large_record(handle, record, identity)
            else:
                handle.seek(record.start)
                entry = handle.read(record.size).rstrip(b"\r\n").decode("utf-8")
            wire_bytes = len(json.dumps(entry, ensure_ascii=True)) + 1
            if entries and page_bytes + wire_bytes > MAX_PAGE_BYTES:
                scan.offset, scan.record_end = record.end, None
                break
            entries.append(RawEvent(entry))
            page_bytes += wire_bytes
            if len(entries) >= limit:
                break
        has_more = scan.offset > 0
        next_token = (
            encode_page_token(
                PageTokenData(
                    RECENT_KIND,
                    offset=scan.offset,
                    file_size=stat.st_size,
                    file_id=identity,
                    record_end=scan.record_end,
                    issued_at=now_ms(),
                )
            )
            if has_more
            else None
        )
    return TranscriptPage(list(reversed(entries)), next_token, has_more)
