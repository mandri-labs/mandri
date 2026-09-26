"""Byte-offset JSONL page reading with torn-tail exclusion."""

import dataclasses
import os
from pathlib import Path
from typing import BinaryIO

from mandri.core.ids import PageToken, RawEvent
from mandri.core.ports.transcripts import TranscriptPage
from mandri.sessions.transcripts.errors import (
    PageTokenStaleError,
    TranscriptStoreError,
)
from mandri.sessions.transcripts.record_view import MAX_INLINE_BYTES, large_record
from mandri.sessions.transcripts.records import CHUNK_SIZE, RecordSpan, file_identity
from mandri.sessions.transcripts.tokens import (
    JSONL_STORE_KIND,
    PageTokenData,
    decode_page_token,
    encode_page_token,
    now_ms,
)


@dataclasses.dataclass(frozen=True)
class JsonlPage:
    entries: list[RawEvent]
    next_token: PageTokenData | None
    has_more: bool


def read_jsonl_page(path: Path, cursor: PageTokenData | None, limit: int) -> JsonlPage:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise TranscriptStoreError(f"cannot stat transcript: {error}") from error
    offset = 0
    if cursor is not None:
        if cursor.offset is None or cursor.file_size is None:
            raise TranscriptStoreError("jsonl cursor without byte fields")
        if cursor.file_size > size or cursor.offset > size:
            raise PageTokenStaleError("transcript truncated since token was issued")
        offset = cursor.offset
    try:
        return _read_from(path, offset, size, limit)
    except OSError as error:
        raise TranscriptStoreError(f"cannot read transcript: {error}") from error


def _read_from(path: Path, offset: int, size: int, limit: int) -> JsonlPage:
    entries: list[RawEvent] = []
    torn_at: int | None = None
    at_eof = False
    with path.open("rb") as handle:
        identity = file_identity(path, os.fstat(handle.fileno()))
        handle.seek(offset)
        while len(entries) < limit:
            start = handle.tell()
            entry, complete = _read_entry(handle, identity)
            if entry is None and complete:
                at_eof = True
                break
            if entry is None:
                torn_at = start
                break
            entries.append(entry)
        if at_eof:
            return JsonlPage(entries=entries, next_token=None, has_more=False)
        if torn_at is not None:
            return JsonlPage(
                entries=entries,
                next_token=_token(torn_at, size),
                has_more=False,
            )
        boundary = handle.tell()
        peek, complete = _read_entry(handle, identity)
        if peek is None and complete:
            return JsonlPage(entries=entries, next_token=None, has_more=False)
        if peek is None:
            return JsonlPage(entries=entries, next_token=_token(boundary, size), has_more=False)
        return JsonlPage(entries=entries, next_token=_token(boundary, size), has_more=True)


def _read_entry(handle: BinaryIO, identity: str) -> tuple[RawEvent | None, bool]:
    start = handle.tell()
    prefix = handle.readline(MAX_INLINE_BYTES + 1)
    if not prefix:
        return None, True
    block = prefix
    while block and not block.endswith(b"\n"):
        block = handle.readline(CHUNK_SIZE)
    if not block:
        return None, False
    end = handle.tell()
    if end - start <= MAX_INLINE_BYTES:
        return RawEvent(prefix.decode("utf-8")), True
    entry = large_record(handle, RecordSpan(start, end), identity)
    handle.seek(end)
    return RawEvent(entry), True


def page_from_jsonl(path: Path, cursor: PageToken | None, limit: int) -> TranscriptPage:
    cursor_data = decode_page_token(cursor, JSONL_STORE_KIND) if cursor is not None else None
    result = read_jsonl_page(path, cursor_data, limit)
    next_token = encode_page_token(result.next_token) if result.next_token is not None else None
    return TranscriptPage(entries=result.entries, next_token=next_token, has_more=result.has_more)


def _token(offset: int, size: int) -> PageTokenData:
    return PageTokenData(
        store_kind=JSONL_STORE_KIND, offset=offset, file_size=size, issued_at=now_ms()
    )
