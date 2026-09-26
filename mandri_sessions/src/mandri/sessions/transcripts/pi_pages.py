import hashlib
import json
import os
from pathlib import Path

from mandri.core.ids import PageToken, RawEvent
from mandri.core.ports.transcripts import TranscriptPage
from mandri.sessions.transcripts.errors import PageTokenStaleError
from mandri.sessions.transcripts.recent import MAX_PAGE_BYTES
from mandri.sessions.transcripts.record_view import MAX_INLINE_BYTES, large_record
from mandri.sessions.transcripts.records import RecordSpan, file_identity
from mandri.sessions.transcripts.tokens import (
    PageTokenData,
    decode_page_token,
    encode_page_token,
    now_ms,
)


def pi_page(
    path: Path, spans: list[RecordSpan], cursor: PageToken | None, limit: int, *, recent: bool
) -> TranscriptPage:
    kind = "pi-branch-recent" if recent else "pi-branch"
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        identity = file_identity(path, stat)
        offset = stat.st_size if recent else 0
        if cursor is not None:
            token = decode_page_token(cursor, kind)
            if (
                token.file_id != _branch_identity(identity, spans, token.file_size or 0)
                or (token.file_size or 0) > stat.st_size
            ):
                raise PageTokenStaleError("Pi transcript replaced or truncated")
            offset = token.offset or 0
            boundaries = {span.end if recent else span.start for span in spans}
            if offset not in boundaries:
                raise PageTokenStaleError("Pi session branch changed")
        selected = (
            [span for span in spans if span.end <= offset]
            if recent
            else [span for span in spans if span.start >= offset]
        )
        if recent:
            selected.reverse()
        entries: list[RawEvent] = []
        page_bytes = 0
        for span in selected:
            if span.size > MAX_INLINE_BYTES:
                raw = large_record(handle, span, identity)
            else:
                handle.seek(span.start)
                raw = handle.read(span.size).decode("utf-8")
            wire_bytes = len(json.dumps(raw, ensure_ascii=True)) + 1
            if entries and page_bytes + wire_bytes > MAX_PAGE_BYTES:
                break
            entries.append(RawEvent(raw))
            page_bytes += wire_bytes
            if len(entries) >= limit:
                break
        has_more = len(entries) < len(selected)
        next_token = None
        if has_more:
            next_span = selected[len(entries)]
            next_token = encode_page_token(
                PageTokenData(
                    kind,
                    offset=next_span.end if recent else next_span.start,
                    file_size=stat.st_size,
                    file_id=_branch_identity(identity, spans, stat.st_size),
                    issued_at=now_ms(),
                )
            )
    return TranscriptPage(list(reversed(entries)) if recent else entries, next_token, has_more)


def _branch_identity(identity: str, spans: list[RecordSpan], size: int) -> str:
    selected = [(span.start, span.end) for span in spans if span.end <= size]
    return hashlib.sha256(json.dumps([identity, selected]).encode()).hexdigest()
