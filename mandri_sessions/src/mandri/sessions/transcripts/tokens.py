"""Opaque page token envelope encode and decode."""

import base64
import binascii
import dataclasses
import json
import time
from typing import TypeGuard

from mandri.core.ids import EpochMs, PageToken
from mandri.sessions.transcripts.errors import PageTokenInvalidError

JSONL_STORE_KIND = "jsonl"
SQLITE_STORE_KIND = "sqlite"


@dataclasses.dataclass(frozen=True)
class PageTokenData:
    store_kind: str
    offset: int | None = None
    file_size: int | None = None
    rowid: int | None = None
    file_id: str | None = None
    record_end: int | None = None
    issued_at: EpochMs = dataclasses.field(default_factory=lambda: EpochMs(0))


def now_ms() -> EpochMs:
    return EpochMs(int(time.time() * 1000))


def _is_index(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def encode_page_token(data: PageTokenData) -> PageToken:
    payload: dict[str, object] = {
        "store_kind": data.store_kind,
        "issued_at": int(data.issued_at),
    }
    if data.offset is not None:
        payload["offset"] = data.offset
    if data.file_size is not None:
        payload["file_size"] = data.file_size
    if data.rowid is not None:
        payload["rowid"] = data.rowid
    if data.file_id is not None:
        payload["file_id"] = data.file_id
    if data.record_end is not None:
        payload["record_end"] = data.record_end
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return PageToken(base64.urlsafe_b64encode(blob).decode("ascii"))


def decode_page_token(raw: PageToken, store_kind: str) -> PageTokenData:
    try:
        payload = json.loads(base64.urlsafe_b64decode(str(raw)).decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PageTokenInvalidError(f"unreadable page token: {error}") from error
    if not isinstance(payload, dict) or payload.get("store_kind") != store_kind:
        raise PageTokenInvalidError("page token store kind mismatch")
    issued_at = payload.get("issued_at")
    if not isinstance(issued_at, int) or isinstance(issued_at, bool) or issued_at < 0:
        raise PageTokenInvalidError("page token missing issued_at")
    issued = EpochMs(issued_at)
    offset = payload.get("offset")
    file_size = payload.get("file_size")
    rowid = payload.get("rowid")
    if store_kind in {
        JSONL_STORE_KIND,
        "jsonl-recent",
        "jsonl-record",
        "pi-branch",
        "pi-branch-recent",
    }:
        if not _is_index(offset) or not _is_index(file_size):
            raise PageTokenInvalidError("jsonl page token missing byte cursor")
        record_end = payload.get("record_end")
        if record_end is not None and (
            not _is_index(record_end) or not offset <= record_end <= file_size
        ):
            raise PageTokenInvalidError("invalid pending record boundary")
        if offset > file_size:
            raise PageTokenInvalidError("cursor exceeds transcript size")
        return PageTokenData(
            store_kind,
            offset=offset,
            file_size=file_size,
            file_id=payload.get("file_id"),
            record_end=record_end,
            issued_at=issued,
        )
    if not _is_index(rowid):
        raise PageTokenInvalidError("sqlite page token missing rowid cursor")
    return PageTokenData(store_kind, rowid=rowid, issued_at=issued)
