"""Opencode sqlite transcript reader with rowid keyset pagination."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from mandri.core.ids import PageToken, RawEvent
from mandri.core.opencode_messages import message_events, message_record
from mandri.core.ports.transcripts import SessionRef, TranscriptPage
from mandri.core.types.agents import AgentState
from mandri.sessions.native_activity import native_model, native_turn_busy
from mandri.sessions.opencode_store import message_table
from mandri.sessions.transcripts.errors import TranscriptStoreError
from mandri.sessions.transcripts.record_preview import MAX_PREVIEW_BYTES, preview_from_prefix
from mandri.sessions.transcripts.record_view import MAX_INLINE_BYTES
from mandri.sessions.transcripts.tokens import (
    SQLITE_STORE_KIND,
    PageTokenData,
    decode_page_token,
    encode_page_token,
    now_ms,
)

SQLITE_TIMEOUT_S = 2.0


class OpencodeTranscriptReader:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    def agent_state(self, session: SessionRef) -> AgentState:
        rows = self._read_rows(str(session.native_id), None, 1, metadata_only=True)
        if not rows or rows[0][1] is None:
            return AgentState.UNKNOWN
        try:
            latest = json.loads(rows[0][1])
        except (ValueError, RecursionError):
            return AgentState.UNKNOWN
        if isinstance(latest, dict) and latest.get("type") == "idle":
            return {
                "failed": AgentState.FAILED,
                "interrupted": AgentState.STOPPED,
                "succeeded": AgentState.COMPLETED,
            }.get(str(latest.get("outcome")), AgentState.UNKNOWN)
        busy, _ = self.status(session)
        if busy is None:
            return AgentState.UNKNOWN
        return AgentState.RUNNING if busy else AgentState.COMPLETED

    def status(self, session: SessionRef) -> tuple[bool | None, str | None]:
        rows = self._read_rows(str(session.native_id), None, 64, metadata_only=True)
        if not rows:
            return False, None
        busy, model = None, None
        for row in rows:
            if row[1] is None:
                break
            try:
                info = json.loads(str(row[1]))
            except (ValueError, RecursionError):
                break
            if not isinstance(info, dict):
                break
            if info.get("type") == "idle":
                if busy is None:
                    busy = False
                continue
            if "type" in info:
                info = message_record(info, str(session.native_id))["message"]
            entry = json.dumps({"type": "message.updated", "properties": {"info": info}})
            if busy is None:
                busy = native_turn_busy(session.harness, [entry])
            if model is None:
                model = native_model([entry])
            if busy is not None and model is not None:
                break
        return busy, model

    def page(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        rowid_cursor = self._decode_cursor(cursor)
        rows = self._read_rows(str(session.native_id), rowid_cursor, limit + 1)
        has_more = len(rows) > limit
        rows = rows[:limit]
        parts = self._read_parts([str(row[2]) for row in rows])
        entries = [self._entry(row, parts, str(session.native_id)) for row in rows]
        next_token = self._encode_token(rows) if has_more and rows else None
        return TranscriptPage(entries=entries, next_token=next_token, has_more=has_more)

    def _decode_cursor(self, cursor: PageToken | None) -> int | None:
        if cursor is None:
            return None
        data = decode_page_token(cursor, SQLITE_STORE_KIND)
        if data.rowid is None:
            raise TranscriptStoreError("sqlite cursor without rowid")
        return data.rowid

    def recent(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        rows = self._read_rows(str(session.native_id), self._decode_cursor(cursor), limit + 1)
        has_more = len(rows) > limit
        rows = rows[:limit]
        parts = self._read_parts([str(row[2]) for row in rows], identities=True)
        entries: list[RawEvent] = []
        for row in reversed(rows):
            message_id = str(row[2])
            info = {**_decode_data(str(row[1]), int(row[3])), "id": message_id}
            if info.get("type") in {"user", "assistant", "idle", "synthetic", "system", "skill"}:
                entries.extend(
                    RawEvent(json.dumps(event))
                    for event in message_events(info, str(session.native_id))
                )
                continue
            entries.append(
                RawEvent(
                    json.dumps(
                        info
                        if info.get("type") == "mandri.transcript_record"
                        else {
                            "type": "message.updated",
                            "properties": {"info": info},
                        }
                    )
                )
            )
            for part in parts.get(message_id, []):
                entries.append(
                    RawEvent(
                        json.dumps(
                            part
                            if part.get("type") == "mandri.transcript_record"
                            else {
                                "type": "message.part.updated",
                                "properties": {"part": part},
                            }
                        )
                    )
                )
        token = self._encode_token(rows) if has_more and rows else None
        return TranscriptPage(entries, token, has_more)

    def revision(self, session: SessionRef) -> tuple[object, ...]:
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_S)
            try:
                values: list[object] = []
                messages = message_table(connection, str(session.native_id))
                tables = (messages,) if messages == "session_message" else ("message", "part")
                for table in tables:
                    row = connection.execute(
                        f"SELECT count(*), max(time_updated) FROM {table} WHERE session_id = ?",
                        (str(session.native_id),),
                    ).fetchone()
                    values.extend(row)
                return tuple(values)
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise TranscriptStoreError(f"cannot observe opencode store: {error}") from error

    def _read_rows(
        self,
        session_id: str,
        rowid_cursor: int | None,
        fetch_limit: int,
        *,
        metadata_only: bool = False,
    ) -> list[tuple[Any, ...]]:
        where = (
            "WHERE session_id = ? AND rowid < ?"
            if rowid_cursor is not None
            else "WHERE session_id = ?"
        )
        params: tuple[object, ...]
        params = (session_id, rowid_cursor) if rowid_cursor is not None else (session_id,)
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        data = (
            "CASE WHEN length(CAST(data AS BLOB)) <= 65536 THEN data END"
            if metadata_only
            else _bounded_data_sql()
        )
        size_column = "" if metadata_only else ", length(CAST(data AS BLOB))"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_S)
            try:
                table = message_table(connection, session_id)
                if metadata_only and table == "session_message":
                    data = _projected_metadata_sql()
                type_column = ", type" if table == "session_message" else ""
                rows = connection.execute(
                    f"SELECT rowid, {data}, id{size_column}{type_column}"
                    f" FROM {table} {where} ORDER BY rowid DESC LIMIT ?",
                    (*params, fetch_limit),
                ).fetchall()
                if table == "session_message":
                    return [_projected_row(row) for row in rows]
                return rows
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise TranscriptStoreError(f"cannot read opencode store: {error}") from error

    def _read_parts(
        self, message_ids: list[str], *, identities: bool = False
    ) -> dict[str, list[Any]]:
        if not message_ids:
            return {}
        placeholders = ", ".join("?" for _ in message_ids)
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_S)
            try:
                if not connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'part'"
                ).fetchone():
                    return {}
                rows = connection.execute(
                    f"SELECT message_id, {_bounded_data_sql()}, id, length(CAST(data AS BLOB))"
                    " FROM part"
                    f" WHERE message_id IN ({placeholders}) ORDER BY rowid ASC",
                    message_ids,
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as error:
            raise TranscriptStoreError(f"cannot read opencode store: {error}") from error
        parts: dict[str, list[Any]] = {}
        for message_id, data, part_id, size in rows:
            part = _decode_data(str(data), int(size))
            if identities:
                part = {**part, "id": str(part_id), "messageID": str(message_id)}
            parts.setdefault(str(message_id), []).append(part)
        return parts

    @staticmethod
    def _entry(row: tuple[Any, ...], parts: dict[str, list[Any]], session_id: str) -> RawEvent:
        message_id = str(row[2])
        message = _decode_data(str(row[1]), int(row[3]))
        if message.get("type") == "mandri.transcript_record":
            return RawEvent(json.dumps(message))
        if message.get("type") in {"user", "assistant", "idle", "synthetic", "system", "skill"}:
            raw = json.dumps(message_record(message, session_id))
            return RawEvent(json.dumps(_decode_data(raw, len(raw.encode("utf-8")))))
        if message_id not in parts:
            return RawEvent(str(row[1]))
        payload = {"message": message, "parts": parts[message_id]}
        raw = json.dumps(payload, separators=(",", ":"))
        size = len(raw.encode("utf-8"))
        truncated = [
            part for part in parts[message_id] if part.get("type") == "mandri.transcript_record"
        ]
        if truncated:
            size += sum(int(part["byte_length"]) for part in truncated)
        return (
            RawEvent(json.dumps(_decode_data(raw, size)))
            if size > MAX_INLINE_BYTES
            else RawEvent(raw)
        )

    @staticmethod
    def _encode_token(rows: list[tuple[Any, ...]]) -> PageToken:
        return encode_page_token(
            PageTokenData(
                store_kind=SQLITE_STORE_KIND,
                rowid=int(rows[-1][0]),
                issued_at=now_ms(),
            )
        )


def _bounded_data_sql() -> str:
    return (
        f"CASE WHEN length(CAST(data AS BLOB)) > {MAX_INLINE_BYTES}"
        f" THEN substr(data, 1, {MAX_PREVIEW_BYTES}) ELSE data END"
    )


def _projected_metadata_sql() -> str:
    keys = ("model", "time", "finish", "error", "tokens", "cost", "outcome")
    fields = ", ".join(f"'{key}', json_extract(data, '$.{key}')" for key in keys)
    metadata = f"json_object({fields})"
    return (
        f"CASE WHEN json_valid(data) THEN CASE WHEN length(CAST({metadata} AS BLOB)) <= 65536"
        f" THEN {metadata} END END"
    )


def _projected_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    raw = row[1]
    size = row[3] if len(row) == 5 else 0
    if raw is not None and size <= MAX_INLINE_BYTES:
        try:
            info = json.loads(raw)
        except (ValueError, RecursionError):
            return (row[0], raw, *row[2:-1])
        if isinstance(info, dict):
            raw = json.dumps({**info, "id": row[2], "type": row[-1]})
    return (row[0], raw, *row[2:-1])


def _decode_data(data: str, size: int) -> dict[str, Any]:
    if size > MAX_INLINE_BYTES:
        return {
            "type": "mandri.transcript_record",
            "byte_length": size,
            **preview_from_prefix(data[:MAX_PREVIEW_BYTES].encode("utf-8")),
        }
    return dict(json.loads(data))
