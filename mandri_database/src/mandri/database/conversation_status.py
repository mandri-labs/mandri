import json
import sqlite3
from collections.abc import Sequence
from typing import Any

from mandri.core.conversation_status import ObservedWork, reduce_observations
from mandri.core.types.conversation_status import (
    ConversationStatus,
    StatusRevisionError,
    WorkObservation,
)
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage_transactions import transaction


class ConversationStatusRepository:
    def __init__(self, database: AiosqliteDatabase) -> None:
        self._database = database

    async def all(self) -> list[ConversationStatus]:
        rows = await self._database.fetch_all("SELECT summary FROM conversation_status")
        return [ConversationStatus.model_validate_json(row["summary"]) for row in rows]

    async def checkpoint(self, target: str, source: str) -> dict[str, Any] | None:
        row = await self._database.fetch_one(
            "SELECT checkpoint FROM conversation_source WHERE target=? AND source=?",
            (target, source),
        )
        return json.loads(row["checkpoint"]) if row else None

    async def observe(
        self,
        target: str,
        observations: Sequence[WorkObservation],
        source: str | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> ConversationStatus:
        def save(connection: sqlite3.Connection) -> ConversationStatus:
            summary, work = _load(connection, target)

            next_revision = summary.completion_revision

            def claim(key: str) -> bool:
                nonlocal next_revision
                claimed = (
                    connection.execute(
                        "INSERT OR IGNORE INTO conversation_completion(target,event_key,revision)"
                        " VALUES(?,?,?)",
                        (target, key, next_revision + 1),
                    ).rowcount
                    == 1
                )
                if claimed:
                    next_revision += 1
                return claimed

            updated = reduce_observations(summary, work, observations, claim)
            _save(connection, updated, work)
            if source is not None and checkpoint is not None:
                connection.execute(
                    "INSERT INTO conversation_source(target,source,checkpoint) VALUES(?,?,?)"
                    " ON CONFLICT(target,source) DO UPDATE SET checkpoint=excluded.checkpoint",
                    (target, source, json.dumps(checkpoint)),
                )
            return updated

        return await transaction(self._database, save, write=True)

    async def acknowledge(
        self, target: str, through_revision: int, completion_key: str | None = None
    ) -> ConversationStatus:
        def acknowledge(connection: sqlite3.Connection) -> ConversationStatus:
            summary, work = _load(connection, target)
            if through_revision < 0 or through_revision > summary.completion_revision:
                raise StatusRevisionError("Unknown completion revision")
            if completion_key is not None:
                receipt = connection.execute(
                    "SELECT revision FROM conversation_completion WHERE target=? AND event_key=?",
                    (target, completion_key),
                ).fetchone()
                if receipt is None or receipt[0] != through_revision:
                    raise StatusRevisionError("Completion identity changed")
            if through_revision > summary.read_revision:
                summary.read_revision = through_revision
                summary.revision += 1
                _save(connection, summary, work)
            return summary

        return await transaction(self._database, acknowledge, write=True)

    async def link(self, source: str, target: str) -> ConversationStatus:
        def link(connection: sqlite3.Connection) -> ConversationStatus:
            original, original_work = _load(connection, source)
            current, work = _load(connection, target)
            unread = original.completion_revision > original.read_revision
            if unread and original.completion_key is not None:
                claimed = connection.execute(
                    "INSERT OR IGNORE INTO conversation_completion(target,event_key,revision)"
                    " VALUES(?,?,?)",
                    (target, original.completion_key, current.completion_revision + 1),
                ).rowcount
                if claimed:
                    current.completion_revision += 1
                    current.outcome = original.outcome
                    current.completion_key = original.completion_key
                    current.completion_content_key = original.completion_content_key
            if original_work.armed and not work.armed:
                work = original_work
                current.work_state = original.work_state
                current.cycle_active = True
            current.revision = max(current.revision, original.revision) + 1
            _save(connection, current, work)
            connection.execute(
                "INSERT OR IGNORE INTO conversation_completion(target,event_key)"
                " SELECT ?,event_key FROM conversation_completion WHERE target=?",
                (target, source),
            )
            connection.execute(
                "INSERT OR IGNORE INTO conversation_source(target,source,checkpoint)"
                " SELECT ?,source,checkpoint FROM conversation_source WHERE target=?",
                (target, source),
            )
            for table in ("conversation_status", "conversation_source", "conversation_completion"):
                connection.execute(f"DELETE FROM {table} WHERE target=?", (source,))
            return current

        return await transaction(self._database, link, write=True)


def _load(connection: sqlite3.Connection, target: str) -> tuple[ConversationStatus, ObservedWork]:
    row = connection.execute(
        "SELECT summary,observed FROM conversation_status WHERE target=?", (target,)
    ).fetchone()
    if row is None:
        return ConversationStatus(target=target), ObservedWork()
    return ConversationStatus.model_validate_json(row["summary"]), ObservedWork(
        **json.loads(row["observed"])
    )


def _save(connection: sqlite3.Connection, summary: ConversationStatus, work: ObservedWork) -> None:
    connection.execute(
        "INSERT INTO conversation_status(target,summary,observed) VALUES(?,?,?)"
        " ON CONFLICT(target) DO UPDATE SET summary=excluded.summary,observed=excluded.observed",
        (
            summary.target,
            summary.model_dump_json(),
            json.dumps(
                {
                    "armed": work.armed,
                    "cycle_key": work.cycle_key,
                    "terminal_key": work.terminal_key,
                    "terminal_content_key": work.terminal_content_key,
                    "terminal_outcome": work.terminal_outcome,
                }
            ),
        ),
    )
