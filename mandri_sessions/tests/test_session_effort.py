"""Tests for reasoning effort persistence on sessions."""

import pytest
from mandri.core.ids import HarnessKind, SessionId
from mandri.sessions.errors import SessionConflictError, SessionNotFoundError
from mandri.sessions.service import SessionsService

from mandri_sessions.tests.substitutes import FakeDatabase, FakeEngine, insert_session_row


async def _service() -> tuple[SessionsService, FakeDatabase]:
    db = await FakeDatabase.create()
    return SessionsService(db, FakeEngine()), db


async def test_create_session_persists_effort() -> None:
    service, db = await _service()
    session = await service.create_session(
        HarnessKind.CLAUDE, model="openai/gpt", reasoning_effort="high"
    )
    row = await db.fetch_one("SELECT * FROM session WHERE id = ?", (str(session.id),))
    assert row is not None
    assert row["reasoning_effort"] == "high"
    assert session.reasoning_effort == "high"


async def test_create_session_defaults_effort_to_none() -> None:
    service, db = await _service()
    session = await service.create_session(HarnessKind.CLAUDE)
    row = await db.fetch_one("SELECT * FROM session WHERE id = ?", (str(session.id),))
    assert row is not None
    assert row["reasoning_effort"] is None
    assert session.reasoning_effort is None


async def test_row_to_session_roundtrip() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1")
    await db.execute("UPDATE session SET reasoning_effort = 'low' WHERE id = 's1'")
    loaded = await service.get_session(SessionId("s1"))
    assert loaded.reasoning_effort == "low"


async def test_row_to_session_missing_column_maps_to_none() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1")
    loaded = await service.get_session(SessionId("s1"))
    assert loaded.reasoning_effort is None


async def test_set_session_effort_updates_row() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1", updated_at=2_000)
    updated = await service.set_session_effort(SessionId("s1"), "medium")
    assert updated.reasoning_effort == "medium"
    row = await db.fetch_one("SELECT reasoning_effort, updated_at FROM session WHERE id = 's1'")
    assert row is not None
    assert row["reasoning_effort"] == "medium"
    assert row["updated_at"] == 2_000


async def test_set_session_effort_clears_value() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1")
    await service.set_session_effort(SessionId("s1"), "high")
    updated = await service.set_session_effort(SessionId("s1"), None)
    assert updated.reasoning_effort is None


async def test_set_session_effort_rejects_blank() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1")
    with pytest.raises(SessionConflictError):
        await service.set_session_effort(SessionId("s1"), "   ")


async def test_set_session_effort_unknown_session_raises() -> None:
    service, _ = await _service()
    with pytest.raises(SessionNotFoundError):
        await service.set_session_effort(SessionId("nope"), "high")


async def test_set_session_effort_rejects_deleted_session() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1", deleted=1)
    with pytest.raises(SessionNotFoundError):
        await service.set_session_effort(SessionId("s1"), "high")
