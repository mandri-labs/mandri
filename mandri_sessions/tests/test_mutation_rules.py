"""Mutation rules for the sessions service: rename free, delete blocked while live."""

import pytest
from mandri.core.ids import SessionId, SessionTitle
from mandri.core.types.sessions import SessionError
from mandri.sessions.errors import SessionRunningError
from mandri.sessions.service import SessionsService

from mandri_sessions.tests.substitutes import FakeDatabase, FakeEngine, insert_session_row


async def _service() -> tuple[SessionsService, FakeDatabase]:
    db = await FakeDatabase.create()
    return SessionsService(db, FakeEngine()), db


def test_session_running_error_is_in_the_sessions_error_chain() -> None:
    assert issubclass(SessionRunningError, SessionError)


async def test_rename_succeeds_while_session_is_running() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1", state="live")
    updated = await service.rename_session(SessionId("s1"), SessionTitle("renamed"))
    assert str(updated.title_overlay) == "renamed"


async def test_delete_running_session_rejected_naming_the_session() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1", state="live")
    with pytest.raises(SessionRunningError) as error:
        await service.delete_session(SessionId("s1"))
    assert "s1" in str(error.value)


async def test_delete_running_session_with_purge_rejected() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1", state="live", native_id="n1")
    with pytest.raises(SessionRunningError):
        await service.delete_session(SessionId("s1"), purge=True)


async def test_delete_after_session_no_longer_running_succeeds() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1", state="stopped")
    await service.delete_session(SessionId("s1"))
    row = await db.fetch_one("SELECT deleted FROM session WHERE id = 's1'")
    assert row is not None
    assert row["deleted"] == 1


async def test_delete_discovered_session_succeeds() -> None:
    service, db = await _service()
    await insert_session_row(db, "s1")
    await service.delete_session(SessionId("s1"))
    row = await db.fetch_one("SELECT deleted FROM session WHERE id = 's1'")
    assert row is not None
    assert row["deleted"] == 1
