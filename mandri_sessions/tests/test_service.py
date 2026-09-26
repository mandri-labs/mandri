"""Tests for the unified sessions read-model service."""

import pytest
from mandri.core.ids import (
    HarnessKind,
    SessionId,
    SessionState,
    SessionTitle,
)
from mandri.core.types.sessions import SessionStateError
from mandri.sessions.errors import SessionConflictError, SessionNotFoundError
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts import HarnessStoreUnavailableError, TranscriptResolver

from mandri_sessions.tests.substitutes import (
    FakeBackend,
    FakeDatabase,
    FakeEngine,
    FakeReader,
    insert_session_row,
)


async def _service(
    harnesses: dict[HarnessKind, FakeBackend] | None = None,
) -> tuple[SessionsService, FakeDatabase, FakeEngine]:
    db = await FakeDatabase.create()
    engine = FakeEngine()
    for kind, backend in (harnesses or {}).items():
        engine.attach(kind, backend)
    return SessionsService(db, engine), db, engine


async def test_create_session_inserts_discovered_row() -> None:
    service, db, _ = await _service()
    session = await service.create_session(
        HarnessKind.CLAUDE, model="openai/gpt", project_path="C:/work/proj"
    )
    row = await db.fetch_one("SELECT * FROM session WHERE id = ?", (str(session.id),))
    assert row is not None
    assert row["state"] == "discovered"
    assert row["model"] == "openai/gpt"


async def test_list_orders_by_updated_at_desc() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", updated_at=100)
    await insert_session_row(db, "s2", updated_at=300)
    await insert_session_row(db, "s3", updated_at=200)
    sessions = await service.list_sessions()
    assert [str(session.id) for session in sessions] == ["s2", "s3", "s1"]


async def test_list_filters_by_harness_state_and_project_path() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", harness="claude", state="live")
    await insert_session_row(db, "s2", harness="codex", state="live")
    await insert_session_row(db, "s3", harness="claude", state="discovered")
    sessions = await service.list_sessions(harness=HarnessKind.CLAUDE, state=SessionState.LIVE)
    assert [str(session.id) for session in sessions] == ["s1"]


async def test_get_unknown_session_raises() -> None:
    service, _, _ = await _service()
    with pytest.raises(SessionNotFoundError):
        await service.get_session(SessionId("nope"))


async def test_rename_sets_title_overlay() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1")
    updated = await service.rename_session(SessionId("s1"), SessionTitle("new title"))
    assert str(updated.title_overlay) == "new title"
    row = await db.fetch_one("SELECT title_overlay FROM session WHERE id = 's1'")
    assert row is not None
    assert row["title_overlay"] == "new title"


async def test_rename_unknown_session_raises() -> None:
    service, _, _ = await _service()
    with pytest.raises(SessionNotFoundError):
        await service.rename_session(SessionId("nope"), SessionTitle("x"))


async def test_set_model_rejects_blank() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1")
    with pytest.raises(SessionConflictError):
        await service.set_session_model(SessionId("s1"), "   ")


async def test_set_model_rejects_unknown_session() -> None:
    service, _, _ = await _service()
    with pytest.raises(SessionNotFoundError):
        await service.set_session_model(SessionId("nope"), "openai/gpt")


async def test_set_model_updates_row() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1")
    updated = await service.set_session_model(SessionId("s1"), "openai/gpt")
    assert updated.model == "openai/gpt"


async def test_delete_marks_row_deleted() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1")
    await service.delete_session(SessionId("s1"))
    row = await db.fetch_one("SELECT deleted FROM session WHERE id = 's1'")
    assert row is not None
    assert row["deleted"] == 1


async def test_delete_purge_calls_backend_for_native_session() -> None:
    backend = FakeBackend()
    service, db, _ = await _service({HarnessKind.CLAUDE: backend})
    await insert_session_row(db, "s1", native_id="n1")
    await service.delete_session(SessionId("s1"), purge=True)
    assert [str(session_id) for session_id in backend.deleted] == ["n1"]


async def test_delete_purge_without_backend_is_noop() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1")
    await service.delete_session(SessionId("s1"), purge=True)


async def test_set_state_rejects_illegal_transition() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", state="stopped")
    with pytest.raises(SessionStateError):
        await service.set_session_state(SessionId("s1"), SessionState.LIVE)


async def test_failed_startup_can_stop_before_becoming_live() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", state="discovered")
    await service.set_session_state(SessionId("s1"), SessionState.STOPPED)
    row = await db.fetch_one("SELECT state FROM session WHERE id = 's1'")
    assert row is not None and row["state"] == "stopped"
    await service.set_session_state(SessionId("s1"), SessionState.DISCOVERED)


async def test_set_state_updates_row() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", state="discovered")
    await service.set_session_state(SessionId("s1"), SessionState.LIVE)
    row = await db.fetch_one("SELECT state FROM session WHERE id = 's1'")
    assert row is not None
    assert row["state"] == "live"


async def test_reveal_native_id_sets_native_id() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1")
    updated = await service.reveal_native_id(SessionId("s1"), "n9")
    assert str(updated.native_id) == "n9"
    row = await db.fetch_one("SELECT native_id FROM session WHERE id = 's1'")
    assert row is not None
    assert row["native_id"] == "n9"


async def test_reveal_native_id_conflict_raises() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", native_id="taken")
    await insert_session_row(db, "s2")
    with pytest.raises(SessionConflictError):
        await service.reveal_native_id(SessionId("s2"), "taken")


async def test_reveal_native_id_does_not_overwrite_existing() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", native_id="n1")
    updated = await service.reveal_native_id(SessionId("s1"), "n2")
    assert str(updated.native_id) == "n1"


async def test_rotate_native_id_replaces_existing() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", native_id="n1")
    updated = await service.rotate_native_id(SessionId("s1"), "n2")
    assert str(updated.native_id) == "n2"
    row = await db.fetch_one("SELECT native_id FROM session WHERE id = 's1'")
    assert row is not None
    assert row["native_id"] == "n2"


async def test_rotate_native_id_fills_null() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1")
    updated = await service.rotate_native_id(SessionId("s1"), "n1")
    assert str(updated.native_id) == "n1"


async def test_rotate_native_id_conflict_raises() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", native_id="taken")
    await insert_session_row(db, "s2", native_id="n2")
    with pytest.raises(SessionConflictError):
        await service.rotate_native_id(SessionId("s2"), "taken")


async def test_rotate_native_id_unknown_session_raises() -> None:
    service, _, _ = await _service()
    with pytest.raises(SessionNotFoundError):
        await service.rotate_native_id(SessionId("nope"), "n1")


async def test_history_requires_transcript_store() -> None:
    service, db, _ = await _service()
    await insert_session_row(db, "s1", native_id="n1")
    with pytest.raises(HarnessStoreUnavailableError):
        await service.history(SessionId("s1"), None, 50)


async def test_history_bounded_limit_and_ref() -> None:
    db = await FakeDatabase.create()
    reader = FakeReader()
    service = SessionsService(db, FakeEngine(), TranscriptResolver({HarnessKind.CLAUDE: reader}))
    await insert_session_row(db, "s1", native_id="n1")
    page = await service.history(SessionId("s1"), None, 10_000)
    assert not page.has_more
    session, cursor, limit = reader.calls[0]
    assert str(session.native_id) == "n1"
    assert cursor is None
    assert limit == 500
