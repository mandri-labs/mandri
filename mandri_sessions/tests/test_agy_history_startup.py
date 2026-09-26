import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, PageToken, SessionState
from mandri.core.ports.transcripts import SessionRef, TranscriptPage
from mandri.sessions.agy_lease import AgyConversationLease
from mandri.sessions.agy_profiles import write_agy_json
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.agy_transcripts import AgyTranscriptReader
from mandri.sessions.transcripts.errors import TranscriptNotFoundError
from mandri.sessions.transcripts.resolver import TranscriptResolver

from mandri_sessions.tests.substitutes import FakeDatabase, FakeEngine


def pending_root(tmp_path: Path, *, pending: bool = True, existing_steps: bool = False):
    canonical, profiles = tmp_path / "native", tmp_path / "profiles"
    write_agy_json(
        profiles / "managed/mandri-session.json",
        {
            "native_id": "new",
            "is_mandri_root": True,
            "history_pending": pending,
        },
    )
    database = canonical / "antigravity-cli/conversations/new.db"
    database.parent.mkdir(parents=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("CREATE TABLE steps (idx INTEGER)")
        if existing_steps:
            connection.execute("INSERT INTO steps VALUES (0)")
    lease = AgyConversationLease(canonical, "new")
    lease.acquire()
    return AgyTranscriptReader(canonical, profiles), lease


@pytest.mark.parametrize("recent", [False, True])
async def test_fresh_managed_identity_has_empty_history_until_native_file_exists(
    tmp_path: Path, recent
):
    reader, lease = pending_root(tmp_path)
    database = await FakeDatabase.create()
    service = SessionsService(database, FakeEngine(), TranscriptResolver({HarnessKind.AGY: reader}))
    session = await service.create_session(HarnessKind.AGY)
    await service.reveal_native_id(session.id, HarnessSessionId("new"))
    await service.set_session_state(session.id, SessionState.LIVE)
    try:
        assert await service.history(session.id, None, 50, recent=recent) == TranscriptPage(
            [], None, False
        )
        ref = SessionRef(HarnessKind.AGY, HarnessSessionId("new"))
        initial = reader.revision(ref)
        assert reader.status(ref) == (None, None)
        path = (
            reader.root / "antigravity-cli/brain/new/.system_generated/logs/transcript_full.jsonl"
        )
        path.parent.mkdir(parents=True)
        record = '{"step_index":0,"type":"USER_INPUT","content":"Synthetic"}'
        path.write_text(record + "\n", encoding="utf-8")
        assert [
            line.strip()
            for line in (await service.history(session.id, None, 50, recent=recent)).entries
        ] == [record]
        assert reader.revision(ref) != initial
    finally:
        lease.release()


@pytest.mark.parametrize("recent", [False, True])
@pytest.mark.parametrize("cause", ["resumed", "old_steps", "stopped", "cursor"])
def test_missing_existing_or_unowned_history_is_not_hidden(tmp_path: Path, recent, cause):
    reader, lease = pending_root(
        tmp_path, pending=cause != "resumed", existing_steps=cause == "old_steps"
    )
    if cause == "stopped":
        lease.release()
    read = reader.recent if recent else reader.page
    cursor = PageToken("old-history-cursor") if cause == "cursor" else None
    try:
        with pytest.raises(TranscriptNotFoundError):
            read(SessionRef(HarnessKind.AGY, HarnessSessionId("new")), cursor, 50)
    finally:
        lease.release()
