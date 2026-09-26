from pathlib import Path

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, PageToken, SessionState
from mandri.core.ports.transcripts import TranscriptPage
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts import (
    CodexTranscriptReader,
    HarnessStoreUnavailableError,
    TranscriptNotFoundError,
    TranscriptResolver,
)

from mandri_sessions.tests.substitutes import FakeDatabase, FakeEngine


@pytest.mark.parametrize("harness", HarnessKind)
@pytest.mark.parametrize("recent", [False, True])
async def test_new_session_history_is_empty_before_native_identity(harness, recent):
    db = await FakeDatabase.create()
    service = SessionsService(db, FakeEngine())
    session = await service.create_session(harness)

    assert await service.history(session.id, None, 50, recent=recent) == TranscriptPage(
        [], None, False
    )

    await service.set_session_state(session.id, SessionState.LIVE)

    assert await service.history(session.id, None, 50, recent=recent) == TranscriptPage(
        [], None, False
    )


@pytest.mark.parametrize("recent", [False, True])
async def test_history_reads_native_rollout_after_identity_is_revealed(tmp_path: Path, recent):
    db = await FakeDatabase.create()
    reader = CodexTranscriptReader(tmp_path / "sessions")
    service = SessionsService(db, FakeEngine(), TranscriptResolver({HarnessKind.CODEX: reader}))
    session = await service.create_session(HarnessKind.CODEX)
    await service.set_session_state(session.id, SessionState.LIVE)

    assert (await service.history(session.id, None, 50, recent=recent)).entries == []

    rollout = tmp_path / "sessions" / "rollout-synthetic-thread.jsonl"
    rollout.parent.mkdir()
    record = '{"type":"response_item","payload":{"role":"assistant","content":"OK"}}'
    rollout.write_text(record + "\n", encoding="utf-8")
    await service.reveal_native_id(session.id, HarnessSessionId("synthetic-thread"))

    page = await service.history(session.id, None, 50, recent=recent)
    assert [entry.rstrip() for entry in page.entries] == [record]
    assert page.next_token is None
    assert page.has_more is False


@pytest.mark.parametrize("state", [SessionState.DISCOVERED, SessionState.LIVE])
async def test_native_session_with_missing_rollout_remains_an_error(tmp_path: Path, state):
    db = await FakeDatabase.create()
    reader = CodexTranscriptReader(tmp_path / "sessions")
    service = SessionsService(db, FakeEngine(), TranscriptResolver({HarnessKind.CODEX: reader}))
    session = await service.create_session(HarnessKind.CODEX)
    await service.reveal_native_id(session.id, HarnessSessionId("missing-thread"))
    await service.set_session_state(session.id, state)

    with pytest.raises(TranscriptNotFoundError):
        await service.history(session.id, None, 50, recent=True)


async def test_history_cursor_is_not_silently_discarded_before_identity():
    db = await FakeDatabase.create()
    service = SessionsService(db, FakeEngine())
    session = await service.create_session(HarnessKind.CODEX)

    with pytest.raises(HarnessStoreUnavailableError):
        await service.history(session.id, PageToken("previous-identity-cursor"), 50)
