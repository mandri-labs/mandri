import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import mandri.sessions.service as module
import pytest
from mandri.core.ids import HarnessKind, SessionId
from mandri.core.types.availability import SessionOwner
from mandri.sessions.ownership.service import NativeOwnership
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts.claude_transcripts import ClaudeTranscriptReader
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader
from mandri.sessions.transcripts.opencode_transcripts import OpencodeTranscriptReader
from mandri.sessions.transcripts.resolver import TranscriptResolver

from .substitutes import make_session


@pytest.mark.parametrize("running", [False, True])
async def test_incomplete_opencode_message_requires_a_running_process(
    tmp_path, monkeypatch, running
):
    path = tmp_path / "store.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE message (id TEXT, session_id TEXT, data TEXT)")
        connection.execute(
            "INSERT INTO message VALUES ('message', 'native', ?)",
            (
                json.dumps(
                    {
                        "role": "assistant",
                        "modelID": "m",
                        "providerID": "p",
                        "time": {"created": 1},
                    }
                ),
            ),
        )
    reader = OpencodeTranscriptReader(path)
    service = SessionsService(
        AsyncMock(), AsyncMock(), TranscriptResolver({HarnessKind.OPENCODE: reader})
    )
    monkeypatch.setattr(
        service,
        "get_session",
        AsyncMock(
            return_value=make_session(
                harness=HarnessKind.OPENCODE, native_id="native", project_path="/workspace"
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "inspect_owner",
        lambda *args: NativeOwnership(SessionOwner.EXTERNAL if running else SessionOwner.UNOWNED),
    )
    assert await service.external_status(SessionId("one")) == (running, "p/m")


@pytest.mark.parametrize(
    "harness,event",
    [
        (HarnessKind.CLAUDE, {"type": "user", "message": {"content": [{"type": "tool_result"}]}}),
        (HarnessKind.CODEX, {"type": "event_msg", "payload": {"type": "task_started"}}),
    ],
)
@pytest.mark.parametrize("running", [False, True])
async def test_interrupted_transcript_requires_a_running_harness(
    tmp_path, monkeypatch, harness, event, running
):
    filename = "rollout-native.jsonl" if harness is HarnessKind.CODEX else "native.jsonl"
    (tmp_path / filename).write_text(json.dumps(event) + "\n", encoding="utf8")
    reader = (
        CodexTranscriptReader(tmp_path)
        if harness is HarnessKind.CODEX
        else ClaudeTranscriptReader(tmp_path)
    )
    service = SessionsService(AsyncMock(), AsyncMock(), TranscriptResolver({harness: reader}))
    service.get_session = AsyncMock(
        return_value=make_session(harness=harness, native_id="native", project_path="/workspace")
    )
    monkeypatch.setattr(
        module,
        "inspect_owner",
        lambda *args: NativeOwnership(SessionOwner.EXTERNAL if running else SessionOwner.UNOWNED),
    )
    assert await service.external_status(SessionId("one")) == (running, None)


async def test_process_exit_changes_revision_without_transcript_changes(monkeypatch):
    reader = SimpleNamespace(revision=lambda ref: ("file", 1, 100))
    service = SessionsService(AsyncMock(), AsyncMock())
    service._transcripts = SimpleNamespace(for_session=lambda session: reader)
    service.get_session = AsyncMock(
        return_value=make_session(
            harness=HarnessKind.CLAUDE, native_id="native", project_path="/workspace"
        )
    )
    monkeypatch.setattr(
        module, "inspect_owner", lambda *args: NativeOwnership(SessionOwner.EXTERNAL)
    )
    before = await service.transcript_revision(SessionId("one"))
    monkeypatch.setattr(
        module, "inspect_owner", lambda *args: NativeOwnership(SessionOwner.UNOWNED)
    )
    after = await service.transcript_revision(SessionId("one"))
    assert before == (("file", 1, 100), NativeOwnership(SessionOwner.EXTERNAL))
    assert after == (("file", 1, 100), NativeOwnership(SessionOwner.UNOWNED))
