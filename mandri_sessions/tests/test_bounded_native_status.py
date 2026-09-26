import json
import sqlite3
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, SessionId
from mandri.core.ports.transcripts import SessionRef
from mandri.sessions.service import SessionsService
from mandri.sessions.transcripts import status
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader
from mandri.sessions.transcripts.errors import HarnessStoreUnavailableError
from mandri.sessions.transcripts.opencode_transcripts import OpencodeTranscriptReader
from mandri.sessions.transcripts.resolver import TranscriptResolver

from .substitutes import make_session


def event(kind):
    return json.dumps({"type": "event_msg", "payload": {"type": kind}}) + "\n"


def test_latest_status_does_not_read_older_compaction(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        "x" * (9 * 1024**2)
        + "\n"
        + json.dumps({"type": "turn_context", "payload": {"model": "test-model"}})
        + "\n"
        + event("task_complete"),
        encoding="utf8",
    )
    assert status.jsonl_status(path, HarnessKind.CODEX) == (False, "test-model")


@pytest.mark.parametrize(
    "barrier",
    ['{"incomplete":\n', '"not an object"\n', "x" * 70000 + "\n"],
    ids=["invalid-json", "scalar", "oversized"],
)
def test_unreadable_newer_record_never_exposes_an_older_idle_status(tmp_path, barrier):
    path = tmp_path / "session.jsonl"
    path.write_text(event("task_complete") + barrier, encoding="utf8")
    assert status.jsonl_status(path, HarnessKind.CODEX) == (None, None)


def test_scan_budget_returns_unknown_and_does_not_guess_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "MAX_STATUS_SCAN_BYTES", 128)
    path = tmp_path / "session.jsonl"
    path.write_text(event("task_complete") + '{"type":"progress"}\n' * 100, encoding="utf8")
    assert status.jsonl_status(path, HarnessKind.CODEX) == (None, None)


async def test_status_is_independent_of_history_and_unknown_blocks_resume(tmp_path):
    path = tmp_path / "rollout-native.jsonl"
    path.write_text('{"type":"progress"}\n', encoding="utf8")
    reader = CodexTranscriptReader(tmp_path)
    service = SessionsService(
        AsyncMock(), AsyncMock(), TranscriptResolver({HarnessKind.CODEX: reader})
    )
    service.get_session = AsyncMock(
        return_value=make_session(
            harness=HarnessKind.CODEX,
            native_id="native",
            project_path="/workspace",
        )
    )
    service.history = AsyncMock(side_effect=AssertionError("history must not be read"))
    assert await service.external_status(SessionId("session")) == (None, None)
    with pytest.raises(HarnessStoreUnavailableError, match="activity is unknown"):
        await service.external_busy(SessionId("session"))
    service.history.assert_not_awaited()


def test_opencode_status_never_loads_message_parts(tmp_path):
    path = tmp_path / "store.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE message (id TEXT, session_id TEXT, data TEXT)")
        connection.execute(
            "INSERT INTO message VALUES ('one', 'native', ?)",
            (
                json.dumps(
                    {
                        "role": "assistant",
                        "modelID": "model",
                        "providerID": "provider",
                        "time": {"completed": 1},
                    }
                ),
            ),
        )
    reader = OpencodeTranscriptReader(path)
    ref = SessionRef(HarnessKind.OPENCODE, HarnessSessionId("native"))
    assert reader.status(ref) == (False, "provider/model")
