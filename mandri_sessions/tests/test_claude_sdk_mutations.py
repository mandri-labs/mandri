import json

from mandri.core.ids import SessionId, SessionTitle
from mandri.sessions.adapters.claude_fetch_sessions import ClaudeSdkFetchSessionsAdapter
from mandri.sessions.adapters.claude_mutations import (
    ClaudeSdkCheckSessionExistsAdapter,
    ClaudeSdkDeleteSessionAdapter,
    ClaudeSdkRenameSessionAdapter,
)

NATIVE_ID = "019b0000-0000-7000-8000-000000000001"


def test_sdk_fetch_rename_and_delete_share_native_session_identity(tmp_path, monkeypatch):
    config = tmp_path / ".claude"
    path = config / "projects" / "-workspace" / f"{NATIVE_ID}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": NATIVE_ID,
                "cwd": "/workspace",
                "timestamp": "2026-01-01T10:00:00.000Z",
                "message": {"role": "user", "content": "Synthetic session history"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    fetch = ClaudeSdkFetchSessionsAdapter()
    exists = ClaudeSdkCheckSessionExistsAdapter()
    rename = ClaudeSdkRenameSessionAdapter()
    delete = ClaudeSdkDeleteSessionAdapter()
    session_id = SessionId(NATIVE_ID)

    before = fetch.fetch()
    assert len(before) == 1
    assert before[0].native_id == NATIVE_ID
    assert before[0].native_title == "Synthetic session history"
    assert before[0].project_path == "/workspace"
    assert exists.exists(session_id)

    rename.rename(session_id, SessionTitle("Renamed synthetic session"))
    after = fetch.fetch()
    assert len(after) == 1
    assert after[0].id == before[0].id
    assert after[0].native_id == before[0].native_id
    assert after[0].created_at == before[0].created_at
    assert after[0].native_title == "Renamed synthetic session"
    assert exists.exists(session_id)

    delete.delete(session_id)
    assert not exists.exists(session_id)
    assert fetch.fetch() == []
