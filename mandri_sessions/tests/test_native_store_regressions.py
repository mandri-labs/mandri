import json
import sqlite3
from pathlib import Path

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.ports.transcripts import SessionRef
from mandri.sessions.adapters.codex_fetch_sessions import CodexStateDbFetchAdapter
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader
from mandri.sessions.transcripts.errors import PageTokenStaleError
from mandri.sessions.transcripts.recent import recent_jsonl


def test_recent_pages_exclude_partial_tail_and_survive_append(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(b'"one"\n"two"\n"three"\n"partial')
    page = recent_jsonl(path, None, 2)
    assert page.entries == ['"two"', '"three"']
    assert page.has_more
    with path.open("ab") as handle:
        handle.write(b'"\n')
    older = recent_jsonl(path, page.next_token, 2)
    assert older.entries == ['"one"']
    assert not older.has_more
    assert recent_jsonl(path, None, 1).entries == ['"partial"']


def test_recent_cursor_rejects_different_file_and_truncation(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    other = tmp_path / "b.jsonl"
    path.write_bytes(b"1\n2\n3\n")
    other.write_bytes(path.read_bytes())
    cursor = recent_jsonl(path, None, 1).next_token
    with pytest.raises(PageTokenStaleError):
        recent_jsonl(other, cursor, 1)
    path.write_bytes(b"1\n")
    with pytest.raises(PageTokenStaleError):
        recent_jsonl(path, cursor, 1)


def test_codex_uses_name_and_tracks_canonical_rollout_changes(tmp_path: Path) -> None:
    db = tmp_path / "state_5.sqlite"
    root = tmp_path / "sessions"
    root.mkdir()
    old = root / "rollout-old-thread.jsonl"
    active = root / "rollout-new-thread_rollout.jsonl"
    old.write_text('"old"\n')
    active.write_text('"active"\n')
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE threads (id TEXT, cwd TEXT, model TEXT, title TEXT, "
            "name TEXT, first_user_message TEXT, created_at_ms INT, "
            "updated_at_ms INT, archived INT, rollout_path TEXT, archived_at INT)"
        )
        connection.execute(
            "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "thread",
                "/project",
                "model",
                "prompt",
                "User name",
                "prompt",
                1,
                2,
                0,
                str(active),
                None,
            ),
        )
    assert CodexStateDbFetchAdapter(db).fetch()[0].native_title == "User name"
    reader = CodexTranscriptReader(root)
    ref = SessionRef(HarnessKind.CODEX, HarnessSessionId("thread"))
    assert [json.loads(line) for line in reader.page(ref, None, 10).entries] == ["active"]
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE threads SET rollout_path = ?, name = ?", (str(old), ""))
    assert [json.loads(line) for line in reader.page(ref, None, 10).entries] == ["old"]
    assert CodexStateDbFetchAdapter(db).fetch()[0].native_title == "prompt"


def test_recent_pages_span_chunks(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    lines = ['"' + str(i) + "x" * 1000 + '"' for i in range(300)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cursor = None
    pages: list[str] = []
    while True:
        page = recent_jsonl(path, cursor, 31)
        pages = list(page.entries) + pages
        if not page.has_more:
            break
        cursor = page.next_token
    assert pages == lines
