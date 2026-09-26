"""Opencode transcript reader: message entries joined with their part rows."""

import json
import sqlite3
from pathlib import Path

from mandri.core.ids import HarnessKind, HarnessSessionId, ProjectPath
from mandri.core.ports.transcripts import SessionRef
from mandri.sessions.transcripts.opencode_transcripts import OpencodeTranscriptReader

SCHEMA = (
    """
    CREATE TABLE message (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      time_created INTEGER NOT NULL,
      time_updated INTEGER NOT NULL,
      data TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE part (
      id TEXT PRIMARY KEY,
      message_id TEXT NOT NULL,
      session_id TEXT NOT NULL,
      time_created INTEGER NOT NULL,
      time_updated INTEGER NOT NULL,
      data TEXT NOT NULL
    )
    """,
)

MESSAGE_DATA = '{"role":"user","agent":"build"}'
REPLY_DATA = '{"role":"assistant","modelID":"m"}'
PART_TEXT_A = '{"type":"text","text":"JOURNEY_OK"}'
PART_STEP_START = '{"type":"step-start"}'
PART_TEXT_B = '{"type":"text","text":"done"}'


def _session() -> SessionRef:
    return SessionRef(
        harness=HarnessKind.OPENCODE,
        native_id=HarnessSessionId("ses_a"),
        project_path=ProjectPath("C:/work"),
    )


def _store(
    tmp_path: Path,
    messages: list[tuple[str, str]],
    parts: list[tuple[str, str, str]],
) -> Path:
    db_path = tmp_path / "opencode.db"
    connection = sqlite3.connect(db_path)
    for statement in SCHEMA:
        connection.execute(statement)
    for index, (message_id, data) in enumerate(messages):
        connection.execute(
            "INSERT INTO message (id, session_id, time_created, time_updated, data)"
            " VALUES (?, 'ses_a', ?, ?, ?)",
            (message_id, 1_000 + index, 1_000 + index, data),
        )
    for part_id, message_id, data in parts:
        connection.execute(
            "INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)"
            " VALUES (?, ?, 'ses_a', 1, 1, ?)",
            (part_id, message_id, data),
        )
    connection.commit()
    connection.close()
    return db_path


def _entries(reader: OpencodeTranscriptReader, limit: int = 50) -> list[str]:
    return [str(entry) for entry in reader.page(_session(), None, limit).entries]


def test_entry_joins_message_data_with_part_texts(tmp_path: Path) -> None:
    db_path = _store(
        tmp_path,
        [("msg_1", MESSAGE_DATA)],
        [("prt_1", "msg_1", PART_TEXT_A), ("prt_2", "msg_1", PART_STEP_START)],
    )
    entries = _entries(OpencodeTranscriptReader(db_path))
    assert len(entries) == 1
    payload = json.loads(entries[0])
    assert payload["message"] == json.loads(MESSAGE_DATA)
    assert payload["parts"] == [json.loads(PART_TEXT_A), json.loads(PART_STEP_START)]


def test_entry_without_parts_keeps_raw_message_data(tmp_path: Path) -> None:
    db_path = _store(tmp_path, [("msg_1", MESSAGE_DATA)], [])
    assert _entries(OpencodeTranscriptReader(db_path)) == [MESSAGE_DATA]


def test_recent_history_exposes_stable_part_identity_and_user_role(tmp_path: Path) -> None:
    db_path = _store(tmp_path, [("msg_1", MESSAGE_DATA)], [("prt_1", "msg_1", PART_TEXT_A)])
    reader = OpencodeTranscriptReader(db_path)
    revision = reader.revision(_session())
    events = [json.loads(line) for line in reader.recent(_session(), None, 10).entries]
    assert events[0]["properties"]["info"]["role"] == "user"
    assert events[1]["properties"]["part"]["id"] == "prt_1"
    assert events[1]["properties"]["part"]["messageID"] == "msg_1"
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE part SET data = ?, time_updated = 2", (PART_TEXT_B,))
    connection.commit()
    connection.close()
    assert reader.revision(_session()) != revision
    events = [json.loads(line) for line in reader.recent(_session(), None, 10).entries]
    assert events[1]["properties"]["part"]["id"] == "prt_1"
    assert events[1]["properties"]["part"]["text"] == "done"


def test_one_entry_per_message_with_parts_in_rowid_order(tmp_path: Path) -> None:
    db_path = _store(
        tmp_path,
        [("msg_1", MESSAGE_DATA), ("msg_2", REPLY_DATA)],
        [("prt_1", "msg_2", PART_TEXT_B), ("prt_2", "msg_2", PART_TEXT_A)],
    )
    entries = _entries(OpencodeTranscriptReader(db_path))
    assert len(entries) == 2
    assert json.loads(entries[0])["parts"] == [json.loads(PART_TEXT_B), json.loads(PART_TEXT_A)]
    assert entries[1] == MESSAGE_DATA


def test_entries_are_newest_first_across_pages(tmp_path: Path) -> None:
    db_path = _store(
        tmp_path,
        [("msg_1", MESSAGE_DATA), ("msg_2", REPLY_DATA), ("msg_3", REPLY_DATA)],
        [],
    )
    reader = OpencodeTranscriptReader(db_path)
    ref = _session()
    collected: list[str] = []
    cursor = None
    while True:
        page = reader.page(ref, cursor, 2)
        collected.extend(str(entry) for entry in page.entries)
        if not page.has_more:
            break
        cursor = page.next_token
    assert collected == [REPLY_DATA, REPLY_DATA, MESSAGE_DATA]


def test_other_sessions_parts_and_messages_are_excluded(tmp_path: Path) -> None:
    db_path = _store(
        tmp_path,
        [("msg_1", MESSAGE_DATA)],
        [("prt_1", "msg_other", '{"type":"text","text":"other"}')],
    )
    connection = sqlite3.connect(db_path)
    connection.execute(
        "INSERT INTO message (id, session_id, time_created, time_updated, data)"
        " VALUES ('msg_x', 'ses_b', 5, 5, ?)",
        ('{"role":"user"}',),
    )
    connection.commit()
    connection.close()
    assert _entries(OpencodeTranscriptReader(db_path)) == [MESSAGE_DATA]


def test_large_part_is_a_bounded_preview_in_both_history_directions(tmp_path: Path) -> None:
    data = json.dumps({"type": "text", "text": "large output\n" * 100000})
    db_path = _store(tmp_path, [("msg_1", MESSAGE_DATA)], [("prt_1", "msg_1", data)])
    reader = OpencodeTranscriptReader(db_path)
    for page in (reader.recent(_session(), None, 10), reader.page(_session(), None, 10)):
        markers = [json.loads(line) for line in page.entries if "mandri.transcript_record" in line]
        assert len(markers) == 1
        marker = markers[0]
        assert marker["type"] == "mandri.transcript_record"
        assert marker["preview"].startswith("large output\n")
        assert marker["preview_truncated"] is True
        assert len(marker["preview"]) <= 16 * 1024


def test_large_message_is_a_preview_without_losing_following_parts(tmp_path: Path) -> None:
    data = json.dumps({"role": "assistant", "text": "message\n" * 200000})
    db_path = _store(tmp_path, [("msg_1", data)], [("prt_1", "msg_1", PART_TEXT_A)])
    reader = OpencodeTranscriptReader(db_path)
    events = [json.loads(line) for line in reader.recent(_session(), None, 10).entries]
    assert events[0]["preview"].startswith("message\n")
    assert events[1]["properties"]["part"]["id"] == "prt_1"
