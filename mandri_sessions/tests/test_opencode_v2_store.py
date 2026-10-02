import json
import sqlite3

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.ports.transcripts import SessionRef
from mandri.sessions.adapters.opencode_fetch_sessions import OpencodeSqliteFetchSessionsAdapter
from mandri.sessions.agents.opencode import OpencodeAgentDiscovery
from mandri.sessions.transcripts.opencode_transcripts import OpencodeTranscriptReader
from mandri.sessions.usage_opencode import OpencodeUsageReader


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "opencode.db"
    with sqlite3.connect(path) as db:
        for table in ("session", "session_v2"):
            db.execute(
                f"CREATE TABLE {table} (id TEXT PRIMARY KEY, parent_id TEXT, title TEXT,"
                " directory TEXT, time_created INTEGER, time_updated INTEGER,"
                " time_archived INTEGER)"
            )
        db.execute(
            "CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER,"
            " time_updated INTEGER, data TEXT)"
        )
        db.execute(
            "CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,"
            " time_created INTEGER, time_updated INTEGER, data TEXT)"
        )
        db.execute(
            "CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT, type TEXT,"
            " seq INTEGER, time_created INTEGER, time_updated INTEGER, data TEXT)"
        )
        db.executemany(
            "INSERT INTO session VALUES (?, ?, ?, '', 1, 1, NULL)",
            [("legacy", None, "Legacy"), ("migrated", "legacy", "Old child")],
        )
        db.executemany(
            "INSERT INTO session_v2 VALUES (?, ?, ?, '', 1, 2, ?)",
            [
                ("root", None, "Root", None),
                ("child", "root", "Child", None),
                ("migrated", "legacy", "Archived", 3),
            ],
        )
        db.execute(
            "INSERT INTO message VALUES ('legacy-msg', 'legacy', 1, 1, ?)",
            (json.dumps({"role": "user"}),),
        )
        db.execute(
            "INSERT INTO session_message VALUES ('user', 'root', 'user', 1, 10, 10, ?)",
            (json.dumps({"text": "hello", "time": {"created": 10}}),),
        )
        db.execute(
            "INSERT INTO session_message VALUES ('assistant', 'root', 'assistant', 2, 11, 12, ?)",
            (
                json.dumps(
                    {
                        "agent": "build",
                        "model": {"providerID": "opencode-go", "id": "glm-5.3-flash"},
                        "time": {"created": 11, "completed": 12},
                        "finish": "stop",
                        "content": [
                            {"type": "reasoning", "text": "thinking"},
                            {"type": "text", "text": "hello"},
                        ],
                        "tokens": {
                            "input": 10,
                            "output": 2,
                            "reasoning": 1,
                            "cache": {"read": 0, "write": 0},
                            "total": 13,
                        },
                        "cost": 0,
                    }
                ),
            ),
        )
        db.execute(
            "INSERT INTO session_message VALUES ('idle', 'root', 'idle', 3, 13, 13, ?)",
            (json.dumps({"outcome": "succeeded", "time": {"created": 13}}),),
        )
    return path


def ref(identity="root"):
    return SessionRef(HarnessKind.OPENCODE, HarnessSessionId(identity), None)


def test_mixed_store_keeps_legacy_and_new_sessions_without_archived_copies(store):
    before = store.read_bytes()
    sessions = OpencodeSqliteFetchSessionsAdapter(store).fetch()
    assert {str(s.native_id) for s in sessions} == {"legacy", "root", "child"}
    agents = OpencodeAgentDiscovery(store).discover(sessions).agents
    assert [(a.native_id, a.parent_native_id) for a in agents] == [("child", "root")]
    assert store.read_bytes() == before


def test_v2_history_recent_status_and_usage_share_projected_identity(store):
    reader = OpencodeTranscriptReader(store)
    history = [json.loads(e) for e in reader.page(ref(), None, 10).entries]
    recent = [json.loads(e) for e in reader.recent(ref(), None, 10).entries]
    history_parts = [p for e in history for p in e["parts"]]
    recent_parts = [e["properties"]["part"] for e in recent if e["type"] == "message.part.updated"]
    assert {p["id"] for p in history_parts} == {p["id"] for p in recent_parts}
    assert reader.status(ref()) == (False, "opencode-go/glm-5.3-flash")
    usage = OpencodeUsageReader(store).read_usage(ref(), session_id="managed", observed_at_ms=20)
    assert usage.status == "ready"
    assert len(usage.observations) == 1
    assert usage.observations[0].output_tokens == 2
    assert usage.observations[0].pricing_context["message_id"] == "assistant"
    assert json.loads(reader.page(ref("legacy"), None, 10).entries[0])["role"] == "user"
    assert recent[-1]["type"] == "session.idle"


def test_v2_pagination_and_revision_observe_projection_updates(store):
    reader = OpencodeTranscriptReader(store)
    first = reader.page(ref(), None, 1)
    second = reader.page(ref(), first.next_token, 1)
    assert first.has_more and second.has_more
    assert json.loads(first.entries[0])["message"]["id"] == "idle"
    assert json.loads(second.entries[0])["message"]["id"] == "assistant"
    before = reader.revision(ref())
    with sqlite3.connect(store) as db:
        db.execute("UPDATE session_message SET time_updated = 30 WHERE id='assistant'")
    assert reader.revision(ref()) != before


def test_v2_oversized_messages_are_bounded_in_both_history_directions(store):
    with sqlite3.connect(store) as db:
        db.execute(
            "UPDATE session_message SET data = ? WHERE id='assistant'",
            (json.dumps({"content": [{"type": "text", "text": "large output\n" * 100000}]}),),
        )
    reader = OpencodeTranscriptReader(store)
    for page in (reader.page(ref(), None, 10), reader.recent(ref(), None, 10)):
        markers = [json.loads(e) for e in page.entries if "mandri.transcript_record" in e]
        assert len(markers) == 1
        assert markers[0]["preview_truncated"] is True
        assert len(markers[0]["preview"]) <= 16384


@pytest.mark.parametrize(
    "outcome,state", [("succeeded", "completed"), ("failed", "failed"), ("interrupted", "stopped")]
)
def test_v2_historical_agent_state_preserves_execution_outcome(store, outcome, state):
    with sqlite3.connect(store) as db:
        db.execute(
            "UPDATE session_message SET data = ? WHERE id='idle'",
            (json.dumps({"outcome": outcome, "time": {"created": 13}}),),
        )
    assert OpencodeTranscriptReader(store).agent_state(ref()).value == state


def test_v2_usage_metadata_remains_available_for_large_assistant_output(store):
    with sqlite3.connect(store) as db:
        db.execute(
            "UPDATE session_message SET data=json_set(data,'$.content',json(?))"
            " WHERE id='assistant'",
            (json.dumps([{"type": "text", "text": "large output\n" * 100000}]),),
        )
    result = OpencodeUsageReader(store).read_usage(ref(), session_id="managed", observed_at_ms=20)
    assert result.status == "ready"
    assert result.observations[0].output_tokens == 2
    assert OpencodeTranscriptReader(store).status(ref()) == (False, "opencode-go/glm-5.3-flash")
