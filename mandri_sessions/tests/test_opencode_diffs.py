import json
import sqlite3
import subprocess

from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.ports.transcripts import SessionRef
from mandri.sessions.transcripts.opencode_transcripts import OpencodeTranscriptReader


def test_native_turn_snapshots_restore_diffs_in_both_history_directions(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = tmp_path / "snapshot" / "project" / "worktree"
    subprocess.run(["git", "init", "--bare", str(repository)], check=True, capture_output=True)

    def git(*args):
        return (
            subprocess.run(
                ["git", "--git-dir", str(repository), "--work-tree", str(workspace), *args],
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )

    (workspace / "existing.py").write_text("before\n")
    (workspace / "removed.py").write_text("removed\n")
    git("add", "-A")
    start = git("write-tree")
    (workspace / "existing.py").write_text("after\n")
    (workspace / "removed.py").unlink()
    (workspace / "created.py").write_text("created\n")
    git("add", "-A")
    end = git("write-tree")
    database = tmp_path / "opencode.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE session_v2 (id TEXT, project_id TEXT, agent TEXT)")
        db.execute("INSERT INTO session_v2 VALUES ('root','project','plan')")
        db.execute(
            "CREATE TABLE session_message (id TEXT, session_id TEXT, type TEXT, seq INTEGER,"
            " time_created INTEGER, time_updated INTEGER, data TEXT)"
        )
        for index, (identity, kind, data) in enumerate(
            [
                ("user", "user", {"text": "Change files"}),
                (
                    "assistant",
                    "assistant",
                    {"snapshot": {"start": start, "end": end}, "content": []},
                ),
                ("idle", "idle", {"outcome": "succeeded"}),
                ("user2", "user", {"text": "No changes"}),
                (
                    "assistant2",
                    "assistant",
                    {"snapshot": {"start": end, "end": end}, "content": []},
                ),
                ("idle2", "idle", {"outcome": "succeeded"}),
            ]
        ):
            db.execute(
                "INSERT INTO session_message VALUES (?, 'root', ?, ?, ?, ?, ?)",
                (identity, kind, index, index, index, json.dumps(data)),
            )
    before = database.read_bytes()
    reader = OpencodeTranscriptReader(database)
    ref = SessionRef(HarnessKind.OPENCODE, HarnessSessionId("root"), None)
    stored = [json.loads(entry) for entry in reader.page(ref, None, 10).entries]
    live = [json.loads(entry) for entry in reader.recent(ref, None, 10).entries]
    files = [part for entry in stored for part in entry["parts"] if part["type"] == "file_snapshot"]
    parts = [
        entry["properties"]["part"]
        for entry in live
        if entry["type"] == "message.part.updated"
        and entry["properties"]["part"]["type"] == "file_snapshot"
    ]
    assert {part["id"] for part in files} == {part["id"] for part in parts}
    changed = next(part for part in files if part["turnID"] == "user")
    assert [
        (file["file"], file["status"], file["additions"], file["deletions"])
        for file in changed["files"]
    ] == [
        ("created.py", "added", 1, 0),
        ("existing.py", "modified", 1, 1),
        ("removed.py", "deleted", 0, 1),
    ]
    assert "+after" in changed["files"][1]["patch"]
    assert not any(part["turnID"] == "user2" for part in files)
    assert reader.native_agent(ref) == "plan"
    assert database.read_bytes() == before
