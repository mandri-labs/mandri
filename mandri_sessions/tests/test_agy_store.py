import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, SessionId
from mandri.core.ports.transcripts import SessionRef
from mandri.core.types.availability import SessionOwner
from mandri.core.types.model_selection import ModelSource
from mandri.sessions.adapters.agy_sessions import AgySessionsAdapter
from mandri.sessions.agents.agy import AgyAgentDiscovery
from mandri.sessions.agy_profiles import (
    configure_agy_model,
    merge_agy_hooks,
    prepare_agy_profile,
    read_agy_json,
    write_agy_json,
)
from mandri.sessions.errors import SessionDeleteError
from mandri.sessions.native_activity import native_turn_busy
from mandri.sessions.ownership.processes import dedicated_command, is_harness
from mandri.sessions.ownership.service import NativeOwnership
from mandri.sessions.transcripts.agy_transcripts import AgyTranscriptReader


def conversation(root: Path, native_id: str, prompt: str = "Synthetic request") -> Path:
    database = root / "antigravity-cli/conversations" / f"{native_id}.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS steps (idx INTEGER, step_payload BLOB)")
    path = (
        root / "antigravity-cli/brain" / native_id / ".system_generated/logs/transcript_full.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "step_index": 0,
            "type": "USER_INPUT",
            "status": "DONE",
            "created_at": "2026-09-11T08:00:00Z",
            "content": (
                f"<USER_REQUEST>\n{prompt}\n</USER_REQUEST>\n"
                "<ADDITIONAL_METADATA>x</ADDITIONAL_METADATA>"
            ),
        },
        {"step_index": 1, "type": "PLANNER_RESPONSE", "status": "DONE", "content": "answer"},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return path


def summary(root: Path, native_id: str, title: str, parent: str | None = None) -> None:
    database = root / "antigravity-cli/conversation_summaries.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS summaries (conversation_id TEXT, title TEXT, "
            "last_modified_time TEXT, workspace_uris TEXT, parent_conversation_id TEXT)"
        )
        db.execute(
            "INSERT INTO summaries VALUES (?, ?, ?, ?, ?)",
            (native_id, title, "2026-09-11T08:00:00Z", '["file:///C:/work/project"]', parent),
        )


def test_inventory_recovers_missing_summary_and_preserves_native_prompt(tmp_path: Path) -> None:
    conversation(tmp_path, "known")
    path = conversation(tmp_path, "missing", "Recovered title")
    summary(tmp_path, "known", "Native title")
    sessions = {str(session.native_id): session for session in AgySessionsAdapter(tmp_path).fetch()}
    assert set(sessions) == {"known", "missing"}
    assert sessions["known"].project_path == "C:/work/project"
    assert sessions["known"].native_title == "Native title"
    assert sessions["missing"].native_title == "Recovered title"
    assert sessions["missing"].model_source is ModelSource.NATIVE
    assert "ADDITIONAL_METADATA" in path.read_text()


def test_profile_binding_supplies_cli_model_and_cwd(tmp_path: Path) -> None:
    canonical, profiles = tmp_path / "native", tmp_path / "profiles"
    conversation(canonical, "session")
    write_agy_json(
        profiles / "launch/mandri-session.json",
        {
            "native_id": "session",
            "cwd": "/work",
            "model": "provider/model",
            "model_source": "gateway",
        },
    )
    session = AgySessionsAdapter(canonical, profiles).fetch()[0]
    assert session.model == "provider/model"
    assert session.model_source is ModelSource.GATEWAY
    assert session.project_path == "/work"


def test_transcript_paging_excludes_partial_line(tmp_path: Path) -> None:
    path = conversation(tmp_path, "session")
    with path.open("ab") as handle:
        handle.write(b'{"step_index":2')
    reader = AgyTranscriptReader(tmp_path)
    ref = SessionRef(HarnessKind.AGY, HarnessSessionId("session"))
    first = reader.page(ref, None, 1)
    assert first.has_more
    second = reader.page(ref, first.next_token, 1)
    assert len(second.entries) == 1
    assert not second.has_more
    assert len(reader.recent(ref, None, 10).entries) == 2


def test_completed_response_does_not_prove_global_idle() -> None:
    assert (
        native_turn_busy(
            HarnessKind.AGY, [json.dumps({"type": "PLANNER_RESPONSE", "status": "DONE"})]
        )
        is None
    )
    assert (
        native_turn_busy(HarnessKind.AGY, [json.dumps({"event": "Stop", "fullyIdle": False})])
        is True
    )
    assert (
        native_turn_busy(HarnessKind.AGY, [json.dumps({"event": "Stop", "fullyIdle": True})])
        is False
    )


def test_profiles_share_history_without_mutating_global_settings(tmp_path: Path) -> None:
    canonical, profiles = tmp_path / "native", tmp_path / "profiles"
    path = conversation(canonical, "session")
    settings = {
        "model": "native-model",
        "enableTelemetry": True,
        "customModelsConfig": {"customModels": {"user": {"modelName": "custom"}}},
    }
    write_agy_json(canonical / "antigravity-cli/settings.json", settings)
    write_agy_json(canonical / "config/hooks.json", {"user": {"Stop": []}})
    write_agy_json(canonical / "config/projects/project.json", {"name": "Synthetic project"})
    profile = prepare_agy_profile(profiles, "managed", native=False, canonical_root=canonical)
    assert (profile / "antigravity-cli/conversations/session.db").resolve() == (
        canonical / "antigravity-cli/conversations/session.db"
    ).resolve()
    merge_agy_hooks(profile, {"PreToolUse": []})
    assert "user" in read_agy_json(profile / "config/hooks.json")
    prepare_agy_profile(profiles, "managed", native=True, canonical_root=canonical)
    assert "mandri" not in read_agy_json(profile / "config/hooks.json")
    assert "modelProvider" not in read_agy_json(profile / "antigravity-cli/settings.json")
    assert read_agy_json(canonical / "antigravity-cli/settings.json") == settings
    assert read_agy_json(profile / "config/projects/project.json") == {"name": "Synthetic project"}
    assert path.is_file()


def test_gateway_metadata_preserves_native_custom_model_settings(tmp_path: Path) -> None:
    path = tmp_path / "antigravity-cli/settings.json"
    write_agy_json(
        path,
        {
            "customModelsConfig": {
                "customModels": {
                    "mandri": {
                        "contextWindow": 32768,
                        "maxTokens": 4096,
                        "modelFeatures": {"contextWindowCompression": True},
                    },
                    "personal": {"modelName": "personal"},
                }
            }
        },
    )
    configure_agy_model(tmp_path, {"MANDRI_AGY_MODEL": json.dumps({"maxTokens": 65536})})
    models = read_agy_json(path)["customModelsConfig"]["customModels"]
    assert models["mandri"] == {
        "contextWindow": 32768,
        "maxTokens": 65536,
        "modelFeatures": {"contextWindowCompression": True},
        "modelName": "mandri-route",
    }
    assert models["personal"] == {"modelName": "personal"}
    previous = read_agy_json(path)
    configure_agy_model(tmp_path, {})
    assert read_agy_json(path) == previous


def test_purge_removes_only_target_and_reconciles_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = conversation(tmp_path, "first")
    other = conversation(tmp_path, "other")
    summary(tmp_path, "first", "First")
    summary(tmp_path, "other", "Other")
    write_agy_json(
        tmp_path / "antigravity-cli/cache/last_conversations.json",
        {"/first": "first", "/other": "other"},
    )
    monkeypatch.setattr(
        "mandri.sessions.adapters.agy_sessions.inspect_owner",
        lambda *args: NativeOwnership(SessionOwner.UNOWNED),
    )
    adapter = AgySessionsAdapter(tmp_path)
    adapter.delete(SessionId("first"))
    assert not first.exists()
    assert other.exists()
    assert not adapter.exists(SessionId("first"))
    assert [session.native_id for session in adapter.fetch()] == ["other"]
    assert read_agy_json(tmp_path / "antigravity-cli/cache/last_conversations.json") == {
        "/other": "other"
    }


def test_purge_refuses_external_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = conversation(tmp_path, "session")
    monkeypatch.setattr(
        "mandri.sessions.adapters.agy_sessions.inspect_owner",
        lambda *args: NativeOwnership(SessionOwner.EXTERNAL),
    )
    with pytest.raises(SessionDeleteError, match="active native writer"):
        AgySessionsAdapter(tmp_path).delete(SessionId("session"))
    assert path.exists()


@pytest.mark.parametrize("native_id", ["../outside", "a/b", "a\\b", "..", ""])
def test_store_rejects_unsafe_identifiers(tmp_path: Path, native_id: str) -> None:
    with pytest.raises(ValueError):
        AgySessionsAdapter(tmp_path).exists(SessionId(native_id))


def test_agent_relationships_are_discovered_from_native_metadata(tmp_path: Path) -> None:
    conversation(tmp_path, "parent")
    conversation(tmp_path, "child")
    summary(tmp_path, "child", "Child task", "parent")
    result = AgyAgentDiscovery(tmp_path).discover([])
    assert len(result.agents) == 1
    assert result.agents[0].parent_native_id == "parent"
    assert result.agents[0].title == "Child task"


def test_missing_summary_marks_agent_classification_incomplete(tmp_path: Path) -> None:
    conversation(tmp_path, "parent")
    sessions = AgySessionsAdapter(tmp_path).fetch()
    result = AgyAgentDiscovery(tmp_path).discover(sessions)
    assert not result.complete
    assert result.classified_native_ids == frozenset()


def test_managed_root_is_classified_without_native_summary(tmp_path: Path) -> None:
    canonical, profiles = tmp_path / "native", tmp_path / "profiles"
    conversation(canonical, "root")
    write_agy_json(
        profiles / "managed/mandri-session.json",
        {
            "native_id": "root",
            "is_mandri_root": True,
        },
    )
    sessions = AgySessionsAdapter(canonical, profiles).fetch()
    result = AgyAgentDiscovery(canonical, profiles).discover(sessions)
    assert result.classified_native_ids == frozenset({"root"})
    assert result.complete
    assert result.agents == []


def test_unmarked_binding_does_not_invent_root_or_child_relationship(tmp_path: Path) -> None:
    canonical, profiles = tmp_path / "native", tmp_path / "profiles"
    conversation(canonical, "unclassified")
    write_agy_json(
        profiles / "managed/mandri-session.json",
        {
            "native_id": "unclassified",
            "model": "synthetic/model",
        },
    )
    sessions = AgySessionsAdapter(canonical, profiles).fetch()
    result = AgyAgentDiscovery(canonical, profiles).discover(sessions)
    assert result.classified_native_ids == frozenset()
    assert not result.complete
    assert result.agents == []


def test_ownership_understands_explicit_conversation_arguments() -> None:
    assert is_harness("agy.exe", [], HarnessKind.AGY)
    assert is_harness("antigravity", [], HarnessKind.AGY)
    assert not is_harness("python", [], HarnessKind.AGY)
    assert dedicated_command(["agy", "--conversation=session"], HarnessKind.AGY, "session")
    assert not dedicated_command(["agy", "--continue"], HarnessKind.AGY, "session")
