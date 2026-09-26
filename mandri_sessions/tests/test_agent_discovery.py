import json
import sqlite3
from dataclasses import replace

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.agents import NativeAgent
from mandri.sessions.agents.claude import ClaudeAgentDiscovery
from mandri.sessions.agents.codex import CodexAgentDiscovery, parent_of
from mandri.sessions.agents.opencode import OpencodeAgentDiscovery
from mandri.sessions.agents.relationships import resolve_agents

from mandri_sessions.tests.substitutes import make_session


def spawn(parent):
    return {"sub_agent": {"thread_spawn": {"parent_thread_id": parent}}}


@pytest.mark.parametrize("variant", ["subagent", "sub_agent", "subAgent"])
@pytest.mark.parametrize("storage", ["database", "rollout"])
def test_codex_source_variants_discover_child(tmp_path, variant, storage):
    source = {
        variant: {"thread_spawn": {"parent_thread_id": "parent", "agent_path": "/root/review"}}
    }
    assert parent_of(source) == "parent"
    assert parent_of(json.dumps(source)) == "parent"
    if storage == "database":
        with sqlite3.connect(tmp_path / "state_5.sqlite") as db:
            db.execute("CREATE TABLE threads (id TEXT, source TEXT, agent_nickname TEXT)")
            db.execute(
                "INSERT INTO threads VALUES (?, ?, ?)",
                ("child", json.dumps(source), "Researcher"),
            )
    else:
        folder = tmp_path / "sessions"
        folder.mkdir()
        (folder / "rollout-child.jsonl").write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": "child", "source": source, "agent_nickname": "Researcher"},
                }
            )
            + "\n"
        )
    agents = CodexAgentDiscovery(tmp_path).discover(
        [make_session("parent", harness=HarnessKind.CODEX)]
    )
    agents = agents.agents
    assert len(agents) == 1
    assert agents[0].task_id == "/root/review"
    assert (agents[0].native_id, agents[0].parent_native_id, agents[0].title) == (
        "child",
        "parent",
        "Researcher",
    )


@pytest.mark.parametrize(
    "source",
    [
        "cli",
        "vscode",
        {"forked_from_id": "parent"},
        {"sub_agent": "review"},
        {"sub_agent": {"other": "parent"}},
    ],
)
def test_codex_forks_and_non_spawn_sources_are_not_agents(source):
    assert parent_of(source) is None


def test_codex_database_archive_wins_over_existing_rollout(tmp_path):
    rollout = tmp_path / "sessions" / "rollout-child.jsonl"
    rollout.parent.mkdir()
    rollout.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "child", "source": spawn("parent")}})
        + "\n"
    )
    with sqlite3.connect(tmp_path / "state_5.sqlite") as db:
        db.execute(
            "CREATE TABLE threads (id TEXT, source TEXT, archived INTEGER,"
            " created_at INTEGER, updated_at INTEGER)"
        )
        db.execute(
            "INSERT INTO threads VALUES (?, ?, 1, 10, 20)", ("child", json.dumps(spawn("parent")))
        )
    assert (
        CodexAgentDiscovery(tmp_path)
        .discover([make_session("parent", harness=HarnessKind.CODEX)])
        .agents
        == []
    )


def test_codex_legacy_timestamps_are_milliseconds(tmp_path):
    with sqlite3.connect(tmp_path / "state_5.sqlite") as db:
        db.execute(
            "CREATE TABLE threads (id TEXT, source TEXT, created_at INTEGER, updated_at INTEGER)"
        )
        db.execute(
            "INSERT INTO threads VALUES (?, ?, 10, 20)", ("child", json.dumps(spawn("parent")))
        )
    child = (
        CodexAgentDiscovery(tmp_path)
        .discover([make_session("parent", harness=HarnessKind.CODEX)])
        .agents[0]
    )
    assert (child.created_at, child.updated_at) == (10000, 20000)


@pytest.mark.parametrize("broken", ["", '{"type":', "not json", "{}"])
def test_codex_incomplete_rollout_does_not_block_valid_relationships(tmp_path, broken):
    folder = tmp_path / "sessions"
    folder.mkdir()
    damaged = folder / "rollout-damaged.jsonl"
    damaged.write_text(broken)
    (folder / "rollout-child.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "child", "source": spawn("parent")}})
    )
    discovery = CodexAgentDiscovery(tmp_path)
    sessions = [make_session("parent", harness=HarnessKind.CODEX)]
    result = discovery.discover(sessions)
    assert [agent.native_id for agent in result.agents] == ["child"]
    assert result.classified_native_ids == frozenset({"child"})
    assert not result.complete
    damaged.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "damaged", "source": "cli"}})
    )
    recovered = discovery.discover(sessions)
    assert recovered.complete
    assert recovered.classified_native_ids == frozenset({"child", "damaged"})


def test_all_harness_relationships_remain_separate_and_nested():
    sessions = [make_session(f"root-{harness}", harness=harness) for harness in HarnessKind]
    native = []
    for session in sessions:
        native.append(NativeAgent(session.harness, "child", str(session.native_id), "Child"))
        native.append(
            NativeAgent(
                session.harness,
                "nested",
                str(session.native_id) if session.harness is HarnessKind.CLAUDE else "child",
                "Nested",
                parent_agent_native_id="child" if session.harness is HarnessKind.CLAUDE else None,
            )
        )
    agents = resolve_agents(native, sessions)
    assert len(agents) == 2 * len(HarnessKind)
    assert len({agent.id for agent in agents}) == 2 * len(HarnessKind)
    for harness in HarnessKind:
        child, nested = [agent for agent in agents if agent.harness is harness]
        assert nested.parent_agent_id == child.id
        assert nested.parent_session_id == child.parent_session_id


def test_claude_same_agent_id_in_two_sessions_has_composite_identity(tmp_path):
    sessions = [make_session("first"), make_session("second")]
    for session in sessions:
        directory = tmp_path / "project" / str(session.native_id) / "subagents"
        directory.mkdir(parents=True)
        (directory / "agent-child.jsonl").write_text("{}\n")
        (directory / "agent-child.meta.json").write_text(
            json.dumps({"taskId": "task", "toolUseId": "delegation"})
        )
    agents = resolve_agents(ClaudeAgentDiscovery(tmp_path).discover(sessions).agents, sessions)
    assert len({agent.id for agent in agents}) == 2
    assert {agent.parent_session_id for agent in agents} == {"first", "second"}
    assert all(agent.task_id == "task" and agent.delegation_id == "delegation" for agent in agents)


def test_claude_orphan_and_cyclic_metadata_never_create_false_parent_links():
    child = NativeAgent(HarnessKind.CLAUDE, "child", "root", "Child")
    native = [
        replace(child, parent_agent_native_id="missing"),
        replace(child, native_id="cycle", parent_agent_native_id="cycle"),
    ]
    assert resolve_agents(native, [make_session("root")]) == []


def test_opencode_discovers_only_native_parent_relationships(tmp_path):
    path = tmp_path / "opencode.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE session (id TEXT, parent_id TEXT, title TEXT,"
            " time_created INTEGER, time_updated INTEGER, time_archived INTEGER)"
        )
        db.executemany(
            "INSERT INTO session VALUES (?, ?, 'title', 1, 2, ?)",
            [
                ("root", None, None),
                ("child", "root", None),
                ("archived", "root", 3),
                ("self", "self", None),
            ],
        )
    native = OpencodeAgentDiscovery(path).discover(
        [make_session("root", harness=HarnessKind.OPENCODE)]
    )
    assert [agent.native_id for agent in native.agents] == ["child"]
