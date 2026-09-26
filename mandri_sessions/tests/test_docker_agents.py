import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind, SessionId, SessionState
from mandri.core.types.agents import AgentState
from mandri.core.types.execution import ExecutionBackend, ProtectionError
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.sessions.agents.docker import discover_docker_agents
from mandri.sessions.agents.service import AgentHistory
from mandri.sessions.agents.status import historical_state
from mandri.sessions.agents.store import AgentStore
from mandri.sessions.transcripts.resolver import TranscriptResolver

from .substitutes import insert_session_row, make_session


@pytest.fixture
async def docker_agents(tmp_path):
    rows = []
    paths = []
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "sessions.db")
    await db.migrate()
    for name in ("left", "right"):
        workspace = tmp_path / f"workspace-{name}"
        workspace.mkdir()
        state = tmp_path / "native" / f"session-{name}"
        (state / ".codex/sessions").mkdir(parents=True)
        session = replace(
            make_session(f"parent-{name}", harness=HarnessKind.CODEX, project_path=str(workspace)),
            id=SessionId(f"session-{name}"),
            state=SessionState.STOPPED,
            execution_backend=ExecutionBackend.DOCKER,
            execution_context=json.dumps(
                {
                    "version": "1",
                    "workspace_root": str(workspace),
                    "container_root": "/workspace",
                    "native_state_root": str(state),
                    "native_home": "/home/worker",
                }
            ),
        )
        path = state / ".codex/sessions/rollout-shared-child.jsonl"
        metadata = {
            "type": "session_meta",
            "payload": {
                "id": "shared-child",
                "agent_nickname": name,
                "source": {"subagent": {"thread_spawn": {"parent_thread_id": f"parent-{name}"}}},
            },
        }
        event = {
            "type": "event_msg",
            "payload": {"type": "task_complete", "last_agent_message": name},
        }
        path.write_text(json.dumps(metadata) + "\n" + json.dumps(event) + "\n")
        rows.append(session)
        paths.append(path)
        await insert_session_row(db, str(session.id), native_id=str(session.native_id))
    sessions = SimpleNamespace(
        list_sessions=AsyncMock(side_effect=lambda: rows),
        get_session=AsyncMock(
            side_effect=lambda identity: next(row for row in rows if row.id == identity)
        ),
        activity_of=Mock(return_value=None),
    )
    global_discovery = Mock()
    global_discovery.discover.side_effect = AssertionError("Host profiles must not be inspected")
    store = AgentStore(db)
    history = AgentHistory(store, sessions, TranscriptResolver({}), [global_discovery])
    try:
        yield history, store, rows, paths, global_discovery
    finally:
        await db.close()


async def test_docker_children_and_histories_remain_in_each_selected_state(docker_agents):
    history, _, rows, _, global_discovery = docker_agents
    await history.refresh(force=True)
    agents = await history.list()
    assert len(agents) == 2
    assert agents[0].id != agents[1].id
    global_discovery.discover.assert_not_called()
    for parent in rows:
        agent = next(agent for agent in agents if agent.parent_session_id == parent.id)
        assert agent.session_id is None
        page = await history.history(agent.id, None, 20)
        event = json.loads(page.entries[-1])
        assert event["payload"]["last_agent_message"] == agent.title


async def test_pi_discovery_does_not_inspect_other_harness_state(docker_agents):
    history, _, rows, paths, _ = docker_agents
    rows[0] = replace(rows[0], harness=HarnessKind.PI)
    database = paths[0].parents[2] / ".local/share/opencode/opencode.db"
    database.parent.mkdir(parents=True)
    database.write_text("unrelated native state")
    result = discover_docker_agents(rows[0])
    assert result.harness is HarnessKind.PI
    assert result.agents == []
    assert not result.complete
    await history.refresh(force=True)
    assert {agent.parent_session_id for agent in await history.list()} == {str(rows[1].id)}


async def test_docker_child_cannot_read_other_state_or_host_file(docker_agents, tmp_path):
    history, store, rows, paths, _ = docker_agents
    await history.refresh(force=True)
    agent = (await history.list(str(rows[0].id)))[0]
    outside = tmp_path / "private.jsonl"
    outside.write_text('{"private":"synthetic-canary"}\n')
    for path in (outside, paths[1]):
        forged = replace(agent, transcript_path=str(path))
        await store.upsert(forged)
        with pytest.raises(ProtectionError, match="external path"):
            await history.history(agent.id, None, 20)
        assert historical_state(forged, TranscriptResolver({}), rows[0]) is AgentState.UNKNOWN


async def test_discovery_rejects_symlinked_rollout_before_metadata_read(docker_agents, tmp_path):
    _, _, rows, paths, _ = docker_agents
    outside = tmp_path / "private.jsonl"
    outside.write_text('{"private":"synthetic-canary"}\n')
    paths[0].unlink()
    paths[0].symlink_to(outside)
    with pytest.raises(ProtectionError):
        discover_docker_agents(rows[0])


def agy_parent(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "native" / "agy-parent"
    home = state / ".gemini"
    cache = home / "antigravity-cli/cache"
    cache.mkdir(parents=True)
    (cache / "conversation_metadata.json").write_text(
        json.dumps({"child": {"parent_conversation_id": "parent", "title": "Synthetic child"}})
    )
    (home / "mandri-session.json").write_text(
        json.dumps({"native_id": "parent", "is_mandri_root": True})
    )
    session = replace(
        make_session("parent", harness=HarnessKind.AGY, project_path=str(workspace)),
        id=SessionId("agy-parent"),
        execution_backend=ExecutionBackend.DOCKER,
        execution_context=json.dumps(
            {
                "version": "1",
                "workspace_root": str(workspace),
                "container_root": "/workspace",
                "native_state_root": str(state),
                "native_home": "/home/worker",
            }
        ),
    )
    return session, home


def test_agy_discovery_ignores_unread_artifact_and_cache_symlinks(tmp_path):
    session, home = agy_parent(tmp_path)
    outside = tmp_path / "unselected"
    outside.mkdir()
    (home / "antigravity-cli/cli.log").symlink_to(outside / "native-log")
    (home / "antigravity-cli/cache/browser-link").symlink_to(outside, target_is_directory=True)
    conversation = home / "antigravity-cli/brain/child"
    logs = conversation / ".system_generated/logs"
    logs.mkdir(parents=True)
    (conversation / "artifacts").symlink_to(outside, target_is_directory=True)
    path = logs / "transcript.jsonl"
    path.write_text('{"type":"USER_INPUT","content":"Synthetic history"}\n')
    result = discover_docker_agents(session)
    assert result.complete
    assert len(result.agents) == 1
    assert result.agents[0].transcript_path == str(path)


@pytest.mark.parametrize(
    "relative",
    [
        "antigravity-cli/cache/conversation_metadata.json",
        "antigravity-cli/conversation_summaries.db-wal",
        "antigravity-cli/brain/child/.system_generated/logs/transcript.jsonl",
    ],
)
def test_agy_discovery_still_rejects_external_consumed_state(tmp_path, relative):
    session, home = agy_parent(tmp_path)
    outside = tmp_path / "unselected.json"
    outside.write_text('{"synthetic":"unselected"}')
    target = home / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    target.symlink_to(outside)
    with pytest.raises(ProtectionError):
        discover_docker_agents(session)
