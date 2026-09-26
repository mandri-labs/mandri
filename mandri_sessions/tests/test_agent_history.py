import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.ports.agents import AgentDiscoveryResult
from mandri.core.types.agents import Agent, AgentState, NativeAgent
from mandri.sessions.agents.service import AgentHistory
from mandri.sessions.agents.status import historical_state
from mandri.sessions.errors import DatabaseAccessError
from mandri.sessions.transcripts.resolver import TranscriptResolver

from mandri_sessions.tests.substitutes import FakeReader, make_session


def child(path=None):
    return Agent(
        "child",
        "parent",
        HarnessKind.CLAUDE,
        "native-child",
        "Child",
        AgentState.UNKNOWN,
        1,
        2,
        transcript_path=str(path) if path else None,
    )


def history(agent, readers=None):
    store = SimpleNamespace(
        get=AsyncMock(return_value=agent),
        upsert=AsyncMock(),
        set_revision=AsyncMock(),
        classify=AsyncMock(),
        list=AsyncMock(return_value=[]),
    )
    sessions = SimpleNamespace(
        get_session=AsyncMock(return_value=make_session("parent", harness=agent.harness)),
        list_sessions=AsyncMock(return_value=[make_session("parent", harness=agent.harness)]),
    )
    return AgentHistory(store, sessions, readers or TranscriptResolver({}), []), store


async def test_claude_history_pages_backwards_without_loss_or_duplication(tmp_path):
    path = tmp_path / "agent-child.jsonl"
    path.write_text("".join(json.dumps({"index": index}) + "\n" for index in range(7)))
    service, _ = history(child(path))
    cursor, collected = None, []
    while True:
        page = await service.history("child", cursor, 2)
        collected = [json.loads(entry)["index"] for entry in page.entries] + collected
        if not page.has_more:
            break
        assert page.next_token != cursor
        cursor = page.next_token
    assert collected == list(range(7))


@pytest.mark.parametrize("requested,expected", [(0, 1), (10000, 500)])
async def test_native_history_limits_are_bounded(requested, expected):
    reader = FakeReader()
    service, _ = history(
        replace(child(), harness=HarnessKind.CODEX), TranscriptResolver({HarnessKind.CODEX: reader})
    )
    await service.history("child", None, requested)
    assert reader.calls[0][2] == expected


@pytest.mark.parametrize(
    "event,state",
    [
        ({"type": "assistant", "message": {"stop_reason": "end_turn"}}, AgentState.COMPLETED),
        ({"type": "user", "message": {}}, AgentState.RUNNING),
        ({"unrelated": True}, AgentState.UNKNOWN),
    ],
)
def test_historical_state_uses_native_boundaries(tmp_path, event, state):
    path = tmp_path / "child.jsonl"
    path.write_text(json.dumps(event) + "\n")
    assert historical_state(child(path), TranscriptResolver({})) is state


async def test_discovery_refresh_does_not_mask_database_failure():
    service, store = history(child())
    service._discoveries = [
        SimpleNamespace(
            discover=lambda sessions: AgentDiscoveryResult(
                HarnessKind.CLAUDE, [NativeAgent(HarnessKind.CLAUDE, "child", "parent", "Child")]
            )
        )
    ]
    store.get.side_effect = DatabaseAccessError("unavailable")
    with pytest.raises(DatabaseAccessError):
        await service.refresh(force=True)
    store.upsert.assert_not_awaited()


async def test_relationship_refresh_preserves_state_without_reading_transcript(tmp_path):
    path = tmp_path / "child.jsonl"
    path.write_text(
        json.dumps({"type": "assistant", "message": {"stop_reason": "end_turn"}}) + "\n"
    )
    service, store = history(replace(child(path), state=AgentState.RUNNING))
    native = NativeAgent(
        HarnessKind.CLAUDE, "child", "parent", "Child", 1, 100, transcript_path=str(path)
    )
    service._discoveries = [
        SimpleNamespace(
            discover=lambda sessions: AgentDiscoveryResult(HarnessKind.CLAUDE, [native])
        )
    ]
    await service.refresh(force=True)
    assert store.upsert.await_args.args[0].state is AgentState.RUNNING
