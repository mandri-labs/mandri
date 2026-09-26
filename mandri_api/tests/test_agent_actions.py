import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.agent_actions import register_agent_actions
from mandri.core.ids import HarnessKind
from mandri.core.protocol.errors import ProtocolErrorCode
from mandri.core.protocol.frames import RequestFrame
from mandri.core.protocol.registry import ActionRegistry
from mandri.core.types.agents import Agent, AgentState
from mandri.runtime.control.agents.base import UnsupportedAgentOperation
from mandri.runtime.control.errors import ControlTransportError
from mandri.sessions.errors import DatabaseAccessError
from mandri.sessions.transcripts.errors import PageTokenInvalidError


@pytest.mark.parametrize(
    "error,code",
    [
        (UnsupportedAgentOperation("unsupported"), ProtocolErrorCode.AGENT_UNSUPPORTED),
        (ControlTransportError("rejected"), ProtocolErrorCode.CONTROL_DELIVERY_FAILED),
        (DatabaseAccessError("private database path"), ProtocolErrorCode.HARNESS_STORE_UNAVAILABLE),
        (PageTokenInvalidError("private cursor"), ProtocolErrorCode.INVALID_PARAMS),
        (RuntimeError("private unexpected detail"), ProtocolErrorCode.INTERNAL_ERROR),
    ],
)
async def test_agent_failure_is_correlated_and_private_details_are_hidden(error, code):
    registry = ActionRegistry()
    register_agent_actions(registry, SimpleNamespace(history=AsyncMock(side_effect=error)))
    response = await registry.handle(
        RequestFrame(
            type="request",
            op_id="agent-request",
            action="agent.history",
            params={"agent_id": "child"},
        )
    )
    assert response.op_id == "agent-request"
    assert not response.ok
    assert response.error.code is code
    assert "private" not in response.model_dump_json()


async def test_agent_list_exposes_task_relationship_without_transcript_path():
    agent = Agent(
        "child",
        "parent",
        HarnessKind.CLAUDE,
        "native-child",
        "Child",
        AgentState.UNKNOWN,
        1,
        2,
        task_id="/root/child",
        transcript_path="private-path",
    )
    registry = ActionRegistry()
    register_agent_actions(
        registry,
        SimpleNamespace(
            list=AsyncMock(return_value=([agent], {})),
            history_store=SimpleNamespace(
                cache_lock=asyncio.Lock(), classified=AsyncMock(return_value=["parent"])
            ),
        ),
    )
    response = await registry.handle(
        RequestFrame(type="request", op_id="list", action="agent.list", params={})
    )
    assert response.ok
    assert "private" not in response.model_dump_json()
    view = response.result["agents"][0]
    assert "transcript_path" not in view
    assert view["native_id"] == "native-child"
    assert view["task_id"] == "/root/child"


async def test_invalid_history_limit_is_rejected_before_service_call():
    service = SimpleNamespace(history=AsyncMock())
    registry = ActionRegistry()
    register_agent_actions(registry, service)
    response = await registry.handle(
        RequestFrame(
            type="request",
            op_id="bad-limit",
            action="agent.history",
            params={"agent_id": "child", "limit": 501},
        )
    )
    assert not response.ok
    assert response.op_id == "bad-limit"
    assert response.error.code is ProtocolErrorCode.INVALID_PARAMS
    service.history.assert_not_awaited()
