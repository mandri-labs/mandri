from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.agents import Agent, AgentCapabilities, AgentState
from mandri.runtime.agents import AgentService
from mandri.runtime.control.agents.base import UnsupportedAgentOperation
from mandri.runtime.control.errors import ControlTransportError


def setup_service(control=None):
    agent = Agent(
        "child", "parent", HarnessKind.OPENCODE, "native-child", "Child", AgentState.RUNNING, 1, 2
    )
    history = SimpleNamespace(
        get=AsyncMock(return_value=agent),
        set_state=AsyncMock(),
        list=AsyncMock(return_value=[agent]),
    )
    runtime = SimpleNamespace(
        control_for_session=Mock(return_value=SimpleNamespace(agents=control) if control else None)
    )
    sessions = SimpleNamespace(list_sessions=AsyncMock(return_value=[]))
    return AgentService(history, runtime, sessions), history, runtime


async def test_read_only_history_exposes_no_controls_for_inactive_parent():
    service, _, _ = setup_service()
    agents, _ = await service.list()
    assert agents[0].capabilities == AgentCapabilities()
    with pytest.raises(UnsupportedAgentOperation):
        await service.message("child", "hello")
    with pytest.raises(UnsupportedAgentOperation):
        await service.stop("child")


async def test_child_control_uses_own_parent_and_only_updates_child_state():
    control = SimpleNamespace(
        capabilities=Mock(return_value=AgentCapabilities(message=True)), message=AsyncMock()
    )
    service, history, runtime = setup_service(control)
    await service.message("child", "hello")
    runtime.control_for_session.assert_called_once_with("parent")
    assert control.message.await_args.args[0].native_id == "native-child"
    history.set_state.assert_awaited_once_with("child", AgentState.RUNNING)


async def test_failed_delivery_does_not_claim_child_running():
    control = SimpleNamespace(
        capabilities=Mock(return_value=AgentCapabilities(message=True)),
        message=AsyncMock(side_effect=ControlTransportError("rejected")),
    )
    service, history, _ = setup_service(control)
    with pytest.raises(ControlTransportError):
        await service.message("child", "hello")
    history.set_state.assert_not_awaited()


async def test_native_noop_stop_does_not_claim_child_stopped():
    control = SimpleNamespace(
        capabilities=Mock(return_value=AgentCapabilities(stop=True)),
        stop=AsyncMock(return_value=False),
    )
    service, history, _ = setup_service(control)
    assert await service.stop("child") is False
    history.set_state.assert_not_awaited()
