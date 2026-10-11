from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.actions import _make_agent_handler, _make_prompt_handler
from mandri.core.ports.control import PromptOutcome, PromptState
from mandri.core.protocol.errors import ProtocolError, ProtocolErrorCode
from mandri.core.protocol.registry import SessionAgentParams, SessionPromptParams
from mandri.runtime.control.errors import ControlError

SESSION = "00000000-0000-0000-0000-000000000001"


async def test_agent_action_confirms_native_selection_without_changing_permissions():
    runtime = SimpleNamespace(select_session_agent=AsyncMock(), set_session_mode=AsyncMock())
    result = await _make_agent_handler(runtime)(
        SessionAgentParams(session_id=SESSION, agent="custom")
    )
    assert result == {"session_id": SESSION, "agent": "custom"}
    runtime.select_session_agent.assert_awaited_once_with(SESSION, "custom")
    runtime.set_session_mode.assert_not_awaited()


async def test_prompt_selects_the_requested_agent_before_delivery():
    calls = []

    async def select(session_id, agent):
        calls.append(("agent", session_id, agent))

    async def prompt(session_id, content):
        calls.append(("prompt", session_id, content))
        return PromptOutcome(PromptState.QUEUED)

    runtime = SimpleNamespace(select_session_agent=select, send_session_prompt=prompt)
    result = await _make_prompt_handler(runtime)(
        SessionPromptParams(session_id=SESSION, agent="plan", content="Continue")
    )
    assert calls == [("agent", SESSION, "plan"), ("prompt", SESSION, "Continue")]
    assert result["state"] == "queued"


async def test_rejected_agent_never_delivers_the_prompt_with_another_agent():
    runtime = SimpleNamespace(
        select_session_agent=AsyncMock(side_effect=ControlError("Agent removed")),
        send_session_prompt=AsyncMock(),
    )
    with pytest.raises(ProtocolError) as error:
        await _make_prompt_handler(runtime)(
            SessionPromptParams(session_id=SESSION, agent="removed", content="Continue")
        )
    assert error.value.code == ProtocolErrorCode.CONTROL_DELIVERY_FAILED
    runtime.send_session_prompt.assert_not_awaited()
