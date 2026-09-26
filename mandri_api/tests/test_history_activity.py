from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.api.actions import _make_history_handler
from mandri.core.protocol.registry import SessionHistoryParams


@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("managed", [False, True])
async def test_history_restores_working_state_independently_of_ownership(managed, busy):
    sessions = SimpleNamespace(
        history=AsyncMock(
            return_value=SimpleNamespace(entries=[], next_token=None, has_more=False)
        ),
        external_status=AsyncMock(return_value=(busy, None)),
    )
    runtime = SimpleNamespace(
        registry=SimpleNamespace(status=Mock(return_value="live" if managed else "stopped")),
        is_busy=Mock(return_value=busy),
    )
    result = await _make_history_handler(sessions, runtime)(
        SessionHistoryParams(session_id="00000000-0000-0000-0000-000000000001")
    )
    assert result["turn_active"] is busy
    assert result["external_busy"] is (busy and not managed)
    if managed:
        sessions.external_status.assert_not_awaited()
        runtime.is_busy.assert_called_once_with("00000000-0000-0000-0000-000000000001")
    else:
        runtime.is_busy.assert_not_called()
