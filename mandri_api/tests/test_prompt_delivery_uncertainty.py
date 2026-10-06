from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.actions import _make_prompt_handler
from mandri.core.protocol.frames import RequestFrame
from mandri.core.protocol.registry import ActionRegistry
from mandri.runtime.control.errors import (
    HarnessNotInitializedError,
    PromptDeliveryFailedError,
    PromptDeliveryUnknownError,
)


@pytest.mark.parametrize(
    "error,code",
    [
        (PromptDeliveryUnknownError("reply lost"), "delivery_unknown"),
        (TimeoutError("reply expired"), "delivery_unknown"),
        (PromptDeliveryFailedError("explicit rejection"), "prompt_delivery_failed"),
        (HarnessNotInitializedError("not ready"), "prompt_delivery_failed"),
    ],
)
async def test_prompt_uncertainty_survives_protocol_dispatch(error, code):
    runtime = SimpleNamespace(send_session_prompt=AsyncMock(side_effect=error))
    registry = ActionRegistry()
    registry.register("session.prompt", _make_prompt_handler(runtime))
    frame = RequestFrame(
        type="request",
        op_id="a" * 36,
        action="session.prompt",
        params={
            "session_id": "0123456789abcdef0123456789abcdef0123",
            "content": "Synthetic message",
        },
    )
    response = await registry.handle(frame)
    assert response.error.code == code
    runtime.send_session_prompt.assert_awaited_once()
