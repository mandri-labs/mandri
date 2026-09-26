import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.adapters import HarnessAdapters
from mandri.runtime.control.opencode import OpencodeControlAdapter
from mandri.runtime.service import RuntimeService


@pytest.mark.parametrize("resume", [False, True])
async def test_opencode_attachment_does_not_expose_routed_model(monkeypatch, resume):
    control = Mock()
    factory = Mock(return_value=HarnessAdapters(control=control))
    runtime = RuntimeService({}, adapters=factory)
    runtime._session_state("session").launched_model = (
        ModelSource.GATEWAY,
        "provider/new-model",
        "high",
    )
    runtime._reveal_identity = AsyncMock()
    runtime._events.start_event_pump = Mock()
    identity = AsyncMock(return_value=HarnessSessionId("native-session"))
    monkeypatch.setattr("mandri.runtime.native_id.verify_opencode_session_id", identity)
    monkeypatch.setattr("mandri.runtime.native_id.await_opencode_session_id", identity)

    if resume:
        await runtime._reattach_opencode("session", 8123, HarnessSessionId("native-session"))
    else:
        await runtime._attach_opencode("session", 8123)

    context = factory.call_args.args[0]
    assert context.model is None


async def test_prompt_overrides_model_remembered_by_resumed_conversation():
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(204)

    control = OpencodeControlAdapter("http://opencode.invalid", "native-session")
    await control._client.aclose()
    control._client = httpx.AsyncClient(
        base_url="http://opencode.invalid", transport=httpx.MockTransport(respond)
    )
    try:
        await control.send_prompt("continue")
        await control.send_prompt("follow up")
    finally:
        await control.aclose()
    assert [body["model"] for body in sent] == [
        {"providerID": "mandri", "modelID": "mandri_gateway"},
        {"providerID": "mandri", "modelID": "mandri_gateway"},
    ]
    assert sent[0]["parts"] == [{"type": "text", "text": "continue"}]


@pytest.mark.parametrize("harness", list(HarnessKind))
@pytest.mark.parametrize("changed_model", [False, True])
@pytest.mark.parametrize("changed_effort", [False, True])
async def test_gateway_switch_restarts_only_for_launch_bound_settings(
    harness, changed_model, changed_effort
):
    record = SimpleNamespace(
        harness=harness,
        model_source=ModelSource.GATEWAY,
        model="provider/new" if changed_model else "provider/old",
        reasoning_effort="high" if changed_effort else "low",
        native_id=HarnessSessionId("native"),
    )
    runtime = RuntimeService(
        {}, sessions=SimpleNamespace(get_session=AsyncMock(return_value=record))
    )
    runtime._session_state("session").launched_model = (ModelSource.GATEWAY, "provider/old", "low")
    assert await runtime._models.needs_restart("session") is (
        (harness in (HarnessKind.OPENCODE, HarnessKind.PI) and changed_model)
        or (harness is HarnessKind.PI and changed_effort)
    )
