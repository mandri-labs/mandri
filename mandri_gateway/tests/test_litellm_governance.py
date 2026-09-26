"""Tests for gateway reasoning effort governance in the litellm adapter."""

from dataclasses import replace

import pytest
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef
from mandri.gateway.litellm_adapter import AnthropicHandler, OpenAIHandler, ResponsesHandler
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState


def _resolved(effort: str | None) -> ResolvedRoute:
    provider = Provider(
        name="prov",
        kind=ProviderKind.OPENROUTER,
        api_base=None,
        api_key=SecretRef("k"),
        state=ProviderState.VERIFIED,
    )
    model = Model(
        provider=ProviderKind.OPENROUTER,
        model_ref=ModelRef("openrouter/z-ai/glm-5.2"),
        api_base=None,
        api_key=SecretRef("k"),
    )
    return ResolvedRoute(
        route_id=RouteId("r"), provider=provider, model=model, reasoning_effort=effort
    )


class CallCapture:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def __call__(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return {"ok": True}


@pytest.mark.parametrize("target", ["openrouter/first", "openrouter/second"])
async def test_gateway_alias_never_selects_the_upstream_model(monkeypatch, target):
    call = CallCapture()
    monkeypatch.setattr("mandri.gateway.litellm_adapter.litellm.acompletion", call)
    route = _resolved("high")
    route = replace(route, model=replace(route.model, model_ref=ModelRef(target)))
    await OpenAIHandler().chat_completions(
        route,
        {
            "model": "mandri_gateway",
            "messages": [{"role": "user", "content": "synthetic"}],
            "reasoning_effort": "low",
        },
    )
    assert call.kwargs["model"] == target
    assert call.kwargs["reasoning_effort"] == "high"


@pytest.fixture()
def capture(monkeypatch: pytest.MonkeyPatch) -> CallCapture:
    return CallCapture()


async def test_chat_override_strips_and_sets_effort(
    monkeypatch: pytest.MonkeyPatch, capture: CallCapture
) -> None:
    monkeypatch.setattr("mandri.gateway.litellm_adapter.litellm.acompletion", capture)
    body = {
        "model": "openrouter/z-ai/glm-5.2",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"effort": "low"},
        "reasoning_effort": "low",
        "temperature": 0.5,
    }
    await OpenAIHandler().chat_completions(_resolved("high"), body)
    assert capture.kwargs is not None
    assert capture.kwargs["reasoning_effort"] == "high"
    assert "reasoning" not in capture.kwargs
    assert capture.kwargs["temperature"] == 0.5


async def test_chat_no_effort_keeps_current_payload_shape(
    monkeypatch: pytest.MonkeyPatch, capture: CallCapture
) -> None:
    monkeypatch.setattr("mandri.gateway.litellm_adapter.litellm.acompletion", capture)
    body = {
        "model": "openrouter/z-ai/glm-5.2",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
    }
    await OpenAIHandler().chat_completions(_resolved(None), body)
    assert capture.kwargs is not None
    assert "reasoning_effort" not in capture.kwargs
    assert capture.kwargs["temperature"] == 0.5


async def test_responses_override_strips_incoming_reasoning(
    monkeypatch: pytest.MonkeyPatch, capture: CallCapture
) -> None:
    monkeypatch.setattr("mandri.gateway.litellm_adapter.litellm.aresponses", capture)
    body = {
        "model": "openrouter/z-ai/glm-5.2",
        "input": "hi",
        "reasoning": {"effort": "low"},
    }
    await ResponsesHandler().responses(_resolved("xhigh"), body)
    assert capture.kwargs is not None
    assert capture.kwargs["reasoning"] == {"effort": "xhigh"}


async def test_responses_no_effort_keeps_current_behavior(
    monkeypatch: pytest.MonkeyPatch, capture: CallCapture
) -> None:
    monkeypatch.setattr("mandri.gateway.litellm_adapter.litellm.aresponses", capture)
    body = {"model": "openrouter/z-ai/glm-5.2", "input": "hi"}
    await ResponsesHandler().responses(_resolved(None), body)
    assert capture.kwargs is not None
    assert "reasoning" not in capture.kwargs


async def test_responses_no_effort_keeps_non_empty_incoming_effort(
    monkeypatch: pytest.MonkeyPatch, capture: CallCapture
) -> None:
    monkeypatch.setattr("mandri.gateway.litellm_adapter.litellm.aresponses", capture)
    body = {
        "model": "openrouter/z-ai/glm-5.2",
        "input": "hi",
        "reasoning": {"effort": "low"},
    }
    await ResponsesHandler().responses(_resolved(None), body)
    assert capture.kwargs is not None
    assert capture.kwargs["reasoning"] == {"effort": "low"}


async def test_anthropic_override_replaces_thinking_with_output_config(
    monkeypatch: pytest.MonkeyPatch, capture: CallCapture
) -> None:
    monkeypatch.setattr(
        "mandri.gateway.litellm_adapter.litellm.anthropic_interface.messages.acreate",
        capture,
    )
    body = {
        "model": "anthropic/claude-x",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "output_config": {"effort": "low"},
    }
    await AnthropicHandler().messages(_resolved("medium"), body)
    assert capture.kwargs is not None
    assert capture.kwargs["output_config"] == {"effort": "medium"}
    assert "thinking" not in capture.kwargs


async def test_anthropic_no_effort_keeps_payload_passthrough(
    monkeypatch: pytest.MonkeyPatch, capture: CallCapture
) -> None:
    monkeypatch.setattr(
        "mandri.gateway.litellm_adapter.litellm.anthropic_interface.messages.acreate",
        capture,
    )
    body = {
        "model": "anthropic/claude-x",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    await AnthropicHandler().messages(_resolved(None), body)
    assert capture.kwargs is not None
    assert capture.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "output_config" not in capture.kwargs
