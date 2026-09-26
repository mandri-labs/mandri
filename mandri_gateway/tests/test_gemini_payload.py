from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from litellm.types.utils import ModelResponse
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.gateway.litellm_adapter import GeminiHandler
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState


@pytest.mark.parametrize("stream", [False, True])
async def test_gemini_preserves_instructions_tools_and_pins_route(monkeypatch, stream):
    api_base = Url("http://synthetic.invalid/v1")
    key = SecretRef("synthetic-upstream-key")
    provider = Provider("fixture", ProviderKind.OPENAI, api_base, key, ProviderState.VERIFIED)
    route = ResolvedRoute(
        route_id=RouteId("route"),
        provider=provider,
        model=Model(ProviderKind.OPENAI, ModelRef("custom_openai/pinned-model"), api_base, key),
    )
    capture = AsyncMock(return_value={"candidates": []})
    monkeypatch.setattr("mandri.gateway.litellm_adapter.generate_content", capture)
    body = {
        "model": "untrusted-model",
        "api_key": "untrusted-key",
        "api_base": "http://untrusted.invalid",
        "systemInstruction": {"parts": [{"text": "SYSTEM_CONTRACT_MARKER"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "USER_CONTRACT_MARKER"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "run_command",
                            "args": {"CommandLine": "printf synthetic"},
                        },
                        "thoughtSignature": "synthetic-signature",
                    }
                ],
            },
            {
                "role": "model",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "run_command",
                            "response": {"result": "synthetic"},
                        }
                    }
                ],
            },
        ],
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.5},
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "run_command",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {"CommandLine": {"type": "STRING"}},
                            "required": ["CommandLine"],
                        },
                    }
                ]
            }
        ],
    }
    original = deepcopy(body)
    await GeminiHandler().generate_content(route, body, stream)
    args = capture.call_args.kwargs
    assert args["model"] == "custom_openai/pinned-model"
    assert args["api_key"] == "synthetic-upstream-key"
    assert args["api_base"] == str(api_base)
    assert args["systemInstruction"] == body["systemInstruction"]
    expected_contents = deepcopy(body["contents"])
    expected_contents[2]["role"] = "user"
    assert args["contents"] == expected_contents
    assert args["tools"] == body["tools"]
    assert args["config"] == body["generationConfig"]
    assert body == original


async def test_agy_tool_result_survives_real_litellm_conversion(monkeypatch):
    api_base = Url("http://synthetic.invalid/v1")
    key = SecretRef("synthetic")
    provider = Provider("fixture", ProviderKind.OPENAI, api_base, key, ProviderState.VERIFIED)
    route = ResolvedRoute(
        route_id=RouteId("route"),
        provider=provider,
        model=Model(ProviderKind.OPENAI, ModelRef("custom_openai/pinned-model"), api_base, key),
    )
    capture = AsyncMock(
        return_value=ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "DONE"},
                    "finish_reason": "stop",
                }
            ]
        )
    )
    monkeypatch.setattr("litellm.acompletion", capture)
    body = {
        "systemInstruction": {"parts": [{"text": "SYSTEM_MARKER"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "USER_MARKER"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "call-fixture",
                            "name": "run_command",
                            "args": {"CommandLine": "printf AGY_TOOL_OK"},
                        }
                    }
                ],
            },
            {
                "role": "model",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "call-fixture",
                            "name": "run_command",
                            "response": {
                                "output": "The command exited with code 0.\nOutput:\nAGY_TOOL_OK\n"
                            },
                        }
                    }
                ],
            },
        ],
    }
    original = deepcopy(body)
    response = await GeminiHandler().generate_content(route, body, False)
    messages = capture.call_args.kwargs["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[0]["content"] == "SYSTEM_MARKER"
    assert "AGY_TOOL_OK" in messages[-1]["content"]
    assert messages[-1]["tool_call_id"] == messages[-2]["tool_calls"][0]["id"]
    assert response["candidates"][0]["content"]["parts"][0]["text"] == "DONE"
    assert body == original
