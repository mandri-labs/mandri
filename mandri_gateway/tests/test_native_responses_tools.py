import json
from dataclasses import replace

import httpx
import litellm
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.core.ids import ProviderKind
from mandri.gateway.litellm_adapter import ResponsesHandler

from mandri_gateway.tests.test_web_search_options import route


@pytest.mark.parametrize("kind", [ProviderKind.OPENAI, ProviderKind.OPENROUTER])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("search_type", ["web_search", "web_search_preview"])
async def test_native_responses_preserves_search_access_and_function_tools(
    monkeypatch, kind, stream, search_type
):
    GLOBAL_LOGGING_WORKER.start()
    monkeypatch.setattr(
        litellm.utils, "supports_native_streaming", lambda model, custom_llm_provider: True
    )
    captured = []

    async def send(client, request, **kwargs):
        assert request.url.host == "127.0.0.1"
        assert request.url.path == "/v1/responses"
        captured.append(json.loads(request.content))
        response = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "test-model",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        if not stream:
            return httpx.Response(200, request=request, json=response)
        events = [
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_test",
                "output_index": 0,
                "content_index": 0,
                "delta": "OK",
            },
            {"type": "response.completed", "sequence_number": 2, "response": response},
        ]
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
            ),
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    selected = route(kind)
    if kind == ProviderKind.OPENAI:
        selected = replace(selected, model=replace(selected.model, model_ref="openai/test-model"))
    tools = [
        {"type": search_type, "external_web_access": False, "search_context_size": "medium"},
        {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
    ]
    response = await ResponsesHandler().responses(
        selected,
        {"input": "test", "tools": tools, "tool_choice": "auto", "stream": stream},
    )
    if stream:
        events = [event async for event in response]
        assert any(
            event.type == "response.output_text.delta" and event.delta == "OK" for event in events
        )
        response = next(event.response for event in events if event.type == "response.completed")
    await GLOBAL_LOGGING_WORKER.flush()
    assert response.status == "completed"
    assert response.output[0].content[0].text == "OK"
    assert len(captured) == 1
    assert captured[0]["tools"] == tools
    assert captured[0]["tool_choice"] == "auto"
    assert captured[0].get("stream", False) is stream
