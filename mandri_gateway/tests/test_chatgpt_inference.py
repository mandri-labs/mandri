import asyncio
import base64
import json
from dataclasses import replace

import httpx
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.core.ids import ModelRef, ProviderKind, SecretRef, Url
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.gateway import privacy_transport
from mandri.gateway.chatgpt_adapter import complete_response
from mandri.gateway.errors.upstream import UpstreamError
from mandri.gateway.litellm_adapter import (
    AnthropicHandler,
    GeminiHandler,
    OpenAIHandler,
    ResponsesHandler,
)
from mandri.gateway.privacy_egress import EgressGuard
from mandri.gateway.privacy_response import restore_json

from mandri_gateway.tests.test_privacy_transport import scope


@pytest.fixture(autouse=True)
async def logging_worker():
    GLOBAL_LOGGING_WORKER.start()
    yield
    await GLOBAL_LOGGING_WORKER.flush()


def token():
    claim = (
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "synthetic-account",
                        "chatgpt_data_residency": "eu",
                    }
                }
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    return f"header.{claim}.signature"


def route(protected):
    guard = scope().guard
    model = replace(
        guard.route.model,
        provider=ProviderKind.CHATGPT,
        api_key=SecretRef(token()),
        api_base=Url("https://chatgpt.com/backend-api/codex"),
        model_ref=ModelRef("openai/gpt-5.4"),
    )
    resolved = replace(
        guard.route,
        model=model,
        provider=replace(
            guard.route.provider,
            kind=ProviderKind.CHATGPT,
            api_base=model.api_base,
            api_key=model.api_key,
        ),
        privacy_mode=PrivacyMode.SURROGATE if protected else PrivacyMode.NONE,
    )
    guard = EgressGuard(resolved, guard.engine)
    return resolved, guard


def completed(text):
    return {
        "id": "resp_synthetic",
        "object": "response",
        "created_at": 1750000000,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": "",
        "model": "gpt-5.4",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "store": False,
        "output": [
            {
                "id": "msg_synthetic",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 14,
        },
    }


@pytest.mark.parametrize("sparse_terminal", [False, True])
@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.6-luna"])
@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("protocol", ["responses", "chat", "anthropic", "gemini"])
async def test_sdk_bridge_collects_complete_chatgpt_response(
    monkeypatch, protected, protocol, model, sparse_terminal
):
    resolved, guard = route(protected)
    resolved = replace(
        resolved, model=replace(resolved.model, model_ref=ModelRef("openai/" + model))
    )
    guard = EgressGuard(resolved, guard.engine)
    private = "owner-128@example.invalid"
    alias = guard.engine.protect_text(private) if protected else private
    consumed = []
    closed = []
    seen = []
    terminal = completed(alias)
    item = terminal["output"][0]
    if sparse_terminal:
        terminal["output"] = []
    body = (
        'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
        + "data: "
        + json.dumps({"type": "response.output_item.done", "output_index": 0, "item": item})
        + "\n\ndata: "
        + json.dumps({"type": "response.completed", "response": terminal})
        + "\n\n"
    ).encode()

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(body), 7):
                consumed.append(offset)
                yield body[offset : offset + 7]

        async def aclose(self):
            closed.append(True)

    async def send(client, request):
        seen.append(request)
        payload = json.loads(request.content)
        assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
        assert payload["stream"] is True
        assert payload["store"] is False
        assert payload["model"] == model
        assert request.headers["ChatGPT-Account-ID"] == "synthetic-account"
        assert request.headers["Authorization"] == f"Bearer {token()}"
        if protected:
            assert private.encode() not in request.content
        return httpx.Response(
            200, request=request, headers={"content-type": "text/event-stream"}, stream=Chunks()
        )

    privacy_transport.install_transport_observers()
    monkeypatch.setattr(privacy_transport, "_ASYNC_SEND", send)
    kwargs = {"guard": guard} if protected else {}
    if protocol == "responses":
        result = await ResponsesHandler().responses(
            resolved, {"input": alias, "stream": False}, **kwargs
        )
    elif protocol == "chat":
        result = await OpenAIHandler().chat_completions(
            resolved, {"messages": [{"role": "user", "content": alias}]}, **kwargs
        )
    elif protocol == "anthropic":
        result = await AnthropicHandler().messages(
            resolved, {"messages": [{"role": "user", "content": alias}], "max_tokens": 64}, **kwargs
        )
    else:
        result = await GeminiHandler().generate_content(
            resolved, {"contents": [{"role": "user", "parts": [{"text": alias}]}]}, False, **kwargs
        )
    value = result if isinstance(result, dict) else result.model_dump(mode="json")
    assert len(consumed) == len(range(0, len(body), 7))
    assert closed
    assert len(seen) == 1
    assert alias in json.dumps(value)
    if protected:
        assert guard.sends == 1
        assert private in json.dumps(restore_json(value, guard.engine))


async def test_truncated_stream_does_not_return_partial_output(monkeypatch):
    resolved, guard = route(True)

    async def send(client, request):
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content='data: {"type":"response.output_text.delta","delta":"partial"}\n\n',
        )

    privacy_transport.install_transport_observers()
    monkeypatch.setattr(privacy_transport, "_ASYNC_SEND", send)
    with pytest.raises(UpstreamError, match="complete response"):
        await ResponsesHandler().responses(resolved, {"input": "Synthetic"}, guard=guard)


async def test_streaming_cannot_be_enabled_without_completion_adapter():
    resolved, guard = route(True)
    request = httpx.Request(
        "POST",
        str(resolved.model.api_base) + "/responses",
        json={"input": "Synthetic", "stream": True, "model": "gpt-5.4"},
    )
    with pytest.raises(ProtectionError, match="streaming is disabled"):
        await guard.check(request)


async def test_real_http_transport_collects_chatgpt_response():
    resolved, _ = route(False)
    received = []
    wire = (
        "data: "
        + json.dumps({"type": "response.completed", "response": completed("Hello")})
        + "\n\n"
    ).encode()

    async def serve(reader, writer):
        headers = await reader.readuntil(b"\r\n\r\n")
        length = next(
            int(line.split(b":", 1)[1])
            for line in headers.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        )
        received.append(await reader.readexactly(length))
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
            + str(len(wire)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + wire
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    base = Url(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/codex")
    resolved = replace(
        resolved,
        model=replace(resolved.model, api_base=base, model_ref=ModelRef("openai/gpt-5.6-luna")),
    )
    async with server:
        result = await OpenAIHandler().chat_completions(
            resolved, {"messages": [{"role": "user", "content": "Hello"}], "stream": True}
        )
    assert received
    assert result.choices[0].message.content == "Hello"


@pytest.mark.parametrize("terminal_kind", ["response.completed", "response.incomplete"])
async def test_sparse_terminal_recovers_complete_tool_items_in_index_order(terminal_kind):
    reasoning = {"id": "reason_1", "type": "reasoning", "summary": []}
    call = {
        "id": "tool_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": '{"value":"synthetic"}',
        "status": "completed",
    }
    terminal = {**completed(""), "output": [], "status": terminal_kind.split(".")[1]}
    events = [
        {"type": "response.output_item.done", "output_index": 1, "item": call},
        {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
        {"type": terminal_kind, "response": terminal},
    ]
    response = httpx.Response(
        200, content="".join("data: " + json.dumps(event) + "\n\n" for event in events)
    )
    result = await complete_response(response)
    assert result["output"] == [reasoning, call]


@pytest.mark.parametrize("ending", ["missing", "failed", "gap"])
async def test_completed_items_do_not_hide_unfinished_or_invalid_response(ending):
    events = [
        {
            "type": "response.output_item.done",
            "output_index": 1 if ending == "gap" else 0,
            "item": completed("Hello")["output"][0],
        }
    ]
    if ending == "failed":
        events.append({"type": "response.failed"})
    elif ending == "gap":
        events.append({"type": "response.completed", "response": {**completed(""), "output": []}})
    response = httpx.Response(
        200, content="".join("data: " + json.dumps(event) + "\n\n" for event in events)
    )
    with pytest.raises(UpstreamError):
        await complete_response(response)
