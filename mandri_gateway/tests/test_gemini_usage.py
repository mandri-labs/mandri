import json
from types import MappingProxyType
from unittest.mock import AsyncMock

import httpx
import pytest
from litellm.google_genai.adapters.transformation import (
    GoogleGenAIAdapter,
    GoogleGenAIStreamWrapper,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.utils import ModelResponse, ModelResponseStream
from mandri.gateway.gemini_completion import generate_content


def chunk(*, text=None, finish=None, usage=None):
    choices = (
        []
        if usage is not None
        else [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}]
    )
    return ModelResponseStream(choices=choices, usage=usage)


async def chunks(*values):
    for value in values:
        yield value


def test_litellm_usage_only_regression_is_reproduced():
    adapter = GoogleGenAIAdapter()
    state = GoogleGenAIStreamWrapper(chunks())
    terminal = chunk(usage={"prompt_tokens": 123, "completion_tokens": 17, "total_tokens": 140})
    assert adapter.translate_streaming_completion_to_generate_content(terminal, state) is None
    finish = adapter.translate_streaming_completion_to_generate_content(chunk(finish="stop"), state)
    assert finish["usageMetadata"]["totalTokenCount"] == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {
            "prompt_tokens": 123,
            "completion_tokens": 17,
            "total_tokens": 140,
            "prompt_tokens_details": {"cached_tokens": 32},
            "completion_tokens_details": {"reasoning_tokens": 5},
            "cost": 0.001,
        },
    ],
)
async def test_usage_mapping_preserves_known_zero_and_absence(monkeypatch, stream, usage):
    response = ModelResponse(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }
        ],
        usage=usage,
    )
    values = [chunk(text="OK"), chunk(finish="stop")]
    if usage is not None:
        values.append(chunk(usage=usage))
    call = AsyncMock(return_value=chunks(*values) if stream else response)
    monkeypatch.setattr("litellm.acompletion", call)
    result = await generate_content(
        model="custom_openai/fixture",
        stream=stream,
        contents=[{"role": "user", "parts": [{"text": "fixture"}]}],
    )
    payloads = [json.loads(raw.decode()[6:]) async for raw in result] if stream else [result]
    if stream:
        assert call.call_args.kwargs["stream_options"] == {"include_usage": True}
        assert all("usageMetadata" not in item for item in payloads[:2])
    observed = [item["usageMetadata"] for item in payloads if "usageMetadata" in item]
    if usage is None:
        assert observed == []
    elif usage["total_tokens"] == 0:
        assert observed == [
            {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}
        ]
    else:
        assert observed == [
            {
                "promptTokenCount": 123,
                "candidatesTokenCount": 12,
                "totalTokenCount": 140,
                "thoughtsTokenCount": 5,
                "cachedContentTokenCount": 32,
            }
        ]
    assert all("cost" not in item for item in payloads)


async def test_real_litellm_http_stream_preserves_usage_only_tail(monkeypatch):
    requests = []

    async def send(client, request, **kwargs):
        assert request.url.host == "synthetic.invalid"
        requests.append(json.loads(request.content))
        events = [
            chunk(text="OK"),
            chunk(finish="stop"),
            chunk(
                usage={
                    "prompt_tokens": 123,
                    "completion_tokens": 17,
                    "total_tokens": 140,
                }
            ),
        ]
        wire = "".join("data: " + event.model_dump_json() + "\n\n" for event in events)
        return httpx.Response(
            200,
            content=wire + "data: [DONE]\n\n",
            request=request,
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    try:
        result = await generate_content(
            model="custom_openai/fixture",
            stream=True,
            api_base="http://synthetic.invalid/v1",
            api_key="fixture",
            contents=[{"role": "user", "parts": [{"text": "fixture"}]}],
        )
        payloads = [json.loads(raw.decode()[6:]) async for raw in result]
        assert requests[0]["stream_options"] == {"include_usage": True}
        usage = payloads[-1]["usageMetadata"]
        assert usage["promptTokenCount"] == 123
        assert usage["candidatesTokenCount"] == 17
        assert usage["totalTokenCount"] == 140
    finally:
        await GLOBAL_LOGGING_WORKER.flush()


async def test_stream_failure_is_not_silently_swallowed(monkeypatch):
    async def failing():
        yield chunk(text="OK")
        raise RuntimeError("fixture failure")

    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=failing()))
    result = await generate_content(model="custom_openai/fixture", stream=True, contents=[])
    with pytest.raises(RuntimeError, match="fixture failure"):
        _ = [raw async for raw in result]


@pytest.mark.parametrize("stream", [False, True])
async def test_native_gemini_conversion_remains_native(monkeypatch, stream):
    expected = {"usageMetadata": {"promptTokenCount": 123, "totalTokenCount": 140}}
    call = AsyncMock(return_value=expected)
    name = "agenerate_content_stream" if stream else "agenerate_content"
    monkeypatch.setattr("litellm.google_genai." + name, call)
    result = await generate_content(model="gemini/gemini-2.5-pro", stream=stream, contents=[])
    assert result is expected
    assert call.call_args.kwargs == {"model": "gemini/gemini-2.5-pro", "contents": []}


@pytest.mark.parametrize("arguments", ["", '{"path":"fixture"}'])
async def test_stream_tool_call_content_survives_usage_translation(monkeypatch, arguments):
    tool = ModelResponseStream(
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"name": "read_file", "arguments": arguments}}
                    ]
                },
            }
        ]
    )
    call = AsyncMock(
        return_value=chunks(
            tool,
            chunk(finish="tool_calls"),
            chunk(
                usage={
                    "prompt_tokens": 123,
                    "completion_tokens": 17,
                    "total_tokens": 140,
                }
            ),
        )
    )
    monkeypatch.setattr("litellm.acompletion", call)
    result = await generate_content(model="custom_openai/fixture", stream=True, contents=[])
    payloads = [json.loads(raw.decode()[6:]) async for raw in result]
    calls = [
        part["functionCall"]
        for payload in payloads
        for candidate in payload.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
        if "functionCall" in part
    ]
    assert calls == [{"name": "read_file", "args": json.loads(arguments) if arguments else {}}]
    assert [
        item["usageMetadata"]["totalTokenCount"] for item in payloads if "usageMetadata" in item
    ] == [140]


async def test_stream_accepts_read_only_adapter_payload(monkeypatch):
    payload = MappingProxyType({"candidates": [], "usageMetadata": {"totalTokenCount": 0}})
    monkeypatch.setattr(
        GoogleGenAIAdapter,
        "translate_streaming_completion_to_generate_content",
        lambda self, chunk, state: payload,
    )
    terminal = chunk(usage={"prompt_tokens": 123, "completion_tokens": 17, "total_tokens": 140})
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=chunks(terminal)))
    result = await generate_content(model="custom_openai/fixture", stream=True, contents=[])
    observed = [json.loads(raw.decode()[6:]) async for raw in result]
    assert observed == [
        {
            "candidates": [],
            "usageMetadata": {
                "promptTokenCount": 123,
                "candidatesTokenCount": 17,
                "totalTokenCount": 140,
            },
        }
    ]
    assert payload["usageMetadata"] == {"totalTokenCount": 0}
