import json

import httpx
import pytest
from mandri.core.ids import ProviderKind
from mandri.gateway.litellm_adapter import ResponsesHandler
from mandri.gateway.response_stream import normalize_stream

from mandri_gateway.tests.test_responses_tool_history import assert_tool_sequence, output
from mandri_gateway.tests.test_web_search_options import route


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_reasoning_survives_responses_round_trip(monkeypatch, stream):
    requests = []
    reasoning = "Synthetic reasoning for this tool batch"
    commentary = "Checking details"
    tool_calls = [
        {
            "id": name,
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        for name in ("call_a", "call_b")
    ]

    def receive(request):
        body = json.loads(request.content)
        requests.append(body)
        assert_tool_sequence(body["messages"])
        if len(requests) == 2:
            assistants = [item for item in body["messages"] if item["role"] == "assistant"]
            assert len(assistants) == 1
            assert assistants[0]["reasoning_content"] == reasoning
            assert assistants[0]["content"] == [{"type": "text", "text": commentary}]
            assert [item["id"] for item in assistants[0]["tool_calls"]] == ["call_a", "call_b"]
        message = (
            {"reasoning_content": reasoning, "content": commentary, "tool_calls": tool_calls}
            if len(requests) == 1
            else {"content": "Done"}
        )
        response = {
            "id": f"chatcmpl-{len(requests)}",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", **message},
                    "finish_reason": "tool_calls" if len(requests) == 1 else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        if not body.get("stream"):
            return httpx.Response(200, json=response)
        deltas = (
            [
                {"role": "assistant", "reasoning_content": reasoning},
                {"tool_calls": [{"index": i, **item} for i, item in enumerate(tool_calls)]},
                {"content": commentary},
            ]
            if len(requests) == 1
            else [{"role": "assistant", "content": "Done"}]
        )
        chunks = []
        for delta in [*deltas, {}]:
            chunk = {
                **response,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": response["choices"][0]["finish_reason"]
                        if not delta
                        else None,
                    }
                ],
            }
            chunks.append(f"data: {json.dumps(chunk)}\n\n")
        return httpx.Response(
            200,
            content="".join(chunks) + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(receive)
    monkeypatch.setattr(httpx.AsyncClient, "_transport_for_url", lambda self, url: transport)
    history = [{"role": "user", "content": "Inspect the files"}]
    for index in range(2):
        response = await ResponsesHandler().responses(
            route(ProviderKind.OPENCODE_GO),
            {"input": history, "stream": stream, "reasoning": {"effort": "high"}},
        )
        if stream:
            events = [event async for event in normalize_stream(response)]
            assert events[-1]["type"] == "response.completed"
            items = [
                event["item"] for event in events if event["type"] == "response.output_item.done"
            ]
        else:
            items = response.model_dump()["output"]
        if index == 0:
            assert [item["type"] for item in items].count("reasoning") == 1
            history.extend([*items, output("call_a"), output("call_b")])
    assert len(requests) == 2
