import copy
import json

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.privacy_response import restore_json, restore_sse
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope


@pytest.fixture
def protected():
    engine = SurrogateEngine(SurrogateScope("synthetic-response-test"))
    engine.register_root("/home/private/Project", "/home/orchard/Service")
    email = engine.protect_text("alice@example.fr")
    tool = engine.register("read_private_file")
    return engine, email, tool


def sse(payload, *, event=None, identity=None, newline="\n"):
    lines = []
    if event:
        lines.append("event: " + event)
    if identity:
        lines.append("id: " + identity)
    lines.append(
        "data: "
        + (payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False))
    )
    return (newline.join(lines) + newline + newline).encode()


async def collect(wire, engine, protocol, chunk_size=11):
    async def source():
        for offset in range(0, len(wire), chunk_size):
            yield wire[offset : offset + chunk_size]

    return b"".join([part async for part in restore_sse(source(), engine, protocol)])


def events(wire):
    result = []
    for frame in wire.decode().split("\n\n"):
        data = [
            line.removeprefix("data: ") for line in frame.splitlines() if line.startswith("data:")
        ]
        if data and data != ["[DONE]"]:
            result.append(json.loads("\n".join(data)))
    return result


def test_complete_chat_restores_content_and_tool_targets_but_preserves_protocol_controls(protected):
    engine, email, tool = protected
    payload = {
        "id": email,
        "model": email,
        "created": 123,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": email,
                    "tool_calls": [
                        {
                            "id": email,
                            "type": "function",
                            "function": {
                                "name": tool,
                                "arguments": json.dumps(
                                    {
                                        "path": "/home/orchard/Service/mandri/src/new.py",
                                        "email": email,
                                    }
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 42},
    }
    before = copy.deepcopy(payload)
    result = restore_json(payload, engine)
    assert result["id"] == result["model"] == email
    assert result["usage"] == payload["usage"]
    message = result["choices"][0]["message"]
    assert message["content"] == "alice@example.fr"
    call = message["tool_calls"][0]
    assert call["id"] == email and call["function"]["name"] == "read_private_file"
    assert json.loads(call["function"]["arguments"]) == {
        "path": "/home/private/Project/mandri/src/new.py",
        "email": "alice@example.fr",
    }
    assert payload == before


def test_complete_responses_with_null_error_still_restores_every_output_item(protected):
    engine, email, tool = protected
    payload = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "error": None,
        "output": [
            {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": email}]},
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": email}],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": tool,
                "arguments": json.dumps({"email": email}),
            },
        ],
    }
    result = restore_json(payload, engine)
    assert result["error"] is None
    assert result["output"][0]["content"][0]["text"] == "alice@example.fr"
    assert result["output"][1]["summary"][0]["text"] == "alice@example.fr"
    assert json.loads(result["output"][2]["arguments"]) == {"email": "alice@example.fr"}


@pytest.mark.parametrize("protocol", ["anthropic", "gemini"])
def test_complete_structured_provider_content_and_args_restore(protected, protocol):
    engine, email, tool = protected
    if protocol == "anthropic":
        payload = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": email},
                {"type": "tool_use", "id": "tool_1", "name": tool, "input": {"email": email}},
            ],
        }
        result = restore_json(payload, engine)
        assert result["content"][0]["text"] == "alice@example.fr"
        assert result["content"][1]["input"] == {"email": "alice@example.fr"}
        assert result["content"][1]["id"] == "tool_1"
    else:
        payload = {
            "candidates": [
                {
                    "index": 0,
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": email},
                            {"functionCall": {"name": tool, "args": {"email": email}}},
                        ],
                    },
                }
            ],
            "usageMetadata": {"totalTokenCount": 27},
        }
        result = restore_json(payload, engine)
        parts = result["candidates"][0]["content"]["parts"]
        assert parts[0]["text"] == "alice@example.fr"
        assert parts[1]["functionCall"] == {
            "name": "read_private_file",
            "args": {"email": "alice@example.fr"},
        }
        assert result["usageMetadata"] == payload["usageMetadata"]


def test_citation_offsets_are_recomputed_against_restored_text(protected):
    engine, email, _ = protected
    text = f"Contact {email}. More details."
    start, end = len("Contact "), len("Contact " + email)
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "start_index": start,
                                "end_index": end,
                                "url": "https://example.com",
                                "title": email,
                            }
                        ],
                    }
                ],
            }
        ]
    }
    result = restore_json(payload, engine)
    part = result["output"][0]["content"][0]
    citation = part["annotations"][0]
    assert part["text"][citation["start_index"] : citation["end_index"]] == "alice@example.fr"
    assert citation["title"] == "alice@example.fr"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "image", "source": {"type": "base64", "data": "abc"}}]
                    }
                }
            ]
        },
        {
            "type": "message",
            "content": [{"type": "thinking", "thinking": "private", "signature": "signed-value"}],
        },
        {
            "candidates": [
                {"content": {"parts": [{"text": "private", "thoughtSignature": "signed-value"}]}}
            ]
        },
        {"choices": [{"logprobs": {"content": []}, "message": {"content": "text"}}]},
        {"output": [{"type": "computer_call", "action": {}}]},
        {"output": [{"type": ["future"]}]},
        {"choices": [{"message": {"content": [{"type": {"future": True}}]}}]},
    ],
)
def test_unknown_response_surfaces_pass_through(protected, payload):
    assert restore_json(payload, protected[0]) == payload


@pytest.mark.parametrize("arguments", ["{", "{} trailing", '"string"', "[]", '{"a":1,"a":2}'])
def test_complete_tool_arguments_require_valid_unambiguous_object_json(protected, arguments):
    with pytest.raises(ProtectionError):
        restore_json(
            {"output": [{"type": "function_call", "name": "tool", "arguments": arguments}]},
            protected[0],
        )


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 113, 100000])
async def test_chat_live_text_restoration_preserves_ids_usage_and_event_order(
    protected, chunk_size
):
    engine, email, _ = protected
    messages = [
        {
            "id": "chat_1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "é " + email[:13]},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat_1",
            "choices": [{"index": 0, "delta": {"content": email[13:]}, "finish_reason": None}],
        },
        {"id": "chat_1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"id": "chat_1", "choices": [], "usage": {"total_tokens": 99}},
    ]
    wire = b"".join(
        sse(message, identity=f"event-{index}", newline="\r\n")
        for index, message in enumerate(messages)
    ) + sse("[DONE]", newline="\r\n")
    result = await collect(wire, engine, GatewayProtocol.CHAT, chunk_size)
    output = events(result)
    assert len(output) == len(messages)
    assert (
        "".join(
            choice.get("delta", {}).get("content", "")
            for item in output
            for choice in item["choices"]
        )
        == "é alice@example.fr"
    )
    assert [line for line in result.decode().splitlines() if line.startswith("id:")] == [
        f"id: event-{index}" for index in range(4)
    ]
    assert output[-1]["usage"] == {"total_tokens": 99}
    assert b"data: [DONE]" in result


async def test_chat_parallel_tool_json_is_held_and_restored_independently(protected):
    engine, email, tool = protected
    first = json.dumps({"path": "/home/orchard/Service/mandri/src/new.py", "email": email})
    second = json.dumps({"email": email, "operation": "verify"})
    messages = []
    for index in range(max(len(first), len(second))):
        calls = []
        for lane, source in enumerate((first, second)):
            if index < len(source):
                function = {"arguments": source[index]}
                if index == 0:
                    function["name"] = tool
                calls.append(
                    {"index": lane, "id": f"call-{lane}", "type": "function", "function": function}
                )
        messages.append(
            {"choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}]}
        )
    messages.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    wire = b"".join(sse(message) for message in messages) + sse("[DONE]")
    output = events(await collect(wire, engine, "chat", 3))
    arguments = {0: "", 1: ""}
    names = {0: "", 1: ""}
    for item in output:
        for call in item["choices"][0]["delta"].get("tool_calls", []):
            arguments[call["index"]] += call["function"].get("arguments", "")
            names[call["index"]] += call["function"].get("name", "")
    assert json.loads(arguments[0]) == {
        "path": "/home/private/Project/mandri/src/new.py",
        "email": "alice@example.fr",
    }
    assert json.loads(arguments[1]) == {"email": "alice@example.fr", "operation": "verify"}
    assert names == {0: "read_private_file", 1: "read_private_file"}


async def test_responses_interleaved_reasoning_text_and_final_snapshots_restore_once(protected):
    engine, email, _ = protected
    output_item = {
        "id": "msg_1",
        "type": "message",
        "content": [{"type": "output_text", "text": email}],
    }
    frames = [
        {"type": "response.created", "response": {"id": "resp_1", "output": [], "error": None}},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "content_index": 0,
            "delta": email[:12],
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "summary_index": 0,
            "delta": "Consider " + email[:9],
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "content_index": 0,
            "delta": email[12:],
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "summary_index": 0,
            "delta": email[9:],
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "content_index": 0,
            "text": email,
        },
        {"type": "response.output_item.done", "output_index": 0, "item": output_item},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "output": [output_item],
                "error": None,
                "usage": {"total_tokens": 29},
            },
        },
    ]
    for index, frame in enumerate(frames):
        frame["sequence_number"] = index
    result = events(
        await collect(
            b"".join(sse(frame, event=frame["type"]) for frame in frames), engine, "responses", 1
        )
    )
    assert [item["sequence_number"] for item in result] == list(range(len(frames)))
    text = "".join(
        item.get("delta", "") for item in result if item["type"] == "response.output_text.delta"
    )
    thinking = "".join(
        item.get("delta", "")
        for item in result
        if item["type"] == "response.reasoning_summary_text.delta"
    )
    assert text == "alice@example.fr" and thinking == "Consider alice@example.fr"
    assert result[-1]["response"]["output"][0]["content"][0]["text"] == "alice@example.fr"
    assert result[-1]["response"]["usage"] == {"total_tokens": 29}


async def test_anthropic_text_and_tool_blocks_close_without_reordering(protected):
    engine, email, tool = protected
    arguments = json.dumps({"email": email})
    frames = [
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "id": "msg_1",
                "content": [],
                "usage": {"input_tokens": 7},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": email[:10]},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": email[10:]},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "call_1", "name": tool, "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": arguments[:12]},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": arguments[12:]},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 12},
        },
        {"type": "message_stop"},
    ]
    result = events(
        await collect(
            b"".join(sse(frame, event=frame["type"]) for frame in frames), engine, "anthropic", 2
        )
    )
    assert [item["type"] for item in result] == [item["type"] for item in frames]
    assert result[5]["content_block"]["name"] == "read_private_file"
    text = "".join(item.get("delta", {}).get("text", "") for item in result)
    args = "".join(item.get("delta", {}).get("partial_json", "") for item in result)
    assert text == "alice@example.fr" and json.loads(args) == {"email": "alice@example.fr"}
    assert result[-2]["usage"] == {"output_tokens": 12}


async def test_gemini_candidates_and_complete_function_calls_restore(protected):
    engine, email, tool = protected
    frames = [
        {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": email[:15]}]}}
            ]
        },
        {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": email[15:]}]}}
            ]
        },
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": tool, "args": {"email": email}}}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"totalTokenCount": 22},
        },
    ]
    result = events(await collect(b"".join(sse(frame) for frame in frames), engine, "gemini", 1))
    parts = [
        part
        for item in result
        for candidate in item["candidates"]
        for part in candidate["content"]["parts"]
    ]
    assert "".join(part.get("text", "") for part in parts) == "alice@example.fr"
    assert parts[-1]["functionCall"] == {
        "name": "read_private_file",
        "args": {"email": "alice@example.fr"},
    }


@pytest.mark.parametrize(
    "wire,protocol",
    [
        (b"", "chat"),
        (b"data: [DONE]\n\n", "chat"),
        (b"data: {bad}\n\n", "chat"),
        (b"data: {}", "chat"),
        (b"data: \xff\n\n", "chat"),
        (b"data: \xc3", "chat"),
        (
            sse({"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]}),
            "chat",
        ),
        (
            sse({"type": "response.output_text.delta", "item_id": "m", "delta": "hello"}),
            "responses",
        ),
        (
            sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                }
            ),
            "anthropic",
        ),
        (sse({"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}), "gemini"),
    ],
)
async def test_empty_truncated_invalid_and_unclosed_streams_fail(protected, wire, protocol):
    with pytest.raises(ProtectionError) as caught:
        await collect(wire, protected[0], protocol, 1)
    assert caught.value.code == "privacy_response_invalid"


async def test_incomplete_tool_json_never_reaches_harness(protected):
    engine, _, tool = protected
    frames = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": tool, "arguments": '{"path":'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    seen = []

    async def source():
        for frame in frames:
            yield sse(frame)
        yield sse("[DONE]")

    with pytest.raises(ProtectionError):
        async for chunk in restore_sse(source(), engine, "chat"):
            seen.append(chunk)
    assert seen == []


async def test_downstream_cancellation_closes_upstream_iterator(protected):
    engine, _, _ = protected
    closed = False

    async def source():
        nonlocal closed
        try:
            yield sse(
                {
                    "choices": [
                        {"index": 0, "delta": {"content": "plain output "}, "finish_reason": None}
                    ]
                }
            )
            yield sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield sse("[DONE]")
        finally:
            closed = True

    output = restore_sse(source(), engine, "chat")
    await anext(output)
    await output.aclose()
    assert closed


async def test_restored_upstream_error_keeps_code_and_message(protected):
    engine, email, _ = protected
    payload = {
        "error": {"type": "provider_error", "code": "invalid", "message": "Unknown " + email}
    }
    result = events(await collect(sse(payload) + sse("[DONE]"), engine, "chat", 1))
    assert result == [
        {
            "error": {
                "type": "provider_error",
                "code": "invalid",
                "message": "Unknown alice@example.fr",
            }
        }
    ]


@pytest.mark.parametrize("field", ["tool_calls", "function_call", "annotations"])
def test_chat_sdk_nullable_tool_fields_preserve_null_and_restore_content(protected, field):
    engine, email, _ = protected
    payload = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": email,
                    field: None,
                },
                "finish_reason": "stop",
            }
        ]
    }
    restored = restore_json(payload, engine)
    assert restored["choices"][0]["message"][field] is None
    assert restored["choices"][0]["message"]["content"] == "alice@example.fr"


@pytest.mark.parametrize("field", ["tool_calls", "function_call", "annotations"])
async def test_streamed_sdk_nullable_fields_preserve_null_and_finish_choice(protected, field):
    engine, email, _ = protected
    wire = sse(
        {"choices": [{"index": 0, "delta": {"content": email, field: None}, "finish_reason": None}]}
    )
    wire += sse({"choices": [{"index": 0, "delta": {field: None}, "finish_reason": "stop"}]})
    wire += sse("[DONE]")
    output = events(await collect(wire, engine, GatewayProtocol.CHAT))
    assert output[0]["choices"][0]["delta"]["content"] == "alice@example.fr"
    assert all(item["choices"][0]["delta"][field] is None for item in output)


@pytest.mark.parametrize("legacy", [False, True])
async def test_streamed_sdk_nullable_function_fields_preserve_fragment_lanes(protected, legacy):
    engine, email, tool = protected
    arguments = json.dumps({"email": email})
    functions = [
        {"name": tool[:4], "arguments": None},
        {"name": tool[4:], "arguments": arguments[:8]},
        {"name": None, "arguments": arguments[8:]},
        {"name": None, "arguments": None},
    ]
    deltas = [
        {"function_call": function}
        if legacy
        else {"tool_calls": [{"index": 0, "id": None, "type": None, "function": function}]}
        for function in functions
    ]
    wire = b"".join(
        sse({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) for delta in deltas
    )
    wire += sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    wire += sse("[DONE]")
    output = events(await collect(wire, engine, GatewayProtocol.CHAT, chunk_size=1))
    restored = [
        item["choices"][0]["delta"]["function_call"]
        if legacy
        else item["choices"][0]["delta"]["tool_calls"][0]["function"]
        for item in output[:-1]
    ]
    assert "".join(item.get("name") or "" for item in restored) == "read_private_file"
    assert json.loads("".join(item.get("arguments") or "" for item in restored)) == {
        "email": "alice@example.fr"
    }
    assert restored[0]["arguments"] is None
    assert restored[-1] == {"name": None, "arguments": None}


async def test_streamed_sdk_nullable_function_metadata_preserved(protected):
    engine, _, _ = protected
    delta = {"tool_calls": [{"index": 0, "id": "call_0", "type": None, "function": None}]}
    wire = sse({"choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}]})
    wire += sse("[DONE]")
    output = events(await collect(wire, engine, GatewayProtocol.CHAT))
    assert output[0]["choices"][0]["delta"] == delta


@pytest.mark.parametrize(
    "call",
    [
        {"function": False},
        {"function": {"name": 42}},
        {"function": {"arguments": False}},
    ],
)
async def test_streamed_nullable_fields_do_not_accept_malformed_nonnull_values(protected, call):
    engine, _, _ = protected
    wire = sse(
        {"choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": "stop"}]}
    ) + sse("[DONE]")
    with pytest.raises(ProtectionError):
        await collect(wire, engine, GatewayProtocol.CHAT)


async def test_litellm_usage_only_choice_after_finish_preserves_statistics(protected):
    engine, email, _ = protected
    terminal = {"choices": [{"index": 0, "delta": {"content": email}, "finish_reason": "stop"}]}
    usage = {
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
                "delta": {
                    "content": None,
                    "role": None,
                    "function_call": None,
                    "tool_calls": None,
                    "audio": None,
                },
                "logprobs": None,
            }
        ],
        "usage": {"completion_tokens": 2, "prompt_tokens": 9, "total_tokens": 11},
    }
    output = events(
        await collect(sse(terminal) + sse(usage) + sse("[DONE]"), engine, GatewayProtocol.CHAT)
    )
    assert output[0]["choices"][0]["delta"]["content"] == "alice@example.fr"
    assert output[1] == usage


@pytest.mark.parametrize(
    "delta",
    [{"content": "late text"}, {"tool_calls": []}, {"new_content": None}, {"role": "assistant"}],
)
async def test_finished_choice_cannot_reopen_through_usage_frame(protected, delta):
    engine, email, _ = protected
    terminal = {"choices": [{"index": 0, "delta": {"content": email}, "finish_reason": "stop"}]}
    usage = {
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        "usage": {"total_tokens": 11},
    }
    with pytest.raises(ProtectionError, match="already completed choice"):
        await collect(sse(terminal) + sse(usage) + sse("[DONE]"), engine, GatewayProtocol.CHAT)


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
async def test_responses_search_progress_and_results_restore_nested_content(protected, chunk_size):
    engine, email, _ = protected
    item = {
        "type": "web_search_call",
        "id": "ws_123",
        "status": "completed",
        "action": {
            "type": "search",
            "query": email,
            "sources": [{"type": "url", "url": "https://example.com/", "title": email}],
        },
    }
    wire = b"".join(
        sse(
            {
                "type": "response.web_search_call." + phase,
                "item_id": "ws_123",
                "output_index": 0,
                "sequence_number": index,
            }
        )
        for index, phase in enumerate(["in_progress", "searching", "completed"])
    )
    wire += sse({"type": "response.output_item.done", "output_index": 0, "item": item})
    wire += sse(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "object": "response",
                "status": "completed",
                "output": [item],
            },
        }
    )
    result = events(await collect(wire, engine, GatewayProtocol.RESPONSES, chunk_size))
    assert len(result) == 5
    for restored in [result[3]["item"], result[4]["response"]["output"][0]]:
        assert restored["id"] == "ws_123"
        assert restored["action"]["query"] == "alice@example.fr"
        assert restored["action"]["sources"][0]["title"] == "alice@example.fr"


async def test_large_sse_frame_has_no_surrogate_size_limit(protected):
    engine, email, _ = protected
    text = "ordinary text " * 160_000 + email
    wire = sse({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]})
    wire += sse("[DONE]")
    output = events(await collect(wire, engine, GatewayProtocol.CHAT))
    assert output[0]["choices"][0]["delta"]["content"] == engine.restore_text(text)


async def test_unknown_stream_events_and_signed_deltas_pass_through(protected):
    engine, email, _ = protected
    frames = [
        {"type": "message_start", "message": {"type": "message", "content": []}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": email},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "opaque-value"},
        },
        {"type": "future_event", "payload": {"data": "opaque-value"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    output = events(
        await collect(b"".join(sse(frame) for frame in frames), engine, GatewayProtocol.ANTHROPIC)
    )
    assert output[2]["delta"]["thinking"] == engine.restore_text(email)
    assert output[3] == frames[3]
    assert output[4] == frames[4]
