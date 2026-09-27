import copy
from types import SimpleNamespace

import pytest
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.utils import ModelResponseStream
from mandri.gateway.response_stream import normalize_stream, sse_frame


async def normalize(events):
    async def source():
        for event in events:
            yield event

    return [event async for event in normalize_stream(source())]


def item(kind, identity, text=""):
    if kind == "reasoning":
        return {"type": kind, "id": identity, "summary": [{"type": "summary_text", "text": text}]}
    return {
        "type": kind,
        "id": identity,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def added(value, index=0):
    return {"type": "response.output_item.added", "output_index": index, "item": value}


def delta(kind, identity, text, index=0):
    return {
        "type": f"response.{kind}.delta",
        "item_id": identity,
        "output_index": index,
        "delta": text,
    }


@pytest.mark.parametrize("role_first", [False, True])
async def test_litellm_reasoning_and_message_keep_distinct_lifecycles(role_first):
    events = []
    if role_first:
        events.append(added({"type": "message", "id": "msg", "role": "assistant", "content": []}))
        events.append(
            {
                "type": "response.content_part.added",
                "item_id": "msg",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }
        )
    else:
        events.append(added({"type": "reasoning", "id": "rs", "summary": None}))
    events.extend(
        [
            delta("reasoning_summary_text", "rs", "Think "),
            delta("reasoning_summary_text", "rs", "carefully"),
            delta("output_text", "msg", "ANSWER"),
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": item("message", "msg", "ANSWER"),
            },
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        item("reasoning", "rs", "Think carefully"),
                        item("message", "msg", "ANSWER"),
                    ]
                },
            },
        ]
    )
    original = copy.deepcopy(events)
    result = await normalize(events)
    assert events == original
    active = None
    indices = {}
    done = []
    for event in result:
        if event["type"] == "response.output_item.added":
            active = event["item"]
            assert active["id"] not in indices
            indices[active["id"]] = event["output_index"]
        if event["type"] == "response.reasoning_summary_text.delta":
            assert active["type"] == "reasoning"
            assert event["item_id"] == active["id"] == "rs"
        if event["type"] == "response.output_text.delta":
            assert active["type"] == "message"
            assert event["item_id"] == active["id"] == "msg"
        if event["type"] == "response.output_item.done":
            done.append(event["item"]["id"])
        if "item_id" in event:
            assert event["output_index"] == indices[event["item_id"]]
    assert indices == {"rs": 0, "msg": 1}
    assert done == ["rs", "msg"]
    assert result[-1]["response"]["output"] == [
        item("reasoning", "rs", "Think carefully"),
        item("message", "msg", "ANSWER"),
    ]


async def test_valid_stream_preserves_explicit_parts_and_items():
    events = [
        added({"id": "rs", "type": "reasoning", "summary": []}),
        {
            "type": "response.reasoning_summary_part.added",
            "item_id": "rs",
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        },
        {**delta("reasoning_summary_text", "rs", "reason"), "summary_index": 0},
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs",
            "output_index": 0,
            "summary_index": 0,
            "text": "reason",
        },
        {
            "type": "response.reasoning_summary_part.done",
            "item_id": "rs",
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": "reason"},
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item("reasoning", "rs", "reason"),
        },
        {"type": "response.completed", "response": {"output": [item("reasoning", "rs", "reason")]}},
    ]
    events = [{**event, "sequence_number": index} for index, event in enumerate(events)]
    assert await normalize(events) == events


async def test_message_parts_do_not_duplicate_and_complete_snapshot_is_preserved():
    events = [
        added({"type": "message", "id": "msg", "role": "assistant", "content": []}),
        {
            "type": "response.content_part.added",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {**delta("output_text", "msg", "answer"), "content_index": 0},
        {
            "type": "response.completed",
            "response": {
                "output": [item("message", "msg", "answer")],
                "usage": {"total_tokens": 3},
            },
        },
    ]
    result = await normalize(events)
    assert sum(event["type"] == "response.content_part.added" for event in result) == 1
    assert sum(event["type"] == "response.output_item.done" for event in result) == 1
    assert result[-1]["response"] == events[-1]["response"]


@pytest.mark.parametrize("role_first", [False, True])
async def test_installed_litellm_chunks_open_correct_native_active_item(role_first):
    converter = LiteLLMCompletionStreamingIterator(
        model="synthetic-model",
        litellm_custom_stream_wrapper=SimpleNamespace(logging_obj=None),
        request_input="synthetic input",
        responses_api_request={},
    )
    chunks = ([{"role": "assistant", "content": ""}] if role_first else []) + [
        {"reasoning_content": "Think "},
        {"reasoning_content": "carefully"},
        {"content": "ANSWER"},
    ]
    events = []
    for value in chunks:
        chunk = ModelResponseStream(choices=[{"index": 0, "delta": value}])
        converter._ensure_output_item_for_chunk(chunk)
        events.extend(converter._pending_response_events)
        converter._pending_response_events.clear()
        transformed = converter._transform_chat_completion_chunk_to_response_api_chunk(chunk)
        if transformed is not None:
            events.append(transformed)
    result = await normalize(events)
    active = None
    reasoning = ""
    answer = ""
    for event in result:
        if event["type"] == "response.output_item.added":
            active = event["item"]
        if event["type"] == "response.reasoning_summary_text.delta":
            assert active["type"] == "reasoning"
            assert active["id"] == event["item_id"] == converter._cached_reasoning_item_id
            reasoning += event["delta"]
        if event["type"] == "response.output_text.delta":
            assert active["type"] == "message"
            assert active["id"] == event["item_id"] == converter._cached_item_id
            answer += event["delta"]
    assert reasoning == "Think carefully"
    assert answer == "ANSWER"
    for event in result:
        assert sse_frame(event).startswith(b"event: response.")


async def test_valid_message_and_function_stream_is_unchanged():
    message = item("message", "msg", "answer")
    call = {
        "type": "function_call",
        "id": "fc",
        "call_id": "call",
        "name": "tool",
        "arguments": "{}",
    }
    events = [
        added({"type": "message", "id": "msg", "role": "assistant", "content": []}),
        {
            "type": "response.content_part.added",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {**delta("output_text", "msg", "answer"), "content_index": 0},
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "answer",
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": message["content"][0],
        },
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        added(call, 1),
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc",
            "output_index": 1,
            "delta": "{}",
        },
        {"type": "response.output_item.done", "output_index": 1, "item": call},
        {"type": "response.completed", "response": {"output": [message, call]}},
    ]
    events = [{**event, "sequence_number": index + 12} for index, event in enumerate(events)]
    assert await normalize(events) == events


async def test_inserted_events_have_monotone_required_sequence_numbers():
    events = [
        {"type": "response.created", "sequence_number": 8, "response": {"output": []}},
        {**delta("reasoning_summary_text", "rs", "thought"), "sequence_number": 9},
        {**delta("output_text", "msg", "answer"), "sequence_number": 10},
        {
            "type": "response.completed",
            "sequence_number": 11,
            "response": {
                "output": [item("reasoning", "rs", "thought"), item("message", "msg", "answer")]
            },
        },
    ]
    result = await normalize(events)
    assert [event["sequence_number"] for event in result] == list(range(8, 8 + len(result)))


async def test_valid_refusal_stream_keeps_item_before_refusal_delta():
    part = {"type": "refusal", "refusal": "Declined"}
    message = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [part],
    }
    events = [
        added({"id": "msg", "type": "message", "role": "assistant", "content": []}),
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "item_id": "msg",
            "content_index": 0,
            "part": {"type": "refusal", "refusal": ""},
        },
        {
            "type": "response.refusal.delta",
            "output_index": 0,
            "item_id": "msg",
            "content_index": 0,
            "delta": "Declined",
        },
        {
            "type": "response.refusal.done",
            "output_index": 0,
            "item_id": "msg",
            "content_index": 0,
            "refusal": "Declined",
        },
        {
            "type": "response.content_part.done",
            "output_index": 0,
            "item_id": "msg",
            "content_index": 0,
            "part": part,
        },
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": {"output": [message]}},
    ]
    events = [{**event, "sequence_number": index} for index, event in enumerate(events)]
    assert await normalize(events) == events


@pytest.mark.parametrize("terminal", ["failed", "incomplete"])
async def test_unsuccessful_terminal_preserves_announcements_without_inventing_success(terminal):
    events = [
        added({"type": "message", "id": "msg", "role": "assistant", "content": []}),
        {"type": f"response.{terminal}", "response": {"status": terminal, "output": []}},
    ]
    events = [{**event, "sequence_number": index} for index, event in enumerate(events)]
    assert await normalize(events) == events


@pytest.mark.parametrize("terminal", ["failed", "incomplete"])
async def test_unsuccessful_snapshot_matches_remapped_indices_without_closing_message(terminal):
    message = item("message", "msg", "partial answer")
    message["status"] = "incomplete"
    reasoning = item("reasoning", "rs", "thought")
    response = {
        "status": terminal,
        "error": {"code": "server_error", "message": "Synthetic failure"},
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [message, reasoning],
    }
    events = [
        added({"type": "message", "id": "msg", "role": "assistant", "content": []}),
        delta("reasoning_summary_text", "rs", "thought"),
        delta("output_text", "msg", "partial answer"),
        {"type": f"response.{terminal}", "response": response},
    ]
    original = copy.deepcopy(events)
    result = await normalize(events)
    assert events == original
    announced = {
        event["item"]["id"]: event["output_index"]
        for event in result
        if event["type"] == "response.output_item.added"
    }
    assert announced == {"rs": 0, "msg": 1}
    assert result[-1]["response"] == {**response, "output": [reasoning, message]}
    assert result[-1]["type"] == f"response.{terminal}"
    assert not any(
        event["type"] == "response.output_item.done" and event["item"]["id"] == "msg"
        for event in result
    )
    assert not any(event["type"] == "response.completed" for event in result)


async def test_reasoning_in_message_part_is_corrected_without_losing_final_reasoning():
    reasoning = item("reasoning", "rs", "thought")
    message = item("message", "msg", "answer")
    events = [
        delta("reasoning_summary_text", "rs", "thought"),
        delta("output_text", "msg", "answer"),
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "answer",
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "reasoning_text", "reasoning": "thought"},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": message},
        {"type": "response.completed", "response": {"output": [reasoning, message]}},
    ]
    result = await normalize(events)
    parts = [event["part"] for event in result if event["type"] == "response.content_part.done"]
    assert parts == message["content"]
    assert result[-1]["response"]["output"] == [reasoning, message]


async def test_function_index_collision_remaps_every_event_and_preserves_call_identity():
    call = {
        "type": "function_call",
        "id": "fc",
        "call_id": "call",
        "name": "tool",
        "arguments": "{}",
    }
    events = [
        delta("reasoning_summary_text", "rs", "thought"),
        delta("output_text", "msg", "answer"),
        added(call, 1),
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc",
            "output_index": 1,
            "delta": "{}",
        },
        {"type": "response.output_item.done", "output_index": 1, "item": call},
    ]
    result = await normalize(events)
    function_events = [
        event
        for event in result
        if event.get("item_id") == "fc" or event.get("item", {}).get("id") == "fc"
    ]
    assert len(function_events) == 3
    assert all(event["output_index"] == 2 for event in function_events)
    assert function_events[-1]["item"] == call


@pytest.mark.parametrize(
    "status,error",
    [
        ("failed", {"code": "server_error", "message": "Service temporarily overloaded"}),
        ("completed", {"code": "server_error", "message": "Service temporarily overloaded"}),
        ("completed", None),
    ],
)
async def test_mislabeled_completed_event_never_hides_failure_or_empty_output(status, error):
    response = {
        "id": "resp_overloaded",
        "object": "response",
        "status": status,
        "error": error,
        "output": [],
    }
    events = await normalize(
        [
            {"type": "response.created", "response": {**response, "status": "in_progress"}},
            {"type": "response.completed", "response": response},
        ]
    )
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["status"] == "failed"
    assert events[-1]["response"]["error"]["message"] == (
        "Service temporarily overloaded" if error else "Provider completed without any output"
    )
    assert not any(event["type"] == "response.completed" for event in events)


@pytest.mark.parametrize(
    "output,expected",
    [
        ([item("message", "msg")], "response.failed"),
        ([item("reasoning", "rs")], "response.failed"),
        ([item("message", "msg", "Answer")], "response.completed"),
        ([item("reasoning", "rs", "Thinking")], "response.completed"),
        (
            [{"type": "function_call", "id": "call", "name": "test", "arguments": "{}"}],
            "response.completed",
        ),
        (
            [
                {
                    "type": "message",
                    "id": "msg",
                    "content": [{"type": "refusal", "refusal": "Denied"}],
                }
            ],
            "response.completed",
        ),
        ([{"type": "reasoning", "id": "rs", "encrypted_content": "opaque"}], "response.completed"),
    ],
)
async def test_empty_messages_fail_without_rejecting_other_output(output, expected):
    events = await normalize(
        [added(value, index) for index, value in enumerate(output)]
        + [{"type": "response.completed", "response": {"output": output}}]
    )
    assert events[-1]["type"] == expected
