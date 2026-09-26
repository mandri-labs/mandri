from copy import deepcopy
from dataclasses import replace

import litellm
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.core.ids import ModelRef, ProviderKind
from mandri.gateway.litellm_adapter import ResponsesHandler
from mandri.gateway.responses_input import normalize_tool_results

from mandri_gateway.tests.test_web_search_options import route, upstream

__all__ = ["upstream"]


def message(text):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def call(call_id, *, custom=False):
    return {
        "type": "custom_tool_call" if custom else "function_call",
        "call_id": call_id,
        "name": "read_file",
        **({"input": "sample.txt"} if custom else {"arguments": '{"path":"sample.txt"}'}),
    }


def output(call_id, *, custom=False):
    return {
        "type": "custom_tool_call_output" if custom else "function_call_output",
        "call_id": call_id,
        "output": "sample",
    }


def reasoning(text):
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("commentary_first", [False, True])
async def test_reasoning_and_commentary_stay_with_their_tool_calls(
    upstream, stream, commentary_first
):
    items = [{"role": "user", "content": "test"}]
    for index in range(2):
        calls = [call(f"{index}-a"), call(f"{index}-b")]
        commentary = message(f"Checking details {index}")
        items.extend(
            [
                reasoning(f"Synthetic reasoning {index}"),
                *([commentary, *calls] if commentary_first else [*calls, commentary]),
                output(f"{index}-b"),
                output(f"{index}-a"),
            ]
        )
    original = deepcopy(items)
    response = await ResponsesHandler().responses(
        route(ProviderKind.OPENCODE_GO),
        {"input": items, "stream": stream, "reasoning": {"effort": "high"}},
    )
    if stream:
        events = [event async for event in response]
        assert any(event.type == "response.completed" for event in events)
    messages = upstream[0]["messages"]
    assert_tool_sequence(messages)
    assistants = [item for item in messages if item["role"] == "assistant"]
    assert len(assistants) == 2
    for index, item in enumerate(assistants):
        assert item["reasoning_content"] == f"Synthetic reasoning {index}"
        assert item["content"] == [{"type": "text", "text": f"Checking details {index}"}]
        assert [tool["id"] for tool in item["tool_calls"]] == [f"{index}-a", f"{index}-b"]
    assert items == original


CASES = [
    [call("a"), call("b"), output("b"), output("a")],
    [message("Before"), call("a"), message("After"), output("a")],
    [call("a"), call("b"), output("a"), message("Working"), output("b")],
    [call("a"), message("Between"), call("b"), output("b"), output("a")],
    [call("a"), message("First"), message("Second"), output("a")],
    [call("a", custom=True), message("After"), output("a", custom=True)],
    [
        call("a"),
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking"}]},
        message("After"),
        output("a"),
    ],
]


def assert_tool_sequence(messages):
    pending = set()
    seen_calls = []
    seen_results = []
    for item in messages:
        if item["role"] == "tool":
            call_id = item["tool_call_id"]
            assert call_id in pending
            pending.remove(call_id)
            seen_results.append(call_id)
        else:
            assert not pending, f"Unanswered calls before {item}: {pending}"
            for tool in item.get("tool_calls", []):
                assert tool["id"] not in pending
                pending.add(tool["id"])
                seen_calls.append(tool["id"])
    assert not pending
    assert sorted(seen_calls) == sorted(seen_results)
    assert len(seen_calls) == len(set(seen_calls))


@pytest.mark.parametrize("items", CASES)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "kind",
    [ProviderKind.OPENCODE_GO, ProviderKind.OPENCODE, ProviderKind.CUSTOM, ProviderKind.LM_STUDIO],
)
async def test_responses_tool_history_on_chat_wire(upstream, items, stream, kind):
    GLOBAL_LOGGING_WORKER.start()
    original = deepcopy(items)
    response = await ResponsesHandler().responses(
        route(kind),
        {
            "input": [{"role": "user", "content": "test"}, *items],
            "stream": stream,
            "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
        },
    )
    if stream:
        events = [event async for event in response]
        assert any(event.type == "response.completed" for event in events)
    else:
        assert response.output[0].content[0].text == "OK"
    assert len(upstream) == 1
    messages = upstream[0]["messages"]
    assert_tool_sequence(messages)
    assert sum(len(item.get("tool_calls", [])) for item in messages) == sum(
        item.get("type") in {"function_call", "custom_tool_call"} for item in items
    )
    for item in items:
        if item.get("role") == "assistant":
            text = item["content"][0]["text"]
            assert any(text in str(wire.get("content")) for wire in messages)
    assert items == original


@pytest.mark.parametrize("kind", list(ProviderKind))
@pytest.mark.parametrize("model", ["first-model", "second-model"])
async def test_normalization_depends_on_protocol_not_model(monkeypatch, kind, model):
    captured = {}

    async def responses(**kwargs):
        captured.update(kwargs)
        return "OK"

    monkeypatch.setattr(litellm, "aresponses", responses)
    selected = route(kind)
    selected = replace(selected, model=replace(selected.model, model_ref=ModelRef(model)))
    items = [call("a"), message("After"), output("a")]
    assert await ResponsesHandler().responses(selected, {"input": items}) == "OK"
    if kind in {ProviderKind.OPENAI, ProviderKind.OPENROUTER}:
        assert captured["input"] is items
        assert not captured["use_chat_completions_api"]
    else:
        assert captured["input"] == [items[1], items[0], items[2]]
        assert captured["use_chat_completions_api"]


@pytest.mark.parametrize("items", CASES)
def test_normalization_preserves_content_and_is_idempotent(items):
    original = deepcopy(items)
    result = normalize_tool_results(items)
    assert [
        part for item in result if item.get("role") == "assistant" for part in item["content"]
    ] == [part for item in original if item.get("role") == "assistant" for part in item["content"]]
    assert sorted(id(item) for item in result if item.get("role") != "assistant") == sorted(
        id(item) for item in items if item.get("role") != "assistant"
    )
    assert normalize_tool_results(result) == result
    assert items == original


@pytest.mark.parametrize(
    "items",
    [
        "test",
        None,
        [],
        [call("a"), message("Missing result")],
        [call("a"), call("b"), message("Missing b"), output("a")],
        [call("a"), message("Duplicate"), output("a"), output("a")],
        [call("a"), call("a"), message("Duplicate"), output("a")],
        [call("a"), message("Wrong type"), output("a", custom=True)],
        [output("a"), call("a")],
        [output("a")],
    ],
)
def test_incomplete_or_ambiguous_history_is_not_repaired(items):
    assert normalize_tool_results(items) == items


@pytest.mark.parametrize(
    "boundary",
    [
        {"role": "user", "content": "test"},
        {"role": "developer", "content": "test"},
        {"role": "system", "content": "test"},
        {"type": "item_reference", "id": "opaque"},
        {"type": "unknown_item", "role": "assistant", "content": "opaque"},
    ],
)
def test_results_do_not_cross_conversation_boundaries(boundary):
    items = [call("a"), boundary, output("a")]
    assert normalize_tool_results(items) == items


def test_reused_ids_in_separate_turns_and_structured_failure_outputs():
    failed = {**output("a"), "output": [{"type": "input_text", "text": "Exit code: 2"}]}
    first = [call("a"), message("After"), failed]
    boundary = {"role": "user", "content": "test"}
    second = [call("a"), message("Next"), output("a")]
    assert normalize_tool_results([*first, boundary, *second]) == [
        first[1],
        first[0],
        failed,
        boundary,
        second[1],
        second[0],
        second[2],
    ]


@pytest.mark.parametrize("items", CASES)
async def test_reasoning_covers_all_fragments_of_a_tool_batch(upstream, items):
    response = await ResponsesHandler().responses(
        route(ProviderKind.OPENCODE_GO),
        {"input": [{"role": "user", "content": "test"}, reasoning("Batch reasoning"), *items]},
    )
    assert response.output[0].content[0].text == "OK"
    messages = upstream[0]["messages"]
    assert_tool_sequence(messages)
    assistants = [item for item in messages if item["role"] == "assistant"]
    assert len(assistants) == 1
    expected = "\n".join(
        ["Batch reasoning"]
        + [
            part["text"]
            for item in items
            if item.get("type") == "reasoning"
            for part in item["summary"]
        ]
    )
    assert assistants[0]["reasoning_content"] == expected


def test_tool_batch_keeps_mixed_content_blocks_and_string_messages():
    refusal = {"type": "refusal", "refusal": "Unavailable"}
    items = [
        reasoning("Batch reasoning"),
        {"role": "assistant", "content": "Before"},
        call("a"),
        {"role": "assistant", "content": [refusal]},
        message("After"),
        output("a"),
    ]
    original = deepcopy(items)
    result = normalize_tool_results(items)
    assert result == [
        items[0],
        {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Before"},
                refusal,
                {"type": "output_text", "text": "After"},
            ],
        },
        items[2],
        items[-1],
    ]
    assert items == original
    assert normalize_tool_results(result) == result
