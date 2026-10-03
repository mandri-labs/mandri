import json
from unittest.mock import AsyncMock

import httpx
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.core.types.execution import ProtectionError
from mandri.gateway import privacy_transport
from mandri.gateway.chatgpt_adapter import ChatGptAdapter, complete_response
from mandri.gateway.chatgpt_continuation import commentary_response
from mandri.gateway.chatgpt_conversion import chat_request, chat_response
from mandri.gateway.errors.upstream import UpstreamError
from mandri.gateway.litellm_adapter import OpenAIHandler, ResponsesHandler
from mandri.gateway.usage import UsageCollector, usage_scope

from mandri_gateway.tests.test_chatgpt_inference import completed, route


@pytest.fixture(autouse=True)
async def logging_worker():
    GLOBAL_LOGGING_WORKER.start()
    yield
    await GLOBAL_LOGGING_WORKER.flush()


def message_response(text, phase, identity):
    response = completed(text)
    response["id"] = "resp_" + identity
    response["output"][0].update(id="msg_" + identity, phase=phase)
    return response


@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("ending", ["final_answer", "tool"])
async def test_commentary_continues_with_original_phase_and_complete_usage(
    monkeypatch, protected, protocol, ending
):
    resolved, guard = route(protected)
    initial = message_response("Checking files", "commentary", "first")
    terminal = message_response("Checks complete", "final_answer", "last")
    terminal["usage"].update(input_tokens=3, output_tokens=2, total_tokens=5)
    if ending == "tool":
        terminal["output"] = [
            {
                "id": "tool_last",
                "type": "function_call",
                "call_id": "call_last",
                "name": "check",
                "arguments": "{}",
                "status": "completed",
            }
        ]
    sent = []
    records = []

    async def send(client, request):
        sent.append(json.loads(request.content))
        response = initial if len(sent) == 1 else terminal
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content="data: "
            + json.dumps({"type": "response.completed", "response": response})
            + "\n\n",
        )

    async def save(record):
        records.append(record)

    privacy_transport.install_transport_observers()
    monkeypatch.setattr(privacy_transport, "_ASYNC_SEND", send)
    collector = UsageCollector(resolved, protocol, save)
    kwargs = {"guard": guard} if protected else {}
    with usage_scope(collector):
        if protocol == "chat":
            result = await OpenAIHandler().chat_completions(
                resolved, {"messages": [{"role": "user", "content": "Check"}]}, **kwargs
            )
        else:
            result = await ResponsesHandler().responses(resolved, {"input": "Check"}, **kwargs)
    assert len(sent) == 2
    assert sent[1]["input"][-1] == initial["output"][0]
    assert sent[1]["tools"] == sent[0]["tools"] if "tools" in sent[0] else "tools" not in sent[1]
    value = (
        result.model_dump(mode="json", exclude_none=True)
        if hasattr(result, "model_dump")
        else result
    )
    if protocol == "chat":
        assert len(value["choices"]) == 1
        assert "Checking files" in value["choices"][0]["message"]["content"]
        assert value["choices"][0]["finish_reason"] == (
            "tool_calls" if ending == "tool" else "stop"
        )
        if ending == "tool":
            assert value["choices"][0]["message"]["tool_calls"][0]["id"] == "call_last"
    else:
        assert value["output"] == initial["output"] + terminal["output"]
    assert collector.record.input_tokens == 13
    assert collector.record.output_tokens == 6
    assert collector.record.total_tokens == 19
    assert not collector.observation_incomplete
    if protected:
        assert guard.sends == 2


async def test_repeated_commentary_fails_instead_of_reporting_completion(monkeypatch):
    resolved, _ = route(False)
    sent = []

    async def send(client, request):
        sent.append(request)
        response = message_response("Checking", "commentary", str(len(sent)))
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content="data: "
            + json.dumps({"type": "response.completed", "response": response})
            + "\n\n",
        )

    privacy_transport.install_transport_observers()
    monkeypatch.setattr(privacy_transport, "_ASYNC_SEND", send)
    with pytest.raises(UpstreamError, match="without completing"):
        await ResponsesHandler().responses(resolved, {"input": "Check"})
    assert len(sent) == 9


def test_chat_conversion_round_trips_explicit_assistant_phases():
    response = message_response("Checking", "commentary", "first")
    chat = chat_response(response)
    request = chat_request(
        {
            "model": response["model"],
            "messages": [
                {"role": "user", "content": "Check"},
                chat["choices"][0]["message"],
            ],
        }
    )
    assert request["input"][-1]["phase"] == "commentary"
    assert "phase" not in request["input"][0]


async def test_sparse_terminal_recovers_tool_alongside_terminal_message():
    response = message_response("Checking", "commentary", "first")
    tool = {
        "id": "tool",
        "type": "function_call",
        "call_id": "call",
        "name": "check",
        "arguments": "{}",
    }
    events = [
        {"type": "response.output_item.done", "output_index": 0, "item": response["output"][0]},
        {"type": "response.output_item.done", "output_index": 1, "item": tool},
        {"type": "response.completed", "response": response},
    ]
    result = await complete_response(
        httpx.Response(
            200,
            content="".join("data: " + json.dumps(event) + "\n\n" for event in events),
        )
    )
    assert result["output"] == [response["output"][0], tool]


@pytest.mark.parametrize("ending", ["final_answer", "unmarked", "incomplete", "tool"])
def test_only_commentary_without_tools_requests_continuation(ending):
    response = message_response("Checking", "commentary", "first")
    if ending == "final_answer":
        response["output"][0]["phase"] = "final_answer"
    elif ending == "unmarked":
        response["output"][0].pop("phase")
    elif ending == "incomplete":
        response["status"] = "incomplete"
    else:
        response["output"].append({"type": "function_call"})
    assert not commentary_response(response)


async def test_commentary_replay_cannot_bypass_protected_egress(monkeypatch):
    resolved, guard = route(True)
    private = "replay-canary@example.invalid"
    alias = guard.engine.protect_text(private)
    sent = []

    async def send(client, request):
        sent.append(request)
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content="data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": message_response(private, "commentary", "first"),
                }
            )
            + "\n\n",
        )

    privacy_transport.install_transport_observers()
    monkeypatch.setattr(privacy_transport, "_ASYNC_SEND", send)
    with pytest.raises(ProtectionError, match="Unmasked provider content"):
        await ResponsesHandler().responses(resolved, {"input": alias}, guard=guard)
    assert len(sent) == 1
    assert guard.sends == 1


async def test_continuation_upstream_error_is_returned_with_its_body():
    adapter = ChatGptAdapter("https://provider.invalid/v1")
    original = httpx.Request(
        "POST", "https://provider.invalid/v1/responses", json={"input": "Check"}
    )
    upstream = adapter.request(original)
    initial = httpx.Response(
        200,
        request=upstream,
        content="data: "
        + json.dumps(
            {
                "type": "response.completed",
                "response": message_response("Checking", "commentary", "first"),
            }
        )
        + "\n\n",
    )
    failure = httpx.Response(503, request=upstream, json={"error": "synthetic failure"})
    send = AsyncMock(return_value=failure)
    response = await adapter.response(initial, original, send)
    assert response.status_code == 503
    assert response.json() == {"error": "synthetic failure"}
    assert initial.is_closed
    send.assert_awaited_once()
