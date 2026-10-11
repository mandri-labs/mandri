import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mandri.core.ids import ApprovalKind, HarnessSessionId
from mandri.runtime.control.errors import ControlError, ControlTransportError
from mandri.runtime.control.opencode import OpencodeControlAdapter, _normalize_permission_event
from mandri.runtime.control.opencode_questions import question_answers, recovered_inputs


@pytest.fixture
async def control():
    adapter = OpencodeControlAdapter("http://opencode.invalid", "native-session")
    yield adapter
    await adapter.aclose()


async def test_catalog_is_dynamic_and_does_not_expose_templates(control):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(200, json={"data": []}),
            httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "name": "custom",
                            "description": "Custom command",
                            "template": "private prompt",
                            "hints": ["$1", "$ARGUMENTS"],
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "build", "name": "Build", "mode": "primary"},
                        {"id": "plan", "name": "Plan", "mode": "primary"},
                        {"id": "custom-agent", "name": "Custom agent", "mode": "all"},
                        {"id": "explore", "name": "Explore", "mode": "subagent"},
                        {"id": "hidden", "name": "Hidden", "mode": "primary", "hidden": True},
                    ]
                },
            ),
        ]
    )
    catalog = await control.list_commands()
    assert [item["id"] for item in catalog] == [
        "compact",
        "agent:build",
        "agent:plan",
        "agent:custom-agent",
        "custom",
    ]
    assert catalog[-1]["argument_hint"] == "$1 $ARGUMENTS"
    assert all("template" not in item for item in catalog)
    control._request.assert_awaited_with("GET", "/api/agent", None)


async def test_execution_revalidates_catalog_and_preserves_arguments(control):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(200, json={"data": []}),
            httpx.Response(200, json={"data": [{"name": "custom"}]}),
            httpx.Response(204),
            httpx.Response(204),
        ]
    )
    assert (await control.execute_command("custom", '  "two words"\nthree'))["kind"] == "transcript"
    control._request.assert_awaited_with(
        "POST",
        "/api/session/native-session/command",
        {
            "name": "custom",
            "text": '  "two words"\nthree',
        },
    )


async def test_missing_command_never_falls_through_to_prompt(control):
    control._request = AsyncMock(return_value=httpx.Response(200, json={"data": []}))
    with pytest.raises(ControlError, match="no longer available"):
        await control.execute_command("removed", "")
    assert control._request.await_count == 2


async def test_native_agent_selection_uses_native_id_without_permission_changes(control):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(200, json={"data": []}),
            httpx.Response(
                200, json={"data": [{"id": "custom", "name": "Custom", "mode": "primary"}]}
            ),
            httpx.Response(204),
        ]
    )
    await control.execute_command("agent:custom", "")
    control._request.assert_awaited_with(
        "POST", "/api/session/native-session/agent", {"agent": "custom"}
    )
    assert control._request.await_count == 3


@pytest.mark.parametrize("agent", ["build", "plan"])
@pytest.mark.parametrize("via_command", [False, True])
async def test_cold_session_waits_for_plugin_agents_before_selection(control, agent, via_command):
    activated = False
    calls = []

    async def request(method, path, body=None):
        nonlocal activated
        calls.append((method, path, body))
        if path == "/api/integration":
            activated = True
            return httpx.Response(200, json={"data": []})
        if path == "/api/agent":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": name, "name": name.title(), "mode": "primary"}
                        for name in ("build", "plan")
                    ]
                    if activated
                    else []
                },
            )
        assert method == "POST" and path == "/api/session/native-session/agent"
        return httpx.Response(204)

    control._request = AsyncMock(side_effect=request)
    if via_command:
        await control.execute_command(f"agent:{agent}", "")
    else:
        await control.select_agent(agent)
    assert calls == [
        ("GET", "/api/integration", None),
        ("GET", "/api/agent", None),
        ("POST", "/api/session/native-session/agent", {"agent": agent}),
    ]


async def test_agent_selection_does_not_continue_after_activation_failure(control):
    control._request = AsyncMock(return_value=httpx.Response(503))
    with pytest.raises(ControlTransportError, match="plugin activation"):
        await control.select_agent("plan")
    control._request.assert_awaited_once_with("GET", "/api/integration", None)


@pytest.mark.parametrize("agent", ["missing", "explore", "hidden"])
async def test_activated_catalog_rejects_unavailable_primary_agents(control, agent):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(200, json={"data": []}),
            httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "build", "name": "Build", "mode": "primary"},
                        {"id": "explore", "name": "Explore", "mode": "subagent"},
                        {"id": "hidden", "name": "Hidden", "mode": "primary", "hidden": True},
                    ]
                },
            ),
        ]
    )
    with pytest.raises(ControlError, match="no longer available"):
        await control.select_agent(agent)
    assert control._request.await_count == 2


async def test_native_compact_is_available_without_a_prompt_template(control):
    control._request = AsyncMock(
        side_effect=[httpx.Response(204), httpx.Response(200, json={"data": {"id": "compact"}})]
    )
    result = await control.execute_command("compact", "")
    assert result["kind"] == "transcript"
    control._request.assert_awaited_with("POST", "/api/session/native-session/compact", {})


@pytest.mark.parametrize("command", ["compact", "agent:plan"])
async def test_native_actions_reject_arguments_before_side_effects(control, command):
    control._request = AsyncMock()
    with pytest.raises(ControlError, match="does not accept arguments"):
        await control.execute_command(command, "unexpected")
    control._request.assert_not_awaited()


async def test_rejected_gateway_selection_never_executes_command(control):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(200, json={"data": []}),
            httpx.Response(200, json={"data": [{"name": "custom"}]}),
            httpx.Response(400),
        ]
    )
    with pytest.raises(ControlTransportError, match="gateway model"):
        await control.execute_command("custom", "")
    assert control._request.await_count == 3


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"error": "private diagnostics"}),
        httpx.Response(200, json=True),
        httpx.Response(200, json={"info": {"error": {"data": {"message": "private"}}}}),
    ],
)
async def test_failed_or_unconfirmed_execution_is_not_success(control, response):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(200, json={"data": []}),
            httpx.Response(200, json={"data": [{"name": "custom"}]}),
            httpx.Response(204),
            response,
        ]
    )
    with pytest.raises(ControlError):
        await control.execute_command("custom", "")
    assert control._request.await_count == 4


async def test_native_question_reply_and_reject_use_native_request_id(control):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "data": {
                        "fields": [
                            {"key": "first", "type": "string"},
                            {"key": "second", "type": "multiselect"},
                        ]
                    }
                },
            ),
            httpx.Response(204),
        ]
    )
    assert await control.answer_question("q/1", [["first"], ["second", "third"]])
    control._request.assert_awaited_with(
        "POST",
        "/api/session/native-session/form/q%2F1/reply",
        {"answer": {"first": "first", "second": ["second", "third"]}},
    )
    control._request = AsyncMock(return_value=httpx.Response(204))
    assert await control.answer_question("q/1", None)
    control._request.assert_awaited_with("DELETE", "/api/session/native-session/form/q%2F1", None)
    control._request.return_value = httpx.Response(404)
    assert not await control.answer_question("old", [["first"]])


def test_questions_and_recovery_are_session_scoped():
    event = {
        "type": "question.asked",
        "properties": {"id": "question", "sessionID": "session", "questions": []},
    }
    request = _normalize_permission_event(event, HarnessSessionId("session"))
    assert request is not None and request.kind is ApprovalKind.USER_INPUT
    assert _normalize_permission_event(event, HarnessSessionId("other")) is None
    assert recovered_inputs(
        [event["properties"], {"sessionID": "other"}], "question.asked", "session"
    ) == [event]


async def test_pending_inputs_recovered_from_both_native_endpoints(control):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(200, json={"data": []}),
            httpx.Response(
                200, json={"data": [{"id": "permission", "sessionID": "native-session"}]}
            ),
            httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "question",
                            "sessionID": "native-session",
                            "title": "More",
                            "fields": [{"key": "first", "type": "string"}],
                        }
                    ]
                },
            ),
        ]
    )
    await control._recover_pending_permissions()
    assert (await control._queue.get())["type"] == "permission.asked"
    assert (await control._queue.get())["type"] == "question.asked"


def test_question_answers_preserve_native_order_and_validate_multiplicity():
    request = SimpleNamespace(
        native_request=json.dumps(
            {
                "properties": {
                    "questions": [
                        {"question": "Choose", "options": [{"label": "A"}], "custom": False},
                        {"question": "More", "multiple": True},
                    ]
                }
            }
        ),
        answers=[{"question": "1", "answers": ["B", "C"]}, {"question": "0", "answers": ["A"]}],
    )
    assert question_answers(request) == [["A"], ["B", "C"]]
    request.answers[1]["answers"] = ["A", "B"]
    with pytest.raises(ControlTransportError, match="one answer"):
        question_answers(request)
    request.answers[1]["answers"] = ["D"]
    with pytest.raises(ControlTransportError, match="native question options"):
        question_answers(request)
