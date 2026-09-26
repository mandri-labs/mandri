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
        return_value=httpx.Response(
            200,
            json=[
                {
                    "name": "custom",
                    "description": "Custom command",
                    "template": "private prompt",
                    "hints": ["$1", "$ARGUMENTS"],
                }
            ],
        )
    )
    assert await control.list_commands() == [
        {
            "id": "custom",
            "name": "custom",
            "description": "Custom command",
            "aliases": [],
            "argument_hint": "$1 $ARGUMENTS",
            "kind": "prompt",
        }
    ]
    control._request.assert_awaited_once_with("GET", "/command", None)


async def test_execution_revalidates_catalog_and_preserves_arguments(control):
    control._request = AsyncMock(
        side_effect=[
            httpx.Response(200, json=[{"name": "custom"}]),
            httpx.Response(200, json={"info": {"id": "message"}, "parts": []}),
        ]
    )
    assert (await control.execute_command("custom", '  "two words"\nthree'))["kind"] == "transcript"
    control._request.assert_awaited_with(
        "POST",
        "/session/native-session/command",
        {
            "command": "custom",
            "arguments": '  "two words"\nthree',
            "model": "mandri/mandri_gateway",
        },
    )


async def test_missing_command_never_falls_through_to_prompt(control):
    control._request = AsyncMock(return_value=httpx.Response(200, json=[]))
    with pytest.raises(ControlError, match="no longer available"):
        await control.execute_command("removed", "")
    assert control._request.await_count == 1


@pytest.mark.parametrize(
    "command,extra",
    [
        ({"name": "custom", "model": "other/model"}, []),
        (
            {"name": "custom", "agent": "researcher"},
            [
                httpx.Response(
                    200,
                    json=[
                        {"name": "researcher", "model": {"providerID": "other", "modelID": "model"}}
                    ],
                )
            ],
        ),
    ],
)
async def test_model_override_cannot_bypass_gateway(control, command, extra):
    control._request = AsyncMock(side_effect=[httpx.Response(200, json=[command]), *extra])
    with pytest.raises(ControlError, match="outside the Mandri gateway"):
        await control.execute_command("custom", "")
    assert all(call.args[0] == "GET" for call in control._request.call_args_list)


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
        side_effect=[httpx.Response(200, json=[{"name": "custom"}]), response]
    )
    with pytest.raises(ControlError):
        await control.execute_command("custom", "")
    assert control._request.await_count == 2


async def test_native_question_reply_and_reject_use_native_request_id(control):
    control._request = AsyncMock(return_value=httpx.Response(200, json=True))
    assert await control.answer_question("q/1", [["first"], ["second", "third"]])
    control._request.assert_awaited_with(
        "POST", "/question/q%2F1/reply", {"answers": [["first"], ["second", "third"]]}
    )
    assert await control.answer_question("q/1", None)
    control._request.assert_awaited_with("POST", "/question/q%2F1/reject", None)
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
            httpx.Response(200, json=[{"id": "permission", "sessionID": "native-session"}]),
            httpx.Response(200, json=[{"id": "question", "sessionID": "native-session"}]),
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
