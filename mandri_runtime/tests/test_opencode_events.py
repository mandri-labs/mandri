import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from mandri.core.ids import ApprovalDecision, HarnessKind
from mandri.core.opencode_messages import message_record
from mandri.core.types.agents import Agent, AgentState
from mandri.runtime.agent_events import _updated
from mandri.runtime.control.opencode import OpencodeControlAdapter
from mandri.runtime.control.opencode_diffs import OpencodeDiffRecovery
from mandri.runtime.control.opencode_events import OpencodeEvents, form_event
from mandri.runtime.control.opencode_questions import form_answers


def event(kind, **data):
    return {"type": kind, "created": 10, "data": {"sessionID": "child", **data}}


async def test_streamed_parts_replace_by_the_same_identity_as_history():
    converter = OpencodeEvents(AsyncMock())
    parts = []
    for kind in ("reasoning", "text"):
        first = await converter.convert(
            event(f"session.{kind}.started", assistantMessageID="msg", ordinal=0)
        )
        delta = await converter.convert(
            event(f"session.{kind}.delta", assistantMessageID="msg", ordinal=0, delta="hello")
        )
        ended = await converter.convert(
            event(f"session.{kind}.ended", assistantMessageID="msg", ordinal=0, text="hello world")
        )
        assert first[0]["properties"]["part"]["text"] == ""
        assert delta[0]["properties"]["part"]["text"] == "hello"
        parts.append(ended[0]["properties"]["part"])
    stored = message_record(
        {
            "id": "msg",
            "type": "assistant",
            "content": [
                {"type": "reasoning", "text": "hello world"},
                {"type": "text", "text": "hello world"},
            ],
        },
        "child",
    )
    assert [p["id"] for p in parts] == [p["id"] for p in stored["parts"]]
    assert len({p["id"] for p in parts}) == 2


@pytest.mark.parametrize("input_id", [None, "manual"])
async def test_compaction_stream_and_history_share_identity_and_summary(input_id):
    converter = OpencodeEvents(AsyncMock())
    started = await converter.convert(
        {
            **event(
                "session.compaction.started",
                reason="manual" if input_id else "auto",
                inputID=input_id,
            ),
            "id": "evt_compact",
        }
    )
    delta = await converter.convert(event("session.compaction.delta", text="summary"))
    ended = await converter.convert(event("session.compaction.ended", text="full summary"))
    first = started[-1]["properties"]["part"]
    assert first["status"] == "running"
    assert delta[-1]["properties"]["part"]["summary"] == "summary"
    terminal = ended[-1]["properties"]["part"]
    assert terminal["status"] == "completed"
    assert terminal["summary"] == "full summary"
    stored = message_record(
        {
            "id": input_id or "msg_compact",
            "type": "compaction",
            "status": "completed",
            "summary": "full summary",
        },
        "child",
    )
    assert terminal["id"] == first["id"] == stored["parts"][0]["id"]


async def test_completed_turn_recovers_native_snapshot_diff_for_write_tools():
    files = [
        {"file": "hello.py", "patch": "+hello", "status": "added", "additions": 1, "deletions": 0}
    ]
    call = AsyncMock(
        side_effect=[
            httpx.Response(
                200, json={"data": {"id": "user", "type": "user", "text": "Create file"}}
            ),
            httpx.Response(200, json={"data": files}),
        ]
    )
    queue = asyncio.Queue()
    recovery = OpencodeDiffRecovery(call, queue.put_nowait)
    converter = OpencodeEvents(call, recovery.submit)
    try:
        await converter.convert(event("session.inbox.delivered", inboxID="user"))
        converted = await converter.convert(event("session.execution.succeeded"))
        assert converted[0]["properties"]["status"] == {"type": "idle"}
        async with asyncio.timeout(1):
            recovered = await queue.get()
        assert recovered == {
            "type": "session.diff",
            "properties": {"sessionID": "child", "diff": files, "scope": "turn", "turnID": "user"},
        }
        call.assert_awaited_with("GET", "/api/session/child/diff?from=user", None)
    finally:
        await recovery.close()


def test_internal_system_messages_do_not_render_as_assistant_text():
    record = message_record(
        {
            "id": "system",
            "type": "system",
            "content": [{"type": "text", "text": "internal declarations"}],
        },
        "root",
    )
    assert record["parts"] == []


@pytest.mark.parametrize("source", ["shell", "subagent", "custom-tool"])
async def test_synthetic_background_results_are_tools_with_persistent_identity(source):
    message = {
        "description": "Background work",
        "text": "result",
        "metadata": {"source": source, "state": "completed", "childID": "child"},
    }
    converted = await OpencodeEvents(AsyncMock()).convert(
        {**event("session.synthetic", **message), "id": "evt_notification"}
    )
    part = converted[-1]["properties"]["part"]
    assert part["type"] == "tool"
    assert part["tool"] == source
    assert part["state"]["status"] == "completed"
    assert part["state"]["output"] == "result"
    assert part["state"]["metadata"]["sessionID"] == "child"
    stored = message_record({**message, "id": "msg_notification", "type": "synthetic"}, "child")
    assert part["id"] == stored["parts"][0]["id"]


async def test_native_shell_lifecycle_preserves_command_and_output():
    converter = OpencodeEvents(AsyncMock())
    shell = {"id": "sh_job", "command": "echo hello", "status": "running"}
    started = await converter.convert(
        {**event("session.shell.started", shell=shell), "id": "evt_shell"}
    )
    ended = await converter.convert(
        {
            **event(
                "session.shell.ended",
                shell={**shell, "status": "exited", "exit": 0},
                output={"output": "hello"},
            ),
            "id": "evt_shell_end",
        }
    )
    assert started[-1]["properties"]["part"]["state"]["status"] == "running"
    result = ended[-1]["properties"]["part"]
    assert result["id"] == started[-1]["properties"]["part"]["id"]
    assert result["state"]["status"] == "completed"
    assert result["state"]["input"]["command"] == "echo hello"
    assert result["state"]["output"] == "hello"


@pytest.mark.parametrize(
    "status,exit_code,expected",
    [("exited", 1, "error"), ("timeout", None, "error"), ("killed", -9, "cancelled")],
)
def test_shell_failures_and_cancellations_remain_visible(status, exit_code, expected):
    record = message_record(
        {"id": "shell", "type": "shell", "command": "command", "status": status, "exit": exit_code},
        "root",
    )
    assert record["parts"][0]["state"]["status"] == expected


async def test_tools_preserve_input_output_timing_and_metadata():
    converter = OpencodeEvents(AsyncMock())
    await converter.convert(
        event("session.tool.input.started", assistantMessageID="msg", id="call", name="subagent")
    )
    running = await converter.convert(
        event(
            "session.tool.called",
            assistantMessageID="msg",
            id="call",
            input={"description": "hello"},
        )
    )
    terminal = await converter.convert(
        {
            **event(
                "session.tool.success",
                assistantMessageID="msg",
                id="call",
                content=[{"type": "text", "text": "done"}],
                metadata={"sessionID": "nested"},
            ),
            "created": 20,
        }
    )
    part = terminal[0]["properties"]["part"]
    assert running[0]["properties"]["part"]["state"]["status"] == "running"
    assert part["tool"] == "subagent"
    assert part["state"]["output"] == "done"
    assert part["state"]["input"] == {"description": "hello"}
    assert part["state"]["time"] == {"start": 10, "end": 20}
    assert part["state"]["metadata"]["sessionID"] == "nested"
    assert (
        part["id"]
        == message_record(
            {
                "id": "msg",
                "type": "assistant",
                "content": [
                    {
                        "id": "call",
                        "type": "tool",
                        "name": "subagent",
                        "state": {"status": "completed", "input": {}},
                    }
                ],
            },
            "child",
        )["parts"][0]["id"]
    )


@pytest.mark.parametrize(
    "outcome,state",
    [
        ("succeeded", AgentState.COMPLETED),
        ("failed", AgentState.FAILED),
        ("interrupted", AgentState.STOPPED),
    ],
)
async def test_execution_completion_updates_child_state(outcome, state):
    converter = OpencodeEvents(AsyncMock())
    agent = Agent("agent", "root", HarnessKind.OPENCODE, "child", "Child", AgentState.RUNNING, 1, 1)
    converted = await converter.convert(
        event(f"session.execution.{outcome}", error={"message": "failure"})
    )
    assert _updated(agent, converted[0]).state is state


async def test_inbox_projection_recovers_user_text_and_images():
    call = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "id": "user",
                    "type": "user",
                    "text": "hello",
                    "files": [{"mime": "image/png", "data": "aGVsbG8="}],
                }
            },
        )
    )
    converter = OpencodeEvents(call)
    converted = await converter.convert(event("session.inbox.delivered", inboxID="user"))
    call.assert_awaited_once_with("GET", "/api/session/child/message/user", None)
    assert converted[0]["properties"]["info"]["role"] == "user"
    assert converted[1]["properties"]["part"]["text"] == "hello"
    assert converted[2]["properties"]["part"]["url"] == "data:image/png;base64,aGVsbG8="
    assert await converter.convert(event("session.inbox.enqueued", inboxID="user")) == []


async def test_recovery_includes_children_and_routes_their_replies():
    control = OpencodeControlAdapter("http://unused.invalid", "root")
    form = {
        "id": "form",
        "sessionID": "child",
        "title": "Choose",
        "fields": [{"key": "choice", "type": "string", "options": [{"label": "A", "value": "a"}]}],
    }

    async def call(method, path, body=None):
        if "parentID=root" in path:
            return httpx.Response(
                200, json={"data": [{"id": "child", "parentID": "root"}], "cursor": {"next": None}}
            )
        if "parentID=child" in path:
            return httpx.Response(200, json={"data": []})
        if method == "GET" and path.endswith("/form/form"):
            return httpx.Response(200, json={"data": form})
        if method == "GET":
            return httpx.Response(
                200, json={"data": [form] if path == "/api/session/child/form" else []}
            )
        return httpx.Response(204)

    control._request = AsyncMock(side_effect=call)
    control._ensure_event_pump = lambda: None
    try:
        question = await control.next_event()
        assert question["type"] == "question.asked"
        assert await control.answer_question("form", [["A"]])
        control._request.assert_awaited_with(
            "POST", "/api/session/child/form/form/reply", {"answer": {"choice": "a"}}
        )
        control._approval_owners["permission"] = "child"
        assert await control.answer_approval("permission", ApprovalDecision.ALLOW)
        control._request.assert_awaited_with(
            "POST", "/api/session/child/permission/permission/reply", {"decision": "once"}
        )
    finally:
        await control.aclose()


def test_native_form_values_keep_the_frontend_question_contract():
    form = {
        "id": "f",
        "sessionID": "child",
        "title": "Options",
        "fields": [
            {"key": "enabled", "type": "boolean"},
            {"key": "count", "type": "integer"},
            {"key": "selected", "type": "multiselect", "options": [{"label": "A", "value": "a"}]},
        ],
    }
    assert form_event(form)["properties"]["questions"][2]["multiple"] is True
    assert form_answers(form, [["true"], ["2"], ["A"]]) == {
        "enabled": True,
        "count": 2,
        "selected": ["a"],
    }


async def test_catalog_notifications_do_not_become_unknown_transcript_cards():
    converter = OpencodeEvents(AsyncMock())
    for kind in ("agent.updated", "command.updated", "plugin.updated", "models-dev.refreshed"):
        assert (await converter.convert({"type": kind, "data": {}}))[0]["type"] == "catalog.updated"


async def test_aborted_steps_keep_the_frontend_stop_outcome():
    converter = OpencodeEvents(AsyncMock())
    converted = await converter.convert(
        event(
            "session.step.failed",
            assistantMessageID="stopped",
            error={"type": "aborted", "message": "Step interrupted"},
        )
    )
    info = converted[0]["properties"]["info"]
    assert info["time"]["completed"] == 10
    assert info["error"]["name"] == "MessageAbortedError"
    stored = message_record(
        {
            "id": "stopped",
            "type": "assistant",
            "error": {"type": "aborted", "message": "Step interrupted"},
        },
        "child",
    )
    assert stored["message"]["error"] == info["error"]
