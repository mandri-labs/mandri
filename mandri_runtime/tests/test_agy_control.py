import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from mandri.core.ids import (
    ApprovalDecision,
    ApprovalId,
    ApprovalKind,
    ApprovalStatus,
    EpochMs,
    HarnessKind,
    HarnessSessionId,
    ModeApplication,
    RawEvent,
    SessionId,
)
from mandri.core.ports.control import PromptState
from mandri.core.types.approvals import ApprovalRequest
from mandri.runtime.control.agy import AgyControlAdapter
from mandri.runtime.control.agy_bridge import AgyBridge
from mandri.runtime.control.agy_policy import AgyPolicy
from mandri.runtime.control.errors import (
    ControlTransportError,
    ModeRejectedError,
    PromptDeliveryFailedError,
)
from mandri.runtime.pump import LinePump


class Events:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.approval = asyncio.Event()

    async def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if event.get("event") == "approval_request":
            self.approval.set()


class Sink:
    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        if self.closed:
            raise BrokenPipeError()
        self.frames.append(data)

    async def drain(self) -> None:
        pass


def tool(name: str = "run_command", **args: Any) -> dict[str, Any]:
    return {
        "conversationId": "conversation",
        "stepIdx": 2,
        "toolCall": {"name": name, "args": args},
    }


def response(
    decision: ApprovalDecision = ApprovalDecision.ALLOW,
    *,
    answers: list[dict[str, Any]] | None = None,
    status: ApprovalStatus = ApprovalStatus.ANSWERED,
) -> ApprovalRequest:
    return ApprovalRequest(
        ApprovalId("approval"),
        SessionId("session"),
        HarnessKind.AGY,
        RawEvent("{}"),
        "conversation:2",
        ApprovalKind.COMMAND_EXECUTION,
        EpochMs(0),
        EpochMs(1000),
        status,
        decision,
        answers,
    )


async def test_hook_blocks_until_correlated_answer_and_reuses_answer(tmp_path: Path) -> None:
    events = Events()
    bridge = AgyBridge(AgyPolicy("default", tmp_path, {}), events.publish)
    request = tool(CommandLine="echo synthetic")
    pending = asyncio.create_task(bridge.handle("PreToolUse", request))
    await asyncio.wait_for(events.approval.wait(), 1)
    assert not pending.done()
    assert await bridge.deliver(response())
    assert await pending == {"decision": "allow"}
    assert await bridge.handle("PreToolUse", request) == {"decision": "allow"}
    assert not await bridge.deliver(response())
    assert len([event for event in events.events if event["event"] == "approval_request"]) == 1
    await bridge.aclose()


async def test_closed_hook_cancels_pending_and_invalidates_token(tmp_path: Path) -> None:
    events = Events()
    bridge = AgyBridge(AgyPolicy("default", tmp_path, {}), events.publish)
    pending = asyncio.create_task(bridge.handle("PreToolUse", tool(CommandLine="echo synthetic")))
    await asyncio.wait_for(events.approval.wait(), 1)
    assert bridge.authenticates(bridge.token)
    assert not bridge.authenticates("wrong-token")
    await bridge.aclose()
    assert (await pending)["decision"] == "deny"
    assert not bridge.authenticates(bridge.token)
    with pytest.raises(ControlTransportError, match="closed"):
        await bridge.handle("Stop", {})


async def test_expired_hook_refuses_late_answer(tmp_path: Path) -> None:
    events = Events()
    bridge = AgyBridge(AgyPolicy("default", tmp_path, {}), events.publish, timeout=0)
    result = await asyncio.wait_for(bridge.handle("PreToolUse", tool(CommandLine="echo test")), 7)
    assert result["decision"] == "deny"
    assert "expired" in result["reason"]
    assert not await bridge.deliver(response())
    await bridge.aclose()


async def test_questions_require_structured_answers_and_inject_once(tmp_path: Path) -> None:
    events = Events()
    bridge = AgyBridge(AgyPolicy("default", tmp_path, {}), events.publish)
    pending = asyncio.create_task(bridge.handle("PreToolUse", tool("ask_question", questions=[])))
    await asyncio.wait_for(events.approval.wait(), 1)
    assert not await bridge.deliver(response(answers=[{"question": "Colour?", "answers": "blue"}]))
    assert not pending.done()
    answers = [{"question": "Colour?", "answers": ["blue"]}]
    assert await bridge.deliver(response(answers=answers))
    assert (await pending)["decision"] == "deny"
    injected = await bridge.handle("PreInvocation", {"conversationId": "conversation"})
    assert json.loads(injected["injectSteps"][0]["userMessage"]) == answers
    assert await bridge.handle("PreInvocation", {"conversationId": "conversation"}) == {
        "injectSteps": []
    }
    await bridge.aclose()


@pytest.mark.parametrize("step", [None, -1, True, "2"])
async def test_invalid_hook_identity_is_denied(tmp_path: Path, step: object) -> None:
    events = Events()
    bridge = AgyBridge(AgyPolicy("bypassPermissions", tmp_path, {}), events.publish)
    request = tool(CommandLine="echo test")
    request["stepIdx"] = step
    assert (await bridge.handle("PreToolUse", request))["decision"] == "deny"
    await bridge.aclose()


async def test_reused_step_cannot_authorize_different_command(tmp_path: Path) -> None:
    events = Events()
    bridge = AgyBridge(AgyPolicy("default", tmp_path, {}), events.publish)
    pending = asyncio.create_task(bridge.handle("PreToolUse", tool(CommandLine="echo approved")))
    await asyncio.wait_for(events.approval.wait(), 1)
    await bridge.deliver(response())
    await pending
    result = await bridge.handle("PreToolUse", tool(CommandLine="echo different"))
    assert result["decision"] == "deny"
    assert "collision" in result["reason"].lower()
    await bridge.aclose()


@pytest.mark.parametrize(
    "mode,name,args,expected",
    [
        ("default", "view_file", {"AbsolutePath": "file.py"}, "allow"),
        ("default", "view_file", {"AbsolutePath": "../outside.py"}, "ask"),
        ("default", "write_to_file", {"TargetFile": "file.py"}, "ask"),
        ("acceptEdits", "write_to_file", {"TargetFile": "file.py"}, "allow"),
        ("acceptEdits", "write_to_file", {"TargetFile": "../outside.py"}, "ask"),
        ("plan", "run_command", {"CommandLine": "echo test"}, "deny"),
        ("plan", "write_to_file", {"TargetFile": "file.py"}, "deny"),
        ("plan", "ask_question", {}, "ask"),
        ("default", "future_unknown_tool", {}, "ask"),
    ],
)
def test_policy_modes_respect_workspace_and_unknown_tools(
    tmp_path: Path, mode: str, name: str, args: dict[str, Any], expected: str
) -> None:
    assert AgyPolicy(mode, tmp_path, {}).decision(name, args) == expected


def test_permission_denial_overrides_full_access_and_plan_overrides_allow(tmp_path: Path) -> None:
    denied = AgyPolicy("bypassPermissions", tmp_path, {"deny": ["command(echo)"]})
    assert denied.decision("run_command", {"CommandLine": "echo test"}) == "deny"
    plan = AgyPolicy("plan", tmp_path, {"allow": ["*"]})
    assert plan.decision("write_to_file", {"TargetFile": "file.py"}) == "deny"


@pytest.mark.parametrize(
    "command", ["echo safe; other", "echo safe && other", "echo $(other)", "echo safe | other"]
)
def test_command_prefix_grant_does_not_allow_composed_shell(tmp_path: Path, command: str) -> None:
    policy = AgyPolicy("default", tmp_path, {"allow": ["command(echo)"]})
    assert policy.decision("run_command", {"CommandLine": command}) == "ask"


async def interrupt() -> bool:
    return True


def adapter(tmp_path: Path, frames: list[dict[str, Any]], expected: str | None = None):
    async def chunks() -> AsyncIterator[bytes]:
        for frame in frames:
            yield (json.dumps(frame) + "\n").encode()

    sink = Sink()
    events = Events()
    bridge = AgyBridge(AgyPolicy("default", tmp_path, {}), events.publish)
    control = AgyControlAdapter(
        LinePump(chunks),
        sink,
        bridge,
        interrupt,
        HarnessSessionId(expected) if expected else None,
    )
    return control, sink


async def test_control_initializes_and_writes_one_ndjson_frame_per_prompt(tmp_path: Path) -> None:
    control, sink = adapter(tmp_path, [{"event": "init", "conversation_id": "conversation"}])
    assert await control.capture_identity() == "conversation"
    result = await control.send_prompt("first\nsecond")
    assert result.state is PromptState.QUEUED
    assert len(sink.frames) == 1
    assert json.loads(sink.frames[0]) == {"event": "user", "message": {"content": "first\nsecond"}}
    assert await control.set_mode("plan") is ModeApplication.REQUIRES_RESTART
    with pytest.raises(ModeRejectedError):
        await control.set_mode("unknown")
    await control.aclose()


@pytest.mark.parametrize(
    "frames", [[], [{"event": "init"}], [{"event": "init", "conversation_id": "other"}]]
)
async def test_control_rejects_missing_or_replaced_identity(
    tmp_path: Path, frames: list[dict[str, Any]]
) -> None:
    control, sink = adapter(tmp_path, frames, expected="expected")
    with pytest.raises(PromptDeliveryFailedError):
        await control.send_prompt("must not be sent")
    assert sink.frames == []
    await control.aclose()


async def test_control_reports_closed_stdin(tmp_path: Path) -> None:
    control, sink = adapter(tmp_path, [{"event": "init", "conversation_id": "conversation"}])
    sink.closed = True
    with pytest.raises(PromptDeliveryFailedError, match="input closed"):
        await control.send_prompt("message")
    await control.aclose()
