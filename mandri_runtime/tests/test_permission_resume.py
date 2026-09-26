"""Permission overrides and approval recovery regressions."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.config.errors import ConfigError
from mandri.core.hub import Hub
from mandri.core.ids import (
    ApprovalKind,
    ApprovalStatus,
    HarnessKind,
    ModeApplication,
    RawEvent,
    SessionId,
)
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import InteractionMode
from mandri.runtime.control.claude import ClaudeApprovalMessenger
from mandri.runtime.service import RuntimeService
from mandri.runtime.session_feed import session_topic


@pytest.mark.parametrize(
    "harness,stored,override,expected",
    [
        (HarnessKind.CLAUDE, "default", "acceptEdits", "acceptEdits"),
        (HarnessKind.CLAUDE, "default", None, "default"),
        (HarnessKind.CODEX, "untrusted", "on-request", "on-request"),
        (HarnessKind.OPENCODE, "default", "acceptEdits", "acceptEdits"),
    ],
)
def test_resume_resolves_permission_override(harness, stored, override, expected):
    runtime = RuntimeService(harness_commands={})
    record = SimpleNamespace(harness=harness, interaction_mode=InteractionMode(stored, "at_launch"))
    mode = runtime._resume_launch_mode(record, mode=override)
    assert mode.mode == expected
    assert record.harness is harness
    assert record.interaction_mode.mode == stored


@pytest.mark.parametrize(
    "harness,mode",
    [
        (HarnessKind.CLAUDE, "invalid"),
        (HarnessKind.CODEX, "default"),
        (HarnessKind.OPENCODE, "another-agent"),
    ],
)
async def test_invalid_override_fails_before_spawn(harness, mode):
    runtime = RuntimeService(harness_commands={harness.value: [harness.value]})
    runtime._load_resumable_record = AsyncMock(
        return_value=SimpleNamespace(
            harness=harness,
            interaction_mode=None,
        )
    )
    runtime._spawn_harness = AsyncMock()
    runtime._resume_route = AsyncMock()
    with pytest.raises(ConfigError):
        await runtime.resume_session("s1", mode=mode)
    runtime._spawn_harness.assert_not_called()
    runtime._resume_route.assert_not_called()


async def test_pending_approval_replayed_to_new_subscriber():
    hub = Hub()
    runtime = RuntimeService(harness_commands={}, hub=hub)
    request = await runtime._approvals.register(
        SessionId("s1"),
        HarnessKind.CLAUDE,
        RawEvent(json.dumps({"type": "control_request", "request_id": "r1"})),
        "r1",
        ApprovalKind.COMMAND_EXECUTION,
        120,
    )
    handle = hub.subscribe(session_topic("s1"))
    runtime.replay_pending_approvals("s1")
    frame = handle.queue.get_nowait()["payload"]
    assert frame["type"] == "approval.pending"
    assert frame["approval_id"] == request.id
    assert frame["deadline"] == request.deadline
    await runtime._approvals.cancel(request.id)
    runtime.replay_pending_approvals("s1")
    assert handle.queue.empty()


async def test_expiry_is_not_reported_as_user_denial():
    runtime = RuntimeService(harness_commands={})
    request = await runtime._approvals.register(
        SessionId("s1"),
        HarnessKind.CLAUDE,
        RawEvent("{}"),
        "r1",
        ApprovalKind.COMMAND_EXECUTION,
        120,
    )
    request = request.transition(ApprovalStatus.EXPIRED)
    payloads = []
    sink = SimpleNamespace(write=payloads.append, drain=AsyncMock())
    await ClaudeApprovalMessenger(stdin=sink).deliver(request)
    response = json.loads(payloads[0])["response"]["response"]
    assert response["behavior"] == "deny"
    assert response["message"] == "Mandri approval expired without a user response"


@pytest.mark.parametrize("harness", [HarnessKind.CLAUDE, HarnessKind.PI])
async def test_resume_launches_with_override_and_persists_after_initialization(
    monkeypatch, harness
):
    runtime = RuntimeService(harness_commands={harness.value: [harness.value]})
    record = SimpleNamespace(
        id=SessionId("s1"),
        harness=harness,
        native_id="native-1",
        parent_session_id=None,
        interaction_mode=InteractionMode("default", "at_launch"),
        model="provider/model",
        model_source=ModelSource.GATEWAY,
        execution_backend=ExecutionBackend.HOST,
        privacy_mode=PrivacyMode.NONE,
        privacy_scope_id=None,
        reasoning_effort=None,
        project_path="/workspace",
    )
    runtime._load_resumable_record = AsyncMock(return_value=record)
    runtime._resume_route = AsyncMock(return_value="route-1")
    runtime._resolve_metadata = AsyncMock(return_value=None)
    process = SimpleNamespace()
    runtime._spawn_harness = AsyncMock(return_value=process)
    control = SimpleNamespace(
        capture_identity=AsyncMock(return_value="native-1"),
        set_mode=AsyncMock(return_value=ModeApplication.MID_SESSION_APPLIED),
    )
    runtime._session_state("s1").control = control
    monkeypatch.setattr(runtime, "_attach_feed", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_attach_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "_attach_liveness", lambda *args: None)
    monkeypatch.setattr(runtime._lifetime, "arm_zero_viewer_decision", lambda *args: None)
    runtime._reveal_identity = AsyncMock()
    runtime._mark_session_live = AsyncMock()
    persisted = AsyncMock()
    runtime._persist_interaction_mode = persisted
    result = await runtime.resume_session("s1", mode="acceptEdits")
    args = runtime._spawn_harness.call_args.args[0]
    if harness is HarnessKind.CLAUDE:
        assert args == ["claude", "--permission-mode", "acceptEdits"]
        control.capture_identity.assert_not_awaited()
    else:
        assert args[0] == "pi"
        control.set_mode.assert_awaited_once_with("acceptEdits")
    assert result.mode == "acceptEdits"
    assert result.harness == harness.value
    assert result.native_id == "native-1"
    persisted.assert_awaited_once()
    assert persisted.call_args.args[:2] == ("s1", "acceptEdits")


async def test_failed_spawn_does_not_persist_permission_mode():
    runtime = RuntimeService(harness_commands={"claude": ["claude"]})
    runtime._load_resumable_record = AsyncMock(
        return_value=SimpleNamespace(
            harness=HarnessKind.CLAUDE,
            native_id=None,
            interaction_mode=InteractionMode("default", "at_launch"),
            model="provider/model",
            model_source=ModelSource.GATEWAY,
            execution_backend=ExecutionBackend.HOST,
            privacy_mode=PrivacyMode.NONE,
            privacy_scope_id=None,
            reasoning_effort=None,
            project_path="/workspace",
        )
    )
    runtime._resume_route = AsyncMock(return_value="route-1")
    runtime._resolve_metadata = AsyncMock(return_value=None)
    runtime._spawn_harness = AsyncMock(side_effect=OSError("spawn failed"))
    runtime._persist_interaction_mode = AsyncMock()
    with pytest.raises(OSError):
        await runtime.resume_session("s1", mode="acceptEdits")
    runtime._persist_interaction_mode.assert_not_called()


@pytest.mark.parametrize("native_id", ["native-1", "rotated-native"])
async def test_resumed_claude_identity_is_persisted_when_it_arrives(native_id):
    sessions = SimpleNamespace(reveal_native_id=AsyncMock(), rotate_native_id=AsyncMock())
    runtime = RuntimeService(harness_commands={}, sessions=sessions)
    callback = runtime._on_identity("s1", "native-1")
    callback(native_id)
    await asyncio.gather(*runtime._control_tasks)
    if native_id == "native-1":
        sessions.reveal_native_id.assert_awaited_once_with(SessionId("s1"), native_id)
        sessions.rotate_native_id.assert_not_awaited()
    else:
        sessions.rotate_native_id.assert_awaited_once_with(SessionId("s1"), native_id)
