"""Native permission profile contracts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessSessionId, ModeApplication
from mandri.core.types.config import SessionModeConfig
from mandri.runtime.control.claude import ClaudeControlAdapter
from mandri.runtime.control.codex import CodexControlAdapter
from mandri.runtime.control.errors import ModeRejectedError
from mandri.runtime.control.modes import resolve_launch
from mandri.runtime.control.opencode import OpencodeControlAdapter
from mandri.runtime.service import RuntimeService


@pytest.mark.parametrize(
    "profile,policy,sandbox,reviewer",
    [
        ("ask", "on-request", "workspace-write", "user"),
        ("auto", "on-request", "workspace-write", "auto_review"),
        ("full-access", "never", "danger-full-access", "user"),
    ],
)
@pytest.mark.parametrize("resume", [False, True])
async def test_codex_profiles_reach_native_thread(profile, policy, sandbox, reviewer, resume):
    mode = resolve_launch("codex", profile, SessionModeConfig())
    adapter = CodexControlAdapter(
        None, None, mode.codex_params, HarnessSessionId("native-1") if resume else None
    )
    adapter._handshake = AsyncMock()
    adapter._call = AsyncMock(return_value={"result": {"thread": {"id": "native-1"}}})
    await adapter.capture_identity()
    expected = {"approvalPolicy": policy, "sandbox": sandbox, "approvalsReviewer": reviewer}
    if resume:
        expected["threadId"] = "native-1"
    adapter._call.assert_awaited_once_with("thread/resume" if resume else "thread/start", expected)


@pytest.mark.parametrize("mode,action", [("default", "ask"), ("auto", "allow")])
async def test_opencode_permissions_reach_native_session(mode, action):
    launch = resolve_launch("opencode", mode, SessionModeConfig())
    assert launch.opencode_config == {"permission": {"*": action}}
    assert not launch.auto_approve
    adapter = OpencodeControlAdapter("http://localhost", "s1")
    adapter._request = AsyncMock(return_value=SimpleNamespace(status_code=200))
    try:
        await adapter.set_mode(mode)
        adapter._request.assert_awaited_once_with(
            "PATCH",
            "/api/session/s1",
            {"permissions": [{"action": "*", "resource": "*", "effect": action}]},
        )
    finally:
        await adapter.aclose()


async def test_legacy_opencode_rules_use_native_array_shape():
    adapter = OpencodeControlAdapter("http://localhost", "s1")
    adapter._request = AsyncMock(return_value=SimpleNamespace(status_code=200))
    try:
        await adapter.set_mode("acceptEdits")
        adapter._request.assert_awaited_once_with(
            "PATCH",
            "/api/session/s1",
            {"permissions": [{"action": "edit", "resource": "*", "effect": "allow"}]},
        )
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("mode", ["ask", "auto", "full-access"])
@pytest.mark.parametrize("active", [False, True])
async def test_codex_live_permissions_are_acknowledged(mode, active):
    adapter = CodexControlAdapter(None, None)
    adapter._thread_id = "thread"
    adapter._active_turn_id = "turn" if active else None
    adapter._call = AsyncMock(return_value={"result": {}})
    result = await adapter.set_mode(mode)
    if active:
        assert result is ModeApplication.REQUIRES_RESTART
        adapter._call.assert_not_awaited()
        return
    assert result is ModeApplication.MID_SESSION_APPLIED
    adapter._call.assert_awaited_once_with(
        "thread/settings/update",
        {
            "threadId": "thread",
            "approvalPolicy": "never" if mode == "full-access" else "on-request",
            "approvalsReviewer": "auto_review" if mode == "auto" else "user",
            "sandboxPolicy": {
                "type": "dangerFullAccess" if mode == "full-access" else "workspaceWrite"
            },
        },
    )
    assert adapter._turn_params("thread", "hello")["approvalPolicy"] == (
        "never" if mode == "full-access" else "on-request"
    )


async def test_codex_rejected_permissions_do_not_change_next_turn():
    adapter = CodexControlAdapter(None, None)
    adapter._thread_id = "thread"
    adapter._approval_policy = "on-request"
    adapter._call = AsyncMock(return_value={"error": {"message": "unsupported"}})
    with pytest.raises(ModeRejectedError, match="unsupported"):
        await adapter.set_mode("full-access")
    assert adapter._approval_policy == "on-request"


@pytest.mark.parametrize("accepted", [False, True])
async def test_claude_sends_permission_command_and_requires_success(accepted):
    adapter = ClaudeControlAdapter(stdout_pump=None, stderr_pump=None, stdin=None)
    adapter._session_id = HarnessSessionId("native")
    adapter._turn_active = True
    adapter._await_ack = AsyncMock(
        return_value={"subtype": "success" if accepted else "error", "error": "rejected"}
    )
    if accepted:
        assert (await adapter.set_mode("acceptEdits")).value == "mid_session_applied"
    else:
        with pytest.raises(ModeRejectedError):
            await adapter.set_mode("acceptEdits")
    identifier, frame = adapter._await_ack.call_args.args
    assert frame == {
        "type": "control_request",
        "request_id": identifier,
        "request": {"subtype": "set_permission_mode", "mode": "acceptEdits"},
    }


@pytest.mark.parametrize("mode", ["ask", "auto", "full-access", "default", "acceptEdits", "plan"])
async def test_runtime_forwards_modes_and_persists_only_after_ack(mode):
    runtime = RuntimeService(harness_commands={})
    control = SimpleNamespace(set_mode=AsyncMock(return_value=ModeApplication.MID_SESSION_APPLIED))
    runtime._require_control = Mock(return_value=control)
    runtime._capture_identity = AsyncMock()
    runtime._persist_interaction_mode = AsyncMock()
    await runtime.set_session_mode("session", mode)
    control.set_mode.assert_awaited_once_with(mode)
    runtime._persist_interaction_mode.assert_awaited_once_with(
        "session", mode, ModeApplication.MID_SESSION_APPLIED
    )
    runtime._persist_interaction_mode.reset_mock()
    control.set_mode.side_effect = ModeRejectedError("rejected")
    with pytest.raises(ModeRejectedError):
        await runtime.set_session_mode("session", mode)
    runtime._persist_interaction_mode.assert_not_awaited()
