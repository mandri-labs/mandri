from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import ApprovalStatus, HarnessKind, ModeApplication, SessionId
from mandri.runtime.control.errors import ControlTransportError, ModeRejectedError
from mandri.runtime.control.modes import resolve_launch
from mandri.runtime.service import RuntimeService


def permission(kind="permission.asked", reference="permission"):
    return {
        "source": "opencode",
        "raw": {
            "type": kind,
            "properties": {"id": reference, "sessionID": "child", "permission": "bash"},
        },
    }


def setup_runtime(mode="auto"):
    hub = Hub()
    runtime = RuntimeService({}, hub=hub)
    state = runtime._session_state("parent")
    runtime.registry.mark_live("parent", SimpleNamespace(returncode=None), "opencode")
    state.control = SimpleNamespace(
        set_mode=AsyncMock(return_value=ModeApplication.MID_SESSION_APPLIED)
    )
    state.delivery = SimpleNamespace(deliver=AsyncMock(return_value=True))
    runtime._capture_identity = AsyncMock()
    launch = resolve_launch("opencode", mode, runtime._mode_defaults)
    runtime._attach_approvals("parent", HarnessKind.OPENCODE, launch)
    return runtime, state, hub


async def test_automatic_child_permission_is_delivered_without_a_pending_card():
    runtime, state, hub = setup_runtime()
    handle = hub.subscribe(Topic("session.parent"))
    try:
        await state.watcher._process(permission())
        state.delivery.deliver.assert_awaited_once()
        assert not runtime._approvals.pending_for_session(SessionId("parent"))
        assert handle.queue.get_nowait()["payload"]["type"] == "approval.resolved"
        assert handle.queue.empty()
    finally:
        await state.watcher.stop()
        hub.unsubscribe(handle)


@pytest.mark.parametrize("event", ["question.asked", "permission.asked"])
async def test_questions_and_failed_automatic_delivery_remain_visible(event):
    runtime, state, hub = setup_runtime()
    state.delivery.deliver.side_effect = ControlTransportError("Offline")
    handle = hub.subscribe(Topic("session.parent"))
    try:
        await state.watcher._process(permission(event))
        if event == "question.asked":
            state.delivery.deliver.assert_not_awaited()
        else:
            state.delivery.deliver.assert_awaited_once()
        assert handle.queue.get_nowait()["payload"]["type"] == "approval.pending"
        assert runtime._approvals.pending_for_session(SessionId("parent"))
    finally:
        await state.watcher.stop()
        hub.unsubscribe(handle)


async def test_live_mode_switch_approves_existing_child_request_and_stops_when_disabled():
    runtime, state, _ = setup_runtime("default")
    try:
        await state.watcher._process(permission())
        request = runtime._approvals.pending_for_session(SessionId("parent"))[0]
        await runtime.set_session_mode("parent", "auto")
        assert request.status is ApprovalStatus.ANSWERED
        await runtime.set_session_mode("parent", "default")
        await state.watcher._process(permission(reference="second"))
        assert len(runtime._approvals.pending_for_session(SessionId("parent"))) == 1
        assert state.delivery.deliver.await_count == 1
    finally:
        await state.watcher.stop()


async def test_rejected_mode_switch_keeps_manual_approval_policy():
    runtime, state, _ = setup_runtime("default")
    state.control.set_mode.side_effect = ModeRejectedError("Rejected")
    try:
        with pytest.raises(ModeRejectedError):
            await runtime.set_session_mode("parent", "auto")
        await state.watcher._process(permission())
        state.delivery.deliver.assert_not_awaited()
    finally:
        await state.watcher.stop()
