from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.hub import Hub
from mandri.core.ids import HarnessSessionId, SessionId
from mandri.core.types.availability import SessionOwner
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionConflictError


async def test_pi_session_change_adopts_discovery_before_publishing_new_history():
    sessions = SimpleNamespace(adopt_pi_native_id=AsyncMock())
    runtime = RuntimeService({}, sessions=sessions, hub=Hub())
    process = SimpleNamespace(returncode=None, process=SimpleNamespace(pid=123))
    runtime.registry.mark_live("session", process, "pi")
    runtime._session_state("session").native_id = HarnessSessionId("old")
    runtime._events.publish_event = AsyncMock()
    await runtime._on_conversation_reset("session", process)(HarnessSessionId("new"))
    sessions.adopt_pi_native_id.assert_awaited_once_with(
        SessionId("session"), "new", ignored_pid=123
    )
    assert runtime._session_state("session").native_id == "new"
    assert runtime._events.publish_event.call_args.args[1]["raw"] == {
        "type": "history_changed",
        "reset": True,
    }


async def test_pi_rejected_session_switch_stops_writer_and_preserves_original_identity():
    sessions = SimpleNamespace(
        adopt_pi_native_id=AsyncMock(side_effect=SessionConflictError("Already managed"))
    )
    runtime = RuntimeService({}, sessions=sessions)
    process = SimpleNamespace(returncode=None, kill_now=Mock(), process=SimpleNamespace(pid=123))
    runtime.registry.mark_live("session", process, "pi")
    runtime._session_state("session").native_id = HarnessSessionId("old")
    with pytest.raises(ControlTransportError, match="sidebar"):
        await runtime._on_conversation_reset("session", process)(HarnessSessionId("claimed"))
    process.kill_now.assert_called_once()
    assert runtime._session_state("session").native_id == "old"


def test_pi_process_claims_follow_runtime_identity_and_ignore_dead_processes():
    sessions = SimpleNamespace(set_pi_processes=Mock())
    runtime = RuntimeService({}, sessions=sessions)
    for index, native in enumerate(("first", "second", None)):
        session_id = str(index)
        runtime.registry.mark_live(
            session_id,
            SimpleNamespace(returncode=None, process=SimpleNamespace(pid=100 + index)),
            "pi",
        )
        runtime._session_state(session_id).native_id = native
    callback = sessions.set_pi_processes.call_args.args[0]
    assert callback() == {100: "first", 101: "second"}
    runtime._session_state("0").native_id = HarnessSessionId("changed")
    runtime.registry.process("1").returncode = 0
    assert callback() == {100: "changed"}


async def test_pi_resume_waits_for_another_managed_identity_before_rechecking_writer():
    sessions = SimpleNamespace(
        native_ownership=AsyncMock(
            side_effect=[
                SimpleNamespace(owner=SessionOwner.UNKNOWN),
                SimpleNamespace(owner=SessionOwner.UNOWNED),
            ]
        )
    )
    runtime = RuntimeService({}, sessions=sessions)
    process = SimpleNamespace(returncode=None, process=SimpleNamespace(pid=42))
    runtime.registry.mark_live("starting", process, "pi")
    control = SimpleNamespace(
        capture_identity=AsyncMock(return_value=HarnessSessionId("different"))
    )
    runtime._session_state("starting").control = control
    await runtime._ensure_session_available("resuming")
    control.capture_identity.assert_awaited_once()
    assert sessions.native_ownership.await_count == 2
    assert runtime._native_pi_processes() == {42: "different"}


@pytest.mark.parametrize("pending_user_selection", [False, True])
async def test_pi_native_command_selection_persists_without_overwriting_pending_choice(
    pending_user_selection,
):
    record = SimpleNamespace(
        model_source=ModelSource.NATIVE,
        model="user-choice" if pending_user_selection else "original",
        reasoning_effort=None,
    )
    sessions = SimpleNamespace(
        get_session=AsyncMock(return_value=record),
        observe_native_selection=AsyncMock(return_value=True),
    )
    runtime = RuntimeService({}, sessions=sessions)
    process = SimpleNamespace(returncode=None)
    runtime.registry.mark_live("session", process, "pi")
    state = runtime._session_state("session")
    state.launched_model = (ModelSource.NATIVE, "original", None)
    await runtime._observe_pi_selection("session", process, "custom/new", "high")
    if pending_user_selection:
        sessions.observe_native_selection.assert_not_awaited()
        assert state.launched_model == (ModelSource.NATIVE, "original", None)
    else:
        sessions.observe_native_selection.assert_awaited_once_with(record, "custom/new", "high")
        assert state.launched_model == (ModelSource.NATIVE, "custom/new", "high")
