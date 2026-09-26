import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.availability import SessionOwner
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.control.errors import ControlTransportError, ThreadOwnershipError
from mandri.runtime.native_restore import restore_codex_model
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.ownership.service import NativeOwnership


@pytest.fixture
def session():
    return SimpleNamespace(
        id="session",
        harness=HarnessKind.CODEX,
        native_id="native",
        project_path="/workspace",
        model="gateway/model",
        model_source=ModelSource.GATEWAY,
        execution_backend=ExecutionBackend.HOST,
        privacy_mode=PrivacyMode.NONE,
        privacy_scope_id=None,
        gateway_route_id="route",
    )


def process_reply(result):
    return SimpleNamespace(
        read_stdout_line=AsyncMock(
            side_effect=[
                json.dumps({"id": 1, "result": {}}),
                json.dumps({"id": 2, **result}),
                json.dumps({"id": 3, "result": {}}),
            ]
        ),
        read_stderr_line=AsyncMock(return_value=None),
        write_stdin=AsyncMock(),
        stop=AsyncMock(return_value=0),
    )


async def test_restore_uses_native_resume_without_a_turn_or_gateway_mutation(session, monkeypatch):
    process = process_reply(
        {"result": {"thread": {"id": "native", "modelProvider": "openai"}, "model": "gpt-5.6-luna"}}
    )
    spawn = AsyncMock(return_value=process)
    monkeypatch.setenv("MANDRI_API_KEY", "synthetic-session-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://synthetic-gateway")
    await restore_codex_model(session, {"codex": ["codex", "app-server"]}, spawn)
    requests = [json.loads(call.args[0]) for call in process.write_stdin.await_args_list]
    assert [request["method"] for request in requests] == [
        "initialize",
        "initialized",
        "thread/resume",
        "thread/settings/update",
    ]
    assert requests[-2]["params"] == {
        "threadId": "native",
        "model": "gpt-5.6-luna",
        "modelProvider": "openai",
    }
    assert requests[-1]["params"] == {"threadId": "native", "model": "gpt-5.6-luna"}
    assert requests[0]["params"]["capabilities"] == {"experimentalApi": True}
    assert "MANDRI_API_KEY" not in spawn.call_args.kwargs["env"]
    assert "OPENAI_BASE_URL" not in spawn.call_args.kwargs["env"]
    assert session.gateway_route_id == "route"
    assert session.model_source is ModelSource.GATEWAY
    process.stop.assert_awaited_once_with(grace=1)


@pytest.mark.parametrize(
    "reply,error",
    [
        (
            {"error": {"code": -32600, "message": "already has an active writer"}},
            ThreadOwnershipError,
        ),
        ({"result": {"thread": {"id": "other"}, "model": "gpt-5.6-luna"}}, ControlTransportError),
        (
            {"result": {"thread": {"id": "native", "modelProvider": "mandri"}, "model": "old"}},
            ControlTransportError,
        ),
    ],
)
async def test_failed_restore_reports_failure_and_releases_process(session, reply, error):
    process = process_reply(reply)
    with pytest.raises(error):
        await restore_codex_model(
            session, {"codex": ["codex", "app-server"]}, AsyncMock(return_value=process)
        )
    process.stop.assert_awaited_once_with(grace=1)


async def test_successful_resume_without_persistent_update_is_not_success(session):
    process = process_reply({})
    process.read_stdout_line.side_effect = [
        json.dumps({"id": 1, "result": {}}),
        json.dumps(
            {
                "id": 2,
                "result": {
                    "thread": {"id": "native", "modelProvider": "openai"},
                    "model": "gpt-5.6-luna",
                },
            }
        ),
        json.dumps({"id": 3, "error": {"code": -32600, "message": "settings unavailable"}}),
    ]
    with pytest.raises(ControlTransportError):
        await restore_codex_model(
            session, {"codex": ["codex", "app-server"]}, AsyncMock(return_value=process)
        )
    process.stop.assert_awaited_once_with(grace=1)


@pytest.mark.parametrize("source", [ModelSource.GATEWAY, ModelSource.NATIVE])
async def test_normal_release_repairs_gateway_selection_only(session, source):
    sessions = SimpleNamespace(set_session_state=AsyncMock())
    runtime = RuntimeService({}, sessions=sessions)
    process = Mock(returncode=None, stop=AsyncMock(return_value=0))
    runtime.registry.mark_live("session", process, "codex")
    runtime._session_state("session").launched_model = (source, "selected", None)

    async def restore(_session):
        assert runtime.registry.process("session") is None
        process.stop.assert_awaited_once()

    runtime._restore_native_model = AsyncMock(side_effect=restore)
    assert await runtime.stop_session("session") == 0
    assert runtime._restore_native_model.await_count == (1 if source is ModelSource.GATEWAY else 0)


async def test_failed_automatic_restore_does_not_leave_the_old_process_live():
    runtime = RuntimeService({}, sessions=SimpleNamespace(set_session_state=AsyncMock()))
    runtime.registry.mark_live(
        "session", Mock(returncode=None, stop=AsyncMock(return_value=0)), "codex"
    )
    runtime._session_state("session").launched_model = (ModelSource.GATEWAY, "selected", None)
    runtime._restore_native_model = AsyncMock(side_effect=ControlTransportError("unavailable"))
    assert await runtime.stop_session("session") == 0
    assert runtime.registry.process("session") is None
    assert runtime._session_state("session").native_restore_failed


@pytest.mark.parametrize(
    "owner", [SessionOwner.UNOWNED, SessionOwner.EXTERNAL, SessionOwner.UNKNOWN]
)
async def test_manual_restore_requires_inactive_unowned_session(session, owner):
    sessions = SimpleNamespace(
        get_session=AsyncMock(return_value=session),
        native_ownership=AsyncMock(return_value=NativeOwnership(owner)),
        external_status=AsyncMock(return_value=(False, None)),
    )
    runtime = RuntimeService({}, sessions=sessions)
    runtime._restore_native_model = AsyncMock()
    if owner is SessionOwner.UNOWNED:
        await runtime.restore_native_model("session")
        await runtime.restore_native_model("session")
        assert runtime._restore_native_model.await_count == 2
    else:
        with pytest.raises(SessionConflictError):
            await runtime.restore_native_model("session")
        runtime._restore_native_model.assert_not_awaited()
    assert not runtime._session_state("session").resuming
