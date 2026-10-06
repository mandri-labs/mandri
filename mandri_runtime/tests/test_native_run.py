import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.native_run import NativeRunStart
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.runtime.docker_process import DockerProcess
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.service import RuntimeService


@pytest.fixture
def native_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("mandri.runtime.service.require_spawn_executable", lambda value: value)
    sessions = SimpleNamespace(
        create_session=AsyncMock(return_value=SimpleNamespace(id="native-session")),
        set_session_state=AsyncMock(),
    )
    routes = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(id="native-route")),
        delete=AsyncMock(),
    )
    scopes = SimpleNamespace(create=AsyncMock(return_value="native-scope"), delete=AsyncMock())
    runtime = RuntimeService(
        {"claude": ["claude", "--print", "--input-format", "stream-json"]},
        sessions=sessions,
        routes=routes,
        privacy_scopes=scopes,
        gateway_port=8000,
        token_issuer=lambda route: f"token-for-{route}",
    )
    runtime._resolve_metadata = AsyncMock(return_value=None)
    runtime._spawn_execution = AsyncMock(return_value=SimpleNamespace())
    runtime.stop_session = AsyncMock()
    spec = NativeRunStart(
        harness="claude", model="provider/model", cwd=str(tmp_path), privacy_mode="surrogate"
    )
    return runtime, sessions, routes, scopes, spec


async def test_native_binds_privacy_scope_and_launches_native_interface(native_runtime):
    runtime, sessions, routes, scopes, spec = native_runtime
    async with runtime.native_run(spec) as plan:
        assert plan.argv == ["claude"]
        assert plan.env["ANTHROPIC_AUTH_TOKEN"] == "token-for-native-route"
        assert plan.env["CLAUDE_CODE_SUBAGENT_MODEL"] == "provider/model"
        assert routes.create.call_args.kwargs["privacy_scope_id"] == "native-scope"
        assert routes.create.call_args.kwargs["privacy_mode"] is PrivacyMode.SURROGATE
        assert sessions.create_session.call_args.kwargs["privacy_scope_id"] == "native-scope"
        runtime._spawn_execution.assert_not_awaited()
    routes.delete.assert_awaited_once_with("native-route")
    scopes.delete.assert_not_awaited()


async def test_native_failed_launch_retains_scope_bound_to_failed_record(
    native_runtime, monkeypatch
):
    runtime, _, routes, scopes, spec = native_runtime

    def fail(value):
        raise RuntimeError("launch failure")

    monkeypatch.setattr("mandri.runtime.service.require_spawn_executable", fail)
    with pytest.raises(RuntimeError, match="launch failure"):
        async with runtime.native_run(spec):
            pytest.fail("Failed launch became usable")
    routes.delete.assert_awaited_once_with("native-route")
    scopes.delete.assert_not_awaited()
    assert runtime.registry.live_ids() == []


async def test_native_failed_record_creation_discards_unbound_scope(native_runtime):
    runtime, sessions, routes, scopes, spec = native_runtime
    sessions.create_session.side_effect = RuntimeError("record failure")
    with pytest.raises(RuntimeError, match="record failure"):
        async with runtime.native_run(spec):
            pytest.fail("Failed record became usable")
    routes.delete.assert_awaited_once_with("native-route")
    scopes.delete.assert_awaited_once_with("native-scope")
    runtime._spawn_execution.assert_not_awaited()


async def test_unavailable_docker_run_cannot_fall_back_to_host(native_runtime):
    runtime, _, routes, scopes, spec = native_runtime
    spec.execution_backend = ExecutionBackend.DOCKER
    with pytest.raises(DockerExecutionError, match="Docker execution is not configured"):
        async with runtime.native_run(spec):
            pytest.fail("Docker absence was ignored")
    runtime._spawn_execution.assert_not_awaited()
    scopes.create.assert_not_awaited()
    routes.create.assert_not_awaited()


async def test_cancelled_native_run_waits_for_cleanup_to_finish(native_runtime):
    runtime, _, _, _, spec = native_runtime
    stopping = asyncio.Event()
    release = asyncio.Event()
    stopped = asyncio.Event()

    async def stop(*args, **kwargs):
        stopping.set()
        await release.wait()
        stopped.set()

    runtime._rollback_route = stop

    async def run():
        async with runtime.native_run(spec):
            pass

    task = asyncio.create_task(run())
    await asyncio.wait_for(stopping.wait(), 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert stopped.is_set()


async def test_missing_privacy_storage_fails_before_any_harness_launch(native_runtime):
    runtime, _, routes, _, spec = native_runtime
    runtime._privacy_scopes = None
    with pytest.raises(ProtectionError, match="Privacy storage"):
        async with runtime.native_run(spec):
            pytest.fail("Privacy absence was ignored")
    runtime._spawn_execution.assert_not_awaited()
    routes.create.assert_not_awaited()


@pytest.mark.parametrize("tty", [False, True])
async def test_docker_plan_attaches_directly_with_scoped_configuration(native_runtime, tty):
    runtime, _, routes, _, spec = native_runtime
    spec.execution_backend = ExecutionBackend.DOCKER
    spec.tty = tty
    runtime._docker = SimpleNamespace(prepare_image=AsyncMock())
    runtime._validate_policy = AsyncMock()
    process = DockerProcess.__new__(DockerProcess)
    process.native_argv = [
        "docker",
        "exec",
        "--interactive",
        "container",
        "python3",
        "/opt/mandri/worker.py",
        "exec",
        "claude",
    ]
    process._exit_code = None
    runtime._spawn_execution.return_value = process
    async with runtime.native_run(spec) as plan:
        expected = ["docker", "exec", "--interactive"]
        if tty:
            expected.append("--tty")
        assert plan.argv == [
            *expected,
            "container",
            "python3",
            "/opt/mandri/worker.py",
            "exec",
            "claude",
        ]
        assert plan.env == {}
        assert runtime._spawn_execution.call_args.kwargs == {"native": True}
        assert routes.create.call_args.kwargs["privacy_mode"] is PrivacyMode.SURROGATE
    runtime.stop_session.assert_awaited_once_with("native-session", restore_native=False)
    routes.delete.assert_awaited_once()
