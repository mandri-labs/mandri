import asyncio
import os
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import psutil
import pytest
from mandri.core.types.execution import ExecutionPhase
from mandri.runtime.docker_process import DockerProcess
from mandri.runtime.errors import SessionNotRunningError
from mandri.runtime.process import ManagedProcess, spawn
from mandri.runtime.service import RuntimeService


async def test_force_stop_kills_before_persistence_or_control_cleanup():
    runtime = RuntimeService({})
    process = SimpleNamespace(returncode=-9, kill=AsyncMock(return_value=-9), stop=AsyncMock())
    runtime.registry.mark_live("session", process, "codex")

    async def persist(*args, **kwargs):
        process.kill.assert_awaited_once()

    runtime._persist_execution_exit = AsyncMock(side_effect=persist)
    runtime._close_control = AsyncMock(side_effect=persist)
    runtime._restore_after_release = AsyncMock()
    assert await runtime.stop_session("session", force=True, restore_native=False) == -9
    process.stop.assert_not_awaited()
    assert runtime._persist_execution_exit.await_args_list == [
        call("session", ExecutionPhase.STOPPING),
        call("session", ExecutionPhase.STOPPED, process=process),
    ]
    runtime._restore_after_release.assert_not_awaited()
    assert runtime.registry.process("session") is None


async def test_managed_kill_bypasses_stdin_and_termination_ladder():
    child = SimpleNamespace(returncode=None, wait=AsyncMock(return_value=-9), stdin=Mock())
    tree = Mock()
    process = ManagedProcess(child, tree_terminator=tree)
    assert await process.kill() == -9
    tree.kill.assert_called_once()
    tree.terminate.assert_not_called()
    tree.release.assert_called_once()
    child.stdin.close.assert_not_called()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
async def test_force_stop_kills_uncooperative_process_and_child(tmp_path):
    program = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    process = await spawn([sys.executable, "-c", program], cwd=tmp_path, env={})
    try:
        child_pid = int(await asyncio.wait_for(process.read_stdout_line(), 3))
        assert await asyncio.wait_for(process.kill(), 3) == -signal.SIGKILL
        if psutil.pid_exists(child_pid):
            assert psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        process.kill_now()
        await process.wait()


async def test_docker_force_stop_kills_container_without_grace_and_releases_resources():
    exited = asyncio.Event()

    async def wait():
        await exited.wait()
        return 0

    async def run(*args, **kwargs):
        assert args == ("kill", "container")
        exited.set()

    attach = SimpleNamespace(
        process=SimpleNamespace(), wait=wait, kill=AsyncMock(), stop=AsyncMock()
    )
    client = SimpleNamespace(
        run=AsyncMock(side_effect=run),
        json=AsyncMock(return_value={"Running": False, "ExitCode": 137, "OOMKilled": False}),
    )
    cleanup = AsyncMock()
    process = DockerProcess(attach, client, "container", cleanup)
    assert await process.kill() == 137
    assert await process.kill() == 137
    client.run.assert_awaited_once_with("kill", "container", check=False)
    attach.kill.assert_awaited_once()
    attach.stop.assert_not_awaited()
    cleanup.assert_awaited_once()


async def test_force_stop_prevents_agy_steering_from_restarting():
    runtime = RuntimeService({})
    entered = asyncio.Event()
    interrupted = asyncio.Event()

    async def interrupt():
        entered.set()
        await interrupted.wait()

    runtime.registry.mark_live(
        "session", SimpleNamespace(returncode=None, kill=AsyncMock(return_value=-9)), "agy"
    )
    runtime._require_control = Mock(return_value=SimpleNamespace(interrupt=interrupt))
    runtime._capture_identity = AsyncMock(return_value="native")
    runtime._resume_session = AsyncMock()
    steering = asyncio.create_task(runtime._restart_agy("session"))
    await asyncio.wait_for(entered.wait(), 1)
    await runtime.stop_session("session", force=True, restore_native=False)
    interrupted.set()
    with pytest.raises(SessionNotRunningError):
        await steering
    runtime._resume_session.assert_not_awaited()
