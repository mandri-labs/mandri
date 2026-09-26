import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.availability import SessionOwner
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.ownership import processes, service
from mandri.sessions.ownership.file_lock import try_lock, unlock, writer_locked
from mandri.sessions.ownership.processes import ProcessIdentity
from mandri.sessions.ownership.service import NativeOwnership, inspect_owner, release_writer
from mandri.sessions.ownership.windows_lock import locking_processes


@pytest.mark.parametrize("locked", [False, True])
def test_codex_ownership_uses_native_lock_not_a_process_name(tmp_path, monkeypatch, locked):
    path = tmp_path / "native.lock"
    path.touch()
    monkeypatch.setattr(service, "codex_writer", lambda *args: None)
    monkeypatch.setattr(
        processes.psutil, "process_iter", Mock(side_effect=AssertionError("process scan"))
    )
    with path.open("r+b") as writer:
        if locked:
            assert try_lock(writer)
        try:
            owner = inspect_owner(HarnessKind.CODEX, "native", "/project", path)
            assert owner.owner is (SessionOwner.EXTERNAL if locked else SessionOwner.UNOWNED)
            assert not owner.can_release
        finally:
            if locked:
                unlock(writer)
    assert path.read_bytes() == b""


def test_missing_and_stale_codex_lock_are_unowned_without_removing_files(tmp_path):
    path = tmp_path / "native.lock"
    assert writer_locked(path) is False
    assert not path.exists()
    path.touch()
    assert writer_locked(path) is False
    assert path.exists()


def test_busy_coordination_is_unknown(tmp_path):
    path = tmp_path / "native.lock"
    path.touch()
    coordination = tmp_path / ".coordination.lock"
    coordination.touch()
    with coordination.open("r+b") as guard:
        assert try_lock(guard)
        try:
            assert writer_locked(path) is None
        finally:
            unlock(guard)


def test_unrelated_native_session_process_does_not_claim_this_session(monkeypatch):
    process = SimpleNamespace(
        pid=12,
        info={
            "name": "claude.exe",
            "cmdline": ["claude", "--resume", "other"],
            "cwd": "/project",
            "create_time": 1.0,
        },
    )
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: [process])
    assert inspect_owner(HarnessKind.CLAUDE, "native", "/project").owner is SessionOwner.UNOWNED


def test_shared_process_with_unresolved_session_is_unknown(monkeypatch):
    process = SimpleNamespace(
        pid=12,
        info={
            "name": "opencode",
            "cmdline": ["opencode", "serve"],
            "cwd": "/project",
            "create_time": 1.0,
        },
    )
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: [process])
    owner = inspect_owner(HarnessKind.OPENCODE, "native", "/project")
    assert owner.owner is SessionOwner.UNKNOWN
    assert not owner.can_release


def test_windows_external_process_release_is_not_misreported_as_graceful(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "win32")
    owner = NativeOwnership(SessionOwner.EXTERNAL, ProcessIdentity(1, 10.0, True))
    assert not owner.can_release


@pytest.mark.parametrize("changed", ["owner", "pid", "none"])
def test_external_release_rechecks_identity_before_any_signal(monkeypatch, changed):
    monkeypatch.setattr(service.sys, "platform", "linux")
    owner = NativeOwnership(SessionOwner.EXTERNAL, ProcessIdentity(10, 1.0, True))
    current = NativeOwnership(SessionOwner.UNKNOWN) if changed == "owner" else owner
    monkeypatch.setattr(service, "inspect_owner", lambda *args: current)
    process = Mock()
    process.create_time.return_value = 2.0 if changed == "pid" else 1.0
    monkeypatch.setattr(service.psutil, "Process", lambda pid: process)
    if changed != "none":
        with pytest.raises(SessionConflictError):
            release_writer(HarnessKind.CODEX, "native", "/project", None, owner)
        process.terminate.assert_not_called()
    else:
        release_writer(HarnessKind.CODEX, "native", "/project", None, owner)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        process.kill.assert_not_called()


@pytest.mark.parametrize("changed", [False, True])
def test_forced_release_requires_unchanged_writer_after_grace(monkeypatch, changed):
    monkeypatch.setattr(service.sys, "platform", "linux")
    owner = NativeOwnership(SessionOwner.EXTERNAL, ProcessIdentity(10, 1.0, True))
    next_owner = NativeOwnership(SessionOwner.UNKNOWN) if changed else owner
    monkeypatch.setattr(service, "inspect_owner", Mock(side_effect=[owner, next_owner]))
    process = Mock()
    process.create_time.return_value = 1.0
    process.wait.side_effect = [psutil.TimeoutExpired(5), None]
    monkeypatch.setattr(service.psutil, "Process", lambda pid: process)
    if changed:
        with pytest.raises(SessionConflictError):
            release_writer(HarnessKind.CODEX, "native", "/project", None, owner)
        process.kill.assert_not_called()
    else:
        release_writer(HarnessKind.CODEX, "native", "/project", None, owner)
        process.kill.assert_called_once()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Restart Manager")
async def test_windows_lock_reports_the_exact_synthetic_writer(tmp_path):
    path = tmp_path / "native.lock"
    path.touch()
    script = (
        "import msvcrt,sys,time; "
        "f=open(sys.argv[1],'r+b'); "
        "msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1); "
        "print('ready',flush=True); time.sleep(60)"
    )
    child = await asyncio.create_subprocess_exec(
        sys._base_executable, "-c", script, str(path), stdout=asyncio.subprocess.PIPE
    )
    try:
        assert (await asyncio.wait_for(child.stdout.readline(), 5)).strip() == b"ready"
        owners = await asyncio.to_thread(locking_processes, path)
        assert child.pid in owners
        assert abs(owners[child.pid] - psutil.Process(child.pid).create_time()) < 0.01
        assert writer_locked(path) is True
    finally:
        if child.returncode is None:
            child.terminate()
        await asyncio.wait_for(child.wait(), 5)
    assert writer_locked(path) is False
