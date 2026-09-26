"""Managed child processes with line-oriented pipes and a termination ladder."""

import asyncio
import contextlib
import ctypes
import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from mandri.runtime.errors import (
    ProcessIOError,
    ProcessJobError,
    ProcessSpawnError,
    ProcessTerminationError,
)

DEFAULT_LINE_LIMIT = 10 * 1024 * 1024
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


class TreeTerminator(Protocol):
    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def release(self) -> None: ...


class ProcessGroupTerminator:
    def __init__(self, pid: int) -> None:
        self._pid = pid

    def terminate(self) -> None:
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        self._signal_group(_SIGKILL)

    def release(self) -> None:
        return None

    def _signal_group(self, sig: signal.Signals) -> None:
        killpg = getattr(os, "killpg", None)
        if killpg is None:
            return
        with contextlib.suppress(ProcessLookupError):
            killpg(self._pid, sig)


def platform_spawn_options() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": _CREATE_SUSPENDED | _CREATE_NO_WINDOW}
    return {"start_new_session": True}


def tree_terminator_factory() -> Callable[[int], TreeTerminator]:
    if sys.platform == "win32":
        return attach_job_object_terminator
    return ProcessGroupTerminator


if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _CREATE_SUSPENDED = 0x00000004
    _CREATE_NO_WINDOW = 0x08000000
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001
    _TH32CS_SNAPTHREAD = 0x00000004
    _THREAD_SUSPEND_RESUME = 0x0002
    _TREE_KILL_EXIT_CODE = 1
    _RESUME_THREAD_ERROR = 0xFFFFFFFF
    _RESUME_FAILED = 0xFFFFFFFF

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_uint32),
            ("cntUsage", ctypes.c_uint32),
            ("th32ThreadID", ctypes.c_uint32),
            ("th32OwnerProcessID", ctypes.c_uint32),
            ("tpBasePri", ctypes.c_long),
            ("tpDeltaPri", ctypes.c_long),
            ("dwFlags", ctypes.c_uint32),
        ]

    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    _kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    _kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    _kernel32.SetInformationJobObject.restype = ctypes.c_int32
    _kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _kernel32.AssignProcessToJobObject.restype = ctypes.c_int32
    _kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.TerminateJobObject.restype = ctypes.c_int32
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.OpenThread.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    _kernel32.OpenThread.restype = ctypes.c_void_p
    _kernel32.ResumeThread.argtypes = [ctypes.c_void_p]
    _kernel32.ResumeThread.restype = ctypes.c_uint32
    _kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    _kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    _kernel32.Thread32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(_THREADENTRY32)]
    _kernel32.Thread32First.restype = ctypes.c_int32
    _kernel32.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(_THREADENTRY32)]
    _kernel32.Thread32Next.restype = ctypes.c_int32
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_int32

    class JobObjectTerminator:
        def __init__(self, job_handle: int) -> None:
            self._job_handle: int | None = job_handle

        def terminate(self) -> None:
            self._terminate_job()

        def kill(self) -> None:
            self._terminate_job()
            self.release()

        def release(self) -> None:
            if self._job_handle is None:
                return
            job_handle = self._job_handle
            self._job_handle = None
            _kernel32.CloseHandle(job_handle)

        def _terminate_job(self) -> None:
            job_handle = self._job_handle
            if job_handle is None:
                return
            if not _kernel32.TerminateJobObject(job_handle, _TREE_KILL_EXIT_CODE):
                raise ProcessTerminationError("TerminateJobObject failed")

    def attach_job_object_terminator(pid: int) -> JobObjectTerminator:
        job_handle = _kernel32.CreateJobObjectW(None, None)
        if not job_handle:
            raise ProcessJobError("CreateJobObjectW failed")
        try:
            _set_job_kill_on_close(job_handle)
            _assign_process_to_job(pid, job_handle)
            _resume_main_thread(pid)
        except ProcessJobError:
            _kernel32.CloseHandle(job_handle)
            raise
        return JobObjectTerminator(int(job_handle))

    def _set_job_kill_on_close(job_handle: int) -> None:
        limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            job_handle,
            _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise ProcessJobError("SetInformationJobObject failed")

    def _assign_process_to_job(pid: int, job_handle: int) -> None:
        process_handle = _kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not process_handle:
            raise ProcessJobError(f"OpenProcess failed for pid {pid}")
        try:
            if not _kernel32.AssignProcessToJobObject(job_handle, process_handle):
                raise ProcessJobError(f"AssignProcessToJobObject failed for pid {pid}")
        finally:
            _kernel32.CloseHandle(process_handle)

    def _resume_main_thread(pid: int) -> None:
        thread_id = _first_thread_id(pid)
        if thread_id is None:
            raise ProcessJobError(f"no thread found for pid {pid}")
        thread_handle = _kernel32.OpenThread(_THREAD_SUSPEND_RESUME, False, thread_id)
        if not thread_handle:
            raise ProcessJobError(f"OpenThread failed for tid {thread_id}")
        try:
            if _kernel32.ResumeThread(thread_handle) == _RESUME_THREAD_ERROR:
                raise ProcessJobError(f"ResumeThread failed for tid {thread_id}")
        finally:
            _kernel32.CloseHandle(thread_handle)

    def _first_thread_id(pid: int) -> int | None:
        snapshot = _kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
        if not snapshot:
            raise ProcessJobError("CreateToolhelp32Snapshot failed")
        entry = _THREADENTRY32()
        entry.dwSize = ctypes.sizeof(_THREADENTRY32)
        try:
            if not _kernel32.Thread32First(snapshot, ctypes.byref(entry)):
                return None
            while entry.th32OwnerProcessID != pid:
                if not _kernel32.Thread32Next(snapshot, ctypes.byref(entry)):
                    return None
            return int(entry.th32ThreadID)
        finally:
            _kernel32.CloseHandle(snapshot)


class ManagedProcess:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        tree_terminator: TreeTerminator | None = None,
    ) -> None:
        self._process = process
        self._tree_terminator = tree_terminator

    @property
    def process(self) -> asyncio.subprocess.Process:
        return self._process

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    async def read_stdout_line(self) -> str:
        return await self._read_line(self._process.stdout)

    async def read_stderr_line(self) -> str:
        return await self._read_line(self._process.stderr)

    async def write_stdin(self, data: bytes) -> None:
        stdin = self._process.stdin
        if stdin is None:
            raise ProcessIOError("stdin is not available")
        stdin.write(data)
        try:
            await stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ProcessIOError("stdin write failed") from exc

    async def wait(self) -> int:
        code = await self._process.wait()
        return int(code)

    async def stop(self, grace: float = 5.0) -> int:
        if self._process.returncode is not None:
            if self._tree_terminator is not None:
                self._tree_terminator.kill()
            self._release_tree_terminator()
            return int(self._process.returncode)
        await self._close_stdin()
        code = await self._await_exit(grace)
        if code is None:
            self._terminate_tree()
            code = await self._await_exit(grace)
        if code is None:
            self._kill_tree()
            code = await self._await_exit(grace)
        if code is None:
            raise ProcessTerminationError("process survived the termination ladder")
        self._release_tree_terminator()
        return code

    async def interrupt(self) -> bool:
        if self.returncode is not None:
            return False
        if os.name == "nt":
            await self.stop(grace=0.5)
        else:
            killpg = getattr(os, "killpg", None)
            if killpg is None:
                return False
            try:
                killpg(self._process.pid, signal.SIGINT)
            except ProcessLookupError:
                return False
        return True

    async def kill(self) -> int:
        self._kill_tree()
        code = await self._await_exit(5.0)
        if code is None:
            raise ProcessTerminationError("process survived forced termination")
        self._release_tree_terminator()
        return code

    async def _await_exit(self, grace: float) -> int | None:
        try:
            code = await asyncio.wait_for(self._process.wait(), timeout=grace)
        except TimeoutError:
            return None
        return int(code)

    def _terminate_tree(self) -> None:
        if self._tree_terminator is not None:
            self._tree_terminator.terminate()
            return
        self._signal(self._process.terminate)

    def kill_now(self) -> None:
        self._kill_tree()

    def _kill_tree(self) -> None:
        if self._tree_terminator is not None:
            self._tree_terminator.kill()
            return
        self._signal(self._process.kill)

    def _release_tree_terminator(self) -> None:
        if self._tree_terminator is not None:
            self._tree_terminator.release()
            self._tree_terminator = None

    async def _close_stdin(self) -> None:
        stdin = self._process.stdin
        if stdin is None:
            return
        try:
            stdin.close()
            await stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _signal(self, signal: Callable[[], None]) -> None:
        with contextlib.suppress(ProcessLookupError):
            signal()

    async def _read_line(self, stream: asyncio.StreamReader | None) -> str:
        if stream is None:
            raise ProcessIOError("stream is not available")
        try:
            raw = await stream.readuntil(b"\n")
        except asyncio.IncompleteReadError as exc:
            return _decode(exc.partial)
        except asyncio.LimitOverrunError as exc:
            await stream.read(exc.consumed)
            raise ProcessIOError("line exceeds the configured limit") from exc
        except OverflowError as exc:
            raise ProcessIOError("line exceeds the configured limit") from exc
        return _decode(raw)


def _decode(raw: bytes) -> str:
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    return raw.decode("utf-8", errors="replace")


async def spawn(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    line_limit: int = DEFAULT_LINE_LIMIT,
) -> ManagedProcess:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=line_limit,
            **platform_spawn_options(),
        )
    except (OSError, ValueError) as exc:
        raise ProcessSpawnError(f"spawn failed for {argv[0]!r}") from exc
    factory = tree_terminator_factory()
    try:
        tree_terminator = factory(int(process.pid))
    except (ProcessJobError, OSError) as exc:
        with contextlib.suppress(OSError):
            process.kill()
        await process.wait()
        raise ProcessSpawnError(f"process tree guard failed for {argv[0]!r}") from exc
    return ManagedProcess(process, tree_terminator=tree_terminator)
