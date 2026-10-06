"""Probe, pid-file, liveness, spawn, and readiness helpers for the Mandri daemon."""

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, TypeIs

import httpx
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.ids import EpochMs

__all__ = [
    "DAEMON_TITLE",
    "Address",
    "DaemonPidRecord",
    "ForeignServiceError",
    "ProbeStatus",
    "ReadinessTimeoutError",
    "StopError",
    "await_ready",
    "httpx",
    "is_running",
    "pid_alive",
    "pid_file_is_stale",
    "probe",
    "read_pid_file",
    "remove_pid_file",
    "resolve_address",
    "spawn_daemon",
    "terminate_pid",
    "write_pid_file",
]

DAEMON_TITLE = "Mandri Daemon"
PROBE_TIMEOUT_SECONDS = 1.0
PID_FILE_NAME = "daemon.pid"
DAEMON_LOG_NAME = "daemon.log"
SPAWN_READINESS_TIMEOUT_SECONDS = 10.0
SPAWN_READINESS_POLL_SECONDS = 0.15


class ReadinessTimeoutError(Exception):
    """Raised when the daemon does not become ready in time."""


class ForeignServiceError(Exception):
    """Raised when the target service is not managed by this daemon."""


class StopError(Exception):
    """Raised when the daemon process cannot be terminated."""


@dataclass(frozen=True)
class Address:
    host: str
    port: int


@dataclass(frozen=True)
class DaemonPidRecord:
    pid: int
    started_at: EpochMs


class ProbeStatus(Enum):
    RUNNING = "running"
    FOREIGN = "foreign"
    UNREACHABLE = "unreachable"


def resolve_address(base_dir: Path) -> Address:
    config = TomlConfigAdapter(base_dir).load()
    return Address(host=config.server.host, port=config.server.port)


def probe(address: Address) -> ProbeStatus:
    url = f"http://{address.host}:{address.port}/v1/openapi.json"
    try:
        response = httpx.get(url, timeout=PROBE_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return ProbeStatus.UNREACHABLE
    if response.status_code != 200:
        return ProbeStatus.UNREACHABLE
    if response.json().get("info", {}).get("title") == DAEMON_TITLE:
        return ProbeStatus.RUNNING
    return ProbeStatus.FOREIGN


def is_running(base_dir: Path) -> bool:
    adapter = TomlConfigAdapter(base_dir)
    db_path = base_dir / "mandri.db"
    if not adapter.config_path.is_file() and not db_path.is_file():
        return False
    return probe(resolve_address(base_dir)) is ProbeStatus.RUNNING


def _pid_file_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / PID_FILE_NAME


def write_pid_file(base_dir: Path | str) -> None:
    record = DaemonPidRecord(
        pid=os.getpid(),
        started_at=EpochMs(int(time.time() * 1000)),
    )
    payload = json.dumps({"pid": record.pid, "started_at": record.started_at})
    directory = Path(base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _pid_file_path(directory).write_text(payload, encoding="utf-8")


def read_pid_file(base_dir: Path | str) -> DaemonPidRecord | None:
    try:
        raw = _pid_file_path(base_dir).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    pid_value = payload.get("pid")
    started_at_value = payload.get("started_at")
    if not _is_int(pid_value) or not _is_int(started_at_value):
        return None
    pid = int(pid_value)
    started_at = int(started_at_value)
    return DaemonPidRecord(pid=pid, started_at=EpochMs(started_at))


def _is_int(value: object) -> TypeIs[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def remove_pid_file(base_dir: Path | str) -> None:
    try:
        _pid_file_path(base_dir).unlink()
    except FileNotFoundError:
        return


def pid_file_is_stale(base_dir: Path | str, probe: Callable[[], bool]) -> bool:
    record = read_pid_file(base_dir)
    if record is None:
        return False
    if not pid_alive(record.pid):
        return True
    try:
        return not probe()
    except OSError:
        return True


def _log_file_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / DAEMON_LOG_NAME


def _spawn_argv(
    base_dir: Path,
    address: Address,
    log_level: str = "info",
    enable_litellm_debug: bool = False,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "mandri.daemon",
        "serve",
        "--base-dir",
        str(base_dir),
        "--host",
        address.host,
        "--port",
        str(address.port),
        "--log-level",
        log_level,
    ]
    if enable_litellm_debug:
        argv.append("--enable-litellm-debug")
    return argv


def _spawn_platform_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return kwargs


def spawn_daemon(
    base_dir: Path,
    address: Address,
    log_level: str = "info",
    enable_litellm_debug: bool = False,
) -> None:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    with open(_log_file_path(base_dir), "ab") as log_file:
        kwargs = _spawn_platform_kwargs()
        kwargs["stdout"] = log_file
        kwargs["stderr"] = log_file
        subprocess.Popen(_spawn_argv(base_dir, address, log_level, enable_litellm_debug), **kwargs)


def await_ready(address: Address, deadline_s: float = 10.0, log_path: Path | None = None) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if probe(address) is ProbeStatus.RUNNING:
            return
        time.sleep(SPAWN_READINESS_POLL_SECONDS)
    suffix = f"; see {log_path}" if log_path is not None else ""
    raise ReadinessTimeoutError(
        f"daemon did not become ready at {address.host}:{address.port} within {deadline_s}s{suffix}"
    )


def _probe_is_running(address: Address) -> bool:
    return probe(address) is ProbeStatus.RUNNING


def ensure_running(
    base_dir: Path,
    log_level: str = "info",
    enable_litellm_debug: bool = False,
) -> Address:
    address = resolve_address(base_dir)
    status = probe(address)
    if status is ProbeStatus.RUNNING:
        return address
    if status is ProbeStatus.FOREIGN:
        raise ForeignServiceError(f"port {address.port} is occupied by a foreign service")
    if pid_file_is_stale(base_dir, partial(_probe_is_running, address)):
        remove_pid_file(base_dir)
    spawn_daemon(base_dir, address, log_level, enable_litellm_debug)
    try:
        await_ready(
            address,
            deadline_s=SPAWN_READINESS_TIMEOUT_SECONDS,
            log_path=_log_file_path(base_dir),
        )
    except ReadinessTimeoutError:
        if _probe_is_running(address):
            return address
        raise
    return address


if sys.platform == "win32":
    _SYNCHRONIZE = 0x00100000
    _PROCESS_TERMINATE = 0x0001
    _WAIT_OBJECT_0 = 0x00000000
    _ERROR_ACCESS_DENIED = 0x00000005

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    _kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _kernel32.TerminateProcess.restype = ctypes.c_int32
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.c_int32

    def pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        handle = _kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            wait_result = int(_kernel32.WaitForSingleObject(handle, 0))
        finally:
            _kernel32.CloseHandle(handle)
        return wait_result != _WAIT_OBJECT_0

    def terminate_pid(pid: int) -> bool:
        if pid <= 0:
            return False
        handle = _kernel32.OpenProcess(_PROCESS_TERMINATE, False, pid)
        if not handle:
            if ctypes.get_last_error() == _ERROR_ACCESS_DENIED:
                raise StopError(f"access denied while terminating pid {pid}")
            return False
        try:
            if not int(_kernel32.TerminateProcess(handle, 1)):
                if ctypes.get_last_error() == _ERROR_ACCESS_DENIED:
                    raise StopError(f"access denied while terminating pid {pid}")
                return False
        finally:
            _kernel32.CloseHandle(handle)
        return True

else:

    def pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def terminate_pid(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except PermissionError as error:
            raise StopError(f"access denied while terminating pid {pid}") from error
        return True
