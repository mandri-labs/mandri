"""Typed HTTP client for daemon-mediated CLI commands (sessions start/stop, run)."""

import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from mandri.cli.run_errors import DaemonUnreachableError, ReadinessTimeoutError
from mandri.config.toml_adapter import DEFAULT_BASE_DIR, TomlConfigAdapter

DAEMON_TITLE = "Mandri Daemon"
PROBE_TIMEOUT_SECONDS = 1.0
SPAWN_READINESS_TIMEOUT_SECONDS = 10.0
SPAWN_READINESS_POLL_SECONDS = 0.15
REQUEST_TIMEOUT_SECONDS = 10.0

__all__ = [
    "Address",
    "ProbeStatus",
    "await_ready",
    "base_url",
    "daemon_log_path",
    "error_message",
    "probe",
    "request",
    "resolve_address",
    "resolve_base_url",
    "spawn",
]


@dataclass(frozen=True)
class Address:
    host: str
    port: int


class ProbeStatus(Enum):
    RUNNING = "running"
    FOREIGN = "foreign"
    UNREACHABLE = "unreachable"


def resolve_address(base_dir: Path | None = None) -> Address:
    directory = base_dir if base_dir is not None else DEFAULT_BASE_DIR
    config = TomlConfigAdapter(directory).load()
    return Address(host=config.server.host, port=config.server.port)


def resolve_base_url(base_dir: Path | None = None) -> str:
    return base_url(resolve_address(base_dir))


def base_url(address: Address) -> str:
    return f"http://{address.host}:{address.port}"


def probe(address: Address) -> ProbeStatus:
    url = f"{base_url(address)}/v1/openapi.json"
    try:
        response = httpx.get(url, timeout=PROBE_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return ProbeStatus.UNREACHABLE
    if response.status_code != 200:
        return ProbeStatus.UNREACHABLE
    if response.json().get("info", {}).get("title") == DAEMON_TITLE:
        return ProbeStatus.RUNNING
    return ProbeStatus.FOREIGN


def request(
    method: str,
    url: str,
    body: object | None = None,
    params: dict[str, str] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> httpx.Response:
    try:
        return httpx.request(method, url, timeout=timeout, json=body, params=params)
    except httpx.HTTPError as error:
        raise DaemonUnreachableError(f"daemon request failed: {error}") from error


def error_message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except (ValueError, KeyError):
        return response.text


def daemon_log_path(base_dir: Path) -> Path:
    return base_dir / "daemon.log"


def spawn(base_dir: Path, address: Address) -> None:
    with open(daemon_log_path(base_dir), "ab") as log_file:
        subprocess.Popen(
            _spawn_argv(base_dir, address),
            stdout=log_file,
            stderr=log_file,
            **_spawn_platform_kwargs(),
        )


def _spawn_argv(base_dir: Path, address: Address) -> list[str]:
    return [
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
    ]


def _spawn_platform_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return kwargs


def await_ready(address: Address, deadline_s: float = 10.0, log_path: Path | None = None) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if probe(address) is ProbeStatus.RUNNING:
            return
        time.sleep(SPAWN_READINESS_POLL_SECONDS)
    suffix = f"; see {log_path}" if log_path is not None else ""
    raise ReadinessTimeoutError(
        f"daemon did not become ready at {address.host}:{address.port}"
        f" within {deadline_s}s{suffix}; run mandri-daemon start"
    )
