"""Base class for commands that ensure a Mandri daemon is running."""

from pathlib import Path

from mandri.cli.command import Command
from mandri.cli.daemon_client import (
    SPAWN_READINESS_TIMEOUT_SECONDS,
    Address,
    ProbeStatus,
    await_ready,
    daemon_log_path,
    probe,
    resolve_address,
    spawn,
)
from mandri.cli.run_errors import ForeignServiceError, ReadinessTimeoutError


class DaemonCommand(Command):
    """Command base class owning the daemon probe, spawn, and readiness lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self._address: Address | None = None

    def _ensure_running(self, base_dir: Path) -> Address:
        address = resolve_address(base_dir)
        self._address = address
        status = probe(address)
        if status is ProbeStatus.RUNNING:
            return address
        if status is ProbeStatus.FOREIGN:
            raise ForeignServiceError(f"port {address.port} is occupied by a foreign service")
        base_dir.mkdir(parents=True, exist_ok=True)
        with self._console.status("Starting daemon..."):
            spawn(base_dir, address)
            self._await_ready(base_dir, address)
        self._console.log(f"daemon ready at {address.host}:{address.port}")
        return address

    def _await_ready(self, base_dir: Path, address: Address) -> None:
        try:
            await_ready(
                address,
                deadline_s=SPAWN_READINESS_TIMEOUT_SECONDS,
                log_path=daemon_log_path(base_dir),
            )
        except ReadinessTimeoutError:
            if probe(address) is ProbeStatus.RUNNING:
                return
            raise
