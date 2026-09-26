"""Live/stopped registry for daemon-started sessions."""

import dataclasses

from mandri.runtime.process import ManagedProcess

LIVE = "live"
STOPPED = "stopped"


@dataclasses.dataclass(frozen=True)
class RegistryEntry:
    status: str
    process: ManagedProcess | None
    harness: str | None = None
    stopped_signal: bool = False


class SessionRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def mark_live(
        self, session_id: str, process: ManagedProcess | None = None, harness: str | None = None
    ) -> None:
        self._entries[session_id] = RegistryEntry(LIVE, process, harness)

    def claim_stopped_signal(self, session_id: str) -> bool:
        entry = self._entries.get(session_id)
        if entry is None:
            self._entries[session_id] = RegistryEntry(STOPPED, None, stopped_signal=True)
            return True
        if entry.stopped_signal:
            return False
        self._entries[session_id] = dataclasses.replace(entry, stopped_signal=True)
        return True

    def mark_stopped(self, session_id: str) -> None:
        entry = self._entries.get(session_id)
        if entry is None:
            self._entries[session_id] = RegistryEntry(STOPPED, None)
        else:
            self._entries[session_id] = dataclasses.replace(entry, status=STOPPED, process=None)

    def status(self, session_id: str) -> str | None:
        entry = self._entries.get(session_id)
        return None if entry is None else entry.status

    def process(self, session_id: str) -> ManagedProcess | None:
        entry = self._entries.get(session_id)
        return None if entry is None else entry.process

    def processes(self) -> dict[str, ManagedProcess]:
        return {
            session_id: entry.process
            for session_id, entry in self._entries.items()
            if entry.process is not None
        }

    def harness_of(self, session_id: str) -> str | None:
        entry = self._entries.get(session_id)
        return None if entry is None else entry.harness

    def live_ids(self) -> list[str]:
        return [
            session_id
            for session_id, entry in self._entries.items()
            if entry.status == LIVE
            and entry.process is not None
            and entry.process.returncode is None
        ]

    async def reconcile(self, excluded: frozenset[str] = frozenset()) -> list[str]:
        stopped: list[str] = []
        for session_id, entry in self._entries.items():
            if session_id in excluded or entry.status != LIVE:
                continue
            if entry.process is None or entry.process.returncode is not None:
                if entry.process is not None:
                    await entry.process.stop()
                self._entries[session_id] = dataclasses.replace(entry, status=STOPPED, process=None)
                stopped.append(session_id)
        return stopped
