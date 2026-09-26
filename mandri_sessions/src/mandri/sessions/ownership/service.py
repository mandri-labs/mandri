import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import psutil
from mandri.core.ids import HarnessKind
from mandri.core.types.availability import SessionOwner
from mandri.sessions.errors import SessionConflictError
from mandri.sessions.ownership.file_lock import writer_locked
from mandri.sessions.ownership.processes import ProcessIdentity, codex_writer, session_processes


@dataclass(frozen=True)
class NativeOwnership:
    owner: SessionOwner
    process: ProcessIdentity | None = None
    reason: str | None = None

    @property
    def can_release(self) -> bool:
        return sys.platform != "win32" and self.process is not None and self.process.dedicated


def inspect_owner(
    harness: HarnessKind,
    native_id: str,
    project_path: str,
    lock_path: Path | None = None,
    pi_processes: Mapping[int, str | None] | None = None,
    ignored_pids: frozenset[int] = frozenset(),
) -> NativeOwnership:
    if harness is HarnessKind.CODEX:
        if lock_path is None:
            return NativeOwnership(SessionOwner.UNKNOWN, reason="writer_store_unavailable")
        locked = writer_locked(lock_path)
        if locked is None:
            return NativeOwnership(SessionOwner.UNKNOWN, reason="writer_status_unavailable")
        if not locked:
            return NativeOwnership(SessionOwner.UNOWNED)
        process = codex_writer(lock_path, native_id)
        reason = (
            None if process is not None and process.dedicated else "external_release_unsupported"
        )
        return NativeOwnership(SessionOwner.EXTERNAL, process, reason)
    try:
        processes, uncertain = session_processes(
            harness, native_id, project_path, pi_processes, ignored_pids
        )
    except (OSError, psutil.Error):
        return NativeOwnership(SessionOwner.UNKNOWN, reason="writer_status_unavailable")
    if len(processes) == 1 and not uncertain:
        reason = None if processes[0].dedicated else "external_release_unsupported"
        return NativeOwnership(SessionOwner.EXTERNAL, processes[0], reason)
    if processes:
        return NativeOwnership(SessionOwner.EXTERNAL, reason="external_release_unsupported")
    if uncertain:
        return NativeOwnership(SessionOwner.UNKNOWN, reason="writer_status_unavailable")
    return NativeOwnership(SessionOwner.UNOWNED)


def release_writer(
    harness: HarnessKind,
    native_id: str,
    project_path: str,
    lock_path: Path | None,
    expected: NativeOwnership,
) -> None:
    current = inspect_owner(harness, native_id, project_path, lock_path)
    identity = expected.process
    if identity is None or not expected.can_release or current != expected:
        raise SessionConflictError("Native writer changed or cannot be released safely")
    try:
        process = psutil.Process(identity.pid)
        if process.create_time() != identity.created_at:
            raise SessionConflictError("Native writer changed")
        process.terminate()
        try:
            process.wait(timeout=5)
        except psutil.TimeoutExpired:
            if inspect_owner(harness, native_id, project_path, lock_path) != expected:
                raise SessionConflictError("Native writer changed before forced release") from None
            if process.create_time() != identity.created_at:
                raise SessionConflictError("Native writer changed before forced release") from None
            process.kill()
            process.wait(timeout=5)
    except psutil.NoSuchProcess:
        return
    except psutil.Error as error:
        raise SessionConflictError("Native writer could not be released") from error
