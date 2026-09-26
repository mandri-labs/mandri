import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from mandri.core.codex_versions import CODEX_FORK_VERSIONS
from mandri.core.ids import HarnessKind, SessionState
from mandri.core.types.execution import ExecutionBackend, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.core.types.sessions import Session
from mandri.sessions.execution_context import DockerSessionContext, transcript_reference
from mandri.sessions.ownership.file_lock import try_lock, unlock
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader

_MAX_BYTES = 256 * 1024 * 1024
_MAX_LINE = 16 * 1024 * 1024


def _incompatible() -> ProtectionError:
    return ProtectionError(
        "session_transition_unsupported", "The selected native history cannot be forked"
    )


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _workspace_identity(session: Session, root: Path) -> tuple[int, int]:
    try:
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError
        if session.execution_backend is ExecutionBackend.DOCKER:
            context = json.loads(session.execution_context or "null")
            if (
                not isinstance(context, dict)
                or context.get("workspace_device") != str(metadata.st_dev)
                or context.get("workspace_inode") != str(metadata.st_ino)
            ):
                raise ValueError
        return metadata.st_dev, metadata.st_ino
    except (OSError, ValueError, TypeError):
        raise ProtectionError(
            "workspace_identity_changed", "The selected source workspace identity changed"
        ) from None


@dataclass(frozen=True)
class CodexForkSource:
    session: Session
    native_id: str
    rollout_path: Path
    workspace_root: Path
    _handle: BinaryIO = field(repr=False, compare=False)
    _signature: tuple[int, int, int, int] = field(repr=False)
    _digest: bytes = field(repr=False)
    _workspace_signature: tuple[int, int] = field(repr=False)

    def copy_rollout(self, destination: Path) -> None:
        self.verify()
        digest = hashlib.sha256()
        self._handle.seek(0)
        created = False
        try:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as output:
                created = True
                remaining = self._signature[2]
                while remaining:
                    chunk = self._handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise _incompatible()
                    digest.update(chunk)
                    output.write(chunk)
                    remaining -= len(chunk)
                if self._handle.read(1):
                    raise _incompatible()
                output.flush()
                os.fsync(output.fileno())
            if digest.digest() != self._digest:
                raise _incompatible()
            self.verify()
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise

    def verify(self) -> None:
        if self._handle.closed:
            raise _incompatible()
        if _workspace_identity(self.session, self.workspace_root) != self._workspace_signature:
            raise ProtectionError(
                "workspace_identity_changed", "The selected source workspace identity changed"
            )
        try:
            metadata = self.rollout_path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _identity(metadata) != self._signature
                or _identity(os.fstat(self._handle.fileno())) != self._signature
            ):
                raise _incompatible()
        except OSError:
            raise _incompatible() from None


def _inspect(handle: BinaryIO, session: Session) -> bytes:
    metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= _MAX_BYTES:
        raise _incompatible()
    digest = hashlib.sha256()
    first = True
    total = 0
    while line := handle.readline(_MAX_LINE + 1):
        total += len(line)
        if len(line) > _MAX_LINE or not line.endswith(b"\n") or total > _MAX_BYTES:
            raise _incompatible()
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError, RecursionError):
            raise _incompatible() from None
        if not isinstance(row, dict) or not isinstance(row.get("type"), str):
            raise _incompatible()
        if first:
            payload = row.get("payload")
            if (
                row["type"] != "session_meta"
                or not isinstance(payload, dict)
                or payload.get("id") != str(session.native_id)
                or payload.get("cli_version") not in CODEX_FORK_VERSIONS
                or payload.get("cwd") != str(transcript_reference(session).project_path)
            ):
                raise _incompatible()
            first = False
        elif row["type"] == "session_meta":
            raise _incompatible()
        digest.update(line)
    if total != metadata.st_size or _identity(os.fstat(handle.fileno())) != _identity(metadata):
        raise _incompatible()
    return digest.digest()


def _locations(session: Session, reader: CodexTranscriptReader) -> tuple[Path, Path, Path]:
    if session.execution_backend is ExecutionBackend.DOCKER:
        context = DockerSessionContext.from_session(session)
        _workspace_identity(session, context.workspace_root)
    reference = transcript_reference(session)
    path = Path(reader.revision(reference)[0])
    native_lock = reader.writer_lock_path(reference)
    if native_lock is None:
        raise _incompatible()
    if session.execution_backend is ExecutionBackend.DOCKER:
        context = DockerSessionContext.from_session(session)
        path = context.state_path(path)
        return path, context.state_root.parent / f".{session.id}.lock", context.workspace_root
    home = native_lock.parent.parent.resolve(strict=True)
    if native_lock.parent.is_symlink() or not native_lock.resolve().is_relative_to(home):
        raise _incompatible()
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not any(
        resolved.is_relative_to(home / directory) for directory in ("sessions", "archived_sessions")
    ):
        raise _incompatible()
    return resolved, native_lock, Path(session.project_path).resolve(strict=True)


@contextmanager
def selected_codex_source(
    session: Session, target: ExecutionBackend, reader: object
) -> Iterator[CodexForkSource]:
    if (
        session.harness is not HarnessKind.CODEX
        or session.native_id is None
        or session.state is not SessionState.STOPPED
        or session.deleted
        or session.model_source is not ModelSource.GATEWAY
        or session.execution_backend is not target
        or not isinstance(reader, CodexTranscriptReader)
    ):
        raise _incompatible()
    try:
        path, lock, workspace = _locations(session, reader)
        if lock.is_symlink():
            raise _incompatible()
        lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "r+b") as lease:
            if not try_lock(lease):
                raise ProtectionError("session_writer_conflict", "The native session has a writer")
            try:
                file_descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(file_descriptor, "rb") as handle:
                    signature = _identity(os.fstat(handle.fileno()))
                    digest = _inspect(handle, session)
                    source = CodexForkSource(
                        session,
                        str(session.native_id),
                        path,
                        workspace,
                        handle,
                        signature,
                        digest,
                        _workspace_identity(session, workspace),
                    )
                    source.verify()
                    yield source
                    source.verify()
            finally:
                unlock(lease)
    except OSError:
        raise _incompatible() from None
