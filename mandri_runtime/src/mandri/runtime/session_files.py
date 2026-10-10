import os
import stat
import sys
from pathlib import Path
from typing import BinaryIO

from mandri.core.types.execution import ExecutionBackend
from mandri.core.types.sessions import Session
from mandri.runtime.attachments import MAX_FILE_BYTES, AttachmentError, AttachmentStore
from mandri.runtime.file_context import SessionFileContext


def open_session_file(
    session: Session, store: AttachmentStore, reference: str
) -> tuple[BinaryIO, str, int]:
    context = SessionFileContext.from_session(session, store.directory(str(session.id)))
    candidate = context.resolve(reference)
    roots = [context.workspace, context.attachments]
    try:
        resolved = candidate.resolve(strict=True)
        if session.execution_backend is ExecutionBackend.DOCKER:
            root = next((root for root in roots if resolved.is_relative_to(root)), None)
            if root is None:
                raise AttachmentError("File is outside this session's directories")
            fd = _open_beneath(root, resolved.relative_to(root))
        else:
            selected = Path(os.path.abspath(candidate))
            if not any(
                selected.is_relative_to(root) or resolved.is_relative_to(root) for root in roots
            ):
                raise AttachmentError("File is outside this session's directories")
            fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        stream = os.fdopen(fd, "rb")
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_FILE_BYTES:
            stream.close()
            raise AttachmentError("File is unavailable or exceeds 20 MiB")
        return stream, resolved.name, metadata.st_size
    except OSError as error:
        raise AttachmentError("File is unavailable") from error


def _open_beneath(root: Path, relative: Path) -> int:
    if sys.platform == "win32":
        return os.open(root / relative, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    else:
        if os.open not in os.supports_dir_fd:
            return os.open(root / relative, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent = os.open(root, flags)
        try:
            for part in relative.parts[:-1]:
                child = os.open(part, flags, dir_fd=parent)
                os.close(parent)
                parent = child
            return os.open(
                relative.name or ".",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
        finally:
            os.close(parent)
