import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote

from mandri.core.types.prompt import PromptAttachment, UserPrompt
from mandri.core.types.sessions import Session
from mandri.runtime.file_context import SessionFileContext
from mandri.runtime.image_input import inspect_image

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_PROMPT_IMAGE_BYTES = 4 * 1024 * 1024
MAX_ATTACHMENTS = 4
MAX_SESSION_BYTES = 128 * 1024 * 1024
logger = logging.getLogger(__name__)


class AttachmentError(ValueError):
    pass


class AttachmentStorageError(AttachmentError):
    pass


def _same_content(source: Path, destination: Path) -> bool:
    try:
        if source.stat().st_size != destination.stat().st_size:
            return False
        with source.open("rb") as original, destination.open("rb") as existing:
            while chunk := original.read(65536):
                if chunk != existing.read(len(chunk)):
                    return False
            return existing.read(1) == b""
    except FileNotFoundError:
        return False


@dataclasses.dataclass(frozen=True)
class Attachment:
    id: str
    name: str
    media_type: str
    size: int


def check_support(session: Session) -> None:
    if session.harness.value == "agy":
        raise AttachmentError("Attachments are not supported by this harness")


def safe_name(name: str) -> str:
    name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = "".join(character for character in name if character.isprintable())
    if not name or name in {".", ".."} or len(name.encode("utf-8")) > 200:
        raise AttachmentError("Invalid file name")
    return name


def media_type(path: Path, name: str) -> str:
    try:
        detected = inspect_image(path)
    except ValueError as error:
        raise AttachmentError(str(error)) from error
    if detected:
        return detected
    if Path(name).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".svg",
        ".tiff",
        ".heic",
    }:
        raise AttachmentError("Only PNG and JPEG images are supported")
    return "application/octet-stream"


class AttachmentStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self._locks: dict[str, asyncio.Lock] = {}

    def directory(self, session_id: str) -> Path:
        return self.root / hashlib.sha256(session_id.encode()).hexdigest()

    async def upload(self, session: Session, name: str, chunks: AsyncIterator[bytes]) -> Attachment:
        check_support(session)
        name = safe_name(name)
        try:
            return await self._upload(session, name, chunks)
        except OSError as error:
            logger.warning(
                "Attachment upload storage unavailable session=%s errno=%s winerror=%s",
                session.id,
                error.errno,
                getattr(error, "winerror", None),
            )
            raise AttachmentStorageError("Session file storage is unavailable") from error

    async def _upload(
        self, session: Session, name: str, chunks: AsyncIterator[bytes]
    ) -> Attachment:
        async with self._locks.setdefault(str(session.id), asyncio.Lock()):
            directory = self.directory(str(session.id))
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            used = sum(path.stat().st_size for path in directory.glob("*/content"))
            if used >= MAX_SESSION_BYTES:
                raise AttachmentError("Session attachment storage is full")
            identity = uuid.uuid4().hex
            pending = directory / f".{identity}"
            pending.mkdir(mode=0o700)
            size = 0
            try:
                with (pending / "content").open("xb") as stream:
                    async for chunk in chunks:
                        size += len(chunk)
                        if size > MAX_FILE_BYTES or used + size > MAX_SESSION_BYTES:
                            raise AttachmentError("File exceeds the upload limit")
                        await asyncio.to_thread(stream.write, chunk)
                if size == 0:
                    raise AttachmentError("Empty files cannot be attached")
                mime = await asyncio.to_thread(media_type, pending / "content", name)
                if mime.startswith("image/") and size > MAX_IMAGE_BYTES:
                    raise AttachmentError("Images must not exceed 2 MiB")
                attachment = Attachment(identity, name, mime, size)
                (pending / "metadata.json").write_text(json.dumps(dataclasses.asdict(attachment)))
                pending.rename(directory / identity)
                return attachment
            finally:
                if pending.exists():
                    shutil.rmtree(pending)

    def read(self, session_id: str, identity: str) -> tuple[Attachment, Path]:
        if not re.fullmatch(r"[a-f0-9]{32}", identity):
            raise AttachmentError("Invalid attachment identity")
        directory = self.directory(session_id) / identity
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
            attachment = Attachment(**metadata)
            path = directory / "content"
            if (
                directory.is_symlink()
                or path.is_symlink()
                or path.stat().st_size != attachment.size
            ):
                raise AttachmentError("Attachment is unavailable")
            return attachment, path
        except (OSError, ValueError, TypeError) as error:
            raise AttachmentError("Attachment is unavailable") from error

    def prepare(self, session: Session, text: str, identities: list[str]) -> UserPrompt:
        check_support(session)
        if len(identities) > MAX_ATTACHMENTS or len(set(identities)) != len(identities):
            raise AttachmentError("Attach at most four distinct files")
        selected = [self.read(str(session.id), identity) for identity in identities]
        image_bytes = sum(item.size for item, _ in selected if item.media_type.startswith("image/"))
        if image_bytes > MAX_PROMPT_IMAGE_BYTES:
            raise AttachmentError("Combined images must not exceed 4 MiB")
        parts = []
        links = []
        for item, source in selected:
            try:
                path = self.materialize(session, item, source)
                data = source.read_bytes() if item.media_type.startswith("image/") else b""
            except OSError as error:
                logger.warning(
                    "Attachment materialization unavailable "
                    "session=%s attachment=%s errno=%s winerror=%s",
                    session.id,
                    item.id,
                    error.errno,
                    getattr(error, "winerror", None),
                )
                raise AttachmentStorageError("Session file storage is unavailable") from error
            label = item.name.replace("[", "\\[").replace("]", "\\]")
            links.append(f"[{label}]({quote(Path(path).as_posix(), safe='/@:')})")
            parts.append(PromptAttachment(path, item.media_type, data))
        return UserPrompt(
            "\n\n".join(part for part in [text, "\n".join(links)] if part), tuple(parts)
        )

    def materialize(self, session: Session, item: Attachment, source: Path) -> str:
        context = SessionFileContext.from_session(session, self.directory(str(session.id)))
        destination, runtime_path = context.attachment_path(item.id, safe_name(item.name))
        if any(
            path.is_symlink()
            for path in (destination, *destination.parents)
            if path.is_relative_to(context.attachments)
        ):
            raise AttachmentError("Session file storage contains a symbolic link")
        if _same_content(source, destination):
            return runtime_path
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        temporary = destination.with_name(f".{uuid.uuid4().hex}")
        try:
            shutil.copyfile(source, temporary)
            if sys.platform != "win32":
                temporary.chmod(0o444)
            if sys.platform == "win32" and destination.exists():
                destination.chmod(0o644)
            os.replace(temporary, destination)
        finally:
            if sys.platform == "win32" and destination.exists():
                destination.chmod(0o444)
            if sys.platform == "win32" and temporary.exists():
                temporary.chmod(0o644)
            temporary.unlink(missing_ok=True)
        return runtime_path
