"""Streaming access to a referenced native transcript record."""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from mandri.core.ids import PageToken
from mandri.sessions.transcripts.errors import PageTokenStaleError, TranscriptStoreError
from mandri.sessions.transcripts.records import CHUNK_SIZE, file_identity
from mandri.sessions.transcripts.tokens import decode_page_token


@dataclass
class RecordDownload:
    handle: BinaryIO
    size: int

    def chunks(self) -> Iterator[bytes]:
        remaining = self.size
        try:
            while remaining:
                chunk = self.handle.read(min(remaining, CHUNK_SIZE))
                if not chunk:
                    raise TranscriptStoreError("transcript truncated during download")
                remaining -= len(chunk)
                yield chunk
        finally:
            self.handle.close()

    def close(self) -> None:
        self.handle.close()


def open_record(path: Path, reference: PageToken) -> RecordDownload:
    token = decode_page_token(reference, "jsonl-record")
    start, end = token.offset or 0, token.file_size or 0
    try:
        handle = path.open("rb")
        try:
            stat = os.fstat(handle.fileno())
            if token.file_id != file_identity(path, stat) or not start < end <= stat.st_size:
                raise PageTokenStaleError("transcript record replaced or truncated")
            if start:
                handle.seek(start - 1)
                if handle.read(1) != b"\n":
                    raise PageTokenStaleError("record start is no longer valid")
            handle.seek(end - 1)
            if handle.read(1) != b"\n":
                raise PageTokenStaleError("record end is no longer valid")
            handle.seek(start)
            return RecordDownload(handle, end - start)
        except BaseException:
            handle.close()
            raise
    except OSError as error:
        raise TranscriptStoreError(f"cannot open transcript record: {error}") from error
