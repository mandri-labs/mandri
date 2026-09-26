"""Bounded reverse scanning of JSONL record offsets."""

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

CHUNK_SIZE = 65536


def file_identity(path: Path, stat: os.stat_result) -> str:
    return hashlib.sha256(f"{path.resolve()}:{stat.st_ino}".encode()).hexdigest()


@dataclass(frozen=True)
class RecordSpan:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


class ReverseRecords:
    def __init__(
        self, handle: BinaryIO, offset: int, budget: int, record_end: int | None = None
    ) -> None:
        self.handle = handle
        self.offset = offset
        self.record_end = record_end
        self._floor = max(0, offset - budget)

    def __iter__(self) -> Iterator[RecordSpan]:
        while self.offset > self._floor:
            start = max(self._floor, self.offset - CHUNK_SIZE)
            self.handle.seek(start)
            block = self.handle.read(self.offset - start)
            position = len(block)
            while (position := block.rfind(b"\n", 0, position)) >= 0:
                boundary = start + position + 1
                end = self.record_end
                self.offset = boundary
                self.record_end = None
                if end is not None:
                    yield RecordSpan(boundary, end)
                self.record_end = boundary
            self.offset = start
        if self.offset == 0 and self.record_end is not None:
            end = self.record_end
            self.record_end = None
            yield RecordSpan(0, end)
