"""Child process stream pumps producing typed line events from raw byte streams."""

import asyncio
import dataclasses
import enum
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Protocol, final

from mandri.runtime.errors import ProcessIOError
from mandri.runtime.process import DEFAULT_LINE_LIMIT

_READ_SIZE = 65536


class LineEventKind(enum.StrEnum):
    LINE = "line"
    INCOMPLETE = "incomplete"
    OVERSIZE = "oversize"
    DECODE_ERROR = "decode_error"


@final
@dataclasses.dataclass(frozen=True)
class LineEvent:
    kind: LineEventKind
    text: str
    size: int


class ProcessStreams(Protocol):
    @property
    def stdout(self) -> asyncio.StreamReader | None: ...

    @property
    def stderr(self) -> asyncio.StreamReader | None: ...


@final
class LinePump:
    def __init__(
        self,
        stream_factory: Callable[[], AsyncIterator[bytes]],
        limit: int = DEFAULT_LINE_LIMIT,
    ) -> None:
        self._stream_factory = stream_factory
        self._limit = limit

    async def lines(self) -> AsyncIterator[LineEvent]:
        buffer = bytearray()
        oversized = False
        async for chunk in self._stream_factory():
            buffer.extend(chunk)
            while True:
                index = buffer.find(b"\n")
                if index < 0:
                    if not oversized and len(buffer) > self._limit:
                        yield LineEvent(LineEventKind.OVERSIZE, "", len(buffer))
                        oversized = True
                    break
                raw = bytes(buffer[:index])
                del buffer[: index + 1]
                if oversized:
                    oversized = False
                    continue
                if len(raw) > self._limit:
                    yield LineEvent(LineEventKind.OVERSIZE, "", len(raw))
                    continue
                yield _line_event(raw, LineEventKind.LINE)
        if buffer:
            if len(buffer) > self._limit:
                yield LineEvent(LineEventKind.OVERSIZE, "", len(buffer))
            else:
                yield _line_event(bytes(buffer), LineEventKind.INCOMPLETE)


async def stream_reader_bytes(reader: asyncio.StreamReader | None) -> AsyncIterator[bytes]:
    if reader is None:
        raise ProcessIOError("stream is not available")
    while chunk := await reader.read(_READ_SIZE):
        yield chunk


def pump_process_streams(
    process: ProcessStreams, limit: int = DEFAULT_LINE_LIMIT
) -> tuple[LinePump, LinePump]:
    stdout = LinePump(partial(stream_reader_bytes, process.stdout), limit)
    stderr = LinePump(partial(stream_reader_bytes, process.stderr), limit)
    return stdout, stderr


def _line_event(raw: bytes, kind: LineEventKind) -> LineEvent:
    stripped = raw[:-1] if raw.endswith(b"\r") else raw
    try:
        text = stripped.decode("utf-8")
    except UnicodeDecodeError:
        replacement = stripped.decode("utf-8", errors="replace")
        return LineEvent(LineEventKind.DECODE_ERROR, replacement, len(raw))
    return LineEvent(kind, text, len(raw))
