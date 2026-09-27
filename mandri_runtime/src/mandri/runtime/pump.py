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
        limit: int | None = DEFAULT_LINE_LIMIT,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._stream_factory = stream_factory
        self._limit = limit
        self._on_close = on_close

    def close(self) -> None:
        callback, self._on_close = self._on_close, None
        if callback is not None:
            callback()

    async def lines(self) -> AsyncIterator[LineEvent]:
        buffer = bytearray()
        oversized = False
        scan_start = 0
        async for chunk in self._stream_factory():
            buffer.extend(chunk)
            while True:
                index = buffer.find(b"\n", scan_start)
                if index < 0:
                    scan_start = len(buffer)
                    if not oversized and self._limit is not None and len(buffer) > self._limit:
                        yield LineEvent(LineEventKind.OVERSIZE, "", len(buffer))
                        oversized = True
                    break
                raw = bytes(buffer[:index])
                del buffer[: index + 1]
                scan_start = 0
                if oversized:
                    oversized = False
                    continue
                if self._limit is not None and len(raw) > self._limit:
                    yield LineEvent(LineEventKind.OVERSIZE, "", len(raw))
                    continue
                yield _line_event(raw, LineEventKind.LINE)
        if buffer:
            if self._limit is not None and len(buffer) > self._limit:
                yield LineEvent(LineEventKind.OVERSIZE, "", len(buffer))
            else:
                yield _line_event(bytes(buffer), LineEventKind.INCOMPLETE)


async def stream_reader_bytes(reader: asyncio.StreamReader | None) -> AsyncIterator[bytes]:
    if reader is None:
        raise ProcessIOError("stream is not available")
    while chunk := await reader.read(_READ_SIZE):
        yield chunk


def pump_process_streams(
    process: ProcessStreams,
    limit: int | None = DEFAULT_LINE_LIMIT,
    *,
    stderr_limit: int | None = None,
) -> tuple[LinePump, LinePump]:
    stdout = LinePump(partial(stream_reader_bytes, process.stdout), limit)
    stderr = LinePump(
        partial(stream_reader_bytes, process.stderr),
        stderr_limit if stderr_limit is not None else limit,
    )
    return stdout, stderr


def _line_event(raw: bytes, kind: LineEventKind) -> LineEvent:
    stripped = raw[:-1] if raw.endswith(b"\r") else raw
    try:
        text = stripped.decode("utf-8")
    except UnicodeDecodeError:
        replacement = stripped.decode("utf-8", errors="replace")
        return LineEvent(LineEventKind.DECODE_ERROR, replacement, len(raw))
    return LineEvent(kind, text, len(raw))
