import codecs
import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from mandri.gateway.privacy_response_json import response_error, restore_arguments
from mandri.gateway.surrogate import StreamRestorer, SurrogateEngine
from mandri.gateway.surrogate.json_text import unique_object


@dataclass(slots=True)
class Frame:
    lines: list[str]
    payload: dict[str, Any] | None
    data: str | None
    waiting: int = 0

    def encode(self) -> bytes:
        data = (
            json.dumps(self.payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if self.payload is not None
            else self.data
        )
        lines = [line for line in self.lines if not line.startswith("data:") and line != "data"]
        if data is not None:
            lines.extend("data: " + part for part in data.split("\n"))
        return ("\n".join(lines) + "\n\n").encode("utf-8")


class Decoder:
    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self._parts: list[str] = []
        self._lines: list[str] = []
        self._pending_cr = False

    def feed(self, chunk: bytes, *, final: bool = False) -> list[Frame]:
        try:
            text = self._utf8.decode(chunk, final=final)
        except UnicodeError as error:
            raise response_error("Provider stream contains incomplete or invalid UTF-8") from error
        if self._pending_cr and text:
            text = text.removeprefix("\n")
            self._pending_cr = False
        result = []
        start = 0
        for match in re.finditer(r"\r\n|\r|\n", text):
            self._parts.append(text[start : match.start()])
            line = "".join(self._parts)
            self._parts.clear()
            if line:
                self._lines.append(line)
            else:
                result.append(self._frame())
            start = match.end()
            self._pending_cr = match.group() == "\r" and start == len(text)
        if start < len(text):
            self._parts.append(text[start:])
        if final and (self._parts or self._lines):
            raise response_error("Provider stream ended inside an SSE event")
        return result

    def _frame(self) -> Frame:
        data_lines = []
        for line in self._lines:
            name, colon, value = line.partition(":")
            if name == "data":
                data_lines.append(value.removeprefix(" ") if colon else "")
        data = "\n".join(data_lines) if data_lines else None
        payload = None
        if data is not None and data != "[DONE]":
            try:
                payload = json.loads(data, object_pairs_hook=unique_object)
            except (ValueError, RecursionError) as error:
                raise response_error("Provider SSE data is not valid JSON") from error
            if not isinstance(payload, dict):
                raise response_error("Provider SSE data must be a JSON object")
        frame = Frame(self._lines, payload, data)
        self._lines = []
        return frame


@dataclass(slots=True)
class Slot:
    frame: Frame
    body: dict[str, Any]
    key: str

    def append(self, text: str) -> None:
        self.body[self.key] += text


@dataclass(slots=True)
class TextLane:
    restorer: StreamRestorer
    pending: Slot | None = None
    raw: list[str] = field(default_factory=list)
    closed: bool = False


@dataclass(slots=True)
class ArgumentLane:
    pending: Slot
    raw: list[str] = field(default_factory=list)
    closed: bool = False


class Lanes:
    def __init__(self, engine: SurrogateEngine) -> None:
        self.engine = engine
        self.text: dict[tuple[str, ...], TextLane] = {}
        self.arguments: dict[tuple[str, ...], ArgumentLane] = {}
        self.queue: deque[Frame] = deque()

    def enqueue(self, frame: Frame) -> None:
        self.queue.append(frame)

    def ready(self) -> list[bytes]:
        result = []
        while self.queue and (not self.queue[0].waiting):
            frame = self.queue.popleft()
            result.append(frame.encode())
        return result

    def feed_text(self, identity: tuple[str, ...], slot: Slot) -> None:
        value = slot.body[slot.key]
        if not isinstance(value, str):
            raise response_error("Provider text delta must be a string")
        lane = self.text.get(identity)
        if lane is None:
            lane = TextLane(StreamRestorer(self.engine))
            self.text[identity] = lane
        if lane.closed:
            raise response_error("Provider emitted text after its lane completed")
        lane.raw.append(value)
        output = lane.restorer.feed(value)
        if lane.pending:
            lane.pending.append(output)
            lane.pending.frame.waiting -= 1
            lane.pending = None
            slot.body[slot.key] = ""
        else:
            slot.body[slot.key] = output
        if lane.restorer.pending:
            lane.pending = slot
            slot.frame.waiting += 1

    def feed_arguments(self, identity: tuple[str, ...], slot: Slot) -> None:
        value = slot.body[slot.key]
        if not isinstance(value, str):
            raise response_error("Provider tool argument delta must be a string")
        lane = self.arguments.get(identity)
        if lane is None:
            lane = ArgumentLane(slot)
            self.arguments[identity] = lane
            slot.frame.waiting += 1
        if lane.closed:
            raise response_error("Provider emitted arguments after the tool call completed")
        lane.raw.append(value)
        slot.body[slot.key] = ""

    def close(self, prefix: tuple[str, ...] = ()) -> None:
        for identity, lane in self.text.items():
            if identity[: len(prefix)] != prefix or lane.closed:
                continue
            tail = lane.restorer.finish()
            if tail:
                if lane.pending is None:
                    raise response_error("A restored stream suffix lost its source event")
                lane.pending.append(tail)
            if lane.pending:
                lane.pending.frame.waiting -= 1
                lane.pending = None
            lane.closed = True
        for identity, argument in self.arguments.items():
            if identity[: len(prefix)] != prefix or argument.closed:
                continue
            argument.pending.append(restore_arguments("".join(argument.raw), self.engine))
            argument.pending.frame.waiting -= 1
            argument.closed = True

    def raw_text(self, identity: tuple[str, ...]) -> str | None:
        lane = self.text.get(identity)
        return "".join(lane.raw) if lane else None
