import inspect
from collections.abc import AsyncIterator
from typing import Any

from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.privacy_response_json import (
    response_error,
    restore_annotations,
    restore_arguments,
    restore_item,
    restore_json,
    restore_part,
    text_value,
)
from mandri.gateway.privacy_response_sse import Decoder, Frame, Lanes, Slot
from mandri.gateway.surrogate import SurrogateEngine

__all__ = ["restore_json", "restore_sse"]


class _ResponseStream:
    def __init__(self, engine: SurrogateEngine, protocol: GatewayProtocol) -> None:
        self.engine = engine
        self.protocol = protocol
        self.lanes = Lanes(engine)
        self.terminal = False
        self.failed = False
        self.seen = False
        self.choices: set[str] = set()
        self.finished_choices: set[str] = set()
        self.blocks: set[str] = set()

    def accept(self, frame: Frame) -> list[bytes]:
        self.lanes.enqueue(frame)
        payload = frame.payload
        if frame.data == "[DONE]":
            if self.protocol != GatewayProtocol.CHAT:
                raise response_error("Unexpected provider terminal marker")
            if not self.seen or (not self.failed and self.choices != self.finished_choices):
                raise response_error("Provider stream ended before its choices completed")
            self.lanes.close()
            self.terminal = True
        elif payload is not None:
            self.seen = True
            if payload.get("error") is not None:
                frame.payload = restore_json(payload, self.engine)
                self.lanes.close()
                self.failed = True
                self.terminal = True
            elif self.protocol == GatewayProtocol.CHAT:
                self._chat(frame, payload)
            elif self.protocol == GatewayProtocol.RESPONSES:
                self._responses(frame, payload)
            elif self.protocol == GatewayProtocol.ANTHROPIC:
                self._anthropic(frame, payload)
            elif self.protocol == GatewayProtocol.GEMINI:
                self._gemini(frame, payload)
            else:
                self.terminal = True
        return self.lanes.ready()

    def finish(self) -> list[bytes]:
        if not self.seen or not self.terminal:
            raise response_error("Provider stream ended without a protocol completion event")
        if self.lanes.queue:
            raise response_error("Provider stream has unresolved content lanes")
        return self.lanes.ready()

    def _chat(self, frame: Frame, payload: dict[str, Any]) -> None:
        if self.terminal:
            raise response_error("Provider emitted data after stream completion")
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise response_error("Provider stream choices must be a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise response_error("Provider stream choice must be an object")
            index = str(choice.get("index", 0))
            self.choices.add(index)
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise response_error("Provider message delta must be an object")
            if index in self.finished_choices:
                if (
                    isinstance(payload.get("usage"), dict)
                    and choice.get("finish_reason") is None
                    and (choice.get("logprobs") is None)
                    and (
                        set(delta)
                        <= {
                            "content",
                            "role",
                            "function_call",
                            "tool_calls",
                            "audio",
                            "refusal",
                            "reasoning",
                            "reasoning_content",
                            "provider_specific_fields",
                        }
                    )
                    and all(value is None for value in delta.values())
                ):
                    continue
                raise response_error("Provider emitted an already completed choice")
            for key in ("content", "refusal", "reasoning", "reasoning_content"):
                value = delta.get(key)
                if isinstance(value, list) and key == "content":
                    for position, part in enumerate(value):
                        if not isinstance(part, dict) or part.get("type") not in (
                            "text",
                            "output_text",
                        ):
                            continue
                        self.lanes.feed_text(
                            ("chat", index, key, str(position)), Slot(frame, part, "text")
                        )
                elif value is not None:
                    self.lanes.feed_text(("chat", index, key), Slot(frame, delta, key))
            if delta.get("annotations") is not None:
                original = self.lanes.raw_text(("chat", index, "content"))
                delta["annotations"] = restore_annotations(
                    delta["annotations"], self.engine, original
                )
            calls = delta.get("tool_calls")
            if calls is None:
                calls = []
            if not isinstance(calls, list):
                raise response_error("Provider streamed tool calls must be a list")
            for position, call in enumerate(calls):
                if not isinstance(call, dict) or call.get("type") not in (None, "function"):
                    continue
                function = call.get("function")
                if function is not None:
                    self._chat_function(
                        frame, function, ("chat", index, "tool", str(call.get("index", position)))
                    )
            if delta.get("function_call") is not None:
                self._chat_function(frame, delta["function_call"], ("chat", index, "legacy-tool"))
            if choice.get("finish_reason") is not None:
                self.lanes.close(("chat", index))
                self.finished_choices.add(index)

    def _chat_function(self, frame: Frame, function: Any, prefix: tuple[str, ...]) -> None:
        if not isinstance(function, dict):
            raise response_error("Provider streamed function must be an object")
        if function.get("name") is not None:
            self.lanes.feed_text((*prefix, "name"), Slot(frame, function, "name"))
        if function.get("arguments") is not None:
            self.lanes.feed_arguments((*prefix, "arguments"), Slot(frame, function, "arguments"))

    def _response_key(self, payload: dict[str, Any], lane: str) -> tuple[str, ...]:
        item = str(payload.get("item_id", payload.get("output_index", 0)))
        part = str(payload.get("content_index", payload.get("summary_index", 0)))
        return ("responses", item, lane, part)

    def _responses(self, frame: Frame, payload: dict[str, Any]) -> None:
        if self.terminal:
            raise response_error("Provider emitted data after response completion")
        kind = payload.get("type", "")
        if not isinstance(kind, str):
            raise response_error("Provider response event type must be a string")
        text_kinds = {
            "response.output_text",
            "response.reasoning_summary_text",
            "response.reasoning_text",
            "response.refusal",
        }
        stem, _, suffix = kind.rpartition(".")
        if stem == "response.web_search_call" and suffix in {
            "in_progress",
            "searching",
            "completed",
        }:
            if set(payload) - {"type", "item_id", "output_index", "sequence_number"}:
                raise response_error("Provider search progress has unexpected content")
            return
        if stem in text_kinds and suffix in {"delta", "done"}:
            identity = self._response_key(payload, stem)
            if suffix == "delta":
                self.lanes.feed_text(identity, Slot(frame, payload, "delta"))
            else:
                self.lanes.close(identity)
                key = "refusal" if "refusal" in payload else "text"
                if key in payload:
                    payload[key] = text_value(payload[key], self.engine)
            return
        if kind in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            identity = self._response_key(payload, "arguments")
            if kind.endswith(".delta"):
                self.lanes.feed_arguments(identity, Slot(frame, payload, "delta"))
            else:
                self.lanes.close(identity)
                payload["arguments"] = restore_arguments(payload.get("arguments"), self.engine)
            return
        if kind in {"response.output_item.added", "response.output_item.done"}:
            item = payload.get("item")
            if not isinstance(item, dict):
                raise response_error("Provider output item event is incomplete")
            if kind.endswith(".done"):
                item_identity = str(item.get("id", payload.get("output_index", 0)))
                self.lanes.close(("responses", item_identity))
            restore_item(item, self.engine, partial=kind.endswith(".added"))
            return
        if kind in {
            "response.content_part.added",
            "response.content_part.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        }:
            restore_part(payload.get("part"), self.engine, partial=kind.endswith(".added"))
            return
        if kind == "response.output_text.annotation.added":
            original = self.lanes.raw_text(self._response_key(payload, "response.output_text"))
            payload["annotation"] = restore_annotations(
                [payload.get("annotation")], self.engine, original
            )[0]
            return
        if kind in {"response.completed", "response.failed", "response.incomplete"}:
            self.lanes.close()
            if "response" in payload:
                payload["response"] = restore_json(payload["response"], self.engine)
            self.terminal = True
            self.failed = kind != "response.completed"
            return
        if kind in {"response.created", "response.in_progress", "response.queued"}:
            if "response" in payload:
                payload["response"] = restore_json(payload["response"], self.engine)
            return
        if kind == "error":
            if "message" in payload:
                payload["message"] = text_value(payload["message"], self.engine)
            self.lanes.close()
            self.terminal = self.failed = True
            return
        return

    def _anthropic(self, frame: Frame, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind == "ping":
            return
        if self.terminal:
            raise response_error("Provider emitted data after message completion")
        index = str(payload.get("index", 0))
        prefix = ("anthropic", index)
        if kind == "message_start":
            payload["message"] = restore_json(payload.get("message"), self.engine)
        elif kind == "content_block_start":
            if index in self.blocks:
                raise response_error("Provider reused an open content block")
            self.blocks.add(index)
            restore_part(payload.get("content_block"), self.engine, partial=True)
        elif kind == "content_block_delta":
            if index not in self.blocks:
                raise response_error("Provider content delta has no open block")
            delta = payload.get("delta")
            if not isinstance(delta, dict):
                raise response_error("Provider content block delta must be an object")
            delta_kind = delta.get("type")
            if delta_kind in {"text_delta", "thinking_delta"}:
                key = "text" if delta_kind == "text_delta" else "thinking"
                self.lanes.feed_text((*prefix, key), Slot(frame, delta, key))
            elif delta_kind == "input_json_delta":
                self.lanes.feed_arguments(
                    (*prefix, "arguments"), Slot(frame, delta, "partial_json")
                )
            elif delta_kind == "citations_delta":
                original = self.lanes.raw_text((*prefix, "text"))
                delta["citation"] = restore_annotations(
                    [delta.get("citation")], self.engine, original
                )[0]
            else:
                return
        elif kind == "content_block_stop":
            if index not in self.blocks:
                raise response_error("Provider closed an unknown content block")
            self.lanes.close(prefix)
            self.blocks.remove(index)
        elif kind == "message_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict) and delta.get("stop_sequence") is not None:
                delta["stop_sequence"] = text_value(delta["stop_sequence"], self.engine)
        elif kind == "message_stop":
            if self.blocks:
                raise response_error("Provider message ended with open content blocks")
            self.lanes.close()
            self.terminal = True
        else:
            return

    def _gemini(self, frame: Frame, payload: dict[str, Any]) -> None:
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            raise response_error("Provider streamed candidates must be a list")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise response_error("Provider streamed candidate must be an object")
            index = str(candidate.get("index", 0))
            if index in self.finished_choices:
                raise response_error("Provider emitted an already completed candidate")
            self.choices.add(index)
            content = candidate.get("content", {})
            if not isinstance(content, dict) or not isinstance(content.get("parts", []), list):
                raise response_error("Provider streamed candidate content is invalid")
            for position, part in enumerate(content.get("parts", [])):
                if not isinstance(part, dict):
                    raise response_error("Provider streamed part must be an object")
                if "text" in part:
                    self.lanes.feed_text(
                        ("gemini", index, str(position), str(bool(part.get("thought")))),
                        Slot(frame, part, "text"),
                    )
                else:
                    restore_part(part, self.engine)
            if candidate.get("finishReason"):
                self.lanes.close(("gemini", index))
                self.finished_choices.add(index)
        self.terminal = bool(self.choices) and self.choices == self.finished_choices


async def restore_sse(
    stream: AsyncIterator[bytes], engine: SurrogateEngine, protocol: GatewayProtocol | str
) -> AsyncIterator[bytes]:
    try:
        selected = GatewayProtocol(protocol)
    except ValueError:
        async for chunk in stream:
            yield chunk
        return
    decoder = Decoder()
    response = _ResponseStream(engine, selected)
    try:
        async for chunk in stream:
            if not isinstance(chunk, bytes):
                raise response_error("Provider stream transport must yield bytes")
            for frame in decoder.feed(chunk):
                for encoded in response.accept(frame):
                    yield encoded
        for frame in decoder.feed(b"", final=True):
            for encoded in response.accept(frame):
                yield encoded
        for encoded in response.finish():
            yield encoded
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            closing = close()
            if inspect.isawaitable(closing):
                await closing
