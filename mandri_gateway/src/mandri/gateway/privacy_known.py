from typing import Any

from mandri.gateway.surrogate import SurrogateEngine
from mandri.gateway.surrogate.detectors import select_spans
from mandri.gateway.surrogate.encoded_paths import encoded_roots
from mandri.gateway.surrogate.json_text import rewrite_json
from mandri.gateway.surrogate.matching import LiteralIndex
from mandri.gateway.surrogate.paths import root_occurrences
from mandri.gateway.surrogate.types import Span

_MEDIA_TYPES = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "input_audio",
        "audio",
        "video",
        "file",
        "input_file",
        "document",
    }
)
_MEDIA_FIELDS = frozenset({"inlineData", "inline_data", "fileData", "file_data"})


class KnownValues:
    def __init__(self, engine: SurrogateEngine) -> None:
        self.scope = engine.scope
        self.mappings = {item.original: item for item in self.scope.mappings}
        self.forward = LiteralIndex(list(self.mappings.values()))
        self.reverse = LiteralIndex(self.scope.mappings, reverse=True)

    def text(self, value: str) -> str:
        nested = rewrite_json(value, self.replace)
        if nested is not None:
            return nested
        aliases = [(start, end) for start, end, _ in self.reverse.find(value)]
        spans = []
        replacements = {}
        for start, end, item in self.forward.find(value):
            if item.kind == "path_root":
                continue
            spans.append(Span(start, end, item.kind, 100))
            replacements[start, end] = item.surrogate
        for root in self.scope.roots:
            for start, end, _ in root_occurrences(value, root):
                spans.append(Span(start, end, "path_root", 200))
                replacements[start, end] = root.surrogate
        for encoded in encoded_roots(value, self.scope.roots):
            spans.append(Span(encoded.start, encoded.end, "path_root", 200))
            replacements[encoded.start, encoded.end] = encoded.surrogate
        selected = select_spans(
            [
                span
                for span in spans
                if not any(left <= span.start and span.end <= right for left, right in aliases)
            ]
        )
        pieces: list[str] = []
        cursor = 0
        for span in selected:
            pieces.extend((value[cursor : span.start], replacements[span.start, span.end]))
            cursor = span.end
        pieces.append(value[cursor:])
        return "".join(pieces)

    def replace(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if type(value) is int:
            mapping = self.mappings.get(str(value))
            if mapping is not None:
                return int(mapping.surrogate)
            return value
        if isinstance(value, list):
            return [self.replace(item) for item in value]
        if isinstance(value, dict):
            if isinstance(value.get("type"), str) and value["type"] in _MEDIA_TYPES:
                return value
            return {
                self.text(key): child if key in _MEDIA_FIELDS else self.replace(child)
                for key, child in value.items()
            }
        return value
