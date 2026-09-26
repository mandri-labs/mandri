import re
from collections import defaultdict

from mandri.gateway.surrogate.engine import SurrogateEngine, literal_occurrences
from mandri.gateway.surrogate.paths import root_occurrences
from mandri.gateway.surrogate.types import Mapping, PathRoot


class StreamRestorer:
    def __init__(self, engine: SurrogateEngine) -> None:
        self._index: dict[str, list[Mapping]] = defaultdict(list)
        for mapping in engine.scope.mappings:
            self._index[mapping.surrogate[0]].append(mapping)
        for mappings in self._index.values():
            mappings.sort(
                key=lambda item: (item.kind == "path_root", len(item.surrogate)), reverse=True
            )
        self._buffer = ""
        self._previous = ""
        self._finished = False
        self._max_alias = max((len(item.surrogate) for item in engine.scope.mappings), default=1)
        self._starts = (
            re.compile("[" + re.escape("".join(self._index)) + "]") if self._index else None
        )

    def feed(self, value: str) -> str:
        if self._finished:
            self._finished = False
        self._buffer += value
        return self._drain(False)

    @property
    def pending(self) -> bool:
        return bool(self._buffer)

    def finish(self) -> str:
        if self._finished:
            return ""
        self._finished = True
        return self._drain(True)

    def _drain(self, final: bool) -> str:
        result = []
        cursor = 0
        while cursor < len(self._buffer):
            start = self._starts.search(self._buffer, cursor) if self._starts else None
            if start is None or start.start() > cursor:
                end = start.start() if start else len(self._buffer)
                literal = self._buffer[cursor:end]
                result.append(literal)
                self._remember(literal)
                cursor = end
                continue
            remaining = self._buffer[cursor : cursor + self._max_alias + 1]
            candidates = self._index.get(remaining[0], [])
            matches = []
            incomplete = False
            for mapping in candidates:
                alias = mapping.surrogate
                if not final and (alias.startswith(remaining) or remaining == alias):
                    incomplete = True
                if not remaining.startswith(alias):
                    continue
                text = self._previous + remaining
                offset = len(self._previous)
                if mapping.kind == "path_root":
                    root = PathRoot(mapping.original, alias)
                    valid_match = any(
                        start == offset
                        for start, _, _ in root_occurrences(text, root, reverse=True)
                    )
                else:
                    valid_match = mapping.kind in {"secret", "basic"} or any(
                        start == offset for start, _ in literal_occurrences(text, alias)
                    )
                if valid_match:
                    matches.append(mapping)
            if incomplete:
                break
            if matches:
                mapping = matches[0]
                result.append(mapping.original)
                consumed = mapping.surrogate
            else:
                consumed = remaining[0]
                result.append(consumed)
            cursor += len(consumed)
            self._remember(consumed)
        self._buffer = self._buffer[cursor:]
        return "".join(result)

    def _remember(self, value: str) -> None:
        self._previous = (self._previous + value)[-8:]
