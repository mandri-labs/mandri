from collections import deque
from collections.abc import Iterator

from mandri.gateway.surrogate.types import Mapping


class LiteralIndex:
    def __init__(self, mappings: list[Mapping], *, reverse: bool = False) -> None:
        self._reverse = reverse
        self._next: list[dict[str, int]] = [{}]
        self._failure = [0]
        self._output: list[list[tuple[str, Mapping]]] = [[]]
        for mapping in mappings:
            value = mapping.surrogate if reverse else mapping.original
            node = 0
            for char in value:
                if char not in self._next[node]:
                    self._next[node][char] = len(self._next)
                    self._next.append({})
                    self._failure.append(0)
                    self._output.append([])
                node = self._next[node][char]
            self._output[node].append((value, mapping))
        pending = deque(self._next[0].values())
        while pending:
            parent = pending.popleft()
            for char, child in self._next[parent].items():
                pending.append(child)
                failure = self._failure[parent]
                while failure and char not in self._next[failure]:
                    failure = self._failure[failure]
                self._failure[child] = self._next[failure].get(char, 0)
                self._output[child].extend(self._output[self._failure[child]])

    def find(self, text: str) -> Iterator[tuple[int, int, Mapping]]:
        node = 0
        for index, char in enumerate(text):
            while node and char not in self._next[node]:
                node = self._failure[node]
            node = self._next[node].get(char, 0)
            for value, mapping in self._output[node]:
                start, end = index + 1 - len(value), index + 1
                if (
                    (
                        mapping.kind not in {"secret", "basic"}
                        and (self._reverse or mapping.kind != "email")
                    )
                    and value[0].isalnum()
                    and start
                    and (text[start - 1].isalnum() or text[start - 1] == "_")
                ):
                    continue
                if (
                    (
                        mapping.kind not in {"secret", "basic"}
                        and (self._reverse or mapping.kind != "email")
                    )
                    and value[-1].isalnum()
                    and end < len(text)
                    and (text[end].isalnum() or text[end] == "_")
                ):
                    continue
                yield start, end, mapping
