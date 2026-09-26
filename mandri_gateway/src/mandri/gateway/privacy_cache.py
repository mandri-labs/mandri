import copy
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from mandri.database.privacy import StoredPrivacyScope


@dataclass(frozen=True, slots=True)
class CachedScope:
    scope: StoredPrivacyScope
    size: int


def snapshot(scope: StoredPrivacyScope, max_bytes: int) -> CachedScope:
    pending: list[Any] = [scope.payload, scope.scope_id, scope.key, scope.fingerprint]
    seen: set[int] = set()
    size = sys.getsizeof(scope) + 1024
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if size > max_bytes:
            return CachedScope(scope, size)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
    detached = StoredPrivacyScope(
        scope.scope_id, scope.revision, copy.deepcopy(scope.payload), scope.key, scope.fingerprint
    )
    return CachedScope(detached, size)


class ScopeCache:
    def __init__(self, *, max_bytes: int = 16 * 1024 * 1024, max_entries: int = 32) -> None:
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.size = 0
        self._entries: OrderedDict[str, CachedScope] = OrderedDict()

    def get(self, scope_id: str) -> StoredPrivacyScope | None:
        entry = self._entries.get(scope_id)
        if entry is None:
            return None
        self._entries.move_to_end(scope_id)
        return entry.scope

    def discard(self, scope_id: str) -> None:
        entry = self._entries.pop(scope_id, None)
        if entry is not None:
            self.size -= entry.size

    def put(self, entry: CachedScope) -> None:
        previous = self._entries.get(entry.scope.scope_id)
        if previous and previous.scope.revision > entry.scope.revision:
            return
        self.discard(entry.scope.scope_id)
        if entry.size > self.max_bytes or self.max_entries <= 0:
            return
        self._entries[entry.scope.scope_id] = entry
        self.size += entry.size
        while self.size > self.max_bytes or len(self._entries) > self.max_entries:
            _, removed = self._entries.popitem(last=False)
            self.size -= removed.size
