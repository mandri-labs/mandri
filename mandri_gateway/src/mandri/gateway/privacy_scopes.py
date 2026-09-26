import asyncio
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar
from weakref import WeakValueDictionary

from mandri.core.types.config import PrivacySettings
from mandri.core.types.execution import ProtectionError
from mandri.database.privacy import PrivacyRepository, PrivacyRevisionConflict, StoredPrivacyScope
from mandri.gateway.privacy_cache import ScopeCache, snapshot
from mandri.gateway.privacy_context import metadata_read_flags, seed_context, seed_home
from mandri.gateway.privacy_inventory import inventory
from mandri.gateway.privacy_rules import load_extensions
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope

T = TypeVar("T")


class PrivacyScopes:
    def __init__(
        self,
        repository: PrivacyRepository,
        settings: PrivacySettings,
        credentials: Callable[[], Iterable[str]] = tuple,
    ) -> None:
        self.repository = repository
        self.credentials = credentials
        self.cache = ScopeCache()
        self.rules_file = settings.rules_file
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    async def readiness(self) -> None:
        metadata_read_flags()
        await self.repository.readiness()

    def engine(self, payload: dict[str, Any]) -> SurrogateEngine:
        return SurrogateEngine(SurrogateScope.from_dict(payload))

    async def create(self, workspace_root: str) -> str:
        metadata_read_flags()
        root = Path(workspace_root).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise ProtectionError(
                "workspace_unavailable", "Workspace must be an existing directory"
            )
        extensions = await asyncio.to_thread(load_extensions, self.rules_file, root)
        scope = SurrogateScope(str(uuid.uuid4()), rules=list(extensions.rules))
        engine = SurrogateEngine(scope)
        self._seed_credentials(engine)
        for exact in extensions.exact:
            engine.register(exact.value, exact.kind, exact.context)
        seed_context(engine, root.resolve())
        engine.register_root(str(root))
        canonical = str(root.resolve())
        if canonical != str(root):
            engine.register_root(canonical)
        seed_home(engine)
        await self.repository.create(scope.scope_id, scope.to_dict())
        return scope.scope_id

    def _seed_credentials(self, engine: SurrogateEngine) -> None:
        for credential in self.credentials():
            if credential:
                engine.register(credential, "secret", "provider_config")

    async def inventory(self, scope_id: str) -> tuple[int, list[dict[str, str]]]:
        await self.prepare(scope_id, self._seed_credentials)
        stored = await self._load(scope_id)
        scope = SurrogateScope.from_dict(stored.payload)
        return stored.revision, inventory(scope.mappings)

    async def add_workspace(self, scope_id: str, workspace_root: str) -> None:
        root = Path(workspace_root).resolve(strict=True)

        def register(engine: SurrogateEngine) -> None:
            seed_context(engine, root)
            engine.register_root(str(root))

        await self.prepare(scope_id, register)

    async def validate(self, scope_id: str) -> None:
        stored = await self._load(scope_id)
        await asyncio.to_thread(self.engine, stored.payload)

    async def delete(self, scope_id: str) -> None:
        self.cache.discard(scope_id)
        await self.repository.delete(scope_id)

    async def fork(self, scope_id: str) -> str:
        stored = await self._load(scope_id)
        scope = SurrogateScope.from_dict(stored.payload)
        scope.scope_id = str(uuid.uuid4())
        await self.repository.create(scope.scope_id, scope.to_dict())
        return scope.scope_id

    async def _load(self, scope_id: str) -> StoredPrivacyScope:
        cached = self.cache.get(scope_id)
        if cached is not None:
            try:
                revision, fingerprint = await self.repository.revision(scope_id)
            except BaseException:
                self.cache.discard(scope_id)
                raise
            if (cached.revision, cached.fingerprint) == (revision, fingerprint):
                return cached
            self.cache.discard(scope_id)
        stored = await self.repository.load(scope_id)
        self.cache.put(await asyncio.to_thread(snapshot, stored, self.cache.max_bytes))
        return stored

    def _transform(
        self, stored: StoredPrivacyScope, transform: Callable[[SurrogateEngine], T]
    ) -> tuple[T, SurrogateEngine, dict[str, Any], bool]:
        engine = self.engine(stored.payload)
        self._seed_credentials(engine)
        result = transform(engine)
        payload = engine.scope.to_dict()
        return (result, engine, payload, payload != stored.payload)

    async def prepare(
        self, scope_id: str, transform: Callable[[SurrogateEngine], T]
    ) -> tuple[T, SurrogateEngine]:
        lock = self._locks.setdefault(scope_id, asyncio.Lock())
        async with lock:
            while True:
                stored = await self._load(scope_id)
                result, engine, payload, changed = await asyncio.to_thread(
                    self._transform, stored, transform
                )
                if not changed:
                    return result, engine
                try:
                    saved = await self.repository.save(stored, payload)
                except PrivacyRevisionConflict:
                    self.cache.discard(scope_id)
                    await asyncio.sleep(0)
                    continue
                self.cache.put(await asyncio.to_thread(snapshot, saved, self.cache.max_bytes))
                return result, engine
