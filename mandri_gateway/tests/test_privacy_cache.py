import asyncio
import secrets
import threading

import pytest
from mandri.core.types.config import PrivacySettings
from mandri.core.types.execution import ProtectionError
from mandri.database.privacy import PrivacyRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.gateway.privacy_cache import ScopeCache
from mandri.gateway.privacy_scopes import PrivacyScopes
from mandri.gateway.surrogate import SurrogateScope


class SyntheticKeys:
    def __init__(self):
        self.key = secrets.token_bytes(32)
        self.available = True

    def load(self, *, create=False):
        if not self.available:
            raise ProtectionError("privacy_key_unavailable", "Synthetic key is locked")
        return self.key


class CountingRepository(PrivacyRepository):
    def __init__(self, db, keys):
        super().__init__(db, keys)
        self.loads = 0

    async def load(self, scope_id):
        self.loads += 1
        return await super().load(scope_id)


@pytest.fixture
async def cached(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "cache.sqlite")
    await db.migrate()
    keys = SyntheticKeys()
    repository = CountingRepository(db, keys)
    scope = SurrogateScope("scope")
    await repository.create(scope.scope_id, scope.to_dict())
    service = PrivacyScopes(repository, PrivacySettings())
    yield service, db, keys
    await db.close()


async def test_warm_reuse_avoids_full_decryption_and_detaches_mutable_engines(cached):
    service, _, _ = cached
    alias, returned = await service.prepare(
        "scope", lambda engine: engine.protect_text("author@private.example")
    )
    returned.register("uncommitted-private-value")
    for _ in range(3):
        repeated, engine = await service.prepare(
            "scope", lambda engine: engine.protect_text("author@private.example")
        )
        assert repeated == alias
        assert "uncommitted-private-value" not in {item.original for item in engine.scope.mappings}
    assert service.repository.loads == 1


@pytest.mark.parametrize("change", ["key_locked", "wrong_key", "deleted", "ciphertext", "version"])
async def test_warm_cache_cannot_hide_key_or_persistent_state_failure(cached, change):
    service, db, keys = cached
    await service.prepare("scope", lambda engine: engine.protect_text("author@private.example"))
    if change == "key_locked":
        keys.available = False
    elif change == "wrong_key":
        keys.key = secrets.token_bytes(32)
    elif change == "deleted":
        await db.execute("DELETE FROM privacy_scope WHERE id = 'scope'")
    elif change == "version":
        await db.execute("UPDATE privacy_scope SET version = 999 WHERE id = 'scope'")
    else:
        row = await db.fetch_one("SELECT payload FROM privacy_scope WHERE id = 'scope'")
        damaged = bytes([row["payload"][0] ^ 1]) + row["payload"][1:]
        await db.execute("UPDATE privacy_scope SET payload = ? WHERE id = 'scope'", (damaged,))
    invoked = []
    with pytest.raises(ProtectionError):
        await service.prepare("scope", lambda engine: invoked.append(True))
    assert invoked == []
    assert service.cache.get("scope") is None


async def test_revision_change_from_another_service_invalidates_warm_scope(cached):
    service, _, _ = cached
    await service.validate("scope")
    other = PrivacyScopes(service.repository, PrivacySettings())
    alias, _ = await other.prepare(
        "scope", lambda engine: engine.protect_text("second@private.example")
    )
    original, _ = await service.prepare("scope", lambda engine: engine.restore_text(alias))
    assert original == "second@private.example"
    assert service.repository.loads == 3


async def test_cached_concurrent_allocations_remain_atomic_and_consistent(cached):
    service, _, _ = cached
    await service.validate("scope")
    results = await asyncio.gather(
        *[
            service.prepare("scope", lambda engine, value=value: engine.protect_text(value))
            for value in [
                "shared@private.example",
                "first@private.example",
                "second@private.example",
                "shared@private.example",
            ]
        ]
    )
    assert results[0][0] == results[3][0]
    stored = await service.repository.load("scope")
    restored = service.engine(stored.payload)
    assert [restored.restore_text(result[0]) for result in results] == [
        "shared@private.example",
        "first@private.example",
        "second@private.example",
        "shared@private.example",
    ]


async def test_lru_eviction_limits_local_decrypted_scope_memory(cached):
    service, _, _ = cached
    service.cache = ScopeCache(max_bytes=6000, max_entries=2)
    for identity in ["scope", "second", "third"]:
        if identity != "scope":
            await service.repository.create(identity, SurrogateScope(identity).to_dict())
        await service.validate(identity)
    assert service.cache.get("scope") is None
    assert service.cache.size <= 6000
    before = service.repository.loads
    await service.validate("scope")
    assert service.repository.loads == before + 1
    service.cache = ScopeCache(max_bytes=10, max_entries=2)
    await service.validate("scope")
    assert service.cache.get("scope") is None
    assert service.cache.size == 0


async def test_cancellation_remains_responsive_during_cpu_preparation_without_commit(cached):
    service, _, _ = cached
    await service.validate("scope")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def transform(engine):
        entered.set()
        release.wait(timeout=5)
        engine.register("cancelled-sensitive-value")
        finished.set()

    request = asyncio.create_task(service.prepare("scope", transform))
    assert await asyncio.to_thread(entered.wait, 2)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(request, timeout=1)
    release.set()
    assert await asyncio.to_thread(finished.wait, 2)
    stored = await service.repository.load("scope")
    assert stored.revision == 0
    assert stored.payload["mappings"] == []


async def test_many_parallel_preparations_across_services_keep_shared_aliases(cached):
    service, _, _ = cached
    services = [service, PrivacyScopes(service.repository, PrivacySettings())]
    originals = [f"person{index % 12}@private.example" for index in range(48)]
    results = await asyncio.gather(
        *[
            services[index % 2].prepare(
                "scope", lambda engine, value=value: engine.protect_text(value)
            )
            for index, value in enumerate(originals)
        ]
    )
    stored = await service.repository.load("scope")
    resumed = service.engine(stored.payload)
    assert len(stored.payload["mappings"]) == 12
    assert len({result[0] for result in results}) == 12
    for original, (alias, _) in zip(originals, results, strict=True):
        assert resumed.protect_text(original) == alias
        assert resumed.restore_text(alias) == original
