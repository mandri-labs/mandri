import asyncio
import secrets
from pathlib import Path

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.database.privacy import PrivacyRepository, PrivacyRevisionConflict
from mandri.database.sqlite_adapter import AiosqliteDatabase


class TestKeys:
    __test__ = False

    def __init__(self) -> None:
        self.key = secrets.token_bytes(32)

    def load(self, *, create: bool = False) -> bytes:
        return self.key


@pytest.fixture
async def db(tmp_path: Path):
    database = AiosqliteDatabase()
    await database.connect(tmp_path / "test.sqlite")
    await database.migrate()
    yield database
    await database.close()


async def test_ciphertext_hides_originals_and_reopens(db, tmp_path: Path) -> None:
    keys = TestKeys()
    repository = PrivacyRepository(db, keys)
    secret = "new-customer-61722@example.invalid"
    payload = {"mapping": {secret: "surrogate-88192@example.invalid"}}
    created = await repository.create("scope-a", payload)
    assert created.revision == 0
    row = await db.fetch_one("SELECT payload, wrapped_key FROM privacy_scope WHERE id = 'scope-a'")
    assert secret.encode() not in row["payload"]
    assert secret.encode() not in row["wrapped_key"]
    assert secret not in repr(created)
    await db.close()
    await db.connect(tmp_path / "test.sqlite")
    restored = await PrivacyRepository(db, keys).load("scope-a")
    assert restored.payload == payload


async def test_atomic_revision_rejects_stale_writer(db) -> None:
    repository = PrivacyRepository(db, TestKeys())
    first = await repository.create("scope-a", {"entries": []})
    second = await repository.load("scope-a")
    results = await asyncio.gather(
        repository.save(first, {"entries": ["first"]}),
        repository.save(second, {"entries": ["second"]}),
        return_exceptions=True,
    )
    assert sum(isinstance(result, PrivacyRevisionConflict) for result in results) == 1
    state = await repository.load("scope-a")
    assert state.revision == 1
    assert state.payload in ({"entries": ["first"]}, {"entries": ["second"]})


@pytest.mark.parametrize("field", ["payload", "wrapped_key"])
async def test_tampering_blocks_restore(db, field: str) -> None:
    repository = PrivacyRepository(db, TestKeys())
    await repository.create("scope-a", {"private": "synthetic"})
    row = await db.fetch_one(f"SELECT {field} FROM privacy_scope WHERE id = 'scope-a'")
    damaged = bytes([row[field][0] ^ 1]) + row[field][1:]
    await db.execute(f"UPDATE privacy_scope SET {field} = ? WHERE id = 'scope-a'", (damaged,))
    with pytest.raises(ProtectionError, match="authentication failed"):
        await repository.load("scope-a")


async def test_swapped_scope_or_revision_is_not_decryptable(db) -> None:
    repository = PrivacyRepository(db, TestKeys())
    await repository.create("scope-a", {"private": "synthetic"})
    await db.execute("UPDATE privacy_scope SET id = 'scope-b' WHERE id = 'scope-a'")
    with pytest.raises(ProtectionError):
        await repository.load("scope-b")
    await db.execute("UPDATE privacy_scope SET id = 'scope-a', revision = 12 WHERE id = 'scope-b'")
    with pytest.raises(ProtectionError):
        await repository.load("scope-a")


async def test_wrong_key_and_deleted_state_do_not_start_new_scope(db) -> None:
    keys = TestKeys()
    repository = PrivacyRepository(db, keys)
    await repository.create("scope-a", {"private": "synthetic"})
    with pytest.raises(ProtectionError):
        await PrivacyRepository(db, TestKeys()).load("scope-a")
    await repository.delete("scope-a")
    with pytest.raises(ProtectionError, match="unavailable"):
        await repository.load("scope-a")
    assert await db.fetch_all("SELECT * FROM privacy_scope") == []


async def test_referenced_scope_cannot_be_deleted(db) -> None:
    repository = PrivacyRepository(db, TestKeys())
    await repository.create("scope-a", {})
    await db.execute(
        "INSERT INTO gateway_route (id, provider_name, model_ref, formats, created_at,"
        " privacy_mode, privacy_scope_id) VALUES ('route-a', 'test', 'test', '[]', 0,"
        " 'surrogate', 'scope-a')"
    )
    with pytest.raises(ProtectionError, match="referenced"):
        await repository.delete("scope-a")
    await db.execute("DELETE FROM gateway_route WHERE id = 'route-a'")
    await repository.delete("scope-a")


async def test_non_json_payload_cannot_replace_committed_state(db) -> None:
    repository = PrivacyRepository(db, TestKeys())
    scope = await repository.create("scope-a", {"safe": True})
    with pytest.raises(ValueError):
        await repository.save(scope, {"invalid": float("nan")})
    assert (await repository.load("scope-a")).payload == {"safe": True}
