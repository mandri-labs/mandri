import sqlite3

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.database.privacy import PrivacyRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase


class SyntheticKeys:
    def load(self, *, create=False):
        return bytes(range(32))


@pytest.mark.parametrize("operation", ["create", "save"])
async def test_sqlite_full_preserves_last_committed_ciphertext(tmp_path, operation):
    path = tmp_path / "capacity.sqlite"
    db = AiosqliteDatabase()
    await db.connect(path)
    await db.migrate()
    repository = PrivacyRepository(db, SyntheticKeys())
    try:
        previous = await repository.create("existing", {"committed": "synthetic"})
        before = await db.fetch_one("SELECT * FROM privacy_scope WHERE id = 'existing'")
        pages = await db.fetch_one("PRAGMA page_count")
        limit = await db.fetch_one(f"PRAGMA max_page_count = {pages['page_count']}")
        assert limit["max_page_count"] == pages["page_count"]
        payload = {"private": "new-allocation@example.invalid" * 100_000}
        with pytest.raises(ProtectionError) as raised:
            if operation == "create":
                await repository.create("new", payload)
            else:
                await repository.save(previous, payload)
        assert raised.value.code == "privacy_state_unavailable"
        assert str(raised.value) == "Privacy storage is unavailable"
        cause = raised.value.__context__.__cause__
        assert isinstance(cause, sqlite3.OperationalError)
        assert cause.sqlite_errorcode == sqlite3.SQLITE_FULL
        assert await db.fetch_one("SELECT * FROM privacy_scope WHERE id = 'existing'") == before
        assert await db.fetch_one("SELECT id FROM privacy_scope WHERE id = 'new'") is None
        assert (await db.fetch_one("PRAGMA integrity_check"))["integrity_check"] == "ok"
    finally:
        await db.close()
    await db.connect(path)
    try:
        reopened = await PrivacyRepository(db, SyntheticKeys()).load("existing")
        assert reopened.revision == 0
        assert reopened.payload == {"committed": "synthetic"}
        assert path.stat().st_size <= pages["page_count"] * 4096
    finally:
        await db.close()
