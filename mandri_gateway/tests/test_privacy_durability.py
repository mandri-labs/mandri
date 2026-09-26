import asyncio
import multiprocessing
import sqlite3
import time

import pytest
from mandri.core.types.config import PrivacySettings
from mandri.core.types.execution import ProtectionError
from mandri.database.privacy import PrivacyRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.gateway.privacy_scopes import PrivacyScopes
from mandri.gateway.surrogate import SurrogateScope


class SyntheticKeys:
    def load(self, *, create=False):
        return bytes(range(32))


class CommitGateDatabase(AiosqliteDatabase):
    def __init__(self, stage, channel):
        super().__init__()
        self.stage = stage
        self.channel = channel

    def pause(self):
        self.channel.send("allocation-barrier")
        time.sleep(15)
        raise RuntimeError("synthetic allocation barrier timed out")

    async def fetch_one(self, sql, params=()):
        row = await super().fetch_one(sql, params)
        if self.stage == "after" and sql.startswith("UPDATE privacy_scope SET payload"):
            await asyncio.to_thread(self.pause)
        return row


async def _allocate(path, stage, channel):
    db = CommitGateDatabase(stage, channel)
    await db.connect(path)
    if stage == "before":
        await db._require_connection().create_function("allocation_barrier", 0, db.pause)
        await db.execute(
            "CREATE TEMP TRIGGER privacy_allocation_barrier BEFORE UPDATE ON privacy_scope"
            " BEGIN SELECT allocation_barrier(); END"
        )
    scopes = PrivacyScopes(PrivacyRepository(db, SyntheticKeys()), PrivacySettings())
    await scopes.prepare("scope", lambda engine: engine.protect_text("new@example.invalid"))
    channel.send("allocation-returned")
    await db.close()


def _crash_writer(path, stage, channel):
    asyncio.run(_allocate(path, stage, channel))


@pytest.mark.parametrize("stage,revision", [("before", 0), ("after", 1)])
async def test_process_death_around_allocation_commit_is_atomic(tmp_path, stage, revision):
    path = tmp_path / "crash.sqlite"
    db = AiosqliteDatabase()
    await db.connect(path)
    await db.migrate()
    await PrivacyRepository(db, SyntheticKeys()).create("scope", SurrogateScope("scope").to_dict())
    await db.close()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_crash_writer, args=(path, stage, child))
    process.start()
    child.close()
    try:
        assert await asyncio.to_thread(parent.poll, 10), "allocation did not reach real SQL barrier"
        assert parent.recv() == "allocation-barrier"
        with sqlite3.connect(path) as independent:
            assert independent.execute("SELECT revision FROM privacy_scope").fetchone() == (
                revision,
            )
        assert not parent.poll(), "allocation returned to caller before injected process death"
        process.kill()
        await asyncio.to_thread(process.join, 5)
        assert not process.is_alive() and process.exitcode != 0
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        parent.close()
    await db.connect(path)
    try:
        repository = PrivacyRepository(db, SyntheticKeys())
        recovered = await repository.load("scope")
        assert recovered.revision == revision
        assert len(recovered.payload["mappings"]) == revision
        assert (await db.fetch_one("PRAGMA integrity_check"))["integrity_check"] == "ok"
        service = PrivacyScopes(repository, PrivacySettings())
        result, engine = await service.prepare(
            "scope", lambda engine: engine.protect_text("new@example.invalid")
        )
        assert engine.restore_text(result) == "new@example.invalid"
        persisted = await repository.load("scope")
        assert persisted.revision == 1
        assert len(persisted.payload["mappings"]) == 1
        if stage == "after":
            assert persisted.payload == recovered.payload
    finally:
        await db.close()


async def test_warm_scope_does_not_release_uncommitted_aliases_when_sqlite_is_full(tmp_path):
    path = tmp_path / "warm-full.sqlite"
    db = AiosqliteDatabase()
    await db.connect(path)
    await db.migrate()
    repository = PrivacyRepository(db, SyntheticKeys())
    await repository.create("scope", SurrogateScope("scope").to_dict())
    service = PrivacyScopes(repository, PrivacySettings())
    try:
        stable, _ = await service.prepare(
            "scope", lambda engine: engine.protect_text("stable@example.invalid")
        )
        committed = await repository.load("scope")
        pages = await db.fetch_one("PRAGMA page_count")
        await db.fetch_one(f"PRAGMA max_page_count = {pages['page_count']}")
        originals = [f"new-{index}@example.invalid" for index in range(1000)]
        with pytest.raises(ProtectionError) as raised:
            await service.prepare("scope", lambda engine: engine.protect(originals))
        assert raised.value.code == "privacy_state_unavailable"
        assert raised.value.__context__.__cause__.sqlite_errorcode == sqlite3.SQLITE_FULL
        assert (await repository.load("scope")).payload == committed.payload
        cached = service.cache.get("scope")
        assert cached.revision == committed.revision and cached.payload == committed.payload
        await db.fetch_one("PRAGMA max_page_count = 100000")
        retried, engine = await service.prepare(
            "scope", lambda engine: engine.protect(originals)
        )
        assert engine.restore(retried) == originals
        assert engine.protect_text("stable@example.invalid") == stable
        persisted = await repository.load("scope")
        assert persisted.revision == committed.revision + 1
        assert len(persisted.payload["mappings"]) == 1001
    finally:
        await db.close()
