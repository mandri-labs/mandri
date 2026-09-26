"""Concurrent access coherence across two connections to the same database file."""

import asyncio

from mandri.database.sqlite_adapter import AiosqliteDatabase

_INSERT = (
    "INSERT INTO session (id, harness, project_path, created_at, updated_at,"
    " state, deleted, last_synced_at) VALUES (?, 'claude', 'C:/w', 1, 1,"
    " 'discovered', 0, 1)"
)


async def _connect(tmp_path: object, name: str) -> AiosqliteDatabase:
    adapter = AiosqliteDatabase()
    await adapter.connect(tmp_path / name)
    await adapter.migrate()
    return adapter


async def test_write_on_one_connection_visible_on_the_other(tmp_path: object) -> None:
    writer = await _connect(tmp_path, "mandri.db")
    reader = await _connect(tmp_path, "mandri.db")
    await writer.execute(_INSERT, ("s1",))
    rows = await reader.fetch_all("SELECT id FROM session")
    assert [row["id"] for row in rows] == ["s1"]
    await writer.close()
    await reader.close()


async def test_concurrent_writes_from_both_connections_leave_consistent_state(
    tmp_path: object,
) -> None:
    first = await _connect(tmp_path, "mandri.db")
    second = await _connect(tmp_path, "mandri.db")
    await asyncio.gather(
        *(first.execute(_INSERT, (f"a{index}",)) for index in range(25)),
        *(second.execute(_INSERT, (f"b{index}",)) for index in range(25)),
    )
    rows = await first.fetch_all("SELECT id FROM session")
    assert len(rows) == 50
    assert len({str(row["id"]) for row in rows}) == 50
    await first.close()
    await second.close()


async def test_wal_journal_mode_absorbs_reader_writer_contention(tmp_path: object) -> None:
    writer = await _connect(tmp_path, "mandri.db")
    reader = await _connect(tmp_path, "mandri.db")
    mode = await reader.fetch_one("PRAGMA journal_mode")
    assert mode is not None and str(mode["journal_mode"]).lower() == "wal"
    for index in range(20):
        await writer.execute(_INSERT, (f"s{index}",))
        rows = await reader.fetch_all("SELECT id FROM session")
        assert len(rows) == index + 1
    await writer.close()
    await reader.close()
