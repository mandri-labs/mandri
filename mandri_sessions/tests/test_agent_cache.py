import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import ActivityState, HarnessKind, SessionState
from mandri.core.ports.agents import AgentDiscoveryResult
from mandri.core.types.agents import AgentState, NativeAgent
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.sessions.agents.service import AgentHistory
from mandri.sessions.agents.store import AgentStore
from mandri.sessions.transcripts.resolver import TranscriptResolver

from mandri_sessions.tests.substitutes import insert_session_row, make_session


@pytest.fixture
async def cache(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "cache.db")
    await db.migrate()
    rows = [make_session("old", updated_at=1), make_session("recent", updated_at=2)]
    for row in rows:
        await insert_session_row(db, str(row.id), native_id=str(row.native_id))
    sessions = SimpleNamespace(
        list_sessions=AsyncMock(side_effect=lambda: list(rows)),
        get_session=AsyncMock(
            side_effect=lambda identity: next(row for row in rows if row.id == identity)
        ),
        activity_of=Mock(return_value=None),
    )
    discovery = Mock()
    discovery.discover.return_value = AgentDiscoveryResult(
        HarnessKind.CLAUDE,
        [
            NativeAgent(HarnessKind.CLAUDE, "old-child", "old", "Old child", 1, 1),
            NativeAgent(HarnessKind.CLAUDE, "new-child", "recent", "New child", 1, 2),
        ],
    )
    store = AgentStore(db)
    history = AgentHistory(store, sessions, TranscriptResolver({}), [discovery])
    try:
        yield history, store, rows, sessions, discovery
    finally:
        await db.close()


async def test_listing_never_discovers_or_reads_histories(cache, monkeypatch):
    history, _, _, _, discovery = cache
    read = Mock(side_effect=AssertionError("History read on cache path"))
    monkeypatch.setattr("mandri.sessions.agents.service.historical_state", read)
    assert await history.list() == []
    discovery.discover.assert_not_called()
    await history.refresh(force=True)
    assert len(await history.list()) == 2
    assert await history.classified() == ["old", "recent"]
    read.assert_not_called()


async def test_backfill_newest_first_and_persists_completion(cache, monkeypatch):
    history, store, _, sessions, discovery = cache
    read = Mock(return_value=AgentState.COMPLETED)
    monkeypatch.setattr("mandri.sessions.agents.service.historical_state", read)
    await history.refresh(force=True)
    assert await history.backfill()
    assert read.call_args.args[0].parent_session_id == "recent"
    assert await history.backfill()
    assert await store.pending() == {}
    restarted = AgentHistory(store, sessions, TranscriptResolver({}), [discovery])
    await restarted.refresh(force=True)
    assert not await restarted.backfill()
    assert read.call_count == 2
    assert len(await restarted.list()) == 2


async def test_selected_missing_family_is_prioritized(cache, monkeypatch):
    history, _, _, _, _ = cache
    read = Mock(return_value=AgentState.COMPLETED)
    monkeypatch.setattr("mandri.sessions.agents.service.historical_state", read)
    await history.refresh(force=True)
    await history.list("old")
    assert await history.backfill()
    assert read.call_args.args[0].parent_session_id == "old"


@pytest.mark.parametrize("managed", [True, False])
async def test_active_sessions_are_excluded(cache, monkeypatch, managed):
    history, _, rows, sessions, _ = cache
    if managed:
        rows[1] = replace(rows[1], state=SessionState.LIVE)
    else:
        sessions.activity_of.side_effect = lambda identity: (
            SimpleNamespace(state=ActivityState.ACTIVE) if identity == "recent" else None
        )
    read = Mock(return_value=AgentState.COMPLETED)
    monkeypatch.setattr("mandri.sessions.agents.service.historical_state", read)
    await history.refresh(force=True)
    assert await history.backfill()
    assert read.call_args.args[0].parent_session_id == "old"
    assert not await history.backfill()


@pytest.mark.parametrize("change", ["event", "activation", "revision"])
async def test_inflight_backfill_cannot_overwrite_newer_state(cache, monkeypatch, change):
    history, store, rows, _, discovery = cache
    started, finish = threading.Event(), threading.Event()

    def delayed_read(*args):
        started.set()
        assert finish.wait(5)
        return AgentState.COMPLETED

    monkeypatch.setattr("mandri.sessions.agents.service.historical_state", delayed_read)
    await history.refresh(force=True)
    task = asyncio.create_task(history.backfill())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        agent = (await history.list("recent"))[0]
        if change == "event":
            await history.save(replace(agent, state=AgentState.RUNNING, updated_at=999))
        elif change == "activation":
            rows[1] = replace(rows[1], state=SessionState.LIVE)
        else:
            discovery.discover.return_value.agents[1] = replace(
                discovery.discover.return_value.agents[1], updated_at=99
            )
            await history.refresh(force=True)
    finally:
        finish.set()
    assert not await task
    assert "recent" in await store.pending()
    assert (await history.list("recent"))[0].state is not AgentState.COMPLETED


async def test_verified_empty_family_is_not_rescanned(cache):
    history, store, _, _, discovery = cache
    discovery.discover.return_value = AgentDiscoveryResult(HarnessKind.CLAUDE, [])
    assert await history.classified() == []
    await history.refresh(force=True)
    assert await history.backfill()
    assert await history.backfill()
    assert await store.pending() == {}
    await history.refresh(force=True)
    assert not await history.backfill()


async def test_removed_inactive_child_is_reconciled_but_active_family_is_preserved(cache):
    history, _, rows, _, discovery = cache
    await history.refresh(force=True)
    rows[1] = replace(rows[1], state=SessionState.LIVE)
    discovery.discover.return_value = AgentDiscoveryResult(HarnessKind.CLAUDE, [])
    await history.refresh(force=True)
    assert await history.list("old") == []
    assert len(await history.list("recent")) == 1


async def test_failed_discovery_preserves_cache_and_does_not_classify_new_sessions(cache):
    history, _, rows, _, discovery = cache
    await history.refresh(force=True)
    rows.append(make_session("unclassified"))
    discovery.discover.side_effect = OSError("unavailable")
    with pytest.raises(Exception, match="discovery is unavailable"):
        await history.refresh(force=True)
    assert "unclassified" not in await history.classified()
    assert len(await history.list()) == 2


async def test_partial_discovery_keeps_cached_children_and_withholds_unknown_roots(cache):
    history, _, _, _, discovery = cache
    discovery.discover.return_value = AgentDiscoveryResult(
        HarnessKind.CLAUDE,
        discovery.discover.return_value.agents,
        frozenset({"old"}),
        complete=False,
    )
    await history.refresh(force=True)
    assert await history.classified() == ["old"]
    cached = await history.list()
    discovery.discover.return_value = AgentDiscoveryResult(
        HarnessKind.CLAUDE, [], frozenset({"old"}), complete=False
    )
    await history.refresh(force=True)
    assert await history.list() == cached
    assert "recent" not in await history.classified()
    assert all(agent.state is AgentState.UNKNOWN for agent in cached)


async def test_event_during_discovery_preserves_state_and_links_imported_child(cache):
    history, _, rows, _, discovery = cache
    rows[:] = [replace(row, harness=HarnessKind.CODEX) for row in rows]
    discovery.discover.return_value = AgentDiscoveryResult(
        HarnessKind.CODEX,
        [replace(row, harness=HarnessKind.CODEX) for row in discovery.discover.return_value.agents],
    )
    await history.refresh(force=True)
    child = (await history.list("recent"))[0]
    rows.append(make_session("new-child", harness=HarnessKind.CODEX))
    started, finish = threading.Event(), threading.Event()
    native = discovery.discover.return_value

    def delayed_discovery(sessions):
        started.set()
        assert finish.wait(5)
        return native

    discovery.discover.side_effect = delayed_discovery
    task = asyncio.create_task(history.refresh(force=True))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        await history.save(replace(child, state=AgentState.RUNNING, updated_at=999))
    finally:
        finish.set()
    await task
    linked = (await history.list("recent"))[0]
    assert linked.state is AgentState.RUNNING
    assert linked.updated_at == 999
    assert linked.session_id == "new-child"


async def test_unavailable_state_is_not_marked_complete_or_blocks_older_families(
    cache, monkeypatch
):
    history, store, _, _, _ = cache
    read = Mock(
        side_effect=lambda agent, readers, parent: (
            AgentState.UNKNOWN if agent.parent_session_id == "recent" else AgentState.COMPLETED
        )
    )
    monkeypatch.setattr("mandri.sessions.agents.service.historical_state", read)
    await history.refresh(force=True)
    assert await history.backfill()
    assert await history.backfill()
    assert list(await store.pending()) == ["recent"]
