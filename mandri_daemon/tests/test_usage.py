import asyncio
import contextlib
import json
from decimal import Decimal
from pathlib import Path

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.types.usage import UsageObservation, UsagePrice
from mandri.daemon.usage import UsageCoordinator
from mandri.daemon.usage_history import UsageHistorySync
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository


def token_event(timestamp: str, count: int) -> str:
    return (
        json.dumps(
            {
                "timestamp": timestamp,
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": count,
                            "cached_input_tokens": 0,
                            "output_tokens": 10,
                            "reasoning_output_tokens": 0,
                            "total_tokens": count + 10,
                        }
                    },
                },
            }
        )
        + "\n"
    )


async def test_history_sync_is_bounded_durable_and_retains_deleted_session(tmp_path: Path):
    home = tmp_path / "native"
    (home / "sessions").mkdir(parents=True)
    transcript = home / "sessions" / "rollout-native-test.jsonl"
    transcript.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "native-test"}})
        + "\n"
        + json.dumps({"type": "turn_context", "payload": {"model": "gpt-6-astra"}})
        + "\n"
        + token_event("2026-09-20T08:00:00Z", 100)
        + token_event("2026-09-20T08:01:00Z", 150),
        encoding="utf-8",
    )
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "mandri.db")
    await db.migrate()
    repo = UsageRepository(db)
    await db.execute(
        "INSERT INTO session(id,harness,native_id,project_path,created_at,updated_at,state,"
        "last_synced_at,model_source) VALUES('s','codex','native-test','/synthetic',1,1,"
        "'stopped',1,'native')"
    )
    sync = UsageHistorySync(db, repo, home)
    await sync.reconcile()
    first = await repo.overview()
    assert first["summary"]["input_tokens"] == 150
    assert first["summary"]["total_tokens"] == 160
    await db.execute("UPDATE session SET deleted=1 WHERE id='s'")
    await db.close()
    await db.connect(tmp_path / "mandri.db")
    repo = UsageRepository(db)
    await UsageHistorySync(db, repo, home).reconcile()
    replay = await repo.overview()
    assert replay["revision"] == first["revision"]
    assert replay["summary"]["total_tokens"] == 160
    with transcript.open("a", encoding="utf-8") as stream:
        stream.write(token_event("2026-09-20T08:02:00Z", 200))
    await UsageHistorySync(db, repo, home).reconcile()
    assert (await repo.overview())["summary"]["total_tokens"] == 210
    await repo.erase_session("s")
    await UsageHistorySync(db, repo, home).reconcile()
    assert (await repo.overview())["summary"]["fact_count"] == 0
    await db.close()


async def test_coordinator_publishes_committed_revision_and_coalesces_refresh():
    db = AiosqliteDatabase()
    await db.connect(":memory:")
    await db.migrate()
    hub = Hub()
    subscription = hub.subscribe(Topic("usage.changed"))
    coordinator = UsageCoordinator(UsageRepository(db), hub)
    assert await coordinator.refresh()
    assert await coordinator.refresh()
    await coordinator.record(
        UsageObservation(
            source="test",
            source_key="r",
            fact_key="r",
            input_tokens=10,
        )
    )
    assert (await coordinator.repository.overview())["summary"]["input_tokens"] == 10
    frame = await asyncio.wait_for(subscription.queue.get(), timeout=1)
    assert frame is not None
    assert frame["payload"]["revision"] == await coordinator.repository.revision()
    coordinator.close()
    await hub.close_all()
    await db.close()


async def test_history_append_spans_batches_without_stopping_at_unchanged_revision(tmp_path):
    home = tmp_path / "native"
    (home / "sessions").mkdir(parents=True)
    path = home / "sessions" / "rollout-native-test.jsonl"
    path.write_text(
        json.dumps({"type": "turn_context", "payload": {"model": "m"}})
        + "\n"
        + token_event("2026-09-20T08:00:00Z", 1)
    )
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "db")
    await db.migrate()
    try:
        await db.execute(
            "INSERT INTO session(id,harness,native_id,project_path,created_at,updated_at,state,"
            "last_synced_at,model_source) VALUES('s','codex','native-test','/synthetic',1,1,"
            "'stopped',1,'native')"
        )
        repository = UsageRepository(db)
        await UsageHistorySync(db, repository, home).reconcile()
        with path.open("a") as stream:
            stream.write("".join(token_event("2026-09-20T08:01:00Z", n) for n in range(2, 1052)))
        sync = UsageHistorySync(db, repository, home)
        for _ in range(4):
            await sync.reconcile()
        assert (await repository.overview())["summary"]["input_tokens"] == 1051
        prior = await repository.revision()
        for _ in range(4):
            await sync.reconcile()
        assert await repository.revision() == prior
    finally:
        await db.close()


async def test_partial_trailing_record_does_not_hold_completed_history_forever(tmp_path):
    home = tmp_path / "native"
    (home / "sessions").mkdir(parents=True)
    path = home / "sessions" / "rollout-native-test.jsonl"
    prefix = json.dumps({"type": "turn_context", "payload": {"model": "m"}}) + "\n"
    event = token_event("2026-09-20T08:01:00Z", 30)
    path.write_text(prefix + token_event("2026-09-20T08:00:00Z", 20) + event[:30])
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "db")
    await db.migrate()
    try:
        await db.execute(
            "INSERT INTO session(id,harness,native_id,project_path,created_at,updated_at,state,"
            "last_synced_at,model_source) VALUES('s','codex','native-test','/synthetic',1,1,"
            "'stopped',1,'native')"
        )
        repository = UsageRepository(db)
        sync = UsageHistorySync(db, repository, home)
        await sync.reconcile()
        assert (await repository.overview())["summary"]["input_tokens"] == 20
        with path.open("a") as stream:
            stream.write(event[30:])
        for _ in range(2):
            await sync.reconcile()
        assert (await repository.overview())["summary"]["input_tokens"] == 30
    finally:
        await db.close()


async def test_restart_revalues_existing_facts_even_when_catalog_is_cached(monkeypatch):
    db = AiosqliteDatabase()
    await db.connect(":memory:")
    await db.migrate()
    repository = UsageRepository(db)
    coordinator = UsageCoordinator(repository, Hub())
    observed = asyncio.Event()
    calls = []

    async def cached_prices():
        return {"cached": True, "price_count": 10}

    async def revalue(**fields):
        calls.append(fields)
        observed.set()
        return {"next_key": None}

    coordinator.refresh_prices = cached_prices
    monkeypatch.setattr(repository, "revalue_unpriced", revalue)
    task = asyncio.create_task(coordinator.run())
    try:
        await asyncio.wait_for(observed.wait(), 1)
        assert calls[0]["all_facts"] is True
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        coordinator.close()
        await db.close()


async def test_claude_native_history_is_priced_per_message_model(tmp_path):
    home = tmp_path / "claude"
    project = home / "projects" / "-synthetic"
    project.mkdir(parents=True)
    events = [
        {
            "type": "assistant",
            "sessionId": "native-test",
            "timestamp": "2026-09-20T08:00:00Z",
            "message": {
                "id": f"msg-{index}",
                "model": model,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
        for index, model in enumerate(("claude-test-a", "claude-test-b"))
    ]
    (project / "native-test.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "db")
    await db.migrate()
    try:
        await db.execute(
            "INSERT INTO session(id,harness,native_id,project_path,created_at,updated_at,state,"
            "last_synced_at,model_source,model) VALUES('s','claude','native-test','/synthetic',"
            "1,1,'stopped',1,'native','claude-test-b')"
        )
        repository = UsageRepository(db)
        for index, model in enumerate(("claude-test-a", "claude-test-b"), 1):
            await repository.add_price(
                UsagePrice(
                    price_id=model,
                    provider="anthropic",
                    model=model,
                    effective_from=0,
                    rates={"input_tokens": Decimal(index), "output_tokens": Decimal(index)},
                )
            )
        sync = UsageHistorySync(db, repository, tmp_path / "codex", claude_home=home)
        await sync.reconcile()
        overview = await repository.overview()
        assert {row["key"] for row in overview["breakdown"]} == {"claude-test-a", "claude-test-b"}
        assert overview["summary"]["usd_equivalent"] == "0.000036"
        assert overview["summary"]["unclassified_fact_count"] == 0
        assert overview["summary"]["request_count"] == 2
        prior = await repository.revision()
        await UsageHistorySync(db, repository, tmp_path / "codex", claude_home=home).reconcile()
        assert await repository.revision() == prior
    finally:
        await db.close()


@pytest.mark.parametrize("model", ["mandri_gateway", "opencode_go/glm-5.3-flash"])
async def test_opencode_history_never_promotes_gateway_harness_models(tmp_path, model):
    db = AiosqliteDatabase()
    await db.connect(":memory:")
    await db.migrate()
    repository = UsageRepository(db)
    sync = UsageHistorySync(db, repository, tmp_path)
    value = UsageObservation(
        source="native:opencode",
        source_key="message",
        fact_key="message",
        provider="mandri",
        model=model,
        input_tokens=100,
        output_tokens=10,
    )
    try:
        await repository.record(sync._opencode_attributed(value))
        assert (await repository.overview())["summary"]["fact_count"] == 0
        stored = await db.fetch_one("SELECT payload FROM usage_observation")
        assert stored is not None
        assert json.loads(stored["payload"])["authoritative"] is False
    finally:
        await db.close()
