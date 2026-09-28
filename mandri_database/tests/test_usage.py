import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from mandri.core.types.usage import UsageAccount, UsageFilters, UsageObservation, UsagePrice
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.database.usage import UsageRepository
from mandri.database.usage_migrations import migrate_usage
from mandri.database.usage_serialization import encode


@pytest.fixture
async def store(tmp_path):
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "usage.sqlite")
    await db.migrate()
    await migrate_usage(db._require_connection())
    yield db
    await db.close()


@pytest.fixture
def repository(store):
    return UsageRepository(store)


def usage(**fields):
    return UsageObservation(
        **{
            "source": "gateway",
            "source_key": "request",
            "fact_key": "request",
            "session_id": "s1",
            "occurred_at": 1000,
            **fields,
        }
    )


async def test_replay_replacement_and_missing_vs_zero(repository):
    pending = usage(request_count=1)
    assert await repository.record(pending) == 1
    assert await repository.record(pending) == 1
    result = await repository.overview()
    assert result["summary"]["input_tokens"] is None
    assert result["summary"]["missing_fields"]["input_tokens"] == 1
    complete = replace(pending, sequence=1, input_tokens=0, output_tokens=7, complete=True)
    await repository.record(complete)
    result = await repository.overview()
    assert result["summary"]["fact_count"] == 1
    assert result["summary"]["input_tokens"] == 0
    assert result["summary"]["output_tokens"] == 7
    assert result["summary"]["request_count"] == 1
    assert result["summary"]["usd_equivalent"] is None
    assert await repository.record(pending) == 2


async def test_attribution_snapshot_and_model_refinement(repository):
    await repository.record(usage(project_path="old", provider="p", model=None))
    await repository.record(usage(sequence=1, project_path="new", provider="q", model="m"))
    old = await repository.overview(UsageFilters(project_path="old", model="m"))
    assert old["summary"]["fact_count"] == 1
    assert (await repository.overview(UsageFilters(project_path="new")))["summary"][
        "fact_count"
    ] == 0


async def test_cumulative_baseline_delta_restart_and_regression(repository):
    baseline = usage(
        source="native",
        kind="cumulative",
        epoch="lifetime:s1",
        baseline=True,
        sequence=1,
        input_tokens=100,
        output_tokens=10,
    )
    await repository.record(baseline)
    assert (await repository.overview())["summary"]["fact_count"] == 0
    next_value = replace(
        baseline,
        baseline=False,
        source_key="second",
        sequence=2,
        occurred_at=2000,
        input_tokens=125,
        output_tokens=15,
    )
    await repository.record(next_value)
    assert await repository.record(next_value) == 2
    assert await repository.record(replace(baseline, source_key="stale")) == 2
    assert (await repository.overview())["summary"]["input_tokens"] == 25
    with pytest.raises(ValueError, match="decreased"):
        await repository.record(replace(next_value, source_key="crash", sequence=3, input_tokens=0))
    assert await repository.revision() == 2
    await repository.record(
        replace(
            next_value,
            source_key="third",
            sequence=3,
            occurred_at=3000,
            input_tokens=150,
            output_tokens=20,
        )
    )
    assert (await repository.overview())["summary"]["input_tokens"] == 50


async def test_first_cumulative_is_undated_and_partial_intervals_are_not_fabricated(repository):
    first = usage(kind="cumulative", epoch="s1", input_tokens=100, sequence=1)
    await repository.record(first)
    all_usage = await repository.overview()
    assert all_usage["summary"]["input_tokens"] == 100
    assert all_usage["undated"]["input_tokens"] == 100
    assert all_usage["timeseries"] == []
    await repository.record(
        replace(
            first, source_key="second", sequence=2, input_tokens=150, occurred_at=3 * 86_400_000
        )
    )
    partial = await repository.overview(UsageFilters(from_ms=86_400_000, to_ms=4 * 86_400_000))
    assert partial["summary"]["input_tokens"] is None
    assert partial["unallocated"]["input_tokens"] == 50
    assert partial["undated"]["input_tokens"] == 100
    assert partial["timeseries"] == []


async def test_erasure_suppresses_replay_and_does_not_change_accounts(repository):
    await repository.record(usage(input_tokens=10))
    account = UsageAccount(account_id="a", harness="codex", observed_at=10, plan="pro")
    await repository.upsert_account(account)
    revision = await repository.erase_session("s1")
    assert await repository.record(usage(sequence=2, input_tokens=20)) == revision
    assert (await repository.overview())["summary"]["fact_count"] == 0
    assert len(await repository.accounts()) == 1
    assert await repository.erase_session("s1") == revision


async def test_inclusive_child_erasure_requires_explicit_enclosing_scope(repository):
    await repository.record(usage(included_session_ids=("child",), input_tokens=10))
    with pytest.raises(ValueError, match="enclosing"):
        await repository.erase_session("child")
    assert (await repository.overview())["summary"]["input_tokens"] == 10


async def test_deleted_sessions_remain_and_no_session_foreign_key(store, repository):
    await store.execute(
        "INSERT INTO session (id,harness,project_path,created_at,updated_at,"
        "state,last_synced_at,deleted) VALUES ('s1','codex','p',1,1,'stopped',1,1)"
    )
    await repository.record(usage(input_tokens=10))
    assert (await repository.overview())["summary"]["input_tokens"] == 10
    assert (await repository.overview(UsageFilters(include_deleted=False)))["summary"][
        "fact_count"
    ] == 0
    await store.execute("DELETE FROM session WHERE id='s1'")
    assert (await repository.overview())["summary"]["input_tokens"] == 10


async def test_concurrent_repositories_and_unrelated_writes_are_atomic(store, repository):
    second = UsageRepository(store)
    await store.execute("CREATE TABLE unrelated (n INTEGER)")
    tasks = []
    for n in range(50):
        value = usage(source_key=str(n), fact_key=str(n), input_tokens=1, request_count=1)
        tasks.extend(
            (
                repository.record(value),
                second.record(value),
                store.execute("INSERT INTO unrelated VALUES (?)", (n,)),
            )
        )
    await asyncio.gather(*tasks)
    snapshot = await repository.overview()
    assert snapshot["revision"] == 50
    assert snapshot["summary"]["input_tokens"] == 50
    assert snapshot["timeseries"][0]["fact_count"] == 50
    assert (await store.fetch_one("SELECT count(*) n FROM unrelated"))["n"] == 50


async def test_explicit_decimal_price_pinned_and_not_retroactive(repository):
    tariff = UsagePrice(
        price_id="v1",
        provider="p",
        model="m",
        effective_from=0,
        rates={
            "input_tokens": Decimal("2"),
            "cache_read_tokens": Decimal("0.2"),
            "output_tokens": Decimal("10"),
        },
    )
    await repository.add_price(tariff)
    fact = usage(
        provider="p",
        model="m",
        input_tokens=1000,
        output_tokens=200,
        cache_read_tokens=600,
        cache_write_tokens=0,
        reasoning_tokens=50,
        input_includes_cache=True,
        output_includes_reasoning=True,
    )
    await repository.record(fact)
    assert (await repository.overview())["summary"]["usd_equivalent"] == "0.00292"
    await repository.add_price(
        replace(tariff, price_id="v2", effective_from=1, rates={"input_tokens": Decimal("200")})
    )
    await repository.record(replace(fact, sequence=1, complete=True))
    assert (await repository.overview())["summary"]["usd_equivalent"] == "0.00292"
    with pytest.raises(ValueError, match="immutable"):
        await repository.add_price(replace(tariff, rates={"input_tokens": Decimal(3)}))


async def test_accounts_ignore_stale_and_snapshot_revision(repository):
    account = UsageAccount(
        account_id="a",
        harness="codex",
        observed_at=20,
        windows=({"used_percent": 10, "resets_at": None},),
    )
    rev = await repository.upsert_account(account)
    assert await repository.upsert_account(replace(account, observed_at=10)) == rev
    snapshot = await repository.account_snapshot()
    assert snapshot["revision"] == rev
    assert snapshot["accounts"][0]["windows"][0]["resets_at"] is None


@pytest.mark.parametrize("status", ["unavailable", "unsupported", "authentication_required"])
async def test_empty_account_probes_do_not_create_accounts(repository, store, status):
    account = UsageAccount(account_id="empty", harness="codex", observed_at=20, status=status)
    rev = await repository.revision()
    assert await repository.upsert_account(account) == rev
    assert await repository.accounts() == []
    assert await store.fetch_all("SELECT * FROM usage_account") == []


async def test_failed_account_refresh_keeps_last_known_snapshot(repository):
    known = UsageAccount(
        account_id="known",
        harness="codex",
        observed_at=20,
        status="available",
        plan="pro",
        windows=({"used_percent": 46},),
    )
    rev = await repository.upsert_account(known)
    expected = await repository.accounts()
    failure = UsageAccount(account_id="known", harness="codex", observed_at=30)
    assert await repository.upsert_account(failure) == rev
    assert await repository.accounts() == expected


async def test_historical_empty_profiles_are_hidden_without_collapsing_known_accounts(
    repository,
    store,
):
    for account in (
        UsageAccount(account_id="empty-1", harness="codex", observed_at=10),
        UsageAccount(account_id="empty-2", harness="codex", observed_at=10),
        UsageAccount(account_id="known-1", harness="codex", observed_at=20, plan="pro"),
        UsageAccount(account_id="known-2", harness="codex", observed_at=20, verified=True),
    ):
        await store.execute(
            "INSERT INTO usage_account VALUES (?,?,?)",
            (account.account_id, account.observed_at, encode(account)),
        )
    snapshot = await repository.account_snapshot()
    assert [account["account_id"] for account in snapshot["accounts"]] == ["known-1", "known-2"]
    assert len(await store.fetch_all("SELECT * FROM usage_account")) == 4


@pytest.mark.parametrize(
    "fields",
    [
        {"plan": "pro"},
        {"auth_mode": "native"},
        {"verified": True},
        {"windows": ({"used_percent": 0},)},
        {"credits": Decimal(0)},
        {"monthly_fee_usd": Decimal(0)},
    ],
)
async def test_known_accounts_without_available_quotas_remain_visible(repository, fields):
    await repository.upsert_account(
        UsageAccount(account_id="known", harness="codex", observed_at=20, **fields),
    )
    accounts = await repository.accounts()
    assert len(accounts) == 1
    assert accounts[0]["status"] == "unavailable"


@pytest.mark.parametrize(
    "filters",
    [
        UsageFilters(from_ms=0, to_ms=3661 * 86_400_000),
        UsageFilters(timezone="bad/zone"),
        UsageFilters(limit=1001),
        UsageFilters(expected_revision=10),
    ],
)
async def test_invalid_queries(repository, filters):
    with pytest.raises(ValueError):
        await repository.overview(filters)


async def test_non_authoritative_observations_never_count(repository):
    await repository.record(usage(authoritative=False, input_tokens=100))
    assert (await repository.overview())["summary"]["fact_count"] == 0


async def test_failed_update_rolls_back_observation_and_revision(repository, store):
    original = usage(input_tokens=20)
    await repository.record(original)
    with pytest.raises(ValueError):
        await repository.record(replace(original, sequence=1, input_tokens=10))
    row = await store.fetch_one("SELECT payload FROM usage_observation")
    assert '"sequence":0' in row["payload"]
    assert await repository.revision() == 1


async def test_cursor_and_fact_batch_atomic_replay_and_erasure(repository, store):
    cursor = {
        "session_id": "s1",
        "identity": "hash",
        "offset": 50,
        "anchor": "digest",
        "parser_version": 1,
        "status": "ready",
    }
    rev = await repository.record_batch([usage(input_tokens=20)], "source-digest", cursor)
    assert await repository.read_cursor("source-digest") == cursor
    assert await repository.record_batch([usage(input_tokens=20)], "source-digest", cursor) == rev
    with pytest.raises(ValueError):
        await repository.record_batch(
            [
                usage(source_key="new", fact_key="new", input_tokens=5),
                usage(sequence=1, input_tokens=10),
            ],
            "source-digest",
            {**cursor, "offset": 100},
        )
    assert await repository.read_cursor("source-digest") == cursor
    assert (await repository.overview())["summary"]["input_tokens"] == 20
    assert await repository.revision() == rev
    await repository.erase_session("s1")
    erased = await repository.revision()
    assert await repository.read_cursor("source-digest") is None
    assert await repository.record_batch([usage()], "source-digest", cursor) == erased
    assert await repository.read_cursor("source-digest") is None
    assert (await store.fetch_one("SELECT count(*) n FROM usage_cursor"))["n"] == 0


async def test_erased_native_identity_cannot_reimport_under_new_session(repository):
    await repository.record(usage(native_session_id="native", input_tokens=10))
    rev = await repository.erase_session("s1")
    assert (
        await repository.record(
            usage(
                session_id="new",
                native_session_id="native",
                source_key="new-record",
                fact_key="new-record",
            )
        )
        == rev
    )
    assert (await repository.overview())["summary"]["fact_count"] == 0


async def test_reopen_persists_replay_and_baseline(tmp_path):
    path = tmp_path / "restart.sqlite"
    db = AiosqliteDatabase()
    await db.connect(path)
    await db.migrate()
    await migrate_usage(db._require_connection())
    repository = UsageRepository(db)
    value = usage(kind="cumulative", epoch="epoch", baseline=True, input_tokens=100, sequence=1)
    await repository.record(value)
    await db.close()
    await db.connect(path)
    try:
        repository = UsageRepository(db)
        assert await repository.record(value) == 1
        await repository.record(
            replace(
                value,
                source_key="next",
                sequence=2,
                baseline=False,
                input_tokens=125,
                occurred_at=2000,
            )
        )
        assert (await repository.overview())["summary"]["input_tokens"] == 25
    finally:
        await db.close()


async def test_memory_connection_transaction_support():
    db = AiosqliteDatabase()
    await db.connect(":memory:")
    try:
        await db.migrate()
        await migrate_usage(db._require_connection())
        repository = UsageRepository(db)
        await repository.record(usage(input_tokens=3))
        assert (await repository.overview())["summary"]["input_tokens"] == 3
    finally:
        await db.close()


async def test_observed_model_and_session_changes_do_not_override_gateway_selection(repository):
    await repository.record(usage(model="auto", project_path="old"))
    await repository.record(usage(model="session-new", project_path="new", sequence=1))
    assert (await repository.overview())["breakdown"][0]["key"] == "auto"
    await repository.record(usage(model="session-new", observed_model="actual", sequence=2))
    assert (await repository.overview(UsageFilters(model="auto")))["summary"]["fact_count"] == 1
    assert (await repository.overview(UsageFilters(project_path="old")))["summary"][
        "fact_count"
    ] == 1


async def test_unknown_price_backfill_bounded_without_changing_pinned(repository):
    fact = usage(
        provider="p",
        model="m",
        input_tokens=100,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        input_includes_cache=False,
        output_includes_reasoning=False,
    )
    for n in range(3):
        await repository.record(replace(fact, source_key=str(n), fact_key=str(n)))
    await repository.add_price(
        UsagePrice(
            price_id="v1",
            provider="p",
            model="m",
            effective_from=0,
            rates={"input_tokens": Decimal("2")},
        )
    )
    assert (await repository.overview())["summary"]["usd_equivalent"] is None
    page = await repository.revalue_unpriced(limit=2)
    assert page["updated"] == 2
    assert page["next_key"] == "1"
    page = await repository.revalue_unpriced(limit=2, after_key=page["next_key"])
    assert page["updated"] == 1
    assert page["next_key"] is None
    assert (await repository.overview())["summary"]["usd_equivalent"] == "0.0006"


async def test_daily_timezone_dst_and_half_open_filters(repository):
    first = int(datetime(2026, 10, 25, 0, 30, tzinfo=UTC).timestamp() * 1000)
    second = int(datetime(2026, 10, 25, 1, 30, tzinfo=UTC).timestamp() * 1000)
    await repository.record(usage(occurred_at=first, input_tokens=5))
    await repository.record(
        usage(source_key="second", fact_key="second", occurred_at=second, input_tokens=7)
    )
    result = await repository.overview(UsageFilters(timezone="Europe/Paris"))
    assert len(result["timeseries"]) == 1
    assert result["timeseries"][0]["date"] == "2026-10-25"
    assert result["timeseries"][0]["input_tokens"] == 12
    result = await repository.overview(UsageFilters(from_ms=first, to_ms=second))
    assert result["summary"]["input_tokens"] == 5


@pytest.mark.parametrize(
    "fields",
    [
        {"input_tokens": -1},
        {"input_tokens": True},
        {"input_tokens": 1.5},
        {"kind": "cumulative", "epoch": None},
        {"baseline": True},
        {"reported_cost_usd": Decimal("NaN")},
    ],
)
async def test_invalid_observations_never_change_revision(repository, fields):
    with pytest.raises(ValueError):
        await repository.record(usage(**fields))
    assert await repository.revision() == 0


async def test_separate_connections_share_atomic_revision(tmp_path):
    db1, db2 = AiosqliteDatabase(), AiosqliteDatabase()
    await db1.connect(tmp_path / "shared.sqlite")
    await db1.migrate()
    await migrate_usage(db1._require_connection())
    await db2.connect(tmp_path / "shared.sqlite")
    try:
        first, second = UsageRepository(db1), UsageRepository(db2)
        await asyncio.gather(
            *[
                repo.record(usage(source_key=str(n), fact_key=str(n), input_tokens=1))
                for n in range(20)
                for repo in (first, second)
            ]
        )
        snapshot = await second.overview()
        assert snapshot["revision"] == 20
        assert snapshot["summary"]["input_tokens"] == 20
    finally:
        await db1.close()
        await db2.close()


async def test_skipped_invalid_batch_commits_gap_and_checkpoint(repository):
    await repository.record(usage(input_tokens=100))
    cursor = {"session_id": "s1", "offset": 10, "status": "ready"}
    invalid = usage(sequence=1, input_tokens=50)
    valid = usage(source_key="next", fact_key="next", input_tokens=10)
    rev = await repository.record_batch([invalid, valid], "history", cursor, skip_invalid=True)
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 110
    assert result["sync_state"]["gap_count"] == 1
    assert result["sync_state"]["status"] == "partial"
    assert result["history_status"] == {"partial": 1}
    assert (await repository.read_cursor("history"))["offset"] == 10
    assert (
        await repository.record_batch([invalid, valid], "history", cursor, skip_invalid=True) == rev
    )
    await repository.erase_session("s1")
    assert (await repository.overview())["sync_state"]["gap_count"] == 0


async def test_freshness_uses_observation_time_and_selected_scope(repository):
    await repository.record(usage(observed_at=100))
    await repository.record(
        usage(session_id="s2", source_key="other", fact_key="other", observed_at=200)
    )
    result = await repository.overview(UsageFilters(session_id="s1"))
    assert result["last_observed_at"] == 100
    assert result["as_of"] > 200
    assert (await repository.overview())["last_observed_at"] == 200


async def test_root_attributed_session_group_and_deleted_filter(repository, store):
    await store.execute(
        "INSERT INTO session (id,harness,project_path,created_at,updated_at,"
        "state,last_synced_at,deleted) VALUES ('root','codex','p',1,1,'stopped',1,1)"
    )
    await repository.record(usage(session_id=None, root_session_id="root", input_tokens=10))
    result = await repository.overview(UsageFilters(group_by="session"))
    assert result["breakdown"][0]["key"] == "root"
    assert result["breakdown"][0]["deleted"] is True
    result = await repository.overview(UsageFilters(include_deleted=False))
    assert result["summary"]["fact_count"] == 0


async def test_stale_cumulative_same_key_different_payload_ignored(repository):
    latest = usage(kind="cumulative", epoch="lifetime", sequence=10, input_tokens=100)
    await repository.record(latest)
    assert await repository.record(replace(latest, observed_at=5000)) == 1
    assert await repository.record(replace(latest, sequence=9, input_tokens=90)) == 1


async def test_unchanged_cumulative_snapshots_update_baseline_without_extra_facts(
    repository, store
):
    empty = usage(kind="cumulative", epoch="lifetime", sequence=1, input_tokens=0, output_tokens=0)
    await repository.record(empty)
    assert (await repository.overview())["summary"]["fact_count"] == 0
    consumed = replace(empty, sequence=2, occurred_at=2000, input_tokens=20, output_tokens=10)
    await repository.record(consumed)
    await repository.record(replace(consumed, sequence=3, occurred_at=3000, observed_at=4000))
    result = await repository.overview()
    assert result["summary"]["fact_count"] == 1
    assert result["summary"]["input_tokens"] == 20
    row = await store.fetch_one("SELECT payload FROM usage_baseline")
    assert '"sequence":3' in row["payload"]


async def test_root_only_erasure_preserves_distinct_child_facts(repository):
    root = usage(session_id=None, root_session_id="root", input_tokens=10)
    child = usage(
        source_key="child",
        fact_key="child",
        session_id="child",
        root_session_id="root",
        input_tokens=5,
    )
    await repository.record(root)
    await repository.record(child)
    rev = await repository.erase_session("root")
    assert await repository.record(replace(root, sequence=1, input_tokens=20)) == rev
    assert (await repository.overview())["summary"]["input_tokens"] == 5
    await repository.record(replace(child, sequence=1, input_tokens=7))
    assert (await repository.overview())["summary"]["input_tokens"] == 7


async def test_two_sources_shared_fact_identity_count_once(repository):
    await repository.record(usage(input_tokens=10))
    await repository.record(usage(source="history", source_key="history", input_tokens=10))
    assert (await repository.overview())["summary"]["input_tokens"] == 10


async def test_native_observed_model_is_not_aggregate_model_allocation(repository):
    value = usage(
        source="native:claude", model=None, observed_model="current-model", input_tokens=100
    )
    await repository.record(value)
    await repository.record(
        replace(value, sequence=1, input_tokens=150, observed_model="new-current-model")
    )
    result = await repository.overview()
    assert result["breakdown"][0]["key"] is None
    assert result["summary"]["usd_equivalent"] is None
    cumulative = usage(
        source="native:codex",
        source_key="total",
        kind="cumulative",
        epoch="mixed",
        model=None,
        observed_model="current",
        input_tokens=100,
    )
    await repository.record(cumulative)
    await repository.record(
        replace(cumulative, sequence=1, input_tokens=200, occurred_at=2000, observed_model="other")
    )
    assert all(row["key"] is None for row in (await repository.overview())["breakdown"])


async def test_native_gateway_native_overlap_advances_baseline_then_recovers(repository):
    native = usage(
        source="native:codex",
        source_key="native",
        fact_key="native",
        kind="cumulative",
        epoch="life",
        input_tokens=100,
        occurred_at=1000,
    )
    await repository.record(native)
    await repository.record(
        usage(occurred_at=2000, observed_at=2100, complete=True, input_tokens=20)
    )
    assert await repository.has_gateway_usage("s1")
    await repository.record(replace(native, sequence=1, occurred_at=3000, input_tokens=130))
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 120
    assert result["sync_state"]["gap_count"] == 1
    await repository.record(replace(native, sequence=2, occurred_at=4000, input_tokens=140))
    assert (await repository.overview())["summary"]["input_tokens"] == 130


async def test_new_proven_native_epoch_after_gateway_can_contribute(repository):
    await repository.record(
        usage(occurred_at=1000, observed_at=1100, complete=True, input_tokens=20)
    )
    native = usage(
        source="native:claude",
        source_key="native",
        fact_key="native",
        kind="cumulative",
        epoch="new-process",
        input_tokens=0,
        occurred_at=2000,
        baseline=True,
    )
    await repository.record(native)
    await repository.record(
        replace(native, sequence=1, occurred_at=3000, baseline=False, input_tokens=10)
    )
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 30
    assert result["sync_state"]["gap_count"] == 0


async def test_retrospective_gateway_suppresses_overlapping_native_interval(repository):
    native = usage(
        source="native:codex",
        source_key="native",
        fact_key="native",
        kind="cumulative",
        epoch="life",
        input_tokens=100,
        occurred_at=1000,
    )
    await repository.record(native)
    await repository.record(replace(native, sequence=1, occurred_at=3000, input_tokens=130))
    await repository.record(
        usage(occurred_at=2000, observed_at=2100, complete=True, input_tokens=20)
    )
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 120
    assert result["sync_state"]["gap_count"] == 1
    await repository.record(replace(native, sequence=2, occurred_at=4000, input_tokens=140))
    assert (await repository.overview())["summary"]["input_tokens"] == 130


async def test_initial_native_lifetime_after_gateway_is_excluded(repository):
    await repository.record(
        usage(
            session_id=None, root_session_id="s1", input_tokens=20, observed_at=1100, complete=True
        )
    )
    await repository.record(
        usage(
            source="native:codex",
            source_key="native",
            fact_key="native",
            kind="cumulative",
            epoch="life",
            input_tokens=100,
            occurred_at=2000,
        )
    )
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 20
    assert result["sync_state"]["gap_count"] == 1


async def test_same_millisecond_distinct_cumulative_records_keep_increments(repository, store):
    first = usage(
        source="native:codex",
        source_key="line:1",
        fact_key="series",
        kind="cumulative",
        epoch="history",
        sequence=1000,
        occurred_at=1000,
        input_tokens=100,
        output_tokens=10,
    )
    second = replace(first, source_key="line:2", input_tokens=125, output_tokens=15)
    third = replace(first, source_key="line:3", input_tokens=140, output_tokens=20)
    checkpoint = {"session_id": "s1", "offset": 300, "status": "complete"}
    revision = await repository.record_batch([first, second, third], "history", checkpoint)
    result = await repository.overview()
    assert result["summary"]["input_tokens"] == 140
    assert result["summary"]["output_tokens"] == 20
    assert result["summary"]["fact_count"] == 3
    assert (await store.fetch_one("SELECT count(DISTINCT fact_key) n FROM usage_fact"))["n"] == 3
    assert await repository.record_batch([first, second, third], "history", checkpoint) == revision
    assert (await repository.overview())["summary"] == result["summary"]


@pytest.mark.parametrize(
    "change",
    [
        {"sequence": 999, "input_tokens": 200},
        {"input_tokens": 90},
        {"input_tokens": 110, "output_tokens": 5},
        {"input_tokens": 110, "output_tokens": None},
        {"input_tokens": 110, "occurred_at": 1001},
        {"input_tokens": 110, "occurred_at": None},
        {},
    ],
)
async def test_same_sequence_requires_timestamp_and_strict_counter_dominance(repository, change):
    first = usage(
        source="native:codex",
        source_key="line:1",
        kind="cumulative",
        epoch="history",
        sequence=1000,
        occurred_at=1000,
        input_tokens=100,
        output_tokens=10,
    )
    revision = await repository.record(first)
    assert await repository.record(replace(first, source_key="line:2", **change)) == revision
    assert (await repository.overview())["summary"]["input_tokens"] == 100


async def test_same_millisecond_after_zero_baseline_and_later_timestamp(repository):
    baseline = usage(
        source="native:codex",
        source_key="start",
        kind="cumulative",
        epoch="history",
        sequence=1000,
        occurred_at=1000,
        input_tokens=0,
        baseline=True,
    )
    await repository.record(baseline)
    first = replace(baseline, source_key="line:1", baseline=False, input_tokens=10)
    second = replace(first, source_key="line:2", input_tokens=20)
    await repository.record(first)
    await repository.record(second)
    later = replace(second, source_key="line:3", sequence=1001, occurred_at=1001, input_tokens=30)
    revision = await repository.record(later)
    assert await repository.record(replace(first, input_tokens=100)) == revision
    assert (await repository.overview())["summary"]["input_tokens"] == 30
