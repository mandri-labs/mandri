import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, SessionId
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.sessions.adapters.agy_sessions import AgySessionsAdapter
from mandri.sessions.adapters.codex_fetch_sessions import (
    CodexRolloutScanAdapter,
    CodexStateDbFetchAdapter,
)
from mandri.sessions.adapters.opencode_fetch_sessions import OpencodeSqliteFetchSessionsAdapter
from mandri.sessions.lineage import MAX_LINEAGE_DEPTH, SessionLineage, ordered_sessions
from mandri.sessions.service import SessionsService
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import (
    FakeBackend,
    FakeDatabase,
    FakeEngine,
    insert_session_row,
    make_session,
)
from mandri_sessions.tests.test_agy_store import conversation, summary
from mandri_sessions.tests.test_session_classification import seed_native_db


def native(identity, parent=None, kind=HarnessKind.CODEX):
    return replace(
        make_session(identity, harness=kind, model="native-engine-model"),
        parent_native_id=HarnessSessionId(parent) if parent else None,
    )


async def protected_parent(db, *, scope="scope-a", deleted=0, kind=HarnessKind.CODEX):
    await insert_session_row(
        db, "parent-mandri", harness=kind.value, native_id="parent", deleted=deleted
    )
    await db.execute(
        "UPDATE session SET privacy_mode='surrogate', privacy_scope_id=?,"
        " model='configured/provider-model', model_source='gateway' WHERE id='parent-mandri'",
        (scope,),
    )


@pytest.mark.parametrize("kind", [HarnessKind.CODEX, HarnessKind.OPENCODE, HarnessKind.AGY])
async def test_reverse_order_grandchildren_inherit_scope_before_discovery_insert(kind):
    db = await FakeDatabase.create()
    await protected_parent(db, kind=kind)
    backend = FakeBackend(
        [
            native("grandchild", "child", kind),
            native("child", "parent", kind),
            native("parent", kind=kind),
        ]
    )
    engine = SyncEngine(db, {kind: backend})
    await engine.sync()
    assert not engine.harness_state(kind).degraded
    records = {row["native_id"]: row for row in await db.fetch_all("SELECT * FROM session")}
    for identity in ("child", "grandchild"):
        assert records[identity]["privacy_mode"] == "surrogate"
        assert records[identity]["privacy_scope_id"] == "scope-a"
        assert records[identity]["model_source"] == "gateway"
        assert records[identity]["model"] == "configured/provider-model"
    assert records["child"]["parent_session_id"] == "parent-mandri"
    assert records["grandchild"]["parent_session_id"] == records["child"]["id"]
    resumed = await SessionsService(db, FakeEngine()).ensure_session_policy(
        SessionId(records["grandchild"]["id"])
    )
    assert resumed.privacy_mode is PrivacyMode.SURROGATE
    assert resumed.privacy_scope_id == "scope-a"


async def test_preexisting_plain_child_upgrades_and_no_metadata_loss_downgrades_it():
    db = await FakeDatabase.create()
    await protected_parent(db)
    await insert_session_row(db, "existing", harness="codex", native_id="child")
    backend = FakeBackend([native("child", "parent"), native("parent")])
    engine = SyncEngine(db, {HarnessKind.CODEX: backend})
    await engine.sync()
    backend.sessions = [native("child"), native("parent")]
    await engine.sync()
    child = await SessionsService(db, FakeEngine()).ensure_session_policy(SessionId("existing"))
    assert child.privacy_mode is PrivacyMode.SURROGATE
    assert child.parent_native_id == "parent"
    assert child.parent_session_id == "parent-mandri"
    assert child.model_source is ModelSource.GATEWAY


async def test_child_cannot_claim_a_pending_root_with_same_cwd_and_time():
    db = await FakeDatabase.create()
    await protected_parent(db)
    await insert_session_row(db, "pending-root", harness="codex")
    engine = SyncEngine(
        db, {HarnessKind.CODEX: FakeBackend([native("child", "parent"), native("parent")])}
    )
    await engine.sync()
    pending = await db.fetch_one("SELECT * FROM session WHERE id='pending-root'")
    assert pending["native_id"] is None
    child = await db.fetch_one("SELECT * FROM session WHERE native_id='child'")
    assert child["id"] != "pending-root"
    assert child["privacy_scope_id"] == "scope-a"


async def test_missing_protected_scope_blocks_child_insertion():
    db = await FakeDatabase.create()
    await protected_parent(db, scope=None)
    engine = SyncEngine(db, {HarnessKind.CODEX: FakeBackend([native("child", "parent")])})
    await engine.sync()
    error = engine.harness_state(HarnessKind.CODEX).last_sync_error
    assert isinstance(error, ProtectionError)
    assert error.code == "privacy_state_unavailable"
    assert await db.fetch_one("SELECT * FROM session WHERE native_id='child'") is None


async def test_orphan_discovery_stays_readable_but_cannot_resume_until_parent_is_found():
    db = await FakeDatabase.create()
    engine = SyncEngine(db, {HarnessKind.CODEX: FakeBackend([native("child", "parent")])})
    await engine.sync()
    service = SessionsService(db, FakeEngine())
    child = (await service.list_sessions())[0]
    with pytest.raises(ProtectionError, match="parent session is unavailable"):
        await service.ensure_session_policy(child.id)
    await protected_parent(db)
    restored = await service.ensure_session_policy(child.id)
    assert restored.privacy_scope_id == "scope-a"


async def test_retained_deleted_parent_and_native_rotation_keep_stable_identity():
    db = await FakeDatabase.create()
    await protected_parent(db)
    engine = SyncEngine(
        db, {HarnessKind.CODEX: FakeBackend([native("child", "parent"), native("parent")])}
    )
    await engine.sync()
    await db.execute(
        "UPDATE session SET deleted=1, native_id='rotated-parent' WHERE id='parent-mandri'"
    )
    child = await db.fetch_one("SELECT id FROM session WHERE native_id='child'")
    session = await SessionsService(db, FakeEngine()).ensure_session_policy(SessionId(child["id"]))
    assert session.privacy_scope_id == "scope-a"
    assert session.parent_session_id == "parent-mandri"


@pytest.mark.parametrize(
    "mutation",
    [
        "DELETE FROM session WHERE id='parent-mandri'",
        "UPDATE session SET privacy_scope_id=NULL WHERE id='parent-mandri'",
        "UPDATE session SET privacy_scope_id='different' WHERE id='parent-mandri'",
        "UPDATE session SET harness='claude' WHERE id='parent-mandri'",
    ],
)
async def test_lineage_failure_never_turns_child_plain(mutation):
    db = await FakeDatabase.create()
    await protected_parent(db)
    engine = SyncEngine(
        db, {HarnessKind.CODEX: FakeBackend([native("child", "parent"), native("parent")])}
    )
    await engine.sync()
    child = await db.fetch_one("SELECT * FROM session WHERE native_id='child'")
    await db.execute(mutation)
    with pytest.raises(ProtectionError):
        await SessionsService(db, FakeEngine()).ensure_session_policy(SessionId(child["id"]))
    stored = await db.fetch_one("SELECT * FROM session WHERE id=?", (child["id"],))
    assert stored["privacy_mode"] == "surrogate"
    assert stored["privacy_scope_id"] == "scope-a"


async def test_child_retains_stronger_policy_than_plain_parent():
    db = await FakeDatabase.create()
    await protected_parent(db)
    engine = SyncEngine(
        db, {HarnessKind.CODEX: FakeBackend([native("child", "parent"), native("parent")])}
    )
    await engine.sync()
    await db.execute(
        "UPDATE session SET privacy_mode='none', privacy_scope_id=NULL WHERE id='parent-mandri'"
    )
    child = await db.fetch_one("SELECT * FROM session WHERE native_id='child'")
    result = await SessionLineage(db).ensure(child["id"])
    assert result["privacy_scope_id"] == "scope-a"
    assert result["privacy_mode"] == "surrogate"


@pytest.mark.parametrize(
    "rows",
    [
        [native("self", "self")],
        [native("a", "b"), native("b", "a")],
        [native("same"), native("same")],
    ],
)
def test_cycles_and_ambiguous_native_rows_fail_closed(rows):
    with pytest.raises(ProtectionError):
        ordered_sessions(rows)


def test_lineage_depth_is_bounded_without_python_recursion():
    rows = [native(str(index), str(index + 1)) for index in range(MAX_LINEAGE_DEPTH + 1)]
    with pytest.raises(ProtectionError):
        ordered_sessions(rows)


async def test_parent_change_cannot_rebind_existing_child():
    db = await FakeDatabase.create()
    await protected_parent(db)
    backend = FakeBackend([native("child", "parent"), native("parent")])
    engine = SyncEngine(db, {HarnessKind.CODEX: backend})
    await engine.sync()
    backend.sessions = [native("child", "replacement")]
    await engine.sync()
    assert engine.harness_state(HarnessKind.CODEX).degraded
    child = await db.fetch_one("SELECT * FROM session WHERE native_id='child'")
    assert child["parent_native_id"] == "parent"
    assert child["privacy_scope_id"] == "scope-a"


@pytest.mark.parametrize("variant", ["subagent", "sub_agent", "subAgent"])
def test_codex_native_parent_is_captured_from_sqlite_and_rollout(tmp_path: Path, variant):
    path = tmp_path / "state.sqlite"
    seed_native_db(path, variant)
    rows = {row.native_id: row for row in CodexStateDbFetchAdapter(path).fetch()}
    assert rows["child"].parent_native_id == "root"
    identity = "10000000-0000-4000-8000-000000000001"
    (tmp_path / f"rollout-2026-09-11T12-00-00-{identity}.jsonl").write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": identity,
                    "source": {variant: {"thread_spawn": {"parent_thread_id": "root"}}},
                },
            }
        )
    )
    assert CodexRolloutScanAdapter(tmp_path).fetch()[0].parent_native_id == "root"


@pytest.mark.parametrize("has_parent_column", [True, False])
def test_opencode_optional_parent_metadata_is_read_only(tmp_path: Path, has_parent_column):
    path = tmp_path / "opencode.db"
    with sqlite3.connect(path) as db:
        suffix = ", parent_id TEXT" if has_parent_column else ""
        db.execute(
            "CREATE TABLE session (id TEXT, title TEXT, directory TEXT, time_created INT,"
            " time_updated INT, time_archived INT"
            + suffix
            + ")"
        )
        values = "'child','title','/project',1,2,NULL" + (",'root'" if has_parent_column else "")
        db.execute("INSERT INTO session VALUES (" + values + ")")
    row = OpencodeSqliteFetchSessionsAdapter(path).fetch()[0]
    assert row.parent_native_id == ("root" if has_parent_column else None)


def test_agy_parent_metadata_is_captured(tmp_path: Path):
    conversation(tmp_path, "child")
    summary(tmp_path, "child", "Child", "root")
    assert AgySessionsAdapter(tmp_path).fetch()[0].parent_native_id == "root"


async def test_host_inventory_does_not_claim_or_tombstone_docker_sessions():
    db = await FakeDatabase.create()
    await insert_session_row(db, "docker-pending", harness="codex")
    await insert_session_row(db, "docker-existing", harness="codex", native_id="docker-native")
    await db.execute("UPDATE session SET execution_backend='docker'")
    engine = SyncEngine(db, {HarnessKind.CODEX: FakeBackend([native("host-native")])})
    await engine.sync()
    rows = {row["id"]: row for row in await db.fetch_all("SELECT * FROM session")}
    assert rows["docker-pending"]["native_id"] is None
    assert rows["docker-existing"]["deleted"] == 0
    host = next(row for row in rows.values() if row["native_id"] == "host-native")
    assert host["execution_backend"] == "host"


async def test_persisted_cycles_and_cross_harness_links_fail_admission():
    db = await FakeDatabase.create()
    await insert_session_row(db, "a", harness="codex", native_id="a")
    await insert_session_row(db, "b", harness="codex", native_id="b")
    await db.execute("UPDATE session SET parent_native_id='b', parent_session_id='b' WHERE id='a'")
    await db.execute("UPDATE session SET parent_native_id='a', parent_session_id='a' WHERE id='b'")
    with pytest.raises(ProtectionError, match="lineage is invalid"):
        await SessionLineage(db).ensure("a")
    await db.execute("UPDATE session SET harness='claude' WHERE id='b'")
    with pytest.raises(ProtectionError, match="parent session is unavailable"):
        await SessionLineage(db).ensure("a")
