import json
import os
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from mandri.core.ids import HarnessKind, SessionId, SessionState
from mandri.core.types.execution import ExecutionBackend, ProtectionError
from mandri.core.types.sessions import Session
from mandri.sessions import fork_lineage
from mandri.sessions.fork_source import selected_codex_source
from mandri.sessions.ownership.file_lock import try_lock, unlock, writer_locked
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader

from .substitutes import make_session

BASE = "019b0000-0000-7000-8000-000000000001"
PARENT = "019b0000-0000-7000-8000-000000000002"
SELECTED = "019b0000-0000-7000-8000-000000000003"


def rollout(home: Path, native_id: str) -> Path:
    return home / "sessions/2026/10/07" / f"rollout-2026-10-07T10-00-00-{native_id}.jsonl"


def record(row: dict) -> bytes:
    return (json.dumps(row) + "\n").encode()


def metadata(native_id: str, ordinal: int, base: dict | None = None) -> dict:
    return {
        "type": "session_meta",
        "ordinal": ordinal,
        "payload": {
            "id": native_id,
            "session_id": native_id,
            "cli_version": "0.161.0",
            "cwd": "/workspace",
            "history_mode": "paginated",
            "history_base": base,
        },
    }


def message(ordinal: int, text: str) -> dict:
    return {
        "type": "response_item",
        "ordinal": ordinal,
        "payload": {"type": "message", "role": "user", "content": text},
    }


def position(native_id: str, ordinal: int, length: int) -> dict:
    return {"thread_id": native_id, "end_ordinal_exclusive": ordinal, "end_byte_offset": length}


@dataclass
class Bundle:
    session: Session
    home: Path
    paths: dict[str, Path]
    prefixes: dict[str, bytes]
    destination: Path

    def reader(self) -> CodexTranscriptReader:
        return CodexTranscriptReader(self.home / "sessions")


@pytest.fixture
def bundle(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "native/session-1"
    home = state / ".codex"
    paths = {native_id: rollout(home, native_id) for native_id in (BASE, PARENT, SELECTED)}
    paths[BASE].parent.mkdir(parents=True)
    base = record(metadata(BASE, 0)) + record(message(1, "Inherited base message"))
    paths[BASE].write_bytes(base + record(message(2, "PRIVATE_BASE_FUTURE")))
    parent = record(metadata(PARENT, 2, position(BASE, 2, len(base)))) + record(
        message(3, "Inherited parent message" + "x" * 16384)
    )
    paths[PARENT].write_bytes(parent + record(message(4, "PRIVATE_PARENT_FUTURE")))
    selected = record(metadata(SELECTED, 4, position(PARENT, 4, len(parent)))) + record(
        message(5, "Selected message")
    )
    paths[SELECTED].write_bytes(selected)
    (home / "auth.json").write_text('{"key":"PRIVATE_AUTH"}')
    with sqlite3.connect(home / "state_5.sqlite") as database:
        database.execute("CREATE TABLE credentials (secret TEXT)")
        database.execute("INSERT INTO credentials VALUES ('PRIVATE_DATABASE')")
    session = replace(
        make_session(SELECTED, harness=HarnessKind.CODEX, project_path=str(workspace)),
        id=SessionId("session-1"),
        state=SessionState.STOPPED,
        execution_backend=ExecutionBackend.DOCKER,
        execution_context=json.dumps(
            {
                "version": "1",
                "workspace_root": str(workspace),
                "workspace_device": str(workspace.stat().st_dev),
                "workspace_inode": str(workspace.stat().st_ino),
                "container_root": "/workspace",
                "native_state_root": str(state),
                "native_home": "/home/worker",
            }
        ),
    )
    destination = rollout(tmp_path / "target/.codex", SELECTED)
    destination.parent.mkdir(parents=True)
    return Bundle(
        session, home, paths, {BASE: base, PARENT: parent, SELECTED: selected}, destination
    )


def test_recursive_lineage_copies_only_inherited_prefixes_and_selected_bytes(bundle):
    originals = {native_id: path.read_bytes() for native_id, path in bundle.paths.items()}
    with selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()) as source:
        source.copy_rollout(bundle.destination)
        for native_id in (BASE, PARENT, SELECTED):
            target = bundle.destination.with_name(bundle.paths[native_id].name)
            assert target.read_bytes() == bundle.prefixes[native_id]
            if os.name != "nt":
                assert target.stat().st_mode & 0o777 == 0o600
        assert writer_locked(bundle.home / "thread-writer-locks" / f"{BASE}.lock")
    files = list(bundle.destination.parents[3].rglob("*"))
    assert sorted(path.name for path in files if path.is_file()) == sorted(
        path.name for path in bundle.paths.values()
    )
    assert all(
        path.read_bytes() == originals[native_id] for native_id, path in bundle.paths.items()
    )
    assert not writer_locked(bundle.home / "thread-writer-locks" / f"{BASE}.lock")


def test_archived_ancestor_is_staged_under_standard_sessions_date(bundle):
    archived = bundle.home / "archived_sessions" / bundle.paths[BASE].name
    archived.parent.mkdir()
    bundle.paths[BASE].rename(archived)
    with selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()) as source:
        source.copy_rollout(bundle.destination)
    assert bundle.destination.with_name(archived.name).read_bytes() == bundle.prefixes[BASE]
    assert not (bundle.destination.parents[3] / "archived_sessions").exists()


def test_reverted_rollouts_can_share_a_logical_id_and_keep_physical_lineage(bundle):
    base = bundle.prefixes[BASE]
    parent = record(metadata(BASE, 2, position(BASE, 2, len(base)))) + record(message(3, "Parent"))
    selected = record(metadata(BASE, 4, position(PARENT, 4, len(parent)))) + record(
        message(5, "Selected")
    )
    bundle.paths[PARENT].write_bytes(parent)
    bundle.paths[SELECTED].write_bytes(selected)
    bundle.session = replace(bundle.session, native_id=BASE)
    with sqlite3.connect(bundle.home / "state_5.sqlite") as database:
        database.execute("CREATE TABLE threads (id TEXT, rollout_path TEXT)")
        database.execute("INSERT INTO threads VALUES (?, ?)", (BASE, str(bundle.paths[SELECTED])))
    with selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()) as source:
        source.copy_rollout(bundle.destination)
    assert bundle.destination.read_bytes() == selected
    assert bundle.destination.with_name(bundle.paths[PARENT].name).read_bytes() == parent
    assert bundle.destination.with_name(bundle.paths[BASE].name).read_bytes() == base


@pytest.mark.parametrize(
    "field,value",
    [
        ("thread_id", "../auth"),
        ("thread_id", []),
        ("end_byte_offset", 0),
        ("end_byte_offset", -1),
        ("end_byte_offset", True),
        ("end_byte_offset", 1),
        ("end_ordinal_exclusive", 0),
        ("end_ordinal_exclusive", True),
        ("end_ordinal_exclusive", 3),
        ("end_ordinal_exclusive", 1 << 64),
    ],
)
def test_invalid_history_boundaries_fail_before_copy(bundle, field, value):
    rows = bundle.paths[SELECTED].read_bytes().splitlines(keepends=True)
    header = json.loads(rows[0])
    header["payload"]["history_base"][field] = value
    bundle.paths[SELECTED].write_bytes(record(header) + b"".join(rows[1:]))
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()),
    ):
        pytest.fail("Invalid native boundary was accepted")
    assert not bundle.destination.exists()


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_ancestor_resolution_rejects_missing_or_ambiguous_files(bundle, change):
    path = bundle.paths[BASE]
    if change == "missing":
        path.unlink()
    else:
        duplicate = bundle.home / "archived_sessions" / path.name
        duplicate.parent.mkdir()
        duplicate.write_bytes(path.read_bytes())
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()),
    ):
        pytest.fail("Unsafe native source was accepted")


@pytest.mark.parametrize("limit", ["MAX_DEPTH", "MAX_BYTES", "MAX_LOOKUP_ENTRIES"])
def test_lineage_resource_limits_fail_before_copy(bundle, monkeypatch, limit):
    value = sum(map(len, bundle.prefixes.values())) - 1 if limit == "MAX_BYTES" else 1
    monkeypatch.setattr(fork_lineage, limit, value)
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()),
    ):
        pytest.fail("Oversized native lineage was accepted")
    assert not bundle.destination.exists()


def test_native_ancestor_writer_blocks_admission(bundle):
    directory = bundle.home / "thread-writer-locks"
    directory.mkdir()
    lock = directory / f"{PARENT}.lock"
    with lock.open("a+b") as lease:
        assert try_lock(lease)
        try:
            with (
                pytest.raises(ProtectionError, match="writer"),
                selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()),
            ):
                pytest.fail("Live native ancestor was accepted")
        finally:
            unlock(lease)


def test_later_collision_rolls_back_created_files_and_preserves_existing_ancestor(bundle):
    retained = bundle.destination.with_name(bundle.paths[BASE].name)
    retained.write_bytes(b"Preexisting destination")
    with (
        selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()) as source,
        pytest.raises(FileExistsError),
    ):
        source.copy_rollout(bundle.destination)
    assert retained.read_bytes() == b"Preexisting destination"
    assert not bundle.destination.exists()
    assert not bundle.destination.with_name(bundle.paths[PARENT].name).exists()


def test_changed_ancestor_digest_rolls_back_even_when_stat_is_restored(bundle, monkeypatch):
    copy = fork_lineage.RolloutPrefix.copy

    def mutate(prefix, output):
        if prefix.path == bundle.paths[PARENT]:
            before = prefix.path.stat()
            prefix.path.write_bytes(prefix.path.read_bytes().replace(b"Inherited", b"Different"))
            os.utime(prefix.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        copy(prefix, output)

    monkeypatch.setattr(fork_lineage.RolloutPrefix, "copy", mutate)
    with (
        selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()) as source,
        pytest.raises(ProtectionError),
    ):
        source.copy_rollout(bundle.destination)
    assert not list(bundle.destination.parent.glob("*.jsonl"))


def test_selected_self_reference_is_rejected_before_copy(bundle):
    header = metadata(SELECTED, 6, position(SELECTED, 6, 100))
    bundle.paths[SELECTED].write_bytes(record(header) + record(message(7, "Selected")))
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()),
    ):
        pytest.fail("Cyclic native history was accepted")


def test_ancestor_metadata_ordinal_must_match_its_inherited_boundary(bundle):
    rows = bundle.paths[PARENT].read_bytes().splitlines(keepends=True)
    header = json.loads(rows[0])
    header["ordinal"] = 3
    bundle.paths[PARENT].write_bytes(record(header) + b"".join(rows[1:]))
    with (
        pytest.raises(ProtectionError),
        selected_codex_source(bundle.session, ExecutionBackend.DOCKER, bundle.reader()),
    ):
        pytest.fail("Inconsistent inherited ordinal was accepted")
