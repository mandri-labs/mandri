import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, SessionId, SessionTitle
from mandri.core.ports.transcripts import SessionRef
from mandri.core.types.availability import SessionOwner
from mandri.core.types.execution import ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.sessions.adapters import pi_sessions
from mandri.sessions.adapters.pi_sessions import PiSessionsAdapter
from mandri.sessions.errors import SessionDeleteError, SessionRenameError
from mandri.sessions.execution_context import DockerSessionContext
from mandri.sessions.native_activity import native_model, native_turn_busy
from mandri.sessions.ownership import processes
from mandri.sessions.ownership.service import NativeOwnership, inspect_owner
from mandri.sessions.pi_store import MAX_METADATA_RECORD_BYTES, PiSessionStore, default_pi_agent_dir
from mandri.sessions.transcripts.docker_readers import DockerPiReader
from mandri.sessions.transcripts.errors import PageTokenStaleError
from mandri.sessions.transcripts.pi_transcripts import PiTranscriptReader
from mandri.sessions.usage.adapter import to_usage_observation
from mandri.sessions.usage.history import read_usage_batch
from mandri.sessions.usage.normalize import normalize_native_usage
from mandri.sessions.usage.types import NativeUsageContext


def write_session(path, native_id="native", entries=(), **header):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "session",
            "version": 3,
            "id": native_id,
            "timestamp": "2026-09-01T12:00:00Z",
            "cwd": "/workspace",
            **header,
        },
        *entries,
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return records


def assistant(stop="stop"):
    return {
        "type": "message",
        "id": "assistant",
        "parentId": "user",
        "message": {
            "role": "assistant",
            "provider": "openrouter",
            "model": "test-model",
            "timestamp": 1788264010000,
            "content": [{"type": "text", "text": "Done"}],
            "stopReason": stop,
            "usage": {
                "input": 10,
                "output": 20,
                "cacheRead": 30,
                "cacheWrite": 40,
                "totalTokens": 100,
                "cost": {"total": 0.01},
            },
        },
    }


def test_discovers_names_models_effort_and_parent_without_cli(tmp_path):
    parent = tmp_path / "--workspace--" / "arbitrary-parent.jsonl"
    child = tmp_path / "--workspace--" / "child.jsonl"
    write_session(parent, "parent")
    write_session(
        child,
        entries=[
            {"type": "message", "message": {"role": "user", "content": "First prompt"}},
            {"type": "thinking_level_change", "thinkingLevel": "high"},
            assistant(),
            {"type": "session_info", "name": "Custom title", "id": "last"},
        ],
        parentSession=str(parent),
    )
    sessions = PiSessionsAdapter(tmp_path).fetch()
    session = next(item for item in sessions if item.native_id == "native")
    assert session.native_title == "Custom title"
    assert session.model == "openrouter/test-model"
    assert session.reasoning_effort == "high"
    assert session.model_source is ModelSource.NATIVE
    assert session.parent_native_id == "parent"
    assert session.project_path == "/workspace"
    session.validate()


def test_cache_reads_only_appended_records_and_reloads_rewrites(tmp_path, monkeypatch):
    path = tmp_path / "workspace" / "native.jsonl"
    records = write_session(path, entries=[assistant()])
    store = PiSessionStore(tmp_path)
    original = store._read
    offsets = []

    def tracked(handle, *args):
        offsets.append(handle.tell())
        return original(handle, *args)

    monkeypatch.setattr(store, "_read", tracked)
    first = store.fetch()
    assert store.fetch() == first
    size = path.stat().st_size
    with path.open("a") as handle:
        handle.write(json.dumps({"type": "session_info", "name": "Renamed"}) + "\n")
    assert store.fetch()[0].name == "Renamed"
    assert offsets == [0, size]
    write_session(path, "replacement", entries=records[1:])
    assert store.fetch()[0].native_id == "replacement"
    assert offsets[-1] == 0
    path.unlink()
    assert store.fetch() == []


def test_partial_record_completion_is_reparsed(tmp_path):
    path = tmp_path / "native.jsonl"
    write_session(path)
    with path.open("a") as handle:
        handle.write('{"type":"session_info","name":"New')
    store = PiSessionStore(tmp_path)
    assert store.fetch()[0].name is None
    with path.open("a") as handle:
        handle.write(' title"}\n')
    assert store.fetch()[0].name == "New title"


def test_discovery_skips_huge_custom_records_and_invalid_files(tmp_path):
    path = tmp_path / "native.jsonl"
    write_session(
        path,
        entries=[
            {"type": "custom", "data": "x" * (MAX_METADATA_RECORD_BYTES + 1)},
            {"type": "session_info", "name": "After large record"},
        ],
    )
    (tmp_path / "broken.jsonl").write_text('{"type":"message","id":"wrong"}\n')
    (tmp_path / "traversal.jsonl").write_text('{"type":"session","id":"../bad"}\n')
    (tmp_path / "alias.jsonl").symlink_to(path)
    rows = PiSessionStore(tmp_path).fetch()
    assert len(rows) == 1
    assert rows[0].name == "After large record"


def test_custom_native_directories_follow_pi_environment(tmp_path, monkeypatch):
    home = tmp_path / "agent"
    external = tmp_path / "custom-sessions"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(home))
    monkeypatch.setenv("PI_CODING_AGENT_SESSION_DIR", str(external))
    write_session(home / "sessions/workspace/one.jsonl", "one")
    write_session(external / "two.jsonl", "two")
    assert default_pi_agent_dir() == home
    assert {row.native_id for row in PiSessionStore().fetch()} == {"one", "two"}
    assert {row.native_id for row in PiSessionStore(home / "sessions").fetch()} == {"one"}


def test_resolve_uses_header_id_and_validates_project(tmp_path):
    path = tmp_path / "folder" / "arbitrary.jsonl"
    write_session(path)
    store = PiSessionStore(tmp_path)
    assert store.resolve("native", "/workspace") == path
    assert store.resolve("native", "/other") is None
    with pytest.raises(ValueError):
        store.resolve("../../escape")


def test_custom_session_paths_survive_store_recreation_and_deferred_persistence(tmp_path):
    root = tmp_path / "agent/sessions"
    custom = tmp_path / "extension-selected/session.jsonl"
    store = PiSessionStore(root)
    store.register("native", custom)
    assert PiSessionStore(root).fetch() == []
    write_session(custom)
    fresh = PiSessionStore(root)
    assert fresh.resolve("native") == custom
    assert fresh.fetch()[0].path == custom
    assert custom.read_text().count('"type": "session"') == 1
    assert (root.parent / ".mandri-session-paths/native.json").is_file()


def test_index_rejects_foreign_headers_and_changed_files(tmp_path):
    root = tmp_path / "agent/sessions"
    custom = tmp_path / "custom/session.jsonl"
    write_session(custom)
    store = PiSessionStore(root)
    with pytest.raises(ValueError):
        store.register("wrong", custom)
    store.register("native", custom)
    assert store.resolve("native") == custom
    write_session(custom, "replacement")
    assert store.resolve("native") is None
    assert store.fetch() == []


def test_docker_path_index_is_validated_before_reading_external_files(tmp_path, monkeypatch):
    state = tmp_path / "state/session"
    state.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "private/native.jsonl"
    write_session(outside)
    PiSessionStore(state / ".pi/agent/sessions").register("native", outside)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path != outside, "External transcript must not be read"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    context = DockerSessionContext(
        state, PurePosixPath("/home/worker"), workspace, PurePosixPath("/workspace")
    )
    reference = SessionRef(HarnessKind.PI, HarnessSessionId("native"))
    with pytest.raises(ProtectionError):
        DockerPiReader(context).page(reference, None, 10)


def test_docker_extension_session_in_workspace_can_be_read_and_cannot_escape(tmp_path):
    state = tmp_path / "state/session"
    state.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    native = workspace / ".custom-pi/session.jsonl"
    write_session(native, entries=[assistant()])
    PiSessionStore(state / ".pi/agent/sessions").register("native", native)
    context = DockerSessionContext(
        state, PurePosixPath("/home/worker"), workspace, PurePosixPath("/workspace")
    )
    reference = SessionRef(HarnessKind.PI, HarnessSessionId("native"))
    reader = DockerPiReader(context)
    assert len(reader.page(reference, None, 10).entries) == 2
    outside = tmp_path / "private.jsonl"
    write_session(outside)
    native.unlink()
    native.symlink_to(outside)
    with pytest.raises(ProtectionError):
        reader.page(reference, None, 10)


def test_rename_retains_oversized_extension_as_parent(tmp_path, monkeypatch):
    path = tmp_path / "native.jsonl"
    write_session(
        path,
        entries=[
            {
                "type": "custom",
                "data": "x" * (MAX_METADATA_RECORD_BYTES + 1),
                "id": "extension",
                "parentId": None,
            }
        ],
    )
    monkeypatch.setattr(
        pi_sessions, "inspect_owner", lambda *args: NativeOwnership(SessionOwner.UNOWNED)
    )
    PiSessionsAdapter(tmp_path).rename(SessionId("native"), SessionTitle("Name"))
    assert json.loads(path.read_text().splitlines()[-1])["parentId"] == "extension"


def test_rename_appends_metadata_preserving_extension_state_and_tree(tmp_path, monkeypatch):
    path = tmp_path / "native.jsonl"
    write_session(path, entries=[{"type": "custom", "id": "extension", "data": {"x": 1}}])
    original = path.read_bytes().rstrip(b"\n")
    path.write_bytes(original)
    monkeypatch.setattr(
        pi_sessions, "inspect_owner", lambda *args: NativeOwnership(SessionOwner.UNOWNED)
    )
    adapter = PiSessionsAdapter(tmp_path)
    adapter.rename(SessionId("native"), SessionTitle("New\nTitle"))
    raw = path.read_bytes()
    assert raw.startswith(original + b"\n")
    record = json.loads(raw.splitlines()[-1])
    assert record["parentId"] == "extension"
    assert record["name"] == "New Title"
    assert adapter.fetch()[0].native_title == "New Title"
    adapter.delete(SessionId("native"))
    assert not adapter.exists(SessionId("native"))


@pytest.mark.parametrize("owner", [SessionOwner.EXTERNAL, SessionOwner.UNKNOWN])
def test_native_writers_block_mutations(tmp_path, monkeypatch, owner):
    path = tmp_path / "native.jsonl"
    write_session(path)
    original = path.read_bytes()
    monkeypatch.setattr(pi_sessions, "inspect_owner", lambda *args: NativeOwnership(owner))
    adapter = PiSessionsAdapter(tmp_path)
    with pytest.raises(SessionRenameError):
        adapter.rename(SessionId("native"), SessionTitle("Blocked"))
    with pytest.raises(SessionDeleteError):
        adapter.delete(SessionId("native"))
    assert path.read_bytes() == original


def test_transcript_preserves_custom_events_and_native_branch_structure(tmp_path):
    path = tmp_path / "workspace" / "native.jsonl"
    events = write_session(
        path, entries=[assistant(), {"type": "custom", "id": "extension", "parentId": "assistant"}]
    )
    reference = SessionRef(HarnessKind.PI, HarnessSessionId("native"))
    reader = PiTranscriptReader(tmp_path)
    page = reader.page(reference, None, 2)
    assert [json.loads(raw) for raw in page.entries] == events[:2]
    second = reader.page(reference, page.next_token, 2)
    assert [json.loads(raw) for raw in second.entries] == events[2:]
    assert reader.status(reference) == (False, "openrouter/test-model")
    assert reader.revision(reference)[2] == path.stat().st_size


def test_native_branch_pages_exclude_abandoned_messages_and_follow_appends(tmp_path):
    path = tmp_path / "native.jsonl"
    common = {"type": "message", "id": "root", "parentId": None, "message": {"role": "user"}}
    abandoned = {**assistant(), "id": "abandoned", "parentId": "root"}
    branch = {**assistant(), "id": "branch", "parentId": "root"}
    write_session(path, entries=[common, abandoned, branch])
    reference = SessionRef(HarnessKind.PI, HarnessSessionId("native"))
    reader = PiTranscriptReader(tmp_path)
    page = reader.page(reference, None, 100)
    assert [json.loads(raw)["id"] for raw in page.entries] == ["native", "root", "branch"]
    recent = reader.recent(reference, None, 1)
    assert json.loads(recent.entries[0])["id"] == "branch"
    prior = reader.recent(reference, recent.next_token, 1)
    assert json.loads(prior.entries[0])["id"] == "root"
    with path.open("a") as handle:
        handle.write(json.dumps({**assistant(), "id": "new", "parentId": "branch"}) + "\n")
    assert json.loads(reader.recent(reference, None, 1).entries[0])["id"] == "new"


def write_leaf(path, leaf_id, **changes):
    stat = path.stat()
    pointer = path.with_name(path.name + ".mandri-leaf")
    pointer.write_text(
        json.dumps(
            {
                "sessionId": "native",
                "leafId": leaf_id,
                "size": str(stat.st_size),
                "mtimeNs": str(stat.st_mtime_ns),
                **changes,
            }
        )
    )
    return pointer


def branched_session(path):
    write_session(
        path,
        entries=[
            {"type": "message", "id": "root", "parentId": None, "message": {"role": "user"}},
            {
                "type": "thinking_level_change",
                "id": "effort-a",
                "parentId": "root",
                "thinkingLevel": "high",
            },
            {**assistant(), "id": "branch-a", "parentId": "effort-a"},
            {
                "type": "thinking_level_change",
                "id": "effort-b",
                "parentId": "root",
                "thinkingLevel": "low",
            },
            {**assistant(), "id": "branch-b", "parentId": "effort-b"},
        ],
    )


def test_managed_branch_pointer_selects_history_and_invalidates_shared_cursor(tmp_path):
    path = tmp_path / "native.jsonl"
    branched_session(path)
    reference = SessionRef(HarnessKind.PI, HarnessSessionId("native"))
    reader = PiTranscriptReader(tmp_path)
    old = reader.page(reference, None, 1)
    before = reader.revision(reference)
    assert reader.store.fetch()[0].thinking_level == "low"
    write_leaf(path, "branch-a")
    assert reader.revision(reference) != before
    assert [json.loads(raw)["id"] for raw in reader.page(reference, None, 20).entries] == [
        "native",
        "root",
        "effort-a",
        "branch-a",
    ]
    assert json.loads(reader.recent(reference, None, 1).entries[0])["id"] == "branch-a"
    assert reader.store.fetch()[0].thinking_level == "high"
    with pytest.raises(PageTokenStaleError):
        reader.page(reference, old.next_token, 1)
    write_leaf(path, None)
    assert [json.loads(raw)["id"] for raw in reader.page(reference, None, 20).entries] == ["native"]
    assert reader.store.fetch()[0].leaf_id is None
    assert reader.store.fetch()[0].thinking_level is None


@pytest.mark.parametrize(
    "changes",
    [{"sessionId": "other"}, {"size": "0"}, {"mtimeNs": "0"}, {"leafId": "absent"}],
)
def test_invalid_managed_branch_pointer_is_ignored(tmp_path, changes):
    path = tmp_path / "native.jsonl"
    branched_session(path)
    write_leaf(path, "branch-a", **changes)
    reference = SessionRef(HarnessKind.PI, HarnessSessionId("native"))
    reader = PiTranscriptReader(tmp_path)
    assert json.loads(reader.recent(reference, None, 1).entries[0])["id"] == "branch-b"
    assert reader.store.fetch()[0].thinking_level == "low"


def test_branch_pointer_expires_on_native_append_and_cursor_follows_same_branch(tmp_path):
    path = tmp_path / "native.jsonl"
    branched_session(path)
    write_leaf(path, "branch-a")
    reference = SessionRef(HarnessKind.PI, HarnessSessionId("native"))
    reader = PiTranscriptReader(tmp_path)
    page = reader.page(reference, None, 2)
    with path.open("a") as handle:
        handle.write(json.dumps({**assistant(), "id": "new", "parentId": "branch-a"}) + "\n")
    following = reader.page(reference, page.next_token, 20)
    assert [json.loads(raw)["id"] for raw in following.entries] == ["effort-a", "branch-a", "new"]


def test_large_extension_state_preserves_branch_ancestors_and_is_downloadable(tmp_path):
    path = tmp_path / "native.jsonl"
    extension = {
        "type": "custom",
        "data": "x" * (MAX_METADATA_RECORD_BYTES + 1),
        "id": "extension",
        "parentId": "assistant",
    }
    write_session(path, entries=[assistant(), extension])
    reference = SessionRef(HarnessKind.PI, HarnessSessionId("native"))
    reader = PiTranscriptReader(tmp_path)
    page = reader.page(reference, None, 10)
    assert json.loads(page.entries[1])["id"] == "assistant"
    placeholder = json.loads(page.entries[2])
    assert placeholder["type"] == "mandri.transcript_record"
    assert placeholder["byte_length"] > MAX_METADATA_RECORD_BYTES


@pytest.mark.parametrize(
    "stop,busy", [("stop", False), ("length", False), ("toolUse", True), ("aborted", False)]
)
def test_pi_turn_boundaries_and_model(stop, busy):
    entries = [json.dumps(assistant(stop))]
    assert native_turn_busy(HarnessKind.PI, entries) is busy
    assert native_model(entries) == "openrouter/test-model"


@pytest.mark.parametrize("selector", ["native", "nati", "/elsewhere/time_native.jsonl"])
def test_explicit_pi_session_is_locked_even_from_different_cwd(monkeypatch, selector):
    process = SimpleNamespace(
        pid=12,
        info={
            "name": "node",
            "cmdline": ["node", "/usr/local/bin/pi", "--session", selector],
            "cwd": "/elsewhere",
            "create_time": 1.0,
        },
    )
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: [process])
    owner = inspect_owner(HarnessKind.PI, "native", "/workspace")
    assert owner.owner is SessionOwner.EXTERNAL
    assert owner.reason == "external_release_unsupported"
    assert not owner.can_release


def test_pi_with_switchable_session_keeps_same_workspace_guarded(monkeypatch):
    process = SimpleNamespace(
        pid=12,
        info={
            "name": "pi",
            "cmdline": ["pi", "--session", "other"],
            "cwd": "/workspace",
            "create_time": 1.0,
        },
    )
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: [process])
    assert inspect_owner(HarnessKind.PI, "native", "/workspace").owner is SessionOwner.UNKNOWN


@pytest.mark.parametrize(
    ("known_id", "owner"),
    [
        ("other", SessionOwner.UNOWNED),
        ("native", SessionOwner.EXTERNAL),
        (None, SessionOwner.UNKNOWN),
    ],
)
def test_managed_pi_identity_disambiguates_same_workspace(monkeypatch, known_id, owner):
    process = SimpleNamespace(
        pid=12,
        info={
            "name": "pi",
            "cmdline": ["pi", "--mode", "rpc"],
            "cwd": "/workspace",
            "create_time": 1.0,
        },
    )
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: [process])
    assert (
        inspect_owner(HarnessKind.PI, "native", "/workspace", pi_processes={12: known_id}).owner
        is owner
    )


def test_known_pi_process_does_not_hide_another_unknown_writer(monkeypatch):
    writers = [
        SimpleNamespace(
            pid=pid,
            info={
                "name": "pi",
                "cmdline": ["pi", "--mode", "rpc"],
                "cwd": "/workspace",
                "create_time": 1.0,
            },
        )
        for pid in (12, 13)
    ]
    monkeypatch.setattr(processes.psutil, "process_iter", lambda *args, **kwargs: writers)
    assert (
        inspect_owner(HarnessKind.PI, "native", "/workspace", pi_processes={12: "other"}).owner
        is SessionOwner.UNKNOWN
    )


def test_pi_usage_live_and_history_share_identity_and_disjoint_tokens(tmp_path):
    context = NativeUsageContext(
        "session", "pi", "process", native_id="native", routing="native", inherited_history="none"
    )
    event = assistant()
    (live,) = normalize_native_usage(context, {**event, "type": "message_end"}, observed_at_ms=1)
    path = tmp_path / "usage.jsonl"
    write_session(path, entries=[event])
    batch = read_usage_batch(path, replace(context, process_epoch="history"), observed_at_ms=2)
    (history,) = batch.observations
    assert live.source_key == history.source_key
    value = to_usage_observation(history, sequence=0)
    assert value.authoritative and value.complete
    assert value.total_tokens == 100
    assert value.input_tokens == 10 and value.cache_read_tokens == 30
    assert value.cache_write_tokens == 40 and value.output_tokens == 20
    assert value.request_count == 1
    assert value.model == "test-model"
    assert value.pricing_context["provider_kind"] == "openrouter"
    assert value.pricing_context["context_tokens"] == 80
    assert value.occurred_at == event["message"]["timestamp"]
    assert not value.input_includes_cache
    assert value.output_includes_reasoning


def test_pi_gateway_usage_and_inherited_history_are_not_double_counted():
    context = NativeUsageContext("session", "pi", "process", native_id="native", routing="gateway")
    (item,) = normalize_native_usage(context, assistant(), observed_at_ms=1)
    assert not to_usage_observation(item, sequence=0).authoritative
    fork = replace(
        context, routing="native", parent_session_id="parent", inherited_history="unknown"
    )
    (item,) = normalize_native_usage(fork, assistant(), observed_at_ms=1)
    assert not to_usage_observation(item, sequence=0).authoritative


@pytest.mark.parametrize("kind", ["usage", "compaction", "branch_summary"])
def test_pi_auxiliary_history_preserves_tokens_and_only_attributes_proven_models(tmp_path, kind):
    context = NativeUsageContext(
        "session", "pi", "history", native_id="native", routing="native", inherited_history="none"
    )
    entry = {
        "type": kind,
        "id": "auxiliary",
        "parentId": None,
        "timestamp": "2026-09-01T12:00:00Z",
        "usage": assistant()["message"]["usage"],
        **(
            {"provider": "openrouter", "model": "cache-model", "kind": "cache_warm"}
            if kind == "usage"
            else {}
        ),
    }
    path = tmp_path / "usage.jsonl"
    write_session(path, entries=[entry])
    (item,) = read_usage_batch(path, context, observed_at_ms=2).observations
    value = to_usage_observation(item, sequence=0)
    assert value.total_tokens == 100
    assert value.reported_cost_usd is not None
    assert value.authoritative is (kind == "usage")
    assert value.model == ("cache-model" if kind == "usage" else None)
    assert value.request_count is None
    assert not value.complete


def test_pi_tool_usage_is_preserved_without_claiming_ownership_of_descendants():
    context = NativeUsageContext(
        "session", "pi", "process", native_id="native", routing="native", inherited_history="none"
    )
    message = {
        "role": "toolResult",
        "toolCallId": "delegation",
        "timestamp": 1788264010000,
        "provider": "openrouter",
        "model": "child-model",
        "usage": assistant()["message"]["usage"],
    }
    (live,) = normalize_native_usage(
        context, {"type": "message_end", "message": message}, observed_at_ms=1
    )
    (history,) = normalize_native_usage(
        replace(context, process_epoch="history"),
        {"type": "message", "id": "entry", "message": message},
        observed_at_ms=2,
        origin="history",
    )
    assert live.source_key == history.source_key
    value = to_usage_observation(history, sequence=0)
    assert value.total_tokens == 100
    assert value.model == "child-model"
    assert not value.authoritative


def test_pi_gateway_provider_history_never_counts_again_as_native_usage():
    context = NativeUsageContext(
        "session", "pi", "history", native_id="native", routing="native", inherited_history="none"
    )
    event = assistant()
    event["message"]["provider"] = "mandri"
    (item,) = normalize_native_usage(context, event, observed_at_ms=1, origin="history")
    assert not to_usage_observation(item, sequence=0).authoritative


def test_discovery_does_not_launch_pi(tmp_path, monkeypatch):
    write_session(tmp_path / "native.jsonl")
    monkeypatch.setattr(
        processes.psutil, "process_iter", Mock(side_effect=AssertionError("process"))
    )
    assert len(PiSessionsAdapter(tmp_path).fetch()) == 1
