import json
from types import SimpleNamespace

import pytest
from mandri.core.ids import HarnessSessionId
from mandri.runtime.control.errors import ControlTransportError
from mandri.runtime.pi_session_paths import (
    PiSessionCheckpoint,
    docker_resume_args,
    record_session_path,
)
from mandri.sessions.pi_store import PiSessionStore


def docker_context(tmp_path):
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    state.mkdir()
    workspace.mkdir()
    return {
        "native_state_root": str(state),
        "workspace_root": str(workspace),
        "native_home": "/home/worker",
        "container_root": "/workspace",
    }


@pytest.mark.parametrize("directory", ["/home/worker/custom", "/workspace/custom"])
def test_pi_docker_resume_uses_persisted_native_path(tmp_path, directory):
    context = docker_context(tmp_path)
    process = SimpleNamespace(execution_context=context)
    native_path = directory + "/arbitrary.jsonl"
    record_session_path(process, HarnessSessionId("native-session"), native_path)
    root = "native_state_root" if directory.startswith("/home") else "workspace_root"
    path = tmp_path / ("state" if root == "native_state_root" else "workspace")
    path = path / "custom/arbitrary.jsonl"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "type": "session",
                "id": "native-session",
                "version": 3,
                "cwd": "/workspace",
                "timestamp": "2026-09-25T12:00:00Z",
            }
        )
        + "\n"
    )
    args = ["pi", "--mode", "rpc", "--session", "native-session"]
    assert docker_resume_args(args, context) == [*args[:-1], native_path]
    assert args[-1] == "native-session"


@pytest.mark.parametrize(
    "value", ["relative.jsonl", "/tmp/session.jsonl", "/workspace/../outside.jsonl"]
)
def test_pi_docker_registration_requires_persistent_storage(tmp_path, value):
    process = SimpleNamespace(execution_context=docker_context(tmp_path))
    with pytest.raises(ControlTransportError):
        record_session_path(process, HarnessSessionId("native-session"), value)


def test_pi_docker_registration_rejects_workspace_symlink_escape(tmp_path):
    context = docker_context(tmp_path)
    (tmp_path / "workspace/custom").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ControlTransportError):
        record_session_path(
            SimpleNamespace(execution_context=context),
            HarnessSessionId("native-session"),
            "/workspace/custom/outside.jsonl",
        )


def test_pi_host_registration_preserves_custom_profile_and_pending_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "profile"))
    path = tmp_path / "custom/session.jsonl"
    record_session_path(SimpleNamespace(), HarnessSessionId("native-session"), str(path))
    path.parent.mkdir()
    path.write_text(
        json.dumps({"type": "session", "id": "native-session", "cwd": "/workspace"}) + "\n"
    )
    assert PiSessionStore().resolve("native-session") == path


@pytest.mark.parametrize("existing", [False, True])
def test_pi_checkpoint_recovers_empty_session_without_overwriting_native_records(
    tmp_path, existing
):
    path = tmp_path / "session.jsonl"
    header = json.dumps({"type": "session", "id": "native-session", "version": 3}) + "\n"
    pending = tmp_path / "session.jsonl.mandri-pending"
    pending.write_text(header)
    complete = header + json.dumps({"type": "session_info", "name": "Native title"}) + "\n"
    if existing:
        path.write_text(complete)
    PiSessionCheckpoint(HarnessSessionId("native-session"), path).recover()
    assert path.read_text() == (complete if existing else header)
    assert not pending.exists()


def test_pi_checkpoint_rejects_mismatched_native_identity(tmp_path):
    path = tmp_path / "session.jsonl"
    (tmp_path / "session.jsonl.mandri-pending").write_text(
        json.dumps({"type": "session", "id": "other-session"}) + "\n"
    )
    with pytest.raises(ControlTransportError, match="identity"):
        PiSessionCheckpoint(HarnessSessionId("native-session"), path).recover()
    assert not path.exists()


def test_pi_checkpoint_release_discards_transient_branch_for_external_resume(tmp_path):
    path = tmp_path / "session.jsonl"
    leaf = tmp_path / "session.jsonl.mandri-leaf"
    leaf.write_text("{}")
    PiSessionCheckpoint(HarnessSessionId("native-session"), path).recover()
    assert not leaf.exists()
