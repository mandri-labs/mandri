import json
from pathlib import Path
from typing import Any

import pytest
from mandri.runtime.agy_launch import prepare_agy_launch
from mandri.runtime.errors import SessionNotResumableError
from mandri.runtime.launch_preparation import PreparedLaunch
from mandri.sessions.agy_lease import AgyConversationLease
from mandri.sessions.agy_profiles import read_agy_json, write_agy_json
from mandri.sessions.errors import SessionConflictError, SessionRunningError


async def publish(event: dict[str, Any]) -> None:
    pass


def prepare(tmp_path: Path, resume: str | None = None, mode: str = "default"):
    argv = ["agy", "--conversation", resume] if resume else ["agy"]
    return prepare_agy_launch(
        PreparedLaunch(argv, {}, None),
        "managed",
        tmp_path,
        tmp_path / "profiles",
        tmp_path / "native",
        False,
        "synthetic/model",
        mode,
        1234,
        1,
        publish,
    )


async def test_resumed_launch_holds_lease_until_resource_closes(tmp_path: Path) -> None:
    store = tmp_path / "native/antigravity-cli/conversations"
    store.mkdir(parents=True)
    (store / "existing.db").touch()
    prepared, resource = prepare(tmp_path, "existing")
    contender = AgyConversationLease(tmp_path / "native", "existing")
    with pytest.raises(SessionRunningError):
        contender.acquire()
    assert "--gemini_dir" in prepared.argv
    resource.bind("existing")
    await resource.aclose()
    contender.acquire()
    contender.release()
    assert "mandri" not in read_agy_json(resource.profile / "config/hooks.json")


async def test_new_launch_binds_identity_and_rejects_replacement(tmp_path: Path) -> None:
    _, resource = prepare(tmp_path)
    resource.bind("new")
    with pytest.raises(SessionConflictError, match="different conversation"):
        resource.bind("different")
    assert read_agy_json(resource.profile / "mandri-session.json")["native_id"] == "new"
    assert read_agy_json(resource.profile / "mandri-session.json")["is_mandri_root"] is True
    await resource.aclose()


async def test_approval_hooks_are_journaled_before_forwarding(tmp_path: Path) -> None:
    _, resource = prepare(tmp_path)
    await resource.bridge.handle("Stop", {"conversationId": "new", "fullyIdle": False})
    record = json.loads((resource.profile / "mandri-events.jsonl").read_text())
    assert record["hook"] == "Stop"
    assert record["data"]["fullyIdle"] is False
    await resource.aclose()


async def test_second_launch_cannot_change_first_profile(tmp_path: Path) -> None:
    store = tmp_path / "native/antigravity-cli/conversations"
    store.mkdir(parents=True)
    (store / "existing.db").touch()
    _, resource = prepare(tmp_path, "existing")
    original = (resource.profile / "config/hooks.json").read_text()
    with pytest.raises(SessionRunningError):
        prepare(tmp_path, "existing")
    assert (resource.profile / "config/hooks.json").read_text() == original
    await resource.aclose()


def test_missing_conversation_is_rejected_before_profile_creation(tmp_path: Path) -> None:
    with pytest.raises(SessionNotResumableError, match="no longer exists"):
        prepare(tmp_path, "missing")
    assert not (tmp_path / "profiles").exists()
    assert not (tmp_path / "native/antigravity-cli/conversations/missing.db").exists()
    lease = AgyConversationLease(tmp_path / "native", "missing")
    lease.acquire()
    lease.release()


async def test_managed_root_provenance_survives_resume_of_same_native_identity(
    tmp_path: Path,
) -> None:
    _, resource = prepare(tmp_path)
    resource.bind("new")
    (tmp_path / "native/antigravity-cli/conversations/new.db").touch()
    await resource.aclose()
    _, resumed = prepare(tmp_path, "new")
    assert read_agy_json(resumed.profile / "mandri-session.json")["is_mandri_root"] is True
    assert read_agy_json(resumed.profile / "mandri-session.json")["history_pending"] is False
    await resumed.aclose()


async def test_resuming_external_conversation_does_not_inherit_another_root_flag(
    tmp_path: Path,
) -> None:
    store = tmp_path / "native/antigravity-cli/conversations"
    store.mkdir(parents=True)
    (store / "external.db").touch()
    write_agy_json(
        tmp_path / "profiles/managed/mandri-session.json",
        {
            "native_id": "different",
            "is_mandri_root": True,
        },
    )
    _, resource = prepare(tmp_path, "external")
    assert read_agy_json(resource.profile / "mandri-session.json")["is_mandri_root"] is False
    await resource.aclose()


async def test_first_root_step_closes_initial_empty_history_window(tmp_path: Path) -> None:
    _, resource = prepare(tmp_path)
    resource.bind("new")
    resource.record(
        {"event": "step_update", "step_update": {"conversation_id": "child", "state": "ACTIVE"}}
    )
    assert read_agy_json(resource.profile / "mandri-session.json")["history_pending"] is True
    resource.record(
        {"event": "step_update", "step_update": {"conversation_id": "new", "state": "DONE"}}
    )
    assert read_agy_json(resource.profile / "mandri-session.json")["history_pending"] is False
    await resource.aclose()


@pytest.mark.parametrize("mode,native", [("acceptEdits", "accept-edits"), ("plan", "plan")])
async def test_permission_restart_passes_native_mode_and_hook_policy(tmp_path, mode, native):
    prepared, resource = prepare(tmp_path, mode=mode)
    assert prepared.argv[-2:] == ["--mode", native]
    assert resource.bridge.policy.mode == mode
    await resource.aclose()
