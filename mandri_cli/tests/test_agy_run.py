import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from mandri.cli import agy_run, run
from mandri.cli.agy_run import AgyRunProfile, close_agy_run, prepare_agy_run, wait_agy_process
from mandri.cli.run_errors import HarnessBinaryNotFoundError, RunError
from mandri.cli.types import RunSpec
from mandri.core.ids import HarnessKind
from mandri.core.types.availability import SessionOwner
from mandri.core.types.config import DaemonConfig, SessionsConfig
from mandri.sessions.agy_lease import AgyConversationLease
from mandri.sessions.agy_observe import observe_agy
from mandri.sessions.agy_profiles import read_agy_json, write_agy_json
from mandri.sessions.agy_store import agy_metadata
from mandri.sessions.ownership.service import NativeOwnership


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    canonical = tmp_path / "native"
    configuration = DaemonConfig(sessions=SessionsConfig(agy_home=str(canonical)))
    monkeypatch.setattr(agy_run, "TomlConfigAdapter", lambda _: Mock(load=lambda: configuration))
    return canonical


def test_cli_builds_private_profile_and_keeps_native_interaction(
    tmp_path: Path, config: Path
) -> None:
    spec = RunSpec(HarnessKind.AGY, "synthetic/model", tmp_path, cwd=tmp_path)
    profile = prepare_agy_run(spec)
    settings = read_agy_json(profile.root / "antigravity-cli/settings.json")
    assert settings["modelProvider"] == "gemini"
    assert settings["customModelsConfig"]["customModels"]["mandri"]["modelName"] == "mandri-route"
    assert "--print" not in profile.args
    assert "--dangerously-skip-permissions" not in profile.args
    hooks = read_agy_json(profile.root / "config/hooks.json")["mandri"]
    assert set(hooks) == {"PreInvocation", "PostInvocation", "Stop"}
    assert not (config / "antigravity-cli/settings.json").exists()


def test_cli_explicit_resume_keeps_identity(
    tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = config / "antigravity-cli/conversations/existing.db"
    database.parent.mkdir(parents=True)
    database.touch()
    monkeypatch.setattr(
        agy_run, "inspect_owner", lambda *args: NativeOwnership(SessionOwner.UNOWNED)
    )
    spec = RunSpec(
        HarnessKind.AGY, "synthetic/model", tmp_path, passthrough_args=("--conversation=existing",)
    )
    profile = prepare_agy_run(spec)
    assert profile.args[-2:] == ("--conversation", "existing")
    assert read_agy_json(profile.root / "mandri-session.json")["native_id"] == "existing"
    close_agy_run(profile)


def test_cli_missing_resume_does_not_create_replacement(tmp_path: Path, config: Path) -> None:
    spec = RunSpec(
        HarnessKind.AGY, "synthetic/model", tmp_path, passthrough_args=("--conversation", "missing")
    )
    with pytest.raises(RunError, match="not found"):
        prepare_agy_run(spec)
    assert not (tmp_path / "agy-profiles").exists()


def test_cli_refuses_active_native_writer(
    tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = config / "antigravity-cli/conversations/existing.db"
    database.parent.mkdir(parents=True)
    database.touch()
    monkeypatch.setattr(
        agy_run, "inspect_owner", lambda *args: NativeOwnership(SessionOwner.EXTERNAL)
    )
    spec = RunSpec(
        HarnessKind.AGY,
        "synthetic/model",
        tmp_path,
        passthrough_args=("--conversation", "existing"),
    )
    with pytest.raises(RunError, match="already has a writer"):
        prepare_agy_run(spec)


def test_cli_observer_preserves_models_across_conversations(tmp_path: Path) -> None:
    profile = tmp_path / "profiles/cli"
    write_agy_json(profile / "mandri-launch.json", {"cwd": "/work", "model": "synthetic/model"})
    observe_agy(profile, "PostInvocation", {"conversationId": "one", "modelName": "mandri"})
    observe_agy(profile, "Stop", {"conversationId": "two", "fullyIdle": True})
    metadata = agy_metadata(tmp_path / "native", tmp_path / "profiles")
    assert metadata["one"]["model"] == "synthetic/model"
    assert metadata["two"]["model"] == "synthetic/model"
    journal = (profile / "mandri-events.jsonl").read_text().splitlines()
    assert json.loads(journal[1])["fullyIdle"] is True


def test_cli_child_hook_cannot_replace_selected_conversation_or_claim_root(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    write_agy_json(
        profile / "mandri-launch.json",
        {
            "cwd": "/work",
            "model": "synthetic/model",
            "native_id": "root",
            "is_mandri_root": True,
        },
    )
    observe_agy(profile, "PreInvocation", {"conversationId": "root"})
    observe_agy(profile, "Stop", {"conversationId": "child", "fullyIdle": True})
    assert read_agy_json(profile / "mandri-session.json")["native_id"] == "root"
    assert read_agy_json(profile / "bindings/child.json").get("is_mandri_root") is not True
    assert read_agy_json(profile / "bindings/root.json")["is_mandri_root"] is True


def test_cli_new_identity_requires_native_workspace_cache_evidence(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    write_agy_json(
        profile / "mandri-launch.json",
        {
            "cwd": "/work",
            "model": "synthetic/model",
            "new_conversation": True,
        },
    )
    observe_agy(profile, "PreInvocation", {"conversationId": "child"})
    assert not (profile / "mandri-session.json").exists()
    write_agy_json(profile / "antigravity-cli/cache/last_conversations.json", {"/work": "root"})
    observe_agy(profile, "PreInvocation", {"conversationId": "root"})
    assert read_agy_json(profile / "mandri-session.json")["native_id"] == "root"
    assert read_agy_json(profile / "mandri-session.json")["is_mandri_root"] is True


def test_cli_lease_collision_becomes_clean_run_error(tmp_path: Path, config: Path) -> None:
    lease = AgyConversationLease(config, "existing")
    lease.acquire()
    spec = RunSpec(
        HarnessKind.AGY,
        "synthetic/model",
        tmp_path,
        passthrough_args=("--conversation", "existing"),
    )
    try:
        with pytest.raises(RunError, match="already has a Mandri writer"):
            prepare_agy_run(spec)
    finally:
        lease.release()


def test_cli_new_identity_lease_collision_stops_child_and_becomes_run_error(tmp_path: Path) -> None:
    canonical = tmp_path / "native"
    profile = AgyRunProfile(tmp_path / "profile", (), canonical)
    write_agy_json(profile.root / "mandri-session.json", {"native_id": "new"})
    lease = AgyConversationLease(canonical, "new")
    lease.acquire()
    process = Mock()
    process.wait.return_value = 0
    try:
        with pytest.raises(RunError, match="already has a Mandri writer"):
            wait_agy_process(process, profile)
        process.terminate.assert_called_once()
    finally:
        lease.release()


def test_cli_resume_preserves_managed_root_provenance(
    tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = config / "antigravity-cli/conversations/root.db"
    database.parent.mkdir(parents=True)
    database.touch()
    write_agy_json(
        tmp_path / "agy-profiles/managed/mandri-session.json",
        {
            "native_id": "root",
            "is_mandri_root": True,
        },
    )
    monkeypatch.setattr(
        agy_run, "inspect_owner", lambda *args: NativeOwnership(SessionOwner.UNOWNED)
    )
    spec = RunSpec(
        HarnessKind.AGY, "synthetic/model", tmp_path, passthrough_args=("--conversation", "root")
    )
    profile = prepare_agy_run(spec)
    try:
        assert read_agy_json(profile.root / "mandri-session.json")["is_mandri_root"] is True
        observe_agy(profile.root, "Stop", {"conversationId": "root", "fullyIdle": True})
        assert read_agy_json(profile.root / "mandri-session.json")["is_mandri_root"] is True
    finally:
        close_agy_run(profile)


def test_cli_detects_agy_outside_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "agy.exe"
    monkeypatch.setattr(run, "find_agy_binary", lambda: binary)
    spec = RunSpec(HarnessKind.AGY, "synthetic/model", tmp_path)
    assert run.RunCommand(spec)._resolve_binary() == binary


def test_cli_reports_missing_agy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "find_agy_binary", lambda: None)
    spec = RunSpec(HarnessKind.AGY, "synthetic/model", tmp_path)
    with pytest.raises(HarnessBinaryNotFoundError, match="not installed"):
        run.RunCommand(spec)._resolve_binary()


@pytest.mark.parametrize("args", [("--model", "native"), ("--gemini_dir=other",)])
def test_cli_prevents_route_configuration_override(
    tmp_path: Path, config: Path, args: tuple[str, ...]
) -> None:
    spec = RunSpec(HarnessKind.AGY, "synthetic/model", tmp_path, passthrough_args=args)
    with pytest.raises(RunError, match="managed by Mandri"):
        prepare_agy_run(spec)
