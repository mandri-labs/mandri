from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from mandri.cli import session_continuation
from mandri.cli.main import build_parser, dispatch_args
from mandri.cli.run import RunCommand
from mandri.cli.session_continuation import ForkSessionCommand, ResumeSessionCommand
from mandri.cli.sessions import StartSessionCommand
from mandri.cli.types import RunSpec
from mandri.core.ids import HarnessKind
from mandri.core.types.execution import ExecutionBackend, PrivacyMode


def test_cli_protection_and_permission_fields_are_independent():
    args = build_parser().parse_args(
        [
            "sessions",
            "start",
            "--harness",
            "codex",
            "--model",
            "p/model",
            "--cwd",
            "/work",
            "--execution-backend",
            "docker",
            "--privacy-mode",
            "surrogate",
            "--mode",
            "full-access",
        ]
    )
    assert StartSessionCommand(args)._body == {
        "harness": "codex",
        "model": "p/model",
        "cwd": "/work",
        "execution_backend": "docker",
        "privacy_mode": "surrogate",
        "mode": "full-access",
    }


def test_cli_resume_does_not_accept_policy_changes():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["sessions", "resume", "session-one", "--privacy-mode", "none"])


@pytest.mark.parametrize("action", ["resume", "fork"])
def test_continuation_uses_daemon_native_session_operation(monkeypatch, action, tmp_path):
    argv = ["sessions", action, "session-one", "--base-dir", str(tmp_path)]
    if action == "fork":
        argv.extend(["--execution-backend", "host", "--privacy-mode", "surrogate"])
    args = build_parser().parse_args(argv)
    request = Mock(return_value=httpx.Response(200, json={"id": "session-two"}))
    monkeypatch.setattr(session_continuation, "request", request)
    monkeypatch.setattr(session_continuation, "resolve_base_url", lambda base: "http://daemon")
    command = ForkSessionCommand(args) if action == "fork" else ResumeSessionCommand(args)
    assert command.run() == 0
    assert request.call_args.args == ("POST", f"http://daemon/v1/sessions/session-one/{action}")
    assert request.call_args.kwargs["body"] == (
        {"execution_backend": "host", "privacy_mode": "surrogate"} if action == "fork" else {}
    )


def test_foreground_run_uses_managed_terminal_for_protected_defaults(monkeypatch, tmp_path: Path):
    (tmp_path / "config.toml").write_text('[defaults]\nprivacy_mode = "surrogate"\n')
    command = RunCommand(RunSpec(HarnessKind.CODEX, "provider/model", tmp_path))
    launch = Mock(side_effect=AssertionError("No real harness may run in this test"))
    monkeypatch.setattr(command, "_resolve_binary", launch)
    managed = Mock()
    managed.return_value.run.return_value = 17
    monkeypatch.setattr("mandri.cli.run.TerminalRunCommand", managed)
    assert command.run() == 17
    assert managed.call_args.args[0].privacy_mode is PrivacyMode.SURROGATE
    launch.assert_not_called()


def test_foreground_docker_and_privacy_options_reach_terminal(monkeypatch, tmp_path):
    managed = Mock()
    managed.return_value.run.return_value = 0
    monkeypatch.setattr("mandri.cli.run.TerminalRunCommand", managed)
    args = build_parser().parse_args(
        [
            "run",
            "--model",
            "provider/model",
            "--execution-backend",
            "docker",
            "--privacy-mode",
            "surrogate",
            "--base-dir",
            str(tmp_path),
            "codex",
            "--",
            "exec",
            "hello",
        ]
    )
    assert dispatch_args(args) == 0
    spec = managed.call_args.args[0]
    assert spec.execution_backend is ExecutionBackend.DOCKER
    assert spec.privacy_mode is PrivacyMode.SURROGATE
    assert spec.passthrough_args == ("exec", "hello")


def test_explicit_unprotected_run_overrides_protected_defaults(monkeypatch, tmp_path):
    (tmp_path / "config.toml").write_text(
        '[defaults]\nexecution_backend = "docker"\nprivacy_mode = "surrogate"\n'
    )
    command = RunCommand(
        RunSpec(
            HarnessKind.CODEX,
            "provider/model",
            tmp_path,
            execution_backend=ExecutionBackend.HOST,
            privacy_mode=PrivacyMode.NONE,
        )
    )
    monkeypatch.setattr(command, "_resolve_binary", Mock())
    monkeypatch.setattr(command, "_build_run_plan", Mock())
    monkeypatch.setattr(command, "_exec_harness", Mock(return_value=0))
    monkeypatch.setattr(command, "_delete_route", Mock())
    managed = Mock(side_effect=AssertionError("Managed execution was explicitly disabled"))
    monkeypatch.setattr("mandri.cli.run.TerminalRunCommand", managed)
    assert command.run() == 0
