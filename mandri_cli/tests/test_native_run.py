import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from mandri.cli.native_run import NativeRunCommand
from mandri.cli.run_errors import RunError
from mandri.cli.types import RunSpec
from mandri.core.ids import HarnessKind
from mandri.core.types.execution import ExecutionBackend, PrivacyMode


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
@pytest.mark.parametrize("tty", [False, True])
def test_cli_inherits_streams_and_native_exit_on_every_platform(
    tmp_path, monkeypatch, platform, tty
):
    spec = RunSpec(
        HarnessKind.CODEX, "provider/model", tmp_path, execution_backend=ExecutionBackend.DOCKER
    )
    command = NativeRunCommand(spec)
    monkeypatch.setattr(
        command, "_ensure_running", lambda base: SimpleNamespace(host="localhost", port=8787)
    )
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)
    request = Mock(
        side_effect=[
            httpx.Response(
                201, json={"id": "run", "argv": ["docker", "exec", "-i", "worker", "codex"]}
            ),
            httpx.Response(204),
        ]
    )
    monkeypatch.setattr("mandri.cli.native_run.request", request)
    spawn = Mock(return_value=SimpleNamespace(wait=lambda: 27))
    monkeypatch.setattr("mandri.cli.native_run.subprocess.Popen", spawn)
    assert command.run() == 27
    assert spawn.call_args.args[0] == ["docker", "exec", "-i", "worker", "codex"]
    assert not ({"stdin", "stdout", "stderr"} & spawn.call_args.kwargs.keys())
    assert spawn.call_args.kwargs["env"] is None
    assert request.call_args_list[0].kwargs["body"]["tty"] == tty
    assert request.call_args.args == ("DELETE", "http://localhost:8787/v1/runtime/runs/run")


def test_failed_docker_start_still_releases_prepared_resources(tmp_path, monkeypatch):
    command = NativeRunCommand(RunSpec(HarnessKind.CODEX, "provider/model", tmp_path))
    monkeypatch.setattr(
        command, "_ensure_running", lambda base: SimpleNamespace(host="localhost", port=8787)
    )
    request = Mock(
        side_effect=[
            httpx.Response(
                201, json={"id": "run", "argv": ["docker", "exec", "-i", "worker", "codex"]}
            ),
            httpx.Response(204),
        ]
    )
    monkeypatch.setattr("mandri.cli.native_run.request", request)
    monkeypatch.setattr(
        "mandri.cli.native_run.subprocess.Popen", Mock(side_effect=OSError("no Docker"))
    )
    with pytest.raises(OSError, match="no Docker"):
        command.run()
    assert request.call_args.args[0] == "DELETE"


def test_failed_preparation_never_runs_docker(tmp_path, monkeypatch):
    command = NativeRunCommand(RunSpec(HarnessKind.CODEX, "provider/model", tmp_path))
    monkeypatch.setattr(
        command, "_ensure_running", lambda base: SimpleNamespace(host="localhost", port=8787)
    )
    monkeypatch.setattr(
        "mandri.cli.native_run.request",
        Mock(
            return_value=httpx.Response(
                422,
                json={"error": {"message": "Privacy key unavailable"}},
            )
        ),
    )
    spawn = Mock()
    monkeypatch.setattr("mandri.cli.native_run.subprocess.Popen", spawn)
    with pytest.raises(RunError, match="Privacy key unavailable"):
        command.run()
    spawn.assert_not_called()


def test_host_privacy_does_not_reintroduce_stripped_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-bypass-gateway")
    spec = RunSpec(
        HarnessKind.CLAUDE, "provider/model", tmp_path, privacy_mode=PrivacyMode.SURROGATE
    )
    command = NativeRunCommand(spec)
    monkeypatch.setattr(
        command, "_ensure_running", lambda base: SimpleNamespace(host="localhost", port=8787)
    )
    request = Mock(
        side_effect=[
            httpx.Response(
                201,
                json={
                    "id": "run",
                    "argv": ["claude"],
                    "env": {"ANTHROPIC_AUTH_TOKEN": "scoped-token"},
                },
            ),
            httpx.Response(204),
        ]
    )
    monkeypatch.setattr("mandri.cli.native_run.request", request)
    spawn = Mock(return_value=SimpleNamespace(wait=lambda: 0))
    monkeypatch.setattr("mandri.cli.native_run.subprocess.Popen", spawn)
    assert command.run() == 0
    assert spawn.call_args.kwargs["env"] == {"ANTHROPIC_AUTH_TOKEN": "scoped-token"}
