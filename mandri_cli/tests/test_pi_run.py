from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from mandri.cli.main import build_parser
from mandri.cli.run import RunCommand
from mandri.cli.types import RunSpec
from mandri.core.ids import HarnessKind


def test_pi_run_keeps_native_customization_and_applies_gateway_effort(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "custom-profile"))
    monkeypatch.setenv("USER_EXTENSION_OPTION", "enabled")
    command = RunCommand(
        RunSpec(
            harness=HarnessKind.PI,
            model_arg="fixture/model",
            base_dir=tmp_path,
            effort="high",
            passthrough_args=("--extension", "./custom.ts", "--skill", "./skills"),
        )
    )
    command._ensure_running = Mock(return_value=SimpleNamespace(host="127.0.0.1", port=8000))
    command._create_route = Mock(return_value=("route", "scoped-key"))
    command._fetch_route_metadata = Mock(return_value=None)
    command._plan = command._build_run_plan()
    command._binary = Path("pi")
    argv = command._harness_command()
    assert argv[0] == "pi"
    assert argv[-6:] == ["--thinking", "high", "--extension", "./custom.ts", "--skill", "./skills"]
    assert command._plan.env["PI_CODING_AGENT_DIR"] == str(tmp_path / "custom-profile")
    assert command._plan.env["USER_EXTENSION_OPTION"] == "enabled"
    assert command._plan.env["MANDRI_API_KEY"] == "scoped-key"
    assert "--mode" not in argv
    assert "--no-extensions" not in argv


def test_pi_cli_parser_preserves_extra_native_flags():
    args = build_parser().parse_args(
        [
            "run",
            "--model",
            "fixture/model",
            "pi",
            "--",
            "--extension",
            "./custom.ts",
        ]
    )
    assert args.harness == "pi"
    assert args.passthrough == ["--extension", "./custom.ts"]
