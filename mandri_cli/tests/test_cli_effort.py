"""Tests for the optional effort flag wiring in the CLI."""

import argparse
from pathlib import Path

import pytest
from mandri.cli import run as run_module
from mandri.cli.main import build_parser
from mandri.cli.run import RunCommand
from mandri.cli.sessions import StartSessionCommand
from mandri.cli.types import RunSpec
from mandri.core.ids import HARNESS_WIRE_FORMATS, HarnessKind


class FakeRouteResponse:
    status_code = 200

    def json(self) -> dict[str, str]:
        return {"id": "route-1", "child_token": "tok"}


def _spec(effort: str | None) -> RunSpec:
    return RunSpec(harness=HarnessKind.CLAUDE, model_arg="p/m", base_dir=Path("b"), effort=effort)


def test_run_parser_accepts_effort() -> None:
    args = build_parser().parse_args(["run", "--model", "p/m", "--effort", "high", "claude"])
    assert args.effort == "high"


def test_run_parser_effort_defaults_to_none() -> None:
    args = build_parser().parse_args(["run", "--model", "p/m", "claude"])
    assert args.effort is None


def test_sessions_start_parser_accepts_effort() -> None:
    args = build_parser().parse_args(
        [
            "sessions",
            "start",
            "--harness",
            "claude",
            "--model",
            "p/m",
            "--cwd",
            "C:/w",
            "--effort",
            "low",
        ]
    )
    assert args.effort == "low"


def test_sessions_start_parser_effort_defaults_to_none() -> None:
    args = build_parser().parse_args(
        ["sessions", "start", "--harness", "claude", "--model", "p/m", "--cwd", "C:/w"]
    )
    assert args.effort is None


def test_run_spec_carries_effort() -> None:
    assert _spec("high").effort == "high"
    assert _spec(None).effort is None


def test_run_command_holds_spec_effort() -> None:
    command = RunCommand(_spec("medium"))
    assert command._spec.effort == "medium"


def test_create_route_includes_effort_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_request(method: str, url: str, body: object = None, params: object = None) -> object:
        calls.append(body)
        return FakeRouteResponse()

    monkeypatch.setattr(run_module, "request", fake_request)
    route_id, token = RunCommand._create_route("http://x", "p/m", HarnessKind.CLAUDE, "high")
    assert (route_id, token) == ("route-1", "tok")
    assert calls[0] == {
        "model": "p/m",
        "formats": [f.value for f in HARNESS_WIRE_FORMATS[HarnessKind.CLAUDE]],
        "effort": "high",
    }


def test_create_route_omits_effort_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_request(method: str, url: str, body: object = None, params: object = None) -> object:
        calls.append(body)
        return FakeRouteResponse()

    monkeypatch.setattr(run_module, "request", fake_request)
    RunCommand._create_route("http://x", "p/m", HarnessKind.CLAUDE)
    assert calls[0] == {
        "model": "p/m",
        "formats": [f.value for f in HARNESS_WIRE_FORMATS[HarnessKind.CLAUDE]],
    }


def _start_args(effort: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        base_dir=Path("b"), harness="claude", model="p/m", cwd="C:/w", effort=effort
    )


def test_start_session_body_includes_effort_only_when_provided() -> None:
    with_effort = StartSessionCommand(_start_args("high"))
    assert with_effort._body == {
        "harness": "claude",
        "model": "p/m",
        "cwd": "C:/w",
        "effort": "high",
    }
    without_effort = StartSessionCommand(_start_args(None))
    assert without_effort._body == {"harness": "claude", "model": "p/m", "cwd": "C:/w"}
