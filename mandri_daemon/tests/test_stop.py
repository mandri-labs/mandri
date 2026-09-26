"""Tests for the mandri-daemon stop command."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mandri.daemon import daemonctl
from mandri.daemon.main import main


def _wait_exited(process: subprocess.Popen[bytes], deadline_s: float = 5.0) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    return False


def _write_pid_file(base_dir: Path, pid: int) -> None:
    payload = json.dumps({"pid": pid, "started_at": 0})
    (base_dir / "daemon.pid").write_text(payload, encoding="utf-8")


def test_terminate_pid_returns_false_for_dead_pid() -> None:
    assert daemonctl.terminate_pid(999999999) is False


def test_stop_when_not_running_is_graceful(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["stop", "--base-dir", str(tmp_path)])
    assert exit_code == 3
    assert "not running" in capsys.readouterr().out


def test_stop_with_stale_pid_file_cleans_up(tmp_path: Path) -> None:
    _write_pid_file(tmp_path, 999999999)
    exit_code = main(["stop", "--base-dir", str(tmp_path)])
    assert exit_code == 3
    assert daemonctl.read_pid_file(tmp_path) is None


def test_stop_when_running_terminates_process_and_removes_pid_file(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        _write_pid_file(tmp_path, process.pid)
        assert daemonctl.pid_alive(process.pid)
        exit_code = main(["stop", "--base-dir", str(tmp_path)])
        assert exit_code == 0
        assert _wait_exited(process)
        assert daemonctl.read_pid_file(tmp_path) is None
    finally:
        process.wait()


def test_stop_success_prints_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        _write_pid_file(tmp_path, process.pid)
        main(["stop", "--base-dir", str(tmp_path)])
        assert f"stopped (pid {process.pid})" in capsys.readouterr().out
    finally:
        process.wait()
