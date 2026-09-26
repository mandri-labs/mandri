import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from mandri.core.types.execution import ProtectionError
from mandri.gateway import privacy_key_lock

_HOLD_LOCK = """
import sys
from pathlib import Path
from mandri.gateway.privacy_key_lock import key_creation_lock

root = Path(sys.argv[1])
with key_creation_lock(root):
    (root / "acquired").touch()
    sys.stdin.read(1)
"""


def test_creation_lock_excludes_other_processes_and_recovers_after_crash(monkeypatch, tmp_path):
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "acquired").exists():
            assert child.poll() is None, child.communicate(timeout=1)
            assert time.monotonic() < deadline, "Independent process did not acquire lock"
            time.sleep(0.01)
        ticks = iter([0.0, 6.0])
        with monkeypatch.context() as context:
            context.setattr(
                privacy_key_lock,
                "time",
                SimpleNamespace(monotonic=lambda: next(ticks), sleep=time.sleep),
            )
            with (
                pytest.raises(ProtectionError, match="initialization is busy"),
                privacy_key_lock.key_creation_lock(tmp_path),
            ):
                pytest.fail("A second process entered the key creation critical section")
        child.kill()
        child.communicate(timeout=5)
        with privacy_key_lock.key_creation_lock(tmp_path):
            assert child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)
