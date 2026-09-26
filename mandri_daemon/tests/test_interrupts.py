import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mandri.daemon.interrupts import InterruptController


@pytest.mark.parametrize("delay", [0.0, 4.99, 5.0])
def test_second_interrupt_forces_exit_within_window(monkeypatch, delay):
    server = SimpleNamespace(should_exit=False)
    kill = Mock()
    controller = InterruptController(server, kill)
    clock = Mock(side_effect=[100.0, 100.0 + delay])
    monkeypatch.setattr("mandri.daemon.interrupts.time.monotonic", clock)
    exit_now = Mock(side_effect=SystemExit(130))
    monkeypatch.setattr("mandri.daemon.interrupts.os._exit", exit_now)
    controller.handle(signal.SIGINT, None)
    assert server.should_exit
    kill.assert_not_called()
    with pytest.raises(SystemExit):
        controller.handle(signal.SIGINT, None)
    kill.assert_called_once()
    exit_now.assert_called_once_with(130)


def test_late_interrupt_rearms_window_and_force_survives_kill_error(monkeypatch):
    server = SimpleNamespace(should_exit=False)
    kill = Mock(side_effect=OSError("unavailable"))
    controller = InterruptController(server, kill)
    monkeypatch.setattr("mandri.daemon.interrupts.time.monotonic", Mock(side_effect=[0, 6, 7]))
    exit_now = Mock(side_effect=SystemExit(130))
    monkeypatch.setattr("mandri.daemon.interrupts.os._exit", exit_now)
    controller.handle(signal.SIGINT, None)
    controller.handle(signal.SIGINT, None)
    exit_now.assert_not_called()
    with pytest.raises(SystemExit):
        controller.handle(signal.SIGINT, None)
    exit_now.assert_called_once_with(130)


def test_signal_handler_is_restored_after_cleanup():
    before = signal.getsignal(signal.SIGINT)
    controller = InterruptController(SimpleNamespace(should_exit=False), Mock())
    with controller.installed():
        assert signal.getsignal(signal.SIGINT) == controller.handle
    assert signal.getsignal(signal.SIGINT) == before


def test_real_second_sigint_exits_during_stuck_async_cleanup():
    code = """
import asyncio, signal
from types import SimpleNamespace
from mandri.daemon.interrupts import InterruptController
server = SimpleNamespace(should_exit=False)
async def run():
    loop = asyncio.get_running_loop()
    loop.call_later(0.1, signal.raise_signal, signal.SIGINT)
    loop.call_later(0.3, signal.raise_signal, signal.SIGINT)
    while not server.should_exit:
        await asyncio.sleep(0.01)
    print('graceful cleanup entered', flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        print('cleanup completed', flush=True)
with InterruptController(server, lambda: print('force children', flush=True)).installed():
    asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 130
    assert "graceful cleanup entered" in result.stdout
    assert "force children" in result.stdout
    assert "cleanup completed" not in result.stdout
