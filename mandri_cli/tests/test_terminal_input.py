import asyncio
import sys

import pytest
from mandri.cli import terminal_input


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX readiness polling")
async def test_idle_terminal_can_be_cancelled_without_waiting_for_input(monkeypatch):
    polls = asyncio.Event()

    def idle(*args):
        polls.set()
        return [], [], []

    monkeypatch.setattr(terminal_input.select, "select", idle)
    reader = asyncio.create_task(terminal_input.read_terminal_line())
    await asyncio.wait_for(polls.wait(), 1)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(reader, 1)
