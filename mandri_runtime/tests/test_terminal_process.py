import asyncio
import os
import sys

import pytest
from mandri.core.terminal import TerminalSize
from mandri.runtime.terminal_process import spawn_terminal

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX terminal required")


async def test_native_terminal_has_controlling_tty_input_resize_and_exit_status():
    code = (
        "import os,sys; "
        "device=os.ttyname(os.open('/dev/tty', os.O_RDONLY)); "
        "print('TTY', os.isatty(0), os.isatty(1), device, flush=True); "
        "print('SIZE', os.get_terminal_size().columns, os.get_terminal_size().lines, flush=True); "
        "line=input(); "
        "print('INPUT', line, flush=True); "
        "size=os.get_terminal_size(); print('RESIZED', size.columns, size.lines, flush=True); "
        "sys.exit(23)"
    )
    process = await spawn_terminal([sys.executable, "-c", code], TerminalSize(rows=31, columns=97))
    try:
        stdout = process.process.stdout
        assert stdout is not None
        async with asyncio.timeout(10):
            assert await stdout.readline() == b"TTY True True /dev/tty\r\n"
            assert await stdout.readline() == b"SIZE 97 31\r\n"
            process.resize(TerminalSize(rows=43, columns=121))
            await process.write_stdin(b"terminal-canary\n")
            remaining = await stdout.read()
            assert b"INPUT terminal-canary\r\n" in remaining
            assert b"RESIZED 121 43\r\n" in remaining
            assert await process.wait() == 23
    finally:
        await process.stop()


async def test_terminal_stop_closes_descriptors_and_terminates_child():
    process = await spawn_terminal(
        [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(60)"],
        TerminalSize(),
    )
    stdout = process.process.stdout
    assert stdout is not None
    async with asyncio.timeout(10):
        assert await stdout.readline() == b"READY\r\n"
        await process.stop(grace=0.1)
    assert process.returncode is not None
    with pytest.raises(OSError):
        os.fstat(process._master)
