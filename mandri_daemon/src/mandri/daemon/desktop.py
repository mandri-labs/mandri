import asyncio
import contextlib
import os
import shutil
import sqlite3
from pathlib import Path

import psutil
import uvicorn
from fastapi import FastAPI
from mandri.core.version import __version__


def backup_profile(base_dir: Path) -> None:
    marker = base_dir / "desktop-version"
    if marker.is_file() and marker.read_text() == __version__:
        return
    base_dir.mkdir(parents=True, exist_ok=True)
    destination = base_dir / "backups" / f"before-{__version__}"
    destination.mkdir(parents=True, exist_ok=True)
    database = base_dir / "mandri.db"
    if database.is_file() and not (destination / "mandri.db").exists():
        with (
            sqlite3.connect(database) as source,
            sqlite3.connect(destination / "mandri.db") as target,
        ):
            source.backup(target)
    config = base_dir / "config.toml"
    if config.is_file() and not (destination / config.name).exists():
        shutil.copy2(config, destination / config.name)


def install_controls(app: FastAPI, server: uvicorn.Server) -> None:
    async def status() -> dict[str, str | int]:
        return {"version": __version__, "pid": os.getpid()}

    async def shutdown() -> dict[str, bool]:
        server.should_exit = True
        return {"stopping": True}

    app.add_api_route("/_desktop/status", status, methods=["GET"], include_in_schema=False)
    app.add_api_route("/_desktop/shutdown", shutdown, methods=["POST"], include_in_schema=False)


async def watch_parent(server: uvicorn.Server, parent: psutil.Process) -> None:
    while not server.should_exit:
        if not parent.is_running():
            server.should_exit = True
            return
        await asyncio.sleep(1)


async def serve_owned(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    parent = psutil.Process(int(os.environ.pop("MANDRI_DESKTOP_PARENT_PID", str(os.getppid()))))
    watcher = asyncio.create_task(watch_parent(server, parent))
    try:
        await task
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
