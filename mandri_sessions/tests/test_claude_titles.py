import json
from pathlib import Path

import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import HarnessKind
from mandri.sessions.adapters.claude_fetch_sessions import ClaudeSdkFetchSessionsAdapter
from mandri.sessions.docker_titles import docker_title
from mandri.sessions.sync import SyncEngine

from mandri_sessions.tests.substitutes import FakeDatabase
from mandri_sessions.tests.test_docker_titles import NATIVE_ID, stored_session


def append(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


@pytest.mark.parametrize("docker", [False, True])
async def test_claude_title_stays_stable_then_adopts_native_title(tmp_path, monkeypatch, docker):
    db = await FakeDatabase.create()
    try:
        state = (
            await stored_session(db, tmp_path, HarnessKind.CLAUDE, "session")
            if docker
            else tmp_path
        )
        config = state / ".claude"
        path = config / "projects/-workspace" / f"{NATIVE_ID}.jsonl"
        path.parent.mkdir(parents=True)
        append(
            path,
            {
                "type": "user",
                "sessionId": NATIVE_ID,
                "cwd": "/workspace",
                "message": {"role": "user", "content": "Fix the session model selector"},
            },
        )
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
        backends = {} if docker else {HarnessKind.CLAUDE: ClaudeSdkFetchSessionsAdapter()}
        hub = Hub()
        feed = hub.subscribe(Topic("sessions.all"))
        engine = SyncEngine(db, backends, hub=hub, docker_title_reader=docker_title)
        await engine.sync()
        original = await db.fetch_one("SELECT * FROM session")
        assert original["native_title"] == "Fix the session model selector"
        for prompt in ("continue", "Why?", "Try again"):
            append(path, {"type": "user", "message": {"role": "user", "content": prompt}})
            append(path, {"type": "last-prompt", "lastPrompt": prompt, "sessionId": NATIVE_ID})
            await engine.sync()
            assert (await db.fetch_one("SELECT * FROM session"))["native_title"] == original[
                "native_title"
            ]
        while not feed.queue.empty():
            feed.queue.get_nowait()
        append(
            path, {"type": "ai-title", "aiTitle": "Session model selection", "sessionId": NATIVE_ID}
        )
        await engine.sync()
        generated = await db.fetch_one("SELECT * FROM session")
        assert generated["id"] == original["id"]
        assert generated["native_title"] == "Session model selection"
        assert feed.queue.get_nowait()["payload"]["raw"]["type"] == "sessions_changed"
        assert generated["title_overlay"] == original["title_overlay"]
        append(path, {"type": "custom-title", "customTitle": "My title", "sessionId": NATIVE_ID})
        append(
            path, {"type": "ai-title", "aiTitle": "Another generated title", "sessionId": NATIVE_ID}
        )
        append(path, {"type": "last-prompt", "lastPrompt": "continue", "sessionId": NATIVE_ID})
        await engine.sync()
        assert (await db.fetch_one("SELECT * FROM session"))["native_title"] == "My title"
    finally:
        await db.close()
