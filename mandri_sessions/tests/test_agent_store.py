from dataclasses import replace

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.agents import Agent, AgentState
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.sessions.agents.store import AgentStore
from mandri.sessions.errors import SessionNotFoundError


async def test_agent_relationship_and_task_identity_survive_database_reopen(tmp_path):
    path = tmp_path / "agents.db"
    db = AiosqliteDatabase()
    await db.connect(path)
    await db.migrate()
    child = Agent(
        "child",
        "parent",
        HarnessKind.CLAUDE,
        "native-child",
        "Child",
        AgentState.RUNNING,
        1,
        2,
        parent_agent_id="ancestor",
        task_id="task",
        delegation_id="delegation",
        transcript_path="synthetic.jsonl",
    )
    try:
        await AgentStore(db).upsert(child)
        await AgentStore(db).upsert(replace(child, state=AgentState.COMPLETED, updated_at=3))
        await AgentStore(db).upsert(replace(child, id="other", parent_session_id="other-parent"))
    finally:
        await db.close()
    await db.connect(path)
    try:
        store = AgentStore(db)
        restored = await store.get("child")
        assert restored == replace(child, state=AgentState.COMPLETED, updated_at=3)
        assert [agent.id for agent in await store.list("parent")] == ["child"]
        with pytest.raises(SessionNotFoundError):
            await store.get("absent")
    finally:
        await db.close()
