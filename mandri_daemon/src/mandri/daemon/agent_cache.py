import asyncio
import contextlib
import logging

from mandri.sessions.agents.service import AgentHistory

logger = logging.getLogger(__name__)


async def discover_agents(history: AgentHistory) -> None:
    while True:
        try:
            await history.refresh()
        except Exception:
            logger.exception("Agent relationship discovery failed")
        await asyncio.sleep(3)


async def backfill_agents(history: AgentHistory) -> None:
    while True:
        history.changed.clear()
        try:
            if await history.backfill():
                await asyncio.sleep(0)
                continue
        except Exception:
            logger.exception("Agent cache backfill failed")
            await asyncio.sleep(5)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(history.changed.wait(), timeout=3)
