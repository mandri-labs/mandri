import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.types.execution import ExecutionPhase
from mandri.core.types.execution_generation import ExecutionGeneration
from mandri.runtime.service import RuntimeService


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("oom", [False, True])
async def test_natural_worker_exit_persists_real_outcome(persist, oom):
    record = ExecutionGeneration("session", 1, "owner", ExecutionPhase.READY, 5, "{}", 1, 2)
    repository = SimpleNamespace(
        update=AsyncMock(
            return_value=dataclasses.replace(record, phase=ExecutionPhase.FAILED, revision=6)
        )
    )
    runtime = RuntimeService({}, executions=repository)
    runtime._executions._records["session"] = record
    process = SimpleNamespace(returncode=137 if oom else 23, oom_killed=oom, stop=AsyncMock())
    runtime.registry.mark_live("session", process, "codex")
    reconcile = runtime.reconcile_and_persist if persist else runtime.reconcile
    assert await reconcile() == ["session"]
    update = repository.update.call_args
    assert update.args == (record, ExecutionPhase.FAILED)
    assert json.loads(update.kwargs["context"]) == {
        "exit_code": 137 if oom else 23,
        "oom_killed": oom,
    }
    assert update.kwargs["reason"] == ("execution_oom" if oom else "execution_exited")
    process.stop.assert_awaited_once()
    assert runtime.registry.process("session") is None


async def test_persistence_failure_does_not_prevent_stopping_worker():
    runtime = RuntimeService({})
    runtime._executions.phase = AsyncMock(side_effect=OSError("database unavailable"))
    process = SimpleNamespace(returncode=None, stop=AsyncMock(return_value=0))
    runtime.registry.mark_live("session", process, "codex")
    assert await runtime.stop_session("session", restore_native=False) == 0
    process.stop.assert_awaited_once_with(5.0)
    assert runtime.registry.process("session") is None
    assert runtime._executions.phase.await_count == 2
