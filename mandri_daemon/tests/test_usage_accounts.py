import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.hub import Hub
from mandri.core.types.config import SessionsConfig
from mandri.core.types.usage import UsageAccount
from mandri.daemon.usage import UsageCoordinator
from mandri.daemon.usage_accounts import NativeQuotaSync
from mandri.runtime.usage_profiles import usage_profile_id


async def test_refresh_queries_every_installed_native_harness_with_no_sessions(monkeypatch):
    calls = []

    async def read(harness, profile, command, cwd, env, spawn):
        calls.append((harness, profile, command, cwd, env))
        assert cwd.is_dir() and not list(cwd.iterdir())
        return [
            UsageAccount(
                account_id=profile,
                harness=harness,
                observed_at=1,
                status="available",
                windows=({"used_percent": 20},),
            )
        ]

    monkeypatch.setattr("mandri.daemon.usage_accounts.read_native_usage", read)
    sink = AsyncMock()
    commands = {name: [name] for name in ("codex", "claude", "agy", "pi", "opencode")}
    sessions = SessionsConfig(
        codex_home="/native/codex", claude_config_dir="/native/claude", agy_home="/native/agy"
    )
    sync = NativeQuotaSync(
        commands,
        sessions,
        sink,
        env={
            "HOME": "/synthetic",
            "OPENAI_API_KEY": "gateway",
            "ANTHROPIC_AUTH_TOKEN": "gateway",
            "PI_CODING_AGENT_DIR": "/native/pi",
        },
    )
    await sync.refresh()
    assert {call[0] for call in calls} == {"codex", "claude", "agy", "pi"}
    assert sink.await_count == 4
    for harness, profile, command, cwd, env in calls:
        assert not cwd.exists()
        assert profile == usage_profile_id(harness, env, agy_root=Path("/native/agy"))
        if harness == "codex":
            assert env["CODEX_HOME"] == "/native/codex"
            assert "OPENAI_API_KEY" not in env
        if harness == "claude":
            assert env["CLAUDE_CONFIG_DIR"] == "/native/claude"
            assert "ANTHROPIC_AUTH_TOKEN" not in env
            assert "--no-session-persistence" in command
        if harness == "agy":
            assert command == [
                "agy",
                "--gemini_dir",
                str(Path("/native/agy")),
                "-p",
                "/usage",
                "--output-format",
                "json",
            ]
        if harness == "pi":
            assert "--no-session" in command
            assert "--no-extensions" in command
            assert Path(command[-1]).name == "pi_usage.ts"
    assert commands == {name: [name] for name in commands}


async def test_failed_provider_does_not_block_other_accounts_or_overwrite_last_reading(monkeypatch):
    async def read(harness, profile, *args):
        if harness == "codex":
            raise TimeoutError
        return [
            UsageAccount(account_id=profile, harness=harness, observed_at=1, status="unavailable")
        ]

    monkeypatch.setattr("mandri.daemon.usage_accounts.read_native_usage", read)
    sink = AsyncMock()
    sync = NativeQuotaSync(
        {"codex": ["codex"], "claude": ["claude"]}, SessionsConfig(), sink, env={}
    )
    with pytest.raises(RuntimeError, match="No native account quotas"):
        await sync.refresh()
    sink.assert_not_called()


async def test_subscription_is_saved_even_when_quota_windows_are_unavailable(monkeypatch):
    account = UsageAccount(
        account_id="profile",
        harness="agy",
        observed_at=1,
        status="unavailable",
        plan="Google AI Ultra",
    )
    monkeypatch.setattr(
        "mandri.daemon.usage_accounts.read_native_usage", AsyncMock(return_value=[account])
    )
    sink = AsyncMock()
    sync = NativeQuotaSync({"agy": ["agy"]}, SessionsConfig(), sink, env={})
    await sync.refresh()
    sink.assert_awaited_once_with(account)


async def test_authentication_failure_reports_actionable_reason_without_credentials(
    monkeypatch, caplog
):
    monkeypatch.setattr(
        "mandri.daemon.usage_accounts.read_native_usage",
        AsyncMock(side_effect=PermissionError("private authentication output")),
    )
    sink = AsyncMock()
    sync = NativeQuotaSync({"agy": ["agy"]}, SessionsConfig(), sink, env={})
    with pytest.raises(RuntimeError, match="No native account quotas"):
        await sync.refresh()
    assert "agy: authentication required" in caplog.text
    assert "private authentication output" not in caplog.text
    sink.assert_not_called()


async def test_concurrent_refresh_is_coalesced_and_cancellation_cleans_up(monkeypatch):
    entered = asyncio.Event()
    directories = []

    async def read(harness, profile, command, cwd, *args):
        directories.append(cwd)
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("mandri.daemon.usage_accounts.read_native_usage", read)
    sync = NativeQuotaSync({"codex": ["codex"]}, SessionsConfig(), AsyncMock(), env={})
    task = asyncio.create_task(sync.refresh())
    await entered.wait()
    await sync.refresh()
    assert len(directories) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not directories[0].exists()


async def test_coordinator_refreshes_at_startup_every_five_minutes_and_on_request(monkeypatch):
    now = 0.0
    monkeypatch.setattr("mandri.daemon.usage.time", SimpleNamespace(monotonic=lambda: now))
    repository = AsyncMock()
    repository.revalue_pending.return_value = {"next_key": None}
    repository.revision.return_value = 0
    coordinator = UsageCoordinator(repository, Hub())
    collected = []
    cycles = iter((299.0, 300.0, 301.0, 302.0))

    async def refresh():
        collected.append(now)

    async def reconcile():
        nonlocal now
        if now == 301:
            assert await coordinator.refresh()
        if now == 302:
            raise asyncio.CancelledError
        now = next(cycles)
        coordinator._wake.set()
        return False

    coordinator.refresh_accounts = refresh
    coordinator.reconcile = reconcile
    with pytest.raises(asyncio.CancelledError):
        await coordinator.run()
    assert collected == [0.0, 300.0, 301.0]
    coordinator.close()
