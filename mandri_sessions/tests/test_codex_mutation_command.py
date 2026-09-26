from unittest.mock import AsyncMock

from mandri.sessions.adapters import codex_mutations


async def test_explicit_mutation_command_is_executed_without_path_substitution(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(codex_mutations.asyncio, "create_subprocess_exec", spawn)
    command = ["/pinned/codex", "app-server", "-c", "synthetic=true"]
    await codex_mutations.connect_codex_app_server(command=command)
    assert spawn.await_args.args == tuple(command)


async def test_legacy_explicit_binary_parameter_remains_supported(monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(codex_mutations.asyncio, "create_subprocess_exec", spawn)
    await codex_mutations.connect_codex_app_server(binary_path="/custom/codex")
    assert spawn.await_args.args == ("/custom/codex", "app-server")
