from pathlib import Path
from unittest.mock import AsyncMock, Mock

from mandri.core.types.config import SessionsConfig
from mandri.daemon import serve


def test_default_codex_ignores_older_executable_on_path(monkeypatch):
    pinned = Path("/pinned/codex")
    monkeypatch.setattr(serve, "bundled_codex_path", lambda: pinned)
    resolver = Mock(return_value=Path("/older/path-binary"))
    monkeypatch.setattr(serve, "resolve_executable", resolver)
    commands = serve.build_harness_commands()
    assert commands["codex"] == [str(pinned), "app-server"]
    assert "codex" not in [call.args[0] for call in resolver.call_args_list]


def test_explicit_codex_command_does_not_require_bundled_installation(monkeypatch):
    bundled = Mock(side_effect=FileNotFoundError("missing"))
    monkeypatch.setattr(serve, "bundled_codex_path", bundled)
    command = ["/custom/codex", "app-server", "--config", "custom=true"]
    assert serve.build_harness_commands({"codex": command})["codex"] == command
    bundled.assert_not_called()


async def test_session_mutations_receive_same_explicit_command(tmp_path, monkeypatch):
    connect = AsyncMock()
    adapter = Mock()
    monkeypatch.setattr(serve, "connect_codex_app_server", connect)
    monkeypatch.setattr(serve, "CodexMutationsAdapter", adapter)
    command = ["/custom/codex", "app-server", "--config", "custom=true"]
    serve._build_codex_backend(
        SessionsConfig(codex_home=str(tmp_path), launch_args={"codex": command})
    )
    factory = adapter.call_args.args[0]
    await factory()
    connect.assert_awaited_once_with(command=command)
