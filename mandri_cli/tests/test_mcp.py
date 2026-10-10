from mandri.cli.main import build_parser
from mandri.cli.mcp import McpCommand


def test_updates_merge_secrets_and_allow_explicit_removal(monkeypatch, capsys):
    args = build_parser().parse_args(
        [
            "mcp",
            "update",
            "fixture",
            "--secret",
            "KEY",
            "--unset-header",
            "OLD",
            "--no-allow-protected",
            "--no-oauth",
            "--call-timeout",
            "120",
            "--json",
        ]
    )
    command = McpCommand(args)
    requests = []

    def request(method, path, body=None):
        requests.append((method, path, body))
        if method == "GET":
            return {"servers": [{"id": "server", "name": "fixture", "revision": 4}]}
        return {"id": "server", "revision": 5}

    monkeypatch.setattr(command, "_ensure_running", lambda base: None)
    monkeypatch.setattr(command, "_request", request)
    monkeypatch.setattr("mandri.cli.mcp.getpass.getpass", lambda prompt: "private")
    assert command.run() == 0
    assert requests[-1] == (
        "PATCH",
        "/servers/server",
        {
            "revision": 4,
            "changes": {
                "env_updates": {"KEY": "private"},
                "header_updates": {"OLD": None},
                "allow_protected": False,
                "auth": "none",
                "call_timeout": 120,
            },
        },
    )
    assert "private" not in capsys.readouterr().out


def test_stdio_arguments_remain_separate(monkeypatch):
    args = build_parser().parse_args(["mcp", "add", "fixture", "--stdio"])
    args.mcp_argv = ["fixture", "--path", "/directory with spaces", "--flag"]
    command = McpCommand(args)
    changes = command._changes()
    assert changes["command"] == "fixture"
    assert changes["args"] == ["--path", "/directory with spaces", "--flag"]
