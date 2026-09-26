import sqlite3

from fastapi.testclient import TestClient
from mandri.core.version import __version__
from mandri.daemon.desktop import backup_profile
from mandri.daemon.serve import build_server


def test_backup_preserves_database_and_configuration(tmp_path):
    with sqlite3.connect(tmp_path / "mandri.db") as db:
        db.execute("CREATE TABLE example (value TEXT)")
        db.execute("INSERT INTO example VALUES ('before')")
    (tmp_path / "config.toml").write_text("[server]\nport = 8787\n")
    backup_profile(tmp_path)
    backup = tmp_path / "backups" / f"before-{__version__}"
    with sqlite3.connect(backup / "mandri.db") as db:
        assert db.execute("SELECT value FROM example").fetchone() == ("before",)
    (tmp_path / "config.toml").write_text("changed")
    backup_profile(tmp_path)
    assert (backup / "config.toml").read_text() == "[server]\nport = 8787\n"


def test_control_routes_require_desktop_token():
    server, _ = build_server("127.0.0.1", 8787, desktop_token="synthetic-token")
    with TestClient(server.config.app, base_url="http://127.0.0.1") as client:
        assert client.post("/_desktop/shutdown").status_code == 401
        assert not server.should_exit
        headers = {"Authorization": "Bearer synthetic-token"}
        assert client.get("/_desktop/status", headers=headers).json()["version"] == __version__
        assert client.post("/_desktop/shutdown", headers=headers).status_code == 200
        assert server.should_exit
