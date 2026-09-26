import sys

import pytest
from mandri.core.types.config import PrivacySettings
from mandri.database.privacy import PrivacyRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.gateway.privacy_scopes import PrivacyScopes


class SyntheticKeys:
    def load(self, *, create=False):
        return bytes(range(32))


@pytest.mark.skipif(sys.platform == "win32", reason="Secure local metadata reads require POSIX")
async def test_added_worktree_root_is_persisted_and_reversible(tmp_path, monkeypatch):
    monkeypatch.setattr("mandri.gateway.privacy_context.getpass.getuser", lambda: "synthetic-user")
    monkeypatch.setattr(
        "mandri.gateway.privacy_context.socket.gethostname", lambda: "synthetic-host"
    )
    monkeypatch.setattr("mandri.gateway.privacy_context.Path.home", lambda: tmp_path / "home")
    root = tmp_path / "project"
    worktree = tmp_path / "worktrees" / "calm-jade-otter"
    root.mkdir()
    worktree.mkdir(parents=True)
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "state.sqlite")
    try:
        await db.migrate()
        repository = PrivacyRepository(db, SyntheticKeys())
        scopes = PrivacyScopes(repository, PrivacySettings())
        scope_id = await scopes.create(str(root))
        await scopes.add_workspace(scope_id, str(worktree))
        restarted = PrivacyScopes(repository, PrivacySettings())
        values = [str(root / "source.py"), str(worktree / "source.py")]
        protected, engine = await restarted.prepare(scope_id, lambda engine: engine.protect(values))
        assert all(str(tmp_path) not in value for value in protected)
        assert protected[0] != protected[1]
        assert engine.restore(protected) == values
    finally:
        await db.close()
