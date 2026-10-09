import json

import pytest
from mandri.core.types.config import PrivacySettings
from mandri.database.privacy import PrivacyRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.gateway.privacy_inventory import inventory, redact
from mandri.gateway.privacy_scopes import PrivacyScopes
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.surrogate.allocation import unique_alias


class SyntheticKey:
    def load(self, *, create=False):
        return b"s" * 32


@pytest.mark.parametrize("original", ["surrogate", "mandri", "...", "Synthetic Person"])
def test_exhausted_aliases_never_introduce_product_or_placeholder_names(original):
    alias = unique_alias(original, "...", lambda source, value: value != source)
    assert alias != original
    assert all(word not in alias.casefold() for word in ("surrogate", "mandri"))


def test_generated_aliases_reject_known_personal_values():
    engine = SurrogateEngine(SurrogateScope("synthetic"))
    engine.register("Synthetic Person")
    assert not engine._available("other", "Synthetic Person Junior")
    assert not engine._available("Synthetic Person", "Synthetic Person Junior")


@pytest.mark.parametrize(
    "value", ["a", "abc", "abcdef", "a@b.co", "synthetic.person@example.invalid"]
)
def test_display_never_contains_the_complete_value(value):
    for kind in ("identity", "email", "secret"):
        masked = redact(value, kind)
        assert value not in masked
        assert "****" in masked
        if kind == "secret":
            assert masked == ("********" + value[-4:] if len(value) > 8 else "********")


async def test_inventory_uses_committed_registry_and_refreshes_configured_keys(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("mandri.gateway.privacy_context.Path.home", lambda: home)
    monkeypatch.setattr("mandri.gateway.privacy_context.getpass.getuser", lambda: "fictional-user")
    monkeypatch.setattr(
        "mandri.gateway.privacy_context.socket.gethostname", lambda: "fictional-host"
    )
    for variable in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(variable, raising=False)
    (home / ".gitconfig").write_text(
        "[user]\nname = Fictional Author\nemail = author@example.invalid\n"
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "state.sqlite")
    await db.migrate()
    credentials = ["opaque-synthetic-key-one"]
    repository = PrivacyRepository(db, SyntheticKey())
    scopes = PrivacyScopes(repository, PrivacySettings(), lambda: credentials)
    try:
        scope_id = await scopes.create(str(tmp_path))
        credentials.append("opaque-synthetic-key-two")
        protected, engine = await scopes.prepare(
            scope_id,
            lambda engine: engine.protect_text(
                "Fictional Author author@example.invalid "
                "new-person@example.invalid opaque-synthetic-key-two"
            ),
        )
        revision, displayed = await scopes.inventory(scope_id)
        stored = await repository.load(scope_id)
        assert revision == stored.revision
        assert displayed == inventory(engine.scope.mappings)
        originals = {item.original for item in engine.scope.mappings}
        assert (
            set(credentials)
            | {"Fictional Author", "author@example.invalid", "new-person@example.invalid"}
            <= originals
        )
        assert all(original not in protected for original in originals)
        assert all(original not in json.dumps(displayed) for original in originals)
        reloaded = PrivacyScopes(repository, PrivacySettings(), lambda: credentials)
        assert await reloaded.inventory(scope_id) == (revision, displayed)
        assert engine.restore_text(protected).startswith("Fictional Author author@example.invalid")
    finally:
        await db.close()


def test_legacy_placeholder_aliases_are_repaired_without_losing_restoration():
    scope = SurrogateScope.from_dict(
        {
            "scope_id": "legacy",
            "mappings": [
                {"kind": "identity", "original": "Fictional Author", "surrogate": "surrogate1"},
                {
                    "kind": "path_root",
                    "original": "/srv/Private",
                    "surrogate": "/mandri/surrogate2",
                },
            ],
            "roots": [{"original": "/srv/Private", "surrogate": "/mandri/surrogate2"}],
        }
    )
    engine = SurrogateEngine(scope)
    for text in ("Fictional Author /srv/Private/file.py", "surrogate1 /mandri/surrogate2/file.py"):
        protected = engine.protect_text(text)
        assert "surrogate" not in protected.casefold()
        assert "mandri" not in protected.casefold()
        assert engine.restore_text(protected) == "Fictional Author /srv/Private/file.py"
    assert (
        engine.restore_text("surrogate1 /mandri/surrogate2/file.py")
        == "Fictional Author /srv/Private/file.py"
    )
    reloaded = SurrogateEngine(SurrogateScope.from_dict(scope.to_dict()))
    assert reloaded.protect_text("Fictional Author") == engine.protect_text("Fictional Author")


def test_legacy_compound_aliases_upgrade_after_their_components():
    original = "https://internal.example/health"
    scope = SurrogateScope.from_dict(
        {
            "scope_id": "legacy-url",
            "mappings": [
                {
                    "kind": "url",
                    "original": original,
                    "surrogate": "https://surrogate.invalid/health",
                },
                {
                    "kind": "hostname",
                    "original": "internal.example",
                    "surrogate": "surrogate.invalid",
                },
            ],
        }
    )
    engine = SurrogateEngine(scope)
    protected = engine.protect_text(original)
    assert protected.startswith("https://")
    assert "surrogate" not in protected
    assert "internal.example" not in protected
    assert engine.restore_text(protected) == original
